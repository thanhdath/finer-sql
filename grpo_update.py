#!/usr/bin/env python3
# grpo_writer.py – Minimal GRPO trainer for SQL writing

from __future__ import annotations
import os
import re
import argparse
import time
import random
from typing import Any, Dict, List, Optional, Tuple

import requests
from datasets import Dataset, load_from_disk  # type: ignore
from transformers import AutoTokenizer  # type: ignore
from memory.compute_intrinsic_reward import IntrinsicRewardComputer
from trl import GRPOConfig, GRPOTrainer  # type: ignore
import pickle
import pandas as pd  # type: ignore
from mcts_grpo import MCTSGRPOTrainer
from atomic_ops.reward import AtomicOpsReward
from sql_exec_scorer import SQLExecScorer

# ───────────────────────── SCORING CONFIG ─────────────────────────
# Approach 1 (rank correlation) or Approach 3 (footrule)
SCORING_MODE = "rankcorr"   # "rankcorr" or "footrule"
SCORING_ALPHA = 0.5         # order vs unordered correctness
SCORING_GAMMA = 1.2         # coverage harshness
SCORING_NUMERIC_EPS = 1e-6  # numeric tolerance for cell equality

_EXEC_SCORER = SQLExecScorer(
    alpha=SCORING_ALPHA,
    gamma=SCORING_GAMMA,
    mode=SCORING_MODE,
    numeric_eps=SCORING_NUMERIC_EPS,
)

ATOMIC_OPS_REWARD = AtomicOpsReward(dialect="sqlite")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("GRPO training for SQL writer")
    ap.add_argument("--model_path", default="/home/htdat/codes/alignment-handbook/output/Qwen-2.5-0.5B/checkpoint-586/", help="Base model path")
    ap.add_argument("--out_dir", default="output/grpo_sql_writer/", help="Output directory")
    ap.add_argument("--data_root", default="/home/htdat/codes/data/data", help="HF datasets root to load")
    ap.add_argument("--api_url", default="http://100.82.13.136:8001/execute", help="SQL executor API endpoint")
    ap.add_argument("--timeout_ms", type=int, default=60000)
    ap.add_argument("--max_rows", type=int, default=10000)
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--num_generations", type=int, default=32)
    ap.add_argument("--limit_samples", type=int, default=0, help="0 = all")
    ap.add_argument("--max_completion_length", type=int, default=2048)
    ap.add_argument("--use_vllm", action="store_true")
    ap.add_argument("--vllm_mode", default="server", choices=["server", "colocate"])
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--gt_cache", default="/home/htdat/codes/data/gt_rows_cache.pkl", help="Path to pickle of cached GT rows {(dataset, db_id, sql): rows}")
    ap.add_argument("--api_max_wait_time", type=int, default=300, help="Max time to wait for API recovery (seconds)")
    ap.add_argument("--api_check_interval", type=int, default=10, help="Initial interval between API health checks (seconds)")
    ap.add_argument("--api_max_retries", type=int, default=3, help="Max API dead retries before stopping training")
    # Intrinsic reward / memory args
    ap.add_argument("--chroma_path", default="./memory/chroma_db", help="Path to ChromaDB persistent store")
    ap.add_argument("--chroma_collection", default="reasoning_paths", help="Chroma collection name for reasoning paths")
    ap.add_argument("--intrinsic_top_k", type=int, default=20, help="How many similar reasoning paths to retrieve")
    ap.add_argument("--intrinsic_use_explore", action="store_true", help="Use exploration term against failed paths if available")
    return ap.parse_args()

ARGS = parse_args()

MODEL_PATH = ARGS.model_path
OUT_DIR = ARGS.out_dir
DATA_ROOT = ARGS.data_root
API_URL = ARGS.api_url
TIMEOUT_MS = ARGS.timeout_ms
MAX_ROWS = ARGS.max_rows
MAX_STEPS = ARGS.max_steps
NUM_GENERATIONS = ARGS.num_generations
LIMIT_SAMPLES = ARGS.limit_samples
MAX_COMPLETION_LENGTH = ARGS.max_completion_length
USE_VLLM = ARGS.use_vllm
VLLM_MODE = ARGS.vllm_mode
DEBUG = ARGS.debug
GT_CACHE_PATH = ARGS.gt_cache
API_MAX_WAIT_TIME = ARGS.api_max_wait_time
API_CHECK_INTERVAL = ARGS.api_check_interval
API_MAX_RETRIES = ARGS.api_max_retries
CHROMA_PATH = ARGS.chroma_path
CHROMA_COLLECTION = ARGS.chroma_collection
INTRINSIC_TOP_K = ARGS.intrinsic_top_k
INTRINSIC_USE_EXPLORE = ARGS.intrinsic_use_explore

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Load GT cache if provided
_GT_CACHE: Optional[Dict[tuple, List[Dict[str, Any]]]] = None
_TIMEOUT_ENTRIES: set = set()  # Track which entries are timeouts
if GT_CACHE_PATH and os.path.exists(GT_CACHE_PATH):
    with open(GT_CACHE_PATH, "rb") as f:
        _GT_CACHE = pickle.load(f)
    # Identify timeout entries in the cache
    if _GT_CACHE is not None:
        for key, value in _GT_CACHE.items():
            if isinstance(value, dict) and value.get("timeout") is True:
                _TIMEOUT_ENTRIES.add(key)
    if DEBUG:
        print(f"[gt_cache] Loaded {len(_GT_CACHE) if _GT_CACHE else 0} entries from {GT_CACHE_PATH}")
        print(f"[gt_cache] Found {len(_TIMEOUT_ENTRIES)} timeout entries")

# Note: messages are already saved in HF dataset; we render to text at load time
# ─────────────────────── INTRINSIC REWARD (MEMORY) ───────────────────────


# Initialize a singleton intrinsic reward computer (also used for memory updates)
# _INTRINSIC = IntrinsicRewardComputer(CHROMA_PATH, CHROMA_COLLECTION)
_INTRINSIC = None
def _get_intrinsic():
    global _INTRINSIC
    if _INTRINSIC is None:
        _INTRINSIC = IntrinsicRewardComputer(CHROMA_PATH, CHROMA_COLLECTION)
    return _INTRINSIC
_get_intrinsic()

# ───────────────────────── DATASET BUILDING ───────────────────────
def load_training_dataset() -> Dataset:
    # Load all HF datasets saved under DATA_ROOT and combine
    datasets_to_concat: List[Dataset] = []
    if not os.path.isdir(DATA_ROOT):
        raise RuntimeError(f"Data root not found: {DATA_ROOT}")

    for name in sorted(os.listdir(DATA_ROOT)):
        path = os.path.join(DATA_ROOT, name)
        if not os.path.isdir(path):
            continue
        try:
            ds_or_dd = load_from_disk(path)
        except Exception:
            continue
        # Accept DatasetDict with 'train' or a single Dataset
        if hasattr(ds_or_dd, "keys") and callable(getattr(ds_or_dd, "keys", None)):
            # DatasetDict
            if "train" in ds_or_dd:
                datasets_to_concat.append(ds_or_dd["train"])
        else:
            datasets_to_concat.append(ds_or_dd)

    if not datasets_to_concat:
        raise RuntimeError(f"No datasets loaded from {DATA_ROOT}")

    # Concatenate by converting to lists (to avoid Arrow schema mismatches)
    combined_rows: List[Dict[str, Any]] = []
    for ds in datasets_to_concat:
        combined_rows.extend(ds.to_list())

    # Optionally limit number of samples
    if LIMIT_SAMPLES and LIMIT_SAMPLES > 0:
        combined_rows = combined_rows[:LIMIT_SAMPLES]

    # Build runtime 'prompt' for TRL using dataset-provided messages
    for ex in combined_rows:
        if "messages" in ex and ex["messages"]:
            ex["prompt"] = tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=True)
            # Remove messages to satisfy TRL input schema expectations
            del ex["messages"]
        else:
            raise RuntimeError("Dataset row missing 'messages'; no fallback to prompt_raw is allowed.")

    # Basic sanity checks for required fields
    required = ["dataset_name", "db_id", "groundtruth_sqls"]
    for key in required:
        if any(key not in ex for ex in combined_rows):
            raise RuntimeError(f"Missing required field '{key}' in loaded dataset rows")

    return Dataset.from_list(combined_rows)


# ─────────────────────── API HEALTH CHECK ───────────────────
def check_api_health(api_url: str, timeout: int = 5) -> bool:
    """Check if the API is alive using the dedicated health endpoint without database execution."""
    try:
        # Use the dedicated liveness endpoint that doesn't execute any database queries
        health_url = api_url.replace('/execute', '/healthz/live')
        response = requests.get(health_url, timeout=timeout)
        return response.status_code == 200
    except (requests.exceptions.ConnectionError, 
            requests.exceptions.Timeout, 
            requests.exceptions.RequestException):
        return False


def wait_for_api_recovery(api_url: str, max_wait_time: Optional[int] = None, check_interval: Optional[int] = None) -> bool:
    """Wait for API to recover with exponential backoff.
    
    Args:
        api_url: API endpoint URL
        max_wait_time: Maximum time to wait in seconds (default from global)
        check_interval: Initial interval between checks in seconds (default from global)
        
    Returns:
        True if API recovered, False if max wait time exceeded
    """
    max_wait_time = int(max_wait_time if max_wait_time is not None else API_MAX_WAIT_TIME)
    check_interval = int(check_interval if check_interval is not None else API_CHECK_INTERVAL)
    
    start_time = time.time()
    wait_time = float(check_interval)
    
    print(f"[API] API appears to be dead. Waiting for recovery...")
    
    while time.time() - start_time < float(max_wait_time):
        if check_api_health(api_url):
            print(f"[API] API recovered after {time.time() - start_time:.1f} seconds")
            return True
        
        print(f"[API] Still waiting... (checked after {time.time() - start_time:.1f}s)")
        time.sleep(wait_time)
        
        # Exponential backoff with jitter
        wait_time = min(wait_time * 1.5, 60.0)  # Cap at 60 seconds
        wait_time += random.uniform(0, 5)  # Add jitter
    
    print(f"[API] API did not recover within {max_wait_time} seconds")
    return False


def is_api_dead_error(error_msg: str) -> bool:
    """Check if the error indicates the API is dead (not just a timeout)."""
    error_lower = error_msg.lower()
    dead_indicators = [
        "connection refused",
        "connection reset",
        "connection aborted", 
        "connection timeout",
        "name or service not known",
        "no route to host",
        "network is unreachable",
        "connection pool is closed",
        "max retries exceeded"
    ]
    return any(indicator in error_lower for indicator in dead_indicators)


# ─────────────────────── EXECUTION + COMPARISON ───────────────────
SQL_FENCE_RE = re.compile(
    r"<answer>[\s\S]*?(?:```(?:sql)?\s*([\s\S]*?)\s*```|\"\"\"\s*([\s\S]*?)\s*\"\"\")[\s\S]*?</answer>",
    re.IGNORECASE,
)


def extract_sql_from_completion(text: str) -> Optional[str]:
    # Preferred format: thought </think> answer (answer is raw SQL only)
    end_tag = "</think>"
    idx = text.rfind(end_tag)
    if idx == -1:
        return None
    answer = text[idx + len(end_tag):].strip()
    return answer or None


def has_format_bonus(text: str) -> Tuple[Optional[str], bool]:
    # Check for exactly 1 opening <think> and 1 closing </think>
    think_open_count = text.count('<think>')
    think_close_count = text.count('</think>')
    
    # Must have exactly 1 of each
    if think_open_count != 1 or think_close_count != 1:
        return (None, False)
    
    # Extract thought text (everything between <think> and </think>)
    start_tag = '<think>'
    end_tag = '</think>'
    start_idx = text.find(start_tag)
    end_idx = text.find(end_tag)
    
    if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
        return (None, False)
    
    # Additional requirement: no tokens before <think> (allow whitespace only)
    prefix = text[:start_idx]
    no_tokens_before_think = (prefix.strip() == "")

    thought = text[start_idx + len(start_tag):end_idx].strip()
    sql = extract_sql_from_completion(text)
    
    # Format bonus requires: no tokens before <think>, thought > 200 chars, AND valid SQL extraction
    return (thought, no_tokens_before_think and (len(thought) > 100) and bool(sql))


def is_select_like(sql: str) -> bool:
    s = sql.lstrip().lower()
    return s.startswith("select") or s.startswith("with") or s.startswith("explain")


def call_sql_api(dataset_name: str, db_id: str, sql: str, *, mode: Optional[str] = None, timeout_ms: Optional[int] = None, max_rows: Optional[int] = None) -> Dict[str, Any]:
    """Call the mats SQL executor API and return full response dict.

    mode: if None, auto: 'read_only' for SELECT/CTE/EXPLAIN, else 'sandbox_rollback'.
    timeout_ms/max_rows: falls back to CLI values when None.
    """
    req_mode = mode or ("read_only" if is_select_like(sql) else "sandbox_rollback")
    tm = TIMEOUT_MS if timeout_ms is None else timeout_ms
    mr = MAX_ROWS if max_rows is None else max_rows
    
    try:
        resp = requests.post(
            API_URL,
            json={
                "dataset_name": dataset_name,
                "db_id": db_id,
                "sql": sql,
                "mode": req_mode,
                "timeout_ms": tm,
                "max_rows": mr,
            },
            timeout=(tm / 1000.0) + 2,
        )
        # Return full details even on HTTP errors (not truncated)
        if resp.status_code != 200:
            return {
                "ok": False,
                "statement_type": "",
                "rows": None,
                "row_count": None,
                "pandas_result": None,
                "notice": None,
                "error": f"HTTP {resp.status_code}: {resp.text}",
                "timed_out": False,
            }
        data = resp.json()
        return {
            "ok": bool(data.get("ok", False)),
            "statement_type": data.get("statement_type"),
            "rows": data.get("rows"),
            "row_count": data.get("row_count"),
            "pandas_result": data.get("pandas_result"),
            "notice": data.get("notice"),
            "error": data.get("error"),
            "timed_out": bool(data.get("timed_out", False)),
        }
    except Exception as e:
        error_msg = str(e)
        
        # Check if this is an API dead error
        if is_api_dead_error(error_msg):
            print(f"[API] Detected API dead error: {error_msg}")
            print(f"[API] Attempting to wait for API recovery...")
            if wait_for_api_recovery(API_URL):
                print(f"[API] API recovered, retrying request...")
                # Retry the request once after recovery
                try:
                    resp = requests.post(
                        API_URL,
                        json={
                            "dataset_name": dataset_name,
                            "db_id": db_id,
                            "sql": sql,
                            "mode": req_mode,
                            "timeout_ms": tm,
                            "max_rows": mr,
                        },
                        timeout=(tm / 1000.0) + 2,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        return {
                            "ok": bool(data.get("ok", False)),
                            "statement_type": data.get("statement_type"),
                            "rows": data.get("rows"),
                            "row_count": data.get("row_count"),
                            "pandas_result": data.get("pandas_result"),
                            "notice": data.get("notice"),
                            "error": data.get("error"),
                            "timed_out": bool(data.get("timed_out", False)),
                        }
                except Exception as retry_e:
                    error_msg = f"Retry failed after API recovery: {str(retry_e)}"
            else:
                error_msg = f"API dead and did not recover: {error_msg}"
        
        return {
            "ok": False,
            "statement_type": "",
            "rows": None,
            "row_count": None,
            "pandas_result": None,
            "notice": None,
            "error": error_msg,
            "timed_out": False,
        }


def _rows_to_dataframe(rows: Optional[List[Dict[str, Any]]]) -> Optional[pd.DataFrame]:
    if rows is None:
        return None
    if not isinstance(rows, list):
        return None
    try:
        df = pd.DataFrame(rows)
        return df.fillna("")
    except Exception:
        return None


def dataframes_equal(true_df: Optional[pd.DataFrame], pred_df: Optional[pd.DataFrame]) -> bool:
    if true_df is None or pred_df is None:
        return False
    true_set = set(map(tuple, true_df.values.tolist()))
    pred_set = set(map(tuple, pred_df.values.tolist()))
    return true_set == pred_set


# ────────────────────── CORRECTNESS CHECK (separate) ──────────────────────
def check_sql_correct(dataset_name: str, db_id: str, predicted_sql: str, ground_truth_sql: str) -> Dict[str, Any]:
    """Execute predicted and ground-truth SQL via API and compare row-level results.

    Returns dict with keys: ok (bool), pred_res (dict), gt_res (dict).
    ok=True iff both calls succeed and rows_equal(pred, gt).
    """
    pred_res = call_sql_api(dataset_name, db_id, predicted_sql)
    if not pred_res.get("ok"):
        return {"ok": False, "pred_res": pred_res, "gt_res": None}

    # Use cached GT rows rather than executing during training
    if _GT_CACHE is None:
        return {"ok": False, "pred_res": pred_res, "gt_res": {"ok": False, "rows": None, "error": "No GT cache loaded"}}
    key = (dataset_name, db_id, ground_truth_sql)
    gt_rows = _GT_CACHE.get(key)
    if gt_rows is None:
        return {"ok": False, "pred_res": pred_res, "gt_res": {"ok": False, "rows": None, "error": "GT rows not in cache"}}
    
    # Check if this is a timeout entry
    if isinstance(gt_rows, dict) and gt_rows.get("timeout") is True:
        return {"ok": False, "pred_res": pred_res, "gt_res": {"ok": False, "rows": None, "error": f"GT SQL timed out: {gt_rows.get('error', 'Unknown timeout')}"}}

    ok = dataframes_equal(_rows_to_dataframe(gt_rows), _rows_to_dataframe(pred_res.get("rows")))
    return {"ok": ok, "pred_res": pred_res, "gt_res": {"ok": True, "rows": gt_rows}}


# ───────────────────────── REWARD FUNCTION ─────────────────────────
FORMAT_BONUS_W = 1.0

def reward_fn(prompts, completions, **kwargs):
    """
    Emptiness-Aware Reward (EAR):
      Base:
        - 2.0 if predicted rows == gold rows (exact)
        - 1.0 + atomic_ops if both execute but rows differ
        - 0.0 if predicted fails to execute
      EAR shaping (added on top of base):
        - +0.5  if both result sets are empty (correct-empty)
        - -0.6  if predicted empty but gold non-empty (spurious-empty)
        - else  add a soft penalty proportional to |P-G|/G, clipped at -0.25
      Total = exec_score (with EAR) + format_bonus + thought_reward
    """
    from typing import Any, Optional, Dict, List, Tuple

    # ----- EAR hyperparams & clamp -----
    EXEC_SCORE_MIN, EXEC_SCORE_MAX = -1.0, 3.0
    EAR_BONUS_BOTH_EMPTY   = 0.5
    EAR_PENALTY_MISS_EMPTY = 0.6
    EAR_MISMATCH_CLIP      = 0.25  # max magnitude for the soft mismatch penalty

    def _compute_cardinality(rows: Any, row_count_field: Optional[int] = None) -> Tuple[int, bool]:
        """
        Returns (cardinality, is_zero_like).
        - rows: list[dict] or None (each dict is one row)
        - row_count_field: optional numeric shortcut from executor if available
        - is_zero_like: [] or single-row aggregate whose values are all 0/None/""
        """
        if isinstance(row_count_field, int) and row_count_field >= 0:
            card = row_count_field
        else:
            if not rows:
                card = 0
            elif isinstance(rows, list):
                card = len(rows)
            else:
                card = 0

        zero_like = (card == 0)
        if not zero_like and isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict):
            vals = list(rows[0].values())
            zero_like = all(v in (0, None, "", "0") for v in vals)
        return card, zero_like

    # ---------- Inputs ----------
    gt_sqls: List[List[str]] = kwargs["groundtruth_sqls"]
    db_ids: List[str] = kwargs["db_id"]
    datasets: List[str] = kwargs["dataset_name"]

    rewards: List[float] = []

    for i, (comp, gt_sql, db_id, dset) in enumerate(zip(completions, gt_sqls, db_ids, datasets)):
        pred_sql = extract_sql_from_completion(comp)

        # ---- format bonus (keep additive; do NOT gate total reward) ----
        thought, correct_format = has_format_bonus(comp)
        format_reward = FORMAT_BONUS_W if correct_format else 0.0

        exec_score = 0.0
        pred_res_obj: Optional[Dict[str, Any]] = None
        gt_res_obj: Optional[Dict[str, Any]] = None

        if pred_sql:
            verdict = check_sql_correct(dset, db_id, pred_sql, gt_sql[0])
            pred_res_obj = verdict.get("pred_res")
            gt_res_obj   = verdict.get("gt_res")

            # ---- Base execution score (exact / partial+atomic / fail) ----
            if verdict.get("ok"):
                # exact rows equal
                exec_score = 2.0
            elif pred_res_obj and pred_res_obj.get("ok") and gt_res_obj and gt_res_obj.get("ok"):
                # both executed: partial + atomic ops similarity
                exec_score = 1.0 + ATOMIC_OPS_REWARD.score_against_list(pred_sql, gt_sql)
            else:
                # prediction failed to execute (syntax/runtime)
                exec_score = 0.0

            # ---- EAR shaping: compare cardinalities when both sides available ----
            if pred_res_obj and pred_res_obj.get("ok") and gt_res_obj and gt_res_obj.get("ok"):
                p_rows = pred_res_obj.get("rows")
                g_rows = gt_res_obj.get("rows")

                p_card, p_zero = _compute_cardinality(p_rows, pred_res_obj.get("row_count"))
                g_card, g_zero = _compute_cardinality(g_rows, None)

                # treat "one-row aggregate with 0/None/''" as effectively empty
                p_eff = 0 if p_zero else p_card
                g_eff = 0 if g_zero else g_card

                if g_eff == 0 and p_eff == 0:
                    exec_score += EAR_BONUS_BOTH_EMPTY     # correct-empty incl. zero-like
                elif g_eff > 0 and p_eff == 0:
                    exec_score -= EAR_PENALTY_MISS_EMPTY    # spurious-empty
                elif g_eff == 0 and p_eff > 0:
                    # predicted non-empty when gold is zero-like → small penalty
                    exec_score -= min(EAR_MISMATCH_CLIP, 0.2)
                else:
                    # soft penalty for count mismatch (bounded)
                    mismatch = abs(p_eff - g_eff) / max(1, g_eff)
                    exec_score += max(-EAR_MISMATCH_CLIP, -mismatch)

            # clamp for stability
            exec_score = float(max(EXEC_SCORE_MIN, min(EXEC_SCORE_MAX, exec_score)))

        # ---- Thought reward (unchanged logic) ----
        thought_reward = 0.0
        if exec_score < 2.0:
            if thought is None:
                thought = comp.strip()
            if thought and len(thought) > 30:
                try:
                    thought_reward = _get_intrinsic().compute_thought_reward(thought, db_id)
                except Exception:
                    thought_reward = 0.0
        else:
            thought_reward = 1.0
            # Persist successful reasoning traces
            if thought:
                try:
                    _ = _get_intrinsic().save_thought(
                        dataset_name=dset,
                        db_id=db_id,
                        thought=thought,
                        model_name=os.path.basename(MODEL_PATH),
                    )
                except Exception as e:
                    if DEBUG:
                        print(f"[memory] Failed to save thought: {e}")

        # ---- Total reward (no gating by format) ----
        total = exec_score + format_reward + thought_reward

        if DEBUG and i == 0:
            print("\n═════════ REWARD DEBUG (EAR) ═════════")
            print("DB:", dset, db_id)
            print("Extracted PRED SQL:", pred_sql or "<none>")
            print("GT SQL:", gt_sql[0])
            if pred_res_obj is not None:
                print("pred ok:", pred_res_obj.get("ok"), "err:", pred_res_obj.get("error"))
                print("pred row_count:", pred_res_obj.get("row_count"))
            if gt_res_obj is not None:
                print("gt ok:", gt_res_obj.get("ok"))
            print("format_reward:", format_reward)
            print("exec_score (with EAR):", exec_score)
            print("thought_reward:", thought_reward)
            print("TOTAL:", total)
            print("══════════════════════════════════════\n")

        rewards.append(float(total))

    return rewards

# ───────────────────────── TRAINER SETUP ───────────────────────────
MAX_PROMPT_LEN = 4096


class APIAwareGRPOTrainer(GRPOTrainer):
    """Custom GRPO trainer that handles API dead scenarios."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_dead_count = 0
        self.max_api_dead_retries = API_MAX_RETRIES
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Override compute_loss to handle API dead scenarios."""
        try:
            return super().compute_loss(model, inputs, return_outputs, **kwargs)
        except Exception as e:
            error_msg = str(e)
            if "API is dead" in error_msg or "api_dead" in error_msg:
                self.api_dead_count += 1
                print(f"[API] API dead detected in training (count: {self.api_dead_count})")
                
                if self.api_dead_count >= self.max_api_dead_retries:
                    print(f"[API] Too many API dead errors ({self.api_dead_count}). Stopping training.")
                    raise RuntimeError("Training stopped due to persistent API issues")
                
                # Wait for API recovery before continuing
                if wait_for_api_recovery(API_URL):
                    print(f"[API] API recovered, continuing training...")
                    self.api_dead_count = 0  # Reset counter on successful recovery
                    return super().compute_loss(model, inputs, return_outputs, **kwargs)
                else:
                    print(f"[API] API did not recover, stopping training.")
                    raise RuntimeError("Training stopped due to API not recovering")
            else:
                # Re-raise non-API errors
                raise


def keep_if_short(ex: Dict[str, Any]) -> bool:
    try:
        if "messages" in ex and ex["messages"]:
            prompt_text = tokenizer.apply_chat_template(ex["messages"], tokenize=False)
        else:
            prompt_text = ex.get("prompt") or ex.get("prompt_raw") or ""
        ground_truth_sql = ex["groundtruth_sqls"][0]
        if type(ground_truth_sql) == list:
            ground_truth_sql = ground_truth_sql[0]
        return len(tokenizer(prompt_text).input_ids) <= MAX_PROMPT_LEN and len(ground_truth_sql) > 0
    except Exception:
        return False


def filter_out_timeouts(ex: Dict[str, Any]) -> bool:
    """Filter out samples where the ground truth SQL timed out during cache building."""
    if _GT_CACHE is None or not _TIMEOUT_ENTRIES:
        return True  # Keep all if no cache or no timeouts
    
    dataset_name = ex.get("dataset_name")
    db_id = ex.get("db_id") 
    ground_truth_sql = ex["groundtruth_sqls"][0]
    
    if not dataset_name or not db_id or not ground_truth_sql:
        return True  # Keep samples without required fields
    
    key = (dataset_name, db_id, ground_truth_sql)
    is_timeout = key in _TIMEOUT_ENTRIES
    
    if DEBUG and is_timeout:
        print(f"[filter] Skipping timeout sample: {dataset_name}/{db_id}")
    
    return not is_timeout  # Keep if NOT a timeout

from transformers import AutoModelForCausalLM
def init_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2"
    )
    model.config.use_cache = False
    return model

def main():
    dataset = load_training_dataset()
    print(f"Initial dataset size: {len(dataset)} samples")
    
    # Filter out samples with timeout ground truth SQL
    if _GT_CACHE is not None and _TIMEOUT_ENTRIES:
        dataset = dataset.filter(filter_out_timeouts, num_proc=4)
        print(f"Dataset size after filtering timeouts: {len(dataset)} samples")
    
    dataset = dataset.filter(keep_if_short, num_proc=4)
    print(f"Dataset size after filtering: {len(dataset)} samples")
    assert len(dataset) > 0

    training_args = GRPOConfig(
        output_dir=OUT_DIR,
        num_train_epochs=-1,
        max_steps=MAX_STEPS,
        logging_steps=1,
        learning_rate=8e-6,            # full-FT on 0.6B
        gradient_accumulation_steps=32,
        num_generations=NUM_GENERATIONS,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        max_prompt_length=MAX_PROMPT_LEN,
        max_completion_length=MAX_COMPLETION_LENGTH,
        save_steps=200,
        bf16=True,
        save_total_limit=10,
        report_to="tensorboard",
        logging_dir=os.path.join(OUT_DIR, "logs"),
        use_vllm=USE_VLLM,
        vllm_mode=VLLM_MODE,
    )

    trainer = APIAwareGRPOTrainer(
        model=init_model(),
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
    )

    trainer.train(resume_from_checkpoint=True)


if __name__ == "__main__":
    main()


