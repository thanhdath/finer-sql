#!/usr/bin/env python3
# grpo_writer.py – Minimal GRPO trainer for SQL writing

from __future__ import annotations
import os
import re
import argparse
import time
import random
import glob
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import requests
from datasets import Dataset, load_from_disk  # type: ignore
from transformers import AutoTokenizer  # type: ignore
from memory.compute_intrinsic_reward import IntrinsicRewardComputer
from trl import GRPOConfig, GRPOTrainer  # type: ignore
import pickle
import pandas as pd  # type: ignore
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

# Initialized after ARGS is parsed (see below, after ATOMIC_E/BETA/GAMMA are set)
ATOMIC_OPS_REWARD: Optional[AtomicOpsReward] = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("GRPO training for SQL writer")
    ap.add_argument("--model_path", default="/home/htdat/codes/alignment-handbook/output/Qwen-2.5-0.5B/checkpoint-586/", help="Base model path")
    ap.add_argument("--out_dir", default="output/grpo_sql_writer/", help="Output directory")
    ap.add_argument("--data_root", default="/home/htdat/codes/data/data", help="HF datasets root to load")
    ap.add_argument("--api_url", default="http://100.82.13.136:8001/execute", help="SQL executor API endpoint")
    ap.add_argument("--timeout_ms", type=int, default=60000)
    ap.add_argument("--max_rows", type=int, default=10000)
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--save_steps", type=int, default=200, help="Save a checkpoint every N steps")
    ap.add_argument("--save_total_limit", type=int, default=None, help="Max number of checkpoints to keep on disk (oldest deleted first). None = keep all.")
    ap.add_argument("--gradient_accumulation_steps", type=int, default=32, help="Gradient accumulation steps (global_batch = num_gpus * grad_accum)")
    ap.add_argument("--learning_rate", type=float, default=8e-6, help="Learning rate for GRPO")
    ap.add_argument("--num_generations", type=int, default=32)
    ap.add_argument("--limit_samples", type=int, default=0, help="0 = all")
    ap.add_argument("--max_completion_length", type=int, default=2048)
    ap.add_argument("--use_vllm", action="store_true")
    ap.add_argument("--vllm_mode", default="server", choices=["server", "colocate"])
    ap.add_argument("--vllm_server_host", type=str, default="0.0.0.0", help="vLLM server host (server mode)")
    ap.add_argument("--vllm_server_port", type=int, default=8000, help="vLLM server port (server mode)")
    ap.add_argument("--use_liger_kernel", action="store_true", help="Use Liger kernel for ~20%% speed and memory savings")
    ap.add_argument("--vllm-no-sleep", action="store_true", dest="vllm_no_sleep", help="Disable vLLM sleep mode in colocate (faster but needs more GPU headroom)")
    ap.add_argument("--vllm_gpu_memory_utilization", type=float, default=None, help="[E6 05B64] colocate vLLM KV budget; default None = TRL default 0.3 = original behaviour")
    ap.add_argument("--per_device_train_batch_size", type=int, default=1, help="[CORE05F2 speed] micro-batch size; default 1 = original behaviour")
    ap.add_argument("--loss_type", default=None, choices=["grpo", "bnpo", "dr_grpo", "dapo"], help="[CORE05F2 speed] TRL loss_type; default None = TRL default (dapo) = original behaviour")
    ap.add_argument("--seed", type=int, default=42,
                    help="Global seed: transformers.set_seed() before data loading/model init and GRPOConfig.seed. "
                         "Default 42 = TRL/transformers TrainingArguments default, i.e. what every earlier run used.")
    ap.add_argument("--data_seed", type=int, default=None,
                    help="Seed of the training-prompt order (TRL RepeatSampler). Default None = use --seed, which is "
                         "exactly TRL's own behaviour (its sampler is seeded with args.seed).")
    ap.add_argument("--resume_from_checkpoint", default=None,
                    help="Continue from this full TRL checkpoint dir (model, optimizer.pt, scheduler.pt, rng_state.pth, "
                         "trainer_state.json): passed to trainer.train(resume_from_checkpoint=<dir>), so global_step/"
                         "optimizer/scheduler/RNG are restored and the first global_step prompts of the seeded order are "
                         "skipped (HF ignore_data_skip=False -> accelerate skip_first_batches). The LR schedule is "
                         "re-created for --max_steps and the restored optimizer's LR is set to it at train begin. "
                         "Default None = unchanged behaviour (auto-resume only from <out_dir>/checkpoint-*).")
    ap.add_argument("--memory_snapshot_init", default=None,
                    help="Seed the memory (Chroma) store: copy this snapshot dir (e.g. checkpoint-N/"
                         "memory_snapshot_near_stepN) to --chroma_path before the store is opened. The snapshot itself is "
                         "never opened. --chroma_path must not exist or be empty. Default None = unchanged behaviour.")
    ap.add_argument("--log_sample_ids", action="store_true",
                    help="Print one '[sample_ids]' line per reward call with the unique (dataset_name, sample_id) "
                         "pairs it scores (proof of the prompt order, e.g. after --resume_from_checkpoint). "
                         "Logging only; default off = unchanged behaviour.")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--gt_cache", default="/home/htdat/codes/data/gt_rows_cache.pkl", help="Path to pickle of cached GT rows {(dataset, db_id, sql): rows}")
    ap.add_argument("--api_max_wait_time", type=int, default=300, help="Max time to wait for API recovery (seconds)")
    ap.add_argument("--api_check_interval", type=int, default=10, help="Initial interval between API health checks (seconds)")
    ap.add_argument("--api_max_retries", type=int, default=3, help="Max API dead retries before stopping training")
    # Intrinsic reward / memory args
    ap.add_argument("--chroma_path", default="./memory/chroma_db", help="Path to ChromaDB persistent store")
    ap.add_argument("--chroma_collection", default="reasoning_paths", help="Chroma collection name for reasoning paths")
    ap.add_argument("--intrinsic_top_k", "--memory-top-k", type=int, default=20, dest="intrinsic_top_k", help="How many similar reasoning paths to retrieve")
    ap.add_argument("--intrinsic_use_explore", action="store_true", help="Use exploration term against failed paths if available")
    ap.add_argument("--same-db-retrieval", action="store_true", help="Allow querying memory on the same db_id (not just other db_ids)")
    ap.add_argument("--only-same-db-retrieval", action="store_true", help="Only query memory with db_id=this_id (exclude other db_ids)")
    ap.add_argument("--no-memory", action="store_true", help="Disable memory/intrinsic reward")
    ap.add_argument("--no-atomic", action="store_true", help="Disable atomic operations reward")
    ap.add_argument("--filter-empty-gt", action="store_true", help="Filter out training samples with empty ground truth SQL execution results")
    ap.add_argument("--allow-empty-gt-match", action="store_true",
                    help="LEGACY/REPRODUCTION ONLY: let a gold with zero rows certify a prediction "
                         "that also returns zero rows. Off by default -- a gold that errored is cached "
                         "as an empty row list by pre-2026-09-21 caches, so such a 'match' pays full "
                         "execution reward for SELECT ... WHERE 1=0.")
    ap.add_argument("--max_prompt_length", type=int, default=2048, help="Maximum prompt length in tokens")
    # Atomic reward shaping function parameters (S3: e=0.05, beta=0.79, gamma=0.20)
    ap.add_argument("--atomic-e", type=float, default=0.05, help="Atomic reward shaping: weight on linear term (default: 0.05 for S3)")
    ap.add_argument("--atomic-beta", type=float, default=0.79, help="Atomic reward shaping: beta coefficient (default: 0.79 for S3)")
    ap.add_argument("--atomic-gamma", type=float, default=0.20, help="Atomic reward shaping: gamma exponent (default: 0.20 for S3)")
    ap.add_argument("--atomic_canon", default="v1", choices=["v1", "v2"],
                    help="Op extractor of the atomic term. v1 (default) = atomic_ops/atomic_ops.py as before. v2 = atomic_ops/ops_v2.py: "
                         "alias-resolved canonical ops (same Jaccard, same max over the reference list, same shaping, same activation).")
    ap.add_argument("--atomic_ops_file", default=None,
                    help="With --atomic_canon v2: JSON from tools/build_atomic_oplists.py holding the precomputed v2 op sets of the gold and "
                         "references of every prompt (keyed by dataset_name/sample_id) plus the database schemas; prompts missing from the "
                         "file fall back to extracting their reference list at runtime (cached).")
    ap.add_argument("--dump_rollouts", default=None,
                    help="Append one JSON line per scored rollout to this path: step, dataset_name, sample_id, db_id, sql, every reward "
                         "term (format, ex, executable, atomic, memory), total, verdict and the group-relative advantage TRL computed "
                         "for it. Default off; training is unchanged.")
    # GRPO training stability knobs
    ap.add_argument("--beta", type=float, default=0.04, help="KL divergence penalty coefficient (default: 0.04). Lower = more exploration, e.g. 0.001 for expert config")
    ap.add_argument("--warmup_ratio", type=float, default=0.0, help="Fraction of steps to linearly warm up LR from 0 (e.g. 0.03 = 3%% of max_steps)")
    ap.add_argument("--warmup_steps", type=int, default=0, help="Number of absolute warmup steps (overrides warmup_ratio if >0)")
    ap.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping max norm (default: 1.0)")
    return ap.parse_args()

ARGS = parse_args()

MODEL_PATH = ARGS.model_path
OUT_DIR = ARGS.out_dir
DATA_ROOT = ARGS.data_root
API_URL = ARGS.api_url
TIMEOUT_MS = ARGS.timeout_ms
MAX_ROWS = ARGS.max_rows
MAX_STEPS = ARGS.max_steps
SAVE_STEPS = ARGS.save_steps
SAVE_TOTAL_LIMIT = ARGS.save_total_limit
GRADIENT_ACCUMULATION_STEPS = ARGS.gradient_accumulation_steps
LEARNING_RATE = ARGS.learning_rate
USE_LIGER_KERNEL = ARGS.use_liger_kernel
NUM_GENERATIONS = ARGS.num_generations
LIMIT_SAMPLES = ARGS.limit_samples
SEED = ARGS.seed
DATA_SEED = ARGS.data_seed if ARGS.data_seed is not None else ARGS.seed  # resolved prompt-order seed
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
SAME_DB_RETRIEVAL = getattr(ARGS, 'same_db_retrieval', False)
ONLY_SAME_DB_RETRIEVAL = getattr(ARGS, 'only_same_db_retrieval', False)
NO_MEMORY = ARGS.no_memory
NO_ATOMIC = ARGS.no_atomic
MAX_PROMPT_LEN = ARGS.max_prompt_length
ALLOW_EMPTY_GT_MATCH = getattr(ARGS, "allow_empty_gt_match", False)
ATOMIC_E = ARGS.atomic_e
ATOMIC_BETA = ARGS.atomic_beta
ATOMIC_GAMMA = ARGS.atomic_gamma

ATOMIC_OPS_REWARD = AtomicOpsReward(dialect="sqlite", e=ATOMIC_E, beta=ATOMIC_BETA, gamma=ATOMIC_GAMMA)
ATOMIC_CANON = ARGS.atomic_canon
ATOMIC_OPS_FILE = ARGS.atomic_ops_file
DUMP_ROLLOUTS = ARGS.dump_rollouts

# ───────────────────── ATOMIC CANON v2 (opt-in; v1 path untouched) ─────────────────────
_V2_OPSETS: Dict[Any, List[set]] = {}      # (dataset_name, sample_id) -> [set(gold ops), set(ref ops), ...]
_V2_SCHEMAS: Dict[str, Dict[str, List[str]]] = {}
_V2_EXTRACTORS: Dict[str, Any] = {}
_V2_RUNTIME_CACHE: Dict[Any, List[set]] = {}
_V2_STATS = {"prebuilt": 0, "runtime": 0, "pred_parse_fail": 0}
if ATOMIC_CANON == "v2":
    from atomic_ops.ops_v2 import OpExtractor as _V2Extractor, OpParseError as _V2ParseError, jaccard as _v2_jaccard, load_ops_file as _v2_load
    if ATOMIC_OPS_FILE:
        _V2_OPSETS, _V2_SCHEMAS, _v2_meta = _v2_load(ATOMIC_OPS_FILE)
        print(f"[atomic_v2] ops file {ATOMIC_OPS_FILE}: prompts {len(_V2_OPSETS)} schemas {len(_V2_SCHEMAS)} "
              f"meta {{rows: {_v2_meta.get('rows')}, references: {_v2_meta.get('references')}, opsets: {_v2_meta.get('opsets')}, "
              f"parquet_sha256: {str(_v2_meta.get('parquet_sha256'))[:16]}}}")
    else:
        print("[atomic_v2] no --atomic_ops_file: reference op sets are extracted at runtime (no schema: unqualified columns stay unqualified)")


def _v2_extractor(dataset_name: str, db_id: str):
    key = f"{dataset_name}:{db_id}"
    ex = _V2_EXTRACTORS.get(key)
    if ex is None:
        ex = _V2Extractor(_V2_SCHEMAS.get(key))
        _V2_EXTRACTORS[key] = ex
    return ex


def atomic_score_v2(pred_sql: str, dataset_name: str, db_id: str, sample_id: Any, gt_sqls: List[str]) -> float:
    """Same rule as AtomicOpsReward.score_against_list (max over references of the shaped Jaccard; a side that does not
    parse is an empty set; an empty union scores 0), with v2 canonical op sets. Reference op sets come from the ops
    file when the prompt is there, else from extracting gt_sqls once (cached)."""
    ex = _v2_extractor(dataset_name, db_id)
    try:
        pred_ops = ex.keys(pred_sql)
    except _V2ParseError:
        pred_ops = set()
        _V2_STATS["pred_parse_fail"] += 1
    opsets = _V2_OPSETS.get((dataset_name, sample_id))
    if opsets is not None:
        _V2_STATS["prebuilt"] += 1
    else:
        ck = (dataset_name, db_id, tuple(gt_sqls or []))
        opsets = _V2_RUNTIME_CACHE.get(ck)
        if opsets is None:
            opsets = []
            for s_ in (gt_sqls or []):
                try:
                    opsets.append(ex.keys(s_))
                except _V2ParseError:
                    opsets.append(set())
            _V2_RUNTIME_CACHE[ck] = opsets
        _V2_STATS["runtime"] += 1
    best = 0.0
    for ref_ops in opsets:
        sc = ATOMIC_OPS_REWARD.execution_weighted_jaccard(_v2_jaccard(pred_ops, ref_ops))
        if sc > best:
            best = sc
    return float(best)


# ───────────────────── ROLLOUT DUMP (opt-in; one JSONL line per scored rollout) ─────────────────────
class _RolloutDump:
    """reward_fn appends one record per rollout to `pending`; the trainer flushes them after TRL has computed the
    advantages of the same generation batch (same completion order), adding step and advantage. If a flush point is
    missed the records are written at the next reward call with step/advantage None. A write failure never reaches the
    reward."""

    def __init__(self, path: str):
        import json as _json
        self.json = _json
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.fh = open(path, "a", buffering=1)
        self.pending: List[Dict[str, Any]] = []
        self.written = 0
        self.misaligned = 0
        self.errors = 0

    def add(self, rec: Dict[str, Any]) -> None:
        self.pending.append(rec)

    def flush(self, step: Optional[int] = None, advantages=None) -> None:
        recs, self.pending = self.pending, []
        if not recs:
            return
        adv = None
        if advantages is not None:
            try:
                adv = [float(x) for x in advantages.detach().cpu().flatten().tolist()] if hasattr(advantages, "detach") else [float(x) for x in advantages]
            except Exception:
                adv = None
        if adv is not None and len(adv) != len(recs):
            self.misaligned += 1
            adv = None
        for k, r in enumerate(recs):
            r["step"] = step
            r["advantage"] = (adv[k] if adv is not None else None)
            try:
                self.fh.write(self.json.dumps(r, default=str) + "\n")
                self.written += 1
            except Exception:
                self.errors += 1
        try:
            self.fh.flush()
        except Exception:
            pass


_DUMP: Optional[_RolloutDump] = _RolloutDump(DUMP_ROLLOUTS) if DUMP_ROLLOUTS else None
if _DUMP is not None:
    print(f"[dump_rollouts] every scored rollout -> {DUMP_ROLLOUTS} (step/advantage filled after TRL's advantage computation)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# ──────────────────── GOLD USABILITY (reward correctness gate) ────────────────────
# A cached ground truth can be in one of four states.  Only the first can ever
# certify a prediction:
#
#   list, non-empty            -> a real result set; compare rows
#   dict {"gold_error": True}  -> the gold SQL did NOT execute (post-2026-09-21 caches)
#   dict {"timeout": True}     -> the gold SQL timed out
#   list, empty ([])           -> AMBIGUOUS.  build_gt_cache.py used to write `[]` for a
#                                 gold that raised, so `[]` means EITHER "no rows" OR
#                                 "this gold is broken".  Either way it cannot certify:
#                                 dataframes_equal([], pred_rows) is True for ANY
#                                 prediction that returns zero rows, so the old code paid
#                                 exec_score 2.0 for `SELECT <col> FROM <t> WHERE 1=0` on
#                                 every such prompt (330 of them are BIRD retail_world,
#                                 whose golds reference `Order Details`, `Territories`,
#                                 `EmployeeTerritories` and `Customers.CompanyName` --
#                                 objects BIRD's own train release does not ship).
GOLD_OK = "ok"
GOLD_ERROR = "gold_error"
GOLD_TIMEOUT = "gold_timeout"
GOLD_EMPTY = "gold_empty"
GOLD_MISSING = "gold_missing"


def classify_gold(gt_rows: Any) -> str:
    """Classify a GT-cache value. Only GOLD_OK may be compared against a prediction."""
    if gt_rows is None:
        return GOLD_MISSING
    if isinstance(gt_rows, dict):
        if gt_rows.get("timeout") is True:
            return GOLD_TIMEOUT
        if gt_rows.get("gold_error") is True:
            return GOLD_ERROR
        rows = gt_rows.get("rows")
        if not isinstance(rows, list) or len(rows) == 0:
            return GOLD_EMPTY
        return GOLD_OK
    if isinstance(gt_rows, list):
        return GOLD_EMPTY if len(gt_rows) == 0 else GOLD_OK
    return GOLD_MISSING


def gold_may_certify(gt_rows: Any) -> bool:
    """True iff this gold is allowed to mark a prediction correct."""
    kind = classify_gold(gt_rows)
    if kind == GOLD_OK:
        return True
    if kind == GOLD_EMPTY and ALLOW_EMPTY_GT_MATCH:
        return True
    return False


def _gold_guard_selftest() -> None:
    """Startup self-test -- runs in production, not only under pytest.

    Guards the exact defect this replaced: an errored/empty gold used to match any
    prediction that returned zero rows and collect the full execution reward.
    """
    assert gold_may_certify([{"a": 1}]) is True, "a real gold must be able to certify"
    assert gold_may_certify({"gold_error": True, "error": "no such table: Order"}) is False, \
        "a gold that did not execute must never certify a prediction"
    assert gold_may_certify({"timeout": True}) is False, \
        "a gold that timed out must never certify a prediction"
    assert gold_may_certify(None) is False, "a gold absent from the cache must never certify"
    if not ALLOW_EMPTY_GT_MATCH:
        assert gold_may_certify([]) is False, \
            "an empty gold must never certify unless --allow-empty-gt-match is given"


_gold_guard_selftest()

# Load GT cache if provided
_GT_CACHE: Optional[Dict[tuple, List[Dict[str, Any]]]] = None
_TIMEOUT_ENTRIES: set = set()  # Track which entries are timeouts
_GOLD_ERROR_ENTRIES: set = set()  # Golds that did not execute when the cache was built
_EMPTY_GT_ENTRIES: set = set()    # Golds cached with zero rows (ambiguous in old caches)
if GT_CACHE_PATH and os.path.exists(GT_CACHE_PATH):
    with open(GT_CACHE_PATH, "rb") as f:
        _GT_CACHE = pickle.load(f)
    # Identify timeout / gold-error / empty entries in the cache
    if _GT_CACHE is not None:
        for key, value in _GT_CACHE.items():
            kind = classify_gold(value)
            if kind == GOLD_TIMEOUT:
                _TIMEOUT_ENTRIES.add(key)
            elif kind == GOLD_ERROR:
                _GOLD_ERROR_ENTRIES.add(key)
            elif kind == GOLD_EMPTY:
                _EMPTY_GT_ENTRIES.add(key)
    # UNCONDITIONAL (not DEBUG-gated): this census is the log line an operator greps to
    # see how much of the corpus can never be scored on this endpoint.
    print(f"[gt_cache] Loaded {len(_GT_CACHE) if _GT_CACHE else 0} entries from {GT_CACHE_PATH}")
    print(f"[gt_cache] gold_timeout={len(_TIMEOUT_ENTRIES)} "
          f"gold_error={len(_GOLD_ERROR_ENTRIES)} "
          f"gold_empty={len(_EMPTY_GT_ENTRIES)} "
          f"allow_empty_gt_match={ALLOW_EMPTY_GT_MATCH}")
    _unusable = len(_TIMEOUT_ENTRIES) + len(_GOLD_ERROR_ENTRIES) + (
        0 if ALLOW_EMPTY_GT_MATCH else len(_EMPTY_GT_ENTRIES))
    print(f"[gt_cache] {_unusable} cached golds CANNOT certify a prediction "
          f"(reward 0 for every rollout on those prompts)")
    if _EMPTY_GT_ENTRIES and not _GOLD_ERROR_ENTRIES:
        print("[gt_cache] NOTE: this cache predates the gold_error marker -- a gold that "
              "failed to execute is stored as an empty row list and is counted under "
              "gold_empty above. Rebuild with build_gt_cache.py to separate the two.")

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


def seed_memory_store(snapshot_dir: str, chroma_path: str) -> Dict[str, Any]:
    """Copy a memory snapshot into an empty/non-existent chroma_path (the snapshot is only read, never opened by
    chromadb). sqlite side files (-wal/-shm/-journal) are not copied. Returns a small record with sha256s."""
    import hashlib, json as _json, shutil
    src = os.path.abspath(snapshot_dir)
    dst = os.path.abspath(chroma_path)
    if not os.path.isfile(os.path.join(src, "chroma.sqlite3")):
        raise SystemExit(f"[memory_init] no chroma.sqlite3 in snapshot {src}")
    if os.path.exists(dst) and (not os.path.isdir(dst) or os.listdir(dst)):
        raise SystemExit(f"[memory_init] --chroma_path {dst} exists and is not empty; refusing to mix stores")
    if src == dst or dst.startswith(src + os.sep) or src.startswith(dst + os.sep):
        raise SystemExit("[memory_init] snapshot and --chroma_path overlap")
    def _sha(f):
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if os.path.isdir(dst):
        os.rmdir(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*-wal", "*-shm", "*-journal"))
    files = sorted(os.path.relpath(os.path.join(r, f), dst) for r, _, fs in os.walk(dst) for f in fs)
    rec = {"snapshot": src, "chroma_path": dst, "n_files": len(files),
           "chroma_sqlite3_sha256_src": _sha(os.path.join(src, "chroma.sqlite3")),
           "chroma_sqlite3_sha256_dst": _sha(os.path.join(dst, "chroma.sqlite3"))}
    assert rec["chroma_sqlite3_sha256_src"] == rec["chroma_sqlite3_sha256_dst"]
    print(f"[memory_init] seeded {dst} from {src}: {len(files)} files, chroma.sqlite3 sha256 "
          f"{rec['chroma_sqlite3_sha256_dst']}")
    with open(dst.rstrip(os.sep) + ".memory_init.json", "w") as f:
        _json.dump(rec, f, indent=1)
    return rec


if ARGS.memory_snapshot_init:
    seed_memory_store(ARGS.memory_snapshot_init, CHROMA_PATH)
_get_intrinsic()

# ───────────────────────── DATASET BUILDING ───────────────────────
def load_training_dataset() -> Dataset:
    datasets_to_concat: List[Dataset] = []
    if not os.path.isdir(DATA_ROOT):
        raise RuntimeError(f"Data root not found: {DATA_ROOT}")

    # Try loading DATA_ROOT directly as a single dataset first
    try:
        ds_or_dd = load_from_disk(DATA_ROOT)
        if hasattr(ds_or_dd, "keys") and callable(getattr(ds_or_dd, "keys", None)):
            if "train" in ds_or_dd:
                datasets_to_concat.append(ds_or_dd["train"])
        else:
            datasets_to_concat.append(ds_or_dd)
    except Exception:
        pass

    # Fall back to scanning subdirectories if direct load found nothing
    if not datasets_to_concat:
        for name in sorted(os.listdir(DATA_ROOT)):
            path = os.path.join(DATA_ROOT, name)
            if not os.path.isdir(path):
                continue
            try:
                ds_or_dd = load_from_disk(path)
            except Exception:
                continue
            if hasattr(ds_or_dd, "keys") and callable(getattr(ds_or_dd, "keys", None)):
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
            #ex['messages'][0]['content'] = """
#You are a meticulous SQL expert. Generate a single, correct SQL query for the user question and the provided database schema.
#Follow this exact response format:

#Rules:
#- Output exactly one SQL statement.
#- The SQL must be executable on SQLite.
#- Do not include any explanatory text.
#- Output one SQL statement only. Do not include any extra text, tags, or code fences.
#"""
            ex["prompt"] = tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=True)
            # Remove messages to satisfy TRL input schema expectations
            # import sys; sys.exit()
            del ex["messages"]
        else:
            raise RuntimeError("Dataset row missing 'messages'; no fallback to prompt_raw is allowed.")
    
    # print random prompt 
    print(random.choice(combined_rows)["prompt"])

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

    # A gold that did not execute -- or that returned no row at all -- cannot certify
    # anything.  Comparing against it used to hand exec_score 2.0 to any prediction that
    # also returned zero rows (dataframes_equal(empty, empty) is True), which is a free
    # full reward for `SELECT <col> FROM <table> WHERE 1=0`.  Reward 0, never a match.
    if not gold_may_certify(gt_rows):
        kind = classify_gold(gt_rows)
        detail = gt_rows.get("error", "") if isinstance(gt_rows, dict) else ""
        return {"ok": False, "pred_res": pred_res, "gold_unusable": kind,
                "gt_res": {"ok": False, "rows": None,
                           "error": f"GOLD UNUSABLE ({kind}): {detail or 'gold cannot certify a prediction'}"}}

    ok = dataframes_equal(_rows_to_dataframe(gt_rows), _rows_to_dataframe(pred_res.get("rows")))
    return {"ok": ok, "pred_res": pred_res, "gt_res": {"ok": True, "rows": gt_rows}}


# ───────────────────────── QUALITY SCORER ─────────────────────────
# Quality scorer is in memory.compute_intrinsic_reward module
from memory.compute_intrinsic_reward import calculate_reasoning_quality


# ───────────────────────── REWARD FUNCTION ─────────────────────────
FORMAT_BONUS_W = 1.0


_reward_call_count = 0  # tracks reward_fn invocations for periodic sample logging
_gold_unusable_rollouts = 0   # rollouts scored against a gold that cannot certify
_gold_unusable_kinds: Dict[str, int] = {}

def reward_fn(prompts, completions, **kwargs):
    global _reward_call_count, _gold_unusable_rollouts
    _reward_call_count += 1

    gt_sqls: List[List[str]] = kwargs["groundtruth_sqls"]
    db_ids: List[str] = kwargs["db_id"]
    datasets: List[str] = kwargs["dataset_name"]
    if ARGS.log_sample_ids:
        _ids = list(dict.fromkeys(zip(datasets, kwargs.get("sample_id") or [None] * len(datasets))))
        print(f"[sample_ids] call={_reward_call_count} n={len(completions)} ids={_ids}", flush=True)

    rewards: List[float] = []
    if _DUMP is not None and _DUMP.pending:
        _DUMP.flush(None, None)     # a previous batch missed its flush point: write it without step/advantage
    sample_ids = kwargs.get("sample_id") or [None] * len(completions)

    # [ORI05F speed] value-preserving: run the unchanged check_sql_correct for every
    # completion with a non-empty extracted SQL concurrently, then consume in order.
    _verdicts: List[Optional[Dict[str, Any]]] = [None] * len(completions)
    with ThreadPoolExecutor(16) as _pool:
        _futs = {}
        for _j, (_comp, _gt, _db, _ds) in enumerate(zip(completions, gt_sqls, db_ids, datasets)):
            _psql = extract_sql_from_completion(_comp)
            if _psql:
                _futs[_j] = _pool.submit(check_sql_correct, _ds, _db, _psql, _gt[0])
        for _j, _f in _futs.items():
            _verdicts[_j] = _f.result()

    for i, (comp, gt_sql, db_id, dset) in enumerate(zip(completions, gt_sqls, db_ids, datasets)):
        pred_sql = extract_sql_from_completion(comp)
        atomic_val = 0.0
        executable_val = 0.0

        # format bonus (kept from your original logic)
        thought, correct_format = has_format_bonus(comp)
        format_bonus = 1.0 if correct_format else 0.0
        format_reward = FORMAT_BONUS_W if format_bonus else 0.0

        exec_score = 0.0
        pred_res_obj: Optional[Dict[str, Any]] = None
        gt_res_obj: Optional[Dict[str, Any]] = None
        api_dead = False
        verdict: Optional[Dict[str, Any]] = None

        if pred_sql:
            verdict = _verdicts[i]
            pred_res_obj = verdict.get("pred_res")
            gt_res_obj = verdict.get("gt_res")

            if i == 0:
                if pred_res_obj and pred_res_obj["rows"] and  len(pred_res_obj["rows"]) > 0:
                    print("Pred: ", pred_res_obj["rows"][0])
                if gt_res_obj and gt_res_obj["rows"] and  len(gt_res_obj["rows"]) > 0:
                    print("GT: ", gt_res_obj["rows"][0])

            # If API dead: signal as before (trainer handles waiting/stop)
            if pred_res_obj and pred_res_obj.get("api_dead"):
                api_dead = True
                print(f"[API] Training stopped due to API being dead. Waiting for recovery...")

            # A gold that cannot certify pays 0 for the whole execution component --
            # no exact-match credit, no "able to execute" credit, no atomic-ops credit
            # (the latter two live in the elif branch below, which gt_res.ok=False closes).
            if verdict.get("gold_unusable"):
                kind = verdict["gold_unusable"]
                _gold_unusable_rollouts += 1
                _gold_unusable_kinds[kind] = _gold_unusable_kinds.get(kind, 0) + 1
                if _gold_unusable_rollouts == 1 or _gold_unusable_rollouts % 500 == 0:
                    print(f"[gold_guard] {_gold_unusable_rollouts} rollouts scored 0 on an "
                          f"unusable gold so far; by kind: {_gold_unusable_kinds} "
                          f"(latest {dset}/{db_id})")

            # Compute similarity score even if not exact-equal:
            # - need pred_res ok and GT rows present in cache
            if verdict.get("ok"):
                exec_score = 2.0
            elif pred_res_obj and pred_res_obj.get("ok") and gt_res_obj and gt_res_obj.get("ok"):
                exec_score += 1.0 # able to execute
                executable_val = 1.0
                # exec_score += _EXEC_SCORER.score_from_rows(
                #     gt_rows=gt_res_obj.get("rows"),
                #     pred_rows=pred_res_obj.get("rows"),
                # )
                if not NO_ATOMIC:
                    if ATOMIC_CANON == "v2":
                        atomic_val = atomic_score_v2(pred_sql, dset, db_id, sample_ids[i], gt_sql)
                    else:
                        atomic_val = ATOMIC_OPS_REWARD.score_against_list(pred_sql, gt_sql)
                    exec_score += atomic_val

        thought_reward = 0
        if not NO_MEMORY:
            if exec_score < 2.0:
                if thought is None:
                    thought = comp.strip()
                if len(thought) > 30:
                    thought_reward = _get_intrinsic().compute_thought_reward(
                        thought, db_id,
                        top_k=INTRINSIC_TOP_K,
                        same_db_retrieval=SAME_DB_RETRIEVAL,
                        only_same_db_retrieval=ONLY_SAME_DB_RETRIEVAL,
                    )
            elif exec_score == 2.0:
                thought_reward = 1.0

        # If the execution score indicates correctness, persist the thought to memory.
        if not NO_MEMORY and exec_score >= 2.0:
            if thought:
                # Quality gate: only save high-quality reasoning traces
                prompt_for_quality = prompts[i] if isinstance(prompts, list) and i < len(prompts) else None
                if not prompt_for_quality:
                    if DEBUG and i == 0:
                        print(f"[memory] WARNING: No prompt available for quality check, rejecting thought")
                    should_insert = False
                    quality_score = 0.0
                else:
                    try:
                        quality_score, should_insert = calculate_reasoning_quality(
                            thought,
                            pred_sql=pred_sql,
                            prompt=prompt_for_quality,
                            debug=DEBUG,
                        )
                    except Exception as e:
                        should_insert = False
                        quality_score = 0.0
                        if DEBUG and i == 0:
                            print(f"[memory] Quality check failed: {e}, rejecting thought")

                if should_insert:
                    try:
                        _ = _get_intrinsic().save_thought(
                            dataset_name=dset,
                            db_id=db_id,
                            thought=thought,
                            model_name=os.path.basename(MODEL_PATH),
                        )
                        if DEBUG and i == 0:
                            print(f"[memory] Saved thought with quality score: {quality_score:.3f}")
                    except Exception as e:
                        if DEBUG:
                            print(f"[memory] Failed to save thought: {e}")
                else:
                    if DEBUG and i == 0:
                        print(f"[memory] Rejected thought due to low quality (score: {quality_score:.3f})")

        # Save failed reasoning traces to separate collection
        if not NO_MEMORY and exec_score < 2.0:
            if thought:
                prompt_for_quality = prompts[i] if isinstance(prompts, list) and i < len(prompts) else None
                if not prompt_for_quality:
                    if DEBUG and i == 0:
                        print(f"[memory] WARNING: No prompt for failed thought quality check")
                    quality_score = 0.0
                    should_insert = False
                else:
                    try:
                        quality_score, should_insert = calculate_reasoning_quality(
                            thought,
                            pred_sql=pred_sql,
                            prompt=prompt_for_quality,
                            debug=DEBUG,
                        )
                    except Exception as e:
                        should_insert = False
                        quality_score = 0.0
                        if DEBUG and i == 0:
                            print(f"[memory] Failed thought quality check error: {e}")

                if should_insert:
                    try:
                        _ = _get_intrinsic().save_failed_thought(
                            dataset_name=dset,
                            db_id=db_id,
                            thought=thought,
                            model_name=os.path.basename(MODEL_PATH),
                        )
                        if DEBUG and i == 0:
                            print(f"[memory] Saved failed thought with quality score: {quality_score:.3f}")
                    except Exception as e:
                        if DEBUG:
                            print(f"[memory] Failed to save failed thought: {e}")

        # Total reward = format bonus + execution similarity.
        # If you want to *require* format bonus to get any reward (your original behavior), keep the gate:
        if format_reward == 0:
            total = 0
        else:
            total = exec_score + format_reward + thought_reward

        if DEBUG and i == 0:
            print("\n═════════ REWARD DEBUG ═════════")
            print("DB:", dset, db_id)
            print("FULL COMPLETION:")
            print("=" * 50)
            print(comp)
            print("=" * 50)
            print("Extracted PRED SQL:", pred_sql or "<none>")
            print("GT SQL:", gt_sql[0])
            if pred_res_obj is not None:
                print("API pred_res ok:", pred_res_obj.get("ok"), "err:", pred_res_obj.get("error"))
            if gt_res_obj is not None:
                print("GT rows ok:", gt_res_obj.get("ok"))
            print("format_reward:", format_reward)
            print("atomic score:", exec_score)
            print("thought_reward:", thought_reward)
            print("TOTAL:", total)
            print("════════════════════════════════\n")

        # Periodic sample logging every 100 reward calls for collapse monitoring.
        # Check the <think> content in training.log after ~500 steps to detect degeneration.
        if i == 0 and _reward_call_count % 100 == 0:
            think_text = (thought or "").strip()
            think_preview = think_text[:500] if think_text else "<empty>"
            print(f"\n[SAMPLE step={_reward_call_count}]")
            print(f"  DB: {dset}/{db_id}")
            print(f"  THINK: {think_preview}")
            print(f"  FORMAT_OK: {bool(format_bonus)}")
            print(f"  EXEC_SCORE: {exec_score:.3f}")
            print(f"  THOUGHT_REWARD: {thought_reward:.3f}")
            print(f"  TOTAL_REWARD: {total:.3f}")
            print(f"[/SAMPLE]\n")

        rewards.append(float(total))
        if _DUMP is not None:
            _DUMP.add({"call": _reward_call_count, "index": i, "dataset_name": dset, "sample_id": sample_ids[i], "db_id": db_id,
                       "sql": pred_sql, "format": float(format_reward), "ex": (2.0 if exec_score >= 2.0 else 0.0),
                       "executable": float(executable_val), "atomic": float(atomic_val), "memory": float(thought_reward),
                       "exec_score": float(exec_score), "total": float(total),
                       "verdict_ok": bool(pred_res_obj is not None and exec_score >= 2.0),
                       "pred_ok": (None if pred_res_obj is None else bool(pred_res_obj.get("ok"))),
                       "pred_error": (None if pred_res_obj is None or pred_res_obj.get("ok") else str(pred_res_obj.get("error"))[:200]),
                       "gold_unusable": (verdict.get("gold_unusable") if verdict else None),
                       "ts": time.time()})

    return rewards

# ───────────────────────── TRAINER SETUP ───────────────────────────


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

    def _generate_and_score_completions(self, inputs):
        """--dump_rollouts: after TRL scored a generation batch, write the pending rollout records with the trainer's
        global_step and the advantages of the same batch (same completion order). No-op when the flag is off."""
        out = super()._generate_and_score_completions(inputs)
        if _DUMP is not None:
            try:
                adv = out.get("advantages") if isinstance(out, dict) else None
                _DUMP.flush(int(self.state.global_step), adv)
            except Exception as e:  # never let logging reach the training step
                print(f"[dump_rollouts] flush failed: {type(e).__name__}: {e}")
        return out

    def _get_train_sampler(self, dataset=None):
        """TRL 0.25.1 seeds its RepeatSampler with args.seed and ignores args.data_seed.
        Keep TRL's sampler (same arguments) and only re-seed it when an explicit, different
        data_seed is given, so the prompt order is a function of data_seed alone.
        With data_seed None (default) or == seed this returns TRL's sampler untouched."""
        sampler = super()._get_train_sampler(dataset)
        data_seed = getattr(self.args, "data_seed", None)
        if data_seed is not None and data_seed != self.args.seed and getattr(sampler, "shuffle", False):
            sampler.seed = data_seed
            sampler.generator.manual_seed(data_seed)
        return sampler


def train_order_fingerprint(sampler, args, dataset, max_steps: int) -> Dict[str, Any]:
    """Fingerprint of the prompt order the trainer will see in its first `max_steps` optimizer steps.

    `sampler` must be a FRESH sampler built with the trainer's own arguments
    (trainer._get_train_sampler() returns a new RepeatSampler with a new generator on every call),
    so iterating it here does not advance the generator of the sampler used for training.
    Single-process consumption model (num_processes == 1, the only setting used by these runs):
    one optimizer step consumes generation_batch_size * gradient_accumulation_steps sampler indices.
    """
    import hashlib
    import json as _json
    n_rows = len(dataset)
    G = int(sampler.mini_repeat_count)
    B = int(sampler.batch_size)
    R = int(sampler.repeat_count)
    per_step = int(args.generation_batch_size) * int(args.gradient_accumulation_steps)
    need = per_step * int(max_steps)
    epoch_len = len(sampler)
    if epoch_len == 0:
        raise RuntimeError(f"sampler is empty (n_rows={n_rows}, prompts per generation={B})")
    stream_sha = hashlib.sha256()
    prompts: List[int] = []
    consumed = 0
    epochs = 0
    while consumed < need:
        epochs += 1
        pos = 0
        for idx in sampler:  # one epoch: a new randperm from the (same, advancing) generator
            if consumed >= need:
                break
            if pos % (G * B * R) < G * B and pos % G == 0:
                prompts.append(int(idx))
            stream_sha.update(int(idx).to_bytes(8, "little", signed=True))
            pos += 1
            consumed += 1
    cols = dataset.column_names
    def _pair(i: int):
        row = dataset[int(i)]
        return [row.get("dataset_name") if "dataset_name" in cols else None,
                row.get("sample_id") if "sample_id" in cols else None]
    ids_sha = hashlib.sha256()
    if "dataset_name" in cols and "sample_id" in cols:
        names = dataset["dataset_name"]
        sids = dataset["sample_id"]
        for i in prompts:
            ids_sha.update(_json.dumps([names[i], sids[i]]).encode() + b"\n")
        ids_hex = ids_sha.hexdigest()
    else:
        ids_hex = None
    idx_sha = hashlib.sha256(_json.dumps(prompts).encode()).hexdigest()
    return {
        "seed": int(args.seed),
        "data_seed": (None if getattr(args, "data_seed", None) is None else int(args.data_seed)),
        "sampler_seed": (None if sampler.seed is None else int(sampler.seed)),
        "shuffle": bool(sampler.shuffle),
        "n_rows": int(n_rows),
        "num_generations": G,
        "prompts_per_generation_batch": B,
        "repeat_count": R,
        "per_device_train_batch_size": int(args.per_device_train_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "steps_per_generation": int(args.steps_per_generation),
        "generation_batch_size": int(args.generation_batch_size),
        "num_iterations": int(getattr(args, "num_iterations", 1)),
        "max_steps": int(max_steps),
        "sampler_indices_per_step": per_step,
        "epoch_step_aligned": bool(epoch_len % per_step == 0),
        "epochs_touched": epochs,
        "n_prompts": len(prompts),
        "first_64_indices": prompts[:64],
        "first_32_pairs": [_pair(i) for i in prompts[:32]],
        "prompt_index_sha256": idx_sha,
        "prompt_id_sha256": ids_hex,
        "sampler_stream_sha256": stream_sha.hexdigest(),
    }


def write_train_order(trainer, out_dir: str, max_steps: int) -> Optional[Dict[str, Any]]:
    """Write <out_dir>/train_order.json. Read-only w.r.t. training: builds a separate sampler."""
    import json as _json
    if not trainer.accelerator.is_main_process:
        return None
    try:
        fp = train_order_fingerprint(trainer._get_train_sampler(), trainer.args, trainer.train_dataset, max_steps)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "train_order.json"), "w") as f:
            _json.dump(fp, f, indent=1)
        print(f"[train_order] seed={fp['seed']} data_seed={fp['data_seed']} sampler_seed={fp['sampler_seed']} "
              f"n_rows={fp['n_rows']} n_prompts={fp['n_prompts']} max_steps={fp['max_steps']}")
        print(f"[train_order] first 10 (dataset_name, sample_id): {fp['first_32_pairs'][:10]}")
        print(f"[train_order] prompt_index_sha256={fp['prompt_index_sha256']} "
              f"prompt_id_sha256={fp['prompt_id_sha256']}")
        return fp
    except Exception as e:  # never let the fingerprint stop a run
        print(f"[train_order] FAILED to write fingerprint: {type(e).__name__}: {e}")
        return None


def make_resume_schedule_callback(record_path: Optional[str] = None):
    """TrainerCallback for --resume_from_checkpoint: HF re-creates the LR scheduler for the new max_steps and then
    loads scheduler.pt (last_epoch = resumed global_step; the lambda itself is not saved), but the optimizer's
    param-group LR comes from optimizer.pt, i.e. the old schedule's value, until the first scheduler.step(). At train
    begin set every group's LR to the re-created schedule's value at the resumed step and record the resume state."""
    import json as _json
    from transformers import TrainerCallback

    class ResumeScheduleCallback(TrainerCallback):
        def __init__(self):
            self.record: Optional[Dict[str, Any]] = None

        def on_train_begin(self, args, state, control, optimizer=None, lr_scheduler=None, **kw):
            sched = lr_scheduler
            lr_before = [float(g["lr"]) for g in optimizer.param_groups]
            new = [float(base * lam(sched.last_epoch)) for base, lam in zip(sched.base_lrs, sched.lr_lambdas)]
            for g, lr in zip(optimizer.param_groups, new):
                g["lr"] = lr
            sched._last_lr = list(new)
            self.record = {"global_step": int(state.global_step), "max_steps": int(state.max_steps),
                           "scheduler_last_epoch": int(sched.last_epoch), "base_lrs": [float(b) for b in sched.base_lrs],
                           "lr_from_checkpoint_optimizer": lr_before, "lr_set_at_resume": new,
                           "lr_scheduler_type": str(args.lr_scheduler_type)}
            print(f"[resume] global_step={state.global_step} max_steps={state.max_steps} "
                  f"lr(optimizer.pt)={lr_before} -> lr(re-created schedule)={new}")
            if record_path and state.is_world_process_zero:
                with open(record_path, "w") as f:
                    _json.dump(self.record, f, indent=1)

    return ResumeScheduleCallback()


def check_resume_checkpoint(ckpt: str, n_rows: int, generation_batch_size: int, num_generations: int) -> int:
    """Validate a full TRL checkpoint and that the resumed step is inside the first epoch (HF does not replay the
    sampler draws of skipped epochs, so a resume past epoch 0 would repeat epoch 0's order). Returns global_step."""
    import json as _json
    need = ["trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"]
    miss = [f for f in need if not os.path.isfile(os.path.join(ckpt, f))]
    if miss:
        raise SystemExit(f"[resume] {ckpt} is not a full TRL checkpoint, missing {miss}")
    gs = int(_json.load(open(os.path.join(ckpt, "trainer_state.json")))["global_step"])
    prompts_per_step = max(1, generation_batch_size // num_generations)
    steps_per_epoch = n_rows // prompts_per_step
    if gs >= steps_per_epoch:
        raise SystemExit(f"[resume] global_step {gs} >= steps per epoch {steps_per_epoch}: unsupported "
                         f"(the skipped epochs' sampler draws would not be replayed)")
    print(f"[resume] checkpoint {ckpt}: global_step={gs}, steps/epoch={steps_per_epoch}; the first {gs} prompts "
          f"of the seed-{SEED if 'SEED' in globals() else '?'} order are skipped")
    return gs


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


def filter_out_empty_gt_results(ex: Dict[str, Any]) -> bool:
    """Filter out samples where the ground truth SQL has empty execution results."""
    if _GT_CACHE is None:
        return True  # Keep all if no cache available
    
    dataset_name = ex.get("dataset_name")
    db_id = ex.get("db_id") 
    ground_truth_sql = ex["groundtruth_sqls"][0]
    
    if not dataset_name or not db_id or not ground_truth_sql:
        return True  # Keep samples without required fields
    
    key = (dataset_name, db_id, ground_truth_sql)
    gt_rows = _GT_CACHE.get(key)
    
    if gt_rows is None:
        return True  # Keep samples not in cache
    
    # Check if this is a timeout entry (already handled by timeout filter)
    if isinstance(gt_rows, dict) and gt_rows.get("timeout") is True:
        return True  # Keep timeout entries (they'll be filtered by timeout filter)
    
    # Check if GT execution results are empty
    if isinstance(gt_rows, list):
        is_empty = len(gt_rows) == 0
    else:
        # Handle case where gt_rows might be a dict with rows field
        rows = gt_rows.get("rows") if isinstance(gt_rows, dict) else None
        is_empty = rows is None or (isinstance(rows, list) and len(rows) == 0)
    
    if DEBUG and is_empty:
        print(f"[filter] Skipping empty GT result sample: {dataset_name}/{db_id}")
    
    return not is_empty  # Keep if NOT empty


def apply_all_filters(dataset: Dataset) -> Dataset:
    """Apply all filtering steps and print statistics."""
    initial_size = len(dataset)
    print(f"Initial dataset size: {initial_size} samples")
    
    # Filter out samples with timeout ground truth SQL
    if _GT_CACHE is not None and _TIMEOUT_ENTRIES:
        dataset = dataset.filter(filter_out_timeouts, num_proc=4)
        timeout_filtered_size = len(dataset)
        timeout_filtered_count = initial_size - timeout_filtered_size
        print(f"Dataset size after filtering timeouts: {timeout_filtered_size} samples (filtered out: {timeout_filtered_count})")
    else:
        timeout_filtered_size = initial_size
        timeout_filtered_count = 0
    
    # Filter out samples with empty ground truth SQL execution results
    if ARGS.filter_empty_gt and _GT_CACHE is not None:
        dataset = dataset.filter(filter_out_empty_gt_results, num_proc=4)
        empty_gt_filtered_size = len(dataset)
        empty_gt_filtered_count = timeout_filtered_size - empty_gt_filtered_size
        print(f"Dataset size after filtering empty GT results: {empty_gt_filtered_size} samples (filtered out: {empty_gt_filtered_count})")
    else:
        empty_gt_filtered_size = timeout_filtered_size
        empty_gt_filtered_count = 0
    
    # Filter by prompt length
    dataset = dataset.filter(keep_if_short, num_proc=4)
    final_size = len(dataset)
    length_filtered_count = empty_gt_filtered_size - final_size
    print(f"Dataset size after filtering by length: {final_size} samples (filtered out: {length_filtered_count})")
    
    # Print summary statistics
    total_filtered = initial_size - final_size
    print(f"\n=== FILTERING SUMMARY ===")
    print(f"Initial samples: {initial_size}")
    print(f"Timeout filtered: {timeout_filtered_count}")
    print(f"Empty GT filtered: {empty_gt_filtered_count}")
    print(f"Length filtered: {length_filtered_count}")
    print(f"Total filtered: {total_filtered}")
    print(f"Final samples: {final_size}")
    print(f"Filtering rate: {total_filtered/initial_size*100:.1f}%")
    print(f"========================")
    
    return dataset

from transformers import AutoModelForCausalLM
def init_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="bfloat16",
       attn_implementation="flash_attention_2",  # nếu còn rác, tạm đổi "eager" để test
       #attn_implementation="eager"
    )
    # bắt buộc với checkpointing
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    #if hasattr(model, "enable_input_require_grads"):
    #    model.enable_input_require_grads()

    return model

def main():
    from transformers import set_seed
    set_seed(SEED)
    print(f"[seed] seed={SEED} data_seed(arg)={ARGS.data_seed} resolved prompt-order seed={DATA_SEED}")
    dataset = load_training_dataset()
    dataset = apply_all_filters(dataset)
    assert len(dataset) > 0

    training_args = GRPOConfig(
        output_dir=OUT_DIR,
        num_train_epochs=-1,
        max_steps=MAX_STEPS,
        logging_steps=1,
        learning_rate=LEARNING_RATE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_generations=NUM_GENERATIONS,
        per_device_train_batch_size=ARGS.per_device_train_batch_size,
        **({"loss_type": ARGS.loss_type} if ARGS.loss_type else {}),
        per_device_eval_batch_size=1,
        max_prompt_length=MAX_PROMPT_LEN,
        max_completion_length=MAX_COMPLETION_LENGTH,
        temperature=1.0,
        beta=ARGS.beta,
        warmup_ratio=ARGS.warmup_ratio,
        warmup_steps=ARGS.warmup_steps,
        max_grad_norm=ARGS.max_grad_norm,
        save_steps=SAVE_STEPS,
        bf16=True,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to="tensorboard",
        logging_dir=os.path.join(OUT_DIR, "logs"),
        use_vllm=USE_VLLM,
        vllm_mode=VLLM_MODE,
        vllm_server_host=ARGS.vllm_server_host,
        vllm_server_port=ARGS.vllm_server_port,
        vllm_enable_sleep_mode=(VLLM_MODE == "colocate" and not ARGS.vllm_no_sleep),
        **({"vllm_gpu_memory_utilization": ARGS.vllm_gpu_memory_utilization} if ARGS.vllm_gpu_memory_utilization is not None else {}),
        use_liger_kernel=USE_LIGER_KERNEL,
        seed=SEED,
        data_seed=ARGS.data_seed,
    ) 

    trainer = APIAwareGRPOTrainer(
        model=init_model(),
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
    )
    write_train_order(trainer, OUT_DIR, MAX_STEPS)

    if ARGS.resume_from_checkpoint:
        ckpt = os.path.abspath(ARGS.resume_from_checkpoint)
        check_resume_checkpoint(ckpt, len(dataset), training_args.generation_batch_size, NUM_GENERATIONS)
        os.makedirs(OUT_DIR, exist_ok=True)
        trainer.add_callback(make_resume_schedule_callback(os.path.join(OUT_DIR, "resume_record.json")))
        trainer.train(resume_from_checkpoint=ckpt)
        return

    # check if checkpoint exists, output-dir/checkpoint-<step>, use glob
    checkpoint_pattern = os.path.join(OUT_DIR, "checkpoint-*")
    existing_checkpoints = glob.glob(checkpoint_pattern)
    if existing_checkpoints:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train(resume_from_checkpoint=False)
    

if __name__ == "__main__":
    main()


