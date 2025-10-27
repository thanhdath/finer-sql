#!/usr/bin/env python3
# evaluate_bird_dev.py – Evaluation script for BIRD dev dataset

from __future__ import annotations
import os
import pickle
import re
import argparse
import json
import multiprocessing
import random
import time
from datetime import timedelta
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# Set multiprocessing start method before importing vLLM
multiprocessing.set_start_method('spawn', force=True)

# Ensure vLLM uses spawn multiprocessing and unbuffered logs by default
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import pandas as pd
from datasets import load_from_disk
from datasets import Dataset, DatasetDict  # type: ignore
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import requests

# SQL extraction function
def extract_sql_from_completion(text: str) -> Optional[str]:
    """Extract SQL from completion text."""
    # Preferred format: thought </think> answer (answer is raw SQL only)
    end_tag = "</think>"
    idx = text.rfind(end_tag)
    if idx == -1:
        return None
    answer = text[idx + len(end_tag):].strip()
    return answer or None



# Minimal SQL API utilities (inlined to avoid importing training code)
API_URL = "http://192.168.1.108:8001/execute"
# API_URL = "http://localhost:8101/execute"

TIMEOUT_MS_DEFAULT = 60000
MAX_ROWS_DEFAULT = 10000


def is_select_like(sql: str) -> bool:
    s = sql.lstrip().lower()
    return s.startswith("select") or s.startswith("with") or s.startswith("explain")


def call_sql_api(dataset_name: str, db_id: str, sql: str, *, mode: Optional[str] = None, timeout_ms: Optional[int] = None, max_rows: Optional[int] = None) -> Dict[str, Any]:
    req_mode = mode or ("read_only" if is_select_like(sql) else "sandbox_rollback")
    tm = TIMEOUT_MS_DEFAULT if timeout_ms is None else int(timeout_ms)
    mr = MAX_ROWS_DEFAULT if max_rows is None else int(max_rows)
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
        return {
            "ok": False,
            "statement_type": "",
            "rows": None,
            "row_count": None,
            "pandas_result": None,
            "notice": None,
            "error": str(e),
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


# Worker utilities for parallel per-sample execution
def _normalize_sql_static(sql: str) -> str:
    s = sql.strip()
    s = re.sub(r";+\s*$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


_G_PRELOADED_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] = {}


def _init_worker(preloaded_cache: Dict[Tuple[str, str, str], Dict[str, Any]]):
    global _G_PRELOADED_CACHE
    _G_PRELOADED_CACHE = preloaded_cache or {}


def _evaluate_sample_entry(entry: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute one sample's GT and candidate SQLs with intra-sample dedup."""
    dataset_name = entry["dataset_name"]
    db_id = entry["db_id"]
    predicted_sqls: List[str] = entry.get("predicted_sqls", [])
    full_completions = entry.get("full_completions", [])
    local_exec_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    def exec_with_local_cache(sql: str) -> Dict[str, Any]:
        if not sql or not sql.strip():
            return {
                "ok": False,
                "statement_type": "",
                "rows": None,
                "row_count": 0,
                "pandas_result": "",
                "notice": None,
                "error": "Empty SQL",
                "timed_out": False,
            }
        key = (dataset_name, db_id, _normalize_sql_static(sql))
        if key in local_exec_cache:
            return local_exec_cache[key]
        # Use preloaded global cache for hits (successful only)
        cached = _G_PRELOADED_CACHE.get(key)
        if cached is not None and cached.get("ok", False) and not cached.get("timed_out", False):
            local_exec_cache[key] = cached
            return cached
        res = call_sql_api(dataset_name, db_id, sql)
        local_exec_cache[key] = res
        return res

    # Execute GT once
    gt_res = exec_with_local_cache(entry["ground_truth_sql"])
    gt_ok = gt_res.get("ok", False)
    gt_df = None
    if gt_ok:
        try:
            gt_df = _rows_to_dataframe(gt_res.get("rows"))
        except Exception:
            gt_ok = False

    any_correct = False
    first_correct_index = -1
    candidate_exec_results: List[Optional[Dict[str, Any]]] = [None] * len(predicted_sqls)
    for idx, candidate_sql in enumerate(predicted_sqls):
        if not candidate_sql or not candidate_sql.strip():
            candidate_exec_results[idx] = {
                "ok": False,
                "statement_type": "",
                "rows": None,
                "row_count": 0,
                "pandas_result": "",
                "notice": None,
                "error": "Empty SQL",
                "timed_out": False,
            }
            continue
        try:
            pred_res = exec_with_local_cache(candidate_sql)
            candidate_exec_results[idx] = pred_res
            if gt_ok and pred_res.get("ok", False):
                pred_df = _rows_to_dataframe(pred_res.get("rows"))
                if dataframes_equal(gt_df, pred_df):
                    any_correct = True
                    if first_correct_index == -1:
                        first_correct_index = idx
        except Exception:
            candidate_exec_results[idx] = {
                "ok": False,
                "statement_type": "",
                "rows": None,
                "row_count": 0,
                "pandas_result": "",
                "notice": None,
                "error": "Exception during execution",
                "timed_out": False,
            }

    result = {
        "sample_id": entry["sample_id"],
        "dataset_name": dataset_name,
        "db_id": db_id,
        "question": entry.get("question", ""),
        "ground_truth_sql": entry["ground_truth_sql"],
        "predicted_sqls": predicted_sqls,
        "full_completions": full_completions,
        "any_execution_correct": any_correct,
        "first_correct_index": first_correct_index,
        "num_candidates": len(predicted_sqls),
    }
    result_with_exec = dict(result)
    result_with_exec["gt_execution_result"] = gt_res
    result_with_exec["candidate_execution_results"] = candidate_exec_results
    return result, result_with_exec

def get_config():
    """Build configuration from argparse for vLLM usage."""
    parser = argparse.ArgumentParser(description="Evaluate BIRD dev with vLLM multi-sample generation and recall@N")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/datht/mats/alignment-handbook/output/Qwen-2.5-Coder-1.5B-SQL-Writer",
        help="Path to the model to load with vLLM"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/home/datht/graph-schema/end2end/data-dev/grpo_sql_writer_bird_dev",
        help="Path to the BIRD dev dataset on disk (save_to_disk)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation outputs"
    )
    parser.add_argument(
        "--exec-cache-file",
        type=str,
        default=None,
        help="Optional path to save execution cache pickle (defaults to <output-dir>/execution_cache.pkl)"
    )
    parser.add_argument(
        "--update-cache",
        action="store_true",
        help="If set, update and persist the execution cache with successful executions"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Max number of samples to evaluate (<= dataset length). Use -1 for all"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling evaluation subset"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for vLLM generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for vLLM"
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=20,
        help="Number of candidates to sample per prompt"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information for a few samples"
    )
    parser.add_argument(
        "--skip-executing",
        action="store_true",
        help="Only generate and save predictions; skip executing SQLs"
    )

    args = parser.parse_args()

    return {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "exec_cache_file": args.exec_cache_file,
        "max_samples": None if args.max_samples == -1 else args.max_samples,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "num_samples": args.num_samples,
        "debug": bool(args.debug),
        "seed": int(args.seed),
        "skip_executing": bool(args.skip_executing),
        "update_cache": bool(args.update_cache),
    }


class BIRDEvaluator:
    """Evaluator for BIRD dev dataset focusing on SQL generation quality."""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Set global random seeds for determinism
        self.seed: int = int(config.get("seed", 42))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
        # Tokenizer will be loaded lazily only when generation is needed
        
        # Load dataset
        print(f"Loading dataset from {config['data_path']}")
        ds_or_dd = load_from_disk(config['data_path'])
        if isinstance(ds_or_dd, DatasetDict):
            split_name = 'train' if 'train' in ds_or_dd else list(ds_or_dd.keys())[0]
            dataset: Dataset = ds_or_dd[split_name]  # choose a default split
        else:
            dataset = ds_or_dd  # type: ignore[assignment]
        if config['max_samples']:
            total_len = len(dataset)
            n = min(int(config['max_samples']), total_len)
            rng = random.Random(int(config.get('seed', 42)))
            indices = rng.sample(range(total_len), n) if n < total_len else list(range(total_len))
            self.dataset = dataset.select(indices)
        else:
            self.dataset = dataset
        
        # Create output directory
        os.makedirs(config['output_dir'], exist_ok=True)
        
        print(f"Dataset size: {len(self.dataset)}")
        print(f"Device: {self.device}")
        
        # Initialize execution cache to avoid duplicate API calls
        if config.get("exec_cache_file"):
            self.exec_cache_file = config["exec_cache_file"]
        else:
            dp = str(config.get("data_path", "")).lower()
            ds = "bird" if "bird" in dp else ("spider" if "spider" in dp else None)
            sp = "train" if "train" in dp else ("dev" if "dev" in dp else None)
            if ds and sp:
                self.exec_cache_file = os.path.join("/home/datht/mats/sql_writer/evaluation", f"execution_cache_{ds}_{sp}.pkl")
            else:
                self.exec_cache_file = os.path.join(config['output_dir'], "execution_cache.pkl")
        self.exec_cache: Dict[tuple, Dict[str, Any]] = {}
        if os.path.exists(self.exec_cache_file):
            try:
                with open(self.exec_cache_file, "rb") as f:
                    self.exec_cache = pickle.load(f)
                print(f"Loaded execution cache: {len(self.exec_cache)} entries from {self.exec_cache_file}")
            except Exception as e:
                self.exec_cache = {}
                print(f"Failed to load execution cache from {self.exec_cache_file}: {e}")
    
    def _ensure_tokenizer(self):
        """Load tokenizer on-demand to avoid unnecessary init when executing only."""
        if not hasattr(self, 'tokenizer'):
            print(f"Loading tokenizer from {self.config['model_path']}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_path'], trust_remote_code=True)

    def _setup_vllm(self):
        """Setup vLLM for inference."""
        from vllm import LLM, SamplingParams
        self.vllm_model_path = self.config['model_path']
        self.llm = LLM(
            model=self.vllm_model_path,
            dtype="bfloat16" if torch.cuda.is_available() else "float32",
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9
        )
        self.sampling_params = SamplingParams(
            temperature=float(self.config.get("temperature", 1.0)),
            max_tokens=2048,
            stop=["<|endoftext|>"],
            n=int(self.config.get("num_samples", 1)),
            seed=int(self.config.get("seed", 42)),
        )
        print(f"vLLM setup complete with model: {self.vllm_model_path}")
    
    def generate_sql(self, prompts: List[str]) -> List[List[str]]:
        """Generate multiple SQL candidates per prompt using vLLM."""
        return self._generate_with_vllm(prompts)
    
    def _generate_with_vllm(self, prompts: List[str]) -> List[List[str]]:
        """Generate using vLLM (multiple candidates per prompt)."""
        # Setup vLLM if not already done
        if not hasattr(self, 'llm'):
            self._setup_vllm()
        
        batch_size = self.config.get('batch_size', 16)
        completions_per_prompt: List[List[str]] = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating with vLLM"):
            batch_prompts = prompts[i:i + batch_size]
            outputs = self.llm.generate(batch_prompts, self.sampling_params)
            
            for output in outputs:
                candidates = [cand.text for cand in output.outputs]
                completions_per_prompt.append(candidates)
        
        return completions_per_prompt
    
    
    
    def analyze_sql_quality(self, predicted_sql: str, ground_truth_sql: str, dataset_name: str, db_id: str) -> Dict[str, Any]:
        """Analyze the quality of generated SQL using API execution."""
        analysis: Dict[str, Any] = {"has_sql": bool(predicted_sql and predicted_sql.strip())}
        
        # Add SQL execution correctness checking using API
        if predicted_sql and predicted_sql.strip():
            try:
                # Execute predicted SQL via API
                pred_res = self._exec_with_cache(dataset_name, db_id, predicted_sql)
                
                # Execute ground truth SQL via API
                gt_res = self._exec_with_cache(dataset_name, db_id, ground_truth_sql)
                
                # Check if both executed successfully and results match
                if pred_res.get("ok") and gt_res.get("ok"):
                    # Compare results using the same logic as grpo_writer
                    pred_df = _rows_to_dataframe(pred_res.get("rows"))
                    gt_df = _rows_to_dataframe(gt_res.get("rows"))
                    analysis["execution_correct"] = dataframes_equal(gt_df, pred_df)
                    analysis["pred_execution_success"] = True
                    analysis["gt_execution_success"] = True
                    
                    # Log execution results in pandas format
                    analysis["pred_result_pandas"] = pred_res.get("pandas_result", "")
                    analysis["gt_result_pandas"] = gt_res.get("pandas_result", "")
                    analysis["pred_row_count"] = pred_res.get("row_count", 0)
                    analysis["gt_row_count"] = gt_res.get("row_count", 0)
                else:
                    analysis["execution_correct"] = False
                    analysis["pred_execution_success"] = pred_res.get("ok", False)
                    analysis["gt_execution_success"] = gt_res.get("ok", False)
                    analysis["pred_error"] = pred_res.get("error") if not pred_res.get("ok") else None
                    analysis["gt_error"] = gt_res.get("error") if not gt_res.get("ok") else None
                    
                    # Log execution results even for failed queries
                    analysis["pred_result_pandas"] = pred_res.get("pandas_result", "") if pred_res.get("ok") else ""
                    analysis["gt_result_pandas"] = gt_res.get("pandas_result", "") if gt_res.get("ok") else ""
                    analysis["pred_row_count"] = pred_res.get("row_count", 0) if pred_res.get("ok") else 0
                    analysis["gt_row_count"] = gt_res.get("row_count", 0) if gt_res.get("ok") else 0
            except Exception as e:
                analysis["execution_correct"] = False
                analysis["pred_execution_success"] = False
                analysis["gt_execution_success"] = False
                analysis["execution_error"] = str(e)
                analysis["pred_result_pandas"] = ""
                analysis["gt_result_pandas"] = ""
                analysis["pred_row_count"] = 0
                analysis["gt_row_count"] = 0
        else:
            analysis["execution_correct"] = False
            analysis["pred_execution_success"] = False
            analysis["gt_execution_success"] = False
            analysis["pred_result_pandas"] = ""
            analysis["gt_result_pandas"] = ""
            analysis["pred_row_count"] = 0
            analysis["gt_row_count"] = 0
        
        return analysis
    
    
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run evaluation on the dataset."""
        print("Starting evaluation...")
        
        # Predictions file path
        predictions_file = os.path.join(self.config['output_dir'], "predicted_sqls.json")
        predictions: List[Dict[str, Any]] = []
        
        # If predictions exist, load them to skip generation and vLLM init
        if os.path.exists(predictions_file):
            try:
                with open(predictions_file, "r") as f:
                    predictions = json.load(f)
                print(f"Loaded existing predicted SQLs from {predictions_file}")
            except Exception as e:
                print(f"Failed to load existing predictions from {predictions_file}: {e}")
                predictions = []
        
        # Generation flow if predictions are missing
        if not predictions:
            # Prepare prompts
            self._ensure_tokenizer()
            prompts: List[str] = []
            for sample in self.dataset:
                sample_dict = dict(sample)
#                 sample_dict["messages"][0]["content"] = """
# You are a meticulous SQL expert. Generate a single, correct SQL query for the user question and the provided database schema.
# Follow this exact response format:

# Rules:
# - Output exactly one SQL statement.
# - The SQL must be executable on SQLite.
# - Do not include any explanatory text.
# - Output one SQL statement only. Do not include any extra text, tags, or code fences.
# """
                prompt = self.tokenizer.apply_chat_template(sample_dict["messages"], tokenize=False, add_generation_prompt=True)
                prompts.append(prompt)
            
            # Generate SQL
            print("Generating SQL predictions (multi-sample)...")
            completions_per_prompt = self.generate_sql(prompts)
            
            # Extract SQL from completions
            predicted_sqls_per_prompt: List[List[str]] = []
            for comps in completions_per_prompt:
                sqls: List[str] = []
                for comp in comps:
                    sql = extract_sql_from_completion(comp)
                    sqls.append(sql or "")
                predicted_sqls_per_prompt.append(sqls)
            
            # Build predictions list with sample metadata
            for i, sample in enumerate(self.dataset):
                sample_dict = dict(sample)
                entry = {
                    "sample_id": sample_dict["sample_id"],
                    "dataset_name": sample_dict["dataset_name"],
                    "db_id": sample_dict["db_id"],
                    "question": sample_dict["question"],
                    "ground_truth_sql": sample_dict["groundtruth_sqls"][0],
                    "predicted_sqls": predicted_sqls_per_prompt[i],
                    "full_completions": completions_per_prompt[i],
                }
                predictions.append(entry)
            
            # Save predictions for later execution
            try:
                with open(predictions_file, "w") as f:
                    json.dump(predictions, f, indent=2, default=str)
                print(f"Saved predicted SQLs to {predictions_file}")
            except Exception as e:
                print(f"Failed to save predictions to {predictions_file}: {e}")

            # If generation-only requested, exit now
            if self.config.get("skip_executing", False):
                return {
                    "generated_only": True,
                    "predictions_file": predictions_file,
                    "num_predictions": len(predictions),
                }
        else:
            # Predictions already exist; if skip-executing is set, do nothing further
            if self.config.get("skip_executing", False):
                print("--skip-executing set and predictions exist; skipping execution.")
                return {
                    "generated_only": True,
                    "predictions_file": predictions_file,
                    "num_predictions": len(predictions),
                }
        
        # Analyze each predicted entry with recall@N using 16 processes (per-sample parallelism)
        print("Analyzing samples...")
        results: List[Dict[str, Any]] = []
        results_with_exec: List[Dict[str, Any]] = []
        # Build a preloaded cache dict of successful entries for worker reuse
        preloaded_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for key, val in self.exec_cache.items():
            if val.get("ok", False) and not val.get("timed_out", False):
                preloaded_cache[key] = val

        with multiprocessing.Pool(processes=8, initializer=_init_worker, initargs=(preloaded_cache,)) as pool:
            for res_pair in tqdm(pool.imap(_evaluate_sample_entry, predictions), total=len(predictions)):
                result, result_with_exec = res_pair
                results.append(result)
                results_with_exec.append(result_with_exec)
        # Ingest successful execution results from workers into the in-memory cache
        try:
            ingested_count = self._ingest_execution_results_into_cache(results_with_exec)
            print(f"Ingested {ingested_count} successful execution results into cache")
        except Exception as e:
            print(f"Failed to ingest execution results into cache: {e}")

        # Compute metrics
        metrics = self._compute_metrics(results)
        
        # Save results (JSON) and detailed results with execution outputs (PKL)
        self._save_results(results, metrics)

        # Save detailed results with execution outputs as PKL (same list-of-dicts shape as JSON, plus exec fields)
        detailed_pkl = os.path.join(self.config['output_dir'], "detailed_results.pkl")
        try:
            with open(detailed_pkl, "wb") as f:
                pickle.dump(results_with_exec, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Detailed results with exec saved to {detailed_pkl}")
        except Exception as e:
            print(f"Failed to save detailed results PKL to {detailed_pkl}: {e}")
        
        # Persist execution cache to disk if updating is enabled or n==30
        if self.config.get("update_cache", False) or int(self.config.get("num_samples", 0)) >= 30:
            try:
                self.save_execution_cache()
            except Exception as e:
                print(f"Failed to save execution cache to {self.exec_cache_file}: {e}")
        
        return metrics

    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL for caching: trim, collapse whitespace, drop trailing semicolons."""
        s = sql.strip()
        s = re.sub(r";+\s*$", "", s)
        s = re.sub(r"\s+", " ", s)
        return s

    def _exec_with_cache(self, dataset_name: str, db_id: str, sql: str, *, mode: Optional[str] = None, timeout_ms: Optional[int] = None, max_rows: Optional[int] = None) -> Dict[str, Any]:
        """Execute SQL with in-memory cache keyed by (dataset, db, normalized SQL)."""
        if not sql or not sql.strip():
            return {
                "ok": False,
                "statement_type": "",
                "rows": None,
                "row_count": 0,
                "pandas_result": "",
                "notice": None,
                "error": "Empty SQL",
                "timed_out": False,
            }
        key = (dataset_name, db_id, self._normalize_sql(sql))
        cached = self.exec_cache.get(key)
        if cached is not None and cached.get("ok", False) and not cached.get("timed_out", False):
            return cached
        res = call_sql_api(dataset_name, db_id, sql, mode=mode, timeout_ms=timeout_ms, max_rows=max_rows)
        if res.get("ok", False) and not res.get("timed_out", False):
            self.exec_cache[key] = res
        return res
    
    def _ingest_execution_results_into_cache(self, results_with_exec: List[Dict[str, Any]]) -> int:
        """Ingest successful execution results from worker outputs into self.exec_cache.

        This ensures the cache is populated even when execution happened in worker processes.
        Returns the number of new cache entries added.
        """
        added_count = 0
        for entry in results_with_exec:
            try:
                dataset_name = entry.get("dataset_name", "")
                db_id = entry.get("db_id", "")

                # Ground-truth SQL result
                gt_sql = entry.get("ground_truth_sql", "")
                gt_res = entry.get("gt_execution_result")
                if (
                    dataset_name
                    and db_id
                    and gt_sql
                    and isinstance(gt_res, dict)
                    and gt_res.get("ok", False)
                    and not gt_res.get("timed_out", False)
                ):
                    key_gt = (dataset_name, db_id, self._normalize_sql(gt_sql))
                    if key_gt not in self.exec_cache:
                        self.exec_cache[key_gt] = gt_res
                        added_count += 1

                # Candidate SQL results
                cand_sqls = entry.get("predicted_sqls", []) or []
                cand_results = entry.get("candidate_execution_results", []) or []
                for cand_sql, cand_res in zip(cand_sqls, cand_results):
                    if (
                        dataset_name
                        and db_id
                        and cand_sql
                        and isinstance(cand_res, dict)
                        and cand_res.get("ok", False)
                        and not cand_res.get("timed_out", False)
                    ):
                        key_cand = (dataset_name, db_id, self._normalize_sql(cand_sql))
                        if key_cand not in self.exec_cache:
                            self.exec_cache[key_cand] = cand_res
                            added_count += 1
            except Exception:
                # Ignore ingestion errors per-entry to avoid breaking the run
                continue
        return added_count

    def save_execution_cache(self) -> int:
        """Persist the execution cache, preferring the cache with more SQL entries.

        If an existing cache file is present on disk, this compares the number of
        entries (unique normalized SQL keys) between the current in-memory cache
        and the on-disk cache. The cache with more entries is kept. If equal,
        the existing on-disk cache is preserved to avoid unnecessary writes.

        Returns the number of entries in the resulting (kept) cache.
        """
        os.makedirs(os.path.dirname(self.exec_cache_file), exist_ok=True)

        current_len = len(self.exec_cache)
        # Do not save extremely large caches to disk to avoid huge files
        if current_len > 50000:
            print(
                f"Execution cache not saved because it exceeds the limit: {current_len} entries (> 100000)"
            )
            return current_len
        disk_cache: Dict[tuple, Dict[str, Any]] = {}
        disk_len = 0

        if os.path.exists(self.exec_cache_file):
            try:
                with open(self.exec_cache_file, "rb") as f:
                    disk_cache = pickle.load(f)
                disk_len = len(disk_cache)
            except Exception:
                disk_cache = {}
                disk_len = 0

        # Decide which cache to keep
        if disk_len >= current_len:
            # Keep existing cache; do not overwrite
            print(
                f"Execution cache kept existing file {self.exec_cache_file} (entries {disk_len}) "
                f"over in-memory (entries {current_len})"
            )
            return disk_len
        else:
            # Save in-memory cache; it has more entries
            with open(self.exec_cache_file, "wb") as f:
                pickle.dump(self.exec_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(
                f"Execution cache saved to {self.exec_cache_file} ({current_len} entries) "
                f"over existing ({disk_len} entries)"
            )
            return current_len
    
    def _compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        total_samples = len(results)
        
        # Metrics for multi-candidate recall@N
        has_any_sql_count = sum(1 for r in results if any((s or "").strip() for s in r.get("predicted_sqls", [])))
        execution_recall_count = sum(1 for r in results if r.get("any_execution_correct", False))
        
        # Failed samples are those with no correct candidate
        failed_samples = [r for r in results if not r.get("any_execution_correct", False)]
        failed_sample_ids = [r["sample_id"] for r in failed_samples]
        
        metrics = {
            "total_samples": total_samples,
            "has_any_sql_rate": has_any_sql_count / total_samples if total_samples else 0.0,
            "execution_recall_at_n": execution_recall_count / total_samples if total_samples else 0.0,
            "failed_samples": failed_samples,
            "failed_sample_ids": failed_sample_ids
        }
        
        return metrics
    
    def _save_results(self, results: List[Dict[str, Any]], metrics: Dict[str, Any]):
        """Save evaluation results to files."""
        # Save detailed results
        results_file = os.path.join(self.config['output_dir'], "detailed_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save metrics
        metrics_file = os.path.join(self.config['output_dir'], "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        
        # Save failed samples with IDs
        failed_samples_file = os.path.join(self.config['output_dir'], "failed_samples.json")
        failed_samples_data = {
            "failed_sample_ids": metrics["failed_sample_ids"],
            "failed_samples": metrics["failed_samples"]
        }
        with open(failed_samples_file, "w") as f:
            json.dump(failed_samples_data, f, indent=2, default=str)
        
        # Save summary
        summary_file = os.path.join(self.config['output_dir'], "summary.txt")
        with open(summary_file, "w") as f:
            f.write("BIRD Dev Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total samples: {metrics['total_samples']}\n")
            f.write(f"Has any-SQL rate: {metrics['has_any_sql_rate']:.3f}\n")
            f.write(f"Execution recall@N: {metrics['execution_recall_at_n']:.3f}\n")
            f.write(f"Failed sample IDs: {metrics['failed_sample_ids']}\n")
        
        print(f"Results saved to {self.config['output_dir']}")
        print(f"Has any-SQL rate: {metrics['has_any_sql_rate']:.3f}")
        print(f"Execution recall@N: {metrics['execution_recall_at_n']:.3f}")
        print(f"Failed samples: {len(metrics['failed_sample_ids'])}")
        print(f"Failed sample IDs: {metrics['failed_sample_ids']}")


def main():
    config = get_config()
    start_time = time.time()
    
    evaluator = BIRDEvaluator(config)
    metrics = evaluator.run_evaluation()
    
    elapsed_seconds = time.time() - start_time
    print("\nEvaluation completed!")
    print(f"Results saved to: {config['output_dir']}")
    print(f"Total running time: {elapsed_seconds:.2f}s ({timedelta(seconds=int(elapsed_seconds))})")


if __name__ == "__main__":
    main()
