#!/usr/bin/env python3
"""
Majority Voting Analysis for SQL Evaluation Results

This script performs majority voting using execution results embedded in
the evaluator's detailed_results.pkl. It does not execute SQL; it only
loads the PKL and groups candidates by their cached execution results.
"""

import json
import pickle
import argparse
import os
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
import multiprocessing as mp
import requests
import re
import sys
import subprocess as sp

# Global selection strategy configured via argparse and pool initializer
SELECTION_STRATEGY = "vav"

def _init_pool(strategy: str):
    global SELECTION_STRATEGY
    SELECTION_STRATEGY = (strategy or "vav").lower()

def choose_group_vav(groups):
    """
    Size-only voting with hard skip:
      - Chỉ xét các nhóm SUCCESS_VALUES:...
      - BỎ HẲN (skip) nhóm rỗng và nhóm toàn số 0 (không cộng/trừ điểm)
      - Chọn nhóm còn lại có size lớn nhất; tie-break theo key
      - Nếu sau lọc hết sạch, fallback về nhóm SUCCESS_VALUES lớn nhất, else 'NO_RESULTS'
    Input:
      - groups: Dict[str, Dict] giống result_groups trong code của bạn
      - question: str (không dùng ở bản này; giữ tham số để drop-in)
    Output:
      - best_key: str
    """
    import re

    if not groups:
        return "NO_RESULTS"

    def parse_vals(key: str):
        pfx = "SUCCESS_VALUES:"
        if not key.startswith(pfx):
            return []
        s = key[len(pfx):]
        if s == "":
            return []
        return [t.strip() for t in s.split(";") if t.strip() != ""]

    num_re = re.compile(r"^[\s\-+]?(\d+(\.\d+)?)$")
    def to_num(tok: str):
        # bỏ dấu % nếu có, chỉ để kiểm tra số 0/0.0
        t = tok[:-1].strip() if isinstance(tok, str) and tok.strip().endswith("%") else tok
        if not isinstance(t, str):
            t = str(t)
        m = num_re.match(t)
        return float(m.group(1)) if m else None

    def non_empty_count(vals):
        return sum(1 for v in vals if v != "")

    def is_empty(vals):
        return non_empty_count(vals) == 0

    def is_all_zero(vals):
        nums = [to_num(v) for v in vals if v != ""]
        nums = [x for x in nums if x is not None]
        return len(nums) > 0 and all(abs(x) < 1e-12 for x in nums)

    # 1) Lọc chỉ còn SUCCESS_VALUES
    sv_items = [(k, meta) for k, meta in groups.items() if k.startswith("SUCCESS_VALUES:")]

    # 2) Skip hẳn nhóm rỗng hoặc toàn 0
    filtered = []
    for k, meta in sv_items:
        vals = parse_vals(k)
        if is_empty(vals):
            continue
        if is_all_zero(vals):
            continue
        filtered.append((k, meta))

    # 3) Nếu còn ứng viên, chọn theo size lớn nhất; tie-break by key
    if filtered:
        best_key = max(filtered, key=lambda km: (int(km[1].get("size", 0)), km[0]))[0]
        return best_key

    # 4) Fallback: nếu lọc hết, chọn nhóm SUCCESS_VALUES lớn nhất ban đầu
    if sv_items:
        return max(sv_items, key=lambda km: (int(km[1].get("size", 0)), km[0]))[0]

    # 5) Không có SUCCESS_VALUES nào
    return "NO_RESULTS"


def is_select_like(sql: str) -> bool:
    s = sql.lstrip().lower()
    return s.startswith("select") or s.startswith("with") or s.startswith("explain")

def call_sql_api(dataset_name: str, db_id: str, sql: str, *, mode: Optional[str] = None, timeout_ms: Optional[int] = None, max_rows: Optional[int] = None) -> Dict[str, Any]:
    """Call SQL API to execute SQL queries."""
    API_URL = "http://192.168.1.108:8001/execute"
    TIMEOUT_MS_DEFAULT = 60000
    MAX_ROWS_DEFAULT = 10000
    
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

def execute_sql(dataset_name: str, db_id: str, sql: str) -> Dict[str, Any]:
    """Execute SQL once without retry."""
    return call_sql_api(dataset_name, db_id, sql)

def _rows_to_tuples(rows: Optional[List[Dict[str, Any]]]) -> Optional[List[tuple]]:
    """Convert rows to list of tuples for comparison."""
    if rows is None:
        return None
    if not isinstance(rows, list):
        return None
    try:
        # Convert each row to a tuple, handling None values
        tuples = []
        for row in rows:
            if isinstance(row, dict):
                # Sort keys for consistent ordering
                sorted_items = sorted(row.items())
                tuple_row = tuple(value if value is not None else "" for key, value in sorted_items)
                tuples.append(tuple_row)
        return tuples
    except Exception:
        return None

def _rows_to_value_tuples_agnostic(rows: Optional[List[Dict[str, Any]]]) -> Optional[List[tuple]]:
    """Convert rows to header-agnostic tuples of values (pure Python).
    - Ignores column names; uses only values per row
    - Sorts values within each row to avoid dependence on column order
    """
    if rows is None or not isinstance(rows, list):
        return None
    normalized_rows: List[tuple] = []
    for row in rows:
        if isinstance(row, dict):
            values = ["" if v is None else v for v in row.values()]
            values_sorted = sorted(map(lambda x: str(x), values))
            normalized_rows.append(tuple(values_sorted))
    return normalized_rows

def results_equal(true_rows: Optional[List[Dict[str, Any]]], pred_rows: Optional[List[Dict[str, Any]]]) -> bool:
    """Check equality header-agnostically using sets of per-row sorted value tuples."""
    true_vals = _rows_to_value_tuples_agnostic(true_rows)
    pred_vals = _rows_to_value_tuples_agnostic(pred_rows)

    if true_vals is None or pred_vals is None:
        return False

    true_set = set(true_vals)
    pred_set = set(pred_vals)
    return true_set == pred_set

def is_syntax_error(result: Dict[str, Any]) -> bool:
    """Check if the execution result indicates a syntax error.
    All SQL execution failures are considered syntax errors, except infrastructure errors.
    """
    if not result.get("ok", False):
        error = result.get("error", "").lower()
        # Infrastructure errors that should NOT be considered syntax errors
        infrastructure_error_indicators = [
            "timeout",
            "network",
            "http",
            "request",
            "api",
            "server",
            "service unavailable",
            "timed out",
            "connection refused",
            "connection timeout"
        ]
        # Database connection errors (not SQL syntax errors)
        database_connection_errors = [
            "connection refused",
            "connection timeout",
            "database connection",
            "connection pool"
        ]
        # If it's an infrastructure error, don't count as syntax error
        if any(indicator in error for indicator in infrastructure_error_indicators):
            return False
        # If it's a database connection error (not SQL syntax), don't count as syntax error
        if any(indicator in error for indicator in database_connection_errors):
            return False
        # All other SQL execution failures are syntax errors (including "no such table", "no such column", etc.)
        return True
    return False

def normalize_execution_result(result: Dict[str, Any]) -> str:
    """Normalize execution result to a header-agnostic signature for grouping via NumPy values.
    - Failures: error prefix
    - Successes: signature from per-row sorted values (ignores headers and column order)
    """
    if not result.get("ok", False):
        error = result.get("error", "Unknown error")
        return f"ERROR: {error[:100]}"

    vals = _rows_to_value_tuples_agnostic(result.get("rows"))
    if vals is None:
        row_count = result.get("row_count", 0)
        return f"SUCCESS_ROWS_COUNT:{row_count}"
    row_strings = ["|".join(map(str, row)) for row in vals]
    signature = ";".join(sorted(set(row_strings)))
    return f"SUCCESS_VALUES:{signature[:200]}"

def build_cache_key(sample_id: Any, sql: str) -> str:
    return f"{sample_id}:::{(sql or '').strip()}"

def majority_voting_for_sample(sample: Dict[str, Any], selection_strategy: Optional[str] = None) -> Dict[str, Any]:
    """Perform majority voting for a single sample."""
    sample_id = sample["sample_id"]
    dataset_name = str(sample.get("dataset_name", ""))
    db_id = str(sample.get("db_id", ""))
    question = sample["question"]
    ground_truth_sql = sample["ground_truth_sql"]
    predicted_sqls = sample["predicted_sqls"]
    gt_result_cached = sample.get("gt_execution_result")
    candidate_execution_results_cached = sample.get("candidate_execution_results")
    
    # print(f"Processing sample {sample_id}: {question[:50]}...")
    
    pred_results = []
    # If cached execution results exist, use them; otherwise execute via API
    if gt_result_cached is not None and candidate_execution_results_cached is not None:
        # print(f"  Using cached ground truth SQL result...")
        gt_result = gt_result_cached
        gt_rows = gt_result.get("rows") if gt_result.get("ok") else None
        # print(f"  Evaluating {len(predicted_sqls)} predicted SQLs from cache...")
        for i, sql in enumerate(tqdm(predicted_sqls, desc="    Candidates", leave=False)):
            cached_result = candidate_execution_results_cached[i] if i < len(candidate_execution_results_cached) else None
            if not sql or not sql.strip():
                pred_results.append({"sql": sql, "result": {"ok": False, "error": "Empty SQL"}, "index": i})
                continue
            if isinstance(cached_result, dict):
                result = cached_result
            else:
                result = {"ok": False, "error": "No cached result"}
                # Explicitly log missing cached results for visibility
                preview = (sql or "").replace("\n", " ")[:120]
                print(f"    [WARN] sample {sample_id}: candidate #{i} has no cached result. SQL preview: {preview}")
            pred_results.append({"sql": sql, "result": result, "index": i})
    else:
        # print(f"  Executing ground truth SQL via API...")
        gt_result = execute_sql(dataset_name, db_id, ground_truth_sql)
        gt_rows = gt_result.get("rows") if gt_result.get("ok") else None
        # print(f"  Executing {len(predicted_sqls)} predicted SQLs via API...")
        for i, sql in enumerate(tqdm(predicted_sqls, desc="    Executing SQLs", leave=False)):
            if not sql or not sql.strip():
                pred_results.append({"sql": sql, "result": {"ok": False, "error": "Empty SQL"}, "index": i})
                continue
            result = execute_sql(dataset_name, db_id, sql)
            pred_results.append({"sql": sql, "result": result, "index": i})
    
    # Filter out syntax errors and group predicted SQLs by execution result
    valid_pred_results = []
    syntax_error_count = 0
    infrastructure_failures: List[Dict[str, Any]] = []
    
    for pred in pred_results:
        if is_syntax_error(pred["result"]):
            syntax_error_count += 1
        else:
            # non-syntax: either ok or failed for infrastructure reasons
            valid_pred_results.append(pred)
            if not pred["result"].get("ok", False):
                infrastructure_failures.append(pred)
    
    # Group valid predicted SQLs by execution result (only successful executions)
    result_groups = defaultdict(list)
    for pred in valid_pred_results:
        # Only group successful executions, not errors
        if pred["result"].get("ok", False):
            normalized_result = normalize_execution_result(pred["result"])
            result_groups[normalized_result].append(pred)
    
    # Sort groups by size (descending)
    sorted_groups = sorted(result_groups.items(), key=lambda kv: len(kv[1]), reverse=True)

    # Find majority group (group with most SQLs)
    if not sorted_groups:
        majority_group = []
        majority_result = "NO_RESULTS"
    else:
        majority_result, majority_group = sorted_groups[0]

    # Decide selection strategy and choose group
    strategy = (selection_strategy or SELECTION_STRATEGY or "vav").lower()
    chosen_group_items: List[Dict[str, Any]] = []
    if strategy == "majority":
        chosen_group_items = majority_group
    else:
        groups_meta = {}
        for res_key, items in sorted_groups:
            group_is_correct = False
            if gt_result.get("ok", False) and items and items[0]["result"].get("ok", False):
                group_rows = items[0]["result"].get("rows")
                group_is_correct = results_equal(gt_rows, group_rows)
            groups_meta[res_key] = {
                "size": len(items),
                "is_majority": (res_key == majority_result),
                "is_correct": group_is_correct,
                "sqls": [{"index": item["index"], "sql": item["sql"]} for item in items],
            }
        vav_result_key = choose_group_vav(groups_meta)
        chosen_group_items = result_groups.get(vav_result_key, [])

    # Select one SQL from the chosen group (first one)
    if chosen_group_items:
        selected_sql = chosen_group_items[0]["sql"]
        selected_result = chosen_group_items[0]["result"]
    else:
        selected_sql = ""
        selected_result = {"ok": False, "error": "No valid SQLs"}
    
    # Check if selected SQL is correct
    is_correct = False
    if selected_result.get("ok", False) and gt_result.get("ok", False):
        selected_rows = selected_result.get("rows")
        is_correct = results_equal(gt_rows, selected_rows)
    
    # Log each group info with correctness
    for grp_idx, (res_key, items) in enumerate(sorted_groups, start=1):
        group_is_correct = False
        if gt_result.get("ok", False) and items and items[0]["result"].get("ok", False):
            group_rows = items[0]["result"].get("rows")
            group_is_correct = results_equal(gt_rows, group_rows)
        # print(f"    Group {grp_idx}: size={len(items)}, is_correct={'TRUE' if group_is_correct else 'FALSE'}")

    # Log infrastructure failure breakdown
    if infrastructure_failures:
        reason_counts: Dict[str, int] = defaultdict(int)
        for item in infrastructure_failures:
            msg = str(item.get("result", {}).get("error", "")).strip()
            reason = (msg or "UNKNOWN").split("\n", 1)[0][:120]
            reason_counts[reason] += 1
        top_reasons = sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True)
        # print(f"    Infrastructure failures: {len(infrastructure_failures)}")
        # for reason, cnt in top_reasons[:5]:
        #     print(f"      - {cnt}x: {reason}")

    # Totals: ensure group sizes equal number of valid successful SQLs
    grouped_total = sum(len(items) for _, items in sorted_groups)
    valid_success_count = sum(1 for p in valid_pred_results if p["result"].get("ok", False))
    # print(f"    Groups total={grouped_total}, valid_success_sqls={valid_success_count}")
    assert grouped_total == valid_success_count, (
        f"Group size sum {grouped_total} != valid successful count {valid_success_count} for sample {sample_id}"
    )

    # Prepare result
    result = {
        "sample_id": sample_id,
        "question": question,
        "ground_truth_sql": ground_truth_sql,
        "ground_truth_execution_success": gt_result.get("ok", False),
        "ground_truth_error": gt_result.get("error") if not gt_result.get("ok") else None,
        "ground_truth_execution_result": gt_result,
        "ground_truth_result_key": normalize_execution_result(gt_result),
        "num_predicted_sqls": len(predicted_sqls),
        "num_syntax_errors": syntax_error_count,
        # Redefined to count only successful (ok=True) non-syntax-error SQLs
        "num_valid_sqls_after_filtering": valid_success_count,
        "num_infrastructure_failures": len(infrastructure_failures),
        "infrastructure_failure_summary": {
            reason: count for reason, count in (
                sorted(
                    (
                        lambda rc: rc.items()
                    )(
                        (lambda xs: { (str(x.get('result', {}).get('error', '')).strip() or 'UNKNOWN').split('\n',1)[0][:120]: xs.count(x) for x in xs })
                        (infrastructure_failures)
                    ),
                    key=lambda kv: kv[1], reverse=True
                )
            )
        },
        "num_result_groups": len(result_groups),
        "majority_group_size": len(majority_group),
        "majority_result": majority_result,
        "selected_sql": selected_sql,
        "selected_sql_index": chosen_group_items[0]["index"] if chosen_group_items else -1,
        "is_sample_correct": is_correct,
        "result_groups": {
            result: {
                "size": len(items),
                "is_majority": result == majority_result,
                "is_correct": (
                    bool(gt_result.get("ok", False))
                    and bool(items)
                    and bool(items[0]["result"].get("ok", False))
                    and results_equal(gt_rows, items[0]["result"].get("rows"))
                ),
                "sqls": [{"index": item["index"], "sql": item["sql"][:100]} for item in items]
            }
            for result, items in sorted_groups
        }
    }
    
    # print(f"  Result: {'CORRECT' if is_correct else 'INCORRECT'} (strategy: {strategy}, chosen group size: {len(chosen_group_items)}, majority group size: {len(majority_group)})")
    return result

def process_sample_safe(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper to safely process a single sample, returning an error record on failure.
    This is suitable for use with multiprocessing pools where exceptions must not
    terminate the whole job.
    """
    return majority_voting_for_sample(sample, None)

def main():
    parser = argparse.ArgumentParser(description="Majority voting analysis for SQL evaluation results")
    parser.add_argument(
        "--input-pkl",
        type=str,
        default="/home/datht/mats/sql_writer/evaluation/evaluation_results-3b/detailed_results.pkl",
        help="Path to detailed_results.pkl produced by evaluate_bird_dev.py"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="/home/datht/mats/sql_writer/evaluation/evaluation_results-3b/detailed_results.json",
        help="Fallback path to detailed_results.json when PKL is unavailable"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/home/datht/mats/sql_writer/evaluation/majority_voting_results.json",
        help="Path to save majority voting results"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum number of samples to process (for testing)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel worker processes (set 1 to disable multiprocessing)"
    )
    parser.add_argument(
        "--selection",
        type=str,
        choices=["majority", "vav"],
        default="vav",
        help="Selection strategy: 'majority' (top-1 by size) or 'vav' (value-aware)"
    )
    parser.add_argument(
        "--focus-sample-id",
        type=int,
        default=None,
        help="If provided, process only this sample_id for debugging"
    )
    
    args = parser.parse_args()
    
    print("Majority Voting Analysis for SQL Evaluation")
    print("=" * 50)
    
    # Load detailed results: try PKL, fallback to JSON
    samples = None
    try:
        if args.input_pkl:
            with open(args.input_pkl, 'rb') as pf:
                loaded_obj = pickle.load(pf)
            # Detect and transform execution cache format (dict keyed by sample_id)
            if isinstance(loaded_obj, dict) and loaded_obj and all(isinstance(v, dict) for v in loaded_obj.values()):
                print(f"Loaded execution cache from {args.input_pkl}; transforming to samples list...")
                transformed: List[Dict[str, Any]] = []
                for key, entry in loaded_obj.items():
                    # Ensure candidates are sorted by index for alignment
                    candidates = list(entry.get("candidates", []))
                    try:
                        candidates.sort(key=lambda c: int(c.get("index", 0)))
                    except Exception:
                        pass
                    predicted_sqls = [c.get("sql", "") for c in candidates]
                    candidate_exec_results = [c.get("result") for c in candidates]
                    gt = entry.get("ground_truth", {}) or {}
                    transformed.append({
                        "sample_id": entry.get("sample_id", key),
                        "dataset_name": entry.get("dataset_name", ""),
                        "db_id": entry.get("db_id", ""),
                        "question": entry.get("question", ""),
                        "ground_truth_sql": gt.get("sql", ""),
                        "predicted_sqls": predicted_sqls,
                        "gt_execution_result": gt.get("result"),
                        "candidate_execution_results": candidate_exec_results,
                    })
                samples = transformed
                print(f"Transformed {len(samples)} samples from execution cache.")
            elif isinstance(loaded_obj, list):
                samples = loaded_obj
                print(f"Loaded samples list with exec from {args.input_pkl} (len={len(samples)})")
            else:
                samples = None
                print(f"Unrecognized PKL structure in {args.input_pkl}; will fallback to JSON")
    except Exception as e:
        print(f"Failed to load input PKL {args.input_pkl}: {e}")
    if samples is None:
        print(f"Loading JSON results from {args.input_file} and will execute via API...")
        with open(args.input_file, 'r') as f:
            samples = json.load(f)
    
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
        print(f"Processing first {args.max_samples} samples for testing")
    
    # Optionally focus on a single sample for debugging
    if args.focus_sample_id is not None:
        samples = [s for s in samples if str(s.get("sample_id")) == str(args.focus_sample_id)]
        print(f"Focusing on sample_id={args.focus_sample_id}. Samples to process: {len(samples)}")
    else:
        print(f"Total samples to process: {len(samples)}")
    
    # Process each sample (optionally in parallel)
    results = []
    correct_count = 0
    total_syntax_errors = 0
    total_valid_sqls = 0

    workers = max(1, int(args.workers)) if hasattr(args, "workers") and args.workers is not None else 1
    if workers > 1:
        print(f"Processing samples in parallel with {workers} workers...")
        with mp.Pool(processes=workers, initializer=_init_pool, initargs=(args.selection,)) as pool:
            for result in tqdm(
                pool.imap_unordered(process_sample_safe, samples, chunksize=1),
                total=len(samples),
                desc="Processing samples"
            ):
                results.append(result)
                if result.get("is_sample_correct"):
                    correct_count += 1
                total_syntax_errors += result.get("num_syntax_errors", 0)
                total_valid_sqls += result.get("num_valid_sqls_after_filtering", 0)
    else:
        _init_pool(args.selection)
        for sample in tqdm(samples, desc="Processing samples"):
            result = process_sample_safe(sample)
            results.append(result)
            if result.get("is_sample_correct"):
                correct_count += 1
            total_syntax_errors += result.get("num_syntax_errors", 0)
            total_valid_sqls += result.get("num_valid_sqls_after_filtering", 0)
    
    # Calculate overall accuracy
    total_samples = len(results)
    accuracy = correct_count / total_samples if total_samples > 0 else 0.0
    
    # Calculate syntax error statistics
    total_predicted_sqls = sum(len(sample.get("predicted_sqls", [])) for sample in samples)
    syntax_error_rate = total_syntax_errors / total_predicted_sqls if total_predicted_sqls > 0 else 0.0
    
    # Prepare summary
    summary = {
        "total_samples": total_samples,
        "correct_samples": correct_count,
        "accuracy": accuracy,
        "accuracy_percentage": f"{accuracy * 100:.2f}%",
        "total_predicted_sqls": total_predicted_sqls,
        "total_syntax_errors": total_syntax_errors,
        "syntax_error_rate": f"{syntax_error_rate:.4f} ({syntax_error_rate * 100:.2f}%)",
        "total_valid_sqls_after_filtering": total_valid_sqls
    }
    
    # Save results
    output_data = {
        "summary": summary,
        "results": results
    }
    
    print(f"\nSaving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 50)
    print("MAJORITY VOTING RESULTS SUMMARY")
    print("=" * 50)
    print(f"Total samples: {total_samples}")
    print(f"Correct samples: {correct_count}")
    print(f"Overall accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Total predicted SQLs: {total_predicted_sqls}")
    print(f"Total syntax errors: {total_syntax_errors}")
    print(f"Syntax error rate: {syntax_error_rate:.4f} ({syntax_error_rate * 100:.2f}%)")
    print(f"Valid SQLs after filtering: {total_valid_sqls}")
    print(f"Results saved to: {args.output_file}")
    
    # Export format depending on dataset: BIRD -> predict_dev.json, Spider -> pred_dev.sql
    pkl_dir = os.path.dirname(args.input_pkl) if args.input_pkl else os.path.dirname(args.input_file)
    # Determine dataset by inspecting samples; fallback to input path
    dataset_name_concat = " ".join([str(s.get("dataset_name", "")).lower() for s in samples])
    input_path_lower = (args.input_pkl or args.input_file or "").lower()
    is_bird = ("bird" in dataset_name_concat) or ("bird" in input_path_lower)
    is_spider = ("spider" in dataset_name_concat) or ("spider" in input_path_lower)

    # Sort results by sample_id for stable export order
    sorted_results = sorted(results, key=lambda x: int(x.get("sample_id", 0)))

    if is_bird and not is_spider:
        bird_output_file = os.path.join(pkl_dir, "predict_dev.json")
        print(f"\nExporting BIRD format to {bird_output_file}...")
        bird_results_list = []

        # Create a mapping from sample_id to question and db_id for BIRD format
        sample_id_to_question = {}
        sample_id_to_db_id = {}
        for sample in samples:
            sample_id_to_question[str(sample.get("sample_id"))] = sample.get("question", "")
            sample_id_to_db_id[str(sample.get("sample_id"))] = sample.get("db_id", "")

        # Build BIRD format results as list of lists
        for result in sorted_results:
            sample_id = str(result.get("sample_id", ""))
            question = sample_id_to_question.get(sample_id, "")
            selected_sql = result.get("selected_sql", "")
            db_id = sample_id_to_db_id.get(sample_id, "")
            bird_format_entry = f"{selected_sql}\t----- bird -----\t{db_id}"
            bird_results_list.append([question, bird_format_entry])

        with open(bird_output_file, "w", encoding='utf-8') as f:
            f.write(json.dumps(bird_results_list, indent=2, ensure_ascii=False))
        print(f"BIRD format exported to: {bird_output_file}")
        print(f"Total entries: {len(bird_results_list)}")

    elif is_spider and not is_bird:
        spider_pred_path = os.path.join(pkl_dir, "pred_dev.sql")
        print(f"\nExporting Spider format to {spider_pred_path}...")
        # Prepare an index mapping to later join with Spider's evaluation output.
        # Build lookup for db_id and question from original samples.
        sample_id_to_question = {}
        sample_id_to_db_id = {}
        for s in samples:
            sample_id_to_question[str(s.get("sample_id"))] = s.get("question", "")
            sample_id_to_db_id[str(s.get("sample_id"))] = s.get("db_id", "")

        pred_index_mapping = []  # list of {sample_id, db_id, question, selected_sql}
        with open(spider_pred_path, "w", encoding="utf-8") as f:
            for result in sorted_results:
                sql = str(result.get("selected_sql", "") or "")
                sql_one_line = re.sub(r"\s+", " ", sql.replace("\r", " ").replace("\n", " ")).strip()
                # Avoid blank lines which the evaluator treats as session separators
                safe_sql = sql_one_line if sql_one_line != "" else "SELECT 1"
                f.write(safe_sql + "\n")
                sid = str(result.get("sample_id"))
                pred_index_mapping.append({
                    "sample_id": result.get("sample_id"),
                    "db_id": sample_id_to_db_id.get(sid, ""),
                    "question": sample_id_to_question.get(sid, ""),
                    "selected_sql": safe_sql,
                })
        print(f"Spider format exported to: {spider_pred_path}")
        # Persist mapping alongside pred file
        pred_map_path = os.path.join(pkl_dir, "pred_index_mapping.json")
        try:
            with open(pred_map_path, "w", encoding="utf-8") as mf:
                json.dump(pred_index_mapping, mf, indent=2, ensure_ascii=False)
            print(f"Pred index mapping saved to: {pred_map_path}")
        except Exception as e:
            print(f"[WARN] Failed to save pred index mapping: {e}")

        # Run Spider evaluation using the official evaluator
        # Auto-detect Spider dataset root (contains dev_gold.sql, database/, tables.json)
        candidate_roots = [
            "/home/datht/mats/data/spider",
            "/home/datht/mats/data/sft_data_collections/spider",
            "/home/datht/mats/data/Spider",
        ]
        spider_root = None
        for root in candidate_roots:
            if (
                os.path.exists(os.path.join(root, "dev_gold.sql"))
                and os.path.isdir(os.path.join(root, "database"))
                and os.path.exists(os.path.join(root, "tables.json"))
            ):
                spider_root = root
                break
        if spider_root is None:
            print("[WARN] Could not auto-detect Spider dataset root; using default '/home/datht/mats/data/spider'")
            spider_root = "/home/datht/mats/data/spider"

        gold_path = os.path.join(spider_root, "dev_gold.sql")
        db_path = os.path.join(spider_root, "database")
        table_path = os.path.join(spider_root, "tables.json")

        eval_cmd = [
            sys.executable,
            "/home/datht/schema-linking-benchmark/MAC-SQL/evaluation/evaluation_spider.py",
            "--gold", gold_path,
            "--db", db_path,
            "--table", table_path,
            "--pred", spider_pred_path,
            "--etype", "exec",
        ]
        print("\nRunning Spider evaluation...\n" + " ".join(eval_cmd))
        try:
            res = sp.run(eval_cmd, capture_output=True, text=True)
            stdout_path = os.path.join(pkl_dir, "spider_eval_stdout.txt")
            stderr_path = os.path.join(pkl_dir, "spider_eval_stderr.txt")
            with open(stdout_path, "w", encoding="utf-8") as f_out:
                f_out.write(res.stdout or "")
            with open(stderr_path, "w", encoding="utf-8") as f_err:
                f_err.write(res.stderr or "")
            print(f"Spider evaluation finished with return code {res.returncode}")
            # Print outputs directly to the console
            if (res.stdout or "").strip():
                print("\n[Spider eval stdout]\n" + res.stdout)
            if (res.stderr or "").strip():
                print("\n[Spider eval stderr]\n" + res.stderr)
            print(f"Stdout saved to: {stdout_path}")
            if res.stderr:
                print(f"Stderr saved to: {stderr_path}")

            # Auto-recover: install nltk if missing, then retry
            if res.returncode != 0 and (res.stderr or "").find("No module named 'nltk'") != -1:
                print("nltk not found in current env. Attempting to install and retry...")
                pip_cmd = [sys.executable, "-m", "pip", "install", "nltk"]
                pip_res = sp.run(pip_cmd, capture_output=True, text=True)
                if pip_res.returncode != 0:
                    print("Failed to install nltk. See pip stderr below:")
                    print((pip_res.stderr or "").splitlines()[-1:] or pip_res.stderr)
                else:
                    # Download tokenizer data commonly required by evaluation scripts
                    sp.run([sys.executable, "-c", "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"], capture_output=True, text=True)
                    print("Retrying Spider evaluation after installing nltk...")
                    res2 = sp.run(eval_cmd, capture_output=True, text=True)
                    with open(stdout_path, "a", encoding="utf-8") as f_out:
                        f_out.write("\n\n=== RETRY AFTER NLTK INSTALL ===\n")
                        f_out.write(res2.stdout or "")
                    with open(stderr_path, "a", encoding="utf-8") as f_err:
                        f_err.write("\n\n=== RETRY AFTER NLTK INSTALL ===\n")
                        f_err.write(res2.stderr or "")
                    print(f"Retry finished with return code {res2.returncode}")
                    if (res2.stdout or "").strip():
                        print("\n[Spider eval stdout - retry]\n" + res2.stdout)
                    if (res2.stderr or "").strip():
                        print("\n[Spider eval stderr - retry]\n" + res2.stderr)
        except Exception as e:
            print(f"Failed to run Spider evaluation: {e}")

        # Build mismatch log: majority-correct but Spider incorrect
        try:
            spider_eval_json = os.path.join(pkl_dir, "evaluation.json")
            if not os.path.exists(spider_eval_json):
                # Some versions may write to spider_eval_stdout.txt only
                print(f"[WARN] Spider evaluation JSON not found at {spider_eval_json}; skipping mismatch log generation")
            else:
                with open(spider_eval_json, "r", encoding="utf-8") as ef:
                    spider_entries = json.load(ef)
                # Load mapping just written
                with open(pred_map_path, "r", encoding="utf-8") as mf:
                    idx_map = json.load(mf)
                # Build quick index: sample_id -> majority result entry
                maj_by_sample_id = {}
                for r in results:
                    maj_by_sample_id[str(r.get("sample_id"))] = r
                def _norm(s: str) -> str:
                    s = (s or "").replace("\r", " ").replace("\n", " ")
                    return re.sub(r"\s+", " ", s).strip()
                # Iterate paired by index
                mismatches: List[Dict[str, Any]] = []
                infra_timeout_count = 0
                infra_conn_aborted_count = 0
                infra_other_count = 0
                infra_skipped = 0

                def _is_infra_error(msg: Optional[str]) -> Tuple[bool, Optional[str]]:
                    if not msg:
                        return False, None
                    low = str(msg).lower()
                    if "timeout" in low or "timed out" in low:
                        return True, "timeout"
                    if "connection reset" in low or "connection aborted" in low or "connection refused" in low:
                        return True, "connection_aborted"
                    if any(k in low for k in ["network", "http", "request", "api", "server", "service unavailable"]):
                        return True, "infrastructure"
                    return False, None
                for idx, sp_entry in enumerate(spider_entries):
                    # Safety if lengths diverge
                    if idx >= len(idx_map):
                        break
                    map_item = idx_map[idx]
                    sample_id = str(map_item.get("sample_id"))
                    maj_res = maj_by_sample_id.get(sample_id)
                    if not maj_res:
                        continue
                    # Consider only majority-correct samples
                    if not bool(maj_res.get("is_sample_correct", False)):
                        continue
                    spider_ok = bool(sp_entry.get("exec_result", False))
                    if spider_ok:
                        continue
                    # Compose log entry
                    db_id = str(map_item.get("db_id", ""))
                    question = map_item.get("question", "")
                    maj_gt_sql = maj_res.get("ground_truth_sql", "")
                    maj_sel_sql = map_item.get("selected_sql", "")
                    # Majority side exec results (via API, best-effort) - include full return structure
                    maj_gt_res = None
                    maj_sel_res = None
                    try:
                        gt_res = execute_sql(dataset_name="spider", db_id=db_id, sql=maj_gt_sql)  # type: ignore[arg-type]
                        maj_gt_res = {
                            "ok": bool(gt_res.get("ok", False)),
                            "row_count": gt_res.get("row_count"),
                            "rows": gt_res.get("rows"),
                            "error": gt_res.get("error"),
                        }
                    except Exception:
                        maj_gt_res = None
                    try:
                        sel_res = execute_sql(dataset_name="spider", db_id=db_id, sql=maj_sel_sql)  # type: ignore[arg-type]
                        maj_sel_res = {
                            "ok": bool(sel_res.get("ok", False)),
                            "row_count": sel_res.get("row_count"),
                            "rows": sel_res.get("rows"),
                            "error": sel_res.get("error"),
                        }
                    except Exception:
                        maj_sel_res = None
                    # Check infra errors to filter out
                    is_infra_gt, tag_gt = _is_infra_error((maj_gt_res or {}).get("error") if isinstance(maj_gt_res, dict) else None)
                    is_infra_sel, tag_sel = _is_infra_error((maj_sel_res or {}).get("error") if isinstance(maj_sel_res, dict) else None)
                    if is_infra_gt or is_infra_sel:
                        tag = tag_gt or tag_sel
                        if tag == "timeout":
                            infra_timeout_count += 1
                        elif tag == "connection_aborted":
                            infra_conn_aborted_count += 1
                            # Print detailed line for connection aborted cases
                            err_msg = None
                            if isinstance(maj_sel_res, dict) and maj_sel_res.get("error"):
                                err_msg = str(maj_sel_res.get("error"))
                            elif isinstance(maj_gt_res, dict) and maj_gt_res.get("error"):
                                err_msg = str(maj_gt_res.get("error"))
                            preview_sql = (maj_sel_sql or maj_gt_sql)[:200]
                            preview_q = (question or "")[:160]
                            print(
                                f"[INFRA] connection_aborted sample_id={sample_id} db_id={db_id} "
                                f"sql='{preview_sql}' error='{(err_msg or '')[:160]}'\n  Q: {preview_q}"
                            )
                        else:
                            infra_other_count += 1
                        infra_skipped += 1
                        continue

                    entry = {
                        "sample_id": sample_id,
                        "db_id": db_id,
                        "question": question,
                        "majority": {
                            "gt_sql": maj_gt_sql,
                            "sample_id": sample_id,
                            "gt_result": maj_gt_res,
                            "selected_result": maj_sel_res,
                        },
                        "spider_eval": {
                            "gt_sql": sp_entry.get("gold", ""),
                            "sample_id": sample_id,
                            "db_id": db_id,
                            "exec_result": bool(sp_entry.get("exec_result", False)),
                            "pred_sql": sp_entry.get("predictSQL", "") or map_item.get("selected_sql", ""),
                        },
                    }
                    mismatches.append(entry)
                out_log = os.path.join(pkl_dir, "spider_mismatch_log.json")
                with open(out_log, "w", encoding="utf-8") as lf:
                    json.dump(mismatches, lf, indent=2, ensure_ascii=False)
                print(f"Mismatch log written to: {out_log} (entries: {len(mismatches)})")
                if infra_skipped > 0:
                    print("[WARN] Infrastructure errors encountered during majority-side exec approximation; skipped from mismatches")
                    print(f"         timeouts={infra_timeout_count}, connection_aborted={infra_conn_aborted_count}, other_infra={infra_other_count}, total_skipped={infra_skipped}")
                # Save infra summary
                infra_summary_path = os.path.join(pkl_dir, "spider_infra_summary.json")
                with open(infra_summary_path, "w", encoding="utf-8") as sf:
                    json.dump({
                        "timeouts": infra_timeout_count,
                        "connection_aborted": infra_conn_aborted_count,
                        "other_infra": infra_other_count,
                        "skipped_from_mismatch": infra_skipped,
                    }, sf, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to build mismatch log: {e}")
    else:
        # Unknown or mixed dataset; skip special export
        print("\nDataset not clearly identified as BIRD or Spider; skipping dataset-specific export.")

if __name__ == "__main__":
    main()
