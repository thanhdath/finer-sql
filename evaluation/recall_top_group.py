#!/usr/bin/env python3
"""
Compute recall@2 using top-2 result groups from majority_voting_results.json.

Notes/assumptions:
- majority_voting_results.json contains only truncated SQLs inside groups; we
  recover full SQL via predicted_sqls[index] from the detailed results file.
- We execute ground-truth SQL and the two candidate SQLs via the existing SQL
  API and compare result sets (order-insensitive) to determine correctness.

Inputs:
- --input-file: path to majority_voting_results.json (required)
- --details-file: path to the detailed_results.json produced during evaluation
                  (used to recover full SQLs by index). Defaults to the same
                  path used by majority_voting.py.
- --output-file: where to write recall summary and per-sample details (JSON).

Output JSON structure:
{
  "summary": {
    "total_samples": int,
    "evaluable_samples": int,         # GT executed successfully and we found candidates
    "recall_at_1": float,             # proportion over evaluable_samples
    "recall_at_2": float,
  },
  "results": [
    {
      "sample_id": int,
      "gt_ok": bool,
      "top_groups": [
        {"result_key": str, "size": int, "candidate_index": int}
      ],
      "candidates": [
        {"index": int, "executed_ok": bool, "matched_gt": bool}
      ],
      "recall_at_1": bool,
      "recall_at_2": bool,
      "notes": str | null
    },
    ...
  ]
}
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm


def is_select_like(sql: str) -> bool:
    s = sql.lstrip().lower()
    return s.startswith("select") or s.startswith("with") or s.startswith("explain")


def call_sql_api(dataset_name: str, db_id: str, sql: str, *, mode: Optional[str] = None, timeout_ms: Optional[int] = None, max_rows: Optional[int] = None) -> Dict[str, Any]:
    """Call SQL API to execute SQL queries.

    Mirrors the behavior used in other evaluation scripts for consistency.
    """
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
    except Exception as e:  # noqa: BLE001
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


def _rows_to_tuples(rows: Optional[List[Dict[str, Any]]]) -> Optional[List[tuple]]:
    """Convert rows to list of tuples for order-insensitive comparison."""
    if rows is None:
        return None
    if not isinstance(rows, list):
        return None
    try:
        tuples: List[tuple] = []
        for row in rows:
            if isinstance(row, dict):
                # Sort keys for consistent ordering
                sorted_items = sorted(row.items())
                tuple_row = tuple(value if value is not None else "" for _key, value in sorted_items)
                tuples.append(tuple_row)
        return tuples
    except Exception:  # noqa: BLE001
        return None


def results_equal(true_rows: Optional[List[Dict[str, Any]]], pred_rows: Optional[List[Dict[str, Any]]]) -> bool:
    true_tuples = _rows_to_tuples(true_rows)
    pred_tuples = _rows_to_tuples(pred_rows)
    if true_tuples is None or pred_tuples is None:
        return False
    return set(true_tuples) == set(pred_tuples)


def pick_top_groups(result_groups: Dict[str, Any], k: int = 2) -> List[Tuple[str, Dict[str, Any]]]:
    """Return the top-k groups by size as list of (group_key, group_value)."""
    items = list(result_groups.items())
    items.sort(key=lambda kv: int(kv[1].get("size", 0)), reverse=True)
    return items[:k]


def main():
    parser = argparse.ArgumentParser(description="Compute recall@2 from majority voting groups")
    parser.add_argument(
        "--input-file",
        type=str,
        default="/home/datht/mats/sql_writer/evaluation/majority_voting_results.json",
        help="Path to majority_voting_results.json",
    )
    parser.add_argument(
        "--details-file",
        type=str,
        default="/home/datht/mats/sql_writer/evaluation/evaluation_results-3b/detailed_results.json",
        help="Path to detailed_results.json (for full predicted_sqls & metadata)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/home/datht/mats/sql_writer/evaluation/recall_top_group_results.json",
        help="Where to save recall results JSON",
    )
    args = parser.parse_args()

    # Load inputs
    with open(args.input_file, "r") as f:
        mv_data: Dict[str, Any] = json.load(f)

    with open(args.details_file, "r") as f:
        detailed_results: List[Dict[str, Any]] = json.load(f)

    # Build lookup by sample_id for detailed results
    details_by_id: Dict[int, Dict[str, Any]] = {}
    for rec in detailed_results:
        try:
            sid = int(rec["sample_id"])  # sample_id present in evaluate_bird_dev outputs
            details_by_id[sid] = rec
        except Exception:  # noqa: BLE001
            continue

    results: List[Dict[str, Any]] = []

    total_samples = 0
    evaluable_samples = 0
    recall1_count = 0
    recall2_count = 0

    samples_list: List[Dict[str, Any]] = mv_data.get("results", []) or []
    for sample in tqdm(samples_list, total=len(samples_list), desc="Computing recall@2"):
        total_samples += 1

        sample_id = int(sample.get("sample_id", -1))
        gt_sql = sample.get("ground_truth_sql", "")
        result_groups = sample.get("result_groups", {}) or {}
        selected_sql_index = int(sample.get("selected_sql_index", -1))

        # Recover detailed record for dataset metadata and predicted SQLs
        detail = details_by_id.get(sample_id)
        if not detail:
            results.append({
                "sample_id": sample_id,
                "gt_ok": False,
                "top_groups": [],
                "candidates": [],
                "recall_at_1": False,
                "recall_at_2": False,
                "notes": "No detailed record found for sample_id (cannot execute)",
            })
            continue

        dataset_name = detail.get("dataset_name", "")
        db_id = detail.get("db_id", "")
        predicted_sqls: List[str] = detail.get("predicted_sqls", []) or []

        # Identify top-2 groups
        top_groups = pick_top_groups(result_groups, k=2)
        group_summaries: List[Dict[str, Any]] = []

        # Candidate indices: majority -> selected_sql_index; runner-up -> first index in its sqls list
        candidate_indices: List[int] = []
        if top_groups:
            # Majority group
            group_key_0, group_val_0 = top_groups[0]
            group_summaries.append({
                "result_key": group_key_0,
                "size": int(group_val_0.get("size", 0)),
                "candidate_index": selected_sql_index,
            })
            if selected_sql_index >= 0:
                candidate_indices.append(selected_sql_index)
        if len(top_groups) > 1:
            group_key_1, group_val_1 = top_groups[1]
            idx2 = -1
            sqls_list = group_val_1.get("sqls", []) or []
            if sqls_list:
                try:
                    idx2 = int(sqls_list[0].get("index", -1))
                except Exception:  # noqa: BLE001
                    idx2 = -1
            group_summaries.append({
                "result_key": group_key_1,
                "size": int(group_val_1.get("size", 0)),
                "candidate_index": idx2,
            })
            if idx2 >= 0:
                candidate_indices.append(idx2)

        # Ensure we have at least one candidate
        if not candidate_indices:
            results.append({
                "sample_id": sample_id,
                "gt_ok": False,
                "top_groups": group_summaries,
                "candidates": [],
                "recall_at_1": False,
                "recall_at_2": False,
                "notes": "No candidate indices available in top groups",
            })
            continue

        # Execute ground-truth
        gt_res = call_sql_api(dataset_name, db_id, gt_sql)
        gt_ok = bool(gt_res.get("ok", False))
        gt_rows = gt_res.get("rows") if gt_ok else None

        if not gt_ok:
            results.append({
                "sample_id": sample_id,
                "gt_ok": False,
                "top_groups": group_summaries,
                "candidates": [],
                "recall_at_1": False,
                "recall_at_2": False,
                "notes": f"GT execution failed: {gt_res.get('error')}",
            })
            continue

        # Build candidate SQLs from indices
        candidate_exec_info: List[Dict[str, Any]] = []
        any_match = False
        first_match = False

        for pos, idx in enumerate(candidate_indices):
            sql_text = predicted_sqls[idx] if 0 <= idx < len(predicted_sqls) else ""
            if not sql_text or not sql_text.strip():
                candidate_exec_info.append({
                    "index": idx,
                    "executed_ok": False,
                    "matched_gt": False,
                })
                continue
            pred_res = call_sql_api(dataset_name, db_id, sql_text)
            pred_ok = bool(pred_res.get("ok", False))
            matched = False
            if pred_ok and gt_ok:
                matched = results_equal(gt_rows, pred_res.get("rows"))
            candidate_exec_info.append({
                "index": idx,
                "executed_ok": pred_ok,
                "matched_gt": matched,
            })
            if pos == 0 and matched:
                first_match = True
            if matched:
                any_match = True

        evaluable_samples += 1
        if first_match:
            recall1_count += 1
        if any_match:
            recall2_count += 1

        results.append({
            "sample_id": sample_id,
            "gt_ok": True,
            "top_groups": group_summaries,
            "candidates": candidate_exec_info,
            "recall_at_1": first_match,
            "recall_at_2": any_match,
            "notes": None,
        })

    recall1 = (recall1_count / evaluable_samples) if evaluable_samples else 0.0
    recall2 = (recall2_count / evaluable_samples) if evaluable_samples else 0.0

    output = {
        "summary": {
            "total_samples": total_samples,
            "evaluable_samples": evaluable_samples,
            "recall_at_1": recall1,
            "recall_at_2": recall2,
        },
        "results": results,
    }

    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print("Recall computation complete")
    print(f"Total samples: {total_samples}")
    print(f"Evaluable samples (GT OK): {evaluable_samples}")
    print(f"Recall@1: {recall1:.4f}")
    print(f"Recall@2: {recall2:.4f}")


if __name__ == "__main__":
    main()


