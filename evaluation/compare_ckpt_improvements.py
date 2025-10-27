#!/usr/bin/env python3
"""
Compare two BIRD detailed_results.pkl files and log samples that were incorrect
at the first checkpoint but correct at the second checkpoint.

Inputs are PKLs produced by evaluate_bird_dev.py, which contain a list of dicts
with keys including: sample_id, question, db_id, any_execution_correct,
first_correct_index, predicted_sqls, etc.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Any, Dict, List, Optional

try:
    from pymongo import MongoClient  # type: ignore
except Exception:
    MongoClient = None  # type: ignore


def load_pkl_list(path: str) -> Optional[List[Dict[str, Any]]]:
    """Load a PKL expected to be a list[dict]. Return None on failure."""
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, list) and (not obj or isinstance(obj[0], dict)):
            return obj  # type: ignore[return-value]
        return None
    except Exception:
        return None


def build_index_by_sample_id(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for it in items:
        sid = str(it.get("sample_id"))
        if sid is None or sid == "None":
            continue
        index[sid] = it
    return index


def extract_first_correct_sql(entry: Dict[str, Any]) -> str:
    idx = int(entry.get("first_correct_index", -1))
    sqls = entry.get("predicted_sqls") or []
    if isinstance(sqls, list) and 0 <= idx < len(sqls):
        return str(sqls[idx])
    return ""

def extract_first_correct_response(entry: Dict[str, Any]) -> str:
    idx = int(entry.get("first_correct_index", -1))
    completions = entry.get("full_completions") or []
    if isinstance(completions, list) and 0 <= idx < len(completions):
        return str(completions[idx])
    return ""


def main():
    parser = argparse.ArgumentParser(description="Find samples incorrect at cp0 but correct at cpN.")
    parser.add_argument("--cp0-pkl", required=True, type=str, help="Path to cp0 detailed_results.pkl")
    parser.add_argument("--cpn-pkl", required=True, type=str, help="Path to cpN detailed_results.pkl")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file to write improvements (default: alongside cpN PKL)",
    )
    parser.add_argument("--mongo-host", type=str, default="192.168.1.108", help="MongoDB host")
    parser.add_argument("--mongo-port", type=int, default=27017, help="MongoDB port")
    parser.add_argument("--mongo-db", type=str, default="mats", help="MongoDB database name")
    parser.add_argument("--mongo-collection", type=str, default="dev_samples", help="MongoDB collection name")

    args = parser.parse_args()

    cp0_list = load_pkl_list(args.cp0_pkl)
    if cp0_list is None:
        raise SystemExit(f"Failed to load list-of-dicts from {args.cp0_pkl}")
    cpn_list = load_pkl_list(args.cpn_pkl)
    if cpn_list is None:
        raise SystemExit(f"Failed to load list-of-dicts from {args.cpn_pkl}")

    cp0_idx = build_index_by_sample_id(cp0_list)
    cpn_idx = build_index_by_sample_id(cpn_list)

    common_ids = sorted(set(cp0_idx.keys()) & set(cpn_idx.keys()), key=lambda x: int(x))

    improvements: List[Dict[str, Any]] = []

    # Optional Mongo connection for difficulty lookup
    client = None
    collection = None
    if MongoClient is not None:
        try:
            client = MongoClient(host=args.mongo_host, port=int(args.mongo_port), serverSelectionTimeoutMS=1500)
            db = client[args.mongo_db]
            collection = db[args.mongo_collection]
            _ = db.command("ping")
        except Exception:
            client = None
            collection = None

    for sid in common_ids:
        a = cp0_idx[sid]
        b = cpn_idx[sid]

        a_ok = bool(a.get("any_execution_correct", False))
        b_ok = bool(b.get("any_execution_correct", False))

        if (not a_ok) and b_ok:
            # fetch difficulty
            difficulty = "unknown"
            if collection is not None:
                try:
                    doc = None
                    try:
                        doc = collection.find_one({"_id": int(sid)})
                    except Exception:
                        doc = None
                    if doc is None:
                        doc = collection.find_one({"_id": sid})
                    if doc is not None:
                        dv = doc.get("difficulty")
                        if isinstance(dv, str) and dv.strip():
                            difficulty = dv
                except Exception:
                    pass
            improvements.append(
                {
                    "sample_id": sid,
                    "db_id": b.get("db_id"),
                    "question": b.get("question"),
                    "ground_truth_sql": b.get("ground_truth_sql"),
                    # Include raw responses (think + SQL) from both checkpoints
                    "cp0_full_completions": a.get("full_completions") or [],
                    "cpN_full_completions": b.get("full_completions") or [],
                    "difficulty": difficulty,
                    "cp0": {
                        "any_execution_correct": a_ok,
                        "num_candidates": int(a.get("num_candidates", len(a.get("predicted_sqls") or []))),
                        "first_correct_index": int(a.get("first_correct_index", -1)),
                    },
                    "cpN": {
                        "any_execution_correct": b_ok,
                        "num_candidates": int(b.get("num_candidates", len(b.get("predicted_sqls") or []))),
                        "first_correct_index": int(b.get("first_correct_index", -1)),
                        "first_correct_sql": extract_first_correct_sql(b),
                        "first_correct_response": extract_first_correct_response(b),
                    },
                }
            )

    out_path = args.output
    if not out_path:
        cpn_dir = os.path.dirname(os.path.abspath(args.cpn_pkl))
        out_path = os.path.join(cpn_dir, "improvements_from_cp0.json")

    # Compute response length statistics
    def _flatten_lengths(entries: List[Dict[str, Any]], key: str) -> List[int]:
        lengths: List[int] = []
        for ent in entries:
            arr = ent.get(key) or []
            if isinstance(arr, list):
                for s in arr:
                    if isinstance(s, str):
                        lengths.append(len(s))
        return lengths

    cp0_lengths = _flatten_lengths(improvements, "cp0_full_completions")
    cpn_lengths = _flatten_lengths(improvements, "cpN_full_completions")

    def _avg(vals: List[int]) -> float:
        return (sum(vals) / len(vals)) if vals else 0.0

    overall_stats = {
        "cp0_count": len(cp0_lengths),
        "cpN_count": len(cpn_lengths),
        "cp0_avg_length_chars": _avg(cp0_lengths),
        "cpN_avg_length_chars": _avg(cpn_lengths),
    }

    # Per-difficulty averages
    per_diff: Dict[str, Dict[str, Any]] = {}
    for ent in improvements:
        diff = str(ent.get("difficulty", "unknown")).lower()
        if diff not in per_diff:
            per_diff[diff] = {
                "_cp0": [],
                "_cpN": [],
            }
        for s in (ent.get("cp0_full_completions") or []):
            if isinstance(s, str):
                per_diff[diff]["_cp0"].append(len(s))
        for s in (ent.get("cpN_full_completions") or []):
            if isinstance(s, str):
                per_diff[diff]["_cpN"].append(len(s))
    for k, v in per_diff.items():
        cp0c = len(v.get("_cp0", []))
        cpnc = len(v.get("_cpN", []))
        per_diff[k] = {
            "cp0_count": cp0c,
            "cpN_count": cpnc,
            "cp0_avg_length_chars": _avg(v.get("_cp0", [])),
            "cpN_avg_length_chars": _avg(v.get("_cpN", [])),
        }

    stats = {
        "overall": overall_stats,
        "per_difficulty": per_diff,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"improvements": improvements, "count": len(improvements), "stats": stats}, f, indent=2, ensure_ascii=False)

    print(f"Improvements written: {len(improvements)} -> {out_path}")


if __name__ == "__main__":
    main()


