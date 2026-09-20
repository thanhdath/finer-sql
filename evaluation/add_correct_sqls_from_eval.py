#!/usr/bin/env python3
"""
add_correct_sqls_from_eval.py
─────────────────────────────
Dry-run tool to collect all correct SQLs per sample from an evaluation
results pickle (detailed_results.pkl) and show planned MongoDB updates
without performing any writes.

Usage (dry-run only):
    python mats/sql_writer/evaluation/add_correct_sqls_from_eval.py \
        --pkl-path /home/datht/mats/sql_writer/evaluation/evaluation_spider_train/GRPO-3B-cp2000/detailed_results.pkl \
        --mongo-uri mongodb://192.168.1.108:27017 \
        --db-name mats \
        --collection spider_train_samples \
        --out-json planned_updates_spider_train.json

Dry-run by default; pass --execute to perform MongoDB updates.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
from pymongo import MongoClient


def normalize_for_compare(sql: str) -> str:
    """Canonical form for equality checks (no trailing semicolon/newline, single spaces)."""
    s = str(sql)
    s = s.strip()
    s = re.sub(r";+\s*$", "", s)  # drop trailing semicolons
    s = re.sub(r"\s+", " ", s)    # collapse all whitespace to single spaces
    return s.strip()


def canonicalize_for_storage(sql: str) -> str:
    """Normalized form to store/print: single spaces, no trailing newline."""
    s = normalize_for_compare(sql)
    return s


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


def collect_correct_sqls_from_entry(entry: Dict[str, Any]) -> List[str]:
    """Given one result_with_exec entry, return all candidate SQLs that match GT results."""
    gt_res = entry.get("gt_execution_result")
    if not gt_res or not gt_res.get("ok", False):
        return []

    gt_df = _rows_to_dataframe(gt_res.get("rows"))
    if gt_df is None:
        return []

    candidates: List[str] = entry.get("predicted_sqls", []) or []
    cand_execs: List[Optional[Dict[str, Any]]] = entry.get("candidate_execution_results", []) or []

    correct_sqls: List[str] = []
    for idx, cand_sql in enumerate(candidates):
        cand_sql = cand_sql or ""
        exec_res = cand_execs[idx] if idx < len(cand_execs) else None
        if not exec_res or not exec_res.get("ok", False):
            continue
        pred_df = _rows_to_dataframe(exec_res.get("rows"))
        if dataframes_equal(gt_df, pred_df):
            correct_sqls.append(cand_sql)

    return correct_sqls


def unique_sqls(sqls: List[str]) -> List[str]:
    """Return de-duplicated SQLs using compare-normalization; store normalized strings."""
    seen: Set[str] = set()
    unique: List[str] = []
    for s in sqls:
        cmp_norm = normalize_for_compare(s)
        if cmp_norm in seen:
            continue
        seen.add(cmp_norm)
        unique.append(canonicalize_for_storage(s))
    return unique


FIELD_NAME = "augmented_sqls_for_grpo"

# The gold SQL field differs by dataset in mats.<dataset>_<split>_samples.
GOLD_FIELDS = ("SQL", "query", "sql", "ground_truth_sql")


def doc_gold(doc: Dict[str, Any]) -> Optional[str]:
    """The benchmark gold stored on a sample document, whatever it is called."""
    for k in GOLD_FIELDS:
        v = doc.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def find_sample_doc(coll, db_id: Any, sample_id: Any, dataset_name: Any = None):
    """Fetch the sample document these correct SQLs belong to.

    THE BUG THIS REPLACES.  The previous version addressed the document by
    ``{"db_id": db_id, "_id": sample_id}`` only -- it assumed the collection's primary
    key ``_id`` is the same number as the evaluation's ``sample_id``.  Where the two id
    spaces disagree (they run parallel inside a database but are offset by however many
    of that database's samples were dropped from one of them), ``_id == sample_id``
    selects a DIFFERENT sample of the SAME database, a few positions along, and the
    correct SQLs of sample N are written onto sample N-k.  That is how
    bird/airline sample 5849 ("flights that landed at Lake Charles Regional on
    2018/8/15") ended up carrying eleven references that count on-time departures on
    2018/8/1, which are sample 5854's: a +5 drift that holds for that whole database.

    The document is now looked up by its own ``sample_id`` field first and only falls
    back to ``_id``; ``dataset_name`` is included when the collection carries it, because
    BIRD and Spider reuse the same sample_id numbers.  The caller verifies the gold
    before writing, so a wrong document can no longer be written to even if some third
    id space is introduced later.
    """
    base: Dict[str, Any] = {"db_id": db_id}
    if dataset_name is not None:
        for q in ({**base, "dataset_name": dataset_name, "sample_id": sample_id},
                  {**base, "dataset": dataset_name, "sample_id": sample_id}):
            doc = coll.find_one(q)
            if doc:
                return doc
    doc = coll.find_one({**base, "sample_id": sample_id})
    if doc:
        return doc
    return coll.find_one({**base, "_id": sample_id})


def build_planned_updates(
    detailed_results: List[Dict[str, Any]],
    mongo_client: MongoClient,
    db_name: str,
    collection_name: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    coll = mongo_client[db_name][collection_name]

    planned: List[Dict[str, Any]] = []
    stats = {
        "total_entries": 0,
        "entries_with_correct": 0,
        "entries_with_updates": 0,
        "total_correct_sqls": 0,
        "total_new_sqls": 0,
        "entries_no_document": 0,
        "entries_gold_mismatch": 0,
    }
    mismatches: List[Dict[str, Any]] = []

    for entry in detailed_results:
        stats["total_entries"] += 1
        sample_id = entry.get("sample_id")
        db_id = entry.get("db_id")
        gt_sql = entry.get("ground_truth_sql", "") or ""

        correct_sqls = collect_correct_sqls_from_entry(entry)
        correct_sqls = unique_sqls(correct_sqls)
        # Drop GT if included; GT is tracked separately in DB
        gt_norm = normalize_for_compare(gt_sql)
        correct_sqls = [s for s in correct_sqls if normalize_for_compare(s) != gt_norm]

        if not correct_sqls:
            continue
        stats["entries_with_correct"] += 1
        stats["total_correct_sqls"] += len(correct_sqls)

        # Fetch the document these SQLs belong to, by its own sample_id (see
        # find_sample_doc: addressing it by _id wrote them onto the wrong sample).
        existing = find_sample_doc(coll, db_id, sample_id, entry.get("dataset_name"))
        if existing is None:
            stats["entries_no_document"] += 1
            continue
        # THE GUARD: the document's own gold must be the gold these SQLs were judged
        # against.  If it is not, this is not that sample's document and nothing is
        # written -- a wrong-sample write is impossible rather than merely unlikely.
        stored_gold = doc_gold(existing)
        if stored_gold is None or normalize_for_compare(stored_gold) != normalize_for_compare(gt_sql):
            stats["entries_gold_mismatch"] += 1
            mismatches.append({
                "sample_id": sample_id, "db_id": db_id,
                "eval_gold": canonicalize_for_storage(gt_sql),
                "document_gold": canonicalize_for_storage(stored_gold or ""),
                "document_id": str(existing.get("_id")),
            })
            continue
        existing_list: List[str] = []
        if isinstance(existing.get(FIELD_NAME), list):
            existing_list = [str(x) for x in existing.get(FIELD_NAME) if isinstance(x, str)]

        # Normalize GT for storage and comparison
        gt_stored = canonicalize_for_storage(gt_sql)
        gt_norm = normalize_for_compare(gt_sql)

        # Remove any GT duplicates from existing (by compare-normalization), deduplicate existing while preserving order
        seen_existing: Set[str] = set()
        existing_no_gt: List[str] = []
        for s in existing_list:
            cmp_norm = normalize_for_compare(s)
            if cmp_norm == gt_norm:
                continue
            if cmp_norm in seen_existing:
                continue
            seen_existing.add(cmp_norm)
            existing_no_gt.append(s)

        existing_norms = seen_existing
        to_add_cmp = [s for s in correct_sqls if normalize_for_compare(s) not in existing_norms and normalize_for_compare(s) != gt_norm]
        # Store additions in storage-canonical form (no trailing newline)
        to_add = [canonicalize_for_storage(s) for s in to_add_cmp]

        # Build final list: GT at position 0, followed by existing (sans GT), then new additions
        final_list: List[str] = [gt_stored] + existing_no_gt + to_add

        planned_update = {
            "sample_id": sample_id,
            "db_id": db_id,
            "document_id": existing.get("_id"),
            "dataset_name": entry.get("dataset_name"),
            "existing_count": len(existing_list),
            "found_correct_count": len(correct_sqls),
            "additions_count": len(to_add),
            "final_count": len(final_list),
            FIELD_NAME: final_list,
            "action": "set",
        }
        if to_add:
            stats["entries_with_updates"] += 1
            stats["total_new_sqls"] += len(to_add)
        planned.append(planned_update)

    if mismatches:
        print("\nREFUSED %d update(s): the document found for the sample does not carry "
              "the gold these SQLs were judged against. Writing them would attach one "
              "sample's references to another. First few:" % len(mismatches))
        for m in mismatches[:5]:
            print("  sample_id=%s db_id=%s document _id=%s\n     eval gold: %s\n     doc  gold: %s"
                  % (m["sample_id"], m["db_id"], m["document_id"],
                     m["eval_gold"][:160], m["document_gold"][:160]))
    stats["mismatch_examples"] = mismatches[:50]
    return planned, stats


def _noop(_: Any) -> None:
    return None


def execute_updates(
    planned: List[Dict[str, Any]],
    mongo_client: MongoClient,
    db_name: str,
    collection_name: str,
) -> None:
    coll = mongo_client[db_name][collection_name]
    updated = 0
    for upd in planned:
        final_array = upd.get(FIELD_NAME, [])
        if not isinstance(final_array, list) or len(final_array) == 0:
            continue
        # Write to the document build_planned_updates VERIFIED, addressed by its own
        # primary key -- never by re-resolving sample_id, which is what mis-addressed
        # these references in the first place.
        doc_id = upd.get("document_id")
        if doc_id is None:
            continue
        coll.update_one(
            {"_id": doc_id},
            {"$set": {FIELD_NAME: final_array}},
            upsert=False,
        )
        updated += 1
    print(f"Applied updates to {updated} documents (set full array with GT at index 0).")


def main():
    parser = argparse.ArgumentParser(description="Dry-run: collect correct SQLs and show planned Mongo updates")
    parser.add_argument(
        "--pkl-path",
        type=str,
        default="/home/datht/mats/sql_writer/evaluation/evaluation_spider_train/GRPO-3B-cp2000/detailed_results.pkl",
        help="Path to detailed_results.pkl produced by evaluation",
    )
    parser.add_argument("--mongo-uri", type=str, default="mongodb://192.168.1.108:27017", help="MongoDB URI")
    parser.add_argument("--db-name", type=str, default="mats", help="MongoDB database name")
    parser.add_argument(
        "--collection",
        type=str,
        default="spider_train_samples",
        help="MongoDB collection that stores per-sample docs",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional path to save planned updates as JSON",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="If set, perform MongoDB updates instead of dry-run only",
    )

    args = parser.parse_args()

    if not os.path.exists(args.pkl_path):
        raise FileNotFoundError(f"Pickle file not found: {args.pkl_path}")

    with open(args.pkl_path, "rb") as f:
        detailed_results = pickle.load(f)
    if not isinstance(detailed_results, list):
        raise ValueError("Expected a list of result entries in the pickle file")

    mongo_client = MongoClient(args.mongo_uri)

    planned, stats = build_planned_updates(
        detailed_results=detailed_results,
        mongo_client=mongo_client,
        db_name=args.db_name,
        collection_name=args.collection,
    )

    # Print a readable report
    print("\nPlanned updates (dry-run by default):\n")
    for upd in planned:
        if upd.get("final_count", 0) <= 0:
            continue
        print(f"sample_id={upd['sample_id']} db_id={upd['db_id']} | existing={upd['existing_count']} found_correct={upd['found_correct_count']} new_added={upd['additions_count']} final_count={upd['final_count']}")
        for i, s in enumerate(upd.get(FIELD_NAME, [])):
            prefix = "[GT]" if i == 0 else "   "
            print(f"  {prefix} {s}")

    print("\nSummary:")
    print(json.dumps(stats, indent=2))

    if args.out_json:
        try:
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump({"planned": planned, "stats": stats}, f, indent=2, ensure_ascii=False)
            print(f"\nPlanned updates saved to: {args.out_json}")
        except Exception as e:
            print(f"Failed to save planned updates JSON: {e}")

    if args.execute:
        execute_updates(
            planned=planned,
            mongo_client=mongo_client,
            db_name=args.db_name,
            collection_name=args.collection,
        )
    else:
        print("\nDry-run only. No MongoDB updates were performed.")


if __name__ == "__main__":
    main()


