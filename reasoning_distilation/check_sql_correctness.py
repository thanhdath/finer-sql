#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import defaultdict

import requests
import pandas as pd  # type: ignore
from pymongo import MongoClient  # type: ignore
from tqdm import tqdm  # type: ignore
from multiprocessing import Pool


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Evaluate SQL correctness for model outputs in MongoDB")
    ap.add_argument("--mongo_uri", default="mongodb://localhost:27017")
    ap.add_argument("--collection", default="llm_pool_bird", help="Mongo collection under 'mats' DB")
    ap.add_argument("--model_name", default='gpt-5--low-answer', help="Exact model_name stored in Mongo (with any suffix)")
    ap.add_argument("--api_url", default="http://192.168.1.108:8001/execute", help="SQL executor API endpoint")
    ap.add_argument("--timeout_ms", type=int, default=120000)
    ap.add_argument("--max_rows", type=int, default=10000)
    ap.add_argument("--processes", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="0 = all documents")
    ap.add_argument("--out_json", help="Optional path to write detailed JSON results (all samples)")
    ap.add_argument("--fail_json", help="Optional path to write failed-only JSON logs")
    ap.add_argument("--out_csv", help="Optional path to write CSV summary")
    # Optional: enrich fail JSON with selected columns from mats.grast
    ap.add_argument("--grast_uri", help="Mongo URI for mats.grast (defaults to --mongo_uri)")
    ap.add_argument("--grast_collection", help="Override mats.grast collection name; derived from dataset/split if omitted")
    return ap.parse_args()


def is_select_like(sql: str) -> bool:
    s = (sql or "").lstrip().lower()
    return s.startswith("select") or s.startswith("with") or s.startswith("explain")


def call_sql_api(api_url: str, dataset_name: str, db_id: str, sql: str, *, timeout_ms: int, max_rows: int) -> Dict[str, Any]:
    req_mode = "read_only" if is_select_like(sql) else "sandbox_rollback"
    try:
        resp = requests.post(
            api_url,
            json={
                "dataset_name": dataset_name,
                "db_id": db_id,
                "sql": sql,
                "mode": req_mode,
                "timeout_ms": int(timeout_ms),
                "max_rows": int(max_rows),
            },
            timeout=(int(timeout_ms) / 1000.0) + 2,
        )
        if resp.status_code != 200:
            return {
                "ok": False,
                "rows": None,
                "error": f"HTTP {resp.status_code}: {resp.text}",
            }
        data = resp.json()
        if data["ok"] == False:
            print("="*100)
            print(json.dumps({
                "dataset_name": dataset_name,
                "db_id": db_id,
                "sql": sql,
                "mode": req_mode,
                "timeout_ms": int(timeout_ms),
                "max_rows": int(max_rows),
            }, indent=2))
            print(data)
        return {
            "ok": bool(data.get("ok", False)),
            "rows": data.get("rows"),
            "error": data.get("error"),
        }
    except Exception as e:
        return {"ok": False, "rows": None, "error": str(e)}


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


def extract_sql_from_response(text: str) -> Optional[str]:
    if not text:
        return None
    txt = text.strip()
    # After </think> tag is SQL (no other fallback)
    end_tag = "</think>"
    idx = txt.rfind(end_tag)
    if idx == -1:
        return None
    tail = txt[idx + len(end_tag):].strip()
    return tail or None


def _get_ground_truth_sql(doc: Dict[str, Any]) -> Optional[str]:
    return doc.get("ground_truth_sql")


def _try_import_is_execution_correct():
    try:
        # Import symbol from module that already binds it in its namespace
        import mats.check_correct as cc  # type: ignore
        fn = getattr(cc, "is_execution_correct", None)
        if callable(fn):
            return fn
    except Exception:
        pass

    # Fallback: strict DataFrame equality
    def _strict_equal(df_true: Optional[pd.DataFrame], df_pred: Optional[pd.DataFrame]) -> bool:
        if df_true is None or df_pred is None:
            return False
        a = set(map(tuple, df_true.values.tolist()))
        b = set(map(tuple, df_pred.values.tolist()))
        return a == b

    return _strict_equal


def _evaluate_one(args: Tuple[Dict[str, Any], str, int, int]) -> Tuple[Dict[str, Any], bool]:
    d, api_url, timeout_ms, max_rows = args
    is_execution_correct = _try_import_is_execution_correct()

    sample_id = d.get("sample_id")
    dataset_name = d.get("dataset_name") or d.get("dataset")
    db_id = d.get("db_id")
    response_text: str = (d.get("response") or "").strip()
    pred_sql = extract_sql_from_response(response_text)
    gt_sql = _get_ground_truth_sql(d)

    rec: Dict[str, Any] = {
        "sample_id": sample_id,
        "dataset_name": dataset_name,
        "db_id": db_id,
        "pred_sql": pred_sql,
        "ground_truth_sql": gt_sql,
        "ok": False,
        "error": None,
        "messages": d.get("messages"),
        "prompt": d.get("prompt"),
        "pred_exec": None,
        "gt_exec": None,
    }

    if not dataset_name or not db_id:
        rec["error"] = "missing dataset_name/db_id"
        return rec, False
    if not pred_sql:
        rec["error"] = "missing predicted SQL"
        return rec, False
    if not gt_sql:
        rec["error"] = "missing ground-truth SQL"
        return rec, False

    pred_res = call_sql_api(api_url, dataset_name, db_id, pred_sql, timeout_ms=timeout_ms, max_rows=max_rows)
    # rec["pred_exec"] = pred_res
    if not pred_res.get("ok"):
        rec["error"] = f"pred exec failed: {pred_res.get('error')}"
        return rec, False

    gt_res = call_sql_api(api_url, dataset_name, db_id, gt_sql, timeout_ms=timeout_ms, max_rows=max_rows)
    # rec["gt_exec"] = gt_res
    if not gt_res.get("ok"):
        rec["error"] = f"gt exec failed: {gt_res.get('error')}"
        return rec, False

    pred_df = _rows_to_dataframe(pred_res.get("rows"))
    gt_df = _rows_to_dataframe(gt_res.get("rows"))
    try:
        correct = bool(is_execution_correct(gt_df, pred_df))
    except Exception as e:
        rec["error"] = f"correctness error: {str(e)}"
        return rec, False

    rec["ok"] = correct
    return rec, correct


def evaluate_documents(
    docs: List[Dict[str, Any]],
    *,
    api_url: str,
    timeout_ms: int,
    max_rows: int,
    processes: int = 4,
    on_record: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    detailed: List[Dict[str, Any]] = []
    total = 0
    success = 0

    if processes and processes > 1:
        args_iter = ((d, api_url, timeout_ms, max_rows) for d in docs)
        with Pool(processes=int(processes)) as pool:
            for rec, ok in tqdm(pool.imap_unordered(_evaluate_one, args_iter), total=len(docs)):
                total += 1
                if ok:
                    success += 1
                if on_record is not None:
                    try:
                        on_record(rec)
                    except Exception as err:
                        print(err)
                        pass
                detailed.append(rec)
    else:
        for d in tqdm(docs, total=len(docs)):
            total += 1
            rec, ok = _evaluate_one((d, api_url, timeout_ms, max_rows))
            if ok:
                success += 1
            if on_record is not None:
                try:
                    on_record(rec)
                except Exception as err:
                    print(err)
                    pass
            detailed.append(rec)

    summary = {
        "total": total,
        "correct": success,
        "accuracy": (float(success) / float(total)) if total > 0 else 0.0,
    }
    return detailed, summary


def main():
    args = parse_args()

    client = MongoClient(args.mongo_uri)
    coll = client["mats"][args.collection]

    cursor = coll.find({"model_name": args.model_name})
    if args.limit and int(args.limit) > 0:
        cursor = cursor.limit(int(args.limit))
    docs = list(cursor)

    # Precompute map from sample_id -> document _id for immediate updates
    sample_to_id: Dict[str, Any] = {}
    for d in docs:
        sid = d.get("sample_id")
        if sid is not None:
            sample_to_id[str(sid)] = d.get("_id")

    def _on_record(rec: Dict[str, Any]) -> None:
        sid = rec.get("sample_id")
        if sid is None:
            return
        key = str(sid)
        if key not in sample_to_id:
            return
        try:
            coll.update_one({"_id": sample_to_id[key]}, {"$set": {"is_execution_correct": rec["ok"]}})
        except Exception:
            # Ignore individual update failures; continue with others
            pass

    results, summary = evaluate_documents(
        docs,
        api_url=args.api_url,
        timeout_ms=args.timeout_ms,
        max_rows=args.max_rows,
        processes=args.processes,
        on_record=_on_record,
    )

    print(json.dumps(summary, indent=2))

    # Write back execution correctness to MongoDB for equivalent samples
    # Already updated per-record during evaluation; no batch write-back needed

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
    if args.fail_json:
        fails = [r for r in results if not r.get("ok")]

        # Enrich failures with selected columns grouped by table from mats.grast
        try:
            grast_uri = args.grast_uri or args.mongo_uri
            gclient = MongoClient(grast_uri)

            # Group sample_ids by dataset_name (ignore split; always use *_train collections)
            pair_to_ids: Dict[str, List[Any]] = defaultdict(list)
            for r in fails:
                dataset_name = r.get("dataset_name") or r.get("dataset") or ""
                sid = r.get("sample_id")
                if dataset_name and sid is not None:
                    pair_to_ids[str(dataset_name)].append(sid)

            def _derive_grast_collection(dataset_name: str) -> str:
                d = (dataset_name or "").lower()
                # Always use *_train collections, never the dev ones
                if d == "spider":
                    return "grast_qwen_0.6b_spider_train"
                return "grast_qwen_0.6b_bird_train"

            sample_to_grouped: Dict[Any, Dict[str, List[str]]] = {}
            sample_to_pred_cols: Dict[Any, List[str]] = {}
            for dataset_name, id_list in pair_to_ids.items():
                if not id_list:
                    continue
                grast_coll_name = args.grast_collection or _derive_grast_collection(dataset_name)
                gcoll = gclient["mats"][grast_coll_name]
                cursor = gcoll.find({"sample_id": {"$in": id_list}}, {"sample_id": 1, "pred_cols": 1})
                for gd in cursor:
                    sid = gd.get("sample_id")
                    sid_key = str(sid)
                    pred_cols = gd.get("pred_cols") or []
                    grouped: Dict[str, List[str]] = defaultdict(list)
                    if isinstance(pred_cols, list):
                        for fq in pred_cols:
                            if isinstance(fq, str) and "." in fq:
                                table, col = fq.split(".", 1)
                                grouped[str(table)].append(str(col))
                    sample_to_grouped[sid_key] = {t: sorted(cols) for t, cols in grouped.items()}
                    if isinstance(pred_cols, list):
                        sample_to_pred_cols[sid_key] = [str(x) for x in pred_cols]

            for r in fails:
                sid = r.get("sample_id")
                lookup_key = str(sid)
                if lookup_key in sample_to_grouped:
                    r["selected_columns"] = sample_to_grouped[lookup_key]
                # if lookup_key in sample_to_pred_cols:
                #     # Log raw predicted columns directly into fail JSON
                #     r["pred_cols"] = sample_to_pred_cols[lookup_key]
        except Exception as e:
            print(e)
            if fails:
                fails[0]["log_error"] = f"enrichment_error: {type(e).__name__}: {str(e)}"

        with open(args.fail_json, "w") as f:
            json.dump({"summary": summary, "failed": fails}, f, indent=2)
    if args.out_csv:
        df = pd.DataFrame(results)
        df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()


