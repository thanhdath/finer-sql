#!/usr/bin/env python3
from __future__ import annotations
import os
import argparse
import pickle
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple, Union

import requests
from datasets import Dataset, load_from_disk
# from pymongo import MongoClient  # Disabled since MongoDB saving is commented out


def process_single_sample(args_tuple):
    """Process a single (dataset_name, db_id, sql) entry and return the result."""
    (dataset_name, db_id, sql, api_url, timeout_ms, max_rows) = args_tuple

    if not dataset_name or not db_id or not sql:
        return {
            "key": None,
            "result": None,
            "error": "Missing required fields",
            "processed": False
        }

    key = (dataset_name, db_id, sql)

    try:
        res = call_sql_api(api_url, dataset_name, db_id, sql, timeout_ms=timeout_ms, max_rows=max_rows)
        return {
            "key": key,
            "result": res,
            "error": None,
            "processed": True
        }
    except Exception as e:
        return {
            "key": key,
            "result": None,
            "error": str(e),
            "processed": True
        }


def call_sql_api(api_url: str, dataset_name: str, db_id: str, sql: str, *, timeout_ms: int, max_rows: int) -> Dict[str, Any]:
    try:
        data = {
                "dataset_name": dataset_name,
                "db_id": db_id,
                "sql": sql,
                "mode": "read_only",
                "timeout_ms": timeout_ms,
                "max_rows": max_rows,
            }
        print(data["sql"])
        resp = requests.post(
            api_url,
            json=data,
            timeout=(timeout_ms / 1000.0) + 2,
        )
        if resp.status_code != 200:
            print(f"HTTP {resp.status_code}: {resp.text}")
            return {"ok": False, "rows": None, "error": f"HTTP {resp.status_code}: {resp.text}"}
        data = resp.json()
        # print few rows
        print(data.get("rows")[:2])
        return {"ok": bool(data.get("ok", False)), "rows": data.get("rows"), "error": data.get("error")}
    except Exception as e:
        print(str(e))
        return {"ok": False, "rows": None, "error": str(e)}


def iter_train_datasets(data_root: str) -> List[Dataset]:
    ds_list: List[Dataset] = []
    # Try loading data_root directly as a single dataset first
    try:
        dd = load_from_disk(data_root)
        if hasattr(dd, "keys") and callable(getattr(dd, "keys", None)):
            if "train" in dd:
                ds_list.append(dd["train"])
        else:
            ds_list.append(dd)
        if ds_list:
            return ds_list
    except Exception:
        pass
    # Fall back to scanning subdirectories
    for name in sorted(os.listdir(data_root)):
        path = os.path.join(data_root, name)
        if not os.path.isdir(path):
            continue
        if "train" not in name:
            try:
                dd = load_from_disk(path)
                if hasattr(dd, "keys") and "train" in dd:
                    ds_list.append(dd["train"])
            except Exception:
                pass
            continue
        try:
            dd = load_from_disk(path)
            if hasattr(dd, "keys") and "train" in dd:
                ds_list.append(dd["train"])
            else:
                ds_list.append(dd)
        except Exception:
            continue
    return ds_list


def main() -> None:
    ap = argparse.ArgumentParser("Build cached ground-truth execution rows for training")
    ap.add_argument("--data_root", default="/home/datht/graph-schema/end2end/data")
    ap.add_argument("--api_url", default="http://192.168.1.108:8001/execute")
    ap.add_argument("--timeout_ms", type=int, default=120000)
    ap.add_argument("--max_rows", type=int, default=10000)
    ap.add_argument("--out", required=True, help="Output pickle path for GT cache")
    ap.add_argument("--mongo_url", default="mongodb://localhost:27017/", help="MongoDB connection URL")
    ap.add_argument("--collection", default="gt_cache", help="MongoDB collection name")
    ap.add_argument("--limit_samples", type=int, default=0, help="0 = all")
    ap.add_argument("--num_processes", type=int, default=16, help="Number of parallel processes")
    ap.add_argument("--load_existing", action="store_true", default=True, help="Load existing cache file and only process missing entries")
    args = ap.parse_args()

    # MongoDB connection disabled since saving is commented out
    # client = MongoClient(args.mongo_url)
    # db = client.mats
    # collection = db[args.collection]
    
    datasets = iter_train_datasets(args.data_root)
    if not datasets:
        raise RuntimeError(f"No train datasets found under {args.data_root}")

    # Load existing cache if it exists and load_existing is enabled
    existing_cache = {}
    if args.load_existing and os.path.exists(args.out):
        print(f"Loading existing cache from {args.out}...")
        try:
            with open(args.out, "rb") as f:
                existing_cache = pickle.load(f)
            print(f"Loaded {len(existing_cache)} existing cache entries")
        except Exception as e:
            print(f"Warning: Could not load existing cache: {e}")
            existing_cache = {}

    # Collect all samples to process
    all_samples = []
    for ds in datasets:
        rows = ds.to_list()
        if args.limit_samples > 0:
            rows = rows[:args.limit_samples]
        all_samples.extend(rows)
    
    # Flatten samples into individual (dataset_name, db_id, sql) entries, one per SQL
    sql_entries_to_process = []
    skipped_existing = 0

    for ex in all_samples:
        dataset_name = ex.get("dataset_name")
        db_id = ex.get("db_id")
        groundtruth_sqls = ex.get("groundtruth_sqls") or []
        if not dataset_name or not db_id:
            continue
        for sql in groundtruth_sqls:
            if not sql:
                continue
            key = (dataset_name, db_id, sql)
            if key in existing_cache:
                cache_entry = existing_cache[key]
                if isinstance(cache_entry, list) or (
                    isinstance(cache_entry, dict) and cache_entry.get("timeout")
                ):
                    skipped_existing += 1
                    continue
            sql_entries_to_process.append((dataset_name, db_id, sql))
    
    print(f"Total samples: {len(all_samples)}")
    print(f"SQL entries already in cache: {skipped_existing}")
    print(f"SQL entries to process: {len(sql_entries_to_process)}")
    print(f"Processing {len(sql_entries_to_process)} SQL entries with {args.num_processes} processes...")
    
    # Prepare arguments for multiprocessing (one entry per SQL)
    process_args = [
        (dataset_name, db_id, sql, args.api_url, args.timeout_ms, args.max_rows)
        for dataset_name, db_id, sql in sql_entries_to_process
    ]
    
    # Initialize cache with existing entries
    cache: Dict[Tuple[str, str, str], Union[List[Dict[str, Any]], Dict[str, Any]]] = existing_cache.copy()
    total = 0
    processed = 0
    skipped = 0
    timeouts = 0
    timeout_entries = []  # Store timeout details for logging
    SAVE_EVERY = 5000  # Checkpoint every N processed entries

    def _save_cache():
        tmp_path = args.out + ".tmp"
        with open(tmp_path, "wb") as _f:
            pickle.dump(cache, _f)
        os.replace(tmp_path, args.out)
        print(f"[checkpoint] Saved {len(cache)} entries → {args.out}")

    # If no entries to process, just save the existing cache
    if not sql_entries_to_process:
        print("No new samples to process. Saving existing cache...")
    else:
        with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
            # Submit all tasks
            future_to_args = {executor.submit(process_single_sample, arg): arg for arg in process_args}
            
            # Process results as they complete
            for future in as_completed(future_to_args):
                result = future.result()
                
                if not result["processed"]:
                    skipped += 1
                    processed += 1
                    print(f"{processed} (skipped - missing fields)")
                    continue
                
                key = result["key"]
                if key is None:
                    skipped += 1
                    processed += 1
                    print(f"{processed} (skipped - no key)")
                    continue
                
                # Check if already in cache (avoid duplicates)
                if key in cache:
                    processed += 1
                    print(f"{processed} (skipped - duplicate)")
                    continue
                
                res = result["result"]
                if res is None:
                    skipped += 1
                    processed += 1
                    print(f"{processed} (skipped - no result)")
                    continue
                    
                if res.get("ok") and isinstance(res.get("rows"), list):
                    cache[key] = res["rows"]
                else:
                    # Check if it's a timeout error (check timed_out flag first, then error message)
                    error_msg = res.get("error", "")
                    is_timeout = res.get("timed_out", False) or "timed out" in error_msg.lower() or "timeout" in error_msg.lower() or "interrupt" in error_msg.lower()
                    if is_timeout:
                        # Cache timeout result separately
                        cache[key] = {"timeout": True, "error": error_msg}
                        timeouts += 1
                        timeout_entries.append({
                            "dataset_name": key[0],
                            "db_id": key[1],
                            "sql": key[2],
                            "error": error_msg,
                            "processed": processed
                        })
                        print(f"Timeout for query {processed} (db_id: {key[1]}): {error_msg}")
                    else:
                        # Cache negative result to avoid re-querying repeatedly
                        cache[key] = []
                        print(f"Failed query {processed}: {error_msg}")
                
                total += 1
                processed += 1
                print(f"{processed}")
                # Periodic checkpoint save
                if total % SAVE_EVERY == 0:
                    _save_cache()

    # Write final cache to pickle file
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(cache, f)
    
    print(f"Saved GT cache ({len(cache)} entries) → {args.out}")
    print(f"Total cache entries: {len(cache)}")
    print(f"Existing entries: {len(existing_cache)}")
    print(f"New entries processed: {total}")
    print(f"Skipped (missing fields): {skipped}")
    print(f"Timeouts in this run: {timeouts}")
    
    # Log timeout details to file
    if timeout_entries:
        timeout_log_file = f"{args.out}_timeouts.log"
        with open(timeout_log_file, "w") as f:
            f.write(f"Timeout Log - {len(timeout_entries)} queries timed out\n")
            f.write("=" * 80 + "\n\n")
            for i, entry in enumerate(timeout_entries, 1):
                f.write(f"Timeout #{i}:\n")
                f.write(f"  Processed: {entry['processed']}\n")
                f.write(f"  Dataset: {entry['dataset_name']}\n")
                f.write(f"  DB ID: {entry['db_id']}\n")
                f.write(f"  SQL: {entry['sql']}\n")
                f.write(f"  Error: {entry['error']}\n")
                f.write("-" * 80 + "\n\n")
        print(f"Timeout details logged to: {timeout_log_file}")
    
    # client.close()


if __name__ == "__main__":
    main()

