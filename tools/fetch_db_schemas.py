#!/usr/bin/env python3
"""Database schemas through the db_execution API (never sqlite directly): {"<dataset>:<db_id>": {table: [columns]}}.

Columns come from the CREATE statements in sqlite_master (parsed with sqlglot); a table whose DDL does not parse falls
back to the keys of one row.  Used by tools/build_atomic_oplists.py (and at runtime by --atomic_canon v2 through the
ops file) to qualify unqualified columns.

    python tools/fetch_db_schemas.py --parquet train.parquet --api_url http://127.0.0.1:8031/execute --out schemas.json
"""
from __future__ import annotations
import argparse, json, sys
import pandas as pd
import requests
import sqlglot
from sqlglot import exp


def fetch(api_url: str, pairs, timeout_ms: int = 30000):
    s = requests.Session()
    out = {}
    for ds, db in pairs:
        def q(sql, max_rows=100000):
            j = s.post(api_url, json={"dataset_name": ds, "db_id": db, "sql": sql, "mode": "read_only", "timeout_ms": timeout_ms, "max_rows": max_rows}, timeout=timeout_ms / 1000 + 10).json()
            if not j.get("ok"):
                raise RuntimeError(f"{ds}/{db}: {j.get('error')}")
            return j.get("rows_list") or [list(r.values()) for r in (j.get("rows") or [])]
        sch = {}
        for name, ddl in q("SELECT name, sql FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'"):
            cols = []
            try:
                cols = [c.name for c in sqlglot.parse_one(ddl, read="sqlite").find_all(exp.ColumnDef)]
            except Exception:  # noqa: BLE001
                cols = []
            if not cols:
                j = s.post(api_url, json={"dataset_name": ds, "db_id": db, "sql": f'SELECT * FROM "{name}" LIMIT 1', "mode": "read_only", "timeout_ms": timeout_ms, "max_rows": 1}, timeout=60).json()
                cols = list((j.get("rows") or [{}])[0].keys()) if j.get("ok") else []
            sch[name] = cols
        out[f"{ds}:{db}"] = sch
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="training parquet with dataset_name/db_id columns")
    ap.add_argument("--api_url", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_parquet(a.parquet, columns=["dataset_name", "db_id"])
    out = fetch(a.api_url, sorted(set(zip(df.dataset_name, df.db_id))))
    json.dump(out, open(a.out, "w"))
    print(f"[schemas] {len(out)} databases, {sum(len(v) for v in out.values())} tables -> {a.out}")


if __name__ == "__main__":
    main()
