#!/usr/bin/env python3
"""Precompute the canonical (v2) op set of the gold and of every reference of a training parquet.

Output (JSON, default <parquet basename>.atomic_ops_v2.json next to the parquet):
  {"meta": {...}, "schemas": {"<dataset>:<db_id>": {table: [cols]}},
   "prompts": [{"dataset_name", "sample_id", "db_id", "n_refs", "n_opsets", "opsets": [[gold ops], [ref ops], ...]}]}
opsets[0] is always the gold's (empty list if the gold does not parse); duplicate op sets of other references are
collapsed (the max over the list is unchanged by duplicates).  The trainer loads this with --atomic_canon v2
--atomic_ops_file and never parses a reference at runtime.  The reference lists themselves are not changed.

    python tools/build_atomic_oplists.py --parquet train.parquet --schemas schemas.json [--out X.json] [--workers 16]
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from atomic_ops.ops_v2 import OpExtractor, OpParseError  # noqa: E402

_SCH = {}
_EX = {}


def _init(schemas):
    global _SCH
    _SCH = schemas


def _ext(ds, db):
    k = f"{ds}:{db}"
    if k not in _EX:
        _EX[k] = OpExtractor(_SCH.get(k))
    return _EX[k]


def _one(args):
    ds, sid, db, refs = args
    ex = _ext(ds, db)
    opsets, seen, fails = [], set(), 0
    for j, r in enumerate(refs):
        try:
            keys = sorted(ex.keys(r))
        except OpParseError:
            keys = []
            fails += 1
        t = tuple(keys)
        if j == 0:
            opsets.append(keys); seen.add(t)
        elif keys and t not in seen:
            opsets.append(keys); seen.add(t)
    return {"dataset_name": ds, "sample_id": int(sid) if str(sid).isdigit() else sid, "db_id": db, "n_refs": len(refs),
            "n_opsets": len(opsets), "parse_fail": fails, "opsets": opsets}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--schemas", required=True, help="tools/fetch_db_schemas.py output")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    df = pd.read_parquet(a.parquet, columns=["dataset_name", "sample_id", "db_id", "groundtruth_sqls"])
    schemas = json.load(open(a.schemas))
    jobs = [(ds, sid, db, list(refs)) for ds, sid, db, refs in zip(df.dataset_name, df.sample_id, df.db_id, df.groundtruth_sqls)]
    with Pool(a.workers, initializer=_init, initargs=(schemas,)) as pool:
        prompts = pool.map(_one, jobs, chunksize=64)
    out_path = a.out or (os.path.splitext(a.parquet)[0] + ".atomic_ops_v2.json")
    n_refs = sum(p["n_refs"] for p in prompts); n_sets = sum(p["n_opsets"] for p in prompts)
    fails = sum(p["parse_fail"] for p in prompts); gold_fail = sum(1 for p in prompts if not p["opsets"][0])
    meta = {"parquet": os.path.abspath(a.parquet), "parquet_sha256": sha256(a.parquet), "schemas": os.path.abspath(a.schemas),
            "schemas_sha256": sha256(a.schemas), "extractor": "atomic_ops/ops_v2.py",
            "extractor_sha256": sha256(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "atomic_ops", "ops_v2.py")),
            "rows": len(prompts), "references": n_refs, "opsets": n_sets, "parse_fail_refs": fails, "gold_parse_fail": gold_fail,
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump({"meta": meta, "schemas": schemas, "prompts": prompts}, open(out_path, "w"))
    print(f"[oplists] rows {len(prompts)} references {n_refs} distinct op sets {n_sets} parse_fail refs {fails} golds {gold_fail} "
          f"-> {out_path} sha256 {sha256(out_path)[:16]} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
