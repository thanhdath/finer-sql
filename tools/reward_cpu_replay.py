#!/usr/bin/env python3
"""CPU replay of the per-rollout reward path of grpo_writer.py from --dump_rollouts records. No SQL, no embedding
server, no Chroma. The atomic function under test is taken verbatim (ast) from a grpo_writer.py source file, so the
"old" path is the trainer of record's code and not a re-implementation.

  profile : time per component per rollout on N recorded steps
  identity: recompute atomic for dump rollouts with the old function and the new batched path; count differences

  python tools/reward_cpu_replay.py profile  --dump D --ops OPS.json --parquet P --steps 10 --old-src grpo_writer.py
  python tools/reward_cpu_replay.py identity --dump D --ops OPS.json --min 5000 --old-src OLD.py --new-src grpo_writer.py
"""
import argparse
import ast
import json
import os
import statistics
import sys
import tempfile
import time
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atomic_ops.ops_v2 import OpExtractor, OpParseError, jaccard, load_ops_file  # noqa: E402
from atomic_ops.reward import AtomicOpsReward  # noqa: E402


def load_funcs(src_path: str, names: List[str], ns: Dict[str, Any]) -> Dict[str, Any]:
    """exec the named top-level defs/classes/assignments of src_path into ns (verbatim source)."""
    src = open(src_path).read()
    tree = ast.parse(src)
    for node in tree.body:
        nm = getattr(node, "name", None)
        if nm in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), src_path, "exec"), ns)
    missing = [n for n in names if n not in ns]
    assert not missing, f"not found in {src_path}: {missing}"
    return ns


def make_ns(ops_path: str) -> Dict[str, Any]:
    opsets, schemas, _ = load_ops_file(ops_path)
    import typing
    return {"Any": Any, "List": List, "Dict": Dict, "Optional": typing.Optional, "Tuple": typing.Tuple, "Set": typing.Set,
            "_V2Extractor": OpExtractor, "_V2ParseError": OpParseError, "_v2_jaccard": jaccard,
            "_V2_OPSETS": opsets, "_V2_SCHEMAS": schemas, "_V2_EXTRACTORS": {}, "_V2_RUNTIME_CACHE": {},
            "_V2_STATS": {"prebuilt": 0, "runtime": 0, "pred_parse_fail": 0},
            "ATOMIC_OPS_REWARD": AtomicOpsReward(dialect="sqlite", e=0.05, beta=0.79, gamma=0.20),  # trainer defaults, used by E10
            "ATOMIC_SCORE_WORKERS": 0, "os": os}


def read_dump(path: str, max_records: int = None):
    recs = {}
    with open(path) as f:
        for i, line in enumerate(f):
            if max_records is not None and i >= max_records:
                break
            r = json.loads(line)
            if r.get("step") is None:
                continue
            recs[(r["step"], r["dataset_name"], str(r["sample_id"]), r["index"])] = r
    return sorted(recs.values(), key=lambda r: (r["step"], r["index"]))


def gold_lists(parquet: str):
    import pandas as pd
    p = pd.read_parquet(parquet, columns=["dataset_name", "sample_id", "groundtruth_sqls", "prompt_raw"])
    return {(r.dataset_name, int(r.sample_id)): (list(r.groundtruth_sqls), r.prompt_raw) for r in p.itertuples()}


def synthetic_thought(prompt: str, sql: str) -> str:
    """No thought text is dumped. Proxy of a typical 0.5B think block: ~180 words naming schema columns and the SQL."""
    import re
    cols = re.findall(r"\b([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\s*:", prompt or "")[:12]
    words = ["To", "answer", "the", "question", "I", "need", "to", "look", "at", "the", "relevant", "tables", "and", "columns."]
    for t, c in cols:
        words += ["The", "column", c, "in", "table", t, "stores", "the", "value", "needed", "for", "filtering", "or", "joining."]
    words += ["So", "the", "query", "is"] + (sql or "SELECT 1").split()
    return " ".join(words[:400])


def cmd_profile(a):
    ns = make_ns(a.ops)
    load_funcs(a.old_src, ["_v2_extractor", "atomic_score_v2", "has_format_bonus", "extract_sql_from_completion"], ns)
    from memory.compute_intrinsic_reward import calculate_reasoning_quality, _truncate_texts
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        print(f"[profile] tokenizer unavailable: {e}")
    gl = gold_lists(a.parquet)
    recs = read_dump(a.dump)
    steps = sorted({r["step"] for r in recs})
    pick = set(steps[:: max(1, len(steps) // a.steps)][: a.steps])
    recs = [r for r in recs if r["step"] in pick]
    T = {k: [] for k in ["atomic_v2", "atomic_extract_only", "atomic_maxref_only", "json_dump", "quality_gate", "tokenizer_truncate", "format_extract"]}
    n_atomic = 0
    dump_f = tempfile.NamedTemporaryFile("w", delete=False, dir=a.tmp, suffix=".jsonl")
    for r in recs:
        sid = int(r["sample_id"])
        gts, prompt = gl[(r["dataset_name"], sid)]
        thought = synthetic_thought(prompt, r["sql"])
        comp = f"<think>\n{thought}\n</think>\n{r['sql'] or ''}"
        t0 = time.perf_counter(); ns["has_format_bonus"](comp); ns["extract_sql_from_completion"](comp); T["format_extract"].append(time.perf_counter() - t0)
        if r["executable"] == 1.0:  # atomic is computed only on executable-but-not-paid rollouts
            n_atomic += 1
            t0 = time.perf_counter()
            v = ns["atomic_score_v2"](r["sql"], r["dataset_name"], r["db_id"], sid, gts)
            T["atomic_v2"].append(time.perf_counter() - t0)
            ex = ns["_v2_extractor"](r["dataset_name"], r["db_id"])
            t0 = time.perf_counter()
            try:
                po = ex.keys(r["sql"])
            except OpParseError:
                po = set()
            T["atomic_extract_only"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            best = 0.0
            for ro in ns["_V2_OPSETS"][(r["dataset_name"], sid)]:
                sc = ns["ATOMIC_OPS_REWARD"].execution_weighted_jaccard(jaccard(po, ro))
                best = max(best, sc)
            T["atomic_maxref_only"].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); calculate_reasoning_quality(thought, pred_sql=r["sql"], prompt=prompt); T["quality_gate"].append(time.perf_counter() - t0)
        if tok is not None:
            t0 = time.perf_counter(); _truncate_texts([thought], tok, 8192); T["tokenizer_truncate"].append(time.perf_counter() - t0)
        rec = {k: r[k] for k in ["call", "index", "dataset_name", "sample_id", "db_id", "sql", "format", "ex", "executable", "atomic",
                                 "memory", "exec_score", "total", "verdict_ok", "pred_ok", "pred_error", "gold_unusable", "ts"]}
        t0 = time.perf_counter(); rec["step"] = r["step"]; rec["advantage"] = r["advantage"]; dump_f.write(json.dumps(rec, default=str) + "\n"); T["json_dump"].append(time.perf_counter() - t0)
    dump_f.close(); os.unlink(dump_f.name)
    n = len(recs)
    out = {"dump": a.dump, "steps": sorted(pick), "rollouts": n, "atomic_rollouts": n_atomic, "components": {}}
    for k, v in T.items():
        if not v:
            continue
        out["components"][k] = {"calls": len(v), "ms_per_call_mean": round(1000 * statistics.mean(v), 3),
                                "ms_per_call_median": round(1000 * statistics.median(v), 3),
                                "ms_per_rollout_avg": round(1000 * sum(v) / n, 3),
                                "ms_per_32_rollouts": round(32000 * sum(v) / n, 1)}
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


def cmd_identity(a):
    """old: atomic_score_v2 of --old-src called per rollout in order. new: the batch API of --new-src
    (atomic_scores_v2_batch) on each step's executable-unpaid rollouts. Compares floats with ==, also vs the dump value."""
    recs = read_dump(a.dump, a.max_records)
    gl = gold_lists(a.parquet)
    ns_old = make_ns(a.ops)
    load_funcs(a.old_src, ["_v2_extractor", "atomic_score_v2"], ns_old)
    ns_new = make_ns(a.ops)
    ns_new["ATOMIC_SCORE_WORKERS"] = a.workers
    load_funcs(a.new_src, ["_v2_extractor", "atomic_score_v2", "atomic_scores_v2_batch"], ns_new)
    by_step: Dict[int, list] = {}
    for r in recs:
        if r["executable"] == 1.0:
            by_step.setdefault(r["step"], []).append(r)
    n = diff_on = diff_od = 0
    t_old = t_new = 0.0
    examples = []
    for st, rs in sorted(by_step.items()):
        args = [(r["sql"], r["dataset_name"], r["db_id"], int(r["sample_id"]), gl[(r["dataset_name"], int(r["sample_id"]))][0]) for r in rs]
        t0 = time.perf_counter(); old = [ns_old["atomic_score_v2"](*x) for x in args]; t_old += time.perf_counter() - t0
        t0 = time.perf_counter(); new = ns_new["atomic_scores_v2_batch"](args); t_new += time.perf_counter() - t0
        for r, o, w in zip(rs, old, new):
            n += 1
            if o != w:
                diff_on += 1
                if len(examples) < 5:
                    examples.append({"step": st, "sql": r["sql"][:120], "old": o, "new": w})
            if o != r["atomic"]:
                diff_od += 1
        if n >= a.min and a.stop_at_min:
            break
    steps = len({s for s in by_step if s <= st})
    out = {"dump": a.dump, "rollouts_compared": n, "steps": steps, "old_vs_new_differ": diff_on, "old_vs_dump_differ": diff_od,
           "old_s_total": round(t_old, 3), "new_s_total": round(t_new, 3), "workers": a.workers,
           "old_ms_per_32_atomic_rollouts": round(32000 * t_old / max(n, 1), 1), "new_ms_per_32_atomic_rollouts": round(32000 * t_new / max(n, 1), 1),
           "examples": examples}
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    pool = ns_new.get("_ATOMIC_POOL")
    if pool is not None:
        pool.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["profile", "identity"])
    ap.add_argument("--dump", required=True)
    ap.add_argument("--ops", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--old-src", default=os.path.join(ROOT, "grpo_writer.py"))
    ap.add_argument("--new-src", default=os.path.join(ROOT, "grpo_writer.py"))
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--min", type=int, default=5000)
    ap.add_argument("--stop-at-min", action="store_true")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--tmp", default=tempfile.gettempdir())
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    {"profile": cmd_profile, "identity": cmd_identity}[a.cmd](a)


if __name__ == "__main__":
    main()
