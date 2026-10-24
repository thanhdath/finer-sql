"""Review harness (subprocess): import grpo_writer from $REPO with the real GT caches / atomic op list / local APIs and call
reward_fn on groups of rollouts read from $CASES (json list of groups; each group = list of {sql, gold, db_id, sample_id}).
$MODE: none | ori_timeout (ORI call_sql_api answers a server-side timeout) | ori_raise (ORI check raises).
Prints ###RESULT### {rewards, recs, secs, last_spur}."""
import json, os, sys, time
sys.path.insert(0, os.environ["REPO"]); os.chdir(os.environ["REPO"])
out = os.environ["OUT"]
sys.argv = ["grpo_writer.py", "--model_path", os.environ["TOKENIZER"], "--gt_cache", os.environ["CORE_GT"], "--api_url", os.environ["CORE_API"],
            "--no-memory", "--atomic_canon", "v2", "--atomic_ops_file", os.environ["OPS"], "--dump_rollouts", os.path.join(out, "r.jsonl"),
            "--out_dir", out] + json.loads(os.environ["EXTRA"])
import grpo_writer as G
mode = os.environ.get("MODE", "none")
ORI = os.environ.get("ORI_API")
if mode == "ori_timeout":
    _c = G.call_sql_api
    def fake(ds, db, sql, *, api_url=None, **kw):
        if api_url == ORI:
            return {"ok": False, "statement_type": "", "rows": None, "row_count": None, "pandas_result": None, "notice": None,
                    "error": "timeout", "timed_out": True}
        return _c(ds, db, sql, api_url=api_url, **kw)
    G.call_sql_api = fake
elif mode == "ori_raise":
    _k = G.check_sql_correct
    def boom(ds, db, sql, gt, api_url=None, gt_cache=None):
        if api_url is not None:
            raise RuntimeError("ori boom")
        return _k(ds, db, sql, gt)
    G.check_sql_correct = boom
TH = "<think>\nI join the needed tables, filter on the requested values, and return exactly the columns the question asks for in the answer.\n</think>\n"
groups = json.load(open(os.environ["CASES"]))
rewards, secs, spur = [], [], []
for g in groups:
    t = time.time()
    r = G.reward_fn(["p"] * len(g), [TH + c["sql"] for c in g], groundtruth_sqls=[[c["gold"]] for c in g], db_id=[c["db_id"] for c in g],
                    dataset_name=["bird"] * len(g), sample_id=[c["sample_id"] for c in g])
    secs.append(time.time() - t); rewards += r
    G._DUMP.flush(0, None)
    spur += list(getattr(G, "_LAST_SPUR", []))
recs = [json.loads(l) for l in open(os.path.join(out, "r.jsonl"))]
print("###RESULT###" + json.dumps({"rewards": rewards, "recs": recs, "secs": secs, "last_spur": spur}))
