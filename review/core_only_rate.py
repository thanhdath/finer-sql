"""Review: on CORE12 (pod) rollouts that CoRE paid, how many would --pay_requires_ori_ok demote (executed + mismatched on ORI)?
Uses the trainer's own check_sql_correct (7f8863a) against the local ORI API :18231 and the ORI cache 5fd69b6b."""
import json, os, sys, random, collections
from concurrent.futures import ThreadPoolExecutor
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); os.chdir(REPO)
A = list(sys.argv)
D = "/home/datht/finer-sql-tkde/experiments/12_final05b_20261024/data/"
sys.argv = ["grpo_writer.py", "--model_path", A[1], "--gt_cache", D + "gt_cache_core_final05b.pkl", "--api_url", "http://127.0.0.1:18221/execute",
            "--no-memory", "--out_dir", A[2]]
import grpo_writer as G, pickle, pandas as pd
ori = pickle.load(open(D + "gt_cache_ori_final05b.pkl", "rb"))
df = pd.read_parquet(D + "worth_train_final05b_core_enriched.parquet")
gold = {(r.dataset_name, int(r.sample_id)): list(r.groundtruth_sqls)[0] for r in df.itertuples()}
paid = {}
for l in open("/home/datht/finer-sql-tkde/experiments/12_final05b_20261024/scratch/core12_pod/rollouts.jsonl"):
    r = json.loads(l)
    if r.get("verdict_ok"):
        paid.setdefault((r["dataset_name"], r["sample_id"], r["db_id"], r["sql"]), 0); paid[(r["dataset_name"], r["sample_id"], r["db_id"], r["sql"])] += 1
keys = sorted(paid); random.Random(1).shuffle(keys); keys = keys[:int(A[3])]
def f(k):
    ds, sid, db, sql = k
    return k, G.check_sql_correct(ds, db, sql, gold[(ds, sid)], "http://127.0.0.1:18231/execute", ori)
c = collections.Counter(); ex = []
with ThreadPoolExecutor(16) as p:
    for k, v in p.map(f, keys):
        w = paid[k]; c["rollouts"] += w; c["unique"] += 1
        if v["ok"]: c["ori_ok"] += w
        elif (v.get("pred_res") or {}).get("ok") and (v.get("gt_res") or {}).get("ok"):
            c["core_only_demoted"] += w; c["core_only_unique"] += 1; ex.append(k)
        else: c["ori_err_or_unusable_kept"] += w
print(json.dumps(dict(c))); print(json.dumps(ex[:5])); json.dump(ex, open(os.path.join(A[2], "core_only_keys.json"), "w"))
