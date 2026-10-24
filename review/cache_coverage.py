"""Review: every CoRE-arm training gold (groundtruth_sqls[0]) must have an entry under the same key in the ORI cache.
Counts certifiability per cache using the trainer's own classify_gold (imported without running the trainer)."""
import pickle, sys, collections, ast, re
import pandas as pd
D = "/home/datht/finer-sql-tkde/experiments/12_final05b_20261024/data/"
core = pickle.load(open(D + "gt_cache_core_final05b.pkl", "rb"))
ori = pickle.load(open(D + "gt_cache_ori_final05b.pkl", "rb"))
df = pd.read_parquet(D + "worth_train_final05b_core_enriched.parquet")
print("columns", list(df.columns)[:20], len(df))
src = open("grpo_writer.py").read()
# pull classify_gold + constants without executing the trainer
from typing import Any
ns = {"Any": Any}
start = src.index("GOLD_OK"); 
m = re.search(r"\n(GOLD_[A-Z_]+ = .*\n)+", src)
exec(m.group(0), ns)
fn = src[src.index("def classify_gold"):]; fn = fn[:fn.index("\ndef ", 10)]
exec(fn, ns)
cls = ns["classify_gold"]
c = collections.Counter()
for _, r in df.iterrows():
    gts = r["groundtruth_sqls"]; g = list(gts)[0]
    k = (r["dataset_name"], r["db_id"], g)
    c["rows"] += 1
    c["core_has"] += k in core; c["ori_has"] += k in ori
    if k in core and k in ori:
        c[("core", cls(core[k]))] += 1; c[("ori", cls(ori[k]))] += 1
print(dict(c))
print("key sets equal:", set(core) == set(ori))
