"""--dump_rollouts in grpo_writer.py: one JSONL line per scored rollout with step, ids, SQL, every reward term, total
and the group-relative advantage (CPU, tiny random Qwen2 through grpo_writer's own APIAwareGRPOTrainer + reward_fn).

grpo_writer is imported the way the launcher does (module-level argparse; tokenizer from a local checkpoint dir; a
tiny GT cache; --no-memory) in a subprocess; the SQL API is the db_execution endpoint in DUMP_TEST_API_URL (skipped if
down). A 2-step run with 2 prompts x 4 generations per step must write exactly 16 lines, steps {0, 1}, and advantages
that are group-centred (sum ~ 0 within each prompt group when any rollout differs).

    python -m pytest tests/test_dump_rollouts.py -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import urllib.request

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_URL = os.environ.get("DUMP_TEST_API_URL", "http://127.0.0.1:18383/execute")
TOKENIZER = os.environ.get("DUMP_TEST_TOKENIZER",
                           "/home/datht/finer-sql-tkde/experiments/05_core05f_finer_config_20261023/ori_a6000/stage/finer05b_bird")


def _api_up():
    try:
        req = urllib.request.Request(API_URL, data=json.dumps({"dataset_name": "bird", "db_id": "address", "sql": "SELECT 1 AS one", "mode": "read_only"}).encode(),
                                     headers={"Content-Type": "application/json"})
        return bool(json.load(urllib.request.urlopen(req, timeout=10)).get("ok"))
    except Exception:
        return False


WORKER = r'''
import json, os, pickle, sys
sys.path.insert(0, os.environ["REPO"]); os.chdir(os.environ["REPO"])
out = os.environ["OUT"]
sys.argv = ["grpo_writer.py", "--model_path", os.environ["TOKENIZER"], "--gt_cache", os.environ["CACHE"], "--api_url", os.environ["API_URL"],
            "--no-memory", "--chroma_path", os.path.join(out, "chroma"), "--dump_rollouts", os.path.join(out, "rollouts.jsonl"),
            "--max_steps", "2", "--num_generations", "4", "--gradient_accumulation_steps", "8", "--out_dir", out]
import grpo_writer as G
from datasets import Dataset
from transformers import Qwen2Config, Qwen2ForCausalLM, set_seed
from trl import GRPOConfig
set_seed(0)
tok = G.tokenizer
cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256)
model = Qwen2ForCausalLM(cfg)
rows = [{"prompt": f"<think>q{i}</think> SELECT {i}", "dataset_name": "bird", "sample_id": 100 + i, "db_id": "address",
         "groundtruth_sqls": ["SELECT COUNT(*) FROM CBSA"]} for i in range(4)]
args = GRPOConfig(output_dir=out, use_cpu=True, bf16=False, fp16=False, report_to=[], per_device_train_batch_size=1,
                  gradient_accumulation_steps=8, num_generations=4, max_steps=2, max_completion_length=8, logging_steps=1, seed=42, save_strategy="no")
tr = G.APIAwareGRPOTrainer(model=model, args=args, train_dataset=Dataset.from_list(rows), reward_funcs=G.reward_fn, processing_class=tok)
tr.train()
print("###RESULT###" + json.dumps({"written": G._DUMP.written, "misaligned": G._DUMP.misaligned, "errors": G._DUMP.errors, "pending": len(G._DUMP.pending)}))
'''


@pytest.mark.skipif(not _api_up(), reason=f"db_execution API not reachable at {API_URL}")
@pytest.mark.skipif(not os.path.isdir(TOKENIZER), reason="tokenizer dir absent")
def test_two_step_cpu_run_dumps_every_rollout():
    import pickle
    out = tempfile.mkdtemp(prefix="dump_rollouts_")
    cache = os.path.join(out, "gt.pkl")
    with open(cache, "wb") as f:
        pickle.dump({("bird", "address", "SELECT COUNT(*) FROM CBSA"): [{"COUNT(*)": 1}]}, f)
    env = dict(os.environ, REPO=REPO, OUT=out, CACHE=cache, API_URL=API_URL, TOKENIZER=TOKENIZER, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-c", WORKER], env=env, capture_output=True, text=True, timeout=900)
    assert "###RESULT###" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    res = json.loads(r.stdout.split("###RESULT###")[1].strip().splitlines()[0])
    lines = [json.loads(l) for l in open(os.path.join(out, "rollouts.jsonl"))]
    assert res["written"] == len(lines) == 16 and res["misaligned"] == 0 and res["errors"] == 0 and res["pending"] == 0, (res, len(lines))
    assert sorted({l["step"] for l in lines}) == [0, 1]
    keys = {"step", "dataset_name", "sample_id", "db_id", "sql", "format", "ex", "executable", "atomic", "memory", "total", "advantage", "verdict_ok"}
    assert all(keys <= set(l) for l in lines)
    assert all(l["advantage"] is not None for l in lines)
    groups = {}
    for l in lines:
        groups.setdefault((l["step"], l["sample_id"]), []).append(l)
    assert all(len(g) == 4 for g in groups.values()) and len(groups) == 4
    for g in groups.values():
        if len({x["total"] for x in g}) > 1:
            assert abs(sum(x["advantage"] for x in g)) < 1e-4
    # a line's terms recompose its total exactly as reward_fn does (format gate)
    for l in lines:
        exp_total = 0.0 if l["format"] == 0 else l["exec_score"] + l["format"] + l["memory"]
        assert abs(l["total"] - exp_total) < 1e-9
    assert "[dump_rollouts] every scored rollout" in r.stdout
