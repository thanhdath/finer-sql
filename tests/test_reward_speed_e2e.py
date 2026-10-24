"""reward-speed flags end-to-end through grpo_writer.reward_fn (CPU, tiny random model, live db_execution API).

Two 2-step CPU training runs with the same seed: the default reward path vs --atomic_memo --memory_fast (and a third
with --memory_embed_batch). Memory ON with a deterministic local embedding stub and a fresh Chroma store per run; the
policy's completions are replaced inside reward_fn by fixed think+SQL texts (executable, partly correct, near-duplicate
thoughts) so the atomic term, the memory reward, the quality gate and both save paths are all exercised. Every dumped
reward field (format, ex, executable, atomic, memory, exec_score, total) must be identical, and so must the ids added
to both Chroma collections.

    DUMP_TEST_API_URL=http://127.0.0.1:18231/execute python -m pytest tests/test_reward_speed_e2e.py -q
"""
import json
import os
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_dump_rollouts import API_URL, TOKENIZER, _api_up  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WORKER = r'''
import hashlib, json, os, re, sys
import numpy as np
sys.path.insert(0, os.environ["REPO"]); os.chdir(os.environ["REPO"])
out = os.environ["OUT"]
sys.argv = ["grpo_writer.py", "--model_path", os.environ["TOKENIZER"], "--gt_cache", os.environ["CACHE"], "--api_url", os.environ["API_URL"],
            "--chroma_path", os.path.join(out, "chroma"), "--dump_rollouts", os.path.join(out, "rollouts.jsonl"), "--atomic_canon", "v2",
            "--max_steps", "2", "--num_generations", "4", "--gradient_accumulation_steps", "8", "--out_dir", out] + json.loads(os.environ["EXTRA"])
import grpo_writer as G
from memory.compute_intrinsic_reward import IntrinsicRewardComputer

def stub_vec(text):
    v = np.zeros(64)
    toks = re.findall(r"\w+", text.lower())
    for t in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
        h = int.from_bytes(hashlib.md5(t.encode()).digest()[:8], "little"); v[h % 64] += 1.0 if (h >> 63) == 0 else -1.0
    return (v / (np.linalg.norm(v) or 1.0)).tolist()

class _It:
    def __init__(s, i, e): s.index, s.embedding = i, e
class Emb:
    def __init__(s): s.embeddings = s; s.requests = []
    def create(s, model, input, timeout=None):
        s.requests.append(len(input)); return type("R", (), {"data": [_It(i, stub_vec(t)) for i, t in enumerate(input)]})()
class Tok:
    def encode(s, t, add_special_tokens=False): return list(t)
    def decode(s, ids, skip_special_tokens=True): return "".join(ids)
import chromadb
C = IntrinsicRewardComputer.__new__(IntrinsicRewardComputer)
C._emb_client = Emb(); C.tokenizer = Tok(); C.max_embed_len = 8192; C.embedding_model = "stub"; C.request_timeout = 60
C.chroma_client = chromadb.PersistentClient(path=os.path.join(out, "chroma"))
C.reasoning_collection = C.chroma_client.get_or_create_collection("reasoning_paths")
C.failed_collection = C.chroma_client.get_or_create_collection("fail_reasoning_traces")
# seed one entry from another db so the other-db retrieval returns something
C.reasoning_collection.add(ids=["seed0"], embeddings=[stub_vec("count rows of the table zip code cbsa seed thought")], documents=["s"], metadatas=[{"db_id": "other_db"}])
G._INTRINSIC = C
SQLS = ["SELECT COUNT(*) FROM CBSA", "SELECT COUNT(CBSA) FROM CBSA", "SELECT COUNT(*) FROM CBSA WHERE CBSA > 0", "SELECT CBSA FROM CBSA LIMIT 1"]
PROMPT = "Table CBSA: metro areas\n  CBSA.CBSA: INTEGER PRIMARY KEY - cbsa code\n  CBSA.CBSA_name: TEXT\n  CBSA.CBSA_type: TEXT\nQuestion: how many CBSA rows?"
def thought(k):
    base = "I need to count the rows of the CBSA table. The column CBSA in table CBSA is the key and CBSA_name and CBSA_type describe each area so counting CBSA rows answers it"
    return base + (" variant %d " % (k % 3)) * (k % 2 + 1) + " then the final query reads CBSA directly from CBSA with no filter on CBSA_type at all"
_orig = G.reward_fn
def reward_fn(prompts, completions, **kw):
    comps = ["<think>\n" + thought(i) + "\n</think>\n" + SQLS[i % 4] for i in range(len(completions))]
    return _orig([PROMPT] * len(completions), comps, **kw)
reward_fn.__name__ = "reward_fn"
from datasets import Dataset
from transformers import Qwen2Config, Qwen2ForCausalLM, set_seed
from trl import GRPOConfig
set_seed(0)
tok = G.tokenizer
cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256)
model = Qwen2ForCausalLM(cfg)
rows = [{"prompt": f"<think>q{i}</think> SELECT {i}", "dataset_name": "bird", "sample_id": 100 + i, "db_id": "address",
         "groundtruth_sqls": ["SELECT COUNT(*) FROM CBSA", "SELECT COUNT(CBSA) FROM CBSA"]} for i in range(4)]
args = GRPOConfig(output_dir=out, use_cpu=True, bf16=False, fp16=False, report_to=[], per_device_train_batch_size=1,
                  gradient_accumulation_steps=8, num_generations=4, max_steps=2, max_completion_length=8, logging_steps=1, seed=42, save_strategy="no")
tr = G.APIAwareGRPOTrainer(model=model, args=args, train_dataset=Dataset.from_list(rows), reward_funcs=reward_fn, processing_class=tok)
tr.train()
ids = {n: sorted(getattr(C, n).get(include=[])["ids"]) for n in ("reasoning_collection", "failed_collection")}
print("###RESULT###" + json.dumps({"written": G._DUMP.written, "ids": ids, "embed_requests": C._emb_client.requests}))
'''


def _run(extra):
    import pickle
    out = tempfile.mkdtemp(prefix="reward_speed_e2e_")
    cache = os.path.join(out, "gt.pkl")
    import urllib.request
    req = urllib.request.Request(API_URL, data=json.dumps({"dataset_name": "bird", "db_id": "address", "sql": "SELECT COUNT(*) FROM CBSA",
                                                         "mode": "read_only"}).encode(), headers={"Content-Type": "application/json"})
    gold_rows = json.load(urllib.request.urlopen(req, timeout=30))["rows"]   # gold through the db_execution API
    with open(cache, "wb") as f:
        pickle.dump({("bird", "address", "SELECT COUNT(*) FROM CBSA"): gold_rows}, f)
    env = dict(os.environ, REPO=REPO, OUT=out, CACHE=cache, API_URL=API_URL, TOKENIZER=TOKENIZER, CUDA_VISIBLE_DEVICES="",
               EXTRA=json.dumps(extra))
    r = subprocess.run([sys.executable, "-c", WORKER], env=env, capture_output=True, text=True, timeout=900)
    assert "###RESULT###" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    res = json.loads(r.stdout.split("###RESULT###")[1].strip().splitlines()[0])
    lines = [json.loads(l) for l in open(os.path.join(out, "rollouts.jsonl"))]
    return res, lines


@pytest.mark.skipif(not _api_up(), reason=f"db_execution API not reachable at {API_URL}")
@pytest.mark.skipif(not os.path.isdir(TOKENIZER), reason="tokenizer dir absent")
def test_flags_give_identical_rewards_and_store():
    F = ["format", "ex", "executable", "atomic", "memory", "exec_score", "total", "verdict_ok", "sql"]
    base_res, base = _run([])
    assert len(base) == 16
    assert any(l["atomic"] > 0 for l in base) and any(l["ex"] == 2.0 for l in base) and any(0 < l["memory"] < 1 for l in base)
    assert base_res["ids"]["failed_collection"] and base_res["ids"]["reasoning_collection"] != ["seed0"]
    for extra in (["--atomic_memo", "--memory_fast"], ["--atomic_memo", "--memory_embed_batch"]):
        res, got = _run(extra)
        assert [[l[k] for k in F] for l in got] == [[l[k] for k in F] for l in base], extra
        assert res["ids"] == base_res["ids"], extra
        assert len(res["embed_requests"]) == 2 and len(base_res["embed_requests"]) > 2, (extra, res["embed_requests"])
