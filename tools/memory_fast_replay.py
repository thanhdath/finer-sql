#!/usr/bin/env python3
"""--memory_fast validation: replay recorded steps through the sequential memory path (compute_thought_reward,
save_failed_thought, save_thought, in rollout order, as reward_fn calls them) and through memory_fast_step, each against
its own COPY of a Chroma seed store, with a deterministic local embedding stub (no server). Compares per-rollout memory
values (==) and the ids added to both collections after every step. Item construction mirrors reward_fn's pre-pass.

  python tools/memory_fast_replay.py --dump D --parquet P --seed-store DIR --work DIR --steps 20 --out OUT.json
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from memory.compute_intrinsic_reward import IntrinsicRewardComputer, calculate_reasoning_quality  # noqa: E402

DIM = 1024


def stub_vec(text):
    """Hashed bag of words (unigrams + bigrams), L2-normalised: similar texts get high cosine, so the 0.9 dedup and the
    top-k retrieval are exercised."""
    v = np.zeros(DIM, dtype=np.float64)
    toks = re.findall(r"\w+", text.lower())
    for t in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
        h = int.from_bytes(hashlib.md5(t.encode()).digest()[:8], "little")
        v[h % DIM] += 1.0 if (h >> 63) == 0 else -1.0
    n = np.linalg.norm(v)
    return (v / n if n else v).tolist()


class _Item:
    def __init__(self, i, e):
        self.index, self.embedding = i, e


class StubEmb:
    def __init__(self):
        self.requests = 0
        self.texts = 0
        self.embeddings = self

    def create(self, model, input, timeout=None):  # noqa: A002
        self.requests += 1
        self.texts += len(input)
        return type("R", (), {"data": [_Item(i, stub_vec(t)) for i, t in enumerate(input)]})()


class Tok:
    def encode(self, t, add_special_tokens=False):
        return list(t)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(ids)


class ExactCollection:
    """In-memory exact-search stand-in for a Chroma collection (squared L2 in float32, float32 storage returned as
    float64 arrays like Chroma, existing ids ignored on add). Isolates the decision rules from HNSW approximation."""

    def __init__(self, ids, embs, metas):
        self.ids = list(ids)
        self.E = np.asarray(embs, dtype=np.float32).reshape(len(self.ids), -1) if len(self.ids) else np.zeros((0, DIM), np.float32)
        self.metas = list(metas)
        self.docs = [""] * len(self.ids)

    def count(self):
        return len(self.ids)

    def get(self, include=None, **kw):
        return {"ids": list(self.ids)}

    def add(self, ids, embeddings, documents, metadatas):
        for i, e, d, m in zip(ids, embeddings, documents, metadatas):
            if i in self.ids:
                continue
            self.ids.append(i); self.docs.append(d); self.metas.append(m)
            self.E = np.vstack([self.E, np.asarray(e, dtype=np.float32)[None, :]])

    def query(self, query_embeddings, n_results, where=None, include=None):
        out = {"ids": [], "distances": [], "embeddings": [], "documents": [], "metadatas": []}
        mask = np.ones(len(self.ids), bool)
        if where is not None:
            c = where["db_id"]
            dbs = np.array([m.get("db_id") for m in self.metas], dtype=object)
            mask = (dbs != c["$ne"]) if "$ne" in c else (dbs == c["$eq"])
        idx = np.nonzero(mask)[0]
        for q in query_embeddings:
            q = np.asarray(q, dtype=np.float32)
            d = ((self.E[idx] - q) ** 2).sum(axis=1)
            order = idx[np.argsort(d, kind="stable")[:n_results]]
            dd = np.sort(d, kind="stable")[:n_results]
            out["ids"].append([self.ids[j] for j in order]); out["distances"].append([float(x) for x in dd])
            out["embeddings"].append([self.E[j].astype(np.float64) for j in order])
            out["documents"].append([self.docs[j] for j in order]); out["metadatas"].append([self.metas[j] for j in order])
        return out


def computer(store, exact=False):
    import chromadb
    c = IntrinsicRewardComputer.__new__(IntrinsicRewardComputer)
    c._emb_client = StubEmb()
    c.tokenizer = Tok()
    c.max_embed_len = 8192
    c.embedding_model = "stub"
    c.request_timeout = 60
    c.chroma_client = chromadb.PersistentClient(path=store)
    c.reasoning_collection = c.chroma_client.get_or_create_collection("reasoning_paths")
    c.failed_collection = c.chroma_client.get_or_create_collection("fail_reasoning_traces")
    if exact:
        for name in ("reasoning_collection", "failed_collection"):
            g = getattr(c, name).get(include=["embeddings", "metadatas"])
            setattr(c, name, ExactCollection(g["ids"], g["embeddings"], g["metadatas"]))
    return c


def thought_for(prompt, r, k):
    cols = re.findall(r"\b([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\s*:", prompt)[:10]
    w = ["To", "answer", "this", "I", "check", "the", "schema", "first."]
    for t, c in cols[: 4 + (k % 5)]:
        w += ["Column", c, "of", t, "is", "relevant", "for", "the", "filter", "and", "the", "join."]
    w += ["Rollout", "variant", str(k % 7), "so", "the", "final", "query", "is"] + (r["sql"] or "SELECT 1").split()
    return " ".join(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--seed-store", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--out", default=None)
    ap.add_argument("--exact", action="store_true", help="exact-search in-memory collections (no HNSW) for both paths")
    a = ap.parse_args()
    import pandas as pd
    p = pd.read_parquet(a.parquet, columns=["dataset_name", "sample_id", "prompt_raw"])
    prompts = {(r.dataset_name, int(r.sample_id)): r.prompt_raw for r in p.itertuples()}
    by = {}
    for line in open(a.dump):
        r = json.loads(line)
        by.setdefault(r["step"], []).append(r)
    steps = sorted(by)[: a.steps]
    stores = {}
    for k in ("seq", "fast"):
        d = os.path.join(a.work, k)
        if os.path.exists(d):
            shutil.rmtree(d)
        shutil.copytree(a.seed_store, d)
        os.system(f"chmod -R u+w {d}")
        stores[k] = computer(d, a.exact)
    S, F = stores["seq"], stores["fast"]
    res = {"steps": len(steps), "rollouts": 0, "reward_rollouts": 0, "value_differ": 0, "max_abs_diff": 0.0,
           "save_candidates": 0, "adds_seq": 0, "adds_fast": 0, "added_ids_differ_steps": 0,
           "seq_s": 0.0, "fast_s": 0.0, "seq_embed_requests": 0, "fast_embed_requests": 0, "examples": []}
    for st in steps:
        rs = sorted(by[st], key=lambda r: r["index"])
        items, trs, seq_vals = [], [], []
        before = {k: set(getattr(S, c).get(include=[])["ids"]) for k, c in (("r", "reasoning_collection"), ("f", "failed_collection"))}
        t0 = time.perf_counter()
        for k, r in enumerate(rs):
            prompt = prompts[(r["dataset_name"], int(r["sample_id"]))]
            th = thought_for(prompt, r, k)
            es = 2.0 if r["ex"] >= 2.0 else (1.0 + r["atomic"] if r["executable"] == 1.0 else 0.0)
            # ---- sequential path (reward_fn loop order) ----
            tr = 0
            if es < 2.0:
                if len(th) > 30:
                    tr = S.compute_thought_reward(th, r["db_id"], top_k=a.top_k)
            else:
                tr = 1.0
            _, ins = calculate_reasoning_quality(th, pred_sql=r["sql"], prompt=prompt)
            if ins:
                if es >= 2.0:
                    S.save_thought(dataset_name=r["dataset_name"], db_id=r["db_id"], thought=th, model_name="m")
                else:
                    S.save_failed_thought(dataset_name=r["dataset_name"], db_id=r["db_id"], thought=th, model_name="m")
            seq_vals.append(tr)
            # ---- fast-path item (mirrors reward_fn's pre-pass) ----
            it = {"reward_text": th if (es < 2.0 and len(th) > 30) else None, "save": None, "save_text": None,
                  "dataset_name": r["dataset_name"], "db_id": r["db_id"], "model_name": "m"}
            if ins and th.strip():
                it["save"], it["save_text"] = ("reasoning" if es >= 2.0 else "failed"), th.strip()
                res["save_candidates"] += 1
            items.append(it)
            trs.append(1.0 if es >= 2.0 else 0)
        res["seq_s"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        out = F.memory_fast_step(items, top_k=a.top_k)
        res["fast_s"] += time.perf_counter() - t0
        assert out is not None
        fast_vals = [(x if it["reward_text"] else t) for x, it, t in zip(out, items, trs)]
        for k, (u, v, it) in enumerate(zip(seq_vals, fast_vals, items)):
            res["rollouts"] += 1
            res["reward_rollouts"] += bool(it["reward_text"])
            if u != v:
                res["value_differ"] += 1
                res["max_abs_diff"] = max(res["max_abs_diff"], abs(u - v))
                if len(res["examples"]) < 5:
                    res["examples"].append({"step": st, "index": k, "seq": u, "fast": v})
        after_s = {k: set(getattr(S, c).get(include=[])["ids"]) for k, c in (("r", "reasoning_collection"), ("f", "failed_collection"))}
        after_f = {k: set(getattr(F, c).get(include=[])["ids"]) for k, c in (("r", "reasoning_collection"), ("f", "failed_collection"))}
        res["adds_seq"] += sum(len(after_s[k] - before[k]) for k in before)
        res["adds_fast"] += sum(len(after_f[k] - before[k]) for k in before)
        if after_s != after_f:
            res["added_ids_differ_steps"] += 1
    res["seq_embed_requests"] = S._emb_client.requests
    res["fast_embed_requests"] = F._emb_client.requests
    res["seq_s"], res["fast_s"] = round(res["seq_s"], 3), round(res["fast_s"], 3)
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
