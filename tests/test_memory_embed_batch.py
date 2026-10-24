"""--memory_embed_batch (reward-speed): CPU only, no embedding server, no Chroma.

(a) With a deterministic fake embedding server and recording fake collections, the per-rollout memory calls of the
    reward loop (compute_thought_reward on unpaid rollouts, save_failed_thought / save_thought) produce the SAME
    sequence of Chroma calls (method, query embedding, n_results, add ids/documents) and the same rewards with and
    without prefetch_embeddings; with prefetch the server gets exactly ONE request whose texts are the distinct texts
    in first-use order, and no per-rollout request.
(b) The texts reward_fn prefetches (has_format_bonus thought, else completion.strip(), stripped) cover every text the
    loop embeds.
(c) Default (no prefetch) path: requests are one text each, in loop order, identical to the original code.

    python -m pytest tests/test_memory_embed_batch.py -q
"""
from __future__ import annotations

import ast
import hashlib
import os
import sys
import typing

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from memory.compute_intrinsic_reward import IntrinsicRewardComputer  # noqa: E402


def _vec(text):
    h = hashlib.sha256(text.encode()).digest()
    return [((b / 255.0) - 0.5) for b in h[:16]]


class _Item:
    def __init__(self, i, e):
        self.index, self.embedding = i, e


class _Resp:
    def __init__(self, data):
        self.data = data


class FakeEmb:
    def __init__(self):
        self.requests = []
        self.embeddings = self

    def create(self, model, input, timeout=None):  # noqa: A002
        self.requests.append(list(input))
        return _Resp([_Item(i, _vec(t)) for i, t in enumerate(input)])


class FakeTok:
    def encode(self, t, add_special_tokens=False):
        return list(t)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(ids)


class FakeCollection:
    def __init__(self, log, name):
        self.log, self.name, self.docs = log, name, []

    def query(self, query_embeddings, n_results, where=None, include=None, **kw):
        self.log.append((self.name, "query", tuple(map(tuple, query_embeddings)), n_results, repr(where)))
        docs = self.docs[-n_results:]
        return {"ids": [[d[0] for d in docs]], "documents": [[d[1] for d in docs]],
                "metadatas": [[{"db_id": d[2]} for d in docs]], "distances": [[0.1] * len(docs)],
                "embeddings": [[d[3] for d in docs]]}

    def add(self, embeddings, documents, metadatas, ids):
        self.log.append((self.name, "add", tuple(ids), tuple(documents)))
        for e, d, m, i in zip(embeddings, documents, metadatas, ids):
            self.docs.append((i, d, m["db_id"], e))


def _computer(log):
    c = IntrinsicRewardComputer.__new__(IntrinsicRewardComputer)
    c._emb_client = FakeEmb()
    c.tokenizer = FakeTok()
    c.max_embed_len = 8192
    c.embedding_model = "fake"
    c.request_timeout = 30
    c.reasoning_collection = FakeCollection(log, "reasoning")
    c.failed_collection = FakeCollection(log, "failed")
    return c


def _load_format_fn():
    src = open(os.path.join(ROOT, "grpo_writer.py")).read()
    ns = {"Optional": typing.Optional, "Tuple": typing.Tuple}
    for node in ast.parse(src).body:
        if getattr(node, "name", None) in ("has_format_bonus", "extract_sql_from_completion"):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "grpo_writer.py", "exec"), ns)
    return ns["has_format_bonus"]


THOUGHTS = ["I join orders and customers on customer_id then count the rows per city " * 3,
            "The question asks for the maximum salary in the employees table grouped by department " * 3,
            "I join orders and customers on customer_id then count the rows per city " * 3,   # duplicate text
            "Filter singer by age above 30 and list the names in descending order of age " * 3,
            "no think block here"]
COMPLETIONS = [f"<think>\n{t}\n</think>\nSELECT 1" for t in THOUGHTS[:4]] + [THOUGHTS[4] + " SELECT 2"]
PAID = [False, True, False, False, False]


def _run(prefetch: bool):
    """The memory part of reward_fn's loop, in order (exec_score < 2: compute_thought_reward + save_failed_thought;
    exec_score == 2: save_thought), on the completions above."""
    log = []
    c = _computer(log)
    fmt = _load_format_fn()
    if prefetch:
        texts = []
        for comp in COMPLETIONS:
            th, _ = fmt(comp)
            t = (th if th is not None else comp.strip()).strip()
            if t:
                texts.append(t)
        c.prefetch_embeddings(texts)
    rewards, used = [], []
    for comp, paid in zip(COMPLETIONS, PAID):
        thought, _ = fmt(comp)
        if not paid:
            if thought is None:
                thought = comp.strip()
            used.append(thought)
            rewards.append(c.compute_thought_reward(thought, "db_x", top_k=3))
            used.append(thought.strip())
            c.save_failed_thought(dataset_name="bird", db_id="db_x", thought=thought, model_name="m")
        elif thought:
            used.append(thought.strip())
            c.save_thought(dataset_name="bird", db_id="db_x", thought=thought, model_name="m")
            rewards.append(1.0)
    if prefetch:
        c.clear_embedding_cache()
    return log, rewards, c._emb_client.requests, used


def test_batched_prefetch_keeps_call_order_and_values():
    log_seq, rew_seq, req_seq, used = _run(False)
    log_bat, rew_bat, req_bat, _ = _run(True)
    assert log_seq == log_bat                      # same Chroma calls, same order, same embeddings, same adds
    assert rew_seq == rew_bat
    assert len(req_bat) == 1                        # one request per reward call
    assert req_bat[0] == list(dict.fromkeys(req_bat[0]))
    assert set(used) <= set(req_bat[0])             # every loop text was prefetched (no live request)


def test_default_path_one_text_per_request_in_loop_order():
    _, _, req_seq, _ = _run(False)
    assert all(len(r) == 1 for r in req_seq)
    flat = [r[0] for r in req_seq]
    # compute_thought_reward embeds twice (reward + retrieval) and each save once: 3 per unpaid, 1 per paid
    assert len(flat) == 3 * PAID.count(False) + PAID.count(True)


def test_cache_cleared_after_call():
    log = []
    c = _computer(log)
    c.prefetch_embeddings(["abc"])
    c.clear_embedding_cache()
    c._embed_one("abc")
    assert c._emb_client.requests == [["abc"], ["abc"]]
