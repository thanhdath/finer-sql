#!/usr/bin/env python3
"""
Intrinsic reward computation for SQL writer memory – minimal, modeled after
`ChromaRetriever` (vLLM embeddings + Chroma collections).
"""

from __future__ import annotations
import numpy as np  # type: ignore
import chromadb  # type: ignore
from itertools import zip_longest
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib
import re


from vllm import LLM  # type: ignore
from transformers.models.auto.tokenization_auto import AutoTokenizer  # type: ignore


def extract_embedding(output_obj) -> List[float]:
    """Extract embedding from vLLM output object."""
    try:
        return output_obj.outputs.embedding  # vLLM >= 0.8.5
    except Exception:
        if hasattr(output_obj, "outputs") and isinstance(output_obj.outputs, list) and output_obj.outputs:
            maybe = output_obj.outputs[0]
            if hasattr(maybe, "embedding"):
                return maybe.embedding
        raise RuntimeError("Unexpected vLLM embed output structure; cannot extract embedding")


def embed_texts_vllm(texts: List[str], llm, tokenizer, max_length: int = 8192) -> List[List[float]]:
    """Embed texts using vLLM with truncation (mirrors save_reasoning_paths_to_chroma.py)."""
    truncated_texts: List[str] = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > max_length:
            ids = ids[:max_length]
        truncated_text = tokenizer.decode(ids, skip_special_tokens=True)
        truncated_texts.append(truncated_text)

    outputs = llm.embed(truncated_texts)
    embeddings = [extract_embedding(output) for output in outputs]
    return embeddings


from typing import List
import numpy as np
from openai import OpenAI
from transformers import AutoTokenizer

def _truncate_texts(texts: List[str], tokenizer, max_length: int) -> List[str]:
    out = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if len(ids) > max_length:
            ids = ids[:max_length]
        out.append(tokenizer.decode(ids, skip_special_tokens=True))
    return out

class IntrinsicRewardComputer:
    def __init__(
        self,
        chroma_path: str = "./chroma_db",
        collection_name: str = "reasoning_paths",
        *,
        embedding_api_base: str = "http://localhost:9000/v1",  # vLLM OpenAI endpoint
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        embedding_api_key: str = "dummy",  # vLLM will accept anything (or set with --api-key)
        max_embed_len: int = 8192,
        request_timeout: float = 60.0,
    ):
        self.embedding_api_base = embedding_api_base
        self.embedding_model = embedding_model
        self.max_embed_len = max_embed_len
        self.request_timeout = request_timeout

        # OpenAI SDK client pointed at vLLM
        self._emb_client = OpenAI(base_url=self.embedding_api_base, api_key=embedding_api_key)

        # tokenizer only for truncation
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model, trust_remote_code=True)

        import chromadb
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.reasoning_collection = self.chroma_client.get_or_create_collection(collection_name)
        self.failed_collection = self.chroma_client.get_or_create_collection("fail_reasoning_traces")

    def _embed_many(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        inputs = _truncate_texts(texts, self.tokenizer, self.max_embed_len)
        # SDK handles retries/transport; vLLM matches OpenAI schema
        resp = self._emb_client.embeddings.create(model=self.embedding_model, input=inputs, timeout=self.request_timeout)
        # Sort by index to align with inputs (usually already aligned)
        data_sorted = sorted(resp.data, key=lambda x: x.index)
        return [item.embedding for item in data_sorted]

    def _embed_one(self, text: str) -> List[float]:
        cache = getattr(self, "_emb_cache", None)
        if cache is not None:
            hit = cache.get(text)
            if hit is not None:
                return hit
        return self._embed_many([text])[0]

    def prefetch_embeddings(self, texts: List[str]) -> int:
        """--memory_embed_batch: embed the distinct texts in ONE request and serve later _embed_one(text) calls from it
        until clear_embedding_cache(). A text not prefetched is embedded live as before. Returns the number embedded."""
        uniq = list(dict.fromkeys(t for t in texts if isinstance(t, str) and t))
        self._emb_cache = dict(zip(uniq, self._embed_many(uniq))) if uniq else {}
        return len(uniq)

    def clear_embedding_cache(self) -> None:
        self._emb_cache = None

    # ───────────── --memory_fast: one step of the memory term with batched server/store calls ─────────────
    @staticmethod
    def _space(col) -> str:
        try:
            cfg = getattr(col, "configuration_json", None) or getattr(col, "configuration", None) or {}
            sp = (cfg.get("hnsw") or {}).get("space")
            if sp:
                return sp
        except Exception:  # noqa: BLE001
            pass
        md = getattr(col, "metadata", None) or {}
        return md.get("hnsw:space", "l2")

    @staticmethod
    def _dist(space: str, q: np.ndarray, e: np.ndarray) -> float:
        if space == "cosine":
            return float(1.0 - np.dot(q, e) / (np.linalg.norm(q) * np.linalg.norm(e)))
        if space == "ip":
            return float(1.0 - np.dot(q, e))
        d = q - e
        return float(np.dot(d, d))   # hnswlib l2 = squared euclidean

    @staticmethod
    def _where_ok(where: Optional[Dict[str, Any]], db_id: str) -> bool:
        if where is None:
            return True
        cond = where["db_id"]
        if "$ne" in cond:
            return db_id != cond["$ne"]
        return db_id == cond["$eq"]

    def memory_fast_step(self, items: List[Dict[str, Any]], top_k: int = 20, same_db_retrieval: bool = False,
                         only_same_db_retrieval: bool = False) -> Optional[List[Optional[float]]]:
        """The memory calls of one reward call, same decisions as the sequential path, batched.

        items, in rollout order: {"reward_text": str|None (compute_thought_reward on it), "save": None|"reasoning"|"failed",
        "save_text": str, "dataset_name", "db_id", "model_name"}. Sequential semantics reproduced:
          reward  = cosine(emb(text), centroid of the top_k reasoning_paths nearest to emb(text) under the db_id filter)
                    over the store AS IT IS AT THAT ROLLOUT (step-start store + reasoning adds of earlier rollouts of this
                    call); 0.0 when nothing is retrieved / norms vanish / any error (as compute_thought_reward).
          save    = query the target collection for its single nearest entry (no filter) over the store at that rollout;
                    skip if cosine(emb, nearest) >= 0.9, else add (same id / document / metadata fields).
        Implementation: ONE embedding request; one batched store query per (collection, filter); a brute-force shadow over
        this call's earlier adds (same space as the collection) merged with the store result by distance; ONE add per
        collection at the end, in rollout order. Returns None (caller runs the sequential path) if a batched server/store
        call fails before any write. Differences vs sequential: embedding float noise (one request instead of 2-3 per
        rollout) and HNSW approximation of the store part only."""
        try:
            texts = []
            for it in items:
                if it.get("reward_text"):
                    texts.append(it["reward_text"])
                if it.get("save") and it.get("save_text"):
                    texts.append(it["save_text"])
            uniq = list(dict.fromkeys(texts))
            E = dict(zip(uniq, self._embed_many(uniq))) if uniq else {}
            if only_same_db_retrieval:
                wf = lambda db: {"db_id": {"$eq": db}}  # noqa: E731
            elif same_db_retrieval:
                wf = lambda db: None  # noqa: E731
            else:
                wf = lambda db: {"db_id": {"$ne": db}}  # noqa: E731
            cols = {"reasoning": self.reasoning_collection, "failed": self.failed_collection}
            space = {k: self._space(c) for k, c in cols.items()}

            def batch_query(col, idx, emb_of, n, where):
                out = {}
                if not idx:
                    return out
                kw = {"query_embeddings": [emb_of(i) for i in idx], "n_results": n,
                      "include": ["documents", "metadatas", "distances", "embeddings"]}
                if where is not None:
                    kw["where"] = where
                res = col.query(**kw) or {}
                for j, i in enumerate(idx):
                    ids = ((res.get("ids") or [])[j:j + 1] or [[]])[0]
                    dists = ((res.get("distances") or [])[j:j + 1] or [[]])[0]
                    embs = (res.get("embeddings") if res.get("embeddings") is not None else [])
                    embs = embs[j] if len(embs) > j else []
                    out[i] = [(float(dists[k]), np.asarray(embs[k])) for k in range(len(ids))
                              if ids[k] is not None and k < len(dists) and k < len(embs) and embs[k] is not None]
                return out

            by_where: Dict[Any, List[int]] = {}
            for i, it in enumerate(items):
                if it.get("reward_text"):
                    by_where.setdefault(repr(wf(it["db_id"])), []).append(i)
            store_rew: Dict[int, list] = {}
            for _, idx in by_where.items():
                store_rew.update(batch_query(cols["reasoning"], idx, lambda i: E[items[i]["reward_text"]], top_k, wf(items[idx[0]]["db_id"])))
            store_nn = {}
            for k in cols:
                idx = [i for i, it in enumerate(items) if it.get("save") == k and it.get("save_text")]
                store_nn[k] = batch_query(cols[k], idx, lambda i: E[items[i]["save_text"]], 1, None)
        except Exception as e:  # noqa: BLE001
            print(f"[memory_fast] batched call failed ({type(e).__name__}: {e}); sequential path for this step")
            return None

        shadow: Dict[str, List[tuple]] = {"reasoning": [], "failed": []}   # (db_id, emb32)
        pending: Dict[str, Dict[str, list]] = {k: {"ids": [], "embeddings": [], "documents": [], "metadatas": []} for k in cols}
        rewards: List[Optional[float]] = []
        for i, it in enumerate(items):
            r = None
            if it.get("reward_text"):
                try:
                    a = np.array(E[it["reward_text"]], dtype=np.float32)
                    where = wf(it["db_id"])
                    cand = list(store_rew.get(i, []))
                    for db, e in shadow["reasoning"]:
                        if self._where_ok(where, db):
                            cand.append((self._dist(space["reasoning"], a, e.astype(np.float32)), e))
                    cand.sort(key=lambda x: x[0])
                    valid = [e for _, e in cand[:top_k]]
                    r = 0.0
                    if valid:
                        centroid = np.mean(valid, axis=0)
                        if centroid is not None and not np.isnan(centroid).any():
                            an, cn = np.linalg.norm(a), np.linalg.norm(centroid)
                            if an >= 1e-8 and cn >= 1e-8:
                                v = float(np.dot(a / an, centroid / cn))
                                r = 0.0 if np.isnan(v) else v
                except Exception:  # noqa: BLE001
                    r = 0.0
            rewards.append(r)
            k = it.get("save")
            t = it.get("save_text")
            if k and t:
                emb = E[t]
                a = np.array(emb, dtype=np.float32)
                cand = list(store_nn[k].get(i, []))
                for db, e in shadow[k]:
                    cand.append((self._dist(space[k], a, e.astype(np.float32)), e))
                skip = False
                if cand:
                    top = np.array(min(cand, key=lambda x: x[0])[1], dtype=np.float32)
                    na, nb = np.linalg.norm(a), np.linalg.norm(top)
                    if na > 0 and nb > 0 and float(np.dot(a, top) / (na * nb)) >= 0.9:
                        skip = True
                if not skip:
                    uid = self._make_thought_id(it["dataset_name"], it["db_id"], t)
                    md = {"dataset_name": it["dataset_name"], "model_name": it.get("model_name") or "unknown",
                          "schema_prompt": "", "reasoning_path": t, "created_at": datetime.utcnow().isoformat(),
                          "db_id": it["db_id"]}
                    pk = pending[k]
                    if uid in pk["ids"]:
                        continue   # sequential add of an existing id is a no-op
                    pk["ids"].append(uid); pk["embeddings"].append(emb); pk["documents"].append(t); pk["metadatas"].append(md)
                    shadow[k].append((it["db_id"], a.astype(np.float64)))   # as the store returns it: float32 values, float64 array
        for k, pk in pending.items():
            if not pk["ids"]:
                continue
            try:
                cols[k].add(**pk)
            except Exception as e:  # noqa: BLE001
                print(f"[memory_fast] batched add to {k} failed ({type(e).__name__}: {e}); adding one by one")
                for j in range(len(pk["ids"])):
                    try:
                        cols[k].add(ids=[pk["ids"][j]], embeddings=[pk["embeddings"][j]], documents=[pk["documents"][j]], metadatas=[pk["metadatas"][j]])
                    except Exception as e2:  # noqa: BLE001
                        print(f"[memory_fast] add failed: {e2}")
        return rewards

    def retrieve_similar_reasoning_paths_from_other_db_ids(
        self,
        thought: str,
        this_db_id: str,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        # Embed once
        q_emb = self._embed_one(thought)

        # Ask Chroma for everything we might need
        res = self.reasoning_collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where={"db_id": {"$ne": this_db_id}},
            include=["documents", "metadatas", "distances", "embeddings"]
        ) or {}

        # Extract first (and only) query's results with sensible fallbacks
        ids   = (res.get("ids")         or [[]])[0]
        docs  = (res.get("documents")   or [[]])[0]
        metas = (res.get("metadatas")   or [[]])[0]
        dists = (res.get("distances")   or [[]])[0]
        embs  = (res.get("embeddings")  or [[]])[0]

        if not ids:
            return []

        # Zip with padding; keep only rows that have an id
        out = []
        for _id, doc, meta, dist, emb in zip_longest(ids, docs, metas, dists, embs, fillvalue=None):
            if _id is None:
                continue
            out.append({
                "id": _id,
                "document": doc or "",
                "embedding": emb,
                "metadata": meta,
                "distance": dist,
            })
        return out

    def retrieve_similar_reasoning_paths(
        self,
        thought: str,
        this_db_id: str,
        top_k: int = 20,
        include_same_db: bool = False,
        only_same_db: bool = False
    ) -> List[Dict[str, Any]]:
        """Retrieve similar reasoning paths, optionally including the same db_id.
        
        Args:
            thought: The thought to retrieve similar paths for
            this_db_id: The current database ID
            top_k: Number of results to retrieve
            include_same_db: If True, retrieve from all db_ids (including same db_id)
            only_same_db: If True, retrieve only from the same db_id (exclude other db_ids)
        """
        # Embed once
        q_emb = self._embed_one(thought)

        # Build where clause based on include_same_db and only_same_db
        if only_same_db:
            where_clause = {"db_id": {"$eq": this_db_id}}
        elif include_same_db:
            where_clause = None
        else:
            where_clause = {"db_id": {"$ne": this_db_id}}

        # Ask Chroma for everything we might need
        query_kwargs = {
            "query_embeddings": [q_emb],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances", "embeddings"]
        }
        if where_clause is not None:
            query_kwargs["where"] = where_clause

        res = self.reasoning_collection.query(**query_kwargs) or {}

        # Extract first (and only) query's results with sensible fallbacks
        ids   = (res.get("ids")         or [[]])[0]
        docs  = (res.get("documents")   or [[]])[0]
        metas = (res.get("metadatas")   or [[]])[0]
        dists = (res.get("distances")   or [[]])[0]
        embs  = (res.get("embeddings")  or [[]])[0]

        if not ids:
            return []

        # Zip with padding; keep only rows that have an id
        out = []
        for _id, doc, meta, dist, emb in zip_longest(ids, docs, metas, dists, embs, fillvalue=None):
            if _id is None:
                continue
            out.append({
                "id": _id,
                "document": doc or "",
                "embedding": emb,
                "metadata": meta,
                "distance": dist,
            })
        return out


    def compute_thought_reward(
        self,
        thought: str,
        this_db_id: str,
        top_k: int = 20,
        same_db_retrieval: bool = False,
        only_same_db_retrieval: bool = False,
    ) -> float:
        """
        Compute intrinsic reward similar to `ChromaRetriever` style.
        Returns 0.0 when ChromaDB has no similar paths yet (early training / empty DB).

        Args:
            thought: The reasoning thought to compute reward for
            this_db_id: The current database ID
            top_k: Number of similar reasoning paths to retrieve
            same_db_retrieval: If True, allow retrieving from the same db_id
            only_same_db_retrieval: If True, retrieve only from the same db_id
        """
        try:
            a_emb = np.array(self._embed_one(thought), dtype=np.float32)

            if only_same_db_retrieval:
                similar_paths = self.retrieve_similar_reasoning_paths(
                    thought, this_db_id, top_k=top_k, only_same_db=True)
            elif same_db_retrieval:
                similar_paths = self.retrieve_similar_reasoning_paths(
                    thought, this_db_id, top_k=top_k, include_same_db=True)
            else:
                similar_paths = self.retrieve_similar_reasoning_paths_from_other_db_ids(
                    thought, this_db_id, top_k=top_k)

            valid_embeddings = [p["embedding"] for p in similar_paths if p.get("embedding") is not None]
            if not valid_embeddings:
                return 0.0

            centroid = np.mean(valid_embeddings, axis=0)
            if centroid is None or np.isnan(centroid).any():
                return 0.0

            a_norm = np.linalg.norm(a_emb)
            c_norm = np.linalg.norm(centroid)
            if a_norm < 1e-8 or c_norm < 1e-8:
                return 0.0
            a_emb = a_emb / a_norm
            centroid = centroid / c_norm
            R_exploit = float(np.dot(a_emb, centroid))
            if np.isnan(R_exploit):
                return 0.0
            return R_exploit
        except Exception:
            return 0.0

    def retrieve_similar_failed_paths(
        self,
        thought: str,
        this_db_id: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve similar failed reasoning paths."""
        q_emb = self._embed_one(thought)

        res = self.failed_collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances", "embeddings"]
        ) or {}

        ids  = (res.get("ids")         or [[]])[0]
        docs = (res.get("documents")   or [[]])[0]
        metas = (res.get("metadatas")  or [[]])[0]
        dists = (res.get("distances")  or [[]])[0]
        embs  = (res.get("embeddings") or [[]])[0]

        if not ids:
            return []

        out = []
        for _id, doc, meta, dist, emb in zip_longest(ids, docs, metas, dists, embs, fillvalue=None):
            if _id is None:
                continue
            if this_db_id and meta:
                meta_db_id = meta.get("db_id") if isinstance(meta, dict) else None
                if meta_db_id == this_db_id:
                    continue
            out.append({
                "id": _id,
                "document": doc or "",
                "embedding": emb,
                "metadata": meta,
                "distance": dist,
            })
            if len(out) >= top_k:
                break
        return out[:top_k]

    def compute_contrastive_thought_reward(
        self,
        thought: str,
        this_db_id: str,
        top_k: int = 10
    ) -> float:
        """Contrastive reward: pos (correct paths) − neg (failed paths), clipped to [-1, 1]."""
        try:
            a_emb = np.array(self._embed_one(thought), dtype=np.float32)
            a_emb_normalized = a_emb / (np.linalg.norm(a_emb) + 1e-8)

            similar_correct = self.retrieve_similar_reasoning_paths_from_other_db_ids(
                thought, this_db_id, top_k=top_k)
            similar_failed = self.retrieve_similar_failed_paths(thought, this_db_id, top_k=top_k)

            pos_sims = []
            for p in similar_correct:
                if p["embedding"] is not None:
                    emb = np.array(p["embedding"], dtype=np.float32)
                    emb_n = emb / (np.linalg.norm(emb) + 1e-8)
                    pos_sims.append(float(np.dot(a_emb_normalized, emb_n)))
            pos = max(pos_sims) if pos_sims else 0.0

            neg_sims = []
            for p in similar_failed:
                if p["embedding"] is not None:
                    emb = np.array(p["embedding"], dtype=np.float32)
                    emb_n = emb / (np.linalg.norm(emb) + 1e-8)
                    neg_sims.append(float(np.dot(a_emb_normalized, emb_n)))
            neg = max(neg_sims) if neg_sims else 0.0

            return float(np.clip(pos - neg, -1.0, 1.0))
        except Exception:
            return 0.0

    def compute_intrinsic_reward(
        self,
        question: str,
        response: str,
        beta_s: float = 1.0,
        beta_e: float = 1.0
    ) -> float:
        """Cosine similarity of response embedding to centroid of retrieved paths."""
        a_emb = np.array(self._embed_one(response), dtype=np.float32)

        similar_paths = self.retrieve_similar_reasoning_paths(question, question)
        docs = [p["document"] for p in similar_paths if isinstance(p.get("document"), str)]
        if not docs:
            return 0.0
        doc_embs = np.array(self._embed_many(docs), dtype=np.float32)
        if doc_embs.size == 0:
            return 0.0
        centroid = np.mean(doc_embs, axis=0)

        a_emb = a_emb / (np.linalg.norm(a_emb) + 1e-8)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        R_exploit = (np.dot(a_emb, centroid) + 1) / 2
        R_explore = 1.0
        return float(beta_s * float(R_exploit) + beta_e * float(R_explore))

    @staticmethod
    def _make_thought_id(dataset_name: str, db_id: str, thought: str) -> str:
        h = hashlib.sha1()
        h.update(dataset_name.encode('utf-8'))
        h.update(b'\x00')
        h.update(db_id.encode('utf-8'))
        h.update(b'\x00')
        h.update(thought.encode('utf-8'))
        return f"reasoning_{db_id}_{h.hexdigest()[:16]}"

    def save_thought(self, *, dataset_name: str, db_id: str, thought: str,
                     model_name: str | None = None, step: int | None = None,
                     thought_embedding_score: float | None = None) -> str | None:
        """Persist a correct thought into Chroma. Skips if cosine similarity to nearest ≥ 0.9."""
        if not isinstance(thought, str):
            return None
        thought = thought.strip()
        if not thought:
            return None

        embedding = self._embed_one(thought)

        res = self.reasoning_collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["embeddings", "documents", "metadatas", "distances"],
        ) or {}
        ids0  = (res.get("ids")        or [[]])[0]
        embs0 = (res.get("embeddings") or [[]])[0]
        if ids0 and len(embs0) > 0:
            top_emb = np.array(embs0[0], dtype=np.float32)
            a = np.array(embedding, dtype=np.float32)
            na, nb = np.linalg.norm(a), np.linalg.norm(top_emb)
            if na > 0 and nb > 0:
                cos = float(np.dot(a, top_emb) / (na * nb))
                if cos >= 0.9:
                    return None

        uid = self._make_thought_id(dataset_name, db_id, thought)
        metadata: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "model_name": model_name or "unknown",
            "schema_prompt": "",
            "reasoning_path": thought,
            "created_at": datetime.utcnow().isoformat(),
            "db_id": db_id,
        }
        if step is not None:
            metadata["step"] = step
        if thought_embedding_score is not None:
            metadata["thought_embedding_score"] = float(thought_embedding_score)

        self.reasoning_collection.add(
            embeddings=[embedding],
            documents=[thought],
            metadatas=[metadata],
            ids=[uid],
        )
        return uid

    def save_failed_thought(self, *, dataset_name: str, db_id: str, thought: str,
                            model_name: str | None = None, step: int | None = None,
                            thought_embedding_score: float | None = None) -> str | None:
        """Persist a failed thought into the fail_reasoning_traces collection."""
        if not isinstance(thought, str):
            return None
        thought = thought.strip()
        if not thought:
            return None

        embedding = self._embed_one(thought)

        res = self.failed_collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["embeddings", "documents", "metadatas", "distances"],
        ) or {}
        ids0  = (res.get("ids")        or [[]])[0]
        embs0 = (res.get("embeddings") or [[]])[0]
        if ids0 and len(embs0) > 0:
            top_emb = np.array(embs0[0], dtype=np.float32)
            a = np.array(embedding, dtype=np.float32)
            na, nb = np.linalg.norm(a), np.linalg.norm(top_emb)
            if na > 0 and nb > 0:
                cos = float(np.dot(a, top_emb) / (na * nb))
                if cos >= 0.9:
                    return None

        uid = self._make_thought_id(dataset_name, db_id, thought)
        metadata: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "model_name": model_name or "unknown",
            "schema_prompt": "",
            "reasoning_path": thought,
            "created_at": datetime.utcnow().isoformat(),
            "db_id": db_id,
        }
        if step is not None:
            metadata["step"] = step
        if thought_embedding_score is not None:
            metadata["thought_embedding_score"] = float(thought_embedding_score)

        self.failed_collection.add(
            embeddings=[embedding],
            documents=[thought],
            metadatas=[metadata],
            ids=[uid],
        )
        return uid


def batch_compute_intrinsic_reward(
    questions: List[str],
    responses: List[str],
    chroma_path: str = "./chroma_db",
    beta_s: float = 1.0,
    beta_e: float = 1.0,
    top_k: int = 20
) -> List[float]:
    if len(questions) != len(responses):
        raise ValueError("Questions and responses must have the same length")
    if not questions:
        return []
    comp = IntrinsicRewardComputer(chroma_path)
    return [comp.compute_intrinsic_reward(q, r, beta_s=beta_s, beta_e=beta_e)
            for q, r in zip(questions, responses)]


# ───────────────────────── QUALITY SCORER ─────────────────────────

def extract_columns_from_prompt(prompt: str) -> set:
    if not prompt:
        return set()
    columns = set()
    dot_patterns = re.findall(r'\b([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\s*:', prompt)
    for _, col in dot_patterns:
        columns.add(col.lower())
    return columns


def calculate_reasoning_quality(
    reasoning_text: str,
    pred_sql: Optional[str] = None,
    prompt: Optional[str] = None,
    debug: bool = False
) -> Tuple[float, bool]:
    """
    Quality Gate: Informativeness (schema density) + Distinction (TTR) + Conciseness.
    Returns (score: float 0.0-1.0, should_insert: bool).
    """
    if not reasoning_text or not isinstance(reasoning_text, str):
        return 0.0, False

    reasoning_text = reasoning_text.strip()
    tokens = reasoning_text.lower().split()
    total_tokens = len(tokens)
    words = [w for w in reasoning_text.split() if w.strip()]
    total_words = len(words)

    if total_words < 30:
        return 0.0, False
    if total_words > 2000:
        return 0.0, False

    if total_words < 50:
        length_score = 0.0
    elif 50 <= total_words <= 2000:
        length_score = 1.0
    else:
        length_score = max(0.0, 1.0 - (total_words - 2000) / 1000.0)

    prompt_columns = extract_columns_from_prompt(prompt) if prompt else set()
    if not prompt_columns:
        return 0.0, False

    text_lower = reasoning_text.lower()
    schema_hits = 0
    for col in prompt_columns:
        pattern = r'\b' + re.escape(col) + r'\b'
        schema_hits += len(re.findall(pattern, text_lower))

    schema_density = schema_hits / total_words if total_words > 0 else 0.0
    if schema_hits < 2 or schema_density < 0.05:
        return 0.0, False

    bigrams = list(zip(tokens, tokens[1:]))
    ttr_score = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
    if ttr_score < 0.6:
        return 0.0, False

    schema_score = min(1.0, schema_density * 10)
    composite_score = 0.33 * length_score + 0.33 * schema_score + 0.34 * ttr_score

    if debug:
        print(f"\n[quality] Length={total_words}w score={length_score:.2f} "
              f"schema_density={schema_density:.3f} score={schema_score:.2f} "
              f"TTR={ttr_score:.3f} composite={composite_score:.3f} → PASS")

    return composite_score, True
