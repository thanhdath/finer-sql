#!/usr/bin/env python3
"""
Intrinsic reward computation for SQL writer memory – minimal, modeled after
`ChromaRetriever` (vLLM embeddings + Chroma collections).
"""

from __future__ import annotations
import numpy as np  # type: ignore
import chromadb  # type: ignore
from itertools import zip_longest
from typing import List, Dict, Any
from datetime import datetime
import hashlib


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





# class IntrinsicRewardComputer:
#     """
#     Minimal class aligned with `ChromaRetriever` style.
#     Uses vLLM to embed, and Chroma to retrieve reasoning paths.
#     """

#     def __init__(self, chroma_path: str = "./chroma_db", collection_name: str = "reasoning_paths"):
#         print("[✓] Loading Qwen3-Embedding via vLLM...")
#         self.model_path = 'Qwen/Qwen3-Embedding-0.6B'
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
#         self.llm = LLM(model=self.model_path, task="embed", trust_remote_code=True, max_model_len=8192, gpu_memory_utilization=0.4)

#         print(f"[✓] Connecting to ChromaDB at {chroma_path}...")
#         self.chroma_client = chromadb.PersistentClient(path=chroma_path)
#         self.reasoning_collection = self.chroma_client.get_or_create_collection(collection_name)
# def _embed_one(self, text: str) -> List[float]:
#         return embed_texts_vllm([text], self.llm, self.tokenizer)[0]


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
        return self._embed_many([text])[0]

    def retrieve_similar_reasoning_paths(
        self,
        question: str,
        top_k: int = 20
    ) -> List[Dict]:
        q_embedding = self._embed_one(question)

        results = self.reasoning_collection.query(
            query_embeddings=[q_embedding],
            n_results=top_k
        )

        if not results.get("ids") or not results["ids"] or not results["ids"][0]:
            return []
        ids0 = results.get("ids")[0]
        docs0 = (results.get("documents") or [[]])[0] if results.get("documents") else [""] * len(ids0)
        metas0 = (results.get("metadatas") or [[]])[0] if results.get("metadatas") else [None] * len(ids0)
        dists0 = (results.get("distances") or [[]])[0] if results.get("distances") else [None] * len(ids0)
        embeddings0 = (results.get("embeddings") or [[]])[0] if results.get("embeddings") else [None] * len(ids0)

        out = []
        for i in range(len(ids0)):
            doc_i = docs0[i] if i < len(docs0) else ""
            meta_i = metas0[i] if i < len(metas0) else None
            dist_i = dists0[i] if i < len(dists0) else None
            embedding_i = embeddings0[i] if i < len(embeddings0) else None
            out.append({
                "id": ids0[i],
                "document": doc_i,
                "embedding": embedding_i,
                "metadata": meta_i,
                "distance": dist_i,
            })
        return out

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

        # Extract first (and only) query’s results with sensible fallbacks
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
    ) -> float:
        """
        Compute intrinsic reward similar to `ChromaRetriever` style.
        For now, uses only exploitation vs correct reasoning paths (no explicit failed paths stored).
        """
        a_emb = np.array(self._embed_one(thought), dtype=np.float32)
        
        similar_paths = self.retrieve_similar_reasoning_paths_from_other_db_ids(thought, this_db_id, top_k=20)

        centroid = np.mean([p["embedding"] for p in similar_paths if p["embedding"] is not None], axis=0)

        # Exploit: cosine similarity to centroid (closer => higher)
        a_emb = a_emb / np.linalg.norm(a_emb)
        centroid = centroid / np.linalg.norm(centroid)
        R_exploit = np.dot(a_emb, centroid)

        return float(R_exploit)

    def compute_intrinsic_reward(
        self,
        question: str,
        response: str,
        beta_s: float = 1.0,
        beta_e: float = 1.0
    ) -> float:
        """
        vLLM-based intrinsic reward similar to `ChromaRetriever` style.
        Uses cosine similarity of the response embedding to the centroid of
        retrieved reasoning path embeddings for the given question.
        """
        # Embed the response (action)
        a_emb = np.array(self._embed_one(response), dtype=np.float32)

        # Retrieve similar paths by the question and embed their documents
        similar_paths = self.retrieve_similar_reasoning_paths(question)
        docs = [p["document"] for p in similar_paths if isinstance(p.get("document"), str)]
        if not docs:
            return 0.0
        doc_embs = np.array(self._embed_many(docs), dtype=np.float32)
        if doc_embs.size == 0:
            return 0.0
        centroid = np.mean(doc_embs, axis=0)

        # Exploit: cosine similarity to centroid (closer => higher)
        a_emb = a_emb / (np.linalg.norm(a_emb) + 1e-8)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        R_exploit = (np.dot(a_emb, centroid) + 1) / 2

        # Exploration placeholder (no failed paths memory yet)
        R_explore = 1.0

        R_mem = beta_s * float(R_exploit) + beta_e * float(R_explore)
        return float(R_mem)

    @staticmethod
    def _make_thought_id(dataset_name: str, db_id: str, thought: str) -> str:
        h = hashlib.sha1()
        h.update(dataset_name.encode('utf-8'))
        h.update(b'\x00')
        h.update(db_id.encode('utf-8'))
        h.update(b'\x00')
        h.update(thought.encode('utf-8'))
        # Use the same naming prefix style as save_reasoning_paths_to_chroma.py ("reasoning_...")
        return f"reasoning_{db_id}_{h.hexdigest()[:16]}"

    def save_thought(self, *, dataset_name: str, db_id: str, thought: str, model_name: str | None = None) -> str | None:
        """Persist a correct thought into Chroma using this class's vLLM embeddings.

        Returns the added id, or None if skipped (already exists or invalid).
        """
        if not isinstance(thought, str):
            return None
        thought = thought.strip()
        if not thought:
            return None

        # Compute embedding first (reuse vLLM + tokenizer already loaded)
        embedding = self._embed_one(thought)

        # Check top-1 similarity; only save if cosine similarity < 0.9
        res = self.reasoning_collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["embeddings", "documents", "metadatas", "distances"],
        ) or {}
        ids0 = (res.get("ids") or [[]])[0]
        embs0 = (res.get("embeddings") or [[]])[0]
        if (ids0 is not None and hasattr(ids0, "__len__") and len(ids0) > 0) and (embs0 is not None and hasattr(embs0, "__len__") and len(embs0) > 0):
            top_emb = np.array(embs0[0], dtype=np.float32)
            a = np.array(embedding, dtype=np.float32)
            na = np.linalg.norm(a)
            nb = np.linalg.norm(top_emb)
            if na > 0 and nb > 0:
                cos = float(np.dot(a, top_emb) / (na * nb))
                if cos >= 0.9:
                    return None  # similar enough; skip saving

        uid = self._make_thought_id(dataset_name, db_id, thought)

        # Metadata format aligned with save_reasoning_paths_to_chroma.py
        metadata: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "model_name": model_name or "unknown",
            "schema_prompt": "",  # unknown in this context; avoid None for Chroma
            "reasoning_path": thought,
            "created_at": datetime.utcnow().isoformat(),
            # keep db_id to support cross-db filters in retrieval
            "db_id": db_id,
        }

        self.reasoning_collection.add(
            embeddings=[embedding],  # type: ignore
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

    rewards: List[float] = []
    for i in range(len(questions)):
        r = comp.compute_intrinsic_reward(questions[i], responses[i], beta_s=beta_s, beta_e=beta_e)
        rewards.append(r)
    return rewards
