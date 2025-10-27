#!/usr/bin/env python3
"""
Intrinsic reward computation for SQL writer memory.
Computes intrinsic rewards for LLM completions based on reasoning path similarity.
"""

from __future__ import annotations
import os
import torch  # type: ignore
import numpy as np  # type: ignore
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer  # type: ignore
import chromadb  # type: ignore
from chromadb.api.types import QueryResult  # type: ignore
from datetime import datetime
import uuid
import hashlib

# vLLM-based embedding utilities (used by MemoryUpdater)
from vllm import LLM  # type: ignore
from transformers.models.auto.tokenization_auto import AutoTokenizer  # type: ignore


class SQLWriterIntrinsicRewardComputer:
    """
    A class to compute intrinsic rewards for SQL writer completions
    based on reasoning path similarity from ChromaDB memory.
    """

    def __init__(self, chroma_path: str = "./chroma_db"):
        """
        Initialize the intrinsic reward computer.
        
        Args:
            chroma_path (str): Path to ChromaDB storage
        """
        print("[✓] Loading embedding model...")
        self.model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', truncate_dim=2048)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        print(f"[✓] Connecting to ChromaDB at {chroma_path}...")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        
        # Try to get reasoning paths collection
        try:
            self.reasoning_collection = self.chroma_client.get_collection("reasoning_paths")
        except Exception:
            print("Warning: Could not find 'reasoning_paths' collection. Make sure to run save_reasoning_paths_to_chroma.py first.")
            self.reasoning_collection = None

    def compute_intrinsic_reward(
        self,
        question: str,
        completion: str,
        beta_s: float = 1.0,
        beta_e: float = 1.0,
        top_k: int = 20
    ) -> float:
        """
        Compute the intrinsic reward for a given (question, completion) pair.

        Args:
            question (str): Input question/prompt.
            completion (str): LLM-generated completion.
            beta_s (float): Weight for exploitation reward (similarity to good examples).
            beta_e (float): Weight for exploration reward (dissimilarity to bad examples).
            top_k (int): Number of similar reasoning paths to retrieve.

        Returns:
            float: Intrinsic reward.
        """
        if self.reasoning_collection is None:
            print("Warning: No reasoning collection available. Returning 0.0")
            return 0.0

        # Step 1: Embed the completion
        completion_emb = self.model.encode([completion], convert_to_numpy=True, prompt_name="query")[0]

        # Step 2: Retrieve similar reasoning paths
        similar_reasoning_paths = self._retrieve_similar_reasoning_paths(question, top_k)
        
        if not similar_reasoning_paths:
            print("Warning: No similar reasoning paths found. Returning 0.0")
            return 0.0

        # Step 3: Separate reasoning paths by correctness
        correct_paths = [path for path in similar_reasoning_paths 
                        if path.get("metadata", {}).get("is_correct", True)]  # Default to True if not specified
        incorrect_paths = [path for path in similar_reasoning_paths 
                          if not path.get("metadata", {}).get("is_correct", True)]

        # Step 4: Compute rewards
        R_exploit = self._compute_exploit_reward(completion_emb, correct_paths)
        R_explore = self._compute_explore_reward(completion_emb, incorrect_paths)

        # Step 5: Combine rewards
        R_mem = beta_s * R_exploit + beta_e * R_explore

        return R_mem

    def batch_compute_intrinsic_reward(
        self,
        questions: List[str],
        completions: List[str],
        beta_s: float = 1.0,
        beta_e: float = 1.0,
        top_k: int = 20
    ) -> List[float]:
        """
        Batch compute intrinsic rewards for a list of (question, completion) pairs.

        Args:
            questions (List[str]): List of input questions.
            completions (List[str]): List of LLM-generated completions.
            beta_s (float): Weight for exploitation reward.
            beta_e (float): Weight for exploration reward.
            top_k (int): Number of similar reasoning paths to retrieve.

        Returns:
            List[float]: List of intrinsic rewards.
        """
        if len(questions) != len(completions):
            raise ValueError("Questions and completions must have the same length")

        if not questions:
            return []

        if self.reasoning_collection is None:
            print("Warning: No reasoning collection available. Returning zeros")
            return [0.0] * len(questions)

        # Step 1: Batch encode completions
        completion_embeddings = self.model.encode(completions, convert_to_numpy=True, prompt_name="query")

        R_mem_list = []

        for i in range(len(questions)):
            completion_emb = completion_embeddings[i]
            
            # Retrieve similar reasoning paths for this question
            similar_reasoning_paths = self._retrieve_similar_reasoning_paths(questions[i], top_k)
            
            if not similar_reasoning_paths:
                R_mem_list.append(0.0)
                continue

            # Separate by correctness
            correct_paths = [path for path in similar_reasoning_paths 
                            if path.get("metadata", {}).get("is_correct", True)]
            incorrect_paths = [path for path in similar_reasoning_paths 
                              if not path.get("metadata", {}).get("is_correct", True)]

            # Compute rewards
            R_exploit = self._compute_exploit_reward(completion_emb, correct_paths)
            R_explore = self._compute_explore_reward(completion_emb, incorrect_paths)

            R_mem = beta_s * R_exploit + beta_e * R_explore
            R_mem_list.append(R_mem)

        return R_mem_list

    def _retrieve_similar_reasoning_paths(self, question: str, top_k: int) -> List[Dict]:
        """
        Retrieve similar reasoning paths for a given question.
        
        Args:
            question (str): Input question.
            top_k (int): Number of similar paths to retrieve.
            
        Returns:
            List[Dict]: List of similar reasoning paths with metadata.
        """
        if self.reasoning_collection is None:
            return []

        # Embed the question
        question_embedding = self.model.encode([question], convert_to_numpy=True, prompt_name="query")[0]

        # Query ChromaDB for similar reasoning paths
        results = self.reasoning_collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=top_k
        )

        # Format results
        similar_paths = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                similar_paths.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None
                })

        return similar_paths

    def _compute_exploit_reward(self, completion_emb: np.ndarray, correct_paths: List[Dict]) -> float:
        """
        Compute exploitation reward based on similarity to correct reasoning paths.
        
        Args:
            completion_emb (np.ndarray): Embedding of the completion.
            correct_paths (List[Dict]): List of correct reasoning paths.
            
        Returns:
            float: Exploitation reward.
        """
        if not correct_paths:
            return 0.0

        # Get embeddings of correct reasoning paths
        correct_embeddings = []
        for path in correct_paths:
            if "embedding" in path:
                correct_embeddings.append(path["embedding"])
            else:
                # If embedding not stored, compute it
                reasoning_text = path.get("document", "")
                if reasoning_text:
                    emb = self.model.encode([reasoning_text], convert_to_numpy=True)[0]
                    correct_embeddings.append(emb)

        if not correct_embeddings:
            return 0.0

        # Compute centroid of correct reasoning paths
        centroid = np.mean(correct_embeddings, axis=0)
        
        # Exploitation reward: negative distance to centroid (closer = higher reward)
        return -np.linalg.norm(completion_emb - centroid)

    def _compute_explore_reward(self, completion_emb: np.ndarray, incorrect_paths: List[Dict]) -> float:
        """
        Compute exploration reward based on dissimilarity to incorrect reasoning paths.
        
        Args:
            completion_emb (np.ndarray): Embedding of the completion.
            incorrect_paths (List[Dict]): List of incorrect reasoning paths.
            
        Returns:
            float: Exploration reward.
        """
        if not incorrect_paths:
            return 1.0  # Maximum exploration reward if no incorrect paths

        # Get embeddings of incorrect reasoning paths
        incorrect_embeddings = []
        for path in incorrect_paths:
            if "embedding" in path:
                incorrect_embeddings.append(path["embedding"])
            else:
                # If embedding not stored, compute it
                reasoning_text = path.get("document", "")
                if reasoning_text:
                    emb = self.model.encode([reasoning_text], convert_to_numpy=True)[0]
                    incorrect_embeddings.append(emb)

        if not incorrect_embeddings:
            return 1.0

        # Compute similarities to incorrect paths
        completion_norm = np.linalg.norm(completion_emb)
        similarities = []
        
        for emb in incorrect_embeddings:
            emb_norm = np.linalg.norm(emb)
            if completion_norm > 0 and emb_norm > 0:
                similarity = np.dot(completion_emb, emb) / (completion_norm * emb_norm)
                similarities.append(similarity)

        if not similarities:
            return 1.0

        # Exploration reward: 1 - max similarity (more different = higher reward)
        return 1.0 - max(similarities)


def compute_intrinsic_reward_for_completion(
    question: str,
    completion: str,
    chroma_path: str = "./chroma_db",
    beta_s: float = 1.0,
    beta_e: float = 1.0,
    top_k: int = 20
) -> float:
    """
    Convenience function to compute intrinsic reward for a single completion.
    
    Args:
        question (str): Input question/prompt.
        completion (str): LLM-generated completion.
        chroma_path (str): Path to ChromaDB storage.
        beta_s (float): Weight for exploitation reward.
        beta_e (float): Weight for exploration reward.
        top_k (int): Number of similar reasoning paths to retrieve.
        
    Returns:
        float: Intrinsic reward.
    """
    computer = SQLWriterIntrinsicRewardComputer(chroma_path)
    return computer.compute_intrinsic_reward(question, completion, beta_s, beta_e, top_k)


def batch_compute_intrinsic_reward_for_completions(
    questions: List[str],
    completions: List[str],
    chroma_path: str = "./chroma_db",
    beta_s: float = 1.0,
    beta_e: float = 1.0,
    top_k: int = 20
) -> List[float]:
    """
    Convenience function to compute intrinsic rewards for multiple completions.
    
    Args:
        questions (List[str]): List of input questions.
        completions (List[str]): List of LLM-generated completions.
        chroma_path (str): Path to ChromaDB storage.
        beta_s (float): Weight for exploitation reward.
        beta_e (float): Weight for exploration reward.
        top_k (int): Number of similar reasoning paths to retrieve.
        
    Returns:
        List[float]: List of intrinsic rewards.
    """
    computer = SQLWriterIntrinsicRewardComputer(chroma_path)
    return computer.batch_compute_intrinsic_reward(questions, completions, beta_s, beta_e, top_k)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute intrinsic reward for SQL writer completion")
    parser.add_argument("--question", required=True, help="Input question")
    parser.add_argument("--completion", required=True, help="LLM completion")
    parser.add_argument("--chroma-path", default="./chroma_db", help="Path to ChromaDB")
    parser.add_argument("--beta-s", type=float, default=1.0, help="Exploitation weight")
    parser.add_argument("--beta-e", type=float, default=1.0, help="Exploration weight")
    parser.add_argument("--top-k", type=int, default=20, help="Number of similar paths to retrieve")
    
    args = parser.parse_args()
    
    reward = compute_intrinsic_reward_for_completion(
        args.question,
        args.completion,
        args.chroma_path,
        args.beta_s,
        args.beta_e,
        args.top_k
    )
    
    print(f"Intrinsic reward: {reward:.4f}")


class MemoryUpdater:
    """
    Append correct thoughts to ChromaDB using vLLM embeddings
    (same embedding algorithm as save_reasoning_paths_to_chroma.py).
    """

    def __init__(self, chroma_path: str = "./chroma_db", collection_name: str = "reasoning_paths"):
        self.chroma_path = chroma_path
        self.collection_name = collection_name

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(self.collection_name)

        # Initialize vLLM embedding model
        model_path = 'Qwen/Qwen3-Embedding-0.6B'
        if AutoTokenizer is None or LLM is None:
            raise RuntimeError("vLLM and transformers must be available to use MemoryUpdater")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # Keep the same settings as in save_reasoning_paths_to_chroma.py
        self.llm = LLM(model=model_path, task="embed", trust_remote_code=True, max_model_len=8192, gpu_memory_utilization=0.6)

    def _extract_embedding(self, output_obj) -> List[float]:
        try:
            return output_obj.outputs.embedding  # vLLM >= 0.8.5
        except Exception:
            if hasattr(output_obj, "outputs") and isinstance(output_obj.outputs, list) and output_obj.outputs:
                maybe = output_obj.outputs[0]
                if hasattr(maybe, "embedding"):
                    return maybe.embedding
            raise RuntimeError("Unexpected vLLM embed output structure; cannot extract embedding")

    def _embed_texts_vllm(self, texts: List[str], max_length: int = 8192) -> List[List[float]]:
        # Truncate inputs to max_length in tokens
        truncated_texts: List[str] = []
        for text in texts:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) > max_length:
                ids = ids[:max_length]
            truncated_texts.append(self.tokenizer.decode(ids, skip_special_tokens=True))

        outputs = self.llm.embed(truncated_texts)
        return [self._extract_embedding(o) for o in outputs]

    @staticmethod
    def _make_thought_id(dataset_name: str, db_id: str, thought: str) -> str:
        h = hashlib.sha1()
        h.update(dataset_name.encode('utf-8'))
        h.update(b'\x00')
        h.update(db_id.encode('utf-8'))
        h.update(b'\x00')
        h.update(thought.encode('utf-8'))
        return f"mem_{h.hexdigest()}"

    def save_thought(self, *, dataset_name: str, db_id: str, thought: str, model_name: Optional[str] = None) -> Optional[str]:
        """Save a single thought to ChromaDB if not already present. Returns id or None if skipped."""
        if not isinstance(thought, str) or not thought.strip():
            return None
        thought = thought.strip()
        uid = self._make_thought_id(dataset_name, db_id, thought)

        # Skip if already exists
        try:
            existing = self.collection.get(ids=[uid])
            if existing and existing.get("ids") and existing["ids"]:
                return None
        except Exception:
            # If collection.get fails for some reason, proceed to add
            pass

        # Embed and add
        embedding = self._embed_texts_vllm([thought])[0]
        metadata: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "db_id": db_id,
            "reasoning_path": thought,
            "is_correct": True,
            "source": "MemoryUpdater",
            "model_name": model_name or "unknown",
            "created_at": datetime.utcnow().isoformat(),
        }

        self.collection.add(
            embeddings=[embedding],  # type: ignore
            documents=[thought],
            metadatas=[metadata],
            ids=[uid],
        )
        return uid
