#!/usr/bin/env python3
"""
Save correct reasoning paths to ChromaDB for memory-based retrieval.
Each entry contains: db_id, schema_prompt, reasoning_path (from <think> tags)
"""

from __future__ import annotations
import argparse
import os
import torch
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pymongo import MongoClient
from tqdm import tqdm
from vllm import LLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
import chromadb




def parse_args():
    """Configure command-line arguments"""
    p = argparse.ArgumentParser()
    p.add_argument("--mongo-uri", default=os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    p.add_argument("--database", default="mats")
    p.add_argument("--collection", default="llm_pool_bird")
    p.add_argument("--chroma-path", default="./chroma_db")
    p.add_argument("--collection-name", default="reasoning_paths")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-samples", type=int, default=None, help="Limit number of samples to process")
    p.add_argument("--drop", action="store_true", help="Drop existing ChromaDB collection")
    p.add_argument("--min-reasoning-length", type=int, default=50, help="Minimum length of reasoning path")
    p.add_argument("--only-correct", action="store_true", default=True, help="Only save reasoning paths from correct responses (default: True)")
    p.add_argument("--model-name", default="deepseek-reasoner", help="Filter samples by model name (default: deepseek-reasoner)")
    return p.parse_args()


def extract_reasoning_path(text: str) -> Optional[str]:
    """Extract reasoning path from <think> tags."""
    # Find <think> and </think> tags
    start_tag = '<think>'
    end_tag = '</think>'
    
    start_idx = text.find(start_tag)
    end_idx = text.find(end_tag)
    
    if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
        return None
    
    reasoning = text[start_idx + len(start_tag):end_idx].strip()
    return reasoning if reasoning else None


def extract_schema_prompt(messages: List[Dict[str, str]]) -> Optional[str]:
    """Extract schema prompt from messages."""
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content", "")
            # Look for "Database Schema:" section
            if "Database Schema:" in content:
                return content
    return None


def is_correct_response(sample: Dict[str, Any]) -> bool:
    """Check if the response is correct using the is_execution_correct field from MongoDB."""
    is_correct = sample.get("is_execution_correct")
    return is_correct is True


def batch_items(lst, size):
    """Yield successive size-sized chunks from lst."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


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
    """Embed texts using vLLM with truncation."""
    # Truncate texts to max_length
    truncated_texts = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > max_length:
            ids = ids[:max_length]
        truncated_text = tokenizer.decode(ids, skip_special_tokens=True)
        truncated_texts.append(truncated_text)
    
    # Get embeddings
    outputs = llm.embed(truncated_texts)
    embeddings = [extract_embedding(output) for output in outputs]
    return embeddings


def main():
    args = parse_args()
    
    # Load environment variables
    load_dotenv("../.env")
    
    # Initialize MongoDB client
    print(f"Connecting to MongoDB: {args.mongo_uri}")
    mongo = MongoClient(args.mongo_uri)
    db = mongo[args.database]
    collection = db[args.collection]
    
    # Initialize embedding model with vLLM
    print("Loading Qwen3-Embedding model via vLLM...")
    model_path = 'Qwen/Qwen3-Embedding-0.6B'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(model=model_path, task="embed", trust_remote_code=True, max_model_len=8192, gpu_memory_utilization=0.6)
    print(f"Model loaded via vLLM")
    
    # Initialize ChromaDB client
    print(f"Connecting to ChromaDB at {args.chroma_path}")
    chroma_client = chromadb.PersistentClient(path=args.chroma_path)
    
    # Handle collection creation
    if args.drop:
        try:
            chroma_client.delete_collection(args.collection_name)
        except Exception:
            pass
        reasoning_collection = chroma_client.create_collection(args.collection_name)
    else:
        reasoning_collection = chroma_client.get_or_create_collection(args.collection_name)
    
    # Load samples from MongoDB
    print(f"Loading samples from {args.database}.{args.collection}")
    query = {
        "ok": True, 
        "response": {"$exists": True, "$ne": None}
    }
    
    # Filter by correctness if requested
    if args.only_correct:
        query["is_execution_correct"] = True
    
    # Always filter to the specified model (default: deepseek-reasoner)
    if args.model_name:
        query["model_name"] = args.model_name
    
    cursor = collection.find(query)
    if args.max_samples:
        cursor = cursor.limit(args.max_samples)
    
    samples = list(cursor)
    print(f"Loaded {len(samples)} samples from MongoDB")
    
    # Process samples to find reasoning paths
    if args.only_correct:
        print("Processing samples to find correct reasoning paths...")
    else:
        print("Processing samples to find all reasoning paths...")
    reasoning_entries = []
    
    for sample in tqdm(samples, desc="Processing samples"):
        sample_id = sample.get("sample_id")
        response = sample.get("response", "")
        messages = sample.get("messages", [])
        
        
        # Extract reasoning path
        reasoning_path = extract_reasoning_path(response)
        if not reasoning_path or len(reasoning_path) < args.min_reasoning_length:
            continue
        
        # Extract schema prompt
        schema_prompt = extract_schema_prompt(messages)
        if not schema_prompt:
            continue
        
        # Check if response is correct (only if filtering for correct ones)
        if args.only_correct:
            if not is_correct_response(sample):
                continue
        
        # Create entry for ChromaDB
        # Use a unique ID that includes model name to avoid duplicates
        unique_id = f"reasoning_{sample_id}_{sample.get('model_name', 'unknown')}"
        entry = {
            "id": unique_id,
            "document": reasoning_path,
            "metadata": {
                "db_id": sample["db_id"],
                "sample_id": sample_id,
                "dataset_name": sample["dataset_name"],
                "model_name": sample["model_name"],
                "schema_prompt": schema_prompt,
                "reasoning_path": reasoning_path,
                "created_at": datetime.utcnow().isoformat()
            }
        }
        reasoning_entries.append(entry)
        
    
    if args.only_correct:
        print(f"Found {len(reasoning_entries)} correct reasoning paths")
    else:
        print(f"Found {len(reasoning_entries)} reasoning paths")
    
    if not reasoning_entries:
        print("No reasoning paths found. Exiting.")
        return
    
    # Embed and store in ChromaDB
    print("Embedding and storing reasoning paths in ChromaDB...")
    
    for batch in tqdm(batch_items(reasoning_entries, args.batch_size), 
                      desc="Embedding reasoning paths", 
                      total=len(reasoning_entries) // args.batch_size + 1):
        
        batch_ids = [entry["id"] for entry in batch]
        
        # Check which entries already exist
        existing = reasoning_collection.get(ids=batch_ids)
        existing_ids = set(existing["ids"]) if existing and "ids" in existing else set()
        
        # Filter out existing entries
        filtered_batch = [entry for entry in batch if entry["id"] not in existing_ids]
        if not filtered_batch:
            continue
        
        # Extract documents and metadata
        documents = [entry["document"] for entry in filtered_batch]
        metadatas = [entry["metadata"] for entry in filtered_batch]
        ids = [entry["id"] for entry in filtered_batch]
        
        try:
            # Generate embeddings using vLLM
            embeddings = embed_texts_vllm(documents, llm, tokenizer)
            
            # Add to ChromaDB
            reasoning_collection.add(
                embeddings=embeddings,  # type: ignore
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            print(f"Error processing batch: {e}")
    
    # Final summary
    print("\n[✓] Reasoning Paths Saved to ChromaDB")
    if args.only_correct:
        print(f"Total correct reasoning paths: {len(reasoning_entries)}")
    else:
        print(f"Total reasoning paths: {len(reasoning_entries)}")
    print(f"Collection name: {args.collection_name}")
    print(f"ChromaDB path: {args.chroma_path}")
    
    # Cleanup
    mongo.close()


if __name__ == "__main__":
    main()
