#!/usr/bin/env python3
"""
Script to create supervised fine-tuning (SFT) dataset for text-to-SQL models.
Creates a dataset with sample_id and messages format suitable for LLM fine-tuning.
"""

import json
import os
import argparse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from pymongo import MongoClient
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict


@dataclass
class Message:
    """Represents a single message in a conversation."""
    role: str  # "system", "user", or "assistant"
    content: str


@dataclass
class SFTExample:
    """Represents a single SFT training example."""
    prompt_id: str  # Unique identifier for this specific prompt/response pair
    sample_id: str  # Original sample ID (can be duplicated across different models)
    messages: List[Message]
    model_name: Optional[str] = None  # Which model generated this response
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for HuggingFace dataset."""
        return {
            "prompt_id": self.prompt_id,
            "sample_id": self.sample_id,
            "messages": [
                {"role": msg.role, "content": msg.content} 
                for msg in self.messages
            ],
            "model_name": self.model_name
        }


class SFTDataGenerator:
    """Generator for supervised fine-tuning datasets."""
    
    def __init__(self, output_dir: str = "/home/datht/mats/data/sft/sft_text2sql"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_from_mongodb(
        self, 
        mongo_uri: str = "mongodb://localhost:27017",
        database: str = "mats",
        collection: str = "llm_pool_bird",
        limit: Optional[int] = None,
        model_filter: Optional[List[str]] = None,
        main_models_only: bool = False
    ) -> List[SFTExample]:
        """Load SFT examples from MongoDB collection, including all model responses."""
        
        print(f"Connecting to MongoDB: {mongo_uri}")
        client = MongoClient(mongo_uri)
        db = client[database]
        coll = db[collection]
        
        # Build query
        query = {"ok": True, "response": {"$exists": True, "$ne": None}}
        if model_filter:
            query["model_name"] = {"$in": model_filter}
        elif main_models_only:
            # Find models with > 1000 samples
            pipeline = [
                {"$match": {"ok": True, "response": {"$exists": True, "$ne": None}}},
                {"$group": {"_id": "$model_name", "count": {"$sum": 1}}},
                {"$match": {"count": {"$gt": 1000}}},
                {"$sort": {"count": -1}}
            ]
            results = list(coll.aggregate(pipeline))
            main_model_names = [result["_id"] for result in results]
            print(f"Found {len(main_model_names)} models with >1000 samples:")
            for i, model_name in enumerate(main_model_names, 1):
                count = next((r["count"] for r in results if r["_id"] == model_name), 0)
                print(f"  {i}. {model_name} ({count:,} samples)")
            if not main_model_names:
                print("Warning: No models found with >1000 samples, using all models")
                main_model_names = coll.distinct("model_name")
            query["model_name"] = {"$in": main_model_names}
        
        print(f"Querying collection: {database}.{collection}")
        print(f"Query: {query}")
        
        # Get all documents (not just unique sample_ids)
        examples = []
        prompt_id_counter = 0
        
        if limit:
            # When using limit, sample proportionally from each model
            print(f"Sampling {limit} documents proportionally from each model...")
            
            # Get model counts
            pipeline = [
                {"$match": query},
                {"$group": {"_id": "$model_name", "count": {"$sum": 1}}}
            ]
            model_counts = list(coll.aggregate(pipeline))
            total_docs = sum(result["count"] for result in model_counts)
            
            # Calculate samples per model
            samples_per_model = {}
            for result in model_counts:
                model_name = result["_id"]
                model_count = result["count"]
                # Sample proportionally, but ensure at least 1 from each model
                samples = max(1, int((model_count / total_docs) * limit))
                samples_per_model[model_name] = min(samples, model_count)
            
            print(f"Sampling plan: {samples_per_model}")
            
            # Sample from each model
            for model_name, sample_count in samples_per_model.items():
                model_query = {**query, "model_name": model_name}
                model_cursor = coll.find(model_query).limit(sample_count)
                
                for doc in model_cursor:
                    sample_id = doc.get("sample_id")
                    messages_data = doc.get("messages", [])
                    response = doc.get("response", "")
                    
                    if not messages_data:
                        print(f"Warning: No messages found for sample_id {sample_id}, model {model_name}")
                        continue
                    
                    if not response:
                        print(f"Warning: No response found for sample_id {sample_id}, model {model_name}")
                        continue
                    
                    # Convert to Message objects
                    messages = []
                    for msg_data in messages_data:
                        messages.append(Message(
                            role=msg_data.get("role", "user"),
                            content=msg_data.get("content", "")
                        ))
                    
                    # Add assistant response
                    messages.append(Message(
                        role="assistant",
                        content=response
                    ))
                    
                    # Create unique prompt_id
                    prompt_id = f"prompt_{prompt_id_counter:06d}"
                    prompt_id_counter += 1
                    
                    # Create SFT example with all fields
                    example = SFTExample(
                        prompt_id=prompt_id,
                        sample_id=str(sample_id),
                        messages=messages,
                        model_name=model_name
                    )
                    examples.append(example)
        else:
            # No limit - get all documents
            print("Processing all model responses...")
            cursor = coll.find(query)
            for doc in cursor:
                sample_id = doc.get("sample_id")
                model_name = doc.get("model_name")
                messages_data = doc.get("messages", [])
                response = doc.get("response", "")
                
                if not messages_data:
                    print(f"Warning: No messages found for sample_id {sample_id}, model {model_name}")
                    continue
                
                if not response:
                    print(f"Warning: No response found for sample_id {sample_id}, model {model_name}")
                    continue
                
                # Convert to Message objects
                messages = []
                for msg_data in messages_data:
                    messages.append(Message(
                        role=msg_data.get("role", "user"),
                        content=msg_data.get("content", "")
                    ))
                
                # Add assistant response
                messages.append(Message(
                    role="assistant",
                    content=response
                ))
                
                # Create unique prompt_id
                prompt_id = f"prompt_{prompt_id_counter:06d}"
                prompt_id_counter += 1
                
                # Create SFT example with all fields
                example = SFTExample(
                    prompt_id=prompt_id,
                    sample_id=str(sample_id),
                    messages=messages,
                    model_name=model_name
                )
                examples.append(example)
        
        print(f"Loaded {len(examples)} examples from MongoDB")
        print(f"Unique sample_ids: {len(set(ex.sample_id for ex in examples))}")
        print(f"Models represented: {len(set(ex.model_name for ex in examples if ex.model_name))}")
        return examples
    
    
    def save_to_huggingface(self, examples: List[SFTExample], split: str = "train") -> None:
        """Save examples to HuggingFace dataset format."""
        if Dataset is None:
            print("Error: HuggingFace datasets library not available")
            return
        
        # Convert examples to dictionary format
        data_list = []
        
        for example in examples:
            sample = {
                "prompt_id": example.prompt_id,
                "sample_id": example.sample_id,
                "messages": [
                    {"role": msg.role, "content": msg.content} 
                    for msg in example.messages
                ],
                "model_name": example.model_name
            }
            data_list.append(sample)
        
        # Create dataset
        dataset = Dataset.from_list(data_list)
        test_dataset = Dataset.from_list(data_list[:10])  # First 10 samples for test
        
        # Create dataset dict
        dataset_dict = DatasetDict({
            "train_sft": dataset,
            "test_sft": test_dataset
        })
        
        # Save to disk
        dataset_dict.save_to_disk(str(self.output_dir))
        dataset_dict.push_to_hub("thanhdath/sft_text2sql", private=False)
        print(f"Saved {len(examples)} examples to {self.output_dir}")
        print(f"Dataset info: {dataset_dict}")
    
    def save_to_json(self, examples: List[SFTExample], filename: str = "sft_data.json") -> None:
        """Save examples to JSON file."""
        data = [example.to_dict() for example in examples]
        
        output_file = self.output_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(examples)} examples to {output_file}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create SFT dataset from MongoDB")
    parser.add_argument("--mongo_uri", default="mongodb://localhost:27017", 
                       help="MongoDB connection URI")
    parser.add_argument("--database", default="mats", 
                       help="MongoDB database name")
    parser.add_argument("--collection", default="llm_pool_bird", 
                       help="MongoDB collection name")
    parser.add_argument("--output_dir", default="/home/datht/mats/data/sft/sft_text2sql",
                       help="Output directory for the dataset")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of examples to process")
    parser.add_argument("--model_filter", nargs='+', default=None,
                       help="Filter by specific model names (space-separated)")
    parser.add_argument("--main_models_only", default=False, action="store_true",
                       help="Use only models with >1000 samples (automatically detected)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    return parser.parse_args()


def main():
    """Main function to create SFT dataset."""
    args = parse_args()
    
    # Load environment variables
    try:
        load_dotenv("/home/datht/mats/.env")
    except Exception:
        pass
    
    print("Creating SFT dataset for text-to-SQL fine-tuning...")
    print(f"MongoDB URI: {args.mongo_uri}")
    print(f"Database: {args.database}")
    print(f"Collection: {args.collection}")
    print(f"Output directory: {args.output_dir}")
    
    # Initialize generator
    generator = SFTDataGenerator(args.output_dir)
    
    # Load data from MongoDB
    print("Loading data from MongoDB...")
    
    # Handle main models filter
    model_filter = args.model_filter
    if args.main_models_only:
        model_filter = None  # We'll handle this in the load function
        print("Filtering for models with >1000 samples (automatically detected)")
    
    examples = generator.load_from_mongodb(
        mongo_uri=args.mongo_uri,
        database=args.database,
        collection=args.collection,
        limit=args.limit,
        model_filter=model_filter,
        main_models_only=args.main_models_only
    )
    
    if not examples:
        print("No examples found in MongoDB!")
        return
    
    print(f"Processing {len(examples)} examples...")
    
    # Save to HuggingFace format
    print("Saving to HuggingFace dataset format...")
    generator.save_to_huggingface(examples, split="train")
    
    # Also save as JSON for backup
    print("Saving backup JSON file...")
    generator.save_to_json(examples, "sft_data_backup.json")
    
    print("SFT dataset creation completed!")
    print(f"Dataset saved to: {generator.output_dir}")


if __name__ == "__main__":
    main()
