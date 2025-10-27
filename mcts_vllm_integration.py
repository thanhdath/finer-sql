"""
MCTS-VLLM Integration Module
Provides MCTS sampling integration with VLLM server for MCTSGRPOTrainer.
"""

import torch
import numpy as np
from typing import List, Optional
from transformers import PreTrainedTokenizerBase
from mcts_gen import generate_mcts_candidates
from mcts_config import MCTSConfig, get_mcts_config
from mcts_dpo_vllm import integrate_mcts_dpo_with_vllm

class CompletionObject:
    """Simple completion object to match VLLM format."""
    def __init__(self, token_ids: List[int]):
        self.token_ids = token_ids


class MCTSVLLMClient:
    """
    VLLM client wrapper for MCTS sampling.
    Provides logits extraction from VLLM server for MCTS tree search.
    """
    
    def __init__(self, vllm_client, tokenizer: PreTrainedTokenizerBase):
        self.vllm_client = vllm_client
        self.tokenizer = tokenizer
        self._logits_cache = {}  # Cache for logits to avoid redundant calls
    
    def get_next_logits(self, input_ids: List[int]) -> np.ndarray:
        """
        Get logits for the next token given input_ids.
        This is the core function needed by MCTS sampling.
        """
        # Convert to tensor if needed
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # Create cache key
        cache_key = tuple(input_ids.tolist())
        if cache_key in self._logits_cache:
            return self._logits_cache[cache_key]
        
        # Get logits from VLLM
        try:
            # Decode input_ids to text for VLLM
            input_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
            
            # Use VLLM to get logits (this is a simplified approach)
            # In practice, you might need to modify VLLM client to expose logits
            # For now, we'll use a placeholder that calls the VLLM client
            logits = self._get_logits_from_vllm(input_text)
            
            # Cache the result
            self._logits_cache[cache_key] = logits
            return logits
            
        except Exception as e:
            print(f"Error getting logits from VLLM: {e}")
            # Return uniform logits as fallback
            vocab_size = self.tokenizer.vocab_size
            return np.zeros(vocab_size, dtype=np.float32)
    
    def _get_logits_from_vllm(self, input_text: str) -> np.ndarray:
        """
        Extract logits from VLLM server.
        This implementation uses VLLM's internal model to get logits.
        """
        try:
            # Use VLLM's internal model to get logits
            # This requires access to the underlying model
            if hasattr(self.vllm_client, 'llm') and hasattr(self.vllm_client.llm, 'llm_engine'):
                # Access the underlying model
                model = self.vllm_client.llm.llm_engine.model_executor.driver_worker.model_runner.model
                
                # Tokenize input
                input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
                if hasattr(input_ids, 'cuda'):
                    input_ids = input_ids.cuda()
                
                # Get logits from the model
                with torch.no_grad():
                    outputs = model(input_ids)
                    logits = outputs.logits[0, -1, :].cpu().numpy()  # Get last token logits
                
                return logits.astype(np.float32)
            else:
                # Fallback: use VLLM client to generate and extract logits
                # This is a workaround - ideally you'd want direct logits access
                vocab_size = self.tokenizer.vocab_size
                
                # Generate a single token to get logits
                from vllm import SamplingParams
                sampling_params = SamplingParams(
                    temperature=0.0,  # Deterministic
                    max_tokens=1,
                    logprobs=vocab_size,  # Get all logprobs
                )
                
                outputs = self.vllm_client.generate([input_text], sampling_params=sampling_params)
                if outputs and outputs[0].outputs:
                    # Extract logits from logprobs
                    logprobs = outputs[0].outputs[0].logprobs[0]
                    logits = np.zeros(vocab_size, dtype=np.float32)
                    for token_id, logprob in logprobs.items():
                        logits[token_id] = logprob
                    return logits
                else:
                    # Fallback to random logits
                    return np.random.randn(vocab_size).astype(np.float32)
                    
        except Exception as e:
            print(f"Error extracting logits from VLLM: {e}")
            # Fallback to random logits
            vocab_size = self.tokenizer.vocab_size
            return np.random.randn(vocab_size).astype(np.float32)
    
    def clear_cache(self):
        """Clear the logits cache."""
        self._logits_cache.clear()


class MCTSVLLMGenerator:
    """
    MCTS-based generation using VLLM server.
    Integrates MCTS sampling with VLLM for efficient generation.
    """
    
    def __init__(
        self,
        vllm_client,
        tokenizer: PreTrainedTokenizerBase,
        mcts_config: Optional[MCTSConfig] = None,
        **kwargs
    ):
        self.vllm_client = vllm_client
        self.tokenizer = tokenizer
        
        # Use provided config or default
        self.mcts_config = mcts_config or get_mcts_config("default")
        
        # Extract parameters from config
        self.max_new_tokens = self.mcts_config.max_new_tokens
        self.num_simulations = self.mcts_config.num_simulations
        self.c_puct = self.mcts_config.c_puct
        self.topk_expand = self.mcts_config.topk_expand
        self.temperature = self.mcts_config.temperature
        
        # Create MCTS-VLLM client
        self.mcts_client = MCTSVLLMClient(vllm_client, tokenizer)
    
    def generate_mcts_candidates(self, prompts: List[str]) -> List[List[str]]:
        """
        Generate MCTS candidates for a batch of prompts.
        
        Args:
            prompts: List of prompt strings
            
        Returns:
            List of lists, where each inner list contains num_generations candidates for each prompt
        """
        all_candidates = []
        
        for prompt in prompts:
            # Generate MCTS candidates for this prompt
            candidates = generate_mcts_candidates(
                prompt=prompt,
                num_generations=self.num_generations,
                tokenizer=self.tokenizer,
                get_next_logits=self.mcts_client.get_next_logits,
                max_new_tokens=self.max_new_tokens,
                num_simulations=self.num_simulations,
                c_puct=self.c_puct,
                topk_expand=self.topk_expand,
                temperature=self.temperature,
            )
            all_candidates.append(candidates)
        
        return all_candidates
    
    def clear_cache(self):
        """Clear the MCTS client cache."""
        self.mcts_client.clear_cache()


def create_mcts_vllm_generator(
    vllm_client,
    tokenizer: PreTrainedTokenizerBase,
    **kwargs
) -> MCTSVLLMGenerator:
    """
    Create an MCTS-VLLM generator with default parameters.
    
    Args:
        vllm_client: VLLM client instance
        tokenizer: Tokenizer for the model
        **kwargs: Additional parameters for MCTSVLLMGenerator
        
    Returns:
        Configured MCTSVLLMGenerator instance
    """
    return MCTSVLLMGenerator(
        vllm_client=vllm_client,
        tokenizer=tokenizer,
        **kwargs
    )


# Integration functions for MCTSGRPOTrainer
def integrate_mcts_with_vllm_generation(
    trainer,
    prompts_text: List[str],
    num_generations: int,
    **generation_kwargs
) -> List[List[int]]:
    """
    Integrate MCTS-DPO algorithm with VLLM generation in MCTSGRPOTrainer.
    
    Args:
        trainer: MCTSGRPOTrainer instance
        prompts_text: List of prompt texts
        num_generations: Number of generations per prompt
        **generation_kwargs: Additional generation parameters
        
    Returns:
        List of completion objects (same format as VLLM generate)
    """
    # Use MCTS-DPO algorithm with VLLM
    return integrate_mcts_dpo_with_vllm(
        trainer=trainer,
        prompts_text=prompts_text,
        num_generations=num_generations,
        **generation_kwargs
    )


def create_mcts_vllm_logits_extractor(vllm_client, tokenizer):
    """
    Create a logits extractor function for MCTS sampling.
    
    Args:
        vllm_client: VLLM client instance
        tokenizer: Tokenizer for the model
        
    Returns:
        Function that takes input_ids and returns logits
    """
    mcts_client = MCTSVLLMClient(vllm_client, tokenizer)
    return mcts_client.get_next_logits
