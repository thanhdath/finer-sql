"""
MCTS Configuration for SQL Writer
Configurable parameters for MCTS sampling in MCTSGRPOTrainer.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MCTSConfig:
    """Configuration for MCTS sampling parameters."""
    
    # MCTS tree search parameters
    num_simulations: int = 4
    c_puct: float = 1.5
    topk_expand: int = 8
    temperature: float = 1.0
    
    # Generation parameters
    # node_max_tokens: number of tokens generated per node expansion
    node_max_tokens: int = 256
    # rollout_max_tokens: max total tokens added along a path (terminal condition)
    rollout_max_tokens: int = 2048
    # # Deprecated: kept for backward compatibility
    max_new_tokens: int = 1024
    depth_limit: int = 16

    top_k: int = 8
    
    
    def __post_init__(self):
        return


# Default MCTS configuration
DEFAULT_MCTS_CONFIG = MCTSConfig()



def get_mcts_config(config_name: str = "default") -> MCTSConfig:
    """
    Get MCTS configuration by name.
    
    Args:
        config_name: Name of the configuration ("default", "high_quality", "fast", "sql")
        
    Returns:
        MCTSConfig instance
    """
    configs = {
        "default": DEFAULT_MCTS_CONFIG,
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config name: {config_name}. Available: {list(configs.keys())}")
    
    return configs[config_name]


def create_custom_mcts_config(**kwargs) -> MCTSConfig:
    """
    Create a custom MCTS configuration.
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        MCTSConfig instance with cuddstom parameters
    """
    return MCTSConfig(**kwargs)
