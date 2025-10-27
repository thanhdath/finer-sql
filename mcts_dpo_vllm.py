"""
MCTS-DPO Integration with VLLM
Adapts MCTS-DPO algorithm to work with VLLM server for GRPO training.
"""

from openai.types.chat.completion_create_params import CompletionCreateParams
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from transformers import PreTrainedTokenizerBase
from mcts_config import MCTSConfig, get_mcts_config
import torch.nn.functional as F


class VLLMState:
    """State representation for VLLM-based MCTS."""
    def __init__(self, input_ids: List[int], attention_mask: List[int] = None, depth: int = 0):
        self.input_ids = input_ids
        self.attention_mask = attention_mask or [1] * len(input_ids)
        self.is_terminal = False
        self.depth = depth
    
    def __repr__(self):
        return f"VLLMState(input_ids={self.input_ids[:10]}..., depth={self.depth}, terminal={self.is_terminal})"


class VLLMAction:
    """Action representation for VLLM-based MCTS (sequence selection)."""
    def __init__(self, token_id: int, log_prob: float = 0.0, token_ids: Optional[List[int]] = None):
        self.token_id = token_id  # first token (for compatibility)
        self.token_ids = token_ids or [token_id]  # full sequence chosen for this edge
        self.log_prob = log_prob
    
    def __repr__(self):
        return f"VLLMAction(token_id={self.token_id}, len_seq={len(self.token_ids)}, log_prob={self.log_prob:.4f})"


class VLLMMCTSNode:
    """MCTS Node following MCTS-DPO structure."""
    def __init__(self, state: VLLMState, action: Optional[VLLMAction] = None, parent: Optional["VLLMMCTSNode"] = None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children: Dict[int, VLLMMCTSNode] = {}
        
        # MCTS-DPO style attributes
        self.N = 0  # visit count
        self.Q = 0.0  # Q-value
        self.p = 0.0  # prior probability
        self.value = 0.0  # node value
        self.is_terminal = state.is_terminal
        self.depth = state.depth
        
        # Additional attributes for VLLM
        self.log_probs = None
        self.ref_log_probs = None
        self.base_rewards = None
    
    def __repr__(self):
        return f"VLLMMCTSNode(depth={self.depth}, N={self.N}, Q={self.Q:.4f}, terminal={self.is_terminal})"


class VLLMWorldModel:
    """World model for VLLM-based MCTS following MCTS-DPO interface."""
    
    def __init__(self, vllm_client, tokenizer: PreTrainedTokenizerBase, max_length: int = 512):
        self.vllm_client = vllm_client
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.example = None
    
    def init_state(self, prompt: str) -> VLLMState:
        """Initialize state from prompt."""
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        return VLLMState(input_ids=input_ids, depth=0)
    
    def step(self, state: VLLMState, action: VLLMAction, log_probs: Optional[torch.Tensor] = None) -> VLLMState:
        """Take a step by appending the selected sequence tokens."""
        append_ids = action.token_ids if hasattr(action, "token_ids") and action.token_ids else [action.token_id]
        new_input_ids = state.input_ids + append_ids
        new_attention_mask = state.attention_mask + [1] * len(append_ids)
        
        # Check if terminal (max length or EOS token)
        is_terminal = (
            len(new_input_ids) >= self.max_length or 
            (append_ids and append_ids[-1] == self.tokenizer.eos_token_id)
        )
        
        new_state = VLLMState(
            input_ids=new_input_ids,
            attention_mask=new_attention_mask
        )
        new_state.is_terminal = is_terminal
        # Tree depth should reflect number of expansions, not token count
        new_state.depth = state.depth + 1
        
        return new_state
    
    def is_terminal(self, state: VLLMState) -> bool:
        """Check if state is terminal."""
        return state.is_terminal
    
    def update_example(self, example: Any) -> None:
        """Update example for the world model."""
        self.example = example


class VLLMSearchConfig:
    """Search configuration for VLLM-based MCTS following MCTS-DPO interface."""
    
    def __init__(self, vllm_client, tokenizer: PreTrainedTokenizerBase, mcts_config: MCTSConfig):
        self.vllm_client = vllm_client
        self.tokenizer = tokenizer
        self.mcts_config = mcts_config
        self.example = None
    
    def get_actions(self, policy_model, state: VLLMState, add_kl: bool = False) -> List[Tuple[VLLMAction, Tuple[torch.Tensor, torch.Tensor], Dict]]:
        """Get possible actions from a state."""
        # Get logits from VLLM
        input_text = self.tokenizer.decode(state.input_ids, skip_special_tokens=True)
        
        # Use VLLM to get next sequence candidates (up to node_max_tokens)
        print(f"[MCTS DEBUG] Requesting {self.mcts_config.topk_expand} sequences of up to {self.mcts_config.node_max_tokens} tokens from vLLM (len={len(state.input_ids)})")
        outputs = self.vllm_client.generate(
            prompts=[input_text],
            n=self.mcts_config.topk_expand,
            temperature=self.mcts_config.temperature,  # allow sampling diversity
            max_tokens=self.mcts_config.node_max_tokens,
        )
        print(f"[MCTS DEBUG] vLLM returned {len(outputs) if outputs is not None else 'None'} outputs")
        
        actions: List[Tuple[VLLMAction, Tuple[torch.Tensor, torch.Tensor], Dict]] = []
        # Parse multiple sampled candidates for breadth
        # Create a child per completion (full sequence edge)
        added = 0
        
        for seq in outputs:
            token_id = seq[0]
            log_prob = 0.0
            action = VLLMAction(token_id=token_id, log_prob=log_prob, token_ids=seq)
            log_probs = torch.tensor([log_prob], dtype=torch.float32)
            ref_log_probs = torch.tensor([log_prob], dtype=torch.float32)
            actions.append((action, (log_probs, ref_log_probs), {}))
            added += 1
            if added >= self.mcts_config.topk_expand:
                break
        print(f"[MCTS DEBUG] Created {added} child actions (sequence edges)")
        return actions

    
    def get_values(self, policy_model, state: VLLMState, action_batch: List[VLLMAction], 
                   log_probs_batch: List[torch.Tensor], ref_log_probs_batch: List[torch.Tensor],
                   add_kl: bool = False, parent_depth: int = 0, parent_value: float = 0.0) -> List[float]:
        """Get values for actions (simplified for now)."""
        # For now, return simple values based on token probabilities
        values = []
        for action in action_batch:
            # Simple value based on log probability
            value = float(action.log_prob)
            values.append(value)
        return values
    
    def reward(self, state: VLLMState, action: VLLMAction, **kwargs) -> Tuple[float, Dict]:
        """Get reward for state-action pair."""
        # Simple reward based on log probability
        reward = float(action.log_prob)
        return reward, {}
    
    def update_example(self, example: Any) -> None:
        """Update example for the search config."""
        self.example = example


class VLLMMCTS:
    """MCTS implementation following MCTS-DPO structure but using VLLM."""
    
    def __init__(self, vllm_client, tokenizer: PreTrainedTokenizerBase, mcts_config: MCTSConfig):
        self.vllm_client = vllm_client
        self.tokenizer = tokenizer
        self.mcts_config = mcts_config
        
        # MCTS-DPO style parameters
        self.w_exp = 1.0  # exploration weight
        self.depth_limit = mcts_config.depth_limit
        self.breadth_limit = mcts_config.topk_expand
        self.n_iters = mcts_config.num_simulations
        self.gamma = 1.0
        self.add_kl = False
        self.temperature = mcts_config.temperature
        self.temperature_decay_ratio = 0.75
        
        # Initialize world model and search config
        self.world_model = VLLMWorldModel(vllm_client, tokenizer, mcts_config.rollout_max_tokens)
        self.search_config = VLLMSearchConfig(vllm_client, tokenizer, mcts_config)
        self.policy_model = None  # Not needed for VLLM
        
        self.root: Optional[VLLMMCTSNode] = None
    
    def search(self, prompt: str, top_k: int = 8) -> List[VLLMMCTSNode]:
        """Run MCTS and return top-K distinct leaf nodes."""
        print(f"[MCTS SEARCH DEBUG] Starting search for prompt: {prompt[:50]}...")
        initial_state = self.world_model.init_state(prompt)
        self.root = VLLMMCTSNode(state=initial_state)
        print(f"[MCTS SEARCH DEBUG] Root state terminal: {initial_state.is_terminal}")

        for iter_idx in range(self.n_iters):
            path = self._select(self.root)
            if not path:
                continue
            self._expand_and_evaluate(path[-1])
            current = path[-1]
            steps = 0
            while (not self._is_terminal_with_depth_limit(current)) and len(current.children) > 0 and steps < self.depth_limit:
                current = self._puct_select(current)
                path.append(current)
                self._expand_and_evaluate(current)
                steps += 1
            self._back_propagate(path)

        return self._extract_topk_paths(top_k)

    def _extract_topk_paths(self, k: int) -> List[VLLMMCTSNode]:
        """Collect leaves, rank by (N, Q), and return K best distinct leaves by final sequence."""
        leaves: List[VLLMMCTSNode] = []

        def dfs(n: VLLMMCTSNode):
            if not n.children:
                leaves.append(n)
                return
            for c in n.children.values():
                dfs(c)

        if self.root is None:
            return []
        dfs(self.root)

        if not leaves:
            return [self.root]

        # Rank leaves: visit count first, then Q
        leaves.sort(key=lambda n: (n.N, n.Q), reverse=True)

        # Deduplicate by full generated sequence from root (total, not suffix)
        seen = set()
        picked = []
        for leaf in leaves:
            seq_key = tuple(leaf.state.input_ids)  # using full ids here
            if seq_key in seen:
                continue
            seen.add(seq_key)
            picked.append(leaf)
            if len(picked) >= k:
                break

        if len(picked) < k:
            # pad if needed
            for leaf in leaves:
                picked.append(leaf)
                if len(picked) >= k:
                    break
        return picked

    def _select(self, node: VLLMMCTSNode) -> List[VLLMMCTSNode]:
        """Selection phase following MCTS-DPO."""
        path = []
        while True:
            path.append(node)
            if node.children is None or len(node.children) == 0 or self._is_terminal_with_depth_limit(node):
                return path
            node = self._puct_select(node)
    
    def _puct(self, node: VLLMMCTSNode) -> float:
        """PUCT formula from MCTS-DPO."""
        if node.parent is None:
            return 0.0
        return node.Q + self.w_exp * node.p * np.sqrt(node.parent.N) / (1 + node.N)
    
    def _puct_select(self, node: VLLMMCTSNode) -> VLLMMCTSNode:
        """Select child using PUCT."""
        return max(node.children.values(), key=self._puct)
    
    def _expand_and_evaluate(self, node: VLLMMCTSNode):
        """Expansion and evaluation phase."""
        if node.state is None:
            node.state = self.world_model.step(node.parent.state, node.action, node.log_probs)
            node.is_terminal = self.world_model.is_terminal(node.state)
        
        if node.is_terminal:
            return
        
        # Get actions from search config
        actions = self.search_config.get_actions(self.policy_model, node.state, add_kl=self.add_kl)
        
        # Limit to breadth_limit
        if len(actions) > self.breadth_limit:
            actions = actions[:self.breadth_limit]

        # sibling priors from logprobs, normalized via softmax
        # log_probs is shaped [1], stack into [B]
        sib_logps = torch.stack([lp for _, (lp, _), _ in actions]).float().view(-1)
        sib_priors = F.softmax(sib_logps, dim=0)  # [B]
        
        # Create children keyed by FULL sequence to avoid collisions
        for i, (action, (log_probs, ref_log_probs), _) in enumerate(actions):
            child_state = self.world_model.step(node.state, action, log_probs)
            child_node = VLLMMCTSNode(state=child_state, action=action, parent=node)
            child_node.log_probs = log_probs
            child_node.ref_log_probs = ref_log_probs
            child_node.p = float(sib_priors[i].item())  # normalized prior in (0,1)

            key = tuple(action.token_ids)  # <-- avoid first-token collisions
            node.children[key] = child_node

        print(f"[MCTS DEBUG] Expanded node at depth {node.depth}: added {len(actions)} children")
        
        # Values for children
        action_batch = [a for a, _, _ in actions]
        log_probs_batch = [lp for _, (lp, _), _ in actions]
        ref_log_probs_batch = [rlp for _, (_, rlp), _ in actions]

        values = self.search_config.get_values(
            self.policy_model, node.state, action_batch,
            log_probs_batch, ref_log_probs_batch,
            add_kl=self.add_kl, parent_depth=node.depth, parent_value=node.value
        )

        # Assign values
        for (action, _, _), v in zip(actions, values):
            key = tuple(action.token_ids)
            if key in node.children:
                node.children[key].value = v
    
    def _back_propagate(self, path: List[VLLMMCTSNode]):
        """Backpropagation phase."""
        if not path:
            return
        
        # Calculate cumulative reward
        cum_reward = 0.0
        for i, node in enumerate(path):
            if node.action:
                reward, _ = self.search_config.reward(node.state, node.action)
                cum_reward += (self.gamma ** i) * reward
        
        # Update nodes
        for node in path:
            node.N += 1
            node.Q = (node.Q * (node.N - 1) + cum_reward) / node.N
            node.value = cum_reward
    
    def _is_terminal_with_depth_limit(self, node: VLLMMCTSNode) -> bool:
        """Check if node is terminal with depth limit."""
        return node.is_terminal or (node.depth - self.root.depth) >= self.mcts_config.depth_limit
    
    def _get_best_path(self) -> List[VLLMMCTSNode]:
        """Get best path from root."""
        if not self.root or not self.root.children:
            return [self.root]
        
        # Find best child by visit count
        best_child = max(self.root.children.values(), key=lambda x: x.N)
        path = [self.root, best_child]
        
        # Continue until terminal or no children
        current = best_child
        while current.children:
            best_child = max(current.children.values(), key=lambda x: x.N)
            path.append(best_child)
            current = best_child
        
        return path


def integrate_mcts_dpo_with_vllm(
    trainer,
    prompts_text: List[str],
    num_generations: int,   # set to 1 in your caller
    **generation_kwargs
) -> List[Any]:
    """
    Single MCTS search per prompt returning top_k distinct completions.
    Returns a flat list of token-id suffixes (one per candidate).
    """
    mcts_config = get_mcts_config("default")
    mcts = VLLMMCTS(
        vllm_client=trainer.vllm_client,
        tokenizer=trainer.tokenizer,
        mcts_config=mcts_config
    )

    result = []
    for prompt_text in prompts_text:
        print(f"[MCTS DEBUG] Processing prompt: {prompt_text[:100]}...")
        # One search ⇒ K leaves
        leaves = mcts.search(prompt_text, top_k=mcts_config.top_k)

        result.extend([leaf.state.input_ids for leaf in leaves])

    return result

        