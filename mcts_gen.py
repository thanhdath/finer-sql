from __future__ import annotations
from typing import Callable, Dict, List, Optional
import math

# Defaults centralized here so grpo_writer.py doesn't need CLI args
MCTS_DEFAULT_MAX_NEW_TOKENS = 32
MCTS_DEFAULT_NUM_SIMULATIONS = 64
MCTS_DEFAULT_C_PUCT = 1.5
MCTS_DEFAULT_TOPK_EXPAND = 20
MCTS_DEFAULT_TEMPERATURE = 1.0


class _MCTSNode:
    def __init__(self, input_ids: List[int], logprob_sum: float = 0.0, parent: Optional["_MCTSNode"] = None):
        self.input_ids: List[int] = input_ids
        self.parent = parent
        self.children: Dict[int, _MCTSNode] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior_probs: Optional[Dict[int, float]] = None  # token_id -> prior prob
        self.logprob_sum: float = logprob_sum

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / float(self.visit_count)

    def ucb_score(self, child_token: int, c_puct: float) -> float:
        child = self.children[child_token]
        prior = 0.0
        if self.prior_probs is not None:
            prior = self.prior_probs.get(child_token, 0.0)
        if child.visit_count == 0:
            q = 0.0
        else:
            q = child.value
        u = c_puct * prior * math.sqrt(max(1, self.visit_count)) / (1 + child.visit_count)
        return q + u


def _softmax(logits, temperature: float = 1.0):
    import numpy as np
    x = np.array(logits, dtype=float)
    x = x / max(temperature, 1e-8)
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    if s <= 0:
        return e * 0.0
    return e / s


def _topk_indices(logits, k: int):
    import numpy as np
    x = np.array(logits, dtype=float)
    if k >= x.size:
        return np.argsort(-x)
    idx = np.argpartition(-x, k)[:k]
    return idx[np.argsort(-x[idx])]


def _default_leaf_value(input_ids: List[int], max_steps: int, temperature: float) -> float:
    # You can replace this with a task-specific value estimator (e.g., RM or heuristic)
    return 0.0


def _mcts_generate_one(
    prompt_ids: List[int],
    *,
    get_next_logits: Callable[[List[int]], "np.ndarray"],  # returns 1D numpy array of logits
    max_new_tokens: int,
    num_simulations: int,
    c_puct: float,
    topk_expand: int,
    temperature: float,
    leaf_value_fn: Optional[Callable[[List[int], int, float], float]] = None,
) -> List[int]:
    import numpy as np  # local import to avoid hard dependency where unused
    root = _MCTSNode(input_ids=prompt_ids)
    leaf_value_fn = leaf_value_fn or _default_leaf_value

    for _ in range(num_simulations):
        # Selection
        node = root
        while True:
            if len(node.input_ids) - len(prompt_ids) >= max_new_tokens:
                break
            if not node.children:
                break
            best_t = None
            best_score = -1e9
            for t in node.children.keys():
                score = node.ucb_score(t, c_puct)
                if score > best_score:
                    best_score = score
                    best_t = t
            node = node.children[best_t]

        # Expansion
        if len(node.input_ids) - len(prompt_ids) < max_new_tokens:
            logits = get_next_logits(node.input_ids)  # implement this in your codebase
            probs = _softmax(logits, temperature=max(temperature, 1e-6))
            top_idx = _topk_indices(logits, topk_expand)
            node.prior_probs = {int(t): float(probs[int(t)]) for t in top_idx}
            for t in top_idx:
                t_int = int(t)
                child_ids = node.input_ids + [t_int]
                node.children[t_int] = _MCTSNode(input_ids=child_ids, parent=node)

            if len(top_idx) > 0:
                priors = np.array([probs[int(t)] for t in top_idx], dtype=float)
                priors = priors / (priors.sum() + 1e-12)
                pick = int(np.random.choice(top_idx, p=priors))
                leaf = node.children[pick]
            else:
                leaf = node
        else:
            leaf = node

        # Simulation/value
        value = leaf_value_fn(leaf.input_ids, max_new_tokens, temperature)

        # Backpropagate
        cur = leaf
        while cur is not None:
            cur.visit_count += 1
            cur.value_sum += float(value)
            cur = cur.parent

    # Greedy extraction by visit-count
    out_node = root
    while len(out_node.input_ids) - len(prompt_ids) < max_new_tokens and out_node.children:
        best_t = max(out_node.children.keys(), key=lambda t: out_node.children[t].visit_count)
        out_node = out_node.children[best_t]

    return out_node.input_ids


def generate_mcts_candidates(
    prompt: str,
    *,
    num_generations: int,
    tokenizer,
    get_next_logits: Callable[[List[int]], "np.ndarray"],
    max_new_tokens: int = MCTS_DEFAULT_MAX_NEW_TOKENS,
    num_simulations: int = MCTS_DEFAULT_NUM_SIMULATIONS,
    c_puct: float = MCTS_DEFAULT_C_PUCT,
    topk_expand: int = MCTS_DEFAULT_TOPK_EXPAND,
    temperature: float = MCTS_DEFAULT_TEMPERATURE,
    leaf_value_fn: Optional[Callable[[List[int], int, float], float]] = None,
) -> List[str]:
    """
    Return exactly num_generations sequences using simple MCTS.

    You must pass a callable get_next_logits(input_ids) -> numpy.ndarray of shape [vocab_size].
    If you need to define it elsewhere, add a function matching that signature and pass it here.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    outputs: List[str] = []
    for _ in range(int(num_generations)):
        out_ids = _mcts_generate_one(
            prompt_ids,
            get_next_logits=get_next_logits,
            max_new_tokens=max_new_tokens,
            num_simulations=num_simulations,
            c_puct=c_puct,
            topk_expand=topk_expand,
            temperature=temperature,
            leaf_value_fn=leaf_value_fn,
        )
        text = tokenizer.decode(out_ids[len(prompt_ids):], skip_special_tokens=True)
        outputs.append(text)
    return outputs


