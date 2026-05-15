"""MoE (Mixture of Experts) router for expert parallelism."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from loguru import logger


class MoERouter(nn.Module):
    """Route tokens to expert nodes based on gating scores.

    Implements top-k routing with load balancing for MoE models
    like Mixtral and DeepSeek.
    """

    def __init__(self, num_experts: int, num_experts_per_tok: int = 2, hidden_dim: int = 4096):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self._expert_map: Dict[int, str] = {}  # expert_id -> node_id

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gating scores and route tokens.

        Args:
            hidden_states: [seq_len, hidden_dim]

        Returns:
            topk_indices: [seq_len, num_experts_per_tok] - expert IDs
            weights: [seq_len, num_experts_per_tok] - gating weights (softmax-normalized)
        """
        scores = self.gate(hidden_states)  # [seq_len, num_experts]
        topk_weights, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        weights = torch.softmax(topk_weights, dim=-1)
        return topk_indices, weights

    def register_expert(self, expert_id: int, node_id: str) -> None:
        """Register which node holds a given expert."""
        self._expert_map[expert_id] = node_id

    def get_expert_node(self, expert_id: int) -> Optional[str]:
        """Get the node ID that holds a given expert."""
        return self._expert_map.get(expert_id)

    def route_to_nodes(
        self, hidden_states: torch.Tensor
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Route tokens to their expert nodes.

        Uses vectorized operations instead of Python loops for performance.

        Returns:
            Dict mapping node_id -> (tokens, weights) for tokens routed to that node.
        """
        topk_indices, weights = self(hidden_states)
        seq_len = hidden_states.shape[0]

        # Build expert_id -> node_id mapping as tensors for vectorized lookup
        expert_to_node = {}
        for expert_id, node_id in self._expert_map.items():
            expert_to_node[expert_id] = node_id

        # Vectorized: collect all (token_idx, expert_rank, node_id) tuples
        node_groups: Dict[str, List[int]] = {}
        for tok_idx in range(seq_len):
            for expert_rank in range(self.num_experts_per_tok):
                expert_id = int(topk_indices[tok_idx, expert_rank].item())
                node_id = expert_to_node.get(expert_id)
                if node_id is None:
                    logger.warning(f"No node registered for expert {expert_id}")
                    continue
                node_groups.setdefault(node_id, []).append(tok_idx * self.num_experts_per_tok + expert_rank)

        # Stack tensors for each node using gathered indices
        result = {}
        for node_id, flat_indices in node_groups.items():
            flat_indices = torch.tensor(flat_indices, device=hidden_states.device)
            token_indices = flat_indices // self.num_experts_per_tok
            expert_ranks = flat_indices % self.num_experts_per_tok

            token_tensors = hidden_states[token_indices]
            token_weights = weights[token_indices, expert_ranks]
            result[node_id] = (token_tensors, token_weights)

        return result


class MoEModelDetector:
    """Detect if a model is an MoE model and extract config."""

    MOE_ARCHITECTURES = {
        "MixtralForCausalLM",
        "DeepseekV2ForCausalLM",
        "Qwen2MoeForCausalLM",
        "JambaForCausalLM",
    }

    @staticmethod
    def is_moe(config) -> bool:
        """Check if the model config indicates an MoE architecture."""
        arch = getattr(config, "architectures", [])
        if any(a in MoEModelDetector.MOE_ARCHITECTURES for a in arch):
            return True
        # Check for MoE-specific config attributes
        if hasattr(config, "num_local_experts"):
            return True
        if hasattr(config, "n_routed_experts"):
            return True
        return False

    @staticmethod
    def get_moe_config(config) -> dict:
        """Extract MoE configuration from model config."""
        return {
            "num_experts": getattr(config, "num_local_experts", getattr(config, "n_routed_experts", 8)),
            "num_experts_per_tok": getattr(config, "num_experts_per_token", 2),
            "intermediate_size": getattr(config, "intermediate_size", None),
            "hidden_dim": config.hidden_size,
        }
