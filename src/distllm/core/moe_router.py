"""MoE (Mixture of Experts) router for expert parallelism."""

import torch
import torch.nn as nn


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
        self._expert_map: dict[int, str] = {}  # expert_id -> node_id

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def get_expert_node(self, expert_id: int) -> str | None:
        """Get the node ID that holds a given expert."""
        return self._expert_map.get(expert_id)

    def route_to_nodes(
        self, hidden_states: torch.Tensor
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Route tokens to their expert nodes.

        Uses fully vectorized operations: no Python loops over tokens.
        Builds a node_mask tensor and uses torch.unique + boolean indexing
        to group tokens by destination node.

        Returns:
            Dict mapping node_id -> (tokens, weights) for tokens routed to that node.
        """
        topk_indices, weights = self(hidden_states)
        seq_len = hidden_states.shape[0]

        # Flatten: each token sends num_experts_per_tok assignments
        flat_indices = topk_indices.flatten()  # [seq_len * topk]
        flat_weights = weights.flatten()

        # Build expert_id -> node_id lookup as a tensor for vectorized indexing
        max_expert_id = max(self._expert_map.keys(), default=-1)
        if max_expert_id < 0:
            return {}

        # Create a lookup table: expert_id -> node_id (as integer codes)
        node_id_to_code: dict[str, int] = {}
        code_to_node_id: dict[int, str] = {}
        for idx, (expert_id, node_id) in enumerate(self._expert_map.items()):
            if node_id not in node_id_to_code:
                code = len(node_id_to_code)
                node_id_to_code[node_id] = code
                code_to_node_id[code] = node_id

        # Map each expert to its node code; unmapped experts get -1
        expert_to_node_code = torch.full((max_expert_id + 1,), -1, dtype=torch.long, device=hidden_states.device)
        for expert_id, node_id in self._expert_map.items():
            expert_to_node_code[expert_id] = node_id_to_code[node_id]

        # Vectorized lookup: for each (token, rank) assignment, get node code
        node_codes = expert_to_node_code[flat_indices]  # [seq_len * topk]

        # Filter out unmapped experts
        valid_mask = node_codes >= 0
        if not valid_mask.any():
            return {}

        valid_codes = node_codes[valid_mask]
        valid_weights = flat_weights[valid_mask]
        # Repeat hidden_states for each expert assignment
        token_indices = torch.arange(seq_len, device=hidden_states.device).repeat_interleave(self.num_experts_per_tok)
        valid_token_indices = token_indices[valid_mask]
        valid_tokens = hidden_states[valid_token_indices]

        # Group by node using unique
        unique_codes, inverse_indices = torch.unique(valid_codes, return_inverse=True)

        result = {}
        for i, code in enumerate(unique_codes.tolist()):
            node_id = code_to_node_id[code]
            mask = inverse_indices == i
            result[node_id] = (valid_tokens[mask], valid_weights[mask])

        return result
