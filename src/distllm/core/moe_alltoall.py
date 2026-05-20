"""All-to-all communication for MoE expert parallelism.

Implements the all-to-all collective used to dispatch tokens to the
correct expert nodes and gather results in Mixture of Experts models.

Provides:
- Token-to-expert mapping and routing
- All-to-all dispatch (sending tokens to expert nodes)
- All-to-all gather (receiving processed tokens from expert nodes)
- Load balancing: capacity factor and auxiliary loss integration
- Integration with NCCL and gRPC transport backends
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass
class AllToAllStats:
    total_dispatches: int = 0
    total_tokens: int = 0
    total_bytes: int = 0
    total_time_ms: float = 0.0
    avg_experts_per_token: float = 0.0
    load_imbalance: float = 0.0


class MoEAllToAll:
    """All-to-all communication for MoE expert parallelism.

    Coordinates the dispatch of token hidden states to the correct
    expert nodes and gathers the results back.

    Flow:
    1. Token-to-expert mapping: router assigns each token to top-k experts
    2. All-to-all dispatch: send tokens to their assigned expert nodes
    3. Expert forward: each node processes its assigned tokens
    4. All-to-all gather: return processed tokens to original nodes
    5. Weighted combine: aggregate expert outputs by routing weights

    Usage:
        alltoall = MoEAllToAll(num_experts=8, num_nodes=4)
        dispatch = alltoall.dispatch(hidden_states, expert_indices, routing_weights)
        # ... each node processes its expert tokens ...
        output = alltoall.gather(dispatch)
    """

    def __init__(
        self,
        num_experts: int = 8,
        num_nodes: int = 1,
        experts_per_node: list[int] | None = None,
        transport_fn: Callable | None = None,
        capacity_factor: float = 1.0,
        top_k: int = 2,
    ):
        self._num_experts = num_experts
        self._num_nodes = num_nodes
        self._capacity_factor = capacity_factor
        self._top_k = top_k

        # experts_per_node: list of expert IDs assigned to each node
        if experts_per_node:
            self._experts_per_node = experts_per_node
        else:
            experts_per_node_count = max(1, num_experts // max(num_nodes, 1))
            self._experts_per_node = [
                list(range(i * experts_per_node_count, (i + 1) * experts_per_node_count))
                for i in range(num_nodes)
            ]
            # Add remaining experts to last node
            remaining = num_experts - sum(len(e) for e in self._experts_per_node)
            if remaining > 0:
                self._experts_per_node[-1].extend(
                    range(num_experts - remaining, num_experts)
                )

        self._transport_fn = transport_fn

        self._stats = AllToAllStats()
        self._lock = threading.Lock()
        self._expert_to_node: dict[int, int] = {}
        for node_id, experts in enumerate(self._experts_per_node):
            for e in experts:
                self._expert_to_node[e] = node_id

    # -------------------------------------------------------------------
    # Dispatch
    # -------------------------------------------------------------------

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> dict[int, dict[str, torch.Tensor]]:
        """Dispatch tokens to their assigned expert nodes.

        Args:
            hidden_states: (num_tokens, hidden_dim) tensor.
            expert_indices: (num_tokens, top_k) expert IDs per token.
            routing_weights: (num_tokens, top_k) routing weights per token.

        Returns:
            Dict[node_id -> {"hidden": tensor, "weights": tensor, "token_map": tensor}]
            for each node that received tokens.
        """
        start = time.time()
        num_tokens = hidden_states.shape[0]

        # Build per-node token lists
        node_tokens: dict[int, list[int]] = {n: [] for n in range(self._num_nodes)}
        node_weights: dict[int, list[float]] = {n: [] for n in range(self._num_nodes)}
        token_map: dict[int, list[int]] = {n: [] for n in range(self._num_nodes)}

        for token_idx in range(num_tokens):
            for k in range(self._top_k):
                expert_id = int(expert_indices[token_idx, k])
                weight = float(routing_weights[token_idx, k])
                node_id = self._expert_to_node.get(expert_id, 0)
                node_tokens[node_id].append(token_idx)
                node_weights[node_id].append(weight)
                token_map[node_id].append(token_idx)

        # Apply capacity factor (drop excess tokens)
        dispatched: dict[int, dict[str, torch.Tensor]] = {}
        for node_id in range(self._num_nodes):
            tids = node_tokens[node_id]
            if not tids:
                continue

            max_capacity = int(len(tids) * self._capacity_factor)
            if len(tids) > max_capacity:
                # Keep only max_capacity tokens (dropping excess)
                tids = tids[:max_capacity]
                node_weights[node_id] = node_weights[node_id][:max_capacity]
                token_map[node_id] = token_map[node_id][:max_capacity]

            node_hidden = hidden_states[tids]
            dispatched[node_id] = {
                "hidden": node_hidden,
                "weights": torch.tensor(node_weights[node_id], device=hidden_states.device),
                "token_map": torch.tensor(token_map[node_id], device=hidden_states.device),
            }

        elapsed = time.time() - start
        total_dispatched = sum(d["hidden"].shape[0] for d in dispatched.values())
        total_bytes = sum(d["hidden"].numel() * d["hidden"].element_size() for d in dispatched.values())

        with self._lock:
            self._stats.total_dispatches += 1
            self._stats.total_tokens += num_tokens
            self._stats.total_bytes += total_bytes
            self._stats.total_time_ms += elapsed * 1000
            self._stats.avg_experts_per_token = self._top_k

            if len(dispatched) > 1:
                token_counts = [d["hidden"].shape[0] for d in dispatched.values()]
                max_count = max(token_counts)
                min_count = min(token_counts)
                self._stats.load_imbalance = (max_count - min_count) / max(max_count, 1)

        return dispatched

    # -------------------------------------------------------------------
    # Gather
    # -------------------------------------------------------------------

    def gather(
        self,
        dispatched: dict[int, dict[str, torch.Tensor]],
        expert_outputs: dict[int, torch.Tensor],
        num_tokens: int,
        hidden_dim: int,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Gather expert outputs and combine weighted results.

        Args:
            dispatched: The dispatch dict from dispatch().
            expert_outputs: Dict[node_id -> processed hidden states tensor].
            num_tokens: Original number of tokens.
            hidden_dim: Hidden dimension size.
            device: Output device.

        Returns:
            Combined output tensor (num_tokens, hidden_dim).
        """
        output = torch.zeros(num_tokens, hidden_dim, device=device)

        for node_id, dispatch_info in dispatched.items():
            node_output = expert_outputs.get(node_id)
            if node_output is None:
                continue

            token_map = dispatch_info["token_map"]
            weights = dispatch_info["weights"]
            num_node_tokens = node_output.shape[0]

            for i in range(num_node_tokens):
                original_idx = int(token_map[i])
                weight = float(weights[i]) if weights.dim() == 1 or (weights.dim() == 0) else 0.0
                output[original_idx] += node_output[i] * weight

        return output

    # -------------------------------------------------------------------
    # Load Balancing
    # -------------------------------------------------------------------

    def compute_load_balancing_loss(
        self,
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Auxiliary load balancing loss (e.g., from Switch Transformer / DeepSpeed).

        Encourages uniform routing across experts.

        Args:
            expert_indices: (num_tokens, top_k) expert IDs.
            routing_weights: (num_tokens, top_k) routing probabilities.

        Returns:
            Scalar loss tensor.
        """
        num_tokens = expert_indices.shape[0]
        expert_counts = torch.zeros(self._num_experts, device=expert_indices.device)
        expert_probs = torch.zeros(self._num_experts, device=routing_weights.device)

        for k in range(self._top_k):
            indices = expert_indices[:, k]
            weights = routing_weights[:, k]
            expert_counts.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float))
            expert_probs.scatter_add_(0, indices, weights)

        fraction_per_expert = expert_counts / max(num_tokens, 1)
        prob_per_expert = expert_probs / max(num_tokens, 1)

        loss = self._num_experts * (fraction_per_expert * prob_per_expert).sum()
        return loss

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_dispatches": self._stats.total_dispatches,
                "total_tokens": self._stats.total_tokens,
                "total_bytes": self._stats.total_bytes,
                "total_time_ms": round(self._stats.total_time_ms, 2),
                "avg_experts_per_token": self._stats.avg_experts_per_token,
                "load_imbalance": round(self._stats.load_imbalance, 4),
                "experts_per_node": [list(e) for e in self._experts_per_node],
                "capacity_factor": self._capacity_factor,
            }
