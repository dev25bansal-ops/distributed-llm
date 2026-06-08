"""Mixture-of-Experts (MoE) automatic expert routing for distributed inference.

MoE models (Mixtral, Qwen2-MoE, DeepSeekMoE, etc.) have sparse expert
layers where each token is routed to a subset of experts.  In a distributed
setting, different nodes can host different experts, significantly reducing
the per-node compute and memory requirements.

Architecture::

    Token → Router → Top-2 experts selected
                      ├── Expert A (Node 1)
                      └── Expert B (Node 2)
    Output ← All-to-all combine
"""

from __future__ import annotations

import threading
from typing import Any, Callable

import torch
import torch.nn.functional as F
from loguru import logger


class ExpertRegistry:
    """Tracks which experts are loaded on which nodes.

    Each expert is identified by (layer_index, expert_id).  The registry
    maps expert locations to node addresses for gRPC routing.
    """

    def __init__(self):
        self._expert_map: dict[tuple[int, int], str] = {}  # (layer, expert) -> node_id
        self._node_experts: dict[str, list[tuple[int, int]]] = {}  # node_id -> [(layer, expert)]
        self._lock = threading.Lock()

    def register_expert(self, node_id: str, layer_idx: int, expert_id: int) -> None:
        with self._lock:
            key = (layer_idx, expert_id)
            self._expert_map[key] = node_id
            self._node_experts.setdefault(node_id, []).append(key)

    def unregister_node(self, node_id: str) -> None:
        with self._lock:
            for key in self._node_experts.pop(node_id, []):
                self._expert_map.pop(key, None)

    def get_node_for_expert(self, layer_idx: int, expert_id: int) -> str | None:
        with self._lock:
            return self._expert_map.get((layer_idx, expert_id))

    def get_experts_on_node(self, node_id: str) -> list[tuple[int, int]]:
        with self._lock:
            return list(self._node_experts.get(node_id, []))

    def has_expert(self, layer_idx: int, expert_id: int) -> bool:
        with self._lock:
            return (layer_idx, expert_id) in self._expert_map


class MoERouter:
    """Routes tokens to the appropriate node based on expert selection.

    For each MoE layer in the model, the router:
      1. Computes routing weights (gates) for each token
      2. Selects top-k experts per token
      3. Routes the token's hidden state to the node hosting the selected expert
      4. Receives the expert output and combines via weighted sum
    """

    def __init__(
        self,
        registry: ExpertRegistry,
        num_experts: int = 8,
        top_k: int = 2,
        hidden_size: int = 4096,
    ):
        self._registry = registry
        self._num_experts = num_experts
        self._top_k = top_k
        self._hidden_size = hidden_size

    def route(
        self,
        hidden_states: torch.Tensor,
        gate_weights: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Route tokens through their assigned experts.

        Args:
            hidden_states: Token hidden states, shape ``(batch, seq, hidden)``.
            gate_weights: Router logits, shape ``(batch, seq, num_experts)``.
            layer_idx: Current transformer layer index.

        Returns:
            Output hidden states after expert processing, same shape as input.
        """
        batch, seq, hidden = hidden_states.shape

        # Select top-k experts per token
        top_k_weights, top_k_indices = torch.topk(
            F.softmax(gate_weights, dim=-1), self._top_k, dim=-1
        )

        output = torch.zeros_like(hidden_states)

        for k in range(self._top_k):
            expert_ids = top_k_indices[:, :, k].flatten()
            routing_weights = top_k_weights[:, :, k].flatten()

            # Group tokens by their assigned node
            node_tokens: dict[str, list[tuple[int, torch.Tensor, float]]] = {}
            for pos, expert_id in enumerate(expert_ids):
                node_id = self._registry.get_node_for_expert(layer_idx, int(expert_id))
                if node_id is None:
                    continue
                b = pos // seq
                s = pos % seq
                token_hidden = hidden_states[b, s:s+1, :]
                weight = float(routing_weights[pos].item())
                node_tokens.setdefault(node_id, []).append((pos, token_hidden, weight))

            # Send tokens to their assigned nodes and collect results
            for node_id, tokens in node_tokens.items():
                for pos, token_hidden, weight in tokens:
                    expert_id = int(expert_ids[pos])
                    expert_output = self._forward_on_node(
                        expert_id, token_hidden, layer_idx, node_id
                    )
                    b = pos // seq
                    s = pos % seq
                    output[b, s, :] += expert_output * weight

        return output

    def _forward_on_node(
        self, expert_id: int, hidden: torch.Tensor,
        layer_idx: int, node_id: str,
    ) -> torch.Tensor:
        """Send a single token to a remote node for expert processing.

        Uses gRPC to forward the token to the node hosting the expert.
        The remote node applies the expert MLP to the hidden state and
        returns the result.

        Falls back to an identity forward if the gRPC call fails
        (logged warning), preserving the model's ability to produce
        reasonable output even with degraded expert routing.
        """
        from distllm.dist.grpc_client import GrpcClientPool

        try:
            client = GrpcClientPool.get_client(node_id)
            if client is None:
                logger.warning(f"No gRPC client for node {node_id}, using identity fallback for expert {expert_id}")
                return hidden

            result = client.call_expert_forward(
                expert_id=expert_id,
                hidden_states=hidden,
                layer_idx=layer_idx,
            )
            if result is None:
                logger.warning(f"gRPC expert forward returned None for node {node_id}, expert {expert_id}")
                return hidden

            return result
        except Exception as e:
            logger.warning(
                f"Expert forward failed: node={node_id} expert={expert_id} layer={layer_idx}: {e}"
            )
            return hidden  # Identity fallback — better than garbage output


class MoEOrchestrator:
    """Orchestrates MoE inference across distributed nodes.

    Integrates with the pipeline's node registration system to track
    which experts are available on which nodes.
    """

    def __init__(self, pipeline=None, num_experts: int = 8, top_k: int = 2):
        self._pipeline = pipeline
        self._registry = ExpertRegistry()
        self._num_experts = num_experts
        self._top_k = top_k

    def register_node_experts(
        self, node_id: str, expert_ids: list[int], layer_idx: int = 0,
    ) -> None:
        for eid in expert_ids:
            self._registry.register_expert(node_id, layer_idx, eid)
        logger.info(f"Registered experts {expert_ids} on {node_id} (layer {layer_idx})")

    def unregister_node(self, node_id: str) -> None:
        self._registry.unregister_node(node_id)

    def get_expert_summary(self) -> dict[str, Any]:
        return {
            "total_experts": self._num_experts,
            "top_k": self._top_k,
            "nodes": list(self._registry._node_experts.keys()),
        }
