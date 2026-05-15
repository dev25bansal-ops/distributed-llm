"""Distributed MoE inference orchestrator.

Coordinates expert routing, dispatch, and aggregation across
distributed worker nodes for Mixture of Experts models.
"""

from typing import Dict, List, Optional

import torch
from loguru import logger


class MoEForwardRequest:
    """Container for an MoE forward request to a single node."""

    def __init__(self, node_id: str, hidden_states: torch.Tensor,
                 expert_ids: List[int], routing_weights: List[float],
                 request_id: str):
        self.node_id = node_id
        self.hidden_states = hidden_states
        self.expert_ids = expert_ids
        self.routing_weights = routing_weights
        self.request_id = request_id


class MoEForwardResponse:
    """Container for an MoE forward response from a node."""

    def __init__(self, node_id: str, output: torch.Tensor,
                 success: bool, error_message: str = "",
                 processing_time_ms: float = 0.0):
        self.node_id = node_id
        self.output = output
        self.success = success
        self.error_message = error_message
        self.processing_time_ms = processing_time_ms


class MoEOrchestrator:
    """Coordinates distributed expert execution.

    Flow:
    1. route: Use MoE router to get expert assignments per token
    2. dispatch: Send expert requests to appropriate nodes in parallel
    3. aggregate: Combine expert outputs weighted by routing weights
    """

    def __init__(self, expert_registry=None):
        self.expert_registry = expert_registry

    def route(
        self,
        hidden_states: torch.Tensor,
        moe_router,
    ) -> Dict[str, MoEForwardRequest]:
        """Route hidden states to expert nodes.

        Uses the MoE router to compute expert assignments, then groups
        tokens by target node.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim].
            moe_router: MoERouter instance with registered expert-to-node mapping.

        Returns:
            Dict mapping node_id to MoEForwardRequest.
        """
        # Get expert assignments from router
        # moe_router.forward returns routing weights [batch, seq_len, num_experts_per_tok]
        # and we need to map experts to nodes
        routing_output = moe_router(hidden_states)  # [batch, seq_len, num_experts_per_tok]

        # Group by expert
        expert_tokens: Dict[int, List[tuple]] = {}  # expert_id -> list of (token_idx, weight)

        # Simplified: assume flat routing for demonstration
        # In production, this would handle the full [batch, seq_len, k] routing
        requests_by_node: Dict[str, MoEForwardRequest] = {}

        if self.expert_registry is None:
            # Fallback: treat as standard forward pass
            logger.debug("No expert registry, skipping MoE routing")
            return requests_by_node

        # For each token, determine which experts handle it and which nodes hold them
        # This is a simplified single-token version
        num_experts_per_tok = routing_output.shape[-1] if routing_output.dim() > 1 else 1

        if routing_output.dim() == 3:
            # [batch, seq_len, num_experts_per_tok]
            batch_size, seq_len, k = routing_output.shape
            hidden_dim = hidden_states.shape[-1]

            # Flatten: treat each (batch, seq) position as a token
            for b in range(batch_size):
                for s in range(seq_len):
                    top_experts = routing_output[b, s, :].tolist()
                    for expert_idx, weight in enumerate(top_experts):
                        if weight > 0:
                            if expert_idx not in expert_tokens:
                                expert_tokens[expert_idx] = []
                            expert_tokens[expert_idx].append((b * seq_len + s, weight))
        elif routing_output.dim() == 2:
            # [num_tokens, num_experts_per_tok]
            num_tokens, k = routing_output.shape
            for i in range(num_tokens):
                weights = routing_output[i].tolist()
                for expert_idx, weight in enumerate(weights):
                    if weight > 0:
                        if expert_idx not in expert_tokens:
                            expert_tokens[expert_idx] = []
                        expert_tokens[expert_idx].append((i, weight))

        # Group by node
        node_expert_tokens: Dict[str, List[tuple]] = {}
        node_expert_ids: Dict[str, set] = {}
        node_weights: Dict[str, List[float]] = {}

        for expert_id, token_weights in expert_tokens.items():
            nodes = self.expert_registry.get_expert_nodes(expert_id)
            if not nodes:
                logger.warning(f"No node found for expert {expert_id}")
                continue

            # Select best node for this expert
            node = self.expert_registry.select_best_node(expert_id)
            if node is None:
                continue

            if node not in node_expert_tokens:
                node_expert_tokens[node] = []
                node_expert_ids[node] = set()
                node_weights[node] = []

            node_expert_tokens[node].extend(token_weights)
            node_expert_ids[node].add(expert_id)
            node_weights[node].extend([w for _, w in token_weights])

        # Build requests
        for node_id, tokens in node_expert_tokens.items():
            token_indices = [idx for idx, _ in tokens]
            weights = [w for _, w in tokens]

            # Gather hidden states for these tokens
            node_hidden = hidden_states.view(-1, hidden_states.shape[-1])[token_indices]

            requests_by_node[node_id] = MoEForwardRequest(
                node_id=node_id,
                hidden_states=node_hidden,
                expert_ids=list(node_expert_ids[node_id]),
                routing_weights=weights,
                request_id=f"moe-{node_id}",
            )

        return requests_by_node

    def dispatch(
        self,
        requests: Dict[str, MoEForwardRequest],
        node_clients: Dict[str, object],
    ) -> Dict[str, MoEForwardResponse]:
        """Dispatch expert requests to nodes in parallel.

        Args:
            requests: Dict mapping node_id to MoEForwardRequest.
            node_clients: Dict mapping node_id to gRPC client.

        Returns:
            Dict mapping node_id to MoEForwardResponse.
        """
        responses: Dict[str, MoEForwardResponse] = {}

        for node_id, request in requests.items():
            client = node_clients.get(node_id)
            if client is None:
                responses[node_id] = MoEForwardResponse(
                    node_id=node_id,
                    output=torch.zeros(1),
                    success=False,
                    error_message=f"No client for node {node_id}",
                )
                continue

            # In production, this would be an async dispatch via asyncio.gather
            # For now, sequential dispatch
            try:
                # Construct proto request (placeholder — would use actual proto)
                response = self._send_moe_forward(client, request)
                responses[node_id] = response

                if self.expert_registry:
                    self.expert_registry.record_request(node_id)
            except Exception as e:
                responses[node_id] = MoEForwardResponse(
                    node_id=node_id,
                    output=torch.zeros(1),
                    success=False,
                    error_message=str(e),
                )

        return responses

    def aggregate(
        self,
        responses: Dict[str, MoEForwardResponse],
        routing_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Combine expert outputs weighted by routing weights.

        Args:
            responses: Dict mapping node_id to MoEForwardResponse.
            routing_weights: Optional routing weights tensor.

        Returns:
            Aggregated output tensor.
        """
        successful = [r for r in responses.values() if r.success]
        if not successful:
            raise RuntimeError(
                f"All expert requests failed: "
                f"{[r.error_message for r in responses.values()]}"
            )

        # Simple aggregation: sum outputs (weighted sum in production)
        outputs = [r.output for r in successful]
        if len(outputs) == 1:
            return outputs[0]

        # Pad and sum
        max_dim = max(o.shape[-1] if o.dim() > 0 else 1 for o in outputs)
        result = torch.zeros_like(outputs[0])
        for o in outputs:
            if o.shape == result.shape:
                result = result + o
            else:
                # Handle shape mismatch
                result = result + torch.nn.functional.pad(
                    o, (0, max_dim - o.shape[-1])
                )

        return result

    def forward(
        self,
        hidden_states: torch.Tensor,
        moe_router,
        node_clients: Dict[str, object],
    ) -> torch.Tensor:
        """Full MoE forward pass: route → dispatch → aggregate.

        Args:
            hidden_states: Input tensor.
            moe_router: MoERouter instance.
            node_clients: Dict mapping node_id to gRPC client.

        Returns:
            Aggregated expert output tensor.
        """
        requests = self.route(hidden_states, moe_router)
        if not requests:
            # No expert routing needed
            return hidden_states

        responses = self.dispatch(requests, node_clients)
        return self.aggregate(responses)

    def _send_moe_forward(self, client, request: MoEForwardRequest) -> MoEForwardResponse:
        """Send MoE forward request to a node (placeholder).

        In production, this constructs the proto message and calls the RPC.
        """
        # Placeholder — actual implementation uses node_pb2.MoEForwardRequest
        raise NotImplementedError("MoE RPC not yet implemented in proto layer")
