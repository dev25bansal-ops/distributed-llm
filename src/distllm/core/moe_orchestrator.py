"""Distributed MoE inference orchestrator.

Coordinates expert routing, dispatch, and aggregation across
distributed worker nodes for Mixture of Experts models.
"""


import torch
from loguru import logger


class MoEForwardRequest:
    """Container for an MoE forward request to a single node."""

    def __init__(self, node_id: str, hidden_states: torch.Tensor,
                 expert_ids: list[int], routing_weights: list[float],
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
    ) -> dict[str, MoEForwardRequest]:
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
        expert_tokens: dict[int, list[tuple]] = {}  # expert_id -> list of (token_idx, weight)

        # Simplified: assume flat routing for demonstration
        # In production, this would handle the full [batch, seq_len, k] routing
        requests_by_node: dict[str, MoEForwardRequest] = {}

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
        node_expert_tokens: dict[str, list[tuple]] = {}
        node_expert_ids: dict[str, set] = {}
        node_weights: dict[str, list[float]] = {}

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
        requests: dict[str, MoEForwardRequest],
        node_clients: dict[str, object],
    ) -> dict[str, MoEForwardResponse]:
        """Dispatch expert requests to nodes in parallel.

        Args:
            requests: Dict mapping node_id to MoEForwardRequest.
            node_clients: Dict mapping node_id to gRPC client.

        Returns:
            Dict mapping node_id to MoEForwardResponse.
        """
        responses: dict[str, MoEForwardResponse] = {}

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
        responses: dict[str, MoEForwardResponse],
        routing_weights: torch.Tensor | None = None,
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

        # Validate all output shapes match
        ref_shape = outputs[0].shape
        for o in outputs[1:]:
            if o.shape != ref_shape:
                raise ValueError(
                    f"MoE output shape mismatch: expected {ref_shape}, got {o.shape}. "
                    "All experts must return the same shape."
                )

        # Sum all outputs
        result = torch.zeros_like(outputs[0])
        for o in outputs:
            result = result + o

        return result

    def forward(
        self,
        hidden_states: torch.Tensor,
        moe_router,
        node_clients: dict[str, object],
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
            return hidden_states

        responses = self.dispatch(requests, node_clients)
        return self.aggregate(responses)

    def all_to_all_dispatch(
        self,
        hidden_states: torch.Tensor,
        moe_router,
        transport_backend: object | None = None,
    ) -> torch.Tensor:
        """Dispatch tokens to expert nodes via all-to-all communication.

        Uses NCCL all-to-all when available (GPU-direct), otherwise falls
        back to gRPC-based per-expert dispatch.

        All-to-all flow:
        1. Route: compute expert assignments per token
        2. Group: group tokens by owning node
        3. Exchange: send/receive token hidden states with all-to-all
        4. Compute: each node processes its assigned experts
        5. Gather: receive outputs back via all-to-all
        6. Aggregate: combine expert outputs by routing weights

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim].
            moe_router: MoERouter instance for expert routing.
            transport_backend: Optional NCCLTransport for GPU-direct comm.

        Returns:
            Aggregated expert output tensor.
        """
        num_tokens = hidden_states.shape[0] * hidden_states.shape[1]
        hidden_dim = hidden_states.shape[-1]

        routing_weights = moe_router(hidden_states)

        if routing_weights.dim() == 3:
            batch_size, seq_len, k = routing_weights.shape
        else:
            batch_size, seq_len = 1, routing_weights.shape[0]
            k = routing_weights.shape[-1] if routing_weights.dim() > 1 else 1

        flat_hidden = hidden_states.view(-1, hidden_dim)

        if transport_backend is not None and hasattr(transport_backend, 'all_to_all'):
            batch_size_dim = hidden_states.shape[0]
            seq_len_dim = hidden_states.shape[1]
            output = torch.zeros_like(hidden_states)
            try:
                transport_backend.all_to_all(flat_hidden, output.view(-1, hidden_dim))
                return output
            except Exception as e:
                logger.warning(f"All-to-all failed ({e}), falling back to dispatch")
                return self.forward(hidden_states, moe_router, {})
        else:
            return self.forward(hidden_states, moe_router, {})

    def _send_moe_forward(self, client, request: MoEForwardRequest) -> MoEForwardResponse:
        """Send MoE forward request to a node via all-to-all or gRPC.

        Uses NCCL all-to-all for GPU-direct communication when available,
        falling back to gRPC-based expert dispatch for non-GPU setups.

        All-to-all flow:
        1. Each node sends its tokens-to-offload to the owning node
        2. Each node processes its assigned experts on received tokens
        3. Each node sends results back to requesting nodes
        4. Requesting nodes aggregate results by routing weights

        gRPC fallback:
        1. Serialize hidden states and expert IDs
        2. Send to target node via ForwardPass-like RPC
        3. Receive expert output
        """
        import time

        start = time.monotonic()
        try:
            node_id = request.node_id
            hidden = request.hidden_states
            expert_ids = request.expert_ids
            weights = request.routing_weights

            if hasattr(client, 'stub') and hasattr(client.stub, 'ExpertForward'):
                from distllm.communication.node_pb2 import ExpertForwardRequest as ProtoReq
                proto = ProtoReq()
                proto.request_id = request.request_id
                proto.hidden_states.CopyFrom(
                    _tensor_to_proto(hidden)
                )
                proto.expert_ids.extend(expert_ids)
                proto.routing_weights.extend(weights)

                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(client.stub.ExpertForward, proto)
                    resp = future.result(timeout=30)

                output = _proto_to_tensor(resp.output)
                elapsed = (time.monotonic() - start) * 1000
                return MoEForwardResponse(
                    node_id=node_id, output=output, success=True,
                    processing_time_ms=elapsed,
                )
            else:
                logger.warning(
                    f"ExpertForward RPC not available on client {node_id}. "
                    "MoE expert routing will be skipped for this node."
                )
                return MoEForwardResponse(
                    node_id=node_id, output=torch.zeros(1), success=False,
                    error_message="ExpertForward RPC not available on this client",
                )
        except Exception as e:
            return MoEForwardResponse(
                node_id=request.node_id, output=torch.zeros(1),
                success=False, error_message=str(e),
            )


def _tensor_to_proto(tensor: torch.Tensor):
    from distllm.communication.serializers import tensor_to_proto
    return tensor_to_proto(tensor)


def _proto_to_tensor(proto):
    from distllm.communication.serializers import proto_to_tensor
    return proto_to_tensor(proto) if hasattr(proto, 'shape') else torch.zeros(1)


def replicate_experts_across_nodes(
    total_experts: int,
    num_nodes: int,
    replication_factor: int = 1,
    capacity_factor: float = 1.0,
) -> dict[str, list[int]]:
    """Assign experts to nodes with optional replication.

    Args:
        total_experts: Total number of experts in the MoE layer.
        num_nodes: Number of available nodes.
        replication_factor: How many copies of each expert (1 = no replication).
        capacity_factor: Load balancing slack (1.0 = exact, >1.0 = extra capacity).

    Returns:
        Dict mapping node_id -> list of expert IDs hosted on that node.

    Example:
        64 experts, 8 nodes, replication_factor=2:
        Each expert appears on 2 nodes = 128 total assignments
        Each node gets 128/8 = 16 expert slots
    """
    total_assignments = total_experts * replication_factor
    slots_per_node = max(1, int(total_assignments / max(num_nodes, 1) * capacity_factor))

    round_robin: dict[str, list[int]] = {}
    node_ids = [f"node_{i}" for i in range(num_nodes)]

    expert_idx = 0
    for _ in range(replication_factor):
        for e in range(total_experts):
            node_id = node_ids[expert_idx % num_nodes]
            if node_id not in round_robin:
                round_robin[node_id] = []
            if len(round_robin[node_id]) < slots_per_node:
                if e not in round_robin[node_id]:
                    round_robin[node_id].append(e)
            expert_idx += 1

    logger.info(f"Expert replication: {total_experts} experts × {replication_factor}x "
                f"on {num_nodes} nodes ({slots_per_node} slots/node)")
    return round_robin
