"""Pipeline orchestrator for distributed LLM inference.

Manages node topology, layer routing, and distributed pipeline execution.
Extracted from the Coordinator class.
"""

import concurrent.futures
from typing import Any

import grpc
import torch
from loguru import logger

from distllm.communication.grpc import is_debug_mode
from distllm.communication.node_pb2 import ForwardPassRequest
from distllm.communication.serializers import (
    dequantize_activation,
    kv_cache_to_proto,
    proto_to_tensor,
    quantize_activation,
    tensor_to_proto,
)
from distllm.config.loader import NodeRole
from distllm.core.kv_cache import KVCache
from distllm.core.latency_tracker import LatencyTracker
from distllm.core.resource_manager import NodeRegistration, ResourceManager
from distllm.errors.types import (
    ConfigValidationError,
    GRPCTimeoutError,
    InputValidationError,
    NodeUnreachableError,
    OOMError,
)


class PipelineOrchestrator:
    """Manages distributed inference pipeline execution.

    Handles node registration, topology management, layer assignment validation,
    and running input through the distributed pipeline.

    Attributes:
        nodes: Dict of node_id -> NodeRegistration.
        node_order: Ordered list of node IDs by layer assignment.
        prefill_nodes: Nodes assigned to prefill role.
        decode_nodes: Nodes assigned to decode role.
        total_layers: Total number of model layers.
        resource_mgr: ResourceManager for health and circuit breaking.
    """

    def __init__(
        self,
        resource_mgr: ResourceManager | None = None,
        total_layers: int = 0,
        max_workers: int | None = None,
    ):
        self.nodes: dict[str, NodeRegistration] = {}
        self.node_order: list[str] = []
        self.prefill_nodes: dict[str, NodeRegistration] = {}
        self.decode_nodes: dict[str, NodeRegistration] = {}
        self.total_layers = total_layers
        self.resource_mgr = resource_mgr or ResourceManager()
        self._latency_tracker: LatencyTracker | None = None
        self._max_workers = max_workers or 4
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers)
        self.enable_overlap: bool = False
        # Track last KV length sent per (node_id, request_id) for delta-only gRPC
        self._node_kv_sent_lens: dict[tuple[str, str], int] = {}

    def set_latency_tracker(self, tracker: LatencyTracker) -> None:
        """Set the latency tracker for rebalancer integration."""
        self._latency_tracker = tracker

    def register_node(
        self,
        node_id: str,
        host: str,
        port: int,
        start_layer: int,
        end_layer: int,
        role: NodeRole = NodeRole.AUTO,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        use_tls: bool = True,
        ca_cert: str | None = None,
        expert_ids: list[int] | None = None,
        cluster_id: str = "default",
    ) -> None:
        """Register a new node in the pipeline.

        Args:
            node_id: Unique node identifier.
            host: Node hostname.
            port: Node gRPC port.
            start_layer: First layer index this node handles.
            end_layer: Last layer index this node handles.
            role: Node role (AUTO, PREFILL, DECODE).
            max_retries: Max retry attempts for gRPC calls.
            retry_delay: Base delay between retries.
            use_tls: Whether to use TLS for gRPC.
            ca_cert: Path to CA certificate file.
            expert_ids: Expert IDs this node hosts for MoE inference.
        """
        self.validate_layer_assignment(node_id, start_layer, end_layer)

        registration = NodeRegistration(
            node_id=node_id,
            host=host,
            port=port,
            start_layer=start_layer,
            end_layer=end_layer,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_tls=use_tls,
            ca_cert=ca_cert,
            role=role,
            expert_ids=expert_ids,
            cluster_id=cluster_id,
        )
        self.nodes[node_id] = registration

        self.node_order = sorted(
            self.nodes.keys(),
            key=lambda nid: self.nodes[nid].start_layer,
        )

        # Assign to role pools
        if role == NodeRole.PREFILL:
            self.prefill_nodes[node_id] = registration
        elif role == NodeRole.DECODE:
            self.decode_nodes[node_id] = registration

        # Auto-scale thread pool to match node count
        target_workers = max(self._max_workers, min(32, len(self.nodes) * 2))
        if target_workers != self._executor._max_workers:
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=target_workers)

        logger.info(f"Registered {node_id}: layers {start_layer}-{end_layer}")

    def group_nodes_into_stages(self, num_stages: int | None = None) -> list[list[str]]:
        """Group nodes into hierarchical stages to reduce pipeline depth.

        Nodes within a stage can run in parallel (tensor/data parallel),
        while stages run sequentially. This converts a deep N-hop pipeline
        into a shallow (N/stage_size)-hop hierarchical pipeline.

        Example: 16 nodes → 4 stages × 4 nodes each = 4 sequential hops.

        Args:
            num_stages: Target number of stages. If None, uses sqrt(N).

        Returns:
            List of stages, where each stage is a list of node_ids.
        """
        if not self.node_order:
            return []

        n = len(self.node_order)
        target = num_stages or max(1, int(n ** 0.5))
        stages: list[list[str]] = [[] for _ in range(target)]

        # Assign nodes to stages in round-robin by layer position
        # This keeps each stage's layer range contiguous
        for i, nid in enumerate(self.node_order):
            stage_idx = min(i * target // n, target - 1)
            stages[stage_idx].append(nid)

        # Filter empty stages
        stages = [s for s in stages if s]
        logger.info(f"Hierarchical pipeline: {n} nodes → {len(stages)} stages ({n // len(stages)} nodes/stage)")
        return stages

    def validate_layer_assignment(self, node_id: str, start_layer: int, end_layer: int) -> None:
        """Validate layer assignment for correctness.

        Checks:
        - Layers within model bounds (0 to total_layers-1)
        - No overlapping layer ranges with existing nodes
        - start_layer <= end_layer

        Args:
            node_id: Node identifier for error messages.
            start_layer: First layer index.
            end_layer: Last layer index.

        Raises:
            ConfigValidationError: If validation fails.
        """
        if self.total_layers <= 0:
            return  # Can't validate without total_layers

        if start_layer < 0 or end_layer >= self.total_layers:
            raise ConfigValidationError(
                field="layer_assignment",
                message=(
                    f"Node {node_id}: layers {start_layer}-{end_layer} out of bounds "
                    f"(model has {self.total_layers} layers, valid range: 0-{self.total_layers - 1})"
                ),
            )

        if start_layer > end_layer:
            raise ConfigValidationError(
                field="layer_assignment",
                message=f"Node {node_id}: start_layer ({start_layer}) > end_layer ({end_layer})",
            )

        for existing_id, existing in self.nodes.items():
            if max(start_layer, existing.start_layer) <= min(end_layer, existing.end_layer):
                raise ConfigValidationError(
                    field="layer_assignment",
                    message=(
                        f"Node {node_id}: layers {start_layer}-{end_layer} overlap with "
                        f"{existing_id} (layers {existing.start_layer}-{existing.end_layer})"
                    ),
                )

    def _prepare_forward_request(
        self,
        node_id: str,
        node: Any,
        is_first_node: bool,
        is_last_node: bool,
        seq_len: int,
        batch_size: int,
        current_hidden: torch.Tensor | None,
        past_kv: list | None,
        request_id: str,
        draft_tokens: list[int] | None,
        input_ids: torch.Tensor | None = None,
    ) -> ForwardPassRequest:
        cached_len = 0
        if past_kv is not None and len(past_kv) > 0:
            cached_len = past_kv[0][0].shape[-2]
        total_len = cached_len + seq_len
        node_attention_mask = torch.ones(batch_size, total_len, dtype=torch.long)
        node_position_ids = torch.arange(cached_len, total_len).unsqueeze(0)

        request = ForwardPassRequest(request_id=request_id, use_cache=True)
        if is_first_node and input_ids is not None:
            request.input_ids.extend(input_ids.squeeze(0).tolist())
        elif current_hidden is not None:
            hidden_to_send, scale = quantize_activation(current_hidden)
            proto_hidden = tensor_to_proto(hidden_to_send)
            if scale is not None:
                proto_hidden.scale.extend(scale.flatten().tolist())
            request.hidden_states.CopyFrom(proto_hidden)
        request.attention_mask.CopyFrom(tensor_to_proto(node_attention_mask))
        request.position_ids.CopyFrom(tensor_to_proto(node_position_ids))
        if past_kv is not None:
            cache = KVCache()
            kv_key = (node_id, request_id)
            prev_len = self._node_kv_sent_lens.get(kv_key, 0)
            current_len = past_kv[0][0].shape[-2] if len(past_kv) > 0 else 0
            if current_len > prev_len:
                delta_kv = [(k.narrow(-2, prev_len, current_len - prev_len).contiguous(),
                             v.narrow(-2, prev_len, current_len - prev_len).contiguous()) for k, v in past_kv]
                cache.set_all(delta_kv)
                request.kv_cache.CopyFrom(kv_cache_to_proto(cache))
                self._node_kv_sent_lens[kv_key] = current_len
        if is_last_node and draft_tokens:
            request.draft_tokens.extend(draft_tokens)
        return request

    def _process_forward_response(
        self,
        response: Any,
        node_id: str,
        node: Any,
        node_kv_caches: dict[str, list | None],
    ) -> torch.Tensor:
        if not response.success:
            self.resource_mgr.record_failure(node_id)
            node.healthy = False
            from distllm.communication.node_pb2 import ErrorCode
            if response.error_code == ErrorCode.OOM:
                raise OOMError(node_id=node_id, detail=response.error_message)
            elif response.error_code == ErrorCode.INVALID_INPUT:
                raise InputValidationError(detail=response.error_message)
            else:
                raise NodeUnreachableError(
                    node_id=node_id,
                    host=node.host,
                    port=node.port,
                    original_error=RuntimeError(response.error_message),
                )
        self.resource_mgr.record_success(node_id)
        if self._latency_tracker is not None and response.processing_time_ms > 0:
            self._latency_tracker.record(node_id, response.processing_time_ms)

        output_proto = response.output
        has_scale = bool(output_proto.scale)
        current_hidden = proto_to_tensor(output_proto)
        if has_scale:
            scale_tensor = torch.tensor(output_proto.scale, dtype=current_hidden.dtype, device=current_hidden.device)
            current_hidden = dequantize_activation(current_hidden, scale_tensor, current_hidden.dtype)
        if response.HasField('kv_cache') and response.kv_cache.layers:
            new_cache = KVCache()
            for layer in response.kv_cache.layers:
                k = proto_to_tensor(layer.key_states)
                v = proto_to_tensor(layer.value_states)
                new_cache.cache.append((k, v))
            node_kv_caches[node_id] = new_cache.cache
        return current_hidden

    def _find_fallback_node(self, failed_node_id: str, failed_node) -> tuple[str, Any] | None:
        """Find a healthy fallback node that can handle the same layer range."""
        for nid, node in self.nodes.items():
            if nid == failed_node_id:
                continue
            if not node.healthy:
                continue
            if self.resource_mgr.check_circuit_breaker(nid):
                continue
            if node.start_layer == failed_node.start_layer and node.end_layer == failed_node.end_layer:
                return nid, node
            # Allow partial overlap: fallback covers at least the failed node's range
            if node.start_layer <= failed_node.start_layer and node.end_layer >= failed_node.end_layer:
                return nid, node
        return None

    def run_pipeline(
        self,
        input_ids: torch.Tensor,
        node_kv_caches: dict[str, list | None],
        request_id: str,
        draft_tokens: list[int] | None = None,
    ) -> torch.Tensor:
        """Run input through all nodes via gRPC.

        First node receives input_ids (embeds them internally).
        Middle/last nodes receive hidden_states (activations).

        Args:
            input_ids: Input token IDs tensor.
            node_kv_caches: Dict of node_id -> KV cache (None for first step).
            request_id: Unique request identifier.
            draft_tokens: Optional draft tokens for speculative decoding (sent to last node).

        Returns:
            Logits or hidden states from the last node.

        Raises:
            NodeUnreachableError: If a node fails.
        """
        seq_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        current_hidden = None
        total_nodes = len(self.node_order)

        if is_debug_mode():
            logger.debug(f"[pipeline] Starting pipeline: input_ids shape={input_ids.shape}, batch_size={batch_size}, seq_len={seq_len}")

        for i, node_id in enumerate(self.node_order):
            node = self.nodes[node_id]
            past_kv = node_kv_caches.get(node_id)
            is_first_node = (i == 0)
            is_last_node = (i == total_nodes - 1)
            node_log = logger.bind(request_id=request_id, node_id=node_id)

            if is_debug_mode():
                layer_info = f"layers {node.start_layer}-{node.end_layer}"
                if is_first_node:
                    logger.debug(f"[pipeline] Node {node_id} ({layer_info}): first node, sending input_ids")
                else:
                    logger.debug(f"[pipeline] Node {node_id} ({layer_info}): hidden_states shape={current_hidden.shape}")
                if past_kv is not None and len(past_kv) > 0:
                    cache_len = past_kv[0][0].shape[-2]
                    logger.debug(f"[pipeline] Node {node_id}: KV cache seq_len={cache_len}")

            if self.resource_mgr.check_circuit_breaker(node_id):
                fallback = self._find_fallback_node(node_id, node)
                if fallback is not None:
                    fallback_id, fallback_node = fallback
                    node_log.warning(f"Circuit breaker open, falling back to {fallback_id}")
                    node_id = fallback_id
                    node = fallback_node
                    past_kv = node_kv_caches.get(node_id)
                    node_log = logger.bind(request_id=request_id, node_id=node_id)
                else:
                    node_log.warning("Circuit breaker open, no fallback available, skipping")
                    node.healthy = False
                    continue

            try:
                request = self._prepare_forward_request(
                    node_id, node, is_first_node, is_last_node,
                    seq_len, batch_size, current_hidden, past_kv,
                    request_id, draft_tokens, input_ids,
                )

                response = self._executor.submit(node.client.stub.ForwardPass, request).result()
                current_hidden = self._process_forward_response(response, node_id, node, node_kv_caches)

                if is_debug_mode():
                    logger.debug(f"[pipeline] Node {node_id}: output shape={current_hidden.shape}")

            except (NodeUnreachableError, OOMError, InputValidationError, GRPCTimeoutError):
                raise
            except grpc.RpcError as e:
                self.resource_mgr.record_failure(node_id)
                node_log.error(f"Pipeline gRPC error: {e}")
                node.healthy = False
                raise NodeUnreachableError(
                    node_id=node_id, host=node.host, port=node.port, original_error=e
                )
            except Exception as e:
                self.resource_mgr.record_failure(node_id)
                node_log.error(f"Pipeline failed: {e}")
                node.healthy = False
                raise

        return current_hidden

    async def run_pipeline_async(
        self,
        input_ids: torch.Tensor,
        node_kv_caches: dict[str, list | None],
        request_id: str,
        draft_tokens: list[int] | None = None,
    ) -> torch.Tensor:
        """Run input through all nodes via async gRPC.

        First node receives input_ids (embeds them internally).
        Middle/last nodes receive hidden_states (activations).

        Args:
            input_ids: Input token IDs tensor.
            node_kv_caches: Dict of node_id -> KV cache (None for first step).
            request_id: Unique request identifier.
            draft_tokens: Optional draft tokens for speculative decoding (sent to last node).

        Returns:
            Logits or hidden states from the last node.

        Raises:
            NodeUnreachableError: If a node fails.
        """
        import asyncio

        seq_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        current_hidden = None
        total_nodes = len(self.node_order)

        if is_debug_mode():
            logger.debug(f"[pipeline] Starting async pipeline: input_ids shape={input_ids.shape}, batch_size={batch_size}, seq_len={seq_len}")

        for i, node_id in enumerate(self.node_order):
            node = self.nodes[node_id]
            past_kv = node_kv_caches.get(node_id)
            is_first_node = (i == 0)
            is_last_node = (i == total_nodes - 1)
            node_log = logger.bind(request_id=request_id, node_id=node_id)

            if is_debug_mode():
                layer_info = f"layers {node.start_layer}-{node.end_layer}"
                if is_first_node:
                    logger.debug(f"[pipeline] Node {node_id} ({layer_info}): first node, sending input_ids")
                else:
                    logger.debug(f"[pipeline] Node {node_id} ({layer_info}): hidden_states shape={current_hidden.shape}")
                if past_kv is not None and len(past_kv) > 0:
                    cache_len = past_kv[0][0].shape[-2]
                    logger.debug(f"[pipeline] Node {node_id}: KV cache seq_len={cache_len}")

            if self.resource_mgr.check_circuit_breaker(node_id):
                fallback = self._find_fallback_node(node_id, node)
                if fallback is not None:
                    fallback_id, fallback_node = fallback
                    node_log.warning(f"Circuit breaker open, falling back to {fallback_id}")
                    node_id = fallback_id
                    node = fallback_node
                    past_kv = node_kv_caches.get(node_id)
                    node_log = logger.bind(request_id=request_id, node_id=node_id)
                else:
                    node_log.warning("Circuit breaker open, no fallback available, skipping")
                    node.healthy = False
                    continue

            try:
                request = self._prepare_forward_request(
                    node_id, node, is_first_node, is_last_node,
                    seq_len, batch_size, current_hidden, past_kv,
                    request_id, draft_tokens, input_ids,
                )

                if hasattr(node, 'async_client') and node.async_client is not None:
                    response = await node.async_client.stub.ForwardPass(request)
                else:
                    response = await asyncio.to_thread(node.client.stub.ForwardPass, request)

                current_hidden = self._process_forward_response(response, node_id, node, node_kv_caches)

                if is_debug_mode():
                    logger.debug(f"[pipeline] Node {node_id}: output shape={current_hidden.shape}")

            except (NodeUnreachableError, OOMError, InputValidationError, GRPCTimeoutError):
                raise
            except grpc.RpcError as e:
                self.resource_mgr.record_failure(node_id)
                node_log.error(f"Pipeline gRPC error: {e}")
                node.healthy = False
                raise NodeUnreachableError(
                    node_id=node_id, host=node.host, port=node.port, original_error=e
                )
            except Exception as e:
                self.resource_mgr.record_failure(node_id)
                node_log.error(f"Pipeline failed: {e}")
                node.healthy = False
                raise

        return current_hidden

    def run_pipeline_overlap(
        self,
        input_ids: torch.Tensor,
        node_kv_caches: dict[str, list | None],
        request_id: str,
        draft_tokens: list[int] | None = None,
    ) -> torch.Tensor:
        """Run pipeline with compute/communication overlap."""
        seq_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        current_hidden = None
        total_nodes = len(self.node_order)

        if total_nodes == 0:
            raise RuntimeError("No nodes registered in pipeline")

        if is_debug_mode():
            logger.debug(f"[pipeline/overlap] Starting overlapped pipeline: input_ids shape={input_ids.shape}, nodes={total_nodes}")

        first_node_id = self.node_order[0]
        first_node = self.nodes[first_node_id]
        first_past_kv = node_kv_caches.get(first_node_id)

        req0 = self._prepare_forward_request(
            first_node_id, first_node, True, (total_nodes == 1),
            seq_len, batch_size, None, first_past_kv,
            request_id, draft_tokens, input_ids,
        )

        future0 = self._executor.submit(first_node.client.stub.ForwardPass, req0)
        resp0 = future0.result()
        current_hidden = self._process_forward_response(resp0, first_node_id, first_node, node_kv_caches)

        for i in range(1, total_nodes):
            node_id = self.node_order[i]
            node = self.nodes[node_id]
            past_kv = node_kv_caches.get(node_id)
            is_last = (i == total_nodes - 1)
            node_log = logger.bind(request_id=request_id, node_id=node_id)

            if self.resource_mgr.check_circuit_breaker(node_id):
                fallback = self._find_fallback_node(node_id, node)
                if fallback is not None:
                    fallback_id, fallback_node = fallback
                    node_log.warning(f"Circuit breaker open (overlap), falling back to {fallback_id}")
                    node_id = fallback_id
                    node = fallback_node
                    past_kv = node_kv_caches.get(node_id)
                    node_log = logger.bind(request_id=request_id, node_id=node_id)
                else:
                    node_log.warning("Circuit breaker open (overlap), no fallback, skipping")
                    node.healthy = False
                    continue

            req = self._prepare_forward_request(
                node_id, node, False, is_last,
                seq_len, batch_size, current_hidden, past_kv,
                request_id, draft_tokens if is_last else None, None,
            )

            next_req = None
            next_node_id = None
            if i + 1 < total_nodes:
                next_node_id = self.node_order[i + 1]
                next_node = self.nodes[next_node_id]
                next_past_kv = node_kv_caches.get(next_node_id)
                next_req = self._prepare_forward_request(
                    next_node_id, next_node, False,
                    (i + 1 == total_nodes - 1),
                    seq_len, batch_size, current_hidden, next_past_kv,
                    request_id, None, None,
                )

            resp_future = self._executor.submit(node.client.stub.ForwardPass, req)

            if next_req is not None and next_node_id is not None:
                hidden_copy = current_hidden.detach().clone()
                self._executor.submit(self._overlap_forward, next_req, hidden_copy, next_node_id)

            response = resp_future.result()
            current_hidden = self._process_forward_response(response, node_id, node, node_kv_caches)

            if is_debug_mode():
                logger.debug(f"[pipeline/overlap] Node {node_id}: output shape={current_hidden.shape}")

        return current_hidden

    def _overlap_forward(self, request: Any, hidden_states: torch.Tensor, node_id: str) -> None:
        """Update request with hidden states and forward concurrently with current node."""
        node = self.nodes[node_id]
        try:
            hidden_to_send, scale = quantize_activation(hidden_states)
            proto_hidden = tensor_to_proto(hidden_to_send)
            if scale is not None:
                proto_hidden.scale.extend(scale.flatten().tolist())
            request.hidden_states.CopyFrom(proto_hidden)
            node.client.stub.ForwardPass(request)
        except Exception as e:
            logger.warning(f"Overlapped forward to {node_id} failed: {e}")
            self.resource_mgr.record_failure(node_id)
            node.healthy = False
