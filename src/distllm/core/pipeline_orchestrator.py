"""Pipeline orchestrator for distributed LLM inference.

Manages node topology, layer routing, and distributed pipeline execution.
Extracted from the Coordinator class.
"""

import asyncio
import concurrent.futures
import os
import threading
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
from distllm.communication.transport import TensorTransport, TransportBackend
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
        self._max_workers = max_workers or os.cpu_count() or 4
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers)
        self.enable_overlap: bool = False
        # Track last KV length sent per (node_id, request_id) for delta-only gRPC
        self._node_kv_sent_lens: dict[tuple[str, str], int] = {}

        # Tensor transport (NCCL for GPU-direct, gRPC fallback)
        self._transport: TensorTransport | None = None
        self._transport_rank_map: dict[str, int] = {}  # node_id -> rank

        # Thread-safe topology state (nodes, node_order, role pools)
        self._topology_lock = threading.Lock()
        self._node_kv_lock = threading.Lock()

        # Cached tensors for forward request preparation
        self._cached_ones: dict[int, torch.Tensor] = {}
        self._cached_arange: dict[int, torch.Tensor] = {}

    def set_tensor_transport(
        self,
        transport: TensorTransport,
        node_rank_map: dict[str, int] | None = None,
    ) -> None:
        """Set the tensor transport backend for GPU-direct transfers.

        When NCCL transport is enabled, hidden_states are transferred
        directly between GPUs without serialization. gRPC is still used
        for control plane metadata.

        Args:
            transport: TensorTransport instance (NCCL or gRPC backend).
            node_rank_map: Mapping from node_id to NCCL rank.
        """
        self._transport = transport
        self._transport_rank_map = node_rank_map or {}
        logger.info(f"Tensor transport set: {transport.backend.value} (available: {transport.is_available})")

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
        use_tls: bool = False,
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

        with self._topology_lock:
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
                old_executor = self._executor
                self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=target_workers)
                old_executor.shutdown(wait=True)

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

        with self._topology_lock:
            node_order_copy = list(self.node_order)
        n = len(node_order_copy)
        target = num_stages or max(1, int(n ** 0.5))
        stages: list[list[str]] = [[] for _ in range(target)]

        # Assign nodes to stages in round-robin by layer position
        for i, nid in enumerate(node_order_copy):
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

        with self._topology_lock:
            nodes_snapshot = dict(self.nodes)
        for existing_id, existing in nodes_snapshot.items():
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
        if total_len not in self._cached_ones:
            self._cached_ones[total_len] = torch.ones(1, total_len, dtype=torch.long)
        if total_len not in self._cached_arange:
            self._cached_arange[total_len] = torch.arange(0, total_len).unsqueeze(0)
        node_attention_mask = self._cached_ones[total_len].expand(batch_size, -1)
        node_position_ids = self._cached_arange[total_len][:, cached_len:]

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
            with self._node_kv_lock:
                prev_len = self._node_kv_sent_lens.get(kv_key, 0)
            current_len = past_kv[0][0].shape[-2] if len(past_kv) > 0 else 0
            if current_len > prev_len:
                delta_kv = [(k.narrow(-2, prev_len, current_len - prev_len).contiguous(),
                             v.narrow(-2, prev_len, current_len - prev_len).contiguous()) for k, v in past_kv]
                cache.set_all(delta_kv)
                request.kv_cache.CopyFrom(kv_cache_to_proto(cache))
                with self._node_kv_lock:
                    self._node_kv_sent_lens[kv_key] = current_len
        if is_last_node and draft_tokens:
            request.draft_tokens.extend(draft_tokens)
        return request

    def _prepare_control_request(
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
        transport_mode: str = "nccl",
    ) -> ForwardPassRequest:
        """Prepare control-plane request when NCCL handles tensor data.

        Only sends metadata (request_id, config, KV cache deltas).
        Tensor data (hidden_states) is transferred via NCCL separately.
        """
        cached_len = 0
        if past_kv is not None and len(past_kv) > 0:
            cached_len = past_kv[0][0].shape[-2]
        total_len = cached_len + seq_len
        if total_len not in self._cached_ones:
            self._cached_ones[total_len] = torch.ones(1, total_len, dtype=torch.long)
        if total_len not in self._cached_arange:
            self._cached_arange[total_len] = torch.arange(0, total_len).unsqueeze(0)
        node_attention_mask = self._cached_ones[total_len].expand(batch_size, -1)
        node_position_ids = self._cached_arange[total_len][:, cached_len:]

        request = ForwardPassRequest(request_id=request_id, use_cache=True)
        # Don't send hidden_states - transferred via NCCL
        request.attention_mask.CopyFrom(tensor_to_proto(node_attention_mask))
        request.position_ids.CopyFrom(tensor_to_proto(node_position_ids))

        # KV cache still sent via gRPC (smaller than hidden states)
        if past_kv is not None:
            from distllm.core.kv_cache import KVCache
            cache = KVCache()
            kv_key = (node_id, request_id)
            with self._node_kv_lock:
                prev_len = self._node_kv_sent_lens.get(kv_key, 0)
            current_len = past_kv[0][0].shape[-2] if len(past_kv) > 0 else 0
            if current_len > prev_len:
                delta_kv = [(k.narrow(-2, prev_len, current_len - prev_len).contiguous(),
                             v.narrow(-2, prev_len, current_len - prev_len).contiguous()) for k, v in past_kv]
                cache.set_all(delta_kv)
                request.kv_cache.CopyFrom(kv_cache_to_proto(cache))
                with self._node_kv_lock:
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
        has_scale = bool(getattr(output_proto, "scale", None))
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
        with self._topology_lock:
            nodes_snapshot = dict(self.nodes)
        for nid, node in nodes_snapshot.items():
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

        with self._topology_lock:
            node_order_snapshot = list(self.node_order)
            nodes_snapshot = dict(self.nodes)
        total_nodes = len(node_order_snapshot)

        if is_debug_mode():
            logger.debug(f"[pipeline] Starting pipeline: input_ids shape={input_ids.shape}, batch_size={batch_size}, seq_len={seq_len}")

        # Pipeline-level circuit breaker: short-circuit if any downstream node is broken
        for node_id in node_order_snapshot:
            node = nodes_snapshot[node_id]
            if self.resource_mgr.check_circuit_breaker(node_id):
                fallback = self._find_fallback_node(node_id, node)
                if fallback is not None:
                    continue
                logger.warning(
                    f"Pipeline short-circuit: node {node_id} circuit breaker open, "
                    f"no fallback available"
                )
                raise NodeUnreachableError(
                    node_id=node_id, host=node.host, port=node.port,
                    original_error=Exception(f"Circuit breaker open for {node_id}"),
                )

        for i, node_id in enumerate(node_order_snapshot):
            node = nodes_snapshot[node_id]
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
                    with self._topology_lock:
                        if node_id in self.nodes:
                            self.nodes[node_id].healthy = False
                    continue

            # Use NCCL transport for hidden_states if available
            use_nccl = (
                self._transport is not None
                and self._transport.is_available
                and self._transport.backend == TransportBackend.NCCL
                and current_hidden is not None
                and not is_first_node
            )

            if use_nccl:
                # Send hidden_states via NCCL (GPU-direct)
                dst_rank = self._transport_rank_map.get(node_id, i)
                self._transport.send_tensor(current_hidden, dst_rank=dst_rank)

                # Send control metadata via gRPC (request_id, config, etc.)
                request = self._prepare_control_request(
                    node_id, node, is_first_node, is_last_node,
                    seq_len, batch_size, current_hidden, past_kv,
                    request_id, draft_tokens,
                    transport_mode="nccl",
                )
                self._executor.submit(node.client.stub.ForwardControl, request).result()

                # Receive output via NCCL
                response_shape = current_hidden.shape
                current_hidden = self._transport.recv_tensor(
                    src_rank=dst_rank,
                    shape=response_shape,
                    dtype=current_hidden.dtype,
                    device=str(current_hidden.device),
                )
            else:
                current_hidden = self._execute_node_grpc(
                    node_id, node, node_kv_caches,
                    is_first_node, is_last_node,
                    seq_len, batch_size, current_hidden,
                    request_id, draft_tokens, input_ids,
                )

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

        Pipeline improvements over basic sequential execution:
        - Pipeline-level circuit breaker: short-circuits if a downstream node is broken

        Args:
            input_ids: Input token IDs tensor.
            node_kv_caches: Dict of node_id -> KV cache (None for first step).
            request_id: Unique request identifier.
            draft_tokens: Optional draft tokens for speculative decoding (sent to last node).

        Returns:
            Logits or hidden states from the last node.

        Raises:
            NodeUnreachableError: If a node fails and no fallback exists.
        """
        seq_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]

        with self._topology_lock:
            node_order_snapshot = list(self.node_order)
            nodes_snapshot = dict(self.nodes)
        total_nodes = len(node_order_snapshot)

        if total_nodes == 0:
            raise RuntimeError("No nodes registered in pipeline")

        if is_debug_mode():
            logger.debug(f"[pipeline] Starting async pipeline: input_ids shape={input_ids.shape}, batch_size={batch_size}, seq_len={seq_len}")

        # Pipeline-level circuit breaker: check all nodes upfront and short-circuit
        # if any downstream node is broken with no fallback
        for node_id in node_order_snapshot:
            node = nodes_snapshot[node_id]
            if self.resource_mgr.check_circuit_breaker(node_id):
                fallback = self._find_fallback_node(node_id, node)
                if fallback is not None:
                    continue
                logger.warning(
                    f"Pipeline short-circuit: node {node_id} circuit breaker open, "
                    f"no fallback available"
                )
                raise NodeUnreachableError(
                    node_id=node_id, host=node.host, port=node.port,
                    original_error=Exception(f"Circuit breaker open for {node_id}"),
                )

        current_hidden: torch.Tensor | None = None

        for i, node_id in enumerate(node_order_snapshot):
            node = nodes_snapshot[node_id]
            past_kv = node_kv_caches.get(node_id)
            is_first_node = (i == 0)
            is_last_node = (i == total_nodes - 1)

            node_log = logger.bind(request_id=request_id, node_id=node_id)

            if is_debug_mode():
                layer_info = f"layers {node.start_layer}-{node.end_layer}"
                if is_first_node:
                    node_log.debug(f"[pipeline] Node {node_id} ({layer_info}): first node, sending input_ids")
                elif current_hidden is not None:
                    node_log.debug(f"[pipeline] Node {node_id} ({layer_info}): hidden_states shape={current_hidden.shape}")
                if past_kv is not None and len(past_kv) > 0:
                    node_log.debug(f"[pipeline] Node {node_id}: KV cache seq_len={past_kv[0][0].shape[-2]}")

            # Circuit breaker per-node with fallback
            if self.resource_mgr.check_circuit_breaker(node_id):
                fallback = self._find_fallback_node(node_id, node)
                if fallback is not None:
                    fallback_id, fallback_node = fallback
                    node_log.warning(f"Circuit breaker open, falling back to {fallback_id}")
                    node_id = fallback_id
                    node = fallback_node
                    past_kv = node_kv_caches.get(fallback_id)
                    node_log = logger.bind(request_id=request_id, node_id=fallback_id)
                else:
                    node_log.warning("Circuit breaker open, no fallback available, skipping")
                    with self._topology_lock:
                        if node_id in self.nodes:
                            self.nodes[node_id].healthy = False
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

            except (NodeUnreachableError, OOMError, InputValidationError, GRPCTimeoutError):
                raise
            except grpc.RpcError as e:
                self.resource_mgr.record_failure(node_id)
                node_log.error(f"Pipeline gRPC error: {e}")
                with self._topology_lock:
                    if node_id in self.nodes:
                        self.nodes[node_id].healthy = False
                raise NodeUnreachableError(
                    node_id=node_id, host=node.host, port=node.port, original_error=e
                )
            except Exception as e:
                self.resource_mgr.record_failure(node_id)
                node_log.error(f"Pipeline failed: {e}")
                with self._topology_lock:
                    if node_id in self.nodes:
                        self.nodes[node_id].healthy = False
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

        with self._topology_lock:
            node_order_snapshot = list(self.node_order)
            nodes_snapshot = dict(self.nodes)
        total_nodes = len(node_order_snapshot)

        if total_nodes == 0:
            raise RuntimeError("No nodes registered in pipeline")

        if is_debug_mode():
            logger.debug(f"[pipeline/overlap] Starting overlapped pipeline: input_ids shape={input_ids.shape}, nodes={total_nodes}")

        first_node_id = node_order_snapshot[0]
        first_node = nodes_snapshot[first_node_id]
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
            node_id = node_order_snapshot[i]
            node = nodes_snapshot[node_id]
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
                    with self._topology_lock:
                        if node_id in self.nodes:
                            self.nodes[node_id].healthy = False
                    continue

            req = self._prepare_forward_request(
                node_id, node, False, is_last,
                seq_len, batch_size, current_hidden, past_kv,
                request_id, draft_tokens if is_last else None, None,
            )

            response = node.client.stub.ForwardPass(req)
            current_hidden = self._process_forward_response(response, node_id, node, node_kv_caches)

            if is_debug_mode():
                logger.debug(f"[pipeline] Node {node_id}: output shape={current_hidden.shape}")

        return current_hidden

    def unregister_node(self, node_id: str) -> NodeRegistration | None:
        """Remove a node from the pipeline topology (used by self-healing).

        Removes the node from ``nodes``, ``node_order``, ``prefill_nodes``,
        and ``decode_nodes``. Returns the removed registration if found.

        Args:
            node_id: Node identifier to remove.

        Returns:
            The removed ``NodeRegistration``, or ``None`` if not found.
        """
        with self._topology_lock:
            removed = self.nodes.pop(node_id, None)
            if node_id in self.node_order:
                self.node_order = [
                    nid for nid in self.node_order if nid != node_id
                ]
            self.prefill_nodes.pop(node_id, None)
            self.decode_nodes.pop(node_id, None)

            # Rebuild transport rank map if needed
            if self._transport_rank_map:
                self._transport_rank_map.pop(node_id, None)
                for i, nid in enumerate(self.node_order):
                    self._transport_rank_map[nid] = i

        if removed:
            try:
                removed.close()
            except Exception:
                pass
            logger.info(f"Unregistered node {node_id} from pipeline")
        return removed

    def shutdown(self) -> None:
        """Shut down the orchestrator and release resources.

        Destroys NCCL transport process groups and shuts down the thread pool.
        """
        if self._transport is not None and hasattr(self._transport, 'destroy'):
            try:
                self._transport.destroy()
            except Exception as e:
                logger.debug(f"Error destroying tensor transport: {e}")
            self._transport = None
        self._executor.shutdown(wait=True)

    def create_node_kv_caches(self) -> dict[str, list | None]:
        """Create a fresh KV cache dict for all registered nodes.

        Returns:
            Dict mapping node_id -> None (KV caches populated on first decode step).
        """
        with self._topology_lock:
            return {nid: None for nid in self.node_order}

    def _execute_node_grpc(
        self,
        node_id: str,
        node,
        node_kv_caches: dict[str, list | None],
        is_first_node: bool,
        is_last_node: bool,
        seq_len: int,
        batch_size: int,
        current_hidden: torch.Tensor | None,
        request_id: str,
        draft_tokens: list[int] | None,
        input_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        """Execute a single node's forward pass via gRPC.

        Handles circuit breaker check, fallback, request preparation,
        gRPC call, response processing, and error handling.

        Returns:
            Updated hidden states tensor, or None if node was skipped.

        Raises:
            NodeUnreachableError: If node fails and no fallback.
        """
        node_log = logger.bind(request_id=request_id, node_id=node_id)
        past_kv = node_kv_caches.get(node_id)

        # Circuit breaker check
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
                with self._topology_lock:
                    if node_id in self.nodes:
                        self.nodes[node_id].healthy = False
                return current_hidden

        try:
            request = self._prepare_forward_request(
                node_id, node, is_first_node, is_last_node,
                seq_len, batch_size, current_hidden, past_kv,
                request_id, draft_tokens, input_ids,
            )
            response = self._executor.submit(node.client.stub.ForwardPass, request).result()
            return self._process_forward_response(response, node_id, node, node_kv_caches)

        except (NodeUnreachableError, OOMError, InputValidationError, GRPCTimeoutError):
            raise
        except grpc.RpcError as e:
            self.resource_mgr.record_failure(node_id)
            node_log.error(f"Pipeline gRPC error: {e}")
            with self._topology_lock:
                if node_id in self.nodes:
                    self.nodes[node_id].healthy = False
            raise NodeUnreachableError(
                node_id=node_id, host=node.host, port=node.port, original_error=e
            )
        except Exception as e:
            self.resource_mgr.record_failure(node_id)
            node_log.error(f"Pipeline failed: {e}")
            with self._topology_lock:
                if node_id in self.nodes:
                    self.nodes[node_id].healthy = False
            raise
