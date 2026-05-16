"""Pipeline orchestrator for distributed LLM inference.

Manages node topology, layer routing, and distributed pipeline execution.
Extracted from the Coordinator class.
"""

from typing import Any, Dict, List, Optional, Tuple

import grpc
import torch
from loguru import logger

from distllm.communication.grpc import NodeClient, is_debug_mode
from distllm.communication.grpc import AsyncNodeClient
from distllm.communication.serializers import tensor_to_proto, proto_to_tensor, kv_cache_to_proto
from distllm.communication.node_pb2 import ForwardPassRequest
from distllm.core.kv_cache import KVCache
from distllm.core.resource_manager import NodeRegistration, ResourceManager
from distllm.core.latency_tracker import LatencyTracker
from distllm.config.loader import NodeRole
from distllm.errors.types import (
    ConfigValidationError, NodeUnreachableError, OOMError,
    InputValidationError, GRPCTimeoutError,
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
        resource_mgr: Optional[ResourceManager] = None,
        total_layers: int = 0,
    ):
        self.nodes: Dict[str, NodeRegistration] = {}
        self.node_order: List[str] = []
        self.prefill_nodes: Dict[str, NodeRegistration] = {}
        self.decode_nodes: Dict[str, NodeRegistration] = {}
        self.total_layers = total_layers
        self.resource_mgr = resource_mgr or ResourceManager()
        self._latency_tracker: Optional[LatencyTracker] = None

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
        ca_cert: Optional[str] = None,
        expert_ids: Optional[List[int]] = None,
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

        logger.info(f"Registered {node_id}: layers {start_layer}-{end_layer}")

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

    def run_pipeline(
        self,
        input_ids: torch.Tensor,
        node_kv_caches: Dict[str, Optional[List]],
        request_id: str,
        draft_tokens: Optional[List[int]] = None,
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
                node_log.warning(f"Circuit breaker open, attempting fallback")
                node.healthy = False

            cached_len = 0
            if past_kv is not None and len(past_kv) > 0:
                cached_len = past_kv[0][0].shape[-2]
            total_len = cached_len + seq_len
            node_attention_mask = torch.ones(batch_size, total_len, dtype=torch.long)
            node_position_ids = torch.arange(cached_len, total_len).unsqueeze(0)

            try:
                request = ForwardPassRequest(request_id=request_id, use_cache=True)

                if is_first_node:
                    request.input_ids.extend(input_ids.squeeze(0).tolist())
                else:
                    request.hidden_states.CopyFrom(tensor_to_proto(current_hidden))

                request.attention_mask.CopyFrom(tensor_to_proto(node_attention_mask))
                request.position_ids.CopyFrom(tensor_to_proto(node_position_ids))

                if past_kv is not None:
                    cache = KVCache()
                    cache.set_all(past_kv)
                    request.kv_cache.CopyFrom(kv_cache_to_proto(cache))

                # Attach draft tokens to last node for speculative verification
                if is_last_node and draft_tokens:
                    request.draft_tokens.extend(draft_tokens)

                response = node.client.stub.ForwardPass(request)

                if not response.success:
                    self.resource_mgr.record_failure(node_id)
                    node.healthy = False
                    # Map error codes to specific exceptions
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

                # Record latency for rebalancer
                if self._latency_tracker is not None and response.processing_time_ms > 0:
                    self._latency_tracker.record(node_id, response.processing_time_ms)

                current_hidden = proto_to_tensor(response.output)

                if is_debug_mode():
                    logger.debug(f"[pipeline] Node {node_id}: output shape={current_hidden.shape}")

                if response.HasField('kv_cache') and response.kv_cache.layers:
                    new_cache = KVCache()
                    for layer in response.kv_cache.layers:
                        k = proto_to_tensor(layer.key_states)
                        v = proto_to_tensor(layer.value_states)
                        new_cache.cache.append((k, v))
                    node_kv_caches[node_id] = new_cache.cache

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
        node_kv_caches: Dict[str, Optional[List]],
        request_id: str,
        draft_tokens: Optional[List[int]] = None,
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
                node_log.warning(f"Circuit breaker open, attempting fallback")
                node.healthy = False

            cached_len = 0
            if past_kv is not None and len(past_kv) > 0:
                cached_len = past_kv[0][0].shape[-2]
            total_len = cached_len + seq_len
            node_attention_mask = torch.ones(batch_size, total_len, dtype=torch.long)
            node_position_ids = torch.arange(cached_len, total_len).unsqueeze(0)

            try:
                request = ForwardPassRequest(request_id=request_id, use_cache=True)

                if is_first_node:
                    request.input_ids.extend(input_ids.squeeze(0).tolist())
                else:
                    request.hidden_states.CopyFrom(tensor_to_proto(current_hidden))

                request.attention_mask.CopyFrom(tensor_to_proto(node_attention_mask))
                request.position_ids.CopyFrom(tensor_to_proto(node_position_ids))

                if past_kv is not None:
                    cache = KVCache()
                    cache.set_all(past_kv)
                    request.kv_cache.CopyFrom(kv_cache_to_proto(cache))

                # Attach draft tokens to last node for speculative verification
                if is_last_node and draft_tokens:
                    request.draft_tokens.extend(draft_tokens)

                # Use async client if available, otherwise fall back to sync
                if hasattr(node, 'async_client') and node.async_client is not None:
                    response = await node.async_client.stub.ForwardPass(request)
                else:
                    # Wrap sync call in asyncio.to_thread for non-blocking
                    response = await asyncio.to_thread(node.client.stub.ForwardPass, request)

                if not response.success:
                    self.resource_mgr.record_failure(node_id)
                    node.healthy = False
                    # Map error codes to specific exceptions
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

                # Record latency for rebalancer
                if self._latency_tracker is not None and response.processing_time_ms > 0:
                    self._latency_tracker.record(node_id, response.processing_time_ms)

                current_hidden = proto_to_tensor(response.output)

                if is_debug_mode():
                    logger.debug(f"[pipeline] Node {node_id}: output shape={current_hidden.shape}")

                if response.HasField('kv_cache') and response.kv_cache.layers:
                    new_cache = KVCache()
                    for layer in response.kv_cache.layers:
                        k = proto_to_tensor(layer.key_states)
                        v = proto_to_tensor(layer.value_states)
                        new_cache.cache.append((k, v))
                    node_kv_caches[node_id] = new_cache.cache

            except (NodeUnreachableError, OOMError, InputValidationError, GRPCTimeoutError):
                raise
            except grpc.aio.AioRpcError as e:
                self.resource_mgr.record_failure(node_id)
                node_log.error(f"Pipeline async gRPC error: {e}")
                node.healthy = False
                raise NodeUnreachableError(
                    node_id=node_id, host=node.host, port=node.port, original_error=e
                )
            except Exception as e:
                self.resource_mgr.record_failure(node_id)
                node_log.error(f"Pipeline async failed: {e}")
                node.healthy = False
                raise

        return current_hidden
