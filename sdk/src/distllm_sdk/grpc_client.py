"""gRPC client SDK for high-performance inter-node communication.

Provides a Python client matching the NodeService defined in
``proto/node.proto``. Used for direct node-to-node communication
in the distributed pipeline (forward pass, health check, weight transfer).

Requires ``grpcio`` and ``grpcio-tools`` (optional dependency).

Usage::

    from distllm_sdk.grpc_client import NodeGRPCClient

    async with NodeGRPCClient("10.0.0.1:50051") as client:
        response = await client.forward_pass(
            input_ids=[1, 2, 3],
            request_id="req-1",
        )
        print(response.output)
"""

from __future__ import annotations

import asyncio
import io
import struct
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class TensorData:
    """Tensor representation for gRPC transport."""
    shape: list[int]
    dtype: str
    raw_data: bytes = b""
    data: list[float] = field(default_factory=list)


@dataclass
class KVCacheData:
    """KV cache representation for gRPC transport."""
    layers: list[dict[str, TensorData]] = field(default_factory=list)


@dataclass
class ForwardPassRequest:
    """Forward pass request matching ForwardPassRequest proto."""
    request_id: str = ""
    input_ids: list[int] = field(default_factory=list)
    hidden_states: TensorData | None = None
    attention_mask: TensorData | None = None
    position_ids: TensorData | None = None
    kv_cache: KVCacheData | None = None
    use_cache: bool = True
    is_first_pass: bool = True
    draft_tokens: list[int] = field(default_factory=list)
    batch_size: int = 1
    seq_len: int = 0
    is_last_pass: bool = False
    model_name: str = ""
    cluster_key: str = ""


@dataclass
class ForwardPassResponse:
    """Forward pass response matching ForwardPassResponse proto."""
    request_id: str = ""
    output: TensorData | None = None
    kv_cache: KVCacheData | None = None
    success: bool = False
    error_message: str = ""
    error_code: int = 0
    is_logits: bool = True
    processing_time_ms: float = 0.0
    cluster_key: str = ""


@dataclass
class HealthCheckResponse:
    """Health check response matching HealthCheckResponse proto."""
    healthy: bool = False
    node_id: str = ""
    memory_used_bytes: int = 0
    memory_total_bytes: int = 0
    gpu_utilization: float = 0.0
    start_layer: int = 0
    end_layer: int = 0
    total_layers: int = 0
    gpu_name: str = ""
    gpu_memory_total: int = 0
    num_layers_loaded: int = 0


@dataclass
class ProfileResponse:
    """Profile response matching ProfileResponse proto."""
    node_id: str = ""
    gpu_name: str = ""
    total_memory_bytes: int = 0
    free_memory_bytes: int = 0
    compute_tflops: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    sm_count: int = 0


@dataclass
class TransferWeightsResponse:
    """Weight transfer response matching TransferWeightsResponse proto."""
    model_name: str = ""
    start_layer: int = 0
    end_layer: int = 0
    state_dict_bytes: bytes = b""
    success: bool = False
    error_message: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    is_final_chunk: bool = False


@dataclass
class ModelAdvertisement:
    """Model advertisement matching ModelAdvertisement proto."""
    model_name: str = ""
    start_layer: int = 0
    end_layer: int = 0
    total_layers: int = 0
    node_id: str = ""
    host: str = ""
    port: int = 0


def _serialize_tensor(tensor: Any) -> bytes:
    """Serialize a tensor to bytes for gRPC transport."""
    try:
        import torch
        if isinstance(tensor, torch.Tensor):
            buffer = io.BytesIO()
            torch.save(tensor.cpu(), buffer)
            return buffer.getvalue()
    except ImportError:
        pass
    return b""


def _deserialize_tensor(data: bytes) -> Any:
    """Deserialize bytes back to a tensor."""
    if not data:
        return None
    try:
        import torch
        buffer = io.BytesIO(data)
        return torch.load(buffer, weights_only=True)
    except Exception:
        return None


class NodeGRPCClient:
    """gRPC client for communicating with DistLLM worker nodes.

    Wraps the NodeService defined in proto/node.proto with a
    high-level Python API. Supports async context manager for
    automatic connection management.

    Args:
        target: Target address (host:port).
        timeout: Default RPC timeout in seconds.
        cluster_key: Optional cluster authentication key.
        use_tls: Whether to use TLS for the connection.
    """

    def __init__(
        self,
        target: str,
        timeout: float = 30.0,
        cluster_key: str = "",
        use_tls: bool = False,
    ):
        self._target = target
        self._timeout = timeout
        self._cluster_key = cluster_key
        self._use_tls = use_tls
        self._channel = None
        self._stub = None
        self._connected = False

    async def connect(self) -> None:
        """Establish the gRPC channel."""
        try:
            import grpc
            import grpc.aio

            if self._use_tls:
                credentials = grpc.ssl_channel_credentials()
                self._channel = grpc.aio.secure_channel(self._target, credentials)
            else:
                self._channel = grpc.aio.insecure_channel(self._target)

            # Wait for connection
            await asyncio.wait_for(
                self._channel.channel_ready(),
                timeout=self._timeout,
            )
            self._connected = True
            logger.debug(f"gRPC connected to {self._target}")
        except ImportError:
            raise RuntimeError("grpcio not installed. Install with: pip install distllm-sdk[grpc]")
        except asyncio.TimeoutError:
            raise ConnectionError(f"gRPC connection to {self._target} timed out")

    async def close(self) -> None:
        """Close the gRPC channel."""
        if self._channel:
            await self._channel.close()
            self._connected = False

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def forward_pass(
        self,
        input_ids: list[int] | None = None,
        hidden_states: Any = None,
        request_id: str = "",
        use_cache: bool = True,
        is_first_pass: bool = True,
        is_last_pass: bool = False,
        draft_tokens: list[int] | None = None,
        model_name: str = "",
    ) -> ForwardPassResponse:
        """Execute a forward pass on a remote node.

        Args:
            input_ids: Input token IDs.
            hidden_states: Hidden state tensor (for pipeline stages).
            request_id: Request identifier.
            use_cache: Whether to use KV cache.
            is_first_pass: Whether this is the first pass in the pipeline.
            is_last_pass: Whether this is the last pass.
            draft_tokens: Draft tokens for speculative decoding.
            model_name: Model name for multi-model serving.

        Returns:
            ForwardPassResponse with output logits and KV cache.
        """
        if not self._connected:
            await self.connect()

        try:
            import grpc
            from distllm.dist import node_pb2, node_pb2_grpc

            stub = node_pb2_grpc.NodeServiceStub(self._channel)

            request = node_pb2.ForwardPassRequest(
                request_id=request_id,
                input_ids=input_ids or [],
                use_cache=use_cache,
                is_first_pass=is_first_pass,
                is_last_pass=is_last_pass,
                draft_tokens=draft_tokens or [],
                model_name=model_name,
                cluster_key=self._cluster_key,
            )

            if hidden_states is not None:
                import torch
                if isinstance(hidden_states, torch.Tensor):
                    request.hidden_states.shape.extend(list(hidden_states.shape))
                    request.hidden_states.dtype = str(hidden_states.dtype)
                    request.hidden_states.raw_data = hidden_states.cpu().numpy().tobytes()

            response = await asyncio.wait_for(
                stub.ForwardPass(request),
                timeout=self._timeout,
            )

            output_tensor = None
            if response.output.raw_data:
                import numpy as np
                shape = list(response.output.shape)
                dtype_map = {"float32": "f4", "float16": "f2", "bfloat16": "f2"}
                dtype_str = dtype_map.get(response.output.dtype, "f2")
                output_tensor = torch.from_numpy(
                    np.frombuffer(response.output.raw_data, dtype=dtype_str).reshape(shape)
                )

            return ForwardPassResponse(
                request_id=response.request_id,
                output=TensorData(
                    shape=list(response.output.shape),
                    dtype=response.output.dtype,
                    raw_data=response.output.raw_data,
                ) if response.output.raw_data else None,
                success=response.success,
                error_message=response.error_message,
                error_code=response.error_code,
                is_logits=response.is_logits,
                processing_time_ms=response.processing_time_ms,
            )

        except ImportError:
            raise RuntimeError("grpcio-tools not installed. Install with: pip install distllm-sdk[grpc]")
        except asyncio.TimeoutError:
            return ForwardPassResponse(
                request_id=request_id,
                success=False,
                error_message=f"Forward pass timed out after {self._timeout}s",
                error_code=4,  # DEADLINE_EXCEEDED
            )

    async def health_check(self, node_id: str = "") -> HealthCheckResponse:
        """Check health of a remote node.

        Returns:
            HealthCheckResponse with node health status.
        """
        if not self._connected:
            await self.connect()

        try:
            from distllm.dist import node_pb2, node_pb2_grpc

            stub = node_pb2_grpc.NodeServiceStub(self._channel)
            request = node_pb2.HealthCheckRequest(
                node_id=node_id,
                cluster_key=self._cluster_key,
            )
            response = await asyncio.wait_for(
                stub.HealthCheck(request),
                timeout=self._timeout,
            )
            return HealthCheckResponse(
                healthy=response.healthy,
                node_id=response.node_id,
                memory_used_bytes=response.memory_used_bytes,
                memory_total_bytes=response.memory_total_bytes,
                gpu_utilization=response.gpu_utilization,
                start_layer=response.start_layer,
                end_layer=response.end_layer,
                total_layers=response.total_layers,
                gpu_name=response.gpu_name,
                gpu_memory_total=response.gpu_memory_total,
                num_layers_loaded=response.num_layers_loaded,
            )
        except Exception as e:
            return HealthCheckResponse(
                healthy=False,
                node_id=node_id,
            )

    async def profile(self, node_id: str = "") -> ProfileResponse:
        """Profile a remote node's hardware capabilities.

        Returns:
            ProfileResponse with GPU capabilities.
        """
        if not self._connected:
            await self.connect()

        try:
            from distllm.dist import node_pb2, node_pb2_grpc

            stub = node_pb2_grpc.NodeServiceStub(self._channel)
            request = node_pb2.ProfileRequest(
                node_id=node_id,
                cluster_key=self._cluster_key,
            )
            response = await asyncio.wait_for(
                stub.Profile(request),
                timeout=self._timeout,
            )
            return ProfileResponse(
                node_id=response.node_id,
                gpu_name=response.gpu_name,
                total_memory_bytes=response.total_memory_bytes,
                free_memory_bytes=response.free_memory_bytes,
                compute_tflops=response.compute_tflops,
                memory_bandwidth_gbps=response.memory_bandwidth_gbps,
                sm_count=response.sm_count,
            )
        except Exception as e:
            logger.warning(f"Profile failed for {node_id}: {e}")
            return ProfileResponse(node_id=node_id)

    async def transfer_weights(
        self,
        model_name: str,
        start_layer: int,
        end_layer: int,
    ) -> TransferWeightsResponse:
        """Request model weights from a remote node.

        Returns:
            TransferWeightsResponse with weight data.
        """
        if not self._connected:
            await self.connect()

        try:
            from distllm.dist import node_pb2, node_pb2_grpc

            stub = node_pb2_grpc.NodeServiceStub(self._channel)
            request = node_pb2.TransferWeightsRequest(
                model_name=model_name,
                start_layer=start_layer,
                end_layer=end_layer,
                cluster_key=self._cluster_key,
            )
            response = await asyncio.wait_for(
                stub.TransferWeights(request),
                timeout=self._timeout,
            )
            return TransferWeightsResponse(
                model_name=response.model_name,
                start_layer=response.start_layer,
                end_layer=response.end_layer,
                state_dict_bytes=response.state_dict_bytes,
                success=response.success,
                error_message=response.error_message,
            )
        except Exception as e:
            logger.warning(f"Weight transfer failed: {e}")
            return TransferWeightsResponse(
                model_name=model_name,
                success=False,
                error_message=str(e),
            )

    async def advertise_models(self, node_id: str = "") -> list[ModelAdvertisement]:
        """Discover models available on a remote node.

        Returns:
            List of ModelAdvertisement with available models.
        """
        if not self._connected:
            await self.connect()

        try:
            from distllm.dist import node_pb2, node_pb2_grpc

            stub = node_pb2_grpc.NodeServiceStub(self._channel)
            request = node_pb2.AdvertiseModelsRequest(
                node_id=node_id,
                cluster_key=self._cluster_key,
            )
            response = await asyncio.wait_for(
                stub.AdvertiseModels(request),
                timeout=self._timeout,
            )
            return [
                ModelAdvertisement(
                    model_name=m.model_name,
                    start_layer=m.start_layer,
                    end_layer=m.end_layer,
                    total_layers=m.total_layers,
                    node_id=m.node_id,
                    host=m.host,
                    port=m.port,
                )
                for m in response.models
            ]
        except Exception as e:
            logger.warning(f"Model advertisement failed: {e}")
            return []

    async def transfer_weights_stream(
        self,
        model_name: str,
        start_layer: int,
        end_layer: int,
    ):
        """Stream model weights from a remote node (for large models).

        Yields:
            TransferWeightsResponse chunks.
        """
        if not self._connected:
            await self.connect()

        try:
            from distllm.dist import node_pb2, node_pb2_grpc

            stub = node_pb2_grpc.NodeServiceStub(self._channel)
            request = node_pb2.TransferWeightsRequest(
                model_name=model_name,
                start_layer=start_layer,
                end_layer=end_layer,
                cluster_key=self._cluster_key,
            )
            async for chunk in stub.TransferWeightsStream(request):
                yield TransferWeightsResponse(
                    model_name=chunk.model_name,
                    start_layer=chunk.start_layer,
                    end_layer=chunk.end_layer,
                    state_dict_bytes=chunk.state_dict_bytes,
                    success=chunk.success,
                    error_message=chunk.error_message,
                    chunk_index=chunk.chunk_index,
                    total_chunks=chunk.total_chunks,
                    is_final_chunk=chunk.is_final_chunk,
                )
        except Exception as e:
            logger.warning(f"Weight stream failed: {e}")
