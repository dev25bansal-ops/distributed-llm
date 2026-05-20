"""Node gRPC service implementations.

Contains sync and async NodeService servicers that handle
forward pass, health checks, node info, MoE, and ping RPCs.
"""

import threading
from typing import Callable

import grpc
import torch
from loguru import logger

from distllm.communication.node_pb2 import (
    ErrorCode, ForwardPassResponse, HealthCheckResponse,
    NodeInfo, PingResponse,
)
from distllm.communication.node_pb2_grpc import NodeServiceServicer
from distllm.communication.tensor_transport import (
    _parse_forward_request, _build_forward_response, _log_forward_debug,
    is_debug_mode,
)
from distllm.communication.serializers import proto_to_tensor, tensor_to_proto
from distllm.constants import get_tensor_max_bytes
from distllm.errors import InputValidationError

# NVML module-level state: initialize once, shutdown on process exit
_pynvml_handle = None
_pynvml_device_handle = None
_pynvml_lock = threading.Lock()

# Max incoming message bytes for protobuf validation (default from constants, matches gRPC transport)
_MAX_MESSAGE_BYTES = get_tensor_max_bytes()


def _raise_if_oversized(request, context) -> None:
    """Validate incoming request byte size before processing.

    Raises an RPC error via context.abort() if the request exceeds the
    configured maximum message size, preventing OOM from oversized payloads.
    """
    size = request.ByteSize()
    if size > _MAX_MESSAGE_BYTES:
        msg = f"Request size {size} bytes exceeds maximum allowed {_MAX_MESSAGE_BYTES} bytes"
        logger.error(msg)
        context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, msg)


def _get_pynvml_handle():
    """Get or initialize pynvml handle (singleton). Returns None if unavailable."""
    global _pynvml_handle, _pynvml_device_handle
    if not torch.cuda.is_available():
        return None, None
    try:
        import pynvml
    except ImportError:
        return None, None

    if _pynvml_handle is None:
        with _pynvml_lock:
            if _pynvml_handle is None:
                try:
                    pynvml.nvmlInit()
                    _pynvml_handle = pynvml
                    _pynvml_device_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                except Exception as e:
                    logger.debug(f"NVML initialization failed: {e}")
                    _pynvml_handle = False  # Mark as tried and failed
                    return None, None

    if _pynvml_handle is False:
        return None, None

    return _pynvml_handle, _pynvml_device_handle


def _shutdown_pynvml():
    """Shutdown NVML. Call during node shutdown."""
    global _pynvml_handle, _pynvml_device_handle
    if _pynvml_handle and _pynvml_handle is not False:
        try:
            _pynvml_handle.nvmlShutdown()
        except Exception:
            pass
        _pynvml_handle = None
        _pynvml_device_handle = None


class NodeService(NodeServiceServicer):
    """gRPC service implementation for worker nodes."""

    def __init__(self, node_id: str, forward_fn: Callable):
        self.node_id = node_id
        self.forward_fn = forward_fn
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _run_forward(self, request):
        """Shared forward pass logic used by both sync and async services."""
        tensors = _parse_forward_request(request, self.device)
        _log_forward_debug(self.node_id, request, tensors)

        with torch.no_grad():
            output, new_past_kv = self.forward_fn(
                hidden_states=tensors["hidden_states"],
                attention_mask=tensors["attention_mask"],
                position_ids=tensors["position_ids"],
                past_key_values=tensors["past_key_values"],
                input_ids=tensors["input_ids"],
            )

        _log_forward_debug(self.node_id, request, tensors, output)
        return output, new_past_kv, tensors

    def _forward_error_response(self, request, e: Exception) -> ForwardPassResponse:
        """Build a ForwardPassResponse for the given exception."""
        error_msg = str(e)
        if isinstance(e, RuntimeError):
            if "out of memory" in error_msg.lower() or ("cuda" in error_msg.lower() and "memory" in error_msg.lower()):
                error_code = ErrorCode.OOM
                error_msg = f"GPU OOM on {self.node_id}: {error_msg}"
            else:
                error_code = ErrorCode.MODEL_ERROR
                error_msg = f"Model error on {self.node_id}: {error_msg}"
        elif isinstance(e, InputValidationError):
            error_code = ErrorCode.INVALID_INPUT
            error_msg = f"Invalid input on {self.node_id}: {e}"
        elif isinstance(e, ValueError):
            error_code = ErrorCode.INVALID_INPUT
            error_msg = f"Invalid input on {self.node_id}: {e}"
        else:
            error_code = ErrorCode.UNKNOWN
            error_msg = f"Unexpected error on {self.node_id}: {e}"
        logger.error(f"ForwardPass error on {self.node_id} (code={error_code}): {e}")
        return ForwardPassResponse(
            request_id=request.request_id,
            success=False,
            error_message=error_msg,
            error_code=error_code,
        )

    def ForwardPass(self, request, context):
        """Receive input, run forward pass, return output."""
        _raise_if_oversized(request, context)
        try:
            output, new_past_kv, tensors = self._run_forward(request)
            draft_tokens = list(request.draft_tokens) if request.draft_tokens else None
            return _build_forward_response(request.request_id, output, new_past_kv, draft_tokens)
        except (RuntimeError, InputValidationError, ValueError, TypeError, MemoryError, AttributeError) as e:
            return self._forward_error_response(request, e)

    @staticmethod
    def _get_gpu_stats() -> tuple[int, int, float, float, bool]:
        """Get actual GPU stats using pynvml if available."""
        memory_used = 0
        memory_total = 0
        gpu_util = 0.0
        temperature = 0.0
        healthy = True

        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated()
            memory_total = torch.cuda.get_device_properties(0).total_memory

            pynvml, device_handle = _get_pynvml_handle()
            if pynvml and device_handle:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(device_handle)
                    gpu_util = float(util.gpu)
                    temp = pynvml.nvmlDeviceGetTemperature(device_handle, pynvml.NVML_TEMPERATURE_GPU)
                    temperature = float(temp)
                except Exception as e:
                    logger.debug(f"GPU stats unavailable: {e}")

            if memory_total > 0 and memory_used / memory_total > 0.95:
                healthy = False

        return memory_used, memory_total, gpu_util, temperature, healthy

    def HealthCheck(self, request, context):
        """Return node health status with actual GPU metrics."""
        memory_used, memory_total, gpu_util, temperature, healthy = self._get_gpu_stats()
        return HealthCheckResponse(
            node_id=self.node_id, healthy=healthy,
            memory_used=memory_used, memory_total=memory_total,
            gpu_utilization=gpu_util, temperature=temperature,
        )

    def GetNodeInfo(self, request, context):
        """Return node hardware info."""
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        total_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        available_memory = total_memory - torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        return NodeInfo(
            node_id=self.node_id, device_type=device_type,
            device_name=device_name, total_memory=total_memory,
            available_memory=available_memory,
        )

    def MoEForward(self, request, context):
        """Execute MoE expert forward pass."""
        _raise_if_oversized(request, context)
        import time
        start = time.time()
        try:
            hidden_states = proto_to_tensor(request.hidden_states, self.device)
            expert_ids = list(request.expert_ids)
            if is_debug_mode():
                logger.debug(f"[{self.node_id}] MoEForward hidden_states shape: {hidden_states.shape}, expert_ids: {expert_ids}")
            with torch.no_grad():
                output, _ = self.forward_fn(hidden_states, attention_mask=None, position_ids=None, past_key_values=None, input_ids=None)
            processing_time_ms = (time.time() - start) * 1000
            if is_debug_mode():
                logger.debug(f"[{self.node_id}] MoEForward output shape: {output.shape}")
            from distllm.communication.node_pb2 import MoEForwardResponse
            return MoEForwardResponse(output=tensor_to_proto(output), success=True, processing_time_ms=processing_time_ms)
        except (RuntimeError, ValueError) as e:
            processing_time_ms = (time.time() - start) * 1000
            logger.error(f"MoEForward error on {self.node_id}: {e}")
            from distllm.communication.node_pb2 import MoEForwardResponse
            return MoEForwardResponse(success=False, error_message=f"MoE error on {self.node_id}: {e}", processing_time_ms=processing_time_ms)
        except (TypeError, MemoryError, AttributeError) as e:
            processing_time_ms = (time.time() - start) * 1000
            logger.error(f"MoEForward unexpected error on {self.node_id}: {e}")
            from distllm.communication.node_pb2 import MoEForwardResponse
            return MoEForwardResponse(success=False, error_message=f"Unexpected MoE error on {self.node_id}: {e}", processing_time_ms=processing_time_ms)

    def Ping(self, request, context):
        """Lightweight ping for cross-cluster latency measurement."""
        import time
        return PingResponse(node_id=self.node_id, cluster_id="", timestamp=int(time.time() * 1000), latency_ms=0.0)


class AsyncNodeService(NodeServiceServicer):
    """Async gRPC service implementation for worker nodes using grpc.aio.

    Reuses shared logic from NodeService via delegation to avoid duplication.
    """

    def __init__(self, node_id: str, forward_fn: Callable):
        self._sync = NodeService(node_id, forward_fn)

    @property
    def node_id(self) -> str:
        return self._sync.node_id

    async def ForwardPass(self, request, context):
        """Receive input, run forward pass, return output (async)."""
        try:
            output, new_past_kv, tensors = self._sync._run_forward(request)
            draft_tokens = list(request.draft_tokens) if request.draft_tokens else None
            return _build_forward_response(request.request_id, output, new_past_kv, draft_tokens)
        except (RuntimeError, InputValidationError, ValueError, TypeError, MemoryError, AttributeError) as e:
            return self._sync._forward_error_response(request, e)

    async def HealthCheck(self, request, context):
        """Return node health status with actual GPU metrics (async)."""
        return self._sync.HealthCheck(request, context)

    async def GetNodeInfo(self, request, context):
        """Return node hardware info (async)."""
        return self._sync.GetNodeInfo(request, context)

    async def MoEForward(self, request, context):
        """Execute MoE expert forward pass (async)."""
        return self._sync.MoEForward(request, context)

    async def Ping(self, request, context):
        """Lightweight ping for cross-cluster latency measurement (async)."""
        return self._sync.Ping(request, context)
