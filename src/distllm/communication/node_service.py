"""Node gRPC service implementations.

Contains sync and async NodeService servicers that handle
forward pass, health checks, node info, MoE, and ping RPCs.
"""

import torch
from loguru import logger
from typing import Callable

from distllm.communication.node_pb2 import (
    ErrorCode, ForwardPassResponse, HealthCheckRequest, HealthCheckResponse,
    NodeInfo, PingResponse,
)
from distllm.communication.node_pb2_grpc import NodeServiceServicer
from distllm.communication.tensor_transport import (
    _parse_forward_request, _build_forward_response, _log_forward_debug,
    is_debug_mode,
)
from distllm.errors import InputValidationError
from distllm.communication.serializers import proto_to_tensor, tensor_to_proto

# NVML module-level state: initialize once, shutdown on process exit
_pynvml_handle = None
_pynvml_device_handle = None


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
        """
        Args:
            node_id: Unique node identifier
            forward_fn: Function that runs forward pass. Signature:
                forward_fn(hidden_states, attention_mask, position_ids, past_key_values) -> (output, new_past_key_values)
        """
        self.node_id = node_id
        self.forward_fn = forward_fn
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def ForwardPass(self, request, context):
        """Receive input, run forward pass, return output."""
        try:
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

            draft_tokens = list(request.draft_tokens) if request.draft_tokens else None
            response = _build_forward_response(request.request_id, output, new_past_kv, draft_tokens)
            return response

        except RuntimeError as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower() or ("cuda" in error_msg.lower() and "memory" in error_msg.lower()):
                error_code = ErrorCode.OOM
                error_msg = f"GPU OOM on {self.node_id}: {error_msg}"
            else:
                error_code = ErrorCode.MODEL_ERROR
                error_msg = f"Model error on {self.node_id}: {error_msg}"
            logger.error(f"ForwardPass error on {self.node_id} (code={error_code}): {e}")
            return ForwardPassResponse(
                request_id=request.request_id,
                success=False,
                error_message=error_msg,
                error_code=error_code,
            )
        except InputValidationError as e:
            error_msg = f"Invalid input on {self.node_id}: {e}"
            logger.error(f"ForwardPass invalid input on {self.node_id}: {e}")
            return ForwardPassResponse(
                request_id=request.request_id,
                success=False,
                error_message=error_msg,
                error_code=ErrorCode.INVALID_INPUT,
            )
        except ValueError as e:
            error_msg = f"Invalid input on {self.node_id}: {e}"
            logger.error(f"ForwardPass invalid input on {self.node_id}: {e}")
            return ForwardPassResponse(
                request_id=request.request_id,
                success=False,
                error_message=error_msg,
                error_code=ErrorCode.INVALID_INPUT,
            )
        except (TypeError, MemoryError, AttributeError) as e:
            error_msg = f"Unexpected error on {self.node_id}: {e}"
            logger.error(f"ForwardPass error on {self.node_id}: {e}")
            return ForwardPassResponse(
                request_id=request.request_id,
                success=False,
                error_message=error_msg,
                error_code=ErrorCode.UNKNOWN,
            )

    def _get_gpu_stats(self) -> tuple[int, int, float, float, bool]:
        """Get actual GPU stats using pynvml if available.

        Returns:
            Tuple of (memory_used, memory_total, gpu_util, temperature, healthy)
        """
        memory_used = 0
        memory_total = 0
        gpu_util = 0.0
        temperature = 0.0
        healthy = True

        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated()
            memory_total = torch.cuda.get_device_properties(0).total_memory

            # Use module-level NVML singleton (initialized once, not per health check)
            pynvml, device_handle = _get_pynvml_handle()
            if pynvml and device_handle:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(device_handle)
                    gpu_util = float(util.gpu)
                    temp = pynvml.nvmlDeviceGetTemperature(device_handle, pynvml.NVML_TEMPERATURE_GPU)
                    temperature = float(temp)
                except Exception as e:
                    logger.debug(f"GPU stats unavailable: {e}")

            # Check for unhealthy conditions
            if memory_total > 0 and memory_used / memory_total > 0.95:
                healthy = False  # GPU memory nearly full

        return memory_used, memory_total, gpu_util, temperature, healthy

    def HealthCheck(self, request, context):
        """Return node health status with actual GPU metrics."""
        memory_used, memory_total, gpu_util, temperature, healthy = self._get_gpu_stats()

        return HealthCheckResponse(
            node_id=self.node_id,
            healthy=healthy,
            memory_used=memory_used,
            memory_total=memory_total,
            gpu_utilization=gpu_util,
            temperature=temperature,
        )

    def GetNodeInfo(self, request, context):
        """Return node hardware info."""
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        total_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        available_memory = total_memory - torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        return NodeInfo(
            node_id=self.node_id,
            device_type=device_type,
            device_name=device_name,
            total_memory=total_memory,
            available_memory=available_memory,
        )

    def MoEForward(self, request, context):
        """Execute MoE expert forward pass."""
        import time
        start = time.time()
        try:
            hidden_states = proto_to_tensor(request.hidden_states, self.device)
            expert_ids = list(request.expert_ids)

            if is_debug_mode():
                logger.debug(f"[{self.node_id}] MoEForward hidden_states shape: {hidden_states.shape}, expert_ids: {expert_ids}")

            with torch.no_grad():
                output, _ = self.forward_fn(
                    hidden_states,
                    attention_mask=None,
                    position_ids=None,
                    past_key_values=None,
                    input_ids=None,
                )

            processing_time_ms = (time.time() - start) * 1000

            if is_debug_mode():
                logger.debug(f"[{self.node_id}] MoEForward output shape: {output.shape}")

            from distllm.communication.node_pb2 import MoEForwardResponse
            return MoEForwardResponse(
                output=tensor_to_proto(output),
                success=True,
                processing_time_ms=processing_time_ms,
            )

        except (RuntimeError, ValueError) as e:
            processing_time_ms = (time.time() - start) * 1000
            logger.error(f"MoEForward error on {self.node_id}: {e}")
            from distllm.communication.node_pb2 import MoEForwardResponse
            return MoEForwardResponse(
                success=False,
                error_message=f"MoE error on {self.node_id}: {e}",
                processing_time_ms=processing_time_ms,
            )
        except (TypeError, MemoryError, AttributeError) as e:
            processing_time_ms = (time.time() - start) * 1000
            logger.error(f"MoEForward unexpected error on {self.node_id}: {e}")
            from distllm.communication.node_pb2 import MoEForwardResponse
            return MoEForwardResponse(
                success=False,
                error_message=f"Unexpected MoE error on {self.node_id}: {e}",
                processing_time_ms=processing_time_ms,
            )

    def Ping(self, request, context):
        """Lightweight ping for cross-cluster latency measurement."""
        import time
        return PingResponse(
            node_id=self.node_id,
            cluster_id="",  # Nodes don't know their cluster at this level
            timestamp=int(time.time() * 1000),
            latency_ms=0.0,  # Client measures RTT
        )


class AsyncNodeService(NodeServiceServicer):
    """Async gRPC service implementation for worker nodes using grpc.aio."""

    def __init__(self, node_id: str, forward_fn: Callable):
        """
        Args:
            node_id: Unique node identifier
            forward_fn: Function that runs forward pass. Signature:
                forward_fn(hidden_states, attention_mask, position_ids, past_key_values) -> (output, new_past_key_values)
        """
        self.node_id = node_id
        self.forward_fn = forward_fn
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _get_gpu_stats(self) -> tuple[int, int, float, float, bool]:
        """Get actual GPU stats using pynvml if available."""
        return NodeService._get_gpu_stats(self)

    async def ForwardPass(self, request, context):
        """Receive input, run forward pass, return output (async)."""
        try:
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

            draft_tokens = list(request.draft_tokens) if request.draft_tokens else None
            response = _build_forward_response(request.request_id, output, new_past_kv, draft_tokens)
            return response

        except RuntimeError as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower() or ("cuda" in error_msg.lower() and "memory" in error_msg.lower()):
                error_code = ErrorCode.OOM
                error_msg = f"GPU OOM on {self.node_id}: {error_msg}"
            else:
                error_code = ErrorCode.MODEL_ERROR
                error_msg = f"Model error on {self.node_id}: {error_msg}"
            logger.error(f"ForwardPass error on {self.node_id} (code={error_code}): {e}")
            return ForwardPassResponse(
                request_id=request.request_id,
                success=False,
                error_message=error_msg,
                error_code=error_code,
            )
        except InputValidationError as e:
            error_msg = f"Invalid input on {self.node_id}: {e}"
            logger.error(f"ForwardPass invalid input on {self.node_id}: {e}")
            return ForwardPassResponse(
                request_id=request.request_id,
                success=False,
                error_message=error_msg,
                error_code=ErrorCode.INVALID_INPUT,
            )
        except ValueError as e:
            error_msg = f"Invalid input on {self.node_id}: {e}"
            logger.error(f"ForwardPass invalid input on {self.node_id}: {e}")
            return ForwardPassResponse(
                request_id=request.request_id,
                success=False,
                error_message=error_msg,
                error_code=ErrorCode.INVALID_INPUT,
            )
        except (TypeError, MemoryError, AttributeError) as e:
            error_msg = f"Unexpected error on {self.node_id}: {e}"
            logger.error(f"ForwardPass error on {self.node_id}: {e}")
            return ForwardPassResponse(
                request_id=request.request_id,
                success=False,
                error_message=error_msg,
                error_code=ErrorCode.UNKNOWN,
            )

    async def HealthCheck(self, request, context):
        """Return node health status with actual GPU metrics (async)."""
        memory_used, memory_total, gpu_util, temperature, healthy = self._get_gpu_stats()

        return HealthCheckResponse(
            node_id=self.node_id,
            healthy=healthy,
            memory_used=memory_used,
            memory_total=memory_total,
            gpu_utilization=gpu_util,
            temperature=temperature,
        )

    async def GetNodeInfo(self, request, context):
        """Return node hardware info (async)."""
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        total_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        available_memory = total_memory - torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        return NodeInfo(
            node_id=self.node_id,
            device_type=device_type,
            device_name=device_name,
            total_memory=total_memory,
            available_memory=available_memory,
        )

    async def MoEForward(self, request, context):
        """Execute MoE expert forward pass (async)."""
        import time
        start = time.time()
        try:
            hidden_states = proto_to_tensor(request.hidden_states, self.device)
            expert_ids = list(request.expert_ids)

            from distllm.communication.tensor_transport import is_debug_mode
            if is_debug_mode():
                logger.debug(f"[{self.node_id}] MoEForward hidden_states shape: {hidden_states.shape}, expert_ids: {expert_ids}")

            with torch.no_grad():
                output, _ = self.forward_fn(
                    hidden_states,
                    attention_mask=None,
                    position_ids=None,
                    past_key_values=None,
                    input_ids=None,
                )

            processing_time_ms = (time.time() - start) * 1000

            if is_debug_mode():
                logger.debug(f"[{self.node_id}] MoEForward output shape: {output.shape}")

            from distllm.communication.node_pb2 import MoEForwardResponse
            return MoEForwardResponse(
                output=tensor_to_proto(output),
                success=True,
                processing_time_ms=processing_time_ms,
            )

        except (RuntimeError, ValueError) as e:
            processing_time_ms = (time.time() - start) * 1000
            logger.error(f"MoEForward error on {self.node_id}: {e}")
            from distllm.communication.node_pb2 import MoEForwardResponse
            return MoEForwardResponse(
                success=False,
                error_message=f"MoE error on {self.node_id}: {e}",
                processing_time_ms=processing_time_ms,
            )
        except (TypeError, MemoryError, AttributeError) as e:
            processing_time_ms = (time.time() - start) * 1000
            logger.error(f"MoEForward unexpected error on {self.node_id}: {e}")
            from distllm.communication.node_pb2 import MoEForwardResponse
            return MoEForwardResponse(
                success=False,
                error_message=f"Unexpected MoE error on {self.node_id}: {e}",
                processing_time_ms=processing_time_ms,
            )

    async def Ping(self, request, context):
        """Lightweight ping for cross-cluster latency measurement (async)."""
        import time
        return PingResponse(
            node_id=self.node_id,
            cluster_id="",
            timestamp=int(time.time() * 1000),
            latency_ms=0.0,
        )
