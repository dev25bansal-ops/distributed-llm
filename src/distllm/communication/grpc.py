"""gRPC communication layer for distributed LLM inference."""

import asyncio
import grpc
import torch
import numpy as np
from concurrent import futures
from loguru import logger
from typing import Optional, Callable, List, Tuple
from collections import defaultdict

from distllm.communication.node_pb2 import (
    Tensor, ForwardPassRequest, ForwardPassResponse,
    HealthCheckRequest, HealthCheckResponse, NodeInfo,
    NodeRegistration as ProtoNodeRegistration, RegistrationResponse,
    InferenceRequest, LogitsResponse, TokenResponse,
    KVCache as ProtoKVCache, KVLayerCache,
    ErrorCode, QuantizationConfig as ProtoQuantizationConfig,
)
from distllm.communication.node_pb2_grpc import (
    NodeServiceServicer, NodeServiceStub,
    CoordinatorServiceServicer, CoordinatorServiceStub,
    add_NodeServiceServicer_to_server,
    add_CoordinatorServiceServicer_to_server,
)
from distllm.core.kv_cache import KVCache
from distllm.communication.serializers import tensor_to_proto, proto_to_tensor, kv_cache_to_proto, proto_to_kv_cache
from distllm.errors import InputValidationError, SerializationError
from distllm.errors.types import NodeUnreachableError, GRPCTimeoutError, CircuitBreakerError
from distllm.errors.retry import retry_grpc_call

# Debug mode configuration — set via CLI --debug
class DebugConfig:
    """Module-level debug configuration for tensor shape logging."""
    enabled = False


def set_debug_mode(enabled: bool) -> None:
    """Enable or disable debug mode for tensor shape logging."""
    DebugConfig.enabled = enabled


def is_debug_mode() -> bool:
    """Check if debug mode is enabled."""
    return DebugConfig.enabled


def _parse_forward_request(request, device: str) -> dict:
    """Parse input tensors from a ForwardPassRequest proto.

    Returns a dict with keys: input_ids, hidden_states, attention_mask,
    position_ids, past_key_values.
    """
    past_key_values = None
    if request.HasField('kv_cache') and request.use_cache:
        past_key_values = proto_to_kv_cache(request.kv_cache, device).cache

    input_ids = None
    if request.input_ids:
        input_ids = torch.tensor(request.input_ids, dtype=torch.long, device=device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

    hidden_states = None
    if request.HasField('hidden_states'):
        hidden_states = proto_to_tensor(request.hidden_states, device)
    elif input_ids is None:
        raise InputValidationError("Either hidden_states or input_ids must be provided", "input")

    attention_mask = None
    if request.HasField('attention_mask'):
        attention_mask = proto_to_tensor(request.attention_mask, device)

    position_ids = None
    if request.HasField('position_ids'):
        position_ids = proto_to_tensor(request.position_ids, device)

    return {
        "input_ids": input_ids,
        "hidden_states": hidden_states,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": past_key_values,
    }


def _build_forward_response(
    request_id: str,
    output: torch.Tensor,
    new_past_kv,
    draft_tokens: Optional[list] = None,
) -> ForwardPassResponse:
    """Build a ForwardPassResponse from model output.

    Handles draft token verification and KV cache serialization.
    """
    response = ForwardPassResponse(
        request_id=request_id,
        output=tensor_to_proto(output),
        success=True,
    )

    # Handle speculative decoding: verify draft tokens
    if draft_tokens:
        if output.dim() == 3:
            num_positions = min(len(draft_tokens), output.shape[1])
            for i in range(num_positions):
                token_at_pos = torch.argmax(output[:, i, :], dim=-1).item()
                response.verified_tokens.append(token_at_pos)
        elif output.dim() == 2:
            token = torch.argmax(output, dim=-1).item()
            response.verified_tokens.append(token)

    if new_past_kv:
        new_cache = KVCache()
        new_cache.set_all(new_past_kv)
        response.kv_cache.CopyFrom(kv_cache_to_proto(new_cache))

    return response


def _log_forward_debug(node_id: str, request, tensors: dict, output: Optional[torch.Tensor] = None) -> None:
    """Log tensor shapes for debugging."""
    if not is_debug_mode():
        return

    input_ids = tensors.get("input_ids")
    hidden_states = tensors.get("hidden_states")
    attention_mask = tensors.get("attention_mask")
    past_key_values = tensors.get("past_key_values")

    if input_ids is not None:
        logger.debug(f"[{node_id}] ForwardPass input_ids shape: {input_ids.shape}")
    if hidden_states is not None:
        logger.debug(f"[{node_id}] ForwardPass hidden_states shape: {hidden_states.shape}")
    if attention_mask is not None:
        logger.debug(f"[{node_id}] ForwardPass attention_mask shape: {attention_mask.shape}")
    if past_key_values:
        cache_len = past_key_values[0][0].shape[-2]
        logger.debug(f"[{node_id}] ForwardPass KV cache seq_len: {cache_len}")
    if hasattr(request, 'draft_tokens') and request.draft_tokens:
        logger.debug(f"[{node_id}] ForwardPass draft_tokens: {list(request.draft_tokens)}")
    if output is not None:
        logger.debug(f"[{node_id}] ForwardPass output shape: {output.shape}")


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

    def _get_gpu_stats(self) -> Tuple[int, int, float, float, bool]:
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

            # Try pynvml for utilization and temperature
            try:
                import pynvml
                try:
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = float(util.gpu)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    temperature = float(temp)
                except Exception as e:
                    logger.debug(f"GPU stats unavailable: {e}")
            except ImportError:
                pass  # pynvml not installed

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
        from distllm.communication.node_pb2 import PingResponse
        return PingResponse(
            node_id=self.node_id,
            cluster_id="",  # Nodes don't know their cluster at this level
            timestamp=int(time.time() * 1000),
            latency_ms=0.0,  # Client measures RTT
        )


class CoordinatorService(CoordinatorServiceServicer):
    """gRPC service implementation for the coordinator."""

    def __init__(self, quantization_config=None, use_tls: bool = True, ca_cert: Optional[str] = None):
        self.nodes = {}
        self.node_channels = {}
        self.node_stubs = {}
        self.quantization_config = quantization_config
        self._expert_registry = None
        self.use_tls = use_tls
        self.ca_cert = ca_cert

    def RegisterNode(self, request, context):
        """Register a worker node."""
        import os

        api_key = os.environ.get("GRPC_API_KEY")
        if api_key:
            client_key = None
            for key, value in request.metadata:
                if key == "api_key":
                    client_key = value
                    break
            if client_key != api_key:
                context.set_code(grpc.StatusCode.UNAUTHENTICATED)
                context.set_details("Invalid or missing API key")
                return RegistrationResponse(accepted=False)

        node_info = request.node_info
        node_id = node_info.node_id
        expert_ids = list(request.expert_ids) if request.expert_ids else []

        logger.info(f"Registering node: {node_id} at {node_info.host}:{node_info.port}")
        if expert_ids:
            logger.info(f"Node {node_id} hosts experts: {expert_ids}")

        if self.use_tls:
            if self.ca_cert:
                from distllm.core.tls import load_tls_channel_credentials
                credentials = load_tls_channel_credentials(self.ca_cert, node_info.host)
            else:
                import os as _os
                auto_ca = _os.path.join("_auto_certs", "ca.crt")
                if _os.path.exists(auto_ca):
                    from distllm.core.tls import load_tls_channel_credentials
                    credentials = load_tls_channel_credentials(auto_ca, node_info.host)
                else:
                    logger.warning(f"No CA cert found for {node_info.host}:{node_info.port}, falling back to insecure")
                    channel = grpc.insecure_channel(f"{node_info.host}:{node_info.port}")
                    stub = NodeServiceStub(channel)
                    self.nodes[node_id] = node_info
                    self.node_channels[node_id] = channel
                    self.node_stubs[node_id] = stub
                    response = RegistrationResponse(accepted=True)
                    if expert_ids and hasattr(self, '_expert_registry') and self._expert_registry is not None:
                        for eid in expert_ids:
                            self._expert_registry.register_expert(eid, node_id)
                    if self.quantization_config and self.quantization_config.method != "none":
                        proto_q = response.quantization
                        proto_q.method = self.quantization_config.method
                        proto_q.bnb_4bit_compute_dtype = self.quantization_config.bnb_4bit_compute_dtype
                        proto_q.bnb_4bit_quant_type = self.quantization_config.bnb_4bit_quant_type
                        proto_q.bnb_4bit_use_double_quant = self.quantization_config.bnb_4bit_use_double_quant
                        proto_q.llm_int8_threshold = self.quantization_config.llm_int8_threshold
                    return response
            channel = grpc.secure_channel(f"{node_info.host}:{node_info.port}", credentials)
        else:
            channel = grpc.insecure_channel(f"{node_info.host}:{node_info.port}")
        stub = NodeServiceStub(channel)

        self.nodes[node_id] = node_info
        self.node_channels[node_id] = channel
        self.node_stubs[node_id] = stub

        response = RegistrationResponse(accepted=True)

        # Register experts on this node
        if expert_ids and hasattr(self, '_expert_registry') and self._expert_registry is not None:
            for eid in expert_ids:
                self._expert_registry.register_expert(eid, node_id)

        if self.quantization_config and self.quantization_config.method != "none":
            proto_q = response.quantization
            proto_q.method = self.quantization_config.method
            proto_q.bnb_4bit_compute_dtype = self.quantization_config.bnb_4bit_compute_dtype
            proto_q.bnb_4bit_quant_type = self.quantization_config.bnb_4bit_quant_type
            proto_q.bnb_4bit_use_double_quant = self.quantization_config.bnb_4bit_use_double_quant
            proto_q.llm_int8_threshold = self.quantization_config.llm_int8_threshold

        return response

    def Infer(self, request, context):
        """Handle inference request by routing to the appropriate node.

        Routes the inference request to registered worker nodes using
        their gRPC ForwardPass endpoints.
        """
        if not self.node_stubs:
            return LogitsResponse(
                request_id=request.request_id,
                generated_text="",
                success=False,
                error_message="No worker nodes registered",
            )

        try:
            # Route to first available node (simple round-robin for v1)
            node_id = next(iter(self.node_stubs))
            stub = self.node_stubs[node_id]

            # Forward the request to the worker node
            response = stub.ForwardPass(request, timeout=30)

            if response.success:
                return LogitsResponse(
                    request_id=request.request_id,
                    generated_text=response.output.float_data[0] if response.output.float_data else "",
                    success=True,
                )
            else:
                return LogitsResponse(
                    request_id=request.request_id,
                    generated_text="",
                    success=False,
                    error_message=response.error_message,
                )
        except grpc.RpcError as e:
            logger.error(f"Inference routing failed: {e}")
            return LogitsResponse(
                request_id=request.request_id,
                generated_text="",
                success=False,
                error_message=f"Node communication error: {e.details()}",
            )
        except SerializationError as e:
            logger.error(f"Inference serialization error: {e}")
            return LogitsResponse(
                request_id=request.request_id,
                generated_text="",
                success=False,
                error_message=f"Serialization error: {str(e)}",
            )

    def StreamInfer(self, request, context):
        """Stream inference by routing to worker nodes.

        Yields token responses from worker nodes as they become available.
        """
        if not self.node_stubs:
            yield TokenResponse(
                request_id=request.request_id,
                token="",
                is_final=True,
                full_text="",
                success=False,
                error_message="No worker nodes registered",
            )
            return

        try:
            node_id = next(iter(self.node_stubs))
            stub = self.node_stubs[node_id]

            # Forward to worker node and stream back responses
            response = stub.ForwardPass(request, timeout=60)

            if response.success:
                # Stream output tokens if available
                if response.output.float_data:
                    for i, val in enumerate(response.output.float_data):
                        yield TokenResponse(
                            request_id=request.request_id,
                            token=str(val),
                            is_final=(i == len(response.output.float_data) - 1),
                            success=True,
                        )
                else:
                    yield TokenResponse(
                        request_id=request.request_id,
                        token="",
                        is_final=True,
                        success=True,
                    )
            else:
                yield TokenResponse(
                    request_id=request.request_id,
                    token="",
                    is_final=True,
                    full_text="",
                    success=False,
                    error_message=response.error_message,
                )
        except grpc.RpcError as e:
            logger.error(f"Streaming inference failed: {e}")
            yield TokenResponse(
                request_id=request.request_id,
                token="",
                is_final=True,
                success=False,
                error_message=f"Node communication error: {e.details()}",
            )
        except SerializationError as e:
            logger.error(f"Streaming inference serialization error: {e}")
            yield TokenResponse(
                request_id=request.request_id,
                token="",
                is_final=True,
                success=False,
                error_message=f"Serialization error: {str(e)}",
            )


class GRPCServer:
    """Manages gRPC server lifecycle."""

    def __init__(self, port: int, servicer, max_workers: int = 10,
                 use_tls: bool = True, cert_file: Optional[str] = None,
                 key_file: Optional[str] = None, ca_cert: Optional[str] = None):
        self.port = port
        self.servicer = servicer
        self.max_workers = max_workers
        self.use_tls = use_tls
        self.cert_file = cert_file
        self.key_file = key_file
        self.ca_cert = ca_cert
        self._cert_dir = None
        # Propagate TLS settings to the servicer for outgoing connections
        if hasattr(servicer, 'use_tls'):
            servicer.use_tls = use_tls
        if hasattr(servicer, 'ca_cert'):
            servicer.ca_cert = ca_cert
        options = [
            ('grpc.max_send_message_length', 64 * 1024 * 1024),
            ('grpc.max_receive_message_length', 64 * 1024 * 1024),
        ]
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=options,
        )

    def start(self):
        """Start the gRPC server."""
        if isinstance(self.servicer, NodeServiceServicer):
            add_NodeServiceServicer_to_server(self.servicer, self.server)
        elif isinstance(self.servicer, CoordinatorServiceServicer):
            add_CoordinatorServiceServicer_to_server(self.servicer, self.server)

        if self.use_tls:
            if not self.cert_file or not self.key_file:
                from distllm.core.tls import generate_self_signed_certs
                self._cert_dir = "_auto_certs"
                cert_file, key_file, _ = generate_self_signed_certs(self._cert_dir)
            else:
                cert_file, key_file = self.cert_file, self.key_file

            from distllm.core.tls import load_tls_credentials
            credentials = load_tls_credentials(cert_file, key_file)
            self.server.add_secure_port(f"[::]:{self.port}", credentials)
            logger.info(f"gRPC server started on port {self.port} (TLS enabled)")
        else:
            self.server.add_insecure_port(f"[::]:{self.port}")
            logger.info(f"gRPC server started on port {self.port} (TLS disabled)")

        self.server.start()
        return self

    def stop(self, grace: int = 5):
        """Stop the gRPC server and close all tracked channels."""
        self.server.stop(grace)

        if hasattr(self.servicer, 'node_channels'):
            for channel in self.servicer.node_channels.values():
                try:
                    channel.close()
                except Exception as e:
                    logger.debug(f"Error closing channel: {e}")
            self.servicer.node_channels.clear()
        if hasattr(self.servicer, 'node_stubs'):
            self.servicer.node_stubs.clear()

        logger.info(f"gRPC server stopped on port {self.port}")

    def wait_for_termination(self):
        """Block until server is stopped."""
        try:
            self.server.wait_for_termination()
        except KeyboardInterrupt:
            self.stop()


class NodeClient:
    """gRPC client for communicating with worker nodes."""

    def __init__(self, host: str, port: int, max_retries: int = 3, retry_delay: float = 1.0, use_tls: bool = True, ca_cert: Optional[str] = None):
        options = [
            ('grpc.max_send_message_length', 64 * 1024 * 1024),
            ('grpc.max_receive_message_length', 64 * 1024 * 1024),
            # Auto-reconnect options
            ('grpc.keepalive_time_ms', 30000),
            ('grpc.keepalive_timeout_ms', 10000),
            ('grpc.keepalive_permit_without_calls', 1),
            ('grpc.enable_retries', 1),
        ]
        if use_tls:
            if ca_cert:
                from distllm.core.tls import load_tls_channel_credentials
                credentials = load_tls_channel_credentials(ca_cert, host)
            else:
                # Auto-discover self-signed certs
                import os as _os
                auto_ca = _os.path.join("_auto_certs", "ca.crt")
                if _os.path.exists(auto_ca):
                    from distllm.core.tls import load_tls_channel_credentials
                    credentials = load_tls_channel_credentials(auto_ca, host)
                else:
                    # Fallback to insecure if no certs found
                    logger.warning(f"No CA cert found for {host}:{port}, falling back to insecure")
                    self.channel = grpc.insecure_channel(f"{host}:{port}", options=options)
                    self.stub = NodeServiceStub(self.channel)
                    self.max_retries = max_retries
                    self.retry_delay = retry_delay
                    return
            self.channel = grpc.secure_channel(f"{host}:{port}", credentials, options=options)
        else:
            self.channel = grpc.insecure_channel(f"{host}:{port}", options=options)
        self.stub = NodeServiceStub(self.channel)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def health_check(self) -> HealthCheckResponse:
        """Check node health with retry."""
        def _call():
            try:
                return self.stub.HealthCheck(HealthCheckRequest(), timeout=10)
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise GRPCTimeoutError(node_id="unknown", timeout=10)
                raise NodeUnreachableError(
                    node_id="unknown", host=self.host, port=self.port, original_error=e
                )
        return retry_grpc_call(
            _call,
            max_retries=self.max_retries,
            base_delay=self.retry_delay,
            retryable_exceptions=(NodeUnreachableError, GRPCTimeoutError),
        )

    def get_info(self) -> NodeInfo:
        """Get node info with retry."""
        def _call():
            try:
                return self.stub.GetNodeInfo(HealthCheckRequest())
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise GRPCTimeoutError(node_id="unknown", timeout=10)
                raise NodeUnreachableError(
                    node_id="unknown", host=self.host, port=self.port, original_error=e
                )
        return retry_grpc_call(
            _call,
            max_retries=self.max_retries,
            base_delay=self.retry_delay,
            retryable_exceptions=(NodeUnreachableError, GRPCTimeoutError),
        )

    def close(self):
        """Close the gRPC channel."""
        self.channel.close()


# ============================================================================
# Async gRPC implementations using grpc.aio
# ============================================================================


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

    # Reuse GPU stats method from sync version via module-level function
    def _get_gpu_stats(self) -> Tuple[int, int, float, float, bool]:
        """Get actual GPU stats using pynvml if available."""
        # Call the same logic as sync version
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
        from distllm.communication.node_pb2 import PingResponse
        return PingResponse(
            node_id=self.node_id,
            cluster_id="",
            timestamp=int(time.time() * 1000),
            latency_ms=0.0,
        )


class AsyncCoordinatorService(CoordinatorServiceServicer):
    """Async gRPC service implementation for the coordinator using grpc.aio."""

    def __init__(self, quantization_config=None, use_tls: bool = True, ca_cert: Optional[str] = None):
        self.nodes = {}
        self.node_channels = {}
        self.node_stubs = {}
        self.quantization_config = quantization_config
        self._expert_registry = None
        self.use_tls = use_tls
        self.ca_cert = ca_cert

    async def RegisterNode(self, request, context):
        """Register a worker node (async)."""
        import os

        api_key = os.environ.get("GRPC_API_KEY")
        if api_key:
            client_key = None
            for key, value in request.metadata:
                if key == "api_key":
                    client_key = value
                    break
            if client_key != api_key:
                context.set_code(grpc.StatusCode.UNAUTHENTICATED)
                context.set_details("Invalid or missing API key")
                return RegistrationResponse(accepted=False)

        node_info = request.node_info
        node_id = node_info.node_id
        expert_ids = list(request.expert_ids) if request.expert_ids else []

        logger.info(f"Registering node: {node_id} at {node_info.host}:{node_info.port}")
        if expert_ids:
            logger.info(f"Node {node_id} hosts experts: {expert_ids}")

        if self.use_tls:
            if self.ca_cert:
                from distllm.core.tls import load_tls_channel_credentials
                credentials = load_tls_channel_credentials(self.ca_cert, node_info.host)
            else:
                import os as _os
                auto_ca = _os.path.join("_auto_certs", "ca.crt")
                if _os.path.exists(auto_ca):
                    from distllm.core.tls import load_tls_channel_credentials
                    credentials = load_tls_channel_credentials(auto_ca, node_info.host)
                else:
                    logger.warning(f"No CA cert found for {node_info.host}:{node_info.port}, falling back to insecure")
                    channel = grpc.aio.insecure_channel(f"{node_info.host}:{node_info.port}")
                    stub = NodeServiceStub(channel)
                    self.nodes[node_id] = node_info
                    self.node_channels[node_id] = channel
                    self.node_stubs[node_id] = stub
                    response = RegistrationResponse(accepted=True)
                    if expert_ids and self._expert_registry is not None:
                        for eid in expert_ids:
                            self._expert_registry.register_expert(eid, node_id)
                    if self.quantization_config and self.quantization_config.method != "none":
                        proto_q = response.quantization
                        proto_q.method = self.quantization_config.method
                        proto_q.bnb_4bit_compute_dtype = self.quantization_config.bnb_4bit_compute_dtype
                        proto_q.bnb_4bit_quant_type = self.quantization_config.bnb_4bit_quant_type
                        proto_q.bnb_4bit_use_double_quant = self.quantization_config.bnb_4bit_use_double_quant
                        proto_q.llm_int8_threshold = self.quantization_config.llm_int8_threshold
                    return response
            channel = grpc.aio.secure_channel(f"{node_info.host}:{node_info.port}", credentials)
        else:
            channel = grpc.aio.insecure_channel(f"{node_info.host}:{node_info.port}")
        stub = NodeServiceStub(channel)

        self.nodes[node_id] = node_info
        self.node_channels[node_id] = channel
        self.node_stubs[node_id] = stub

        response = RegistrationResponse(accepted=True)

        # Register experts on this node
        if expert_ids and self._expert_registry is not None:
            for eid in expert_ids:
                self._expert_registry.register_expert(eid, node_id)

        if self.quantization_config and self.quantization_config.method != "none":
            proto_q = response.quantization
            proto_q.method = self.quantization_config.method
            proto_q.bnb_4bit_compute_dtype = self.quantization_config.bnb_4bit_compute_dtype
            proto_q.bnb_4bit_quant_type = self.quantization_config.bnb_4bit_quant_type
            proto_q.bnb_4bit_use_double_quant = self.quantization_config.bnb_4bit_use_double_quant
            proto_q.llm_int8_threshold = self.quantization_config.llm_int8_threshold

        return response

    async def Infer(self, request, context):
        """Handle inference request by routing to the appropriate node (async).

        Routes the inference request to registered worker nodes using
        their gRPC ForwardPass endpoints.
        """
        if not self.node_stubs:
            return LogitsResponse(
                request_id=request.request_id,
                generated_text="",
                success=False,
                error_message="No worker nodes registered",
            )

        try:
            node_id = next(iter(self.node_stubs))
            stub = self.node_stubs[node_id]
            response = await stub.ForwardPass(request, timeout=30)

            if response.success:
                return LogitsResponse(
                    request_id=request.request_id,
                    generated_text=response.output.float_data[0] if response.output.float_data else "",
                    success=True,
                )
            else:
                return LogitsResponse(
                    request_id=request.request_id,
                    generated_text="",
                    success=False,
                    error_message=response.error_message,
                )
        except grpc.aio.AioRpcError as e:
            logger.error(f"Async inference routing failed: {e}")
            return LogitsResponse(
                request_id=request.request_id,
                generated_text="",
                success=False,
                error_message=f"Node communication error: {e.details()}",
            )
        except SerializationError as e:
            logger.error(f"Async inference serialization error: {e}")
            return LogitsResponse(
                request_id=request.request_id,
                generated_text="",
                success=False,
                error_message=f"Serialization error: {str(e)}",
            )

    async def StreamInfer(self, request, context):
        """Stream inference by routing to worker nodes (async).

        Yields token responses from worker nodes as they become available.
        """
        if not self.node_stubs:
            yield TokenResponse(
                request_id=request.request_id,
                token="",
                is_final=True,
                full_text="",
                success=False,
                error_message="No worker nodes registered",
            )
            return

        try:
            node_id = next(iter(self.node_stubs))
            stub = self.node_stubs[node_id]
            response = await stub.ForwardPass(request, timeout=60)

            if response.success:
                if response.output.float_data:
                    for i, val in enumerate(response.output.float_data):
                        yield TokenResponse(
                            request_id=request.request_id,
                            token=str(val),
                            is_final=(i == len(response.output.float_data) - 1),
                            success=True,
                        )
                else:
                    yield TokenResponse(
                        request_id=request.request_id,
                        token="",
                        is_final=True,
                        success=True,
                    )
            else:
                yield TokenResponse(
                    request_id=request.request_id,
                    token="",
                    is_final=True,
                    full_text="",
                    success=False,
                    error_message=response.error_message,
                )
        except grpc.aio.AioRpcError as e:
            logger.error(f"Async streaming inference failed: {e}")
            yield TokenResponse(
                request_id=request.request_id,
                token="",
                is_final=True,
                success=False,
                error_message=f"Node communication error: {e.details()}",
            )
        except SerializationError as e:
            logger.error(f"Async streaming inference serialization error: {e}")
            yield TokenResponse(
                request_id=request.request_id,
                token="",
                is_final=True,
                success=False,
                error_message=f"Serialization error: {str(e)}",
            )


class AsyncGRPCServer:
    """Manages async gRPC server lifecycle using grpc.aio."""

    def __init__(self, port: int, servicer, max_workers: int = 10,
                 use_tls: bool = True, cert_file: Optional[str] = None,
                 key_file: Optional[str] = None, ca_cert: Optional[str] = None):
        self.port = port
        self.servicer = servicer
        self.max_workers = max_workers
        self.use_tls = use_tls
        self.cert_file = cert_file
        self.key_file = key_file
        self.ca_cert = ca_cert
        self._cert_dir = None
        self.server = None
        # Propagate TLS settings to the servicer for outgoing connections
        if hasattr(servicer, 'use_tls'):
            servicer.use_tls = use_tls
        if hasattr(servicer, 'ca_cert'):
            servicer.ca_cert = ca_cert
        options = [
            ('grpc.max_send_message_length', 64 * 1024 * 1024),
            ('grpc.max_receive_message_length', 64 * 1024 * 1024),
        ]
        self._server_options = options

    async def start(self):
        """Start the async gRPC server."""
        self.server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers),
            options=self._server_options,
        )

        if isinstance(self.servicer, NodeServiceServicer):
            add_NodeServiceServicer_to_server(self.servicer, self.server)
        elif isinstance(self.servicer, CoordinatorServiceServicer):
            add_CoordinatorServiceServicer_to_server(self.servicer, self.server)

        if self.use_tls:
            if not self.cert_file or not self.key_file:
                from distllm.core.tls import generate_self_signed_certs
                self._cert_dir = "_auto_certs"
                cert_file, key_file, _ = generate_self_signed_certs(self._cert_dir)
            else:
                cert_file, key_file = self.cert_file, self.key_file

            from distllm.core.tls import load_tls_credentials
            credentials = load_tls_credentials(cert_file, key_file)
            self.server.add_secure_port(f"[::]:{self.port}", credentials)
            logger.info(f"Async gRPC server started on port {self.port} (TLS enabled)")
        else:
            self.server.add_insecure_port(f"[::]:{self.port}")
            logger.info(f"Async gRPC server started on port {self.port} (TLS disabled)")

        await self.server.start()
        return self

    async def stop(self, grace: int = 5):
        """Stop the async gRPC server and close all tracked channels."""
        if self.server:
            await self.server.stop(grace)

        if hasattr(self.servicer, 'node_channels'):
            for channel in self.servicer.node_channels.values():
                try:
                    await channel.close()
                except Exception as e:
                    logger.debug(f"Error closing channel: {e}")
            self.servicer.node_channels.clear()
        if hasattr(self.servicer, 'node_stubs'):
            self.servicer.node_stubs.clear()

        logger.info(f"Async gRPC server stopped on port {self.port}")

    async def wait_for_termination(self):
        """Block until server is stopped."""
        if self.server:
            try:
                await self.server.wait_for_termination()
            except KeyboardInterrupt:
                await self.stop()


class AsyncNodeClient:
    """Async gRPC client for communicating with worker nodes using grpc.aio."""

    def __init__(self, host: str, port: int, max_retries: int = 3, retry_delay: float = 1.0, use_tls: bool = True, ca_cert: Optional[str] = None):
        options = [
            ('grpc.max_send_message_length', 64 * 1024 * 1024),
            ('grpc.max_receive_message_length', 64 * 1024 * 1024),
            # Auto-reconnect options
            ('grpc.keepalive_time_ms', 30000),
            ('grpc.keepalive_timeout_ms', 10000),
            ('grpc.keepalive_permit_without_calls', 1),
            ('grpc.enable_retries', 1),
        ]
        self.host = host
        self.port = port
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if use_tls:
            if ca_cert:
                from distllm.core.tls import load_tls_channel_credentials
                credentials = load_tls_channel_credentials(ca_cert, host)
            else:
                # Auto-discover self-signed certs
                import os as _os
                auto_ca = _os.path.join("_auto_certs", "ca.crt")
                if _os.path.exists(auto_ca):
                    from distllm.core.tls import load_tls_channel_credentials
                    credentials = load_tls_channel_credentials(auto_ca, host)
                else:
                    # Fallback to insecure if no certs found
                    logger.warning(f"No CA cert found for {host}:{port}, falling back to insecure")
                    self.channel = grpc.aio.insecure_channel(f"{host}:{port}", options=options)
                    self.stub = NodeServiceStub(self.channel)
                    return
            self.channel = grpc.aio.secure_channel(f"{host}:{port}", credentials, options=options)
        else:
            self.channel = grpc.aio.insecure_channel(f"{host}:{port}", options=options)
        self.stub = NodeServiceStub(self.channel)

    async def health_check(self) -> HealthCheckResponse:
        """Check node health (async) with retry."""
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self.stub.HealthCheck(HealthCheckRequest(), timeout=10)
            except grpc.aio.AioRpcError as e:
                if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    last_exc = GRPCTimeoutError(node_id="unknown", timeout=10)
                else:
                    last_exc = NodeUnreachableError(
                        node_id="unknown", host=self.host, port=self.port, original_error=e
                    )
                if attempt == self.max_retries:
                    raise last_exc
                delay = min(self.retry_delay * (2 ** attempt), 60.0)
                logger.warning(
                    f"health_check failed (attempt {attempt + 1}/{self.max_retries + 1}): "
                    f"{last_exc}, retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    async def get_info(self) -> NodeInfo:
        """Get node info (async) with retry."""
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self.stub.GetNodeInfo(HealthCheckRequest())
            except grpc.aio.AioRpcError as e:
                if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    last_exc = GRPCTimeoutError(node_id="unknown", timeout=10)
                else:
                    last_exc = NodeUnreachableError(
                        node_id="unknown", host=self.host, port=self.port, original_error=e
                    )
                if attempt == self.max_retries:
                    raise last_exc
                delay = min(self.retry_delay * (2 ** attempt), 60.0)
                logger.warning(
                    f"get_info failed (attempt {attempt + 1}/{self.max_retries + 1}): "
                    f"{last_exc}, retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    async def forward(self, request) -> ForwardPassResponse:
        """Run forward pass (async) with retry."""
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self.stub.ForwardPass(request)
            except grpc.aio.AioRpcError as e:
                if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    last_exc = GRPCTimeoutError(node_id="unknown", timeout=10)
                else:
                    last_exc = NodeUnreachableError(
                        node_id="unknown", host=self.host, port=self.port, original_error=e
                    )
                if attempt == self.max_retries:
                    raise last_exc
                delay = min(self.retry_delay * (2 ** attempt), 60.0)
                logger.warning(
                    f"forward failed (attempt {attempt + 1}/{self.max_retries + 1}): "
                    f"{last_exc}, retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    async def close(self):
        """Close the async gRPC channel."""
        await self.channel.close()
