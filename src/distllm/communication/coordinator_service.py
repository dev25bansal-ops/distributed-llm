"""Coordinator gRPC service implementations.

Contains sync and async CoordinatorService servicers that handle
node registration, inference routing, and streaming inference.
"""

import grpc
import hmac
import os
import sys
from loguru import logger

from distllm.communication.node_pb2 import (
    LogitsResponse, TokenResponse, RegistrationResponse,
)
from distllm.communication.node_pb2_grpc import CoordinatorServiceServicer, NodeServiceStub
from distllm.errors import SerializationError


_MAX_GRPC_MESSAGE_BYTES = (2 * 1024 * 1024 * 1024) - 1  # ~2 GB


def _node_service_stub_class():
    shim = sys.modules.get("distllm.communication.grpc")
    if shim is not None and hasattr(shim, "NodeServiceStub"):
        return shim.NodeServiceStub
    return NodeServiceStub


def _auth_is_required() -> bool:
    if os.environ.get("GRPC_API_KEY"):
        return True
    return os.environ.get("GRPC_AUTH_REQUIRE", "").lower() in {"1", "true", "yes"}


def _request_metadata(request) -> list[tuple[str, str]]:
    metadata = getattr(request, "metadata", [])
    try:
        return list(metadata)
    except TypeError:
        return []


class CoordinatorService(CoordinatorServiceServicer):
    """gRPC service implementation for the coordinator."""

    def __init__(self, quantization_config=None, use_tls: bool = False, ca_cert: str | None = None):
        import threading
        self.nodes = {}
        self.node_channels = {}
        self.node_stubs = {}
        self._nodes_lock = threading.Lock()
        self.quantization_config = quantization_config
        self._expert_registry = None
        self.use_tls = use_tls
        self.ca_cert = ca_cert

    def RegisterNode(self, request, context):
        """Register a worker node."""
        api_key = os.environ.get("GRPC_API_KEY")

        if _auth_is_required():
            if not api_key:
                context.set_code(grpc.StatusCode.UNAUTHENTICATED)
                context.set_details("Node registration authentication is required, but GRPC_API_KEY is not set")
                return RegistrationResponse(accepted=False)
            client_key = None
            for key, value in _request_metadata(request):
                if key == "api_key":
                    client_key = value
                    break
            if not hmac.compare_digest(client_key or "", api_key):
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
            import os as _os
            auto_client_cert = _os.path.join("_auto_certs", "client.crt")
            auto_client_key = _os.path.join("_auto_certs", "client.key")
            if self.ca_cert:
                from distllm.core.tls import load_tls_channel_credentials
                credentials = load_tls_channel_credentials(self.ca_cert, node_info.host)
            else:
                auto_ca = _os.path.join("_auto_certs", "ca.crt")
                if _os.path.exists(auto_ca):
                    from distllm.core.tls import load_tls_channel_credentials
                    client_cert = auto_client_cert if _os.path.exists(auto_client_cert) else None
                    client_key = auto_client_key if _os.path.exists(auto_client_key) else None
                    credentials = load_tls_channel_credentials(
                        auto_ca, node_info.host,
                        client_cert_file=client_cert,
                        client_key_file=client_key,
                    )
                else:
                    # Security: Do NOT silently downgrade to insecure channel.
                    context.set_code(grpc.StatusCode.UNAVAILABLE)
                    context.set_details(
                        f"TLS enabled but no CA certificate available for {node_info.host}:{node_info.port}. "
                        "Configure TLS certificates or start with TLS disabled for development only."
                    )
                    return RegistrationResponse(accepted=False)
            channel = grpc.aio.secure_channel(
                f"{node_info.host}:{node_info.port}",
                credentials,
                options=[
                    ("grpc.max_send_message_length", _MAX_GRPC_MESSAGE_BYTES),
                    ("grpc.max_receive_message_length", _MAX_GRPC_MESSAGE_BYTES),
                    ("grpc.default_method_deadline", 30.0),
                ],
            )
        else:
            channel = grpc.insecure_channel(
                f"{node_info.host}:{node_info.port}",
                options=[
                    ("grpc.max_send_message_length", _MAX_GRPC_MESSAGE_BYTES),
                    ("grpc.max_receive_message_length", _MAX_GRPC_MESSAGE_BYTES),
                    ("grpc.default_method_deadline", 30.0),
                ],
            )
        stub = _node_service_stub_class()(channel)

        old_channel = None
        with self._nodes_lock:
            old_channel = self.node_channels.get(node_id)
            self.nodes[node_id] = node_info
            self.node_channels[node_id] = channel
            self.node_stubs[node_id] = stub
        if old_channel is not None:
            old_channel.close()

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
        try:
            with self._nodes_lock:
                if not self.node_stubs:
                    return LogitsResponse(
                        request_id=request.request_id,
                        generated_text="",
                        success=False,
                        error_message="No worker nodes registered",
                    )
                node_id = next(iter(self.node_stubs))
                stub = self.node_stubs[node_id]

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
        try:
            with self._nodes_lock:
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
                node_id = next(iter(self.node_stubs))
                stub = self.node_stubs[node_id]

            response = stub.ForwardPass(request, timeout=60)

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


class AsyncCoordinatorService(CoordinatorServiceServicer):
    """Async gRPC service implementation for the coordinator using grpc.aio."""

    def __init__(self, quantization_config=None, use_tls: bool = False, ca_cert: str | None = None):
        import threading
        self.nodes = {}
        self.node_channels = {}
        self.node_stubs = {}
        self._nodes_lock = threading.Lock()
        self.quantization_config = quantization_config
        self._expert_registry = None
        self.use_tls = use_tls
        self.ca_cert = ca_cert

    async def RegisterNode(self, request, context):
        """Register a worker node (async)."""
        api_key = os.environ.get("GRPC_API_KEY")

        if _auth_is_required():
            if not api_key:
                context.set_code(grpc.StatusCode.UNAUTHENTICATED)
                context.set_details("Node registration authentication is required, but GRPC_API_KEY is not set")
                return RegistrationResponse(accepted=False)
            client_key = None
            for key, value in _request_metadata(request):
                if key == "api_key":
                    client_key = value
                    break
            if not hmac.compare_digest(client_key or "", api_key):
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
            import os as _os
            auto_client_cert = _os.path.join("_auto_certs", "client.crt")
            auto_client_key = _os.path.join("_auto_certs", "client.key")
            if self.ca_cert:
                from distllm.core.tls import load_tls_channel_credentials
                credentials = load_tls_channel_credentials(self.ca_cert, node_info.host)
            else:
                auto_ca = _os.path.join("_auto_certs", "ca.crt")
                if _os.path.exists(auto_ca):
                    from distllm.core.tls import load_tls_channel_credentials
                    client_cert = auto_client_cert if _os.path.exists(auto_client_cert) else None
                    client_key = auto_client_key if _os.path.exists(auto_client_key) else None
                    credentials = load_tls_channel_credentials(
                        auto_ca, node_info.host,
                        client_cert_file=client_cert,
                        client_key_file=client_key,
                    )
                else:
                    # Security: Do NOT silently downgrade to insecure channel.
                    context.set_code(grpc.StatusCode.UNAVAILABLE)
                    context.set_details(
                        f"TLS enabled but no CA certificate available for {node_info.host}:{node_info.port}. "
                        "Configure TLS certificates or start with TLS disabled for development only."
                    )
                    return RegistrationResponse(accepted=False)
            channel = grpc.secure_channel(
                f"{node_info.host}:{node_info.port}",
                credentials,
                options=[
                    ("grpc.max_send_message_length", _MAX_GRPC_MESSAGE_BYTES),
                    ("grpc.max_receive_message_length", _MAX_GRPC_MESSAGE_BYTES),
                ],
            )
        else:
            channel = grpc.aio.insecure_channel(
                f"{node_info.host}:{node_info.port}",
                options=[
                    ("grpc.max_send_message_length", _MAX_GRPC_MESSAGE_BYTES),
                    ("grpc.max_receive_message_length", _MAX_GRPC_MESSAGE_BYTES),
                ],
            )
        stub = _node_service_stub_class()(channel)

        old_channel = None
        with self._nodes_lock:
            old_channel = self.node_channels.get(node_id)
            self.nodes[node_id] = node_info
            self.node_channels[node_id] = channel
            self.node_stubs[node_id] = stub
        if old_channel is not None:
            await old_channel.close()

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
        try:
            with self._nodes_lock:
                if not self.node_stubs:
                    return LogitsResponse(
                        request_id=request.request_id,
                        generated_text="",
                        success=False,
                        error_message="No worker nodes registered",
                    )
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
        try:
            with self._nodes_lock:
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
