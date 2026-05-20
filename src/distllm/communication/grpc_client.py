"""gRPC server and client implementations.

Contains sync and async GRPCServer, NodeClient, and AsyncNodeClient
classes that manage server lifecycle and client connections.
"""

import asyncio
import grpc
import pathlib
import sys
from concurrent import futures
from loguru import logger

from distllm.communication.node_pb2 import (
    HealthCheckRequest, HealthCheckResponse, NodeInfo, ForwardPassResponse,
)
from distllm.communication.node_pb2_grpc import (
    NodeServiceServicer, NodeServiceStub, CoordinatorServiceServicer,
    add_NodeServiceServicer_to_server, add_CoordinatorServiceServicer_to_server,
)
from distllm.errors.types import NodeUnreachableError, GRPCTimeoutError
from distllm.errors.retry import retry_grpc_call


_MAX_GRPC_MESSAGE_BYTES = (2 * 1024 * 1024 * 1024) - 1


def _node_service_stub_class():
    shim = sys.modules.get("distllm.communication.grpc")
    if shim is not None and hasattr(shim, "NodeServiceStub"):
        return shim.NodeServiceStub
    return NodeServiceStub


class GRPCServer:
    """Manages gRPC server lifecycle."""

    def __init__(self, port: int, servicer, max_workers: int = 10,
                 use_tls: bool = True, cert_file: str | None = None,
                 key_file: str | None = None, ca_cert: str | None = None):
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
            ('grpc.max_send_message_length', _MAX_GRPC_MESSAGE_BYTES),
            ('grpc.max_receive_message_length', _MAX_GRPC_MESSAGE_BYTES),
            ('grpc.default_compression_algorithm', grpc.Compression.Gzip),
        ]
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=options,
            compression=grpc.Compression.Gzip,
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
                cert_file = str(pathlib.Path(self.cert_file).resolve())
                key_file = str(pathlib.Path(self.key_file).resolve())
                for path, name in [(cert_file, "cert"), (key_file, "key")]:
                    if not pathlib.Path(path).exists():
                        raise FileNotFoundError(f"TLS {name} file not found: {path}")

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
                    import threading
                    def _close():
                        try:
                            channel.close()
                        except Exception:
                            pass
                    t = threading.Thread(target=_close, daemon=True)
                    t.start()
                    t.join(timeout=grace)
                except Exception as e:
                    logger.debug(f"Error closing channel: {e}")
            self.servicer.node_channels.clear()
        if hasattr(self.servicer, 'node_stubs'):
            self.servicer.node_stubs.clear()

        # Clean up auto-generated TLS certificates
        if self._cert_dir:
            import shutil
            try:
                shutil.rmtree(self._cert_dir, ignore_errors=True)
                logger.debug(f"Cleaned up auto-generated TLS certs: {self._cert_dir}")
            except Exception as e:
                logger.debug(f"Failed to clean up TLS certs: {e}")

        logger.info(f"gRPC server stopped on port {self.port}")

    def wait_for_termination(self):
        """Block until server is stopped."""
        try:
            self.server.wait_for_termination()
        except KeyboardInterrupt:
            self.stop()


class NodeClient:
    """gRPC client for communicating with worker nodes."""

    def __init__(self, host: str, port: int, max_retries: int = 3, retry_delay: float = 1.0, use_tls: bool = True, ca_cert: str | None = None):
        self.host = host
        self.port = port
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        options = [
            ('grpc.max_send_message_length', _MAX_GRPC_MESSAGE_BYTES),
            ('grpc.max_receive_message_length', _MAX_GRPC_MESSAGE_BYTES),
            ('grpc.default_compression_algorithm', grpc.Compression.Gzip),
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
                import os as _os
                auto_ca = _os.path.join("_auto_certs", "ca.crt")
                if _os.path.exists(auto_ca):
                    from distllm.core.tls import load_tls_channel_credentials
                    credentials = load_tls_channel_credentials(auto_ca, host)
                else:
                    raise NodeUnreachableError(
                        node_id="unknown", host=host, port=port,
                        original_error=ConnectionError(
                            f"TLS enabled but no CA cert found for {host}:{port}. "
                            "Pass --insecure to the worker node to disable TLS."
                        ),
                    )
            self.channel = grpc.secure_channel(f"{host}:{port}", credentials, options=options)
        else:
            self.channel = grpc.insecure_channel(f"{host}:{port}", options=options)
        self.stub = _node_service_stub_class()(self.channel)
        self._closed = False

    def health_check(self, timeout: float = 10) -> HealthCheckResponse:
        """Check node health with retry."""
        def _call():
            try:
                return self.stub.HealthCheck(HealthCheckRequest(), timeout=timeout)
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise GRPCTimeoutError(node_id="unknown", timeout=timeout)
                raise NodeUnreachableError(
                    node_id="unknown", host=self.host, port=self.port, original_error=e
                )
        return retry_grpc_call(
            _call,
            max_retries=self.max_retries,
            base_delay=self.retry_delay,
            retryable_exceptions=(NodeUnreachableError, GRPCTimeoutError),
        )

    def get_info(self, timeout: float = 10) -> NodeInfo:
        """Get node info with retry."""
        def _call():
            try:
                return self.stub.GetNodeInfo(HealthCheckRequest(), timeout=timeout)
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise GRPCTimeoutError(node_id="unknown", timeout=timeout)
                raise NodeUnreachableError(
                    node_id="unknown", host=self.host, port=self.port, original_error=e
                )
        return retry_grpc_call(
            _call,
            max_retries=self.max_retries,
            base_delay=self.retry_delay,
            retryable_exceptions=(NodeUnreachableError, GRPCTimeoutError),
        )

    def forward(self, request, timeout: float = 30) -> ForwardPassResponse:
        """Run a blocking forward pass with retry."""
        def _call():
            try:
                return self.stub.ForwardPass(request, timeout=timeout)
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise GRPCTimeoutError(node_id="unknown", timeout=timeout)
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
        """Close the gRPC channel with a timeout to prevent indefinite blocking."""
        if self._closed:
            return
        import threading
        def _close():
            try:
                self.channel.close()
            except Exception:
                pass
        t = threading.Thread(target=_close, daemon=True)
        t.start()
        t.join(timeout=5)
        if t.is_alive():
            logger.warning(f"Channel close timed out for {self.host}:{self.port}")
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class AsyncGRPCServer:
    """Manages async gRPC server lifecycle using grpc.aio."""

    def __init__(self, port: int, servicer, max_workers: int = 10,
                 use_tls: bool = True, cert_file: str | None = None,
                 key_file: str | None = None, ca_cert: str | None = None):
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
                cert_file = str(pathlib.Path(self.cert_file).resolve())
                key_file = str(pathlib.Path(self.key_file).resolve())
                for path, name in [(cert_file, "cert"), (key_file, "key")]:
                    if not pathlib.Path(path).exists():
                        raise FileNotFoundError(f"TLS {name} file not found: {path}")

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

        # Clean up auto-generated TLS certificates
        if self._cert_dir:
            import shutil
            try:
                shutil.rmtree(self._cert_dir, ignore_errors=True)
                logger.debug(f"Cleaned up auto-generated TLS certs: {self._cert_dir}")
            except Exception as e:
                logger.debug(f"Failed to clean up TLS certs: {e}")

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

    def __init__(self, host: str, port: int, max_retries: int = 3, retry_delay: float = 1.0, use_tls: bool = True, ca_cert: str | None = None):
        self.host = host
        self.port = port
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        options = [
            ('grpc.max_send_message_length', _MAX_GRPC_MESSAGE_BYTES),
            ('grpc.max_receive_message_length', _MAX_GRPC_MESSAGE_BYTES),
            ('grpc.default_compression_algorithm', grpc.Compression.Gzip),
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
                import os as _os
                auto_ca = _os.path.join("_auto_certs", "ca.crt")
                if _os.path.exists(auto_ca):
                    from distllm.core.tls import load_tls_channel_credentials
                    credentials = load_tls_channel_credentials(auto_ca, host)
                else:
                    raise NodeUnreachableError(
                        node_id="unknown", host=host, port=port,
                        original_error=ConnectionError(
                            f"TLS enabled but no CA cert found for {host}:{port}. "
                            "Pass --insecure to the worker node to disable TLS."
                        ),
                    )
            self.channel = grpc.aio.secure_channel(f"{host}:{port}", credentials, options=options)
        else:
            self.channel = grpc.aio.insecure_channel(f"{host}:{port}", options=options)
        self.stub = _node_service_stub_class()(self.channel)
        self._closed = False

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
        if self._closed:
            return
        await self.channel.close()
        self._closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()
