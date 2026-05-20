"""gRPC server and client implementations.

Contains sync and async GRPCServer, NodeClient, and AsyncNodeClient
classes that manage server lifecycle and client connections.
"""

import asyncio
import grpc
import os
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
                env = os.environ.get("DISTLLM_ENV", "development")
                if env == "production":
                    raise RuntimeError(
                        "TLS enabled with no cert_file/key_file in production mode. "
                        "Set cert_file and key_file explicitly."
                    )
                logger.critical(
                    "SECURITY: Using auto-generated self-signed TLS certificates. "
                    "This provides NO protection against MITM attacks. "
                    "For production, set use_tls=true, cert_file, and key_file explicitly."
                )
                from distllm.core.tls import generate_self_signed_certs
                self._cert_dir = "_auto_certs"
                cert_file, key_file, ca_cert_file = generate_self_signed_certs(self._cert_dir)
            else:
                cert_file = str(pathlib.Path(self.cert_file).resolve())
                key_file = str(pathlib.Path(self.key_file).resolve())
                for path, name in [(cert_file, "cert"), (key_file, "key")]:
                    if not pathlib.Path(path).exists():
                        raise FileNotFoundError(f"TLS {name} file not found: {path}")
                ca_cert_file = str(pathlib.Path(self.ca_cert).resolve()) if self.ca_cert else None

            from distllm.core.tls import load_tls_credentials
            credentials = load_tls_credentials(cert_file, key_file, ca_cert_file=ca_cert_file)
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


def _build_grpc_options() -> list:
    """Build shared gRPC channel options."""
    return [
        ('grpc.max_send_message_length', _MAX_GRPC_MESSAGE_BYTES),
        ('grpc.max_receive_message_length', _MAX_GRPC_MESSAGE_BYTES),
        ('grpc.default_compression_algorithm', grpc.Compression.Gzip),
        ('grpc.keepalive_time_ms', 30000),
        ('grpc.keepalive_timeout_ms', 10000),
        ('grpc.keepalive_permit_without_calls', 1),
        ('grpc.enable_retries', 1),
    ]


def _convert_grpc_error(e: Exception, host: str, port: int, timeout: float):
    """Convert a gRPC exception into a project-specific exception."""
    if isinstance(e, asyncio.TimeoutError):
        return GRPCTimeoutError(node_id="unknown", timeout=timeout)
    if isinstance(e, (grpc.RpcError, grpc.aio.AioRpcError)):
        if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
            return GRPCTimeoutError(node_id="unknown", timeout=timeout)
        return NodeUnreachableError(
            node_id="unknown", host=host, port=port, original_error=e,
        )
    return NodeUnreachableError(
        node_id="unknown", host=host, port=port, original_error=e,
    )


class UnifiedNodeClient:
    """gRPC client base with shared TLS/channel setup.

    Subclasses override _create_sync_channel / _create_aio_channel
    to choose between grpc (sync) and grpc.aio (async).

    NodeClient (sync) and AsyncNodeClient (async) share:
      - __init__ TLS/credentials setup
      - _convert_grpc_error exception handling
      - _build_grpc_options channel options
    """

    def __init__(self, host: str, port: int, max_retries: int = 3,
                 retry_delay: float = 1.0, use_tls: bool = True,
                 ca_cert: str | None = None, timeout: float = 30.0):
        self.host = host
        self.port = port
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self._closed = False

        options = _build_grpc_options()
        target = f"{host}:{port}"

        if use_tls:
            auto_client_cert = os.path.join("_auto_certs", "client.crt")
            auto_client_key = os.path.join("_auto_certs", "client.key")
            if ca_cert:
                from distllm.core.tls import load_tls_channel_credentials
                credentials = load_tls_channel_credentials(ca_cert, host)
            else:
                auto_ca = os.path.join("_auto_certs", "ca.crt")
                if os.path.exists(auto_ca):
                    from distllm.core.tls import load_tls_channel_credentials
                    client_cert = auto_client_cert if os.path.exists(auto_client_cert) else None
                    client_key = auto_client_key if os.path.exists(auto_client_key) else None
                    credentials = load_tls_channel_credentials(
                        auto_ca, host,
                        client_cert_file=client_cert,
                        client_key_file=client_key,
                    )
                else:
                    raise NodeUnreachableError(
                        node_id="unknown", host=host, port=port,
                        original_error=ConnectionError(
                            f"TLS enabled but no CA cert found for {host}:{port}. "
                            "Pass --insecure to the worker node to disable TLS."
                        ),
                    )
            self.channel = self._create_secure_channel(target, credentials, options)
        else:
            self.channel = self._create_insecure_channel(target, options)
        self.stub = _node_service_stub_class()(self.channel)

    def _create_secure_channel(self, target, credentials, options):
        """Override in subclass — return a secure gRPC channel."""
        raise NotImplementedError

    def _create_insecure_channel(self, target, options):
        """Override in subclass — return an insecure gRPC channel."""
        raise NotImplementedError

    def _close_channel(self):
        """Close the underlying channel — handles both sync and async close."""
        if self._closed:
            return
        self._closed = True
        close_fn = self.channel.close
        if asyncio.iscoroutinefunction(close_fn):
            try:
                asyncio.run(close_fn())
            except RuntimeError:
                pass
        else:
            try:
                close_fn()
            except Exception:
                pass

    def __del__(self):
        try:
            self._close_channel()
        except Exception:
            pass


class NodeClient(UnifiedNodeClient):
    """Sync gRPC client with retry — uses grpc (not grpc.aio) channels."""

    def _create_secure_channel(self, target, credentials, options):
        return grpc.secure_channel(target, credentials, options=options)

    def _create_insecure_channel(self, target, options):
        return grpc.insecure_channel(target, options=options)

    def _retry(self, rpc_fn):
        """Execute rpc_fn with exponential backoff (sync)."""
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                return rpc_fn()
            except (NodeUnreachableError, GRPCTimeoutError, grpc.RpcError) as e:
                last_exc = e if isinstance(e, (NodeUnreachableError, GRPCTimeoutError)) else _convert_grpc_error(e, self.host, self.port, 0)
                if attempt == self.max_retries:
                    raise last_exc
                import time
                delay = min(self.retry_delay * (2 ** attempt), 60.0)
                logger.warning(
                    f"gRPC call failed (attempt {attempt + 1}/{self.max_retries + 1}): "
                    f"{last_exc}, retrying in {delay:.1f}s"
                )
                time.sleep(delay)
        raise last_exc  # type: ignore[misc]

    def health_check(self, timeout: float = 10) -> HealthCheckResponse:
        def _call():
            try:
                return self.stub.HealthCheck(HealthCheckRequest(), timeout=timeout)
            except grpc.RpcError as e:
                raise _convert_grpc_error(e, self.host, self.port, timeout)
        return self._retry(_call)

    def get_info(self, timeout: float = 10) -> NodeInfo:
        def _call():
            try:
                return self.stub.GetNodeInfo(HealthCheckRequest(), timeout=timeout)
            except grpc.RpcError as e:
                raise _convert_grpc_error(e, self.host, self.port, timeout)
        return self._retry(_call)

    def forward(self, request, timeout: float = 30) -> ForwardPassResponse:
        def _call():
            try:
                return self.stub.ForwardPass(request, timeout=timeout)
            except grpc.RpcError as e:
                raise _convert_grpc_error(e, self.host, self.port, timeout)
        return self._retry(_call)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.channel.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class AsyncNodeClient(UnifiedNodeClient):
    """Async gRPC client with retry — uses grpc.aio channels."""

    def _create_secure_channel(self, target, credentials, options):
        return grpc.aio.secure_channel(target, credentials, options=options)

    def _create_insecure_channel(self, target, options):
        return grpc.aio.insecure_channel(target, options=options)

    async def _retry_async(self, rpc_coro, timeout: float, method_name: str):
        """Execute rpc_coro with exponential backoff (async)."""
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                return await rpc_coro()
            except (NodeUnreachableError, GRPCTimeoutError) as e:
                last_exc = e
                if attempt == self.max_retries:
                    raise last_exc
                delay = min(self.retry_delay * (2 ** attempt), 60.0)
                logger.warning(
                    f"{method_name} failed (attempt {attempt + 1}/{self.max_retries + 1}): "
                    f"{last_exc}, retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
            except Exception as e:
                last_exc = _convert_grpc_error(e, self.host, self.port, timeout)
                if attempt == self.max_retries:
                    raise last_exc
                delay = min(self.retry_delay * (2 ** attempt), 60.0)
                logger.warning(
                    f"{method_name} failed (attempt {attempt + 1}/{self.max_retries + 1}): "
                    f"{last_exc}, retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    async def health_check(self, timeout: float = 10) -> HealthCheckResponse:
        return await self._retry_async(
            lambda: self.stub.HealthCheck(HealthCheckRequest(), timeout=timeout),
            timeout, "health_check",
        )

    async def get_info(self, timeout: float = 10) -> NodeInfo:
        return await self._retry_async(
            lambda: self.stub.GetNodeInfo(HealthCheckRequest(), timeout=timeout),
            timeout, "get_info",
        )

    async def forward(self, request, timeout: float | None = None) -> ForwardPassResponse:
        deadline = timeout if timeout is not None else self.timeout
        return await self._retry_async(
            lambda: asyncio.wait_for(
                self.stub.ForwardPass(request, timeout=deadline),
                timeout=deadline,
            ),
            deadline, "forward",
        )

    async def close(self):
        if self._closed:
            return
        self._closed = True
        await self.channel.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()


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
            ('grpc.max_send_message_length', _MAX_GRPC_MESSAGE_BYTES),
            ('grpc.max_receive_message_length', _MAX_GRPC_MESSAGE_BYTES),
            ('grpc.default_compression_algorithm', grpc.Compression.Gzip),
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
                cert_file, key_file, ca_cert_file = generate_self_signed_certs(self._cert_dir)
            else:
                cert_file = str(pathlib.Path(self.cert_file).resolve())
                key_file = str(pathlib.Path(self.key_file).resolve())
                for path, name in [(cert_file, "cert"), (key_file, "key")]:
                    if not pathlib.Path(path).exists():
                        raise FileNotFoundError(f"TLS {name} file not found: {path}")
                ca_cert_file = str(pathlib.Path(self.ca_cert).resolve()) if self.ca_cert else None

            from distllm.core.tls import load_tls_credentials
            credentials = load_tls_credentials(cert_file, key_file, ca_cert_file=ca_cert_file)
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


