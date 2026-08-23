"""Transport abstraction layer with gRPC and in-process backends.

Provides a common interface (``TransportBackend``) for executing forward
passes against worker nodes, with two implementations:

- ``GrpcTransport`` — wraps the existing gRPC client/server machinery
  (``node_client.forward_request``, ``NodeServer``, etc.).
- ``InProcessTransport`` — calls ``WorkerNode.forward_fn`` directly,
  no network required.  Ideal for testing pipeline logic without
  starting real gRPC servers.

A factory function ``get_transport()`` selects the backend by name::

    transport = get_transport("grpc")
    output = transport.forward(hidden_states, host="10.0.0.1", port=50051)

    transport = get_transport("inprocess")
    transport.start_server(worker_node, port=0)
    output = transport.forward(hidden_states)
"""

from __future__ import annotations

import abc
from typing import Any

import torch

from distllm.dist import node_client
from distllm.dist import node_service


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class TransportBackend(abc.ABC):
    """Abstract transport interface for distributed forward passes.

    Subclasses implement both synchronous and asynchronous forward
    execution, client lifecycle, and server lifecycle.
    """

    @abc.abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: list | None = None,
        request_id: str = "",
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run a synchronous forward pass on a (possibly remote) node.

        Args:
            hidden_states: Input tensor to forward through the node's layers.
            kv_cache: Optional KV cache list of ``(key, value)`` tensor pairs.
            request_id: Opaque request identifier for tracing.
            **kwargs: Backend-specific keyword arguments (e.g. ``host``,
                ``port``, ``cluster_key``, ``timeout_s``).

        Returns:
            Output tensor from the node.
        """
        ...

    @abc.abstractmethod
    async def forward_async(
        self,
        hidden_states: torch.Tensor,
        kv_cache: list | None = None,
        request_id: str = "",
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run an asynchronous forward pass (non-blocking variant).

        Args:
            hidden_states: Input tensor to forward through the node's layers.
            kv_cache: Optional KV cache list of ``(key, value)`` tensor pairs.
            request_id: Opaque request identifier for tracing.
            **kwargs: Backend-specific keyword arguments.

        Returns:
            Output tensor from the node.
        """
        ...

    @abc.abstractmethod
    def create_client(
        self,
        host: str,
        port: int,
        **kwargs: Any,
    ) -> Any:
        """Create a client connection to a worker node.

        Args:
            host: Worker node hostname or IP.
            port: Worker node port.
            **kwargs: Backend-specific keyword arguments (e.g. ``use_tls``,
                ``timeout_s``, ``cluster_key``).

        Returns:
            An opaque client object understood by :meth:`close_client`.
        """
        ...

    @abc.abstractmethod
    def close_client(self, client: Any) -> None:
        """Close a client connection.

        Args:
            client: Object previously returned by :meth:`create_client`.
        """
        ...

    @abc.abstractmethod
    def start_server(
        self,
        worker_node: Any,
        port: int,
        **kwargs: Any,
    ) -> None:
        """Start the server that listens for incoming forward requests.

        Args:
            worker_node: The worker node instance (e.g. ``WorkerNode``)
                whose ``forward_fn`` will be called for each request.
            port: Port to bind the server on.
            **kwargs: Backend-specific keyword arguments (e.g. ``max_workers``,
                ``use_tls``, ``cluster_key``).
        """
        ...

    @abc.abstractmethod
    def stop_server(self, grace: float = 5.0) -> None:
        """Stop the server.

        Args:
            grace: Grace period in seconds for in-flight requests to
                complete before forceful shutdown.
        """
        ...


# ---------------------------------------------------------------------------
# gRPC transport
# ---------------------------------------------------------------------------


class GrpcTransport(TransportBackend):
    """Transport backed by gRPC.

    Delegates to ``distllm.dist.node_client`` (forward_request,
    forward_request_async, create_node_client) and
    ``distllm.dist.node_service.NodeServer`` for the server side.
    """

    def __init__(self) -> None:
        # Server state — set when start_server() is called.
        self._server: node_service.NodeServer | None = None

    # -- forward -----------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: list | None = None,
        request_id: str = "",
        **kwargs: Any,
    ) -> torch.Tensor:
        host: str = kwargs.pop("host")
        port: int = kwargs.pop("port")
        cluster_key: str | None = kwargs.pop("cluster_key", None)
        timeout_s: float = kwargs.pop("timeout_s", 30.0)

        return node_client.forward_request(
            host=host,
            port=port,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            request_id=request_id,
            cluster_key=cluster_key,
            timeout_s=timeout_s,
        )

    async def forward_async(
        self,
        hidden_states: torch.Tensor,
        kv_cache: list | None = None,
        request_id: str = "",
        **kwargs: Any,
    ) -> torch.Tensor:
        host: str = kwargs.pop("host")
        port: int = kwargs.pop("port")
        cluster_key: str | None = kwargs.pop("cluster_key", None)
        timeout_s: float = kwargs.pop("timeout_s", 30.0)

        return await node_client.forward_request_async(
            host=host,
            port=port,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            request_id=request_id,
            cluster_key=cluster_key,
            timeout_s=timeout_s,
        )

    # -- client lifecycle --------------------------------------------------

    def create_client(
        self,
        host: str,
        port: int,
        **kwargs: Any,
    ) -> node_client.NodeClient:
        use_tls: bool = kwargs.pop("use_tls", True)
        timeout_s: float = kwargs.pop("timeout_s", 5.0)
        cluster_key: str | None = kwargs.pop("cluster_key", None)

        return node_client.create_node_client(
            host=host,
            port=port,
            use_tls=use_tls,
            timeout_s=timeout_s,
            cluster_key=cluster_key,
        )

    def close_client(self, client: node_client.NodeClient) -> None:
        client.close()

    # -- server lifecycle --------------------------------------------------

    def start_server(
        self,
        worker_node: Any,
        port: int,
        **kwargs: Any,
    ) -> None:
        max_workers: int = kwargs.pop("max_workers", 4)
        cluster_key: str | None = kwargs.pop("cluster_key", None)
        use_tls: bool = kwargs.pop("use_tls", True)

        self._server = node_service.NodeServer(
            worker_node,
            port=port,
            max_workers=max_workers,
            cluster_key=cluster_key,
        )
        self._server.start(use_tls=use_tls)

    def stop_server(self, grace: float = 5.0) -> None:
        if self._server is not None:
            self._server.stop(grace=grace)
            self._server = None


# ---------------------------------------------------------------------------
# In-process transport (testing)
# ---------------------------------------------------------------------------


class InProcessTransport(TransportBackend):
    """Transport that calls ``WorkerNode.forward_fn`` directly.

    No network or gRPC is involved.  Useful for testing pipeline logic
    (scheduling, micro-batching, KV-cache propagation) without the
    complexity of real gRPC servers.

    Usage::

        transport = InProcessTransport()
        transport.start_server(worker_node, port=0)  # port ignored
        output = transport.forward(hidden_states, kv_cache=kv_cache)
    """

    def __init__(self) -> None:
        self._worker: Any = None  # WorkerNode reference

    # -- forward -----------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: list | None = None,
        request_id: str = "",  # noqa: ARG002 (ignored in-process)
        **kwargs: Any,
    ) -> torch.Tensor:
        if self._worker is None:
            raise RuntimeError(
                "InProcessTransport has no worker node. "
                "Call start_server(worker_node, ...) first."
            )

        # Forward additional keyword args that WorkerNode.forward_fn accepts.
        attention_mask = kwargs.pop("attention_mask", None)
        position_ids = kwargs.pop("position_ids", None)
        input_ids = kwargs.pop("input_ids", None)

        output, _new_kv = self._worker.forward_fn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=kv_cache,
            input_ids=input_ids,
        )
        return output

    async def forward_async(
        self,
        hidden_states: torch.Tensor,
        kv_cache: list | None = None,
        request_id: str = "",
        **kwargs: Any,
    ) -> torch.Tensor:
        # For in-process, the async variant simply delegates to the
        # synchronous call in a thread executor so the caller's event
        # loop is not blocked.
        import asyncio

        loop = asyncio.get_running_loop()
        fn = partial(self.forward, hidden_states, kv_cache, request_id, **kwargs)
        return await loop.run_in_executor(None, fn)

    # -- client lifecycle --------------------------------------------------

    def create_client(
        self,
        host: str = "",  # noqa: ARG002 (ignored in-process)
        port: int = 0,  # noqa: ARG002 (ignored in-process)
        **kwargs: Any,  # noqa: ARG002 (ignored in-process)
    ) -> Any:
        """Return the worker node itself as the "client"."""
        if self._worker is None:
            raise RuntimeError(
                "InProcessTransport has no worker node. "
                "Call start_server(worker_node, ...) first."
            )
        return self._worker

    def close_client(self, client: Any) -> None:
        pass  # No-op — no network resources to release.

    # -- server lifecycle --------------------------------------------------

    def start_server(
        self,
        worker_node: Any,
        port: int = 0,  # noqa: ARG002 (ignored in-process)
        **kwargs: Any,  # noqa: ARG002 (ignored in-process)
    ) -> None:
        """Store a reference to *worker_node* for direct calls.

        The *port* and any additional kwargs are silently ignored —
        no actual server is started.
        """
        self._worker = worker_node

    def stop_server(self, grace: float = 5.0) -> None:  # noqa: ARG002
        """Release the worker reference."""
        self._worker = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_TRANSPORT_REGISTRY: dict[str, type[TransportBackend]] = {
    "grpc": GrpcTransport,
    "inprocess": InProcessTransport,
}


def get_transport(name: str = "grpc") -> TransportBackend:
    """Factory: return a :class:`TransportBackend` instance by name.

    Args:
        name: Backend name — ``"grpc"`` (default) or ``"inprocess"``.

    Returns:
        A new instance of the requested transport backend.

    Raises:
        ValueError: If *name* is not a known backend.
    """
    cls = _TRANSPORT_REGISTRY.get(name)
    if cls is None:
        known = ", ".join(sorted(_TRANSPORT_REGISTRY))
        raise ValueError(
            f"Unknown transport backend {name!r}. "
            f"Known backends: {known}"
        )
    return cls()


# ---------------------------------------------------------------------------
# Bandwidth estimation
# ---------------------------------------------------------------------------


def estimate_bandwidth(
    host: str,
    port: int,
    sample_size_bytes: int = 8 * 1024 * 1024,
    timeout_s: float = 5.0,
) -> float:
    """Estimate network bandwidth to a peer node.

    Sends a probe payload of *sample_size_bytes* and measures transfer
    time.  Returns estimated bandwidth in Gbps.  Returns ``0.0`` if the
    probe fails or times out.

    The result is cached (per host) so repeated calls are instantaneous
    within a TTL window.
    """
    import os as _os
    import time as _time

    cache = _os.environ.get("_DISTLLM_BW_CACHE", "")
    if cache:
        try:
            cached_host, cached_bw, cached_at = cache.split(",")
            if cached_host == host and _time.time() - float(cached_at) < 60.0:
                return float(cached_bw)
        except (ValueError, OSError):
            pass

    try:
        import torch
        data = torch.randn(sample_size_bytes // 4, dtype=torch.float32)
        from distllm.dist.node_client import forward_request

        t0 = _time.monotonic()
        _ = forward_request(host, port, data, timeout_s=timeout_s)
        elapsed = _time.monotonic() - t0
        if elapsed <= 0:
            return 0.0
        bw_gbps = (sample_size_bytes * 8) / (elapsed * 1e9)
        _os.environ["_DISTLLM_BW_CACHE"] = f"{host},{bw_gbps},{_time.time()}"
        return round(bw_gbps, 2)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# P2PTransport ABC — unified interface for WebRTC / QUIC / TCP
# ---------------------------------------------------------------------------


class P2PMessage:
    """A single message sent over a P2P transport."""
    data: bytes
    seq: int
    ack: bool = False


class P2PTransport(abc.ABC):
    """Abstract base for peer-to-peer transports.

    Implementations: WebRTC (browser nodes), QUIC (WAN-optimised),
    plain TCP (fallback).  Each provides connection-oriented, ordered
    delivery with optional reliability guarantees.
    """

    @abc.abstractmethod
    async def connect(self, peer_id: str, address: str) -> None:
        """Establish a connection to a peer."""

    @abc.abstractmethod
    async def send(self, peer_id: str, data: bytes) -> None:
        """Send data to a connected peer."""

    @abc.abstractmethod
    async def recv(self, peer_id: str) -> bytes:
        """Receive data from a connected peer (blocks until available)."""

    @abc.abstractmethod
    async def disconnect(self, peer_id: str) -> None:
        """Close the connection to a peer."""

    @abc.abstractmethod
    async def health(self, peer_id: str) -> bool:
        """Check whether the connection to *peer_id* is alive."""


__all__ = [
    "GrpcTransport",
    "InProcessTransport",
    "P2PTransport",
    "TransportBackend",
    "estimate_bandwidth",
    "get_transport",
]
