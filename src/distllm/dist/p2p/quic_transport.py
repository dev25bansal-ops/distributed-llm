"""QUIC-based transport for P2P communication with multiplexed streams.

Provides:
- Multiplexed bidirectional streams over QUIC
- Stream prioritization: gossip > metadata > data
- 0-RTT reconnection support via session ticket caching
- Congestion control via QUIC's built-in CC

Usage::

    transport = QuicTransport(node_id="node-1")
    await transport.listen("localhost", 50053)

    # Client side
    conn = await transport.connect("remote-host", 50053)
    await conn.send(StreamPriority.GOSSIP, b"hello")
    priority, data = await conn.recv()

    # Accept incoming
    conn = await transport.accept()

    # Broadcast to all connected peers
    await transport.send_stream(StreamPriority.GOSSIP, b"advertise")
    priority, data, peer_id = await transport.recv_stream()

Transport auto-selection
------------------------
Use :func:`get_optimal_transport` to obtain the best available transport
(aioquic when installed, HTTP fallback otherwise)::

    transport_cls = get_optimal_transport()
"""

from __future__ import annotations

import asyncio
import enum
import os
import struct
import ssl
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional, Tuple
from loguru import logger

# ---------------------------------------------------------------------------
# Optional aioquic import
# ---------------------------------------------------------------------------
try:
    from aioquic.quic.connection import QuicConnection as _QuicConn
    from aioquic.quic.configuration import QuicConfiguration
    from aioquic.quic.events import (
        StreamDataReceived,
        HandshakeCompleted,
        ConnectionTerminated,
        SessionTicketReceived,
    )
    from aioquic.asyncio import connect as _quic_connect
    from aioquic.asyncio import serve as _quic_serve
    from aioquic.asyncio.protocol import QuicConnectionProtocol
    from aioquic.tls import SessionTicket

    HAS_AIOQUIC = True
except ImportError:  # pragma: no cover
    HAS_AIOQUIC = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Wire format: [1-byte priority][4-byte length][payload]
_MESSAGE_HEADER_FMT = "!BI"
_HEADER_SIZE = struct.calcsize(_MESSAGE_HEADER_FMT)
_DEFAULT_TIMEOUT = 30.0
_SENTINEL_PRIORITY = 255  # Lowest possible priority for close sentinel


class StreamPriority(enum.IntEnum):
    """Priority levels for multiplexed QUIC streams.

    Lower numeric value = higher scheduling priority.

    Members:
        GOSSIP (0): Highest priority — gossip protocol messages.
        METADATA (1): Medium priority — cache metadata, peer info.
        DATA (2): Lowest priority — bulk KV cache transfers.
    """

    GOSSIP = 0
    METADATA = 1
    DATA = 2


# Iteration order for priority-aware receive
_STREAM_PRIORITY_ORDER = [
    StreamPriority.GOSSIP,
    StreamPriority.METADATA,
    StreamPriority.DATA,
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class QuicTransportError(Exception):
    """Base exception for QUIC transport errors."""


class ConnectionClosed(QuicTransportError):
    """Raised when attempting to use a closed connection."""


class ConnectionFailed(QuicTransportError):
    """Raised when a connection attempt fails."""


class Timeout(QuicTransportError):
    """Raised when a recv operation times out."""


# ===================================================================
# QUIC Protocol Handler  (aioquic integration)
# ===================================================================
if HAS_AIOQUIC:

    class _QuicProtocol(QuicConnectionProtocol):
        """Internal QUIC protocol handler with priority-based receive queuing.

        Each protocol instance corresponds to one QUIC connection to/from
        a single peer.  Incoming stream data is tagged with its priority
        and pushed into an ``asyncio.PriorityQueue`` so consumers always
        receive the highest-priority message first.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._recv_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
            self._connected_event = asyncio.Event()
            self._closed_event = asyncio.Event()
            self._session_ticket: SessionTicket | None = None

        # ---- aioquic event callbacks ----

        def quic_event_received(self, event: Any) -> None:
            if isinstance(event, HandshakeCompleted):
                self._connected_event.set()
            elif isinstance(event, StreamDataReceived):
                self._handle_stream_data(event)
            elif isinstance(event, ConnectionTerminated):
                self._signal_closed()
            elif isinstance(event, SessionTicketReceived):
                self._session_ticket = event.ticket

        def _handle_stream_data(self, event: StreamDataReceived) -> None:
            """Parse the priority header and enqueue payload."""
            data = event.data
            if len(data) < _HEADER_SIZE:
                logger.warning(
                    "Dropping stream data: header too short (%d bytes)", len(data)
                )
                return
            priority_byte, payload_len = struct.unpack(
                _MESSAGE_HEADER_FMT, data[:_HEADER_SIZE]
            )
            # Clamp priority to valid range
            try:
                priority = StreamPriority(priority_byte)
            except ValueError:
                priority = StreamPriority.DATA

            payload = data[_HEADER_SIZE : _HEADER_SIZE + payload_len]
            self._recv_queue.put_nowait((int(priority), payload))

        def _signal_closed(self) -> None:
            """Mark the connection as closed and unblock consumers."""
            self._closed_event.set()
            # Push a sentinel (lowest priority so pending messages drain first)
            self._recv_queue.put_nowait((_SENTINEL_PRIORITY, None))

        # ---- helpers ----

        @property
        def ticket(self) -> SessionTicket | None:
            """The session ticket received during the TLS handshake (0-RTT)."""
            return self._session_ticket

        async def wait_connected(self, timeout: float = _DEFAULT_TIMEOUT) -> None:
            """Block until the QUIC handshake completes or *timeout* expires."""
            await asyncio.wait_for(self._connected_event.wait(), timeout=timeout)

        @property
        def is_closed(self) -> bool:
            return self._closed_event.is_set()


# ===================================================================
# QuicConnection — single peer connection
# ===================================================================
if HAS_AIOQUIC:

    class QuicConnection:
        """A QUIC connection to a single peer with prioritized streams.

        Each connection manages one ``_QuicProtocol`` instance and wraps
        its send/receive primitives with priority tracking.  Streams are
        ephemeral — every ``send()`` call creates a new bidirectional QUIC
        stream — providing natural prioritisation at the receiver via the
        priority-tagged message queue.

        Attributes:
            peer_id: Human-readable identifier (``host:port``) of the remote peer.
        """

        def __init__(
            self,
            peer_id: str,
            protocol: _QuicProtocol,
        ) -> None:
            self._peer_id = peer_id
            self._protocol = protocol

        # ---- properties ----

        @property
        def peer_id(self) -> str:
            return self._peer_id

        @property
        def session_ticket(self) -> SessionTicket | None:
            """Cached TLS session ticket for 0-RTT reconnection."""
            return self._protocol.ticket

        # ---- public API ----

        async def send(self, priority: StreamPriority, data: bytes) -> None:
            """Send *data* on a new stream at the given *priority*.

            A small header (1 byte priority + 4 byte length) is prepended
            so the receiver can dispatch the payload to the correct queue
            without inspecting the content.

            Raises:
                ConnectionClosed: If the underlying QUIC connection is gone.
            """
            if self._protocol.is_closed:
                raise ConnectionClosed(
                    f"Cannot send to {self._peer_id}: connection closed"
                )

            stream_id = self._protocol._quic.get_next_available_stream_id()
            header = struct.pack(_MESSAGE_HEADER_FMT, int(priority), len(data))
            self._protocol._quic.send_stream_data(
                stream_id, header + data, end_stream=True
            )
            self._protocol.transmit()

        async def recv(self, timeout: float | None = None) -> tuple[StreamPriority, bytes]:
            """Receive the next available message, favouring higher priority.

            Blocks until at least one message is available or the connection
            closes.  Returns ``(priority, payload)``.

            Raises:
                ConnectionClosed: If the peer has disconnected.
                Timeout: If *timeout* seconds elapse without data.
            """
            try:
                prio_int, payload = await asyncio.wait_for(
                    self._protocol._recv_queue.get(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                raise Timeout("recv timed out")

            if payload is None:
                raise ConnectionClosed(
                    f"Connection to {self._peer_id} closed by peer"
                )

            return (StreamPriority(prio_int), payload)

        async def close(self) -> None:
            """Gracefully close the QUIC connection."""
            if not self._protocol.is_closed:
                self._protocol._quic.close()
                self._protocol.transmit()

else:
    # Stub replacement when aioquic is not installed
    class QuicConnection:  # type: ignore[no-redef]
        """Stub replacement when aioquic is not installed.

        Every operation raises ``QuicTransportError`` with a clear message
        directing the caller to install the optional dependency.
        """

        def __init__(self, peer_id: str, protocol: Any = None) -> None:
            self._peer_id = peer_id

        @property
        def peer_id(self) -> str:
            return self._peer_id

        @property
        def session_ticket(self) -> None:
            return None

        async def send(self, priority: StreamPriority, data: bytes) -> None:
            raise QuicTransportError(
                "aioquic is not installed.  "
                "Install with: pip install aioquic"
            )

        async def recv(self, timeout: float | None = None) -> tuple[StreamPriority, bytes]:
            raise QuicTransportError(
                "aioquic is not installed.  "
                "Install with: pip install aioquic"
            )

        async def close(self) -> None:
            pass


# ===================================================================
# QuicTransport — connection manager
# ===================================================================
if HAS_AIOQUIC:

    class QuicTransport:
        """QUIC-based P2P transport with prioritized multiplexed streams.

        Manages connections to/from peers using QUIC (when aioquic is
        available).  Supports 0-RTT reconnection via cached TLS session
        tickets.

        Usage::

            transport = QuicTransport(node_id="node-1")
            await transport.listen("localhost", 50053)

            # Outgoing connection
            conn = await transport.connect("other-host", 50053)
            await conn.send(StreamPriority.GOSSIP, b"hello")
            priority, data = await conn.recv()

            # Accept incoming
            conn = await transport.accept()

            # Cleanup
            await transport.close()
        """

        def __init__(
            self,
            node_id: str = "",
            cert_file: str | None = None,
            key_file: str | None = None,
            session_ticket_dir: str | None = None,
            ca_file: str | None = None,
        ):
            self._node_id = node_id
            self._cert_file = cert_file or os.environ.get("DISTLLM_QUIC_CERT")
            self._key_file = key_file or os.environ.get("DISTLLM_QUIC_KEY")
            self._session_ticket_dir = session_ticket_dir
            # CA bundle used to verify peer certificates (client side).  Without
            # it, P2P self-signed certs cannot be verified and the client falls
            # back to the (MITM-able) CERT_NONE mode — warned loudly below.
            self._ca_file = ca_file or os.environ.get("DISTLLM_QUIC_CA")
            if self._ca_file:
                logger.warning(
                    "QuicTransport: verifying peer certificates against CA "
                    f"{self._ca_file} (CERT_REQUIRED)"
                )
            else:
                logger.warning(
                    "QuicTransport: NO CA configured (set DISTLLM_QUIC_CA) — "
                    "outgoing QUIC connections will NOT verify the peer "
                    "certificate (CERT_NONE) and are vulnerable to MITM."
                )

            # State
            self._connections: dict[str, QuicConnection] = {}
            self._server: Any = None
            self._server_task: asyncio.Task | None = None
            self._accept_queue: asyncio.Queue[QuicConnection] = asyncio.Queue()
            self._session_tickets: dict[str, SessionTicket] = {}
            self._global_recv_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
            self._closed = False

        # ---- config helpers ----

        def _build_config(self, is_client: bool) -> QuicConfiguration:
            """Build a ``QuicConfiguration`` for client or server use."""
            config = QuicConfiguration(
                alpn_protocols=["distllm-quic/v1"],
                is_client=is_client,
                max_datagram_size=1200,
                idle_timeout=30.0,
            )
            if self._cert_file and self._key_file:
                config.load_cert_chain(self._cert_file, self._key_file)
            if is_client:
                if self._ca_file:
                    # Verify the peer's certificate against the configured CA
                    # bundle so an on-path attacker cannot impersonate a node.
                    config.load_verify_locations(self._ca_file)
                    config.verify_mode = ssl.CERT_REQUIRED
                else:
                    # P2P: self-signed certs are the norm; without a CA we
                    # cannot verify, so fall back to CERT_NONE (warned loudly at
                    # construction).  Configure DISTLLM_QUIC_CA to verify peers.
                    config.verify_mode = ssl.CERT_NONE
            return config

        def _on_new_protocol(self, protocol: _QuicProtocol, peer_key: str) -> None:
            """Wire a newly created protocol into the transport."""

            def _put_to_global(event: Any) -> None:
                if isinstance(event, StreamDataReceived):
                    data = event.data
                    if len(data) >= _HEADER_SIZE:
                        priority_byte, payload_len = struct.unpack(
                            _MESSAGE_HEADER_FMT, data[:_HEADER_SIZE]
                        )
                        payload = data[_HEADER_SIZE : _HEADER_SIZE + payload_len]
                        self._global_recv_queue.put_nowait(
                            (priority_byte, payload, peer_key)
                        )

            # Monkey-patch the protocol to also feed the global queue
            original_handler = protocol.quic_event_received

            def _patched_handler(event: Any) -> None:
                original_handler(event)
                _put_to_global(event)

            protocol.quic_event_received = _patched_handler  # type: ignore[method-assign]

        # ---- connect / listen ----

        async def connect(
            self, host: str, port: int
        ) -> QuicConnection:
            """Establish an outgoing QUIC connection to *host:port*.

            Returns:
                A ``QuicConnection`` representing the connection.

            Raises:
                ConnectionFailed: If the handshake fails.
            """
            peer_key = f"{host}:{port}"
            if peer_key in self._connections:
                return self._connections[peer_key]

            config = self._build_config(is_client=True)

            # 0-RTT: apply cached session ticket if available
            ticket = self._session_tickets.get(peer_key)
            if ticket is not None:
                config.session_ticket = ticket

            ready = asyncio.Event()
            connection_ref: list[QuicConnection | None] = [None]
            error_ref: list[Exception | None] = [None]

            async def _run_connection() -> None:
                try:
                    async with _quic_connect(
                        host,
                        port,
                        configuration=config,
                        create_protocol=lambda: _QuicProtocol(  # noqa: F821
                            _QuicConn(configuration=config)
                        ),
                    ) as protocol:
                        # Wait for handshake (or use 0-RTT immediately)
                        await protocol.wait_connected()

                        # Cache session ticket for future 0-RTT
                        if protocol.ticket is not None:
                            self._session_tickets[peer_key] = protocol.ticket

                        conn = QuicConnection(peer_key, protocol)
                        self._on_new_protocol(protocol, peer_key)
                        self._connections[peer_key] = conn
                        connection_ref[0] = conn
                        ready.set()

                        # Keep connection alive until closed signal
                        await protocol._closed_event.wait()
                except Exception as exc:
                    error_ref[0] = exc
                    ready.set()
                finally:
                    self._connections.pop(peer_key, None)

            asyncio.create_task(_run_connection())
            await ready.wait()

            if error_ref[0] is not None:
                raise ConnectionFailed(str(error_ref[0])) from error_ref[0]

            assert connection_ref[0] is not None
            return connection_ref[0]

        async def listen(self, host: str, port: int) -> None:
            """Start a QUIC server on *host:port*.

            This launches a background server.  Use ``accept()`` to
            retrieve incoming connections.

            Raises:
                ConnectionFailed: If the server fails to bind.
            """
            config = self._build_config(is_client=False)
            transport_ref = self  # capture for closure

            def _create_protocol() -> _QuicProtocol:
                """Factory called by ``_quic_serve`` per new QUIC connection.

                We monkey-patch the protocol's ``quic_event_received`` to
                detect when the handshake completes, then register the
                connection with the transport so that ``accept()`` can
                deliver it and ``recv_stream()`` can read from it.
                """
                proto = _QuicProtocol(_QuicConn(configuration=config))
                _original_handler = proto.quic_event_received

                def _connected_handler(event: Any) -> None:
                    _original_handler(event)
                    # Register the connection once the handshake finishes
                    if isinstance(event, HandshakeCompleted):
                        # Derive peer address from the QUIC connection
                        peer_addr = f"{proto._quic.host}:{proto._quic.port}"
                        if peer_addr not in transport_ref._connections:
                            conn = QuicConnection(peer_addr, proto)
                            transport_ref._on_new_protocol(proto, peer_addr)
                            transport_ref._connections[peer_addr] = conn
                            transport_ref._accept_queue.put_nowait(conn)

                proto.quic_event_received = _connected_handler  # type: ignore[method-assign]
                return proto

            try:
                self._server = await _quic_serve(
                    host,
                    port,
                    configuration=config,
                    create_protocol=_create_protocol,
                )
                logger.info(
                    "QUIC server listening on %s:%d (node=%s)",
                    host,
                    port,
                    self._node_id,
                )
            except Exception as exc:
                raise ConnectionFailed(
                    f"Failed to start QUIC server on {host}:{port}: {exc}"
                ) from exc

        async def accept(self) -> QuicConnection:
            """Wait for and return the next incoming QUIC connection."""
            return await self._accept_queue.get()

        # ---- send / receive ----

        async def send_stream(self, priority: StreamPriority, data: bytes) -> None:
            """Broadcast *data* to all connected peers at *priority*.

            Connections that have been closed are silently removed from the
            pool.
            """
            closed_peers: list[str] = []
            for peer_id, conn in list(self._connections.items()):
                try:
                    await conn.send(priority, data)
                except ConnectionClosed:
                    closed_peers.append(peer_id)
                except Exception as exc:
                    logger.warning(
                        "send_stream to %s failed: %s", peer_id, exc
                    )
            for pid in closed_peers:
                self._connections.pop(pid, None)

        async def recv_stream(
            self,
        ) -> tuple[StreamPriority, bytes, str]:
            """Receive the next message from any connected peer.

            Messages are returned in global priority order: all GOSSIP
            messages are delivered before METADATA, and METADATA before
            DATA, regardless of which peer sent them.

            Returns:
                ``(priority, payload, peer_id)``.
            """
            prio_int, payload, peer_id = await self._global_recv_queue.get()
            if payload is None:  # sentinel
                return await self.recv_stream()
            return (StreamPriority(prio_int), payload, peer_id)

        # ---- lifecycle ----

        async def close(self) -> None:
            """Close all connections and stop the server."""
            self._closed = True
            for conn in list(self._connections.values()):
                try:
                    await conn.close()
                except Exception:
                    pass
            self._connections.clear()
            if self._server is not None:
                self._server.close()
                self._server = None
            if self._server_task is not None:
                self._server_task.cancel()
                self._server_task = None

else:
    # Stub replacement when aioquic is not installed
    class QuicTransport:  # type: ignore[no-redef]
        """Stub replacement when aioquic is not installed.

        Every operation raises ``QuicTransportError`` with a clear message.
        """

        def __init__(
            self,
            node_id: str = "",
            cert_file: str | None = None,
            key_file: str | None = None,
            session_ticket_dir: str | None = None,
        ) -> None:
            self._node_id = node_id

        async def connect(
            self, host: str, port: int
        ) -> QuicConnection:
            raise ConnectionFailed(
                "aioquic is required for QUIC transport.  "
                "Install with: pip install aioquic"
            )

        async def listen(self, host: str, port: int) -> None:
            raise ConnectionFailed(
                "aioquic is required for QUIC transport.  "
                "Install with: pip install aioquic"
            )

        async def accept(self) -> QuicConnection:
            raise QuicTransportError(
                "aioquic is not installed.  "
                "Install with: pip install aioquic"
            )

        async def send_stream(self, priority: StreamPriority, data: bytes) -> None:
            raise QuicTransportError(
                "aioquic is not installed.  "
                "Install with: pip install aioquic"
            )

        async def recv_stream(
            self,
        ) -> tuple[StreamPriority, bytes, str]:
            raise QuicTransportError(
                "aioquic is not installed.  "
                "Install with: pip install aioquic"
            )

        async def close(self) -> None:
            pass


# ===================================================================
# Transport auto-detection
# ===================================================================


def get_optimal_transport() -> type[QuicTransport]:
    """Return the best available transport class.

    When ``aioquic`` is installed, returns the real :class:`QuicTransport`
    that uses QUIC (UDP, multiplexed, prioritised, 0-RTT).

    When ``aioquic`` is not available, returns a stub :class:`QuicTransport`
    that raises ``ConnectionFailed`` on any real operation — the caller
    should fall back to the existing HTTP-based ``GossipTransport`` from
    ``transport.py``.
    """
    return QuicTransport


def quic_available() -> bool:
    """Check whether the aioquic library is installed."""
    return HAS_AIOQUIC
