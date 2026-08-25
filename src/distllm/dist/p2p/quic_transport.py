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
# Wire format per stream: a message is split into ``frame_count`` chunks.
# Each frame is laid out as:
#   [1-byte priority][4-byte index][4-byte count][4-byte chunk length]
#   [chunk bytes]
# The explicit chunk length makes frames self-delimiting: aioquic delivers
# stream bytes in packet-sized ``StreamDataReceived`` events whose boundaries
# bear no relation to frame boundaries, so the receiver must parse frames
# greedily out of a per-stream byte buffer rather than assuming one event ==
# one frame.
_CHUNK_HEADER_FMT = "!BIII"
_CHUNK_HEADER_SIZE = struct.calcsize(_CHUNK_HEADER_FMT)
# Keep each frame small enough that it fits within a single QUIC packet
# (default max_datagram_size=1200 leaves ~1195 bytes of payload capacity).
_MAX_FRAME_PAYLOAD = 1024
_DEFAULT_TIMEOUT = 30.0
_SENTINEL_PRIORITY = 255  # Lowest possible priority for close sentinel


def _frame_chunks(priority: StreamPriority | int, data: bytes) -> list[bytes]:
    """Frame *data* for transmission at the given *priority*.

    Splits the payload into ``_MAX_FRAME_PAYLOAD``-byte chunks and returns
    fully-framed pieces (header + chunk).  An empty message produces exactly
    one zero-payload frame so it still round-trips.
    """
    chunks = [
        data[i : i + _MAX_FRAME_PAYLOAD]
        for i in range(0, len(data), _MAX_FRAME_PAYLOAD)
    ] or [b""]
    total = len(chunks)
    prio = int(priority)
    return [
        struct.pack(_CHUNK_HEADER_FMT, prio, idx, total, len(c)) + c
        for idx, c in enumerate(chunks)
    ]


class _StreamAssembler:
    """Byte-level per-stream reassembly for chunked messages.

    Feed every ``StreamDataReceived`` payload into :meth:`feed`; it returns
    the list of fully-received ``(priority, payload)`` messages (usually zero
    or one).  Frames are self-delimiting, so events may split or coalesce
    frames arbitrarily.  QUIC streams are ordered, so frames arrive strictly
    sequentially.
    """

    __slots__ = ("_buf", "_priority", "_count", "_chunks", "_next_index")

    def __init__(self) -> None:
        self._buf = bytearray()
        self._priority: StreamPriority | None = None
        self._count: int | None = None
        self._chunks: dict[int, bytes] = {}
        self._next_index = 0

    def feed(self, data: bytes) -> list[tuple[StreamPriority, bytes]]:
        self._buf.extend(data)
        messages: list[tuple[StreamPriority, bytes]] = []
        while True:
            done, msg = self._try_parse()
            if not done:
                break  # need more bytes
            if msg is not None:
                messages.append(msg)
        return messages

    def _try_parse(self) -> tuple[bool, tuple[StreamPriority, bytes] | None]:
        """Attempt to consume one frame from the buffer.

        Returns ``(progressed, message)``: ``progressed`` is ``True`` when a
        full frame was consumed (``message`` set only if that completed a
        message); ``False`` means more bytes are needed.
        """
        buf = self._buf
        if len(buf) < _CHUNK_HEADER_SIZE:
            return False, None
        priority_byte, frame_index, frame_count, chunk_len = struct.unpack_from(
            _CHUNK_HEADER_FMT, buf
        )
        frame_total = _CHUNK_HEADER_SIZE + chunk_len
        if len(buf) < frame_total:
            return False, None  # partial frame — wait for more bytes

        chunk = bytes(buf[_CHUNK_HEADER_SIZE:frame_total])
        del buf[:frame_total]

        try:
            priority = StreamPriority(priority_byte)
        except ValueError:
            priority = StreamPriority.DATA

        if (
            frame_index < self._next_index
            or (frame_count <= 0 or frame_index >= frame_count)
            or (self._count is not None and frame_count != self._count)
        ):
            # Out-of-sequence or malformed frame on this stream: the stream
            # is unrecoverable (ordered delivery guarantees sequence), so
            # reset the assembler — buffer included — and discard whatever
            # message was being built.
            logger.warning(
                "Discarding malformed frame (index=%d count=%d expected>=%d)",
                frame_index,
                frame_count,
                self._next_index,
            )
            self._buf = bytearray()
            self._reset_state()
            return True, None

        if self._count is None:
            self._count = frame_count
            self._priority = priority

        self._chunks[frame_index] = chunk
        self._next_index = frame_index + 1

        if frame_index == frame_count - 1:
            payload = b"".join(
                self._chunks[i] for i in range(frame_count)
            )
            result_priority = self._priority or priority
            # Reset per-message state but KEEP the buffer: any residual
            # bytes were coalesced into the same event and belong to the
            # next message on this stream.
            self._reset_state()
            return True, (result_priority, payload)
        return True, None

    def _reset_state(self) -> None:
        self._priority = None
        self._count = None
        self._chunks = {}
        self._next_index = 0


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


def _generate_self_signed_cert() -> tuple[Any, Any]:
    """Generate an ephemeral self-signed certificate + private key.

    Used when a QUIC *server* is started without explicit TLS material
    (P2P convention: peers authenticate via ``DISTLLM_QUIC_CA`` or accept
    unverified connections).  Returns ``(certificate, private_key)`` in
    the cryptography-object form aioquic's ``QuicConfiguration`` expects.
    """
    try:
        import datetime

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec
    except ImportError as exc:  # pragma: no cover - defensive
        raise QuicTransportError(
            "QUIC server requires a certificate.  Set DISTLLM_QUIC_CERT/"
            "DISTLLM_QUIC_KEY or install 'cryptography' to auto-generate "
            "an ephemeral self-signed one."
        ) from exc

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name(
        [x509.NameAttribute(x509.oid.NameOID.COMMON_NAME, "distllm-p2p-node")]
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=30))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    return cert, key


# ===================================================================
# QUIC Protocol Handler  (aioquic integration)
# ===================================================================
if HAS_AIOQUIC:

    class _QuicProtocol(QuicConnectionProtocol):
        """Internal QUIC protocol handler with priority-based receive queuing.

        Each protocol instance corresponds to one QUIC connection to/from
        a single peer.  Incoming stream data is reassembled per stream and
        pushed into an ``asyncio.PriorityQueue`` so consumers always receive
        the highest-priority message first.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._recv_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
            self._connected_event = asyncio.Event()
            self._closed_event = asyncio.Event()
            self._session_ticket: SessionTicket | None = None
            # Per-stream byte-level reassemblers for chunked messages
            self._stream_assemblers: dict[Any, _StreamAssembler] = {}
            # Monotonic sequence for stable FIFO ordering within a priority
            # (PriorityQueue compares tuples; without a sequence tiebreaker
            # equal-priority entries would be ordered by payload bytes).
            self._recv_seq: int = 0
            # Optional callback fired for every completed message (used by
            # QuicTransport to feed the transport-wide receive queue).
            self.completion_callback: Any = None

        def _next_seq(self) -> int:
            seq = self._recv_seq
            self._recv_seq += 1
            return seq

        # ---- aioquic event callbacks ----

        def quic_event_received(self, event: Any) -> None:
            if isinstance(event, HandshakeCompleted):
                self._connected_event.set()
            elif isinstance(event, StreamDataReceived):
                self._handle_stream_data(event)
            elif isinstance(event, ConnectionTerminated):
                self._signal_closed()

        def connection_ticket_received(self, ticket: SessionTicket) -> None:
            """TLS session-ticket callback (wired via ``connect()``)."""
            self._session_ticket = ticket

        def datagram_received(self, data: Any, addr: Any = None) -> None:
            """Capture the remote peer address before processing the datagram.

            aioquic does not expose the peer address on ``QuicConnection``
            objects; the server only sees it here.
            """
            if addr is not None:
                try:
                    host, port = addr[0], addr[1]
                except (TypeError, IndexError):
                    host, port = str(addr), 0
                self._remote_addr = f"{host}:{port}"
            super().datagram_received(data, addr)

        @property
        def remote_addr(self) -> str | None:
            """The remote ``host:port`` learned from received datagrams."""
            return getattr(self, "_remote_addr", None)

        def _handle_stream_data(self, event: StreamDataReceived) -> None:
            """Reassemble chunked stream data and enqueue complete messages.

            The wire format is a sequence of self-delimiting frames per
            stream::

                [1B priority][4B index][4B count][4B chunk length][chunk]

            aioquic delivers stream bytes in packet-sized
            ``StreamDataReceived`` events whose boundaries do not align with
            frame boundaries, so a byte-level per-stream assembler parses
            frames greedily and emits each message once its final frame has
            arrived.
            """
            assembler = self._stream_assemblers.get(event.stream_id)
            if assembler is None:
                assembler = _StreamAssembler()
                self._stream_assemblers[event.stream_id] = assembler

            for priority, payload in assembler.feed(event.data):
                self._recv_queue.put_nowait(
                    (int(priority), self._next_seq(), payload)
                )
                if self.completion_callback is not None:
                    try:
                        self.completion_callback(priority, payload)
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.warning("completion callback failed: %s", exc)

        def _signal_closed(self) -> None:
            """Mark the connection as closed and unblock consumers."""
            self._closed_event.set()
            # Push a sentinel (lowest priority so pending messages drain first)
            self._recv_queue.put_nowait(
                (_SENTINEL_PRIORITY, self._next_seq(), None)
            )

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

            The payload is split into self-delimiting frames (priority,
            index/count and chunk length headers) so arbitrarily large
            messages survive aioquic's packet-sized stream delivery; the
            receiver reassembles them transparently.

            Raises:
                ConnectionClosed: If the underlying QUIC connection is gone.
            """
            if self._protocol.is_closed:
                raise ConnectionClosed(
                    f"Cannot send to {self._peer_id}: connection closed"
                )

            stream_id = self._protocol._quic.get_next_available_stream_id()
            frames = _frame_chunks(priority, data)
            for i, frame in enumerate(frames):
                self._protocol._quic.send_stream_data(
                    stream_id, frame, end_stream=(i == len(frames) - 1)
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
                prio_int, _, payload = await asyncio.wait_for(
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
            # Global receive queue entries are (priority, seq, payload, peer);
            # the sequence tiebreaker preserves FIFO order within a priority.
            self._global_recv_seq: int = 0
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
            elif not is_client:
                # QUIC requires a server certificate.  For P2P deployments
                # without explicit TLS material, generate an ephemeral
                # self-signed cert (peers should set DISTLLM_QUIC_CA to
                # authenticate; see the CERT_NONE warning above).
                config.certificate, config.private_key = _generate_self_signed_cert()
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
            else:
                # Servers present their own certificate but never verify
                # clients at the TLS layer (client auth, when needed, happens
                # at the application layer).
                config.verify_mode = ssl.CERT_NONE
            return config

        def _on_new_protocol(self, protocol: _QuicProtocol, peer_key: str) -> None:
            """Wire a newly created protocol into the transport.

            Completed messages are routed into the transport-wide priority
            queue via the protocol's ``completion_callback``; the per-message
            priority is carried in each frame's header, so the global
            ``recv_stream()`` preserves the same prioritisation semantics as
            per-connection ``recv()``.
            """

            def _on_complete(priority: StreamPriority, payload: bytes) -> None:
                seq = self._global_recv_seq
                self._global_recv_seq += 1
                self._global_recv_queue.put_nowait(
                    (int(priority), seq, payload, peer_key)
                )

            protocol.completion_callback = _on_complete

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
            protocol_ref: list[_QuicProtocol | None] = [None]

            def _ticket_handler(ticket: SessionTicket) -> None:
                """TLS callback: cache the ticket for 0-RTT reconnection."""
                proto = protocol_ref[0]
                if proto is not None:
                    proto.connection_ticket_received(ticket)

            async def _run_connection() -> None:
                try:
                    async with _quic_connect(
                        host,
                        port,
                        configuration=config,
                        session_ticket_handler=_ticket_handler,
                        create_protocol=lambda *args, **kwargs: _QuicProtocol(  # noqa: F821
                            *args, **kwargs
                        ),
                    ) as protocol:
                        protocol_ref[0] = protocol
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

            def _create_protocol(*args: Any, **kwargs: Any) -> _QuicProtocol:
                """Factory called by ``_quic_serve`` per new QUIC connection.

                aioquic supplies the ``QuicConnection`` and ``stream_handler``
                as arguments.  We monkey-patch the protocol's
                ``quic_event_received`` to detect when the handshake
                completes, then register the connection with the transport so
                that ``accept()`` can deliver it and ``recv_stream()`` can
                read from it.
                """
                proto = _QuicProtocol(*args, **kwargs)
                _original_handler = proto.quic_event_received

                def _connected_handler(event: Any) -> None:
                    _original_handler(event)
                    # Register the connection once the handshake finishes
                    if isinstance(event, HandshakeCompleted):
                        # Derive peer address from captured datagrams
                        peer_addr = proto.remote_addr or "unknown"
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
            prio_int, _, payload, peer_id = await self._global_recv_queue.get()
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
