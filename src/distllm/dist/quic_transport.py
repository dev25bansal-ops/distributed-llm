"""QUIC/HTTP3 transport for wide-area tensor and forward-pass transfer.

Uses ``aioquic`` as an alternative to gRPC for :class:`TensorTransport`.
Provides QUIC-based client and server for forward pass requests with
built-in framing (length-prefixed binary messages).

QUIC advantages over TCP/gRPC:
  - 0-RTT connection establishment
  - No head-of-line blocking (independent streams)
  - Better packet-loss behavior (FEC, faster recovery)
"""

from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger

LENGTH_PREFIX_FORMAT = "!I"
LENGTH_PREFIX_SIZE = struct.calcsize(LENGTH_PREFIX_FORMAT)

PROTOCOL_ALPN = "distllm-quic-v1"


@dataclass
class QuicConfig:
    """Configuration for QUIC transport connections."""
    host: str = "0.0.0.0"
    port: int = 4433
    alpn_protocols: list[str] = field(default_factory=lambda: [PROTOCOL_ALPN, "distllm-v1"])
    max_stream_data: int = 104_857_600
    idle_timeout: float = 120.0
    max_packet_size: int = 65_527


def _frame_message(data: bytes) -> bytes:
    """Wrap *data* with a 4-byte big-endian length prefix."""
    return struct.pack(LENGTH_PREFIX_FORMAT, len(data)) + data


def _unframe_message(buffer: bytearray) -> tuple[bytes | None, bytearray]:
    """Extract one framed message from *buffer*.

    Returns ``(message, remaining_buffer)`` or ``(None, buffer)`` if
    a complete message is not yet available.
    """
    if len(buffer) < LENGTH_PREFIX_SIZE:
        return None, buffer
    length = struct.unpack(LENGTH_PREFIX_FORMAT, bytes(buffer[:LENGTH_PREFIX_SIZE]))[0]
    total_size = LENGTH_PREFIX_SIZE + length
    if len(buffer) < total_size:
        return None, buffer
    message = bytes(buffer[LENGTH_PREFIX_SIZE:total_size])
    buffer = buffer[total_size:]
    return message, buffer


def is_quic_available() -> bool:
    """Check whether ``aioquic`` is installed."""
    try:
        import aioquic  # noqa: F401
        return True
    except ImportError:
        return False


class QuicStreamHandler:
    """Server-side handler that processes incoming QUIC streams.

    Dispatches framed protobuf messages to a *forward_fn* and sends
    the response on the same stream.
    """

    def __init__(self, forward_fn: Callable[[bytes], bytes] | None = None):
        self._forward_fn = forward_fn
        self._buffers: dict[int, bytearray] = {}

    @property
    def forward_fn(self) -> Callable[[bytes], bytes] | None:
        return self._forward_fn

    @forward_fn.setter
    def forward_fn(self, fn: Callable[[bytes], bytes]) -> None:
        self._forward_fn = fn

    def quic_stream_received(
        self, stream_id: int, data: bytes,
        end_stream: bool, quic_connection: Any,
    ) -> None:
        """Called by the QUIC protocol when stream data arrives."""
        if stream_id not in self._buffers:
            self._buffers[stream_id] = bytearray()
        self._buffers[stream_id].extend(data)

        message, remaining = _unframe_message(self._buffers[stream_id])
        self._buffers[stream_id] = remaining

        if message is not None and self._forward_fn is not None:
            try:
                response_data = self._forward_fn(message)
                quic_connection.send(
                    stream_id, _frame_message(response_data), end_stream=True,
                )
            except Exception:
                logger.exception(f"QUIC stream {stream_id} handler error")
                quic_connection.send(stream_id, b"", end_stream=True)

    def cleanup(self, stream_id: int) -> None:
        self._buffers.pop(stream_id, None)


class QuicTransportClient:
    """Async QUIC client for forward-pass requests.

    Manages a single QUIC connection.  Each :meth:`forward_pass` call
    opens a new bidirectional stream.
    """

    def __init__(self, config: QuicConfig | None = None):
        self._config = config or QuicConfig()
        self._connection: Any = None
        self._lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        return self._connection is not None

    async def connect(
        self, host: str, port: int, timeout: float = 10.0,
    ) -> None:
        """Establish a QUIC connection to *host*:*port*."""
        from aioquic.asyncio.client import connect as _async_connect
        from aioquic.quic.configuration import QuicConfiguration

        quic_config = QuicConfiguration(
            alpn_protocols=self._config.alpn_protocols,
            is_client=True,
            max_data=self._config.max_stream_data,
            max_stream_data=self._config.max_stream_data,
            idle_timeout=self._config.idle_timeout,
        )
        self._connection = await _async_connect(
            host=host, port=port, configuration=quic_config,
            wait_connected=timeout,
        )
        logger.info("QUIC client connected to {}:{}", host, port)

    async def forward_pass(self, request_data: bytes, timeout: float = 120.0) -> bytes:
        """Send a forward-pass request and return the response.

        Args:
            request_data: Serialized protobuf request.
            timeout: Per-request timeout.

        Returns:
            Serialized protobuf response.
        """
        if self._connection is None:
            raise RuntimeError("QUIC client not connected; call connect() first")

        stream_id = self._connection.get_next_available_stream_id()
        framed = _frame_message(request_data)

        async with self._lock:
            self._connection.send(stream_id, framed, end_stream=True)
            raw = await asyncio.wait_for(
                self._connection.receive(stream_id), timeout=timeout,
            )

        msg, _ = _unframe_message(bytearray(raw))
        if msg is None:
            raise RuntimeError("Empty or incomplete QUIC response")
        return msg

    async def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None


class QuicTransportServer:
    """Async QUIC server that processes forward-pass requests.

    Listens on a *host*:*port* and dispatches incoming streams to a
    handler function.
    """

    def __init__(self, config: QuicConfig | None = None):
        self._config = config or QuicConfig()
        self._server: Any = None
        self._handler = QuicStreamHandler()

    @property
    def handler(self) -> QuicStreamHandler:
        return self._handler

    async def start(
        self, host: str, port: int,
        forward_fn: Callable[[bytes], bytes] | None = None,
    ) -> None:
        """Start the QUIC server.

        Args:
            host: Local address to bind.
            port: Local UDP port.
            forward_fn: Optional immediate handler.  Can also be set
                later via ``handler.forward_fn``.
        """
        from aioquic.asyncio.server import serve as _async_serve
        from aioquic.quic.configuration import QuicConfiguration

        if forward_fn is not None:
            self._handler.forward_fn = forward_fn

        quic_config = QuicConfiguration(
            alpn_protocols=self._config.alpn_protocols,
            is_client=False,
            max_data=self._config.max_stream_data,
            max_stream_data=self._config.max_stream_data,
            idle_timeout=self._config.idle_timeout,
        )

        self._server = await _async_serve(
            host=host, port=port,
            configuration=quic_config,
            create_protocol=lambda: self._handler,
        )
        logger.info("QUIC server listening on {}:{}", host, port)

    async def shutdown(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None

    @property
    def is_serving(self) -> bool:
        return self._server is not None


class TcpFallbackTransport:
    """TCP fallback when QUIC (aioquic) is not available.

    Uses standard asyncio TCP with the same length-prefixed binary
    framing as the QUIC transport. Provides a drop-in replacement
    so WAN transfers work even without aioquic installed.

    Usage::

        transport = TcpFallbackTransport()
        await transport.connect("remote-host", 4433)
        await transport.send(b"hello")
        response = await transport.receive()
    """

    def __init__(self, timeout_s: float = 30.0):
        self._timeout_s = timeout_s
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected = False

    async def connect(self, host: str, port: int) -> None:
        """Connect to a remote TCP server."""
        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=self._timeout_s,
        )
        self._connected = True
        logger.debug(f"TCP fallback connected to {host}:{port}")

    async def send(self, data: bytes) -> None:
        """Send a framed message over TCP."""
        if not self._connected or self._writer is None:
            raise RuntimeError("Not connected")
        framed = _frame_message(data)
        self._writer.write(framed)
        await self._writer.drain()

    async def receive(self) -> bytes | None:
        """Receive a framed message from TCP."""
        if not self._connected or self._reader is None:
            raise RuntimeError("Not connected")

        # Read length prefix
        header = await asyncio.wait_for(
            self._reader.readexactly(LENGTH_PREFIX_SIZE),
            timeout=self._timeout_s,
        )
        if not header:
            return None

        length = struct.unpack(LENGTH_PREFIX_FORMAT, header)[0]
        data = await asyncio.wait_for(
            self._reader.readexactly(length),
            timeout=self._timeout_s,
        )
        return data

    async def close(self) -> None:
        """Close the TCP connection."""
        if self._writer is not None:
            self._writer.close()
            await self._writer.wait_closed()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected


def create_transport(
    prefer_quic: bool = True,
    timeout_s: float = 30.0,
) -> QuicClientTransport | TcpFallbackTransport:
    """Create the best available transport.

    Returns QUIC transport if aioquic is installed, otherwise
    falls back to TCP with the same framing protocol.

    Args:
        prefer_quic: If True, prefer QUIC over TCP when available.
        timeout_s: Connection timeout in seconds.

    Returns:
        A transport instance (QUIC or TCP).
    """
    if prefer_quic and is_quic_available():
        return QuicClientTransport(config=QuicConfig(idle_timeout=timeout_s))

    if not is_quic_available():
        logger.info("aioquic not installed, using TCP fallback transport")

    return TcpFallbackTransport(timeout_s=timeout_s)
