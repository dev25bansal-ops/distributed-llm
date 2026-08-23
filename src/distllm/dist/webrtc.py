"""WebRTC Data Channel transport for NAT traversal.

Replaces fragile hand-rolled STUN/TURN in nat.py with proven ICE
(Interactive Connectivity Establishment) via aiortc. Provides:

- ICE-based NAT traversal with configurable STUN/TURN servers
- DTLS 1.2 encryption over data channels
- SCTP reliable ordered delivery with congestion control
- Binary framing protocol for tensor transport
- HTTP-based signaling for SDP offer/answer and ICE candidate exchange

Usage:
    transport = WebRTCTransport(
        stun_servers=["stun:stun.l.google.com:19302"],
        role="offerer",
    )
    sdp = await transport.create_offer()
    answer = await signaling_exchange(sdp)  # via HTTP/gRPC
    await transport.accept_answer(answer)
    await transport.wait_connected()
    await transport.send_tensor(hidden_states_bytes)
"""


from __future__ import annotations
import asyncio
import io
import json
import struct
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Awaitable, Callable

from loguru import logger

try:
    from distllm.security.e2e import E2EEncryption, decrypt_tensor_payload, encrypt_tensor_payload
    HAS_E2E = True
except ImportError:
    HAS_E2E = False

    class E2EEncryption:  # type: ignore[no-redef]
        pass

    def decrypt_tensor_payload(data, e2e=None):
        return data

    def encrypt_tensor_payload(data, e2e=None):
        return data

HAS_WEBRTC = False
try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
    from aiortc.contrib.media import MediaPlayer, MediaRelay

    HAS_WEBRTC = True
except ImportError:
    RTCPeerConnection = None
    RTCSessionDescription = None
    RTCIceCandidate = None


class MsgType(IntEnum):
    FORWARD_PASS = 0
    FORWARD_PASS_RESPONSE = 1
    HEARTBEAT = 2
    TENSOR_TRANSFER = 3
    KV_CACHE = 4
    SHUTDOWN = 255


MSG_HEADER_FMT = "!BQH"
MSG_HEADER_SIZE = struct.calcsize(MSG_HEADER_FMT)


@dataclass
class WebRTCConfig:
    stun_servers: list[str] = field(
        default_factory=lambda: ["stun:stun.l.google.com:19302"]
    )
    turn_servers: list[dict[str, Any]] = field(default_factory=list)
    ice_servers: list[dict[str, Any]] = field(default_factory=list)
    max_message_size: int = 64 * 1024  # 64 KB default SCTP safe
    data_channel_label: str = "distllm-tensor"
    sctp_buffer_size: int = 16 * 1024 * 1024  # 16 MB

    def to_rtc_config(self) -> dict[str, Any]:
        servers = list(self.ice_servers)
        for stun_url in self.stun_servers:
            servers.append({"urls": stun_url})
        for turn_cfg in self.turn_servers:
            servers.append(turn_cfg)
        return {"iceServers": servers}


class WebRTCError(Exception):
    pass


class TensorFraming:
    """Binary message framing for tensor transport over WebRTC data channels.


    Frame format:
      [msg_type:1B][msg_id:8B][payload_length:2B][payload:...]
    """


    @staticmethod
    def encode(msg_type: MsgType, msg_id: int, payload: bytes) -> bytes:
        header = struct.pack(MSG_HEADER_FMT, msg_type, msg_id, len(payload))
        return header + payload

    @staticmethod
    def decode(frame: bytes) -> tuple[MsgType, int, bytes]:
        msg_type, msg_id, payload_len = struct.unpack_from(
            MSG_HEADER_FMT, frame, 0
        )
        payload = frame[MSG_HEADER_SIZE : MSG_HEADER_SIZE + payload_len]
        return MsgType(msg_type), msg_id, payload

    @staticmethod
    def serialize_tensor(data: Any) -> bytes:
        import torch

        buffer = io.BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def deserialize_tensor(data: bytes) -> Any:
        import torch

        buffer = io.BytesIO(data)
        return torch.load(buffer, weights_only=True)

    @staticmethod
    def make_heartbeat() -> bytes:
        return TensorFraming.encode(MsgType.HEARTBEAT, 0, b"")


class WebRTCSignaling:
    """HTTP-based signaling for SDP offer/answer and ICE candidate exchange.


    Each peer runs a small HTTP server to handle incoming signaling
    requests. The signaling protocol uses two endpoints:

      POST /webrtc/offer  — exchange SDP offer/answer
      POST /webrtc/ice    — exchange ICE candidates

    For use with an existing HTTP server (e.g., FastAPI, aiohttp):
      handler = WebRTCSignaling(node_id)
      app.add_route("/webrtc/offer", handler.handle_offer, methods=["POST"])
      app.add_route("/webrtc/ice", handler.handle_ice, methods=["POST"])
    """


    def __init__(self, node_id: str):
        self._node_id = node_id
        self._pending_offers: dict[str, RTCSessionDescription] = {}
        self._pending_ice: dict[str, list[RTCIceCandidate]] = {}
        self._on_offer: Callable[[RTCSessionDescription], Awaitable[RTCSessionDescription]] | None = None
        self._on_ice: Callable[[str, RTCIceCandidate], Awaitable[None]] | None = None

    def set_offer_handler(
        self,
        handler: Callable[[RTCSessionDescription], Awaitable[RTCSessionDescription]],
    ) -> None:
        self._on_offer = handler

    def set_ice_handler(self, handler: Callable[[str, RTCIceCandidate], Awaitable[None]]) -> None:
        self._on_ice = handler

    async def handle_offer(self, body: bytes) -> bytes:
        data = json.loads(body)
        sdp = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
        if self._on_offer:
            answer = await self._on_offer(sdp)
            return json.dumps({"sdp": answer.sdp, "type": answer.type}).encode()
        self._pending_offers[self._node_id] = sdp
        return json.dumps({"status": "accepted"}).encode()

    async def handle_ice(self, body: bytes) -> bytes:
        data = json.loads(body)
        candidate = self._parse_ice_candidate(data)
        if self._on_ice:
            await self._on_ice(data.get("ufrag", ""), candidate)
        else:
            self._pending_ice.setdefault(self._node_id, []).append(candidate)
        return json.dumps({"status": "ok"}).encode()

    @staticmethod
    def _parse_ice_candidate(data: dict) -> RTCIceCandidate:
        candidate_str = data.get("candidate", "")
        parts = candidate_str.split()
        foundation = parts[0] if len(parts) > 0 else "1"
        component = int(parts[1]) if len(parts) > 1 else 1
        transport = parts[2] if len(parts) > 2 else "UDP"
        priority = int(parts[3]) if len(parts) > 3 else 0
        ip = parts[4] if len(parts) > 4 else "0.0.0.0"
        port = int(parts[5]) if len(parts) > 5 else 0
        typ = parts[7] if len(parts) > 7 else "host"
        return RTCIceCandidate(
            component=component,
            foundation=foundation,
            ip=ip,
            port=port,
            priority=priority,
            protocol=transport,
            type=typ,
        )

    @staticmethod
    def format_ice_candidate(candidate: RTCIceCandidate) -> dict:
        return {
            "foundation": candidate.foundation,
            "component": candidate.component,
            "transport": candidate.protocol,
            "priority": candidate.priority,
            "ip": candidate.ip,
            "port": candidate.port,
            "type": candidate.type,
            "ufrag": "",
        }

    @staticmethod
    async def send_offer(
        url: str, offer: RTCSessionDescription
    ) -> RTCSessionDescription | None:
        import httpx

        payload = json.dumps({"sdp": offer.sdp, "type": offer.type}).encode()
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, content=payload)
                if resp.status_code == 200:
                    data = json.loads(resp.text)
                    if "sdp" in data and "type" in data:
                        return RTCSessionDescription(sdp=data["sdp"], type=data["type"])
            return None
        except Exception as e:
            logger.warning(f"WebRTC signaling offer failed: {e}")
            return None

    @staticmethod
    async def send_ice_candidate(url: str, ufrag: str, candidate: dict) -> bool:
        import httpx

        payload = json.dumps({"ufrag": ufrag, **candidate}).encode()
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, content=payload)
                return resp.status_code == 200
        except Exception:
            return False


class WebRTCTransport:
    """Manages an RTCPeerConnection with data channels for tensor transport.


    Uses ICE (Interactive Connectivity Establishment) for NAT traversal
    with configurable STUN/TURN servers. Provides a high-level async API
    for sending/receiving tensors over WebRTC data channels.

    Two roles:
      - offerer: creates the initial offer (typically the connecting peer)
      - answerer: accepts the offer and creates an answer (the listening peer)

    The full connection flow:
      offerer.create_offer() → signaling → answerer.accept_offer()
      offerer.accept_answer() → both.wait_connected() → send/recv tensors
    """


    def __init__(
        self,
        config: WebRTCConfig | None = None,
        role: str = "offerer",
        node_id: str | None = None,
        e2e_encryption: Any = None,
    ):
        if not HAS_WEBRTC:
            raise WebRTCError(
                "aiortc is required for WebRTC transport. "
                "Install with: pip install distributed-llm[webrtc]"
            )
        self._config = config or WebRTCConfig()
        self._role = role
        self._node_id = node_id or str(uuid.uuid4())[:8]
        self._e2e = e2e_encryption
        self._pc: RTCPeerConnection | None = None
        self._data_channel: Any = None
        self._connected = asyncio.Event()
        self._closed = False
        self._recv_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=128)
        self._msg_counter: int = 0

    async def create_offer(self) -> RTCSessionDescription:
        """Create an SDP offer as the offerer peer.


        Returns:
            RTCSessionDescription to be sent to the answerer via signaling.
        """

        self._pc = RTCPeerConnection(**self._config.to_rtc_config())
        self._setup_listeners()
        self._data_channel = self._pc.createDataChannel(
            self._config.data_channel_label,
        )
        self._setup_data_channel()
        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)
        return offer

    async def accept_offer(
        self, offer: RTCSessionDescription
    ) -> RTCSessionDescription:
        """Accept an incoming SDP offer and create an answer.


        Args:
            offer: The SDP offer received from the offerer.

        Returns:
            RTCSessionDescription answer to send back to the offerer.
        """

        self._pc = RTCPeerConnection(**self._config.to_rtc_config())
        self._setup_listeners()

        @self._pc.on("datachannel")
        def on_datachannel(channel):
            self._data_channel = channel
            self._setup_data_channel()

        await self._pc.setRemoteDescription(offer)
        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)
        return answer

    async def accept_answer(self, answer: RTCSessionDescription) -> None:
        """Accept the answer from the answerer (offerer side).


        Args:
            answer: The SDP answer received from the answerer.
        """

        if self._pc is None:
            raise WebRTCError("No peer connection; call create_offer() first")
        await self._pc.setRemoteDescription(answer)

    async def add_ice_candidate(self, candidate: RTCIceCandidate) -> None:
        """Add a remote ICE candidate for connection establishment.


        Args:
            candidate: ICE candidate from the remote peer.
        """

        if self._pc is None:
            raise WebRTCError("No peer connection")
        await self._pc.addIceCandidate(candidate)

    async def wait_connected(self, timeout: float = 30.0) -> bool:
        """Wait for the data channel to open.


        Args:
            timeout: Maximum seconds to wait.

        Returns:
            True if connected, False if timed out.
        """

        try:
            await asyncio.wait_for(self._connected.wait(), timeout=timeout)
            return self._connected.is_set()
        except asyncio.TimeoutError:
            logger.warning(f"WebRTC connection timed out after {timeout}s")
            return False

    async def send_tensor(self, data: Any, msg_type: MsgType = MsgType.FORWARD_PASS) -> int:
        """Serialize and send a tensor over the data channel.


        Args:
            data: Torch tensor or dict of tensors to send.
            msg_type: Message type identifier.

        Returns:
            Message ID for correlating responses.

        Raises:
            WebRTCError: If not connected or data channel not open.
        """

        if self._data_channel is None or self._data_channel.readyState != "open":
            raise WebRTCError("Data channel not open")
        payload = TensorFraming.serialize_tensor(data)
        if self._e2e is not None:
            payload = encrypt_tensor_payload(payload, self._e2e)
        msg_id = self._next_msg_id()
        frame = TensorFraming.encode(msg_type, msg_id, payload)
        self._data_channel.send(frame)
        return msg_id

    async def recv_tensor(
        self, timeout: float | None = None
    ) -> tuple[MsgType, int, Any] | None:
        """Receive a deserialized tensor from the data channel.


        Args:
            timeout: Receive timeout in seconds (None = blocking).

        Returns:
            Tuple of (msg_type, msg_id, tensor_data), or None on timeout.
        """

        try:
            frame = await asyncio.wait_for(self._recv_queue.get(), timeout=timeout)
            msg_type, msg_id, payload = TensorFraming.decode(frame)
            if self._e2e is not None:
                payload = decrypt_tensor_payload(payload, self._e2e)
            data = TensorFraming.deserialize_tensor(payload)
            return msg_type, msg_id, data
        except asyncio.TimeoutError:
            return None

    async def send_heartbeat(self) -> bool:
        """Send a heartbeat message over the data channel.


        Returns:
            True if sent successfully.
        """

        if self._data_channel is None or self._data_channel.readyState != "open":
            return False
        try:
            self._data_channel.send(TensorFraming.make_heartbeat())
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close the peer connection and data channel."""

        self._closed = True
        self._connected.clear()
        if self._data_channel is not None:
            try:
                self._data_channel.close()
            except Exception:
                pass
        if self._pc is not None:
            try:
                await self._pc.close()
            except Exception:
                pass

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set() and not self._closed

    @property
    def connection_state(self) -> str:
        if self._pc is None:
            return "new"
        return self._pc.connectionState

    def _next_msg_id(self) -> int:
        self._msg_counter += 1
        return self._msg_counter

    def _setup_listeners(self) -> None:
        if self._pc is None:
            return

        @self._pc.on("connectionstatechange")
        def on_connection_state_change():
            state = self._pc.connectionState
            logger.debug(f"WebRTC connection state: {state}")
            if state == "connected":
                self._connected.set()
            elif state in ("failed", "disconnected", "closed"):
                self._connected.clear()

        @self._pc.on("iceconnectionstatechange")
        def on_ice_state_change():
            logger.debug(f"ICE state: {self._pc.iceConnectionState}")

        @self._pc.on("icecandidate")
        async def on_ice_candidate(candidate):
            if candidate is None:
                return
            handler = getattr(self, "_on_ice_candidate", None)
            if handler:
                await handler(candidate)

    def _setup_data_channel(self) -> None:
        if self._data_channel is None:
            return

        @self._data_channel.on("open")
        def on_open():
            logger.info("WebRTC data channel opened")
            self._connected.set()

        @self._data_channel.on("close")
        def on_close():
            logger.info("WebRTC data channel closed")
            self._connected.clear()

        @self._data_channel.on("message")
        def on_message(message):
            if isinstance(message, (bytes, bytearray)):
                self._recv_queue.put_nowait(bytes(message))

    def set_ice_candidate_handler(
        self, handler: Callable[[RTCIceCandidate], Awaitable[None]]
    ) -> None:
        self._on_ice_candidate = handler

    def ice_candidates(self) -> list[dict]:
        """Get gathered ICE candidates for signaling to the remote peer.


        Returns:
            List of candidate dictionaries suitable for signaling.
        """

        if self._pc is None:
            return []
        candidates = []
        for candidate in getattr(self._pc, "_ice_gatherer", None) or []:
            candidates.append(WebRTCSignaling.format_ice_candidate(candidate))
        return candidates


async def connect_with_webrtc(
    peer_host: str,
    peer_port: int,
    local_port: int = 0,
    stun_servers: list[str] | None = None,
    turn_servers: list[dict[str, Any]] | None = None,
    timeout: float = 30.0,
    role: str = "offerer",
    signaling_url: str | None = None,
) -> WebRTCTransport | None:
    """Establish a WebRTC data channel connection to a peer.


    Performs full ICE negotiation (NAT traversal) over a signaling
    channel, then returns a connected WebRTCTransport ready for
    tensor send/recv.

    Args:
        peer_host: Target peer hostname or IP.
        peer_port: Target peer's signaling HTTP port.
        local_port: Local signaling port (0 = auto).
        stun_servers: List of STUN server URLs.
        turn_servers: List of TURN server config dicts.
        timeout: Connection timeout in seconds.
        role: 'offerer' (caller) or 'answerer' (listener).
        signaling_url: Override signaling URL template.
            Default: http://{peer_host}:{peer_port}/webrtc

    Returns:
        Connected WebRTCTransport or None on failure.
    """

    if not HAS_WEBRTC:
        logger.error("aiortc not installed; cannot use WebRTC transport")
        return None

    if role == "offerer" and not signaling_url:
        signaling_url = f"http://{peer_host}:{peer_port}/webrtc"

    config = WebRTCConfig(
        stun_servers=stun_servers or ["stun:stun.l.google.com:19302"],
        turn_servers=turn_servers or [],
    )

    transport = WebRTCTransport(config=config, role=role)
    url = signaling_url or f"http://{peer_host}:{peer_port}/webrtc"

    try:
        if role == "offerer":
            offer = await transport.create_offer()
            answer = await WebRTCSignaling.send_offer(f"{url}/offer", offer)
            if answer is None:
                await transport.close()
                return None
            await transport.accept_answer(answer)
        else:
            signal = WebRTCSignaling("webrtc-answerer")
            offer_received = asyncio.Event()
            received_offer = None

            async def on_offer(sdp):
                nonlocal received_offer
                received_offer = sdp
                offer_received.set()
                ans = await transport.accept_offer(sdp)
                return ans

            signal.set_offer_handler(on_offer)
            answer_ready = asyncio.Event()

            if not await transport.wait_connected(timeout=timeout):
                await transport.close()
                return None

        logger.info(f"WebRTC connected to {peer_host}:{peer_port}")
        return transport

    except Exception as e:
        logger.error(f"WebRTC connection to {peer_host}:{peer_port} failed: {e}")
        await transport.close()
        return None


class SignalingServer:
    """WebRTC signaling HTTP server for SDP offer/answer and ICE exchange.


    Listens on a dedicated port, accepts offers from remote peers,
    creates WebRTCTransport instances as the answerer, and exchanges
    ICE candidates until the data channel opens.

    Usage:
        server = SignalingServer(port=50053)
        await server.start()
        # Remote peers can now connect via WebRTC
    """


    def __init__(self, host: str = "0.0.0.0", port: int = 50053):
        self._host = host
        self._port = port
        self._transports: dict[str, WebRTCTransport] = {}
        self._pending_offers: asyncio.Queue[tuple[str, RTCSessionDescription]] = (
            asyncio.Queue()
        )
        self._server: asyncio.AbstractServer | None = None
        self._on_connection: Callable[[WebRTCTransport], Awaitable[None]] | None = None

    def set_connection_handler(
        self, handler: Callable[[WebRTCTransport], Awaitable[None]]
    ) -> None:
        self._on_connection = handler

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_request, self._host, self._port
        )
        logger.info(f"WebRTC signaling server on {self._host}:{self._port}")

    async def stop(self) -> None:
        for tid, transport in list(self._transports.items()):
            try:
                await transport.close()
            except Exception:
                pass
        self._transports.clear()
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _read_http_body(
        self, reader: asyncio.StreamReader
    ) -> tuple[str, str, bytes]:
        request_line = (await reader.readline()).decode().strip()
        if not request_line:
            return ("", "", b"")
        parts = request_line.split(" ")
        method = parts[0]
        path = parts[1] if len(parts) > 1 else "/"
        content_length = 0
        while True:
            header = (await reader.readline()).decode().strip().lower()
            if not header:
                break
            if header.startswith("content-length:"):
                content_length = int(header.split(":")[1].strip())
        body = await reader.read(content_length) if content_length > 0 else b""
        return (method, path, body)

    @staticmethod
    def _send_response(
        writer: asyncio.StreamWriter, status: int, body: bytes
    ) -> None:
        status_text = {200: "OK", 404: "Not Found", 500: "Internal Server Error"}.get(
            status, "Unknown"
        )
        response = (
            f"HTTP/1.1 {status} {status_text}\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Content-Type: application/json\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode() + body
        try:
            writer.write(response)
        finally:
            writer.close()

    async def _handle_request(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            method, path, body = await self._read_http_body(reader)
            if not method:
                writer.close()
                return

            if method == "POST" and path.endswith("/webrtc/offer"):
                data = json.loads(body)
                transport = WebRTCTransport(role="answerer")
                offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
                answer = await transport.accept_offer(offer)
                session_id = str(uuid.uuid4())[:8]
                self._transports[session_id] = transport
                response = json.dumps(
                    {"sdp": answer.sdp, "type": answer.type, "session_id": session_id}
                )
                self._send_response(writer, 200, response.encode())
                await self._on_connected(transport, session_id)

            elif method == "POST" and path.endswith("/webrtc/ice"):
                data = json.loads(body)
                session_id = data.get("session_id", "")
                if session_id in self._transports:
                    transport = self._transports[session_id]
                    candidate = WebRTCSignaling._parse_ice_candidate(data)
                    await transport.add_ice_candidate(candidate)
                self._send_response(writer, 200, b'{"status":"ok"}')

            else:
                self._send_response(writer, 404, b'{"error":"not found"}')
        except Exception as e:
            logger.error(f"Signaling request error: {e}")
            self._send_response(writer, 500, json.dumps({"error": str(e)}).encode())

    async def _on_connected(self, transport: WebRTCTransport, session_id: str) -> None:
        connected = await transport.wait_connected(timeout=30.0)
        if connected and self._on_connection:
            try:
                await self._on_connection(transport)
            except Exception as e:
                logger.error(f"Connection handler error: {e}")


async def connect_with_webrtc(
    peer_host: str,
    peer_port: int,
    stun_servers: list[str] | None = None,
    turn_servers: list[dict[str, Any]] | None = None,
    timeout: float = 30.0,
    signaling_url: str | None = None,
) -> WebRTCTransport | None:
    """Establish a WebRTC data channel connection to a peer.


    Creates an SDP offer as the offerer, sends it to the peer's
    signaling server, accepts the answer, and waits for the data
    channel to open — all with full ICE NAT traversal.

    Args:
        peer_host: Target peer hostname or IP.
        peer_port: Target peer's signaling HTTP port.
        stun_servers: List of STUN server URLs.
        turn_servers: List of TURN server config dicts.
        timeout: Connection timeout in seconds.
        signaling_url: Override signaling URL.
            Default: http://{peer_host}:{peer_port}/webrtc/offer

    Returns:
        Connected WebRTCTransport or None on failure.
    """

    if not HAS_WEBRTC:
        logger.error("aiortc not installed; cannot use WebRTC transport")
        return None

    offer_url = signaling_url or f"http://{peer_host}:{peer_port}/webrtc/offer"

    config = WebRTCConfig(
        stun_servers=stun_servers or ["stun:stun.l.google.com:19302"],
        turn_servers=turn_servers or [],
    )

    transport = WebRTCTransport(config=config, role="offerer")

    try:
        offer = await transport.create_offer()
        answer_data = await WebRTCSignaling.send_offer(offer_url, offer)
        if answer_data is None:
            logger.error("No answer received from peer")
            await transport.close()
            return None
        await transport.accept_answer(answer_data)
        if not await transport.wait_connected(timeout=timeout):
            logger.warning("WebRTC connection timed out after SDP exchange")
            await transport.close()
            return None
        logger.info(f"WebRTC connected to {peer_host}:{peer_port}")
        return transport
    except Exception as e:
        logger.error(f"WebRTC connection to {peer_host}:{peer_port} failed: {e}")
        await transport.close()
        return None


async def serve_webrtc_signaling(
    host: str = "0.0.0.0",
    port: int = 50053,
    on_connection: Callable[[WebRTCTransport], Awaitable[None]] | None = None,
) -> None:
    """Run a WebRTC signaling server (convenience wrapper).


    Creates a SignalingServer and runs it until cancelled.

    Args:
        host: Bind address.
        port: Bind port (default 50053).
        on_connection: Async callback when a WebRTC connection is established.
    """

    server = SignalingServer(host=host, port=port)
    if on_connection:
        server.set_connection_handler(on_connection)
    await server.start()
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop()
