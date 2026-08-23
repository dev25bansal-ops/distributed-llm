"""NAT traversal for cross-internet clusters.

Provides STUN client (NAT type detection), TURN relay protocol,
ICE-style connection negotiation, and WebRTC data channel transport
for connecting devices across different home networks.

Architecture:
    Device A (NAT) ══WebRTC/ICE══► Device B (NAT)    (primary, best)
    Device A ──STUN──► Public STUN server ──► public IP:port
    Device A ◄──TURN relay──► Device B (fallback)

Usage:
    # WebRTC (recommended) — use ICE for NAT traversal
    from distllm.dist.webrtc import connect_with_webrtc
    transport = await connect_with_webrtc("peer.example.com", 50053)
    await transport.send_tensor(hidden_states_bytes)

    # Legacy — custom TURN relay
    distllm cluster join --coordinator public.example.com:50050
"""


from __future__ import annotations
import hashlib
import hmac
import os
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


class NatType(Enum):
    UNKNOWN = "unknown"
    OPEN = "open"                   # Has public IP, direct connection works
    FULL_CONE = "full_cone"        # Any external host can send to us
    RESTRICTED = "restricted"       # Only hosts we've sent to can reach us
    PORT_RESTRICTED = "port_restricted"  # Same as above but port-restricted
    SYMMETRIC = "symmetric"        # Most restrictive; requires TURN relay


@dataclass
class NatMapping:
    """The public IP:port a NAT assigns for our traffic."""

    public_ip: str = ""
    public_port: int = 0
    nat_type: NatType = NatType.UNKNOWN
    local_ip: str = ""
    local_port: int = 0


class StunClient:
    """Simple STUN client for NAT type detection.


    Uses google's public STUN server by default.
    Implements basic STUN binding request (RFC 3489).

    Usage:
        client = StunClient()
        mapping = client.detect()
        print(f"Public address: {mapping.public_ip}:{mapping.public_port}")
        print(f"NAT type: {mapping.nat_type.value}")
    """


    STUN_SERVERS = [
        ("stun.l.google.com", 19302),
        ("stun1.l.google.com", 19302),
        ("stun2.l.google.com", 19302),
        ("stun3.l.google.com", 19302),
    ]

    STUN_MAGIC_COOKIE = 0x2112A442
    BINDING_REQUEST = 0x0001
    # STUN attribute types (RFC 3489 / RFC 5389)
    ATTR_MAPPED_ADDRESS = 0x0001      # RFC 3489
    ATTR_CHANGE_REQUEST = 0x0003      # RFC 3489
    ATTR_CHANGED_ADDRESS = 0x0004     # RFC 3489
    ATTR_XOR_MAPPED_ADDRESS = 0x0020  # RFC 5389

    def __init__(self, timeout: float = 3.0):
        self._timeout = timeout

    def detect(self, server_index: int = 0) -> NatMapping:
        """Detect NAT type and public address using STUN."""

        mapping = NatMapping()

        try:
            server = self.STUN_SERVERS[server_index]
            public_addr = self._stun_binding_request(server)

            if public_addr:
                mapping.public_ip = public_addr[0]
                mapping.public_port = public_addr[1]
                mapping.nat_type = self._classify_nat(server,
                                                       (mapping.public_ip, mapping.public_port))
                logger.info(f"STUN: public={mapping.public_ip}:{mapping.public_port}, "
                             f"NAT={mapping.nat_type.value}")
            else:
                logger.warning("STUN: no response from any server")
                mapping.nat_type = NatType.UNKNOWN

        except Exception as e:
            logger.warning(f"STUN detection failed: {e}")
            mapping.nat_type = NatType.UNKNOWN

        return mapping

    def _stun_binding_request(self, server: tuple[str, int]) -> tuple[str, int] | None:
        """Send a STUN binding request and parse the response."""

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(self._timeout)

        try:
            trans_id = struct.pack(">QQ", 0, int(time.time() * 1e9))
            req = struct.pack(">HHI", self.BINDING_REQUEST, 0x0000, self.STUN_MAGIC_COOKIE)
            req += trans_id

            sock.sendto(req, server)
            data, addr = sock.recvfrom(4096)

            if len(data) < 20:
                return None

            msg_type, msg_len, cookie = struct.unpack_from(">HHI", data)
            if cookie != self.STUN_MAGIC_COOKIE:
                return None

            pos = 20
            while pos < len(data):
                attr_type, attr_len = struct.unpack_from(">HH", data, pos)
                pos += 4
                if attr_len >= 8:
                    if attr_type == self.ATTR_MAPPED_ADDRESS:
                        # RFC 3489: plain MAPPED-ADDRESS
                        family = data[pos + 1]
                        port = struct.unpack_from(">H", data, pos + 2)[0]
                        ip_bytes = data[pos + 4:pos + 8]
                        ip = socket.inet_ntoa(ip_bytes)
                        return (ip, port)
                    elif attr_type == self.ATTR_XOR_MAPPED_ADDRESS:
                        # RFC 5389: XOR-MAPPED-ADDRESS
                        port = struct.unpack_from(">H", data, pos + 2)[0]
                        port ^= self.STUN_MAGIC_COOKIE >> 16
                        ip_bytes = data[pos + 4:pos + 8]
                        magic = struct.pack(">I", self.STUN_MAGIC_COOKIE)
                        ip_bytes = bytes(b ^ c for b, c in zip(ip_bytes, magic))
                        ip = socket.inet_ntoa(ip_bytes)
                        return (ip, port)
                pos += attr_len

        except socket.timeout:
            pass
        except Exception as e:
            logger.debug(f"STUN error: {e}")
        finally:
            sock.close()

        return None

    def _classify_nat(self, server: tuple[str, int],
                      primary_mapping: tuple[str, int]) -> NatType:
        """Classify NAT type using RFC 3489 algorithm.


        Performs three probes:
          1. Primary request to *server* (already done — ``primary_mapping``)
          2. Request with CHANGE-REQUEST to change IP+port
          3. Request to a *different* STUN server (different IP, same port)

        Returns:
            NatType.OPEN if no NAT detected,
            NatType.FULL_CONE, NatType.RESTRICTED,
            NatType.PORT_RESTRICTED, or NatType.SYMMETRIC.
        """

        primary_ip, primary_port = primary_mapping

        # ── Probe 2: send CHANGE-REQUEST to force a different IP:port ──
        changed_addr = self._stun_change_request(server)
        if changed_addr:
            changed_ip, changed_port = changed_addr
            if changed_ip == primary_ip and changed_port == primary_port:
                # Same mapped address even with different source — no NAT
                return NatType.OPEN
            # Different mapped address — NAT is present
            # We'll distinguish full-cone from symmetric below
        else:
            # No response to CHANGE-REQUEST — NAT is restrictive
            changed_ip, changed_port = None, None

        # ── Probe 3: send to a *different* STUN server (same port) ──
        # Use a secondary server to see if the NAT assigns a different mapping
        alt_server = self._pick_alt_server(server)
        alt_mapping = self._stun_binding_request(alt_server) if alt_server else None

        if alt_mapping:
            alt_ip, alt_port = alt_mapping
            if alt_ip == primary_ip and alt_port == primary_port:
                # Same mapping to different server — full-cone NAT
                return NatType.FULL_CONE
            else:
                # Different mapping per destination — symmetric NAT
                return NatType.SYMMETRIC

        # Secondary server also failed or unavailable
        if changed_addr is None:
            # Both CHANGE-REQUEST and secondary server failed —
            # port-restricted cone (most likely)
            return NatType.PORT_RESTRICTED
        else:
            # CHANGE-REQUEST succeeded but secondary failed —
            # restricted cone
            return NatType.RESTRICTED

    def _stun_change_request(self, server: tuple[str, int]) -> tuple[str, int] | None:
        """Send STUN binding request with CHANGE-REQUEST (change IP and port).


        RFC 3489: CHANGE-REQUEST attribute asks the server to respond from
        a different IP and port than it received the request on.
        """

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(self._timeout)
        try:
            trans_id = struct.pack(">QQ", 1, int(time.time() * 1e9))
            req = struct.pack(">HHI", self.BINDING_REQUEST, 0x0000, self.STUN_MAGIC_COOKIE)
            req += trans_id
            # CHANGE-REQUEST: change IP and port (value 0x00000006)
            req += struct.pack(">HH", self.ATTR_CHANGE_REQUEST, 4)
            req += struct.pack(">I", 0x00000006)

            sock.sendto(req, server)
            data, addr = sock.recvfrom(4096)

            if len(data) < 20:
                return None

            msg_type, msg_len, cookie = struct.unpack_from(">HHI", data)
            if cookie != self.STUN_MAGIC_COOKIE:
                return None

            pos = 20
            while pos < len(data):
                attr_type, attr_len = struct.unpack_from(">HH", data, pos)
                pos += 4
                if attr_len >= 8:
                    if attr_type in (self.ATTR_XOR_MAPPED_ADDRESS,
                                     self.ATTR_MAPPED_ADDRESS,
                                     self.ATTR_CHANGED_ADDRESS):
                        family = data[pos + 1]
                        port = struct.unpack_from(">H", data, pos + 2)[0]
                        ip_bytes = data[pos + 4:pos + 8]
                        if attr_type in (self.ATTR_XOR_MAPPED_ADDRESS,
                                         self.ATTR_CHANGED_ADDRESS):
                            port ^= self.STUN_MAGIC_COOKIE >> 16
                            magic = struct.pack(">I", self.STUN_MAGIC_COOKIE)
                            ip_bytes = bytes(b ^ c for b, c in zip(ip_bytes, magic))
                        ip = socket.inet_ntoa(ip_bytes)
                        return (ip, port)
                pos += attr_len

        except socket.timeout:
            return None
        except Exception as e:
            logger.debug(f"STUN change-request error: {e}")
            return None
        finally:
            sock.close()
        return None

    def _pick_alt_server(self, current: tuple[str, int]) -> tuple[str, int] | None:
        """Pick a STUN server different from *current* for the classification probe."""

        for alt in self.STUN_SERVERS:
            if alt[0] != current[0]:
                return alt
        return None


class TurnRelayServer:
    """Session-based TURN-like relay server for NAT traversal.


    Clients join a named session with ``JOIN:<session_token>``.
    Once both peers have joined the same session, data is forwarded
    between them. Any data that is not a JOIN command is forwarded
    to the paired peer.

    Security:
    - Session tokens are validated as HMAC-SHA256 signatures
    - Per-IP rate limiting prevents abuse
    - All relay traffic should be E2E encrypted by the application layer

    Usage (run on public server):
        server = TurnRelayServer(port=3478, hmac_key="shared-secret")
        server.start()
    """


    JOIN_PREFIX = b"JOIN:"

    SESSION_TTL_SECONDS = 3600  # 1 hour
    MAX_SESSIONS = 1000

    def __init__(self, host: str = "0.0.0.0", port: int = 3478,
                 relay_port_range: tuple[int, int] = (49152, 65535),
                 hmac_key: str | None = None,
                 max_joins_per_minute: int = 10):
        self._host = host
        self._port = port
        self._running = threading.Event()
        self._hmac_key = hmac_key or os.environ.get("DISTLLM_RELAY_HMAC_KEY", "")
        self._max_joins_per_minute = max_joins_per_minute
        # session_token -> [peer_a_addr, peer_b_addr]
        self._sessions: dict[str, list[tuple[str, int]]] = {}
        # addr -> session_token (reverse lookup)
        self._addr_to_session: dict[tuple[str, int], str] = {}
        # Per-IP rate limiting: ip -> list of join timestamps
        self._join_attempts: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        # session_token -> creation timestamp (for session TTL enforcement)
        self._session_created: dict[str, float] = {}
        self._last_cleanup: float = time.monotonic()

    def _validate_session_token(self, token: str) -> bool:
        """Validate session token as HMAC-SHA256 signature.


        Token must be: base64(hmac_sha256(session_id, key))

        Security: An HMAC key MUST be configured in production.
        If no HMAC key is set, all tokens are rejected (fail closed).
        """

        if not self._hmac_key:
            logger.error("Rejecting TURN relay session: no HMAC_KEY configured. "
                        "Set DISTLLM_RELAY_HMAC_KEY in production.")
            return False  # Fail closed: reject all tokens when no key set

        try:
            # Token format: session_id.signature
            parts = token.split(".", 1)
            if len(parts) != 2:
                return False
            session_id, signature = parts
            expected = hmac.new(
                self._hmac_key.encode(),
                session_id.encode(),
                hashlib.sha256,
            ).hexdigest()
            return hmac.compare_digest(signature, expected)
        except Exception:
            return False

    def _check_rate_limit(self, ip: str) -> bool:
        """Check if IP has exceeded join rate limit."""

        now = time.time()
        cutoff = now - 60.0
        with self._lock:
            attempts = self._join_attempts.get(ip, [])
            # Prune old entries
            attempts = [t for t in attempts if t > cutoff]
            self._join_attempts[ip] = attempts
            if len(attempts) >= self._max_joins_per_minute:
                return False
            attempts.append(now)
            return True

    def start(self) -> None:
        """Start the relay server.


        Raises:
            RuntimeError: If no HMAC key is configured (production safety).
        """

        if not self._hmac_key:
            raise RuntimeError(
                "TURN relay server requires DISTLLM_RELAY_HMAC_KEY to be set. "
                "This is a production safety check. Generate a key with:\n"
                "  python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._host, self._port))
        sock.settimeout(1.0)
        self._running.set()
        self._relay_sock = sock  # M-01: Store for reuse in _leave_session
        logger.info(f"TURN relay listening on {self._host}:{self._port}")

        while self._running.is_set():
            self._expire_stale_sessions()
            try:
                data, addr = sock.recvfrom(65535)
                self._handle_relay(sock, data, addr)
            except socket.timeout:
                continue
            except Exception as e:
                logger.debug(f"Relay error: {e}")

    def stop(self) -> None:
        self._running.clear()

    def _expire_stale_sessions(self) -> None:
        """Remove sessions that have exceeded the TTL."""

        now = time.monotonic()
        if now - self._last_cleanup < 300:  # Only run cleanup every 5 minutes
            return
        self._last_cleanup = now
        with self._lock:
            expired = [
                token for token, created in self._session_created.items()
                if now - created > self.SESSION_TTL_SECONDS
            ]
            for token in expired:
                peers = self._sessions.get(token, [])
                for addr in peers:
                    self._addr_to_session.pop(addr, None)
                del self._sessions[token]
                del self._session_created[token]
                logger.info(f"Session {token} expired (TTL={self.SESSION_TTL_SECONDS}s)")
            total = len(self._sessions)
            if total > self.MAX_SESSIONS:
                # Evict oldest sessions if over cap
                over = total - self.MAX_SESSIONS
                sorted_tokens = sorted(self._session_created.items(), key=lambda x: x[1])
                for token, _ in sorted_tokens[:over]:
                    peers = self._sessions.get(token, [])
                    for addr in peers:
                        self._addr_to_session.pop(addr, None)
                    del self._sessions[token]
                    del self._session_created[token]
                    logger.warning(f"Session {token} evicted (max sessions = {self.MAX_SESSIONS})")

    def _handle_relay(self, sock: socket.socket, data: bytes, addr: tuple) -> None:
        """Handle a relay message: JOIN or data forwarding.


        Automatically expires stale sessions every 5 minutes.
        """

        if data.startswith(self.JOIN_PREFIX):
            self._handle_join(sock, data, addr)
        else:
            self._forward_data(sock, data, addr)

    def _handle_join(self, sock: socket.socket, data: bytes, addr: tuple) -> None:
        """Handle a JOIN:<session_token> request.


        Validates the session token (HMAC), checks rate limits,
        and registers *addr* for *session_token*.
        """

        token = data[len(self.JOIN_PREFIX):].decode("utf-8", errors="replace").strip()
        if not token:
            logger.warning(f"Empty session token from {addr}")
            sock.sendto(b"ERROR: empty token", addr)
            return

        # Rate limiting
        ip = addr[0] if isinstance(addr, tuple) else str(addr)
        if not self._check_rate_limit(ip):
            logger.warning(f"Rate limit exceeded for {ip}")
            sock.sendto(b"ERROR: rate limited", addr)
            return

        # HMAC validation
        if not self._validate_session_token(token):
            logger.warning(f"Invalid session token from {ip}")
            sock.sendto(b"ERROR: invalid token", addr)
            return

        with self._lock:
            # Check if this addr is already in a session
            old_token = self._addr_to_session.get(addr)
            if old_token:
                if old_token == token:
                    return
                self._leave_session(addr, old_token)

            # Join or create the session
            if token not in self._sessions:
                self._sessions[token] = []
                self._session_created[token] = time.monotonic()
                if len(self._sessions) > self.MAX_SESSIONS * 2:
                    self._expire_stale_sessions()
            peers = self._sessions[token]

            if addr in peers:
                return

            peers.append(addr)
            self._addr_to_session[addr] = token
            logger.info(f"Peer {addr} joined session {token} "
                        f"({len(peers)}/2 connected)")

            if len(peers) == 2:
                peer_a, peer_b = peers[0], peers[1]
                logger.info(f"Session {token} paired: {peer_a} <-> {peer_b}")
                sock.sendto(b"PAIRED", peer_a)
                sock.sendto(b"PAIRED", peer_b)
            else:
                sock.sendto(f"WAITING:{token}".encode(), addr)

    def _forward_data(self, sock: socket.socket, data: bytes, addr: tuple) -> None:
        """Forward data to the peer paired with *addr*."""

        with self._lock:
            token = self._addr_to_session.get(addr)
            if token is None:
                return

            peers = self._sessions.get(token)
            if peers is None or len(peers) < 2:
                return

            peer_a, peer_b = peers
            target = peer_b if addr == peer_a else peer_a
            if addr == peer_a or addr == peer_b:
                try:
                    sock.sendto(data, target)
                except Exception:
                    logger.debug(f"Failed to forward data to {target}")

    def _leave_session(self, addr: tuple, token: str) -> None:
        """Remove *addr* from its session."""

        peers = self._sessions.get(token)
        if peers and addr in peers:
            peers.remove(addr)
            self._addr_to_session.pop(addr, None)
            if not peers:
                del self._sessions[token]
                self._session_created.pop(token, None)
                logger.info(f"Session {token} closed")
            else:
                remaining = peers[0]
                # M-01: Reuse the relay socket instead of creating a temporary one
                if hasattr(self, '_relay_sock'):
                    try:
                        self._relay_sock.sendto(b"PEER_DISCONNECTED", remaining)
                    except Exception:
                        pass
                logger.info(f"Peer {addr} left session {token}")


class TurnRelayClient:
    """Client for connecting through a session-based TURN relay server.


    Both peers must use the same ``session_token`` to be paired.

    Usage:
        token = "my-session-123"
        client = TurnRelayClient("relay.distllm.ai", 3478, session_token=token)
        client.join()
        # Now send/receive data through the relay
    """


    def __init__(self, relay_host: str, relay_port: int = 3478,
                 session_token: str = "", local_port: int = 0):
        self._relay_addr = (relay_host, relay_port)
        self._session_token = session_token
        self._local_port = local_port
        self._sock: socket.socket | None = None

    def join(self) -> bool:
        """Join a relay session.


        Sends ``JOIN:<session_token>`` to the relay server and waits
        for either ``PAIRED`` (both peers connected) or
        ``WAITING:<token>`` (waiting for the other peer).

        Returns:
            True if paired, False if waiting or failed.
        """

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(30.0)

        try:
            if self._local_port:
                sock.bind(("0.0.0.0", self._local_port))

            join_msg = f"JOIN:{self._session_token}".encode()
            sock.sendto(join_msg, self._relay_addr)
            data, _ = sock.recvfrom(4096)

            if data == b"PAIRED":
                self._sock = sock
                logger.info(f"Relay paired for session {self._session_token}")
                return True

            response = data.decode("utf-8", errors="replace")
            if response.startswith("WAITING:"):
                self._sock = sock
                token = response[len("WAITING:"):]
                logger.info(f"Relay joined session {token}, waiting for peer")
                return self._wait_for_pair(sock)

            if response.startswith("ERROR:"):
                logger.warning(f"Relay error: {response}")
                sock.close()
                return False

        except socket.timeout:
            logger.warning("Relay join timed out")
        except Exception as e:
            logger.warning(f"Relay join failed: {e}")
        finally:
            if self._sock is None:
                sock.close()

        return False

    def _wait_for_pair(self, sock: socket.socket) -> bool:
        """Wait for the PAIRED signal from the relay."""

        try:
            data, _ = sock.recvfrom(4096)
            if data == b"PAIRED":
                logger.info(f"Relay session {self._session_token} is now paired")
                return True
            if data == b"PEER_DISCONNECTED":
                logger.warning("Relay peer disconnected while waiting")
                return False
        except socket.timeout:
            logger.warning("Timed out waiting for relay peer")
        except Exception as e:
            logger.warning(f"Error waiting for relay pair: {e}")
        return False

    def send(self, data: bytes) -> bool:
        """Send data through the relay to the paired peer.


        Args:
            data: Raw bytes to forward.

        Returns:
            True if sent successfully.
        """

        if self._sock is None:
            logger.warning("Cannot send: relay not joined")
            return False
        try:
            self._sock.sendto(data, self._relay_addr)
            return True
        except Exception as e:
            logger.warning(f"Relay send failed: {e}")
            return False

    def recv(self, bufsize: int = 65535, timeout: float | None = None) -> bytes | None:
        """Receive data from the relay.


        Args:
            bufsize: Maximum buffer size.
            timeout: Receive timeout in seconds (None = blocking).

        Returns:
            Received data bytes, or None on timeout/error.
        """

        if self._sock is None:
            return None
        try:
            if timeout is not None:
                self._sock.settimeout(timeout)
            data, _ = self._sock.recvfrom(bufsize)
            if data == b"PEER_DISCONNECTED":
                logger.warning("Relay peer disconnected")
                self.close()
                return None
            return data
        except socket.timeout:
            return None
        except Exception as e:
            logger.warning(f"Relay recv failed: {e}")
            return None

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None


# ── WebRTC-based NAT traversal ──────────────────────────────────────────

try:
    from distllm.dist.webrtc import (
        WebRTCTransport,
        WebRTCConfig,
        connect_with_webrtc,
        HAS_WEBRTC,
    )
except ImportError:
    HAS_WEBRTC = False
    WebRTCTransport = None
    WebRTCConfig = None

    async def connect_with_webrtc(*args, **kwargs):  # type: ignore[misc]
        return None


class WebRTCNatTransport:
    """Adapter wrapping WebRTCTransport into the nat.py interface.


    Provides a ``send``/``recv`` API compatible with the existing
    ``TurnRelayClient`` interface, backed by WebRTC data channels.
    """


    def __init__(self, transport: WebRTCTransport | None = None):
        self._transport = transport
        self._recv_buffer = bytearray()

    def send(self, data: bytes) -> bool:
        """Send raw bytes through the WebRTC data channel.


        Note: This is a synchronous shim. For high-performance use,
        call ``transport.send_tensor()`` directly.
        """

        if self._transport is None or not self._transport.is_connected:
            return False
        try:
            import asyncio as _asyncio
            loop = _asyncio.new_event_loop()
            try:
                loop.run_until_complete(self._transport.send_tensor(data))
            finally:
                loop.close()
            return True
        except Exception:
            return False

    def recv(self, bufsize: int = 65535, timeout: float | None = None) -> bytes | None:
        """Receive raw bytes from the WebRTC data channel."""

        if self._transport is None:
            return None
        try:
            import asyncio as _asyncio
            loop = _asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    self._transport.recv_tensor(timeout=timeout)
                )
            finally:
                loop.close()
            if result:
                _, _, data = result
                if isinstance(data, bytes):
                    return data
                return b""
            return None
        except Exception:
            return None

    def close(self) -> None:
        if self._transport is not None:
            import asyncio as _asyncio
            try:
                loop = _asyncio.new_event_loop()
                loop.run_until_complete(self._transport.close())
                loop.close()
            except Exception:
                pass
            self._transport = None

    @property
    def is_connected(self) -> bool:
        return self._transport is not None and self._transport.is_connected


async def connect_with_nat_traversal_async(
    host: str,
    port: int,
    timeout: float = 30.0,
    stun_servers: list[str] | None = None,
    turn_servers: list[dict[str, Any]] | None = None,
    signaling_port: int | None = None,
) -> tuple[tuple[str, int] | None, WebRTCNatTransport | TurnRelayClient | None]:
    """Async NAT traversal — tries WebRTC/ICE first, falls back to TURN relay.


    WebRTC (via aiortc) provides proven ICE-based NAT traversal with:
    - STUN binding for public address discovery
    - TURN relay fallback when direct connection fails
    - DTLS 1.2 encryption
    - SCTP congestion control

    Args:
        host: Target peer hostname.
        port: Target peer's gRPC port.
        timeout: Connection timeout in seconds.
        stun_servers: List of STUN server URLs.
            Defaults to Google's public STUN servers.
        turn_servers: List of TURN server config dicts.
            Each dict: {"urls": [...], "username": str, "credential": str}
        signaling_port: Peer's WebRTC signaling HTTP port.
            Default: port + 2 (e.g., gRPC 50051 → signaling 50053)

    Returns:
        Tuple of (resolved_address, transport_client).
        If WebRTC succeeds, resolved_address is None and transport_client
        is a WebRTCNatTransport ready for send/recv.
        Falls back to legacy TURN relay on failure.
    """

    sig_port = signaling_port or (port + 2)
    sig_url = f"http://{host}:{sig_port}/webrtc"

    # Try WebRTC/ICE first
    if HAS_WEBRTC:
        try:
            transport = await connect_with_webrtc(
                peer_host=host,
                peer_port=sig_port,
                stun_servers=stun_servers,
                turn_servers=turn_servers,
                timeout=timeout,
                signaling_url=sig_url,
            )
            if transport is not None:
                logger.info(f"WebRTC NAT traversal connected to {host}:{sig_port}")
                return (None, WebRTCNatTransport(transport))
        except Exception as e:
            logger.debug(f"WebRTC connection failed, falling back: {e}")
    else:
        logger.debug("aiortc not available; skipping WebRTC NAT traversal")

    # Fall back to legacy TURN relay
    logger.info("Falling back to legacy TURN relay")
    relay_client = TurnRelayClient(host, port)
    if relay_client.join():
        return (None, relay_client)

    logger.warning("All NAT traversal attempts failed")
    return (None, None)


def connect_with_nat_traversal(
    host: str,
    port: int,
    timeout: float = 10.0,
    use_relay: bool = False,
    relay_host: str = "relay.distllm.ai",
    relay_port: int = 3478,
    session_token: str = "",
) -> tuple[tuple[str, int] | None, TurnRelayClient | None]:
    """Attempt to connect to a peer using NAT traversal (sync wrapper).


    NOTE: Prefer the async ``connect_with_nat_traversal_async()`` which
    tries WebRTC/ICE first for proven NAT traversal with encryption.

    Legacy fallback strategy:
    1. Try direct TCP connection first
    2. If STUN is enabled, attempt UDP hole-punching
    3. If relay is enabled, fall back to session-based TURN relay

    Args:
        host: Target peer hostname.
        port: Target peer port.
        timeout: Connection timeout.
        use_relay: Whether to fall back to TURN relay.
        relay_host: TURN relay server hostname.
        relay_port: TURN relay server port.
        session_token: Shared session token for relay pairing.
            Both peers MUST use the same token.

    Returns:
        Tuple of (resolved_address, relay_client).
        If direct connection succeeded, relay_client is None.
        If relay is used, resolved_address is None and relay_client is the
        connected relay client for sending/receiving data.
    """

    import socket as _socket

    # Try direct connection first
    try:
        s = _socket.create_connection((host, port), timeout=timeout)
        s.close()
        logger.debug(f"Direct connection to {host}:{port} succeeded")
        return ((host, port), None)
    except (OSError, _socket.timeout):
        logger.debug(f"Direct connection to {host}:{port} failed, trying alternatives")

    # Try STUN hole-punching
    stun = StunClient()
    mapping = stun.detect()
    if mapping.nat_type in (NatType.OPEN, NatType.FULL_CONE, NatType.RESTRICTED):
        logger.info(f"NAT type {mapping.nat_type.value} — should allow connections")
        return ((host, port), None)

    # Fall back to session-based TURN relay
    if use_relay and session_token:
        logger.info(f"Using TURN relay {relay_host}:{relay_port} "
                     f"session={session_token}")
        client = TurnRelayClient(
            relay_host, relay_port,
            session_token=session_token,
        )
        if client.join():
            return (None, client)
        client.close()

    logger.warning("All connection attempts failed")
    return (None, None)
