"""ICE transport for NAT traversal — RFC 8445 ICE, RFC 5766 TURN, UDP hole-punching.

Provides four layers of connectivity establishment for peers behind NATs:

1.  **ICEAgent** (RFC 8445) — full Interactive Connectivity Establishment:
    - Candidate gathering: host, server-reflexive (STUN), relayed (TURN)
    - Candidate pairing by priority, connectivity checks via STUN binding
    - Nominated-candidate selection and peer-to-peer data transport

2.  **TURNClient** / **TURNServer** (RFC 5766) — standard TURN relay:
    - Allocate, Refresh, CreatePermission, ChannelBind, Send/Data indications
    - Long-term credential authentication with MESSAGE-INTEGRITY
    - Channel-data messages for reduced per-packet overhead

3.  **UDPHolePuncher** — lightweight direct-connection fallback:
    - Discovers public address via STUN, then pings peer directly
    - Exponential-backoff retry, graceful fallback to TURN

4.  **NATTraversalController** — strategy orchestrator:
    - ICE first, hole-punch fallback, TURN last resort
    - ``stats()`` returns strategy used, success rate, latency

Usage::

    controller = NATTraversalController(
        stun_servers=[("stun.l.google.com", 19302)],
        turn_config={"server": "turn.example.com", "port": 3478, "user": "u", "password": "p"},
    )
    transport = controller.connect([Candidate(type="host", host="1.2.3.4", port=50051)])
    if transport:
        transport.send(b"hello")
"""

from __future__ import annotations

import hashlib
import hmac
import os
import random
import select
import socket
import struct
import threading
import time
import uuid
import zlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


# ── STUN Protocol Constants ───────────────────────────────────────────────────

STUN_MAGIC_COOKIE = 0x2112A442

# STUN message types
STUN_BINDING_REQUEST = 0x0001
STUN_BINDING_RESPONSE = 0x0101
STUN_BINDING_ERROR = 0x0111

# TURN message types (RFC 5766)
TURN_ALLOCATE_REQUEST = 0x0003
TURN_ALLOCATE_RESPONSE = 0x0103
TURN_ALLOCATE_ERROR = 0x0113
TURN_REFRESH_REQUEST = 0x0004
TURN_REFRESH_RESPONSE = 0x0104
TURN_REFRESH_ERROR = 0x0114
TURN_SEND_INDICATION = 0x0006
TURN_DATA_INDICATION = 0x0007
TURN_CREATE_PERMISSION_REQUEST = 0x0008
TURN_CREATE_PERMISSION_RESPONSE = 0x0108
TURN_CREATE_PERMISSION_ERROR = 0x0118
TURN_CHANNEL_BIND_REQUEST = 0x0009
TURN_CHANNEL_BIND_RESPONSE = 0x0109
TURN_CHANNEL_BIND_ERROR = 0x0119

# STUN / TURN attributes
ATTR_MAPPED_ADDRESS = 0x0001
ATTR_USERNAME = 0x0006
ATTR_MESSAGE_INTEGRITY = 0x0008
ATTR_ERROR_CODE = 0x0009
ATTR_UNKNOWN_ATTRIBUTES = 0x000A
ATTR_REALM = 0x0014
ATTR_NONCE = 0x0015
ATTR_XOR_MAPPED_ADDRESS = 0x0020
ATTR_XOR_RELAYED_ADDRESS = 0x0016
ATTR_REQUESTED_TRANSPORT = 0x0019
ATTR_LIFETIME = 0x000D
ATTR_XOR_PEER_ADDRESS = 0x0012
ATTR_DATA = 0x0013
ATTR_CHANNEL_NUMBER = 0x000C
ATTR_DONT_FRAGMENT = 0x001A
ATTR_RESERVATION_TOKEN = 0x0022
ATTR_PRIORITY = 0x0024
ATTR_USE_CANDIDATE = 0x0025
ATTR_ICE_CONTROLLING = 0x002A
ATTR_ICE_CONTROLLED = 0x0029
ATTR_FINGERPRINT = 0x8028

# Channel-data message (RFC 5766 Section 11) — first two bits 0b01
CHANNEL_DATA_MIN = 0x4000
CHANNEL_DATA_MAX = 0x7FFE

# Requested transport: UDP = 17
TRANSPORT_UDP = 17

DEFAULT_STUN_SERVERS = [
    ("stun.l.google.com", 19302),
    ("stun1.l.google.com", 19302),
    ("stun2.l.google.com", 19302),
    ("stun3.l.google.com", 19302),
]

TURN_DEFAULT_PORT = 3478

# Candidate type preferences (RFC 8445 Section 5.1.2.1)
CANDIDATE_TYPE_PREF: dict[str, int] = {
    "host": 126,
    "srflx": 100,
    "relay": 0,
}

# Maximum chunks for channel-data (RFC 5766 Section 11)
CHANNEL_DATA_MAX_LEN = 65535


# ── Exceptions ────────────────────────────────────────────────────────────────


class ICEError(Exception):
    """Base exception for ICE/TURN operations."""


class ICEConnectionError(ICEError):
    """Connectivity check failed or no usable candidate pair found."""


class TURNProtocolError(ICEError):
    """TURN server returned an error response."""


class TURNAuthError(TURNProtocolError):
    """TURN authentication failed."""


# ── ICE Candidate Types ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Candidate:
    """An ICE candidate representing a potential communication endpoint.

    Attributes:
        foundation: Opaque identifier for the candidate type + network interface.
        component: Component ID (1 = RTP / data, 2 = RTCP).
        transport: Transport protocol (``"UDP"``).
        priority: Computed priority per RFC 8445 Section 5.1.2.1.
        host: IP address string (v4 or v6).
        port: UDP port.
        type: ``"host"``, ``"srflx"``, or ``"relay"``.
        rel_addr: Related address (for srflx / relay candidates).
        rel_port: Related port (for srflx / relay candidates).
    """

    foundation: str = ""
    component: int = 1
    transport: str = "UDP"
    priority: int = 0
    host: str = "0.0.0.0"
    port: int = 0
    type: str = "host"
    rel_addr: str = ""
    rel_port: int = 0

    def addr(self) -> tuple[str, int]:
        """Return the network address as (host, port)."""
        return (self.host, self.port)


@dataclass
class CandidatePair:
    """A pair of local and remote candidates with connectivity state.

    Attributes:
        local:   Local candidate.
        remote:  Remote candidate.
        priority: Pair priority (RFC 8445 Section 6.1.2.2).
        state:   ``"frozen"`` | ``"waiting"`` | ``"in_progress"`` |
                 ``"succeeded"`` | ``"failed"``.
        nominated: Whether this pair was nominated for data transport.
        rtt:      Round-trip time in seconds (float).
    """

    local: Candidate
    remote: Candidate
    priority: int = 0
    state: str = "frozen"
    nominated: bool = False
    rtt: float = 0.0


# ── Transport interface for data after ICE/TURN negotiation ─────────────────


class DataTransport:
    """Minimal transport interface for sending/receiving raw bytes.

    Returned by :meth:`NATTraversalController.connect`.
    """

    def send(self, data: bytes) -> bool:
        raise NotImplementedError

    def recv(self, bufsize: int = 65535, timeout: float | None = None) -> bytes | None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    @property
    def is_connected(self) -> bool:
        return True

    @property
    def local_addr(self) -> tuple[str, int]:
        raise NotImplementedError

    @property
    def remote_addr(self) -> tuple[str, int]:
        raise NotImplementedError


class _UDPTransport(DataTransport):
    """Simple UDP transport wrapping a socket + remote address."""

    def __init__(self, sock: socket.socket, remote: tuple[str, int]):
        self._sock = sock
        self._remote = remote
        self._closed = False

    def send(self, data: bytes) -> bool:
        if self._closed:
            return False
        try:
            self._sock.sendto(data, self._remote)
            return True
        except OSError:
            return False

    def recv(self, bufsize: int = 65535, timeout: float | None = None) -> bytes | None:
        if self._closed:
            return None
        try:
            if timeout is not None:
                self._sock.settimeout(timeout)
            data, _ = self._sock.recvfrom(bufsize)
            return data
        except socket.timeout:
            return None
        except OSError:
            return None

    def close(self) -> None:
        self._closed = True
        try:
            self._sock.close()
        except OSError:
            pass

    @property
    def is_connected(self) -> bool:
        return not self._closed

    @property
    def local_addr(self) -> tuple[str, int]:
        return self._sock.getsockname()[:2]

    @property
    def remote_addr(self) -> tuple[str, int]:
        return self._remote


# ── STUN Protocol (RFC 5389) ─────────────────────────────────────────────────


class _StunProtocol:
    """Low-level STUN / TURN message builder and parser.

    All methods are static; the class serves purely as a namespace.
    """

    MAGIC_COOKIE = STUN_MAGIC_COOKIE

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def generate_transaction_id() -> bytes:
        """Return a random 12-byte transaction ID."""
        return os.urandom(12)

    @staticmethod
    def _xor_port(port: int) -> int:
        return port ^ (STUN_MAGIC_COOKIE >> 16)

    @staticmethod
    def _xor_addr(ip_bytes: bytes, transaction_id: bytes = b"") -> bytes:
        """XOR IPv4 (4 bytes) or IPv6 (16 bytes) with magic cookie [+ trans_id]."""
        magic = struct.pack(">I", STUN_MAGIC_COOKIE)
        if len(ip_bytes) == 4:
            return bytes(a ^ b for a, b in zip(ip_bytes, magic))
        # IPv6: XOR with (magic cookie + transaction_id[0:12])
        key = magic + transaction_id[:12]
        return bytes(a ^ b for a, b in zip(ip_bytes, key))

    @staticmethod
    def _pack_address(host: str, port: int) -> bytes:
        """Pack an IP:port into an 8-byte (IPv4) or 20-byte (IPv6) XOR-MAPPED-ADDRESS value."""
        try:
            ip_bytes = socket.inet_pton(socket.AF_INET, host)
            family = 0x01
        except OSError:
            ip_bytes = socket.inet_pton(socket.AF_INET6, host)
            family = 0x02
        xport = _StunProtocol._xor_port(port)
        xaddr = _StunProtocol._xor_addr(ip_bytes)
        return struct.pack("!BBH", 0, family, xport) + xaddr

    # ── Build STUN messages ──────────────────────────────────────────

    @staticmethod
    def build_stun_header(msg_type: int, length: int, trans_id: bytes) -> bytes:
        """Build the 20-byte STUN header."""
        assert len(trans_id) == 12
        return struct.pack(">HHI", msg_type, length, STUN_MAGIC_COOKIE) + trans_id

    @staticmethod
    def _add_attr(data: bytearray, attr_type: int, value: bytes) -> None:
        """Append a TLV attribute (padded to 4 bytes)."""
        data += struct.pack(">HH", attr_type, len(value))
        data += value
        # Pad to 4-byte boundary
        pad = (4 - (len(value) % 4)) % 4
        if pad:
            data += b"\x00" * pad

    @staticmethod
    def build_binding_request(
        trans_id: bytes | None = None,
        priority: int = 0,
        controlling: bool = True,
        ufrag: str = "",
        password: str = "",
    ) -> bytes:
        """Build a STUN Binding Request for ICE connectivity checks.

        Args:
            trans_id:  Transaction ID (auto-generated if ``None``).
            priority:  Candidate-pair priority for the PRIORITY attribute.
            controlling: If ``True``, include ICE-CONTROLLING; else ICE-CONTROLLED.
            ufrag:     Local ufrag for USERNAME attribute (only included if set).
            password:  Local password for MESSAGE-INTEGRITY (only if ufrag set).

        Returns:
            Raw STUN message bytes.
        """
        tid = trans_id or _StunProtocol.generate_transaction_id()
        body = bytearray()

        if priority:
            _StunProtocol._add_attr(body, ATTR_PRIORITY, struct.pack(">I", priority))

        tiebreaker = struct.pack(">Q", random.getrandbits(64))
        if controlling:
            _StunProtocol._add_attr(body, ATTR_ICE_CONTROLLING, tiebreaker)
        else:
            _StunProtocol._add_attr(body, ATTR_ICE_CONTROLLED, tiebreaker)

        if ufrag and password:
            _StunProtocol._add_attr(body, ATTR_USERNAME, ufrag.encode("utf-8"))
            integrity_data = _StunProtocol.build_stun_header(
                STUN_BINDING_REQUEST, len(body), tid
            ) + bytes(body)
            _StunProtocol._add_attr(body, ATTR_MESSAGE_INTEGRITY,
                                     _StunProtocol._compute_integrity(integrity_data, password.encode("utf-8")))

        header = _StunProtocol.build_stun_header(STUN_BINDING_REQUEST, len(body), tid)
        return header + bytes(body)

    @staticmethod
    def build_binding_response(trans_id: bytes, mapped_addr: tuple[str, int]) -> bytes:
        """Build a STUN Binding Response with XOR-MAPPED-ADDRESS."""
        value = _StunProtocol._pack_address(mapped_addr[0], mapped_addr[1])
        body = bytearray()
        _StunProtocol._add_attr(body, ATTR_XOR_MAPPED_ADDRESS, value)
        header = _StunProtocol.build_stun_header(STUN_BINDING_RESPONSE, len(body), trans_id)
        return header + bytes(body)

    @staticmethod
    def build_binding_error(trans_id: bytes, code: int = 400, reason: str = "Bad Request") -> bytes:
        """Build a STUN Binding Error Response."""
        body = bytearray()
        err_value = struct.pack("!BB", 0, code // 100) + struct.pack("!B", code % 100) + reason.encode("utf-8")
        _StunProtocol._add_attr(body, ATTR_ERROR_CODE, err_value)
        header = _StunProtocol.build_stun_header(STUN_BINDING_ERROR, len(body), trans_id)
        return header + bytes(body)

    # ── Parse STUN messages ───────────────────────────────────────────

    @staticmethod
    def parse_message(data: bytes) -> dict[str, Any]:
        """Parse a STUN or TURN message into a dict.

        Returns:
            ``{``
                ``"msg_type"``: int,
                ``"length"``: int,
                ``"magic_cookie"``: int,
                ``"trans_id"``: bytes,
                ``"attributes"``: ``{attr_type: bytes_value}``,
                ``"error_code"``: int | None,
                ``"error_reason"``: str | None,
            ``}``
        """
        if len(data) < 20:
            raise ICEError("STUN message too short")

        msg_type, msg_len, cookie = struct.unpack_from(">HHI", data, 0)
        if cookie != STUN_MAGIC_COOKIE:
            # Could be a ChannelData message — check first two bits
            if data[0] & 0xC0 == 0x40:  # 0b01xxxxxx
                raise ICEError("ChannelData message, not STUN")
            raise ICEError(f"Invalid STUN magic cookie: 0x{cookie:08X}")

        # STUN header: type(2) + length(2) + magic_cookie(4) + trans_id(12)
        trans_id = data[8:20]
        pos = 20
        attrs: dict[int, bytes] = {}
        error_code: int | None = None
        error_reason: str | None = None

        while pos < len(data):
            if pos + 4 > len(data):
                break
            attr_type, attr_len = struct.unpack_from(">HH", data, pos)
            pos += 4
            if pos + attr_len > len(data):
                break
            attr_value = data[pos: pos + attr_len]
            attrs[attr_type] = attr_value

            if attr_type == ATTR_ERROR_CODE and attr_len >= 4:
                _class = attr_value[1]
                _num = attr_value[2]
                error_code = _class * 100 + _num
                error_reason = attr_value[4:].decode("utf-8", errors="replace")

            pos += attr_len
            # Skip padding
            pos += (4 - (attr_len % 4)) % 4

        return {
            "msg_type": msg_type,
            "length": msg_len,
            "magic_cookie": cookie,
            "trans_id": trans_id,
            "attributes": attrs,
            "error_code": error_code,
            "error_reason": error_reason,
        }

    @staticmethod
    def parse_xor_mapped_address(attr_value: bytes, trans_id: bytes = b"") -> tuple[str, int]:
        """Extract (ip, port) from an XOR-MAPPED-ADDRESS / XOR-RELAYED-ADDRESS value."""
        if len(attr_value) < 4:
            raise ICEError("XOR-MAPPED-ADDRESS too short")
        _family = attr_value[1]
        port = struct.unpack_from(">H", attr_value, 2)[0]
        port ^= STUN_MAGIC_COOKIE >> 16

        if _family == 0x01:  # IPv4
            ip_bytes = attr_value[4:8]
            xored = _StunProtocol._xor_addr(ip_bytes)
            ip = socket.inet_ntop(socket.AF_INET, xored)
        elif _family == 0x02:  # IPv6
            ip_bytes = attr_value[4:20]
            key = struct.pack(">I", STUN_MAGIC_COOKIE) + trans_id[:12]
            ip = socket.inet_ntop(socket.AF_INET6,
                                   bytes(a ^ b for a, b in zip(ip_bytes, key)))
        else:
            raise ICEError(f"Unknown address family 0x{_family:02x}")

        return (ip, port)

    @staticmethod
    def parse_mapped_address(attr_value: bytes) -> tuple[str, int]:
        """Extract (ip, port) from a plain (non-XOR) MAPPED-ADDRESS."""
        if len(attr_value) < 8:
            raise ICEError("MAPPED-ADDRESS too short")
        family = attr_value[1]
        port = struct.unpack_from(">H", attr_value, 2)[0]
        if family == 0x01:
            ip = socket.inet_ntop(socket.AF_INET, attr_value[4:8])
        elif family == 0x02:
            ip = socket.inet_ntop(socket.AF_INET6, attr_value[4:20])
        else:
            raise ICEError(f"Unknown address family 0x{family:02x}")
        return (ip, port)

    @staticmethod
    def is_channel_data(data: bytes) -> bool:
        """Check whether *data* is a ChannelData message (RFC 5766 Section 11)."""
        return len(data) >= 4 and (data[0] & 0xC0) == 0x40

    # ── Integrity ─────────────────────────────────────────────────────

    @staticmethod
    def _compute_integrity(message: bytes, key: bytes) -> bytes:
        """HMAC-SHA1 for MESSAGE-INTEGRITY (20 bytes)."""
        return hmac.new(key, message, hashlib.sha1).digest()

    @staticmethod
    def _compute_fingerprint(message: bytes) -> bytes:
        """CRC-32 XOR 0x5354554E for FINGERPRINT (4 bytes)."""
        crc = zlib.crc32(message) ^ 0x5354554E
        return struct.pack(">I", crc & 0xFFFFFFFF)

    @staticmethod
    def add_fingerprint(msg: bytes) -> bytes:
        """Append FINGERPRINT attribute to a STUN message."""
        msg += struct.pack(">HH", ATTR_FINGERPRINT, 4)
        msg += _StunProtocol._compute_fingerprint(msg)
        return msg

    # ── TURN message builders ─────────────────────────────────────────

    @staticmethod
    def build_allocate_request(
        lifetime: int = 600,
        trans_id: bytes | None = None,
    ) -> bytes:
        """Build a TURN Allocate request (RFC 5766 Section 6.1)."""
        tid = trans_id or _StunProtocol.generate_transaction_id()
        body = bytearray()

        # REQUESTED-TRANSPORT: UDP
        _StunProtocol._add_attr(body, ATTR_REQUESTED_TRANSPORT,
                                 struct.pack("!B", TRANSPORT_UDP) + b"\x00" * 3)

        if lifetime != 600:
            _StunProtocol._add_attr(body, ATTR_LIFETIME, struct.pack(">I", lifetime))

        # DONT-FRAGMENT
        _StunProtocol._add_attr(body, ATTR_DONT_FRAGMENT, b"")

        header = _StunProtocol.build_stun_header(TURN_ALLOCATE_REQUEST, len(body), tid)
        return header + bytes(body)

    @staticmethod
    def build_refresh_request(
        lifetime: int = 600,
        trans_id: bytes | None = None,
    ) -> bytes:
        """Build a TURN Refresh request."""
        tid = trans_id or _StunProtocol.generate_transaction_id()
        body = bytearray()
        _StunProtocol._add_attr(body, ATTR_LIFETIME, struct.pack(">I", lifetime))
        header = _StunProtocol.build_stun_header(TURN_REFRESH_REQUEST, len(body), tid)
        return header + bytes(body)

    @staticmethod
    def build_create_permission_request(peer_addr: tuple[str, int], trans_id: bytes | None = None) -> bytes:
        """Build a TURN CreatePermission request."""
        tid = trans_id or _StunProtocol.generate_transaction_id()
        body = bytearray()
        value = _StunProtocol._pack_address(peer_addr[0], peer_addr[1])
        _StunProtocol._add_attr(body, ATTR_XOR_PEER_ADDRESS, value)
        header = _StunProtocol.build_stun_header(TURN_CREATE_PERMISSION_REQUEST, len(body), tid)
        return header + bytes(body)

    @staticmethod
    def build_channel_bind_request(channel: int, peer_addr: tuple[str, int], trans_id: bytes | None = None) -> bytes:
        """Build a TURN ChannelBind request."""
        tid = trans_id or _StunProtocol.generate_transaction_id()
        body = bytearray()
        _StunProtocol._add_attr(body, ATTR_CHANNEL_NUMBER, struct.pack(">H", channel) + b"\x00" * 2)
        value = _StunProtocol._pack_address(peer_addr[0], peer_addr[1])
        _StunProtocol._add_attr(body, ATTR_XOR_PEER_ADDRESS, value)
        header = _StunProtocol.build_stun_header(TURN_CHANNEL_BIND_REQUEST, len(body), tid)
        return header + bytes(body)

    @staticmethod
    def build_send_indication(peer_addr: tuple[str, int], data: bytes) -> bytes:
        """Build a TURN Send indication (RFC 5766 Section 10)."""
        tid = _StunProtocol.generate_transaction_id()
        body = bytearray()
        value = _StunProtocol._pack_address(peer_addr[0], peer_addr[1])
        _StunProtocol._add_attr(body, ATTR_XOR_PEER_ADDRESS, value)
        _StunProtocol._add_attr(body, ATTR_DATA, data)
        header = _StunProtocol.build_stun_header(TURN_SEND_INDICATION, len(body), tid)
        return header + bytes(body)

    @staticmethod
    def build_data_indication(peer_addr: tuple[str, int], data: bytes) -> bytes:
        """Build a TURN Data indication."""
        tid = _StunProtocol.generate_transaction_id()
        body = bytearray()
        value = _StunProtocol._pack_address(peer_addr[0], peer_addr[1])
        _StunProtocol._add_attr(body, ATTR_XOR_PEER_ADDRESS, value)
        _StunProtocol._add_attr(body, ATTR_DATA, data)
        header = _StunProtocol.build_stun_header(TURN_DATA_INDICATION, len(body), tid)
        return header + bytes(body)

    @staticmethod
    def build_channel_data(channel: int, data: bytes) -> bytes:
        """Build a ChannelData message (RFC 5766 Section 11).

        Channel numbers 0x4000 -- 0x7FFE.
        """
        assert CHANNEL_DATA_MIN <= channel <= CHANNEL_DATA_MAX
        assert len(data) <= CHANNEL_DATA_MAX_LEN
        return struct.pack(">HH", channel, len(data)) + data

    @staticmethod
    def parse_channel_data(data: bytes) -> tuple[int, bytes]:
        """Parse a ChannelData message, returning (channel, payload)."""
        if len(data) < 4:
            raise ICEError("ChannelData too short")
        channel, length = struct.unpack_from(">HH", data, 0)
        payload = data[4:4 + length]
        return (channel, payload)

    # ── TURN authentication helpers ───────────────────────────────────

    @staticmethod
    def compute_long_term_key(username: str, realm: str, password: str) -> bytes:
        """MD5(username:realm:password) for long-term credential mechanism."""
        return hashlib.md5(f"{username}:{realm}:{password}".encode("utf-8")).digest()

    @staticmethod
    def add_message_integrity(msg: bytes, key: bytes, *, include_fingerprint: bool = False) -> bytes:
        """Append MESSAGE-INTEGRITY (and optionally FINGERPRINT) to a STUN message.

        The message length field in the header is updated to exclude the
        integrity and fingerprint attributes per RFC 5389 Section 15.4.
        """
        # The length in the header must NOT include MESSAGE-INTEGRITY and FINGERPRINT
        # But we're building the message so we need to compute the correct length.
        # The current msg has the header with the body length up to where we'll add MI.
        body_len = len(msg) - 20  # Subtract the 20-byte header
        mi_attr_len = 4 + 20  # 4-byte header + 20-byte SHA-1
        total_attr_len = mi_attr_len
        if include_fingerprint:
            total_attr_len += 4 + 4  # 4-byte header + 4-byte CRC-32

        # Update length in header
        msg = msg[:2] + struct.pack(">H", body_len) + msg[4:]

        # Compute HMAC-SHA1 over the updated message
        mi_value = _StunProtocol._compute_integrity(msg, key)
        msg = msg + struct.pack(">HH", ATTR_MESSAGE_INTEGRITY, 20) + mi_value

        if include_fingerprint:
            msg = _StunProtocol.add_fingerprint(msg)

        return msg

    @staticmethod
    def authenticate_request(
        request: bytes,
        key: bytes,
    ) -> bool:
        """Verify MESSAGE-INTEGRITY on a STUN request.

        Args:
            request: The full STUN message.
            key:     Expected HMAC key (e.g., MD5(username:realm:password)).

        Returns:
            ``True`` if MESSAGE-INTEGRITY is present and valid.
        """
        parsed = _StunProtocol.parse_message(request)
        attrs = parsed["attributes"]
        if ATTR_MESSAGE_INTEGRITY not in attrs:
            return False

        # Find where MESSAGE-INTEGRITY attribute starts
        # Re-parse to find the byte offset
        msg_len = len(request)
        pos = 20
        mi_pos = -1
        while pos < msg_len:
            if pos + 4 > msg_len:
                break
            a_type, a_len = struct.unpack_from(">HH", request, pos)
            if a_type == ATTR_MESSAGE_INTEGRITY:
                mi_pos = pos
                break
            pos += 4 + a_len
            pos += (4 - (a_len % 4)) % 4

        if mi_pos < 0:
            return False

        # Message up to MI attribute (header lengths already correct)
        msg_up_to_mi = request[:mi_pos]
        # Set the length field to include attributes up to (but not including) MI
        body_len = mi_pos - 20
        msg_up_to_mi = msg_up_to_mi[:2] + struct.pack(">H", body_len) + msg_up_to_mi[4:]

        expected_mi = _StunProtocol._compute_integrity(msg_up_to_mi, key)
        return hmac.compare_digest(attrs[ATTR_MESSAGE_INTEGRITY], expected_mi)


# ── Candidate utility functions ──────────────────────────────────────────────


def candidate_priority(candidate: Candidate) -> int:
    """Compute candidate priority per RFC 8445 Section 5.1.2.1.

    ``priority = (2^24) * type_preference + (2^8) * local_preference + (256 - component_id)``
    """
    type_pref = CANDIDATE_TYPE_PREF.get(candidate.type, 0)
    local_pref = 65535  # Maximum local preference
    return (2 ** 24) * type_pref + (2 ** 8) * local_pref + (256 - candidate.component)


def pair_priority(local: Candidate, remote: Candidate, controlling: bool = True) -> int:
    """Compute pair priority for ordering connectivity checks.

    Uses the formula from RFC 8445 Section 6.1.2.2:
    ``pair_priority = 2^32 * min(G, D) + 2 * max(G, D) + (G > D ? 1 : 0)``
    where *G* is the controlling agent's priority and *D* the controlled agent's.

    If *controlling* is ``True``, *G* = local priority, else *G* = remote priority.
    """
    g = local.priority if controlling else remote.priority
    d = remote.priority if controlling else local.priority
    return (2 ** 32) * min(g, d) + 2 * max(g, d) + (1 if g > d else 0)


def _candidate_sort_key(c: Candidate) -> tuple[int, int]:
    """Sort candidates: highest priority first, then foundation."""
    return (-c.priority, c.foundation)


def _get_local_addresses() -> list[tuple[str, int]]:
    """Enumerate local IP addresses (IPv4 and IPv6) suitable for host candidates.

    Returns a list of (ip, family) where family is ``socket.AF_INET`` or ``socket.AF_INET6``.
    """
    addrs: list[tuple[str, int]] = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, family=socket.AF_UNSPEC,
                                        type=socket.SOCK_DGRAM):
            ip = info[4][0]
            family = info[0]
            if family in (socket.AF_INET, socket.AF_INET6):
                if not ip.startswith("127.") and ip != "::1":
                    addrs.append((ip, family))
    except OSError:
        pass

    # Fallback: guaranteed interfaces
    if not addrs:
        try:
            import subprocess
            result = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=3)
            for ip in result.stdout.strip().split():
                if ":" in ip:
                    addrs.append((ip, socket.AF_INET6))
                else:
                    addrs.append((ip, socket.AF_INET))
        except Exception:
            pass

    if not addrs:
        addrs.append(("0.0.0.0", socket.AF_INET))

    return addrs


# ── ICE Agent (RFC 8445) ─────────────────────────────────────────────────────


class ICEAgent:
    """Interactive Connectivity Establishment (RFC 8445) agent.

    Gathers local candidates (host, server-reflexive via STUN, relayed via TURN),
    pairs them with remote candidates, runs connectivity checks via STUN binding
    requests, and establishes a nominated candidate pair for data transport.

    Usage::

        agent = ICEAgent(role="controlling")
        agent.gather_candidates(stun_servers=[("stun.l.google.com", 19302)])
        agent.set_remote_candidates([
            Candidate(type="host", host="203.0.113.5", port=50051),
        ])
        pair = agent.connect(timeout=10.0)
        if pair:
            transport = agent.transport  # DataTransport for I/O

    Thread-safety: external callers must not call methods concurrently on the
    same agent instance.
    """

    MAX_CANDIDATE_PAIRS = 50
    CHECK_INTERVAL = 0.05  # seconds between select() polls
    MAX_RECV_SIZE = 65535

    def __init__(
        self,
        role: str = "controlling",
        stun_servers: list[tuple[str, int]] | None = None,
        turn_config: dict[str, Any] | None = None,
    ):
        if role not in ("controlling", "controlled"):
            raise ICEError(f"ICE role must be 'controlling' or 'controlled', got {role!r}")
        self._role = role
        self._stun_servers = stun_servers or DEFAULT_STUN_SERVERS
        self._turn_config = turn_config

        # Candidate lists
        self._local_candidates: list[Candidate] = []
        self._remote_candidates: list[Candidate] = []
        self._candidate_pairs: list[CandidatePair] = []
        self._valid_pairs: list[CandidatePair] = []
        self._nominated_pair: CandidatePair | None = None

        # Per-candidate sockets: candidate index -> socket
        self._sockets: list[socket.socket] = []
        self._owned_sockets: set[int] = set()  # indices of sockets we created

        # ICE ufrag / password (short-term credentials)
        self._local_ufrag = uuid.uuid4().hex[:8]
        self._local_password = uuid.uuid4().hex[:16]

        # Remote ufrag / password (set via set_remote_candidates)
        self._remote_ufrag: str = ""
        self._remote_password: str = ""

        # Connectivity check infrastructure
        self._lock = threading.Lock()
        self._pending_checks: dict[bytes, CandidatePair] = {}  # trans_id -> pair
        self._listener_threads: list[threading.Thread] = []
        self._running = threading.Event()
        self._transport: _UDPTransport | None = None
        self._connect_start: float = 0.0
        self._connect_rtt: float = 0.0

        # TURN relay transport (if using relayed candidates)
        self._turn_client: TURNClient | None = None
        self._relay_transport: DataTransport | None = None

    # ── Public API ─────────────────────────────────────────────────

    def gather_candidates(
        self,
        stun_servers: list[tuple[str, int]] | None = None,
        turn_config: dict[str, Any] | None = None,
    ) -> list[Candidate]:
        """Gather local host, server-reflexive, and relayed candidates.

        Args:
            stun_servers: STUN servers for server-reflexive candidates.
            turn_config:  TURN configuration for relayed candidates.

        Returns:
            Sorted list of local candidates (highest priority first).
        """
        stun_servers = stun_servers or self._stun_servers
        turn_config = turn_config or self._turn_config
        candidates: list[Candidate] = []
        sockets: list[socket.socket] = []

        # 1. Host candidates
        for ip, family in _get_local_addresses():
            try:
                sock = socket.socket(family, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((ip, 0))
                port = sock.getsockname()[1]
                cand = Candidate(
                    foundation=f"host-{ip}",
                    component=1,
                    transport="UDP",
                    host=ip,
                    port=port,
                    type="host",
                )
                # Set priority after creation
                object.__setattr__(cand, "priority", candidate_priority(cand))
                candidates.append(cand)
                sockets.append(sock)
                logger.debug(f"ICE host candidate: {ip}:{port}")
            except OSError as e:
                logger.debug(f"ICE: skipping {ip}: {e}")

        # 2. Server-reflexive candidates via STUN
        for stun_host, stun_port in stun_servers:
            for idx, host_cand in enumerate(
                [c for c in candidates if c.type == "host"]
            ):
                try:
                    sock = sockets[candidates.index(host_cand)]
                    public = self._stun_binding(
                        sock, (stun_host, stun_port), timeout=3.0
                    )
                    if public:
                        srflx = Candidate(
                            foundation=f"srflx-{public[0]}",
                            component=1,
                            transport="UDP",
                            host=public[0],
                            port=public[1],
                            type="srflx",
                            rel_addr=host_cand.host,
                            rel_port=host_cand.port,
                        )
                        object.__setattr__(srflx, "priority", candidate_priority(srflx))
                        candidates.append(srflx)
                        # Reuse host socket for srflx
                        sockets.append(sock)
                        logger.debug(f"ICE srflx candidate: {public[0]}:{public[1]}")
                        break  # One srflx per STUN server is enough
                except Exception as e:
                    logger.debug(f"ICE STUN from {stun_host}:{stun_port} failed: {e}")

        # 3. Relayed candidates via TURN
        if turn_config:
            try:
                tc = TURNClient(
                    server_host=turn_config.get("server", ""),
                    server_port=turn_config.get("port", TURN_DEFAULT_PORT),
                    username=turn_config.get("user", ""),
                    password=turn_config.get("password", ""),
                )
                if tc.allocate():
                    relayed = tc.relayed_address
                    if relayed:
                        cand = Candidate(
                            foundation=f"relay-{relayed[0]}",
                            component=1,
                            transport="UDP",
                            host=relayed[0],
                            port=relayed[1],
                            type="relay",
                        )
                        object.__setattr__(cand, "priority", candidate_priority(cand))
                        candidates.append(cand)
                        sockets.append(tc._sock)  # type: ignore[arg-type]
                        self._turn_client = tc
                        logger.debug(f"ICE relayed candidate: {relayed[0]}:{relayed[1]}")
            except Exception as e:
                logger.debug(f"ICE TURN allocate failed: {e}")

        self._local_candidates = sorted(candidates, key=_candidate_sort_key)
        self._sockets = sockets
        self._owned_sockets = set(range(len(sockets)))
        return list(self._local_candidates)

    def set_remote_candidates(
        self,
        candidates: list[Candidate],
        ufrag: str = "",
        password: str = "",
    ) -> list[CandidatePair]:
        """Set remote candidates and create ordered candidate pairs.

        Args:
            candidates: Remote peer's candidates.
            ufrag:      Remote ufrag for STUN USERNAME (optional).
            password:   Remote password for MESSAGE-INTEGRITY (optional).

        Returns:
            Candidate pairs sorted by priority (highest first).
        """
        self._remote_candidates = sorted(candidates, key=_candidate_sort_key)
        self._remote_ufrag = ufrag
        self._remote_password = password
        self._pair_candidates()
        return list(self._candidate_pairs)

    def connect(self, timeout: float = 10.0) -> CandidatePair | None:
        """Run ICE connectivity checks and nominate the best candidate pair.

        Steps:
        1. Start listener threads on all local sockets.
        2. Send STUN binding requests for each candidate pair (priority order).
        3. Handle incoming STUN requests (respond) and responses (match checks).
        4. After timeout / all checks resolved, select and nominate the best pair.
        5. Clean up listener threads (but keep the nominated socket open).

        Args:
            timeout: Maximum time in seconds for connectivity checks.

        Returns:
            The nominated :class:`CandidatePair`, or ``None`` if no check succeeded.

        Raises:
            ICEConnectionError: If no local or remote candidates are set.
        """
        if not self._local_candidates:
            raise ICEConnectionError("No local candidates; call gather_candidates() first")
        if not self._remote_candidates:
            raise ICEConnectionError("No remote candidates; call set_remote_candidates() first")

        self._connect_start = time.monotonic()
        self._running.set()

        # Start listener threads on each socket
        self._start_listeners()

        try:
            checks_ok = self._run_connectivity_checks(timeout)
            if checks_ok:
                self._select_nominated_pair()
        finally:
            self._running.clear()

        return self._nominated_pair

    @property
    def transport(self) -> DataTransport | None:
        """Return a :class:`DataTransport` for the nominated pair, or ``None``.

        Available after :meth:`connect` succeeds.
        """
        return self._transport

    def close(self) -> None:
        """Release all resources.  The agent cannot be reused after this call."""
        self._running.clear()
        if self._transport:
            self._transport.close()
            self._transport = None
        owned = set(self._owned_sockets)
        for idx, sock in enumerate(self._sockets):
            if idx in owned:
                try:
                    sock.close()
                except OSError:
                    pass
        self._sockets.clear()
        self._owned_sockets.clear()
        if self._turn_client:
            self._turn_client.close()
            self._turn_client = None

    @property
    def nominated_pair(self) -> CandidatePair | None:
        """The nominated candidate pair after successful connectivity checks."""
        return self._nominated_pair

    @property
    def local_candidates(self) -> list[Candidate]:
        return list(self._local_candidates)

    @property
    def remote_candidates(self) -> list[Candidate]:
        return list(self._remote_candidates)

    @property
    def candidate_pairs(self) -> list[CandidatePair]:
        return list(self._candidate_pairs)

    @property
    def stun_credentials(self) -> tuple[str, str]:
        """Return (local_ufrag, local_password) for signalling to the peer."""
        return (self._local_ufrag, self._local_password)

    # ── Internal: candidate gathering helpers ─────────────────────────

    def _stun_binding(
        self, sock: socket.socket, server: tuple[str, int], timeout: float = 3.0
    ) -> tuple[str, int] | None:
        """Send a STUN binding request and return the public (ip, port) the server sees."""
        trans_id = _StunProtocol.generate_transaction_id()
        req = _StunProtocol.build_binding_request(trans_id, priority=0, controlling=True)
        try:
            sock.sendto(req, server)
            sock.settimeout(timeout)
            data, _ = sock.recvfrom(self.MAX_RECV_SIZE)
            parsed = _StunProtocol.parse_message(data)
            if (parsed["msg_type"] == STUN_BINDING_RESPONSE
                    and parsed["trans_id"] == trans_id):
                attrs = parsed["attributes"]
                if ATTR_XOR_MAPPED_ADDRESS in attrs:
                    return _StunProtocol.parse_xor_mapped_address(
                        attrs[ATTR_XOR_MAPPED_ADDRESS], parsed["trans_id"]
                    )
                if ATTR_MAPPED_ADDRESS in attrs:
                    return _StunProtocol.parse_mapped_address(attrs[ATTR_MAPPED_ADDRESS])
        except (socket.timeout, OSError, ICEError):
            pass
        return None

    # ── Internal: candidate pairing ──────────────────────────────────

    def _pair_candidates(self) -> None:
        """Pair every local candidate with every remote candidate."""
        pairs: list[CandidatePair] = []
        for local in self._local_candidates:
            for remote in self._remote_candidates:
                if local.component != remote.component:
                    continue
                if local.transport != remote.transport:
                    continue
                prio = pair_priority(local, remote, controlling=(self._role == "controlling"))
                pairs.append(CandidatePair(
                    local=local,
                    remote=remote,
                    priority=prio,
                ))
        # Sort by priority descending
        pairs.sort(key=lambda p: -p.priority)
        self._candidate_pairs = pairs[:self.MAX_CANDIDATE_PAIRS]

    def _socket_for(self, candidate: Candidate) -> socket.socket | None:
        """Return the socket bound to *candidate*."""
        for idx, c in enumerate(self._local_candidates):
            if c is candidate or (c.host == candidate.host and c.port == candidate.port):
                if idx < len(self._sockets):
                    return self._sockets[idx]
        return None

    # ── Internal: connectivity checks ────────────────────────────────

    def _start_listeners(self) -> None:
        """Start a background thread per socket to process incoming STUN."""
        self._listener_threads.clear()
        for idx, sock in enumerate(self._sockets):
            if idx not in self._owned_sockets:
                continue
            t = threading.Thread(
                target=self._listener_loop,
                args=(idx,),
                daemon=True,
                name=f"ice-listener-{idx}",
            )
            t.start()
            self._listener_threads.append(t)

    def _listener_loop(self, sock_idx: int) -> None:
        """Listen for incoming STUN messages on a single socket."""
        sock = self._sockets[sock_idx]
        sock.settimeout(0.5)
        while self._running.is_set():
            try:
                data, addr = sock.recvfrom(self.MAX_RECV_SIZE)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                if _StunProtocol.is_channel_data(data):
                    continue  # ChannelData handled separately
                parsed = _StunProtocol.parse_message(data)
            except ICEError:
                continue

            msg_type = parsed["msg_type"]
            trans_id = parsed["trans_id"]

            if msg_type == STUN_BINDING_REQUEST:
                # Respond with a binding response so the peer knows we're reachable
                mapped = sock.getsockname()[:2]
                resp = _StunProtocol.build_binding_response(trans_id, mapped)
                try:
                    sock.sendto(resp, addr)
                except OSError:
                    pass

                # Triggered check: if this is from a remote candidate we haven't
                # checked yet, mark progress
                self._mark_remote_reachable(addr)

            elif msg_type == STUN_BINDING_RESPONSE:
                self._handle_check_response(trans_id, addr)

    def _mark_remote_reachable(self, addr: tuple[str, int]) -> None:
        """Mark candidate pairs matching *addr* as reachable."""
        with self._lock:
            for pair in self._candidate_pairs:
                if pair.remote.addr() == addr and pair.state == "frozen":
                    pair.state = "waiting"

    def _handle_check_response(self, trans_id: bytes, addr: tuple[str, int]) -> None:
        """Process a STUN binding response for a pending connectivity check."""
        with self._lock:
            pair = self._pending_checks.pop(trans_id, None)
            if pair is None:
                return
            if pair.state == "failed":
                return
            pair.state = "succeeded"
            pair.rtt = time.monotonic() - self._connect_start
            if pair not in self._valid_pairs:
                self._valid_pairs.append(pair)
            logger.debug(f"ICE check succeeded: {pair.local.addr()} -> {pair.remote.addr()} "
                         f"rtt={pair.rtt * 1000:.1f}ms")

    def _run_connectivity_checks(self, timeout: float) -> bool:
        """Send STUN binding requests for candidate pairs and collect responses."""
        end_time = time.monotonic() + timeout
        pairs_to_check = [p for p in self._candidate_pairs if p.state == "frozen"]

        if not pairs_to_check:
            return bool(self._valid_pairs)

        # Send checks in priority order, starting with the highest
        for pair in pairs_to_check:
            if time.monotonic() >= end_time:
                break

            sock = self._socket_for(pair.local)
            if sock is None:
                pair.state = "failed"
                continue

            trans_id = _StunProtocol.generate_transaction_id()
            req = _StunProtocol.build_binding_request(
                trans_id,
                priority=pair.priority,
                controlling=(self._role == "controlling"),
                ufrag=self._remote_ufrag,
                password=self._remote_password,
            )

            with self._lock:
                pair.state = "in_progress"
                self._pending_checks[trans_id] = pair

            try:
                sock.sendto(req, pair.remote.addr())
                logger.debug(f"ICE check sent: {pair.local.addr()} -> {pair.remote.addr()}")
            except OSError:
                with self._lock:
                    pair.state = "failed"
                    self._pending_checks.pop(trans_id, None)

        # Wait for responses until timeout
        while time.monotonic() < end_time:
            remaining = end_time - time.monotonic()
            if remaining <= 0:
                break

            with self._lock:
                if not self._pending_checks:
                    break

            time.sleep(min(self.CHECK_INTERVAL, remaining))

        # Mark any still-pending checks as failed
        with self._lock:
            for tid, pair in list(self._pending_checks.items()):
                if pair.state == "in_progress":
                    pair.state = "failed"
                del self._pending_checks[tid]

        return len(self._valid_pairs) > 0

    def _select_nominated_pair(self) -> None:
        """Select the best valid pair and nominate it for data transport.

        The controlling agent picks the highest-priority valid pair.
        The controlled agent uses the pair the controlling agent nominated.
        """
        if not self._valid_pairs:
            return

        # Sort valid pairs by priority descending
        self._valid_pairs.sort(key=lambda p: (-p.priority, p.rtt))

        best = self._valid_pairs[0]
        best.nominated = True
        self._nominated_pair = best

        # If controlling, send USE-CANDIDATE to confirm nomination
        if self._role == "controlling":
            sock = self._socket_for(best.local)
            if sock:
                trans_id = _StunProtocol.generate_transaction_id()
                req = _StunProtocol.build_binding_request(
                    trans_id,
                    priority=best.priority,
                    controlling=True,
                )
                # Append USE-CANDIDATE (empty attribute, length 0)
                req += struct.pack(">HH", ATTR_USE_CANDIDATE, 0)
                try:
                    sock.sendto(req, best.remote.addr())
                except OSError:
                    pass

        # Create transport
        sock = self._socket_for(best.local)
        if sock:
            self._transport = _UDPTransport(sock, best.remote.addr())
            self._connect_rtt = best.rtt
            logger.info(f"ICE nominated: {best.local.addr()} <-> {best.remote.addr()} "
                        f"rtt={best.rtt * 1000:.1f}ms")

    @property
    def connect_rtt(self) -> float:
        """Round-trip time in seconds for the negotiated connection."""
        return self._connect_rtt


def ice_connect(
    local_candidates: list[Candidate] | None = None,
    remote_candidates: list[Candidate] | None = None,
    stun_servers: list[tuple[str, int]] | None = None,
    turn_config: dict[str, Any] | None = None,
    role: str = "controlling",
    timeout: float = 10.0,
) -> DataTransport | None:
    """Convenience: create an :class:`ICEAgent`, run negotiation, return transport.

    Usage::

        transport = ice_connect(
            remote_candidates=[Candidate(type="host", host="1.2.3.4", port=50051)],
            stun_servers=[("stun.l.google.com", 19302)],
        )
        if transport:
            transport.send(b"hello")
    """
    agent = ICEAgent(role=role, stun_servers=stun_servers, turn_config=turn_config)
    try:
        agent.gather_candidates()
        if local_candidates:
            agent.set_remote_candidates(local_candidates)
        if remote_candidates:
            agent.set_remote_candidates(remote_candidates)
        pair = agent.connect(timeout=timeout)
        return agent.transport if pair else None
    finally:
        if agent.transport is None:
            agent.close()


# ── TURN Client (RFC 5766) ────────────────────────────────────────────────────


class TURNClient:
    """RFC 5766 TURN client for relayed communication.

    Manages a TURN allocation including:
    - Allocate / Refresh (with long-term credential authentication)
    - CreatePermission for peer addresses
    - ChannelBind for reduced overhead
    - Send indication / Data indication for data relay

    Usage::

        client = TURNClient("turn.example.com", 3478, "user", "pass")
        if client.allocate():
            print(f"Relayed address: {client.relayed_address}")
            client.create_permission("203.0.113.5", 50051)
            client.send_data(b"hello", ("203.0.113.5", 50051))
            data = client.recv_data(timeout=5.0)
        client.close()
    """

    _REALM_CACHE: str = ""
    _NONCE_CACHE: str = ""

    def __init__(
        self,
        server_host: str,
        server_port: int = TURN_DEFAULT_PORT,
        username: str = "",
        password: str = "",
        lifetime: int = 600,
    ):
        self._server = (server_host, server_port)
        self._username = username
        self._password = password
        self._lifetime = lifetime

        self._sock: socket.socket | None = None
        self._relayed_addr: tuple[str, int] | None = None
        self._allocation: dict[str, Any] = {}
        self._channel_map: dict[tuple[str, int], int] = {}  # peer -> channel number
        self._channel_reverse: dict[int, tuple[str, int]] = {}  # channel -> peer
        self._next_channel = CHANNEL_DATA_MIN
        self._lock = threading.Lock()
        self._recv_buffer: list[bytes] = []
        self._running = threading.Event()
        self._listener: threading.Thread | None = None
        self._permissions: set[tuple[str, int]] = set()

    # ── Public API ─────────────────────────────────────────────────

    def allocate(
        self,
        lifetime: int | None = None,
        timeout: float = 5.0,
    ) -> bool:
        """Allocate a relayed transport address from the TURN server.

        Implements the long-term credential mechanism (RFC 5389 Section 10.2):
        sends an unauthenticated Allocate, handles the 401 response, and retries
        with USERNAME, REALM, NONCE, and MESSAGE-INTEGRITY.

        Args:
            lifetime: Requested allocation lifetime in seconds (default 600).
            timeout:  Per-message timeout.

        Returns:
            ``True`` if allocation succeeded.
        """
        if lifetime is None:
            lifetime = self._lifetime

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        self._sock = sock

        # Step 1: Send unauthenticated Allocate request
        req = _StunProtocol.build_allocate_request(lifetime)
        try:
            sock.sendto(req, self._server)
            data, _ = sock.recvfrom(self._get_max_recv())
        except socket.timeout:
            logger.warning("TURN Allocate: timeout on initial request")
            return self._allocate_failed()
        except OSError as e:
            logger.warning(f"TURN Allocate: socket error: {e}")
            return self._allocate_failed()

        parsed = self._parse_and_log(data)
        if parsed is None:
            return self._allocate_failed()

        # Step 2: Handle 401 (Unauthorized) — extract realm and nonce
        if parsed["msg_type"] == TURN_ALLOCATE_ERROR and parsed.get("error_code") == 401:
            attrs = parsed["attributes"]
            realm = attrs.get(ATTR_REALM, b"").decode("utf-8", errors="replace")
            nonce = attrs.get(ATTR_NONCE, b"").decode("utf-8", errors="replace")
            if not realm or not nonce:
                logger.warning("TURN Allocate: 401 missing REALM or NONCE")
                return self._allocate_failed()
            self._cache_credentials(realm, nonce)

            # Retry with authentication
            return self._allocate_authenticated(lifetime, timeout)

        # Step 3: Handle successful response (unauthenticated — rare)
        if parsed["msg_type"] == TURN_ALLOCATE_RESPONSE:
            return self._process_allocate_response(parsed)

        # Unexpected response
        err = parsed.get("error_reason") or f"msg_type=0x{parsed['msg_type']:04x}"
        logger.warning(f"TURN Allocate: unexpected response: {err}")
        return self._allocate_failed()

    def refresh(self, lifetime: int | None = None, timeout: float = 5.0) -> bool:
        """Refresh the TURN allocation, extending its lifetime.

        Returns:
            ``True`` if the refresh was accepted.
        """
        if self._sock is None:
            return False
        lifetime = lifetime or self._lifetime
        req = _StunProtocol.build_refresh_request(lifetime)
        req = self._add_auth(req)
        try:
            self._sock.sendto(req, self._server)
            data, _ = self._sock.recvfrom(self._get_max_recv())
        except (socket.timeout, OSError):
            return False

        parsed = self._parse_and_log(data)
        if parsed and parsed["msg_type"] == TURN_REFRESH_RESPONSE:
            self._lifetime = lifetime
            return True
        return False

    def create_permission(self, peer_host: str, peer_port: int, timeout: float = 3.0) -> bool:
        """Create a permission for a peer address so the relay can forward data.

        Must be called for each peer before sending data to it.

        Returns:
            ``True`` if the permission was granted.
        """
        if self._sock is None:
            return False
        addr = (peer_host, peer_port)
        req = _StunProtocol.build_create_permission_request(addr)
        req = self._add_auth(req)
        try:
            self._sock.sendto(req, self._server)
            data, _ = self._sock.recvfrom(self._get_max_recv())
        except (socket.timeout, OSError):
            return False

        parsed = self._parse_and_log(data)
        if parsed and parsed["msg_type"] == TURN_CREATE_PERMISSION_RESPONSE:
            with self._lock:
                self._permissions.add(addr)
            logger.debug(f"TURN permission created for {peer_host}:{peer_port}")
            return True
        return False

    def channel_bind(self, peer_host: str, peer_port: int, timeout: float = 3.0) -> int:
        """Bind a channel number to a peer for reduced message overhead.

        Once bound, data can be sent using :meth:`send_channel_data` instead
        of Send indications, saving ~36 bytes per message.

        Returns:
            The channel number (0x4000-0x7FFE), or -1 on failure.
        """
        if self._sock is None:
            return -1
        addr = (peer_host, peer_port)

        # Return existing channel if already bound
        with self._lock:
            if addr in self._channel_map:
                return self._channel_map[addr]

        channel = self._next_channel
        self._next_channel += 1
        if self._next_channel > CHANNEL_DATA_MAX:
            self._next_channel = CHANNEL_DATA_MIN

        req = _StunProtocol.build_channel_bind_request(channel, addr)
        req = self._add_auth(req)
        try:
            self._sock.sendto(req, self._server)
            data, _ = self._sock.recvfrom(self._get_max_recv())
        except (socket.timeout, OSError):
            return -1

        parsed = self._parse_and_log(data)
        if parsed and parsed["msg_type"] == TURN_CHANNEL_BIND_RESPONSE:
            with self._lock:
                self._channel_map[addr] = channel
                self._channel_reverse[channel] = addr
            logger.debug(f"TURN channel {channel} bound to {peer_host}:{peer_port}")
            return channel
        return -1

    def send_data(self, data: bytes, peer_addr: tuple[str, int]) -> bool:
        """Send data to a peer through the TURN relay.

        Uses a Send indication (or ChannelData if a channel is bound).

        Returns:
            ``True`` if sent successfully.
        """
        if self._sock is None:
            return False
        try:
            with self._lock:
                channel = self._channel_map.get(peer_addr)

            if channel is not None:
                msg = _StunProtocol.build_channel_data(channel, data)
            else:
                msg = _StunProtocol.build_send_indication(peer_addr, data)
            self._sock.sendto(msg, self._server)
            return True
        except OSError:
            return False

    def send_channel_data(self, data: bytes, channel: int) -> bool:
        """Send data over a previously bound channel.

        More efficient than :meth:`send_data` because it skips STUN framing.

        Returns:
            ``True`` if sent successfully.
        """
        if self._sock is None:
            return False
        try:
            msg = _StunProtocol.build_channel_data(channel, data)
            self._sock.sendto(msg, self._server)
            return True
        except OSError:
            return False

    def recv_data(self, timeout: float | None = None) -> tuple[bytes, tuple[str, int]] | None:
        """Receive data from the TURN relay.

        Handles both Data indications and ChannelData messages.

        Returns:
            ``(payload, peer_address)`` or ``None`` on timeout.
        """
        if self._sock is None:
            return None

        # Check buffer first
        with self._lock:
            if self._recv_buffer:
                return self._recv_buffer.pop(0)  # type: ignore[return-value]

        try:
            self._sock.settimeout(timeout)
            raw, _ = self._sock.recvfrom(self._get_max_recv())
        except (socket.timeout, OSError):
            return None

        return self._process_incoming(raw)

    def _process_incoming(self, raw: bytes) -> tuple[bytes, tuple[str, int]] | None:
        """Parse an incoming message (Data indication or ChannelData)."""
        try:
            if _StunProtocol.is_channel_data(raw):
                channel, payload = _StunProtocol.parse_channel_data(raw)
                with self._lock:
                    peer = self._channel_reverse.get(channel)
                if peer is None:
                    logger.debug(f"TURN: data on unknown channel 0x{channel:04x}")
                    return None
                logger.debug(f"TURN: channel data from {peer} (ch=0x{channel:04x})")
                return (payload, peer)

            parsed = _StunProtocol.parse_message(raw)
            if parsed["msg_type"] != TURN_DATA_INDICATION:
                return None

            attrs = parsed["attributes"]
            if ATTR_XOR_PEER_ADDRESS not in attrs or ATTR_DATA not in attrs:
                return None

            peer = _StunProtocol.parse_xor_mapped_address(
                attrs[ATTR_XOR_PEER_ADDRESS], parsed["trans_id"]
            )
            payload = attrs[ATTR_DATA]
            logger.debug(f"TURN: data indication from {peer}")
            return (payload, peer)

        except ICEError:
            return None

    def start_listener(self) -> None:
        """Start a background thread that buffers incoming data.

        Use :meth:`recv_data` to consume from the buffer.  Automatically
        started by :meth:`allocate`.
        """
        if self._listener is not None:
            return
        self._running.set()
        self._listener = threading.Thread(
            target=self._listener_loop,
            daemon=True,
            name="turn-client-listener",
        )
        self._listener.start()

    def _listener_loop(self) -> None:
        """Background receive loop that buffers data."""
        while self._running.is_set() and self._sock is not None:
            try:
                self._sock.settimeout(0.5)
                raw, _ = self._sock.recvfrom(self._get_max_recv())
                result = self._process_incoming(raw)
                if result:
                    with self._lock:
                        self._recv_buffer.append(result)  # type: ignore[arg-type]
            except socket.timeout:
                continue
            except OSError:
                break

    def close(self) -> None:
        """Release the TURN allocation and close the socket."""
        self._running.clear()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    @property
    def relayed_address(self) -> tuple[str, int] | None:
        """The relayed transport address allocated by the TURN server."""
        return self._relayed_addr

    @property
    def is_allocated(self) -> bool:
        """``True`` if a TURN allocation exists."""
        return self._relayed_addr is not None

    # ── Internal ────────────────────────────────────────────────────

    def _get_max_recv(self) -> int:
        return 65535

    def _allocate_failed(self) -> bool:
        self.close()
        return False

    def _parse_and_log(self, data: bytes) -> dict[str, Any] | None:
        try:
            return _StunProtocol.parse_message(data)
        except ICEError as e:
            logger.debug(f"TURN: parse error: {e}")
            return None

    def _cache_credentials(self, realm: str, nonce: str) -> None:
        TURNClient._REALM_CACHE = realm
        TURNClient._NONCE_CACHE = nonce

    def _allocate_authenticated(self, lifetime: int, timeout: float) -> bool:
        """Send Allocate request with long-term credential authentication."""
        req = _StunProtocol.build_allocate_request(lifetime)
        req = self._add_auth(req)
        try:
            self._sock.sendto(req, self._server)
            data, _ = self._sock.recvfrom(self._get_max_recv())
        except (socket.timeout, OSError):
            return self._allocate_failed()

        parsed = self._parse_and_log(data)
        if parsed is None:
            return self._allocate_failed()

        if parsed["msg_type"] == TURN_ALLOCATE_RESPONSE:
            ok = self._process_allocate_response(parsed)
            if ok:
                self.start_listener()
            return ok

        err = parsed.get("error_reason") or f"code={parsed.get('error_code')}"
        logger.warning(f"TURN Allocate (authenticated) failed: {err}")
        return self._allocate_failed()

    def _process_allocate_response(self, parsed: dict[str, Any]) -> bool:
        """Extract the relayed address from an Allocate success response."""
        attrs = parsed["attributes"]
        if ATTR_XOR_RELAYED_ADDRESS in attrs:
            try:
                self._relayed_addr = _StunProtocol.parse_xor_mapped_address(
                    attrs[ATTR_XOR_RELAYED_ADDRESS], parsed["trans_id"]
                )
                self._allocation = {
                    "trans_id": parsed["trans_id"],
                    "lifetime": self._lifetime,
                }
                logger.info(f"TURN allocated {self._relayed_addr[0]}:{self._relayed_addr[1]}")
                return True
            except ICEError as e:
                logger.warning(f"TURN: failed to parse relayed address: {e}")
        else:
            logger.warning("TURN Allocate: no XOR-RELAYED-ADDRESS in response")
        return False

    def _add_auth(self, msg: bytes) -> bytes:
        """Add USERNAME, REALM, NONCE, and MESSAGE-INTEGRITY to a STUN/TURN message."""
        realm = TURNClient._REALM_CACHE
        nonce = TURNClient._NONCE_CACHE
        key = _StunProtocol.compute_long_term_key(self._username, realm, self._password)

        # Build the message with USERNAME, REALM, NONCE attributes
        tid = msg[4:16]
        body = bytearray(msg[20:])  # Existing body after header

        # Prepend authentication attributes to the body
        auth_body = bytearray()
        _StunProtocol._add_attr(auth_body, ATTR_USERNAME, self._username.encode("utf-8"))
        _StunProtocol._add_attr(auth_body, ATTR_REALM, realm.encode("utf-8"))
        _StunProtocol._add_attr(auth_body, ATTR_NONCE, nonce.encode("utf-8"))

        # Combine existing body after auth attributes
        final_body = bytes(auth_body) + bytes(body)

        # Rebuild header with correct length (excluding MI and FINGERPRINT)
        mi_len = len(final_body)
        header = _StunProtocol.build_stun_header(
            struct.unpack_from(">H", msg, 0)[0], mi_len, tid
        )

        # Build message up to (but not including) MESSAGE-INTEGRITY
        pre_mi = header + final_body

        # Add MESSAGE-INTEGRITY
        mi_value = _StunProtocol._compute_integrity(pre_mi, key)
        pre_mi += struct.pack(">HH", ATTR_MESSAGE_INTEGRITY, 20)
        pre_mi += mi_value

        # Add FINGERPRINT
        pre_mi = _StunProtocol.add_fingerprint(pre_mi)

        return pre_mi


# ── TURN Server (RFC 5766 — in memory) ────────────────────────────────────────


@dataclass
class TurnAllocation:
    """A TURN allocation (RFC 5766 Section 6).

    Attributes:
        five_tuple:     ``(client_ip, client_port, server_ip, server_port, transport)``.
        relayed_addr:   The UDP address the server allocates for relay.
        lifetime:       Allocation lifetime in seconds.
        created_at:     ``time.monotonic()`` timestamp.
        transport_proto: Transport protocol code (17 = UDP).
    """
    five_tuple: tuple
    relayed_addr: tuple[str, int]
    lifetime: int = 600
    created_at: float = 0.0
    transport_proto: int = TRANSPORT_UDP
    permissions: set[tuple[str, int]] = field(default_factory=set)
    channel_bindings: dict[int, tuple[str, int]] = field(default_factory=dict)
    channel_reverse: dict[tuple[str, int], int] = field(default_factory=dict)
    username: str = ""
    realm: str = ""


@dataclass
class TurnPeerData:
    """Queued data to be delivered to a TURN client."""
    data: bytes
    peer_addr: tuple[str, int]


class TURNServer:
    """In-memory TURN server (RFC 5766) over UDP.

    Supports:
    - Allocate / Refresh with long-term credential authentication
    - CreatePermission for peer addresses
    - ChannelBind for reduced overhead
    - Send / Data indications
    - ChannelData messages

    Usage::

        server = TURNServer(port=3478, realm="example.com", users={"user": "pass"})
        server.start()  # blocks, runs in a thread
        # ...
        server.stop()
    """

    MAX_ALLOCATIONS = 5000
    MAX_PERMISSIONS_PER_ALLOC = 100
    DEFAULT_LIFETIME = 600
    MAX_LIFETIME = 3600

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = TURN_DEFAULT_PORT,
        realm: str = "turn.example.com",
        users: dict[str, str] | None = None,
        secret_key: str | None = None,
    ):
        """
        Args:
            host:       Bind address.
            port:       Bind port.
            realm:      TURN realm for long-term credentials.
            users:      ``{username: password}`` authentication table.
            secret_key: Optional HMAC key for generating nonces.
        """
        self._host = host
        self._port = port
        self._realm = realm
        self._users = users or {}
        self._secret_key = secret_key or os.urandom(32).hex()

        self._sock: socket.socket | None = None
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

        self._lock = threading.Lock()
        self._allocations: dict[tuple[str, int], TurnAllocation] = {}  # client addr -> alloc
        self._relayed_to_client: dict[tuple[str, int], tuple[str, int]] = {}  # relayed addr -> client
        self._used_ports: set[int] = set()

        # Nonce tracking
        self._nonces: dict[str, float] = {}  # nonce -> timestamp

        # Maximum port for relayed address allocation
        self._relay_port_range: tuple[int, int] = (49152, 65535)
        self._relay_host = host

        # Error counters
        self._stats: dict[str, int] = {
            "allocations_created": 0,
            "allocations_failed": 0,
            "messages_relayed": 0,
            "errors": 0,
        }

    # ── Public API ─────────────────────────────────────────────────

    def start(self) -> None:
        """Start the TURN server in a background thread.

        Raises:
            RuntimeError: If the server is already running.
        """
        if self._running.is_set():
            raise RuntimeError("TURNServer is already running")

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self._host, self._port))
        self._sock.settimeout(1.0)
        self._running.set()

        self._thread = threading.Thread(
            target=self._server_loop,
            daemon=True,
            name="turn-server",
        )
        self._thread.start()
        logger.info(f"TURN server listening on {self._host}:{self._port}, realm={self._realm}")

    def stop(self) -> None:
        """Stop the TURN server and release all allocations."""
        self._running.clear()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread and self._thread != threading.current_thread():
            self._thread.join(timeout=5.0)
        with self._lock:
            self._allocations.clear()
            self._relayed_to_client.clear()
            self._used_ports.clear()
        logger.info("TURN server stopped")

    def wait_until_running(self, timeout: float = 5.0) -> bool:
        """Block until the server is running, or timeout."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self._running.is_set() and self._sock is not None:
                return True
            time.sleep(0.1)
        return False

    @property
    def stats(self) -> dict[str, Any]:
        """Return operational statistics."""
        with self._lock:
            return {
                **self._stats,
                "active_allocations": len(self._allocations),
                "relayed_addresses": len(self._relayed_to_client),
            }

    # ── Internal: server loop ───────────────────────────────────────

    def _server_loop(self) -> None:
        """Main server receive-and-dispatch loop."""
        while self._running.is_set():
            try:
                data, addr = self._sock.recvfrom(65535)
                self._handle_message(data, addr)
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception:
                with self._lock:
                    self._stats["errors"] += 1

    def _handle_message(self, data: bytes, addr: tuple[str, int]) -> None:
        """Route an incoming message to the appropriate handler."""
        try:
            # ChannelData messages
            if _StunProtocol.is_channel_data(data):
                self._handle_channel_data(data, addr)
                return

            parsed = _StunProtocol.parse_message(data)
        except ICEError:
            return

        msg_type = parsed["msg_type"]
        trans_id = parsed["trans_id"]

        if msg_type == TURN_ALLOCATE_REQUEST:
            self._handle_allocate(data, addr, parsed)
        elif msg_type == TURN_REFRESH_REQUEST:
            self._handle_refresh(data, addr, parsed)
        elif msg_type == TURN_CREATE_PERMISSION_REQUEST:
            self._handle_create_permission(data, addr, parsed)
        elif msg_type == TURN_CHANNEL_BIND_REQUEST:
            self._handle_channel_bind(data, addr, parsed)
        elif msg_type == TURN_SEND_INDICATION:
            self._handle_send_indication(data, addr, parsed)
        elif msg_type == STUN_BINDING_REQUEST:
            self._handle_stun_binding(data, addr, parsed)
        else:
            # Unknown / unsupported
            resp = self._build_error_response(
                trans_id, 420, "Unknown Attribute",
                extra_attrs={ATTR_UNKNOWN_ATTRIBUTES: struct.pack(">H", msg_type)},
            )
            self._sock.sendto(resp, addr)

    # ── Internal: Allocate ──────────────────────────────────────────

    def _handle_allocate(
        self, data: bytes, addr: tuple[str, int], parsed: dict[str, Any]
    ) -> None:
        """Handle an Allocate request (RFC 5766 Section 6.2)."""
        trans_id = parsed["trans_id"]
        attrs = parsed["attributes"]

        # Check for authentication
        has_auth = ATTR_MESSAGE_INTEGRITY in attrs
        if not has_auth:
            # 401 Unauthorized — challenge with REALM and NONCE
            nonce = self._generate_nonce(addr)
            resp = _StunProtocol.build_stun_header(TURN_ALLOCATE_ERROR, 0, trans_id)
            body = bytearray()

            # ERROR-CODE: 401 Unauthorized
            _StunProtocol._add_attr(body, ATTR_ERROR_CODE,
                                     b"\x00\x04\x01\x19" + b"Unauthorized")

            _StunProtocol._add_attr(body, ATTR_REALM, self._realm.encode("utf-8"))
            _StunProtocol._add_attr(body, ATTR_NONCE, nonce.encode("utf-8"))

            resp = _StunProtocol.build_stun_header(TURN_ALLOCATE_ERROR, len(body), trans_id)
            resp = resp + bytes(body)
            self._sock.sendto(resp, addr)
            return

        # Verify authentication
        user = attrs.get(ATTR_USERNAME, b"").decode("utf-8", errors="replace")
        realm = attrs.get(ATTR_REALM, b"").decode("utf-8", errors="replace")
        nonce = attrs.get(ATTR_NONCE, b"").decode("utf-8", errors="replace")

        if not self._verify_auth(user, realm, nonce, data):
            resp = self._build_error_response(trans_id, 401, "Unauthorized")
            self._sock.sendto(resp, addr)
            return

        # Check for REQUESTED-TRANSPORT
        if ATTR_REQUESTED_TRANSPORT not in attrs:
            resp = self._build_error_response(trans_id, 400, "Bad Request")
            self._sock.sendto(resp, addr)
            return

        transport_value = attrs[ATTR_REQUESTED_TRANSPORT]
        if len(transport_value) < 1 or transport_value[0] != TRANSPORT_UDP:
            resp = self._build_error_response(trans_id, 442, "Unsupported Transport Protocol")
            self._sock.sendto(resp, addr)
            return

        # Check for existing allocation
        with self._lock:
            if addr in self._allocations:
                resp = self._build_error_response(trans_id, 437, "Allocation Mismatch")
                self._sock.sendto(resp, addr)
                return

            if len(self._allocations) >= self.MAX_ALLOCATIONS:
                resp = self._build_error_response(trans_id, 486, "Allocation Quota Reached")
                self._sock.sendto(resp, addr)
                return

            # Allocate a relayed address
            relayed_addr = self._allocate_relayed_addr()
            if relayed_addr is None:
                resp = self._build_error_response(trans_id, 508, "Insufficient Capacity")
                self._sock.sendto(resp, addr)
                return

            # Get lifetime
            lifetime = self.DEFAULT_LIFETIME
            if ATTR_LIFETIME in attrs and len(attrs[ATTR_LIFETIME]) >= 4:
                requested = struct.unpack(">I", attrs[ATTR_LIFETIME])[0]
                lifetime = min(requested, self.MAX_LIFETIME)

            # Create allocation
            five_tuple = (addr[0], addr[1], self._host, self._port, "UDP")
            alloc = TurnAllocation(
                five_tuple=five_tuple,
                relayed_addr=relayed_addr,
                lifetime=lifetime,
                created_at=time.monotonic(),
                transport_proto=TRANSPORT_UDP,
                username=user,
                realm=realm,
            )
            self._allocations[addr] = alloc
            self._relayed_to_client[relayed_addr] = addr
            self._stats["allocations_created"] += 1

        # Build success response
        body = bytearray()
        # XOR-RELAYED-ADDRESS
        relayed_value = _StunProtocol._pack_address(relayed_addr[0], relayed_addr[1])
        _StunProtocol._add_attr(body, ATTR_XOR_RELAYED_ADDRESS, relayed_value)

        # LIFETIME
        _StunProtocol._add_attr(body, ATTR_LIFETIME, struct.pack(">I", lifetime))

        # XOR-MAPPED-ADDRESS (client's public address as seen by server)
        mapped_value = _StunProtocol._pack_address(addr[0], addr[1])
        _StunProtocol._add_attr(body, ATTR_XOR_MAPPED_ADDRESS, mapped_value)

        resp = _StunProtocol.build_stun_header(TURN_ALLOCATE_RESPONSE, len(body), trans_id)
        resp = resp + bytes(body)
        key = _StunProtocol.compute_long_term_key(user, realm,
                                                    self._users.get(user, ""))
        resp = _StunProtocol.add_message_integrity(resp, key, include_fingerprint=True)
        self._sock.sendto(resp, addr)
        logger.info(f"TURN allocation: {addr} -> relayed {relayed_addr[0]}:{relayed_addr[1]} "
                     f"lifetime={lifetime}s")

    # ── Internal: Refresh ──────────────────────────────────────────

    def _handle_refresh(
        self, data: bytes, addr: tuple[str, int], parsed: dict[str, Any]
    ) -> None:
        """Handle a Refresh request."""
        trans_id = parsed["trans_id"]
        with self._lock:
            alloc = self._allocations.get(addr)
            if alloc is None:
                resp = self._build_error_response(trans_id, 437, "Allocation Mismatch")
                self._sock.sendto(resp, addr)
                return

            # Extract lifetime
            lifetime = self.DEFAULT_LIFETIME
            attrs = parsed["attributes"]
            if ATTR_LIFETIME in attrs and len(attrs[ATTR_LIFETIME]) >= 4:
                requested = struct.unpack(">I", attrs[ATTR_LIFETIME])[0]
                if requested == 0:
                    # Delete allocation
                    self._delete_allocation(addr)
                    resp = _StunProtocol.build_stun_header(
                        TURN_REFRESH_RESPONSE, 0, trans_id
                    )
                    self._sock.sendto(resp, addr)
                    return
                lifetime = min(requested, self.MAX_LIFETIME)

            alloc.lifetime = lifetime
            alloc.created_at = time.monotonic()

        body = bytearray()
        _StunProtocol._add_attr(body, ATTR_LIFETIME, struct.pack(">I", lifetime))
        resp = _StunProtocol.build_stun_header(TURN_REFRESH_RESPONSE, len(body), trans_id)
        resp = resp + bytes(body)
        key = _StunProtocol.compute_long_term_key(
            alloc.username, alloc.realm, self._users.get(alloc.username, "")
        )
        resp = _StunProtocol.add_message_integrity(resp, key, include_fingerprint=True)
        self._sock.sendto(resp, addr)

    # ── Internal: CreatePermission ─────────────────────────────────

    def _handle_create_permission(
        self, data: bytes, addr: tuple[str, int], parsed: dict[str, Any]
    ) -> None:
        """Handle a CreatePermission request."""
        trans_id = parsed["trans_id"]
        attrs = parsed["attributes"]
        with self._lock:
            alloc = self._allocations.get(addr)
            if alloc is None:
                resp = self._build_error_response(trans_id, 437, "Allocation Mismatch")
                self._sock.sendto(resp, addr)
                return

            if ATTR_XOR_PEER_ADDRESS not in attrs:
                resp = self._build_error_response(trans_id, 400, "Bad Request")
                self._sock.sendto(resp, addr)
                return

            try:
                peer = _StunProtocol.parse_xor_mapped_address(
                    attrs[ATTR_XOR_PEER_ADDRESS], parsed["trans_id"]
                )
            except ICEError:
                resp = self._build_error_response(trans_id, 400, "Bad Request")
                self._sock.sendto(resp, addr)
                return

            if len(alloc.permissions) >= self.MAX_PERMISSIONS_PER_ALLOC:
                resp = self._build_error_response(trans_id, 486, "Allocation Quota Reached")
                self._sock.sendto(resp, addr)
                return

            alloc.permissions.add(peer)

        resp = _StunProtocol.build_stun_header(TURN_CREATE_PERMISSION_RESPONSE, 0, trans_id)
        key = _StunProtocol.compute_long_term_key(
            alloc.username, alloc.realm, self._users.get(alloc.username, "")
        )
        resp = _StunProtocol.add_message_integrity(resp, key, include_fingerprint=True)
        self._sock.sendto(resp, addr)
        logger.debug(f"TURN permission: {addr} -> peer {peer[0]}:{peer[1]}")

    # ── Internal: ChannelBind ──────────────────────────────────────

    def _handle_channel_bind(
        self, data: bytes, addr: tuple[str, int], parsed: dict[str, Any]
    ) -> None:
        """Handle a ChannelBind request."""
        trans_id = parsed["trans_id"]
        attrs = parsed["attributes"]
        with self._lock:
            alloc = self._allocations.get(addr)
            if alloc is None:
                resp = self._build_error_response(trans_id, 437, "Allocation Mismatch")
                self._sock.sendto(resp, addr)
                return

            if ATTR_CHANNEL_NUMBER not in attrs or ATTR_XOR_PEER_ADDRESS not in attrs:
                resp = self._build_error_response(trans_id, 400, "Bad Request")
                self._sock.sendto(resp, addr)
                return

            channel = struct.unpack(">H", attrs[ATTR_CHANNEL_NUMBER][:2])[0]
            if channel < CHANNEL_DATA_MIN or channel > CHANNEL_DATA_MAX:
                resp = self._build_error_response(trans_id, 400, "Bad Request")
                self._sock.sendto(resp, addr)
                return

            try:
                peer = _StunProtocol.parse_xor_mapped_address(
                    attrs[ATTR_XOR_PEER_ADDRESS], parsed["trans_id"]
                )
            except ICEError:
                resp = self._build_error_response(trans_id, 400, "Bad Request")
                self._sock.sendto(resp, addr)
                return

            # Remove old binding if this channel is already bound to a different peer
            existing_peer = alloc.channel_bindings.get(channel)
            if existing_peer is not None and existing_peer != peer:
                alloc.channel_reverse.pop(existing_peer, None)

            alloc.channel_bindings[channel] = peer
            alloc.channel_reverse[peer] = channel

        resp = _StunProtocol.build_stun_header(TURN_CHANNEL_BIND_RESPONSE, 0, trans_id)
        key = _StunProtocol.compute_long_term_key(
            alloc.username, alloc.realm, self._users.get(alloc.username, "")
        )
        resp = _StunProtocol.add_message_integrity(resp, key, include_fingerprint=True)
        self._sock.sendto(resp, addr)
        logger.debug(f"TURN channel bind: {addr} ch=0x{channel:04x} -> {peer[0]}:{peer[1]}")

    # ── Internal: Send / Data Indications ──────────────────────────

    def _handle_send_indication(
        self, data: bytes, addr: tuple[str, int], parsed: dict[str, Any]
    ) -> None:
        """Handle a Send indication and relay the data to the peer."""
        attrs = parsed["attributes"]
        with self._lock:
            alloc = self._allocations.get(addr)
            if alloc is None:
                return  # Silently discard per RFC 5766 Section 10

            if ATTR_XOR_PEER_ADDRESS not in attrs or ATTR_DATA not in attrs:
                return

            try:
                peer = _StunProtocol.parse_xor_mapped_address(
                    attrs[ATTR_XOR_PEER_ADDRESS], parsed["trans_id"]
                )
            except ICEError:
                return

            payload = attrs[ATTR_DATA]

            # Check permission
            if peer not in alloc.permissions:
                # Permission not explicitly created; auto-create for this peer
                # May be rejected depending on policy; we'll create it silently
                if len(alloc.permissions) < self.MAX_PERMISSIONS_PER_ALLOC:
                    alloc.permissions.add(peer)

        # Relay the data directly to the peer
        try:
            self._sock.sendto(payload, peer)
            with self._lock:
                self._stats["messages_relayed"] += 1
        except OSError:
            pass

    def _handle_channel_data(self, data: bytes, addr: tuple[str, int]) -> None:
        """Handle a ChannelData message (RFC 5766 Section 11)."""
        try:
            channel, payload = _StunProtocol.parse_channel_data(data)
        except ICEError:
            return

        with self._lock:
            alloc = self._allocations.get(addr)
            if alloc is None:
                return
            peer = alloc.channel_bindings.get(channel)
            if peer is None:
                return

        # Relay the payload directly to the peer
        try:
            self._sock.sendto(payload, peer)
            with self._lock:
                self._stats["messages_relayed"] += 1
        except OSError:
            pass

    def _handle_stun_binding(
        self, data: bytes, addr: tuple[str, int], parsed: dict[str, Any]
    ) -> None:
        """Handle a STUN Binding Request (for NAT detection or ICE checks)."""
        trans_id = parsed["trans_id"]
        resp = _StunProtocol.build_binding_response(trans_id, (addr[0], addr[1]))
        self._sock.sendto(resp, addr)

    # ── Internal: Relayed address pool ─────────────────────────────

    def _allocate_relayed_addr(self) -> tuple[str, int] | None:
        """Allocate a (host, port) for the relayed transport address."""
        if self._relay_host in ("0.0.0.0", "::"):
            host = self._host if self._host not in ("0.0.0.0", "::") else "127.0.0.1"
        else:
            host = self._relay_host

        min_port, max_port = self._relay_port_range
        for _ in range(100):  # Try up to 100 times
            port = random.randint(min_port, max_port)
            if port not in self._used_ports:
                self._used_ports.add(port)
                return (host, port)
        return None

    def _delete_allocation(self, client_addr: tuple[str, int]) -> None:
        """Remove an allocation and free its relayed address."""
        alloc = self._allocations.pop(client_addr, None)
        if alloc:
            self._relayed_to_client.pop(alloc.relayed_addr, None)
            self._used_ports.discard(alloc.relayed_addr[1])

    # ── Internal: Authentication ───────────────────────────────────

    def _generate_nonce(self, addr: tuple[str, int]) -> str:
        """Generate a unique nonce for the long-term credential mechanism."""
        now = time.time()
        data = f"{addr[0]}:{addr[1]}:{now}:{self._secret_key}"
        nonce = hashlib.md5(data.encode()).hexdigest()
        with self._lock:
            self._nonces[nonce] = now
            # Clean old nonces
            cutoff = now - 3600
            for n, t in list(self._nonces.items()):
                if t < cutoff:
                    del self._nonces[n]
        return nonce

    def _verify_auth(self, username: str, realm: str, nonce: str, data: bytes) -> bool:
        """Verify MESSAGE-INTEGRITY on a request."""
        if username not in self._users:
            return False
        if realm != self._realm:
            return False
        with self._lock:
            if nonce not in self._nonces:
                return False
            ntime = self._nonces[nonce]
            if time.time() - ntime > 3600:
                del self._nonces[nonce]
                return False

        password = self._users[username]
        key = _StunProtocol.compute_long_term_key(username, realm, password)
        return _StunProtocol.authenticate_request(data, key)

    # ── Internal: Error response builder ───────────────────────────

    def _build_error_response(
        self,
        trans_id: bytes,
        code: int,
        reason: str,
        extra_attrs: dict[int, bytes] | None = None,
    ) -> bytes:
        """Build a TURN error response."""
        body = bytearray()
        _class = code // 100
        _num = code % 100
        err_value = struct.pack("!BB", 0, _class) + struct.pack("!B", _num) + reason.encode("utf-8")
        _StunProtocol._add_attr(body, ATTR_ERROR_CODE, err_value)

        if extra_attrs:
            for attr_type, attr_value in extra_attrs.items():
                _StunProtocol._add_attr(body, attr_type, attr_value)

        header = _StunProtocol.build_stun_header(TURN_ALLOCATE_ERROR, len(body), trans_id)
        return header + bytes(body)


def turn_connect(
    server_host: str,
    server_port: int = TURN_DEFAULT_PORT,
    username: str = "",
    password: str = "",
    peer_addr: tuple[str, int] | None = None,
    lifetime: int = 600,
) -> TURNClient | None:
    """Convenience: allocate a TURN relay and optionally create a peer permission.

    Usage::

        client = turn_connect("turn.example.com", 3478, "user", "pass",
                               peer_addr=("203.0.113.5", 50051))
        if client:
            client.send_data(b"hello", peer_addr)
    """
    client = TURNClient(server_host, server_port, username, password, lifetime)
    if not client.allocate():
        client.close()
        return None
    if peer_addr:
        client.create_permission(peer_addr[0], peer_addr[1])
    return client


# ── UDP Hole Puncher ─────────────────────────────────────────────────────────


class UDPHolePuncher:
    """UDP hole-punching with STUN discovery and exponential-backoff pings.

    Attempts to establish a direct UDP connection between two peers behind
    NAT by:
    1. Discovering the public address via STUN
    2. Sending pings (STUN binding requests) to the peer's public address
    3. Listening for pings from the peer
    4. Falling back to TURN if all retries are exhausted

    Usage::

        puncher = UDPHolePuncher()
        success, addr = puncher.punch(
            local_addr=("0.0.0.0", 0),
            peer_candidates=[Candidate(type="host", host="203.0.113.5", port=50051)],
        )
        if success:
            transport = _UDPTransport(sock, addr)
    """

    MAX_RETRIES = 5
    BASE_DELAY = 0.2  # seconds
    MAX_DELAY = 3.0
    PING_TIMEOUT = 2.0

    def __init__(
        self,
        stun_servers: list[tuple[str, int]] | None = None,
        max_retries: int = 5,
    ):
        self._stun_servers = stun_servers or DEFAULT_STUN_SERVERS
        self._max_retries = max_retries or self.MAX_RETRIES
        self._public_addr: tuple[str, int] | None = None
        self._latency: float = 0.0

    def punch(
        self,
        local_addr: tuple[str, int] | None = None,
        peer_candidates: list[Candidate] | list[tuple[str, int]] | None = None,
        timeout: float = 15.0,
    ) -> tuple[bool, tuple[str, int] | None, socket.socket | None]:
        """Attempt UDP hole-punching to a peer.

        Strategy:
        1. Discover public address via STUN.
        2. For each peer candidate, send pings with exponential backoff.
        3. Listen for pings from the peer on a STUN listener socket.
        4. Return the first successful (socket, remote_addr) pair.

        Args:
            local_addr:  ``(host, port)`` to bind, or ``None`` for auto-bind.
            peer_candidates: List of :class:`Candidate` or ``(host, port)`` tuples.
            timeout:     Total time budget in seconds.

        Returns:
            ``(success, negotiated_addr, socket)``.
        """
        if not peer_candidates:
            return (False, None, None)

        # Resolve peer addresses
        peer_addrs: list[tuple[str, int]] = []
        for c in peer_candidates:
            if isinstance(c, Candidate):
                peer_addrs.append((c.host, c.port))
            elif isinstance(c, (tuple, list)) and len(c) == 2:
                peer_addrs.append((c[0], c[1]))

        # Create socket
        bind_addr = local_addr or ("0.0.0.0", 0)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(bind_addr)
        except OSError as e:
            logger.warning(f"UDP hole-punch: bind failed: {e}")
            sock.close()
            return (False, None, None)

        start_time = time.monotonic()
        end_time = start_time + timeout

        # Discover public address
        self._public_addr = self._discover_public_addr()
        if self._public_addr:
            logger.info(f"UDP hole-punch: public address {self._public_addr[0]}:{self._public_addr[1]}")

        # Start a background listener for STUN pings from peer
        self._running = True
        reached_peer: tuple[str, int] | None = None
        recv_lock = threading.Lock()
        recv_buffer: list[tuple[str, int]] = []

        def listener():
            nonlocal reached_peer
            sock.settimeout(0.3)
            while self._running and reached_peer is None:
                try:
                    data, raddr = sock.recvfrom(65535)
                    try:
                        parsed = _StunProtocol.parse_message(data)
                        if parsed["msg_type"] == STUN_BINDING_REQUEST:
                            # Respond so the peer knows we received
                            resp = _StunProtocol.build_binding_response(
                                parsed["trans_id"], sock.getsockname()[:2]
                            )
                            sock.sendto(resp, raddr)
                    except ICEError:
                        pass
                    with recv_lock:
                        if raddr not in recv_buffer:
                            recv_buffer.append(raddr)
                except socket.timeout:
                    continue
                except OSError:
                    break

        listener_thread = threading.Thread(target=listener, daemon=True)
        listener_thread.start()

        # Send pings to each peer candidate with exponential backoff
        for peer_addr in peer_addrs:
            if time.monotonic() >= end_time:
                break
            for attempt in range(self._max_retries):
                if time.monotonic() >= end_time:
                    break

                trans_id = _StunProtocol.generate_transaction_id()
                ping = _StunProtocol.build_binding_request(trans_id)
                try:
                    sock.sendto(ping, peer_addr)
                except OSError:
                    continue

                # Wait with backoff
                delay = min(self.BASE_DELAY * (2 ** attempt), self.MAX_DELAY)
                wait_end = time.monotonic() + delay

                while time.monotonic() < wait_end:
                    time.sleep(0.05)
                    with recv_lock:
                        for raddr in recv_buffer:
                            if raddr == peer_addr or True:  # Any response is good
                                # Send a final ping to establish the mapping
                                try:
                                    trans_id2 = _StunProtocol.generate_transaction_id()
                                    sock.sendto(
                                        _StunProtocol.build_binding_request(trans_id2),
                                        peer_addr,
                                    )
                                except OSError:
                                    pass
                                elapsed = time.monotonic() - start_time
                                self._latency = elapsed
                                self._running = False
                                listener_thread.join(timeout=1.0)
                                logger.info(
                                    f"UDP hole-punch succeeded {sock.getsockname()[:2]} -> {peer_addr}"
                                    f" (attempt {attempt + 1}, {elapsed * 1000:.0f}ms)"
                                )
                                return (True, peer_addr, sock)

            logger.debug(f"UDP hole-punch: no response from {peer_addr} after {self._max_retries} attempts")

        # Hole-punch failed — try the full timeout to detect late pings
        remaining = end_time - time.monotonic()
        if remaining > 0:
            time.sleep(min(remaining, 2.0))

        self._running = False
        listener_thread.join(timeout=1.0)
        sock.close()
        return (False, None, None)

    def _discover_public_addr(self) -> tuple[str, int] | None:
        """Discover public IP:port using STUN."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(3.0)
        try:
            sock.bind(("0.0.0.0", 0))
            for host, port in self._stun_servers:
                try:
                    trans_id = _StunProtocol.generate_transaction_id()
                    req = _StunProtocol.build_binding_request(trans_id)
                    sock.sendto(req, (host, port))
                    data, _ = sock.recvfrom(65535)
                    parsed = _StunProtocol.parse_message(data)
                    if parsed["msg_type"] == STUN_BINDING_RESPONSE:
                        attrs = parsed["attributes"]
                        if ATTR_XOR_MAPPED_ADDRESS in attrs:
                            return _StunProtocol.parse_xor_mapped_address(
                                attrs[ATTR_XOR_MAPPED_ADDRESS], parsed["trans_id"]
                            )
                except (socket.timeout, OSError, ICEError):
                    continue
        finally:
            sock.close()
        return None

    def close(self) -> None:
        """Cancel any in-progress hole-punch attempt."""
        self._running = False

    @property
    def public_address(self) -> tuple[str, int] | None:
        """The discovered public address, or ``None``."""
        return self._public_addr

    @property
    def latency(self) -> float:
        """Estimated connection latency in seconds."""
        return self._latency


# ── NAT Traversal Controller ─────────────────────────────────────────────────


class NATTraversalController:
    """Orchestrates NAT traversal strategies: ICE -> hole-punch -> TURN.

    Tries each strategy in order and returns the first working transport.

    Usage::

        controller = NATTraversalController(
            stun_servers=[("stun.l.google.com", 19302)],
            turn_config={"server": "turn.example.com", "port": 3478,
                         "user": "user", "password": "pass"},
        )
        transport = controller.connect(peer_candidates=[...])
        if transport:
            transport.send(b"payload")
        print(controller.stats())
    """

    def __init__(
        self,
        stun_servers: list[tuple[str, int]] | None = None,
        turn_config: dict[str, Any] | None = None,
        use_ice: bool = True,
        use_hole_punch: bool = True,
        use_turn: bool = True,
    ):
        self._stun_servers = stun_servers or DEFAULT_STUN_SERVERS
        self._turn_config = turn_config
        self._use_ice = use_ice
        self._use_hole_punch = use_hole_punch
        self._use_turn = use_turn

        self._stats: dict[str, Any] = {
            "total_attempts": 0,
            "successes": 0,
            "failures": 0,
            "strategy_used": None,
            "latency_ms": 0.0,
            "strategies": {
                "ice": {"attempted": 0, "succeeded": 0, "latency_ms": 0.0},
                "hole_punch": {"attempted": 0, "succeeded": 0, "latency_ms": 0.0},
                "turn": {"attempted": 0, "succeeded": 0, "latency_ms": 0.0},
            },
        }
        self._lock = threading.Lock()

    def connect(
        self,
        peer_candidates: list[Candidate],
        remote_ufrag: str = "",
        remote_password: str = "",
        timeout: float = 30.0,
    ) -> DataTransport | None:
        """Establish a connection to a peer using the best available strategy.

        Args:
            peer_candidates: The peer's ICE candidates.
            remote_ufrag:    Peer's ICE ufrag (optional, for MESSAGE-INTEGRITY).
            remote_password: Peer's ICE password (optional).
            timeout:         Total time budget for all strategies.

        Returns:
            A :class:`DataTransport` or ``None`` if all strategies failed.
        """
        with self._lock:
            self._stats["total_attempts"] += 1

        start = time.monotonic()

        # Strategy 1: ICE
        if self._use_ice:
            ice_start = time.monotonic()
            self._stats["strategies"]["ice"]["attempted"] += 1
            logger.info("NAT traversal: trying ICE...")

            agent = ICEAgent(
                role="controlling",
                stun_servers=self._stun_servers,
                turn_config=self._turn_config,
            )
            try:
                agent.gather_candidates()
                agent.set_remote_candidates(peer_candidates, remote_ufrag, remote_password)
                pair = agent.connect(timeout=timeout * 0.4)
                if pair and agent.transport:
                    ice_latency = (time.monotonic() - ice_start) * 1000
                    with self._lock:
                        self._stats["strategy_used"] = "ice"
                        self._stats["successes"] += 1
                        self._stats["latency_ms"] = ice_latency
                        self._stats["strategies"]["ice"]["succeeded"] += 1
                        self._stats["strategies"]["ice"]["latency_ms"] = ice_latency
                    logger.info(f"NAT traversal: ICE succeeded ({ice_latency:.0f}ms)")
                    return agent.transport
            except Exception as e:
                logger.debug(f"NAT traversal: ICE failed: {e}")
            finally:
                if agent.transport is None:
                    agent.close()

            with self._lock:
                self._stats["strategies"]["ice"]["latency_ms"] = (
                    time.monotonic() - ice_start
                ) * 1000

        # Strategy 2: UDP hole-punching
        if self._use_hole_punch:
            hp_start = time.monotonic()
            self._stats["strategies"]["hole_punch"]["attempted"] += 1
            logger.info("NAT traversal: trying UDP hole-punching...")

            puncher = UDPHolePuncher(
                stun_servers=self._stun_servers,
                max_retries=3,
            )
            try:
                remaining = timeout - (time.monotonic() - start)
                success, addr, sock = puncher.punch(
                    peer_candidates=peer_candidates,
                    timeout=max(remaining, 5.0),
                )
                if success and sock is not None and addr is not None:
                    hp_latency = (time.monotonic() - hp_start) * 1000
                    transport = _UDPTransport(sock, addr)
                    with self._lock:
                        self._stats["strategy_used"] = "hole_punch"
                        self._stats["successes"] += 1
                        self._stats["latency_ms"] = hp_latency
                        self._stats["strategies"]["hole_punch"]["succeeded"] += 1
                        self._stats["strategies"]["hole_punch"]["latency_ms"] = hp_latency
                    logger.info(f"NAT traversal: hole-punch succeeded ({hp_latency:.0f}ms)")
                    return transport
            except Exception as e:
                logger.debug(f"NAT traversal: hole-punch failed: {e}")

            with self._lock:
                self._stats["strategies"]["hole_punch"]["latency_ms"] = (
                    time.monotonic() - hp_start
                ) * 1000

        # Strategy 3: TURN relay
        if self._use_turn and self._turn_config:
            turn_start = time.monotonic()
            self._stats["strategies"]["turn"]["attempted"] += 1
            logger.info("NAT traversal: trying TURN relay...")

            try:
                client = TURNClient(
                    server_host=self._turn_config.get("server", ""),
                    server_port=self._turn_config.get("port", TURN_DEFAULT_PORT),
                    username=self._turn_config.get("user", ""),
                    password=self._turn_config.get("password", ""),
                )
                if client.allocate():
                    # Create permission for first peer candidate
                    if peer_candidates:
                        first = peer_candidates[0]
                        client.create_permission(first.host, first.port)
                        client.channel_bind(first.host, first.port)

                    turn_latency = (time.monotonic() - turn_start) * 1000

                    # Create a DataTransport wrapper for TURNClient
                    transport = _TURNDataTransport(client)
                    with self._lock:
                        self._stats["strategy_used"] = "turn"
                        self._stats["successes"] += 1
                        self._stats["latency_ms"] = turn_latency
                        self._stats["strategies"]["turn"]["succeeded"] += 1
                        self._stats["strategies"]["turn"]["latency_ms"] = turn_latency

                    logger.info(f"NAT traversal: TURN succeeded ({turn_latency:.0f}ms)")
                    return transport
            except Exception as e:
                logger.debug(f"NAT traversal: TURN failed: {e}")

            with self._lock:
                self._stats["strategies"]["turn"]["latency_ms"] = (
                    time.monotonic() - turn_start
                ) * 1000

        # All strategies failed
        with self._lock:
            self._stats["failures"] += 1
        logger.error("NAT traversal: all strategies failed")
        return None

    def stats(self) -> dict[str, Any]:
        """Return traversal statistics.

        Returns a dict with:
        - ``total_attempts``, ``successes``, ``failures``
        - ``strategy_used``: ``"ice"`` | ``"hole_punch"`` | ``"turn"`` | ``None``
        - ``latency_ms``: connection latency in milliseconds
        - ``strategies``: per-strategy breakdown
        """
        with self._lock:
            return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset all statistics counters."""
        with self._lock:
            self._stats = {
                "total_attempts": 0,
                "successes": 0,
                "failures": 0,
                "strategy_used": None,
                "latency_ms": 0.0,
                "strategies": {
                    "ice": {"attempted": 0, "succeeded": 0, "latency_ms": 0.0},
                    "hole_punch": {"attempted": 0, "succeeded": 0, "latency_ms": 0.0},
                    "turn": {"attempted": 0, "succeeded": 0, "latency_ms": 0.0},
                },
            }


class _TURNDataTransport(DataTransport):
    """Bridge between :class:`TURNClient` and :class:`DataTransport` interface."""

    def __init__(self, client: TURNClient):
        self._client = client
        self._closed = False

    def send(self, data: bytes) -> bool:
        if self._closed or not self._client._sock:
            return False
        # Send to the first peer in the channel map (or via send_data)
        with self._client._lock:
            peer = next(iter(self._client._channel_reverse.values()), None)
            channel = self._client._channel_map.get(peer) if peer else None

        if channel is not None:
            return self._client.send_channel_data(data, channel)
        if peer is not None:
            return self._client.send_data(data, peer)
        return False

    def recv(self, bufsize: int = 65535, timeout: float | None = None) -> bytes | None:
        if self._closed:
            return None
        result = self._client.recv_data(timeout=timeout)
        if result:
            return result[0]
        return None

    def close(self) -> None:
        self._closed = True
        self._client.close()

    @property
    def local_addr(self) -> tuple[str, int]:
        if self._client._sock:
            return self._client._sock.getsockname()[:2]
        return ("0.0.0.0", 0)

    @property
    def remote_addr(self) -> tuple[str, int]:
        return self._client._server


# ── Top-level convenience ─────────────────────────────────────────────────────


def nat_traverse(
    peer_candidates: list[Candidate] | list[tuple[str, int]],
    stun_servers: list[tuple[str, int]] | None = None,
    turn_config: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> DataTransport | None:
    """Quick-start NAT traversal.

    Converts plain ``(host, port)`` tuples to :class:`Candidate` objects
    and delegates to :class:`NATTraversalController`.

    Usage::

        transport = nat_traverse(
            peer_candidates=[("203.0.113.5", 50051)],
            stun_servers=[("stun.l.google.com", 19302)],
        )
    """
    candidates: list[Candidate] = []
    for c in peer_candidates:
        if isinstance(c, Candidate):
            candidates.append(c)
        else:
            host, port = c
            cand = Candidate(
                foundation=f"remote-{host}",
                component=1,
                transport="UDP",
                host=host,
                port=port,
                type="host",
                       )
            object.__setattr__(cand, "priority", candidate_priority(cand))
            candidates.append(cand)

    controller = NATTraversalController(
        stun_servers=stun_servers,
        turn_config=turn_config,
    )
    return controller.connect(candidates, timeout=timeout)
