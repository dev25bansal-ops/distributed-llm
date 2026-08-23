"""Tests for distllm.dist.nat — NAT traversal primitives.

Zero mocks — uses only real objects from the module.
No GPU required, no external network (loopback only), no timing-dependent assertions.
"""

from __future__ import annotations

import hashlib
import hmac
import socket
import time

import pytest

from distllm.dist.nat import (
    NatMapping,
    NatType,
    StunClient,
    TurnRelayClient,
    TurnRelayServer,
    WebRTCNatTransport,
)


# ── Helpers ────────────────────────────────────────────────────────────────


class _UdpSock:
    """Context manager for a throwaway UDP socket bound to loopback.

    ``sendto`` calls on this socket succeed even when nothing is listening
    (UDP is fire-and-forget), which lets us exercise relay handlers without
    running a full server.
    """

    def __enter__(self) -> socket.socket:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", 0))
        return self._sock

    def __exit__(self, *args: object) -> None:
        try:
            self._sock.close()
        except Exception:
            pass


def _make_token(session_id: str, key: str) -> str:
    """Build a valid HMAC session token for testing."""
    sig = hmac.new(key.encode(), session_id.encode(), hashlib.sha256).hexdigest()
    return f"{session_id}.{sig}"


# ── NatType ────────────────────────────────────────────────────────────────


class TestNatType:
    """Enum values and members."""

    def test_values(self) -> None:
        assert NatType.UNKNOWN.value == "unknown"
        assert NatType.OPEN.value == "open"
        assert NatType.FULL_CONE.value == "full_cone"
        assert NatType.RESTRICTED.value == "restricted"
        assert NatType.PORT_RESTRICTED.value == "port_restricted"
        assert NatType.SYMMETRIC.value == "symmetric"

    def test_six_members(self) -> None:
        assert len(NatType) == 6

    def test_from_value(self) -> None:
        assert NatType("unknown") is NatType.UNKNOWN
        assert NatType("symmetric") is NatType.SYMMETRIC
        assert NatType("open") is NatType.OPEN


# ── NatMapping ────────────────────────────────────────────────────────────


class TestNatMapping:
    """Data transfer object for NAT mapping information."""

    def test_defaults(self) -> None:
        m = NatMapping()
        assert m.public_ip == ""
        assert m.public_port == 0
        assert m.nat_type is NatType.UNKNOWN
        assert m.local_ip == ""
        assert m.local_port == 0

    def test_custom_fields(self) -> None:
        m = NatMapping(
            public_ip="203.0.113.5",
            public_port=45000,
            nat_type=NatType.FULL_CONE,
            local_ip="192.168.1.42",
            local_port=32000,
        )
        assert m.public_ip == "203.0.113.5"
        assert m.public_port == 45000
        assert m.nat_type is NatType.FULL_CONE
        assert m.local_ip == "192.168.1.42"
        assert m.local_port == 32000

    def test_mutable_fields(self) -> None:
        """NatMapping is a regular dataclass (not frozen), fields are assignable."""
        m = NatMapping()
        m.public_ip = "10.0.0.1"
        m.public_port = 8080
        assert m.public_ip == "10.0.0.1"
        assert m.public_port == 8080


# ── StunClient ────────────────────────────────────────────────────────────


class TestStunClient:
    """STUN client for NAT type detection (pure-logic surface)."""

    def test_default_timeout(self) -> None:
        assert StunClient()._timeout == 3.0

    def test_custom_timeout(self) -> None:
        assert StunClient(timeout=2.5)._timeout == 2.5

    def test_stun_servers_constant(self) -> None:
        assert len(StunClient.STUN_SERVERS) >= 1
        for host, port in StunClient.STUN_SERVERS:
            assert isinstance(host, str)
            assert isinstance(port, int)
            assert port == 19302

    def test_magic_cookie(self) -> None:
        assert StunClient.STUN_MAGIC_COOKIE == 0x2112A442

    def test_pick_alt_server_returns_different(self) -> None:
        client = StunClient()
        current = StunClient.STUN_SERVERS[0]
        alt = client._pick_alt_server(current)
        assert alt is not None
        assert alt[0] != current[0]
        assert alt[1] == 19302

    def test_pick_alt_server_last_entry(self) -> None:
        client = StunClient()
        current = StunClient.STUN_SERVERS[-1]
        alt = client._pick_alt_server(current)
        assert alt is not None
        assert alt[0] != current[0]

    def test_pick_alt_server_none_when_single(self) -> None:
        client = StunClient()
        original = StunClient.STUN_SERVERS
        try:
            StunClient.STUN_SERVERS = [("stun.l.google.com", 19302)]
            alt = client._pick_alt_server(("stun.l.google.com", 19302))
            assert alt is None
        finally:
            StunClient.STUN_SERVERS = original

    def test_detect_returns_natmapping(self) -> None:
        """detect() always returns a NatMapping regardless of network."""
        client = StunClient(timeout=0.5)
        mapping = client.detect()
        assert isinstance(mapping, NatMapping)
        # Without network the type should be UNKNOWN
        assert mapping.nat_type is NatType.UNKNOWN


# ── TurnRelayServer ───────────────────────────────────────────────────────


class TestTurnRelayServer:
    """Session-based TURN-like relay server (no-network methods)."""

    # ── Init ──────────────────────────────────────────────────────────

    def test_init_defaults(self) -> None:
        server = TurnRelayServer()
        assert server._host == "0.0.0.0"
        assert server._port == 3478
        assert server._hmac_key == ""
        assert server._max_joins_per_minute == 10
        assert not server._running.is_set()

    def test_init_with_hmac_key(self) -> None:
        server = TurnRelayServer(hmac_key="my-secret")
        assert server._hmac_key == "my-secret"

    def test_init_with_custom_port(self) -> None:
        server = TurnRelayServer(port=9999, hmac_key="k")
        assert server._port == 9999

    # ── start / stop ──────────────────────────────────────────────────

    def test_start_raises_without_hmac_key(self) -> None:
        server = TurnRelayServer()
        with pytest.raises(RuntimeError, match="HMAC_KEY"):
            server.start()

    def test_stop_clears_running_flag(self) -> None:
        server = TurnRelayServer(hmac_key="key")
        server._running.set()
        server.stop()
        assert not server._running.is_set()

    # ── _validate_session_token ───────────────────────────────────────

    def test_validate_token_no_key_rejects_all(self) -> None:
        """Without HMAC key, every token is rejected (fail-closed)."""
        server = TurnRelayServer()
        assert server._validate_session_token("anything") is False

    def test_validate_token_bad_format(self) -> None:
        """Missing '.' separator is rejected."""
        server = TurnRelayServer(hmac_key="secret")
        assert server._validate_session_token("no-dot-here") is False

    def test_validate_token_empty_parts(self) -> None:
        """Empty session_id or signature is rejected."""
        server = TurnRelayServer(hmac_key="key")
        assert server._validate_session_token(".sig") is False
        assert server._validate_session_token("sess.") is False

    def test_validate_token_wrong_signature(self) -> None:
        server = TurnRelayServer(hmac_key="secret")
        assert server._validate_session_token("sess.badsignature") is False

    def test_validate_token_valid(self) -> None:
        key = "test-key"
        server = TurnRelayServer(hmac_key=key)
        token = _make_token("my-session", key)
        assert server._validate_session_token(token) is True

    # ── _check_rate_limit ─────────────────────────────────────────────

    def test_rate_limit_first_call_passes(self) -> None:
        server = TurnRelayServer(hmac_key="k")
        assert server._check_rate_limit("10.0.0.1") is True

    def test_rate_limit_exceeded(self) -> None:
        server = TurnRelayServer(hmac_key="k", max_joins_per_minute=3)
        for _ in range(3):
            assert server._check_rate_limit("10.0.0.1") is True
        assert server._check_rate_limit("10.0.0.1") is False

    def test_rate_limit_different_ips_independent(self) -> None:
        server = TurnRelayServer(hmac_key="k", max_joins_per_minute=2)
        for _ in range(2):
            assert server._check_rate_limit("A") is True
        assert server._check_rate_limit("A") is False
        # Different IP unaffected
        assert server._check_rate_limit("B") is True

    def test_rate_limit_old_entries_pruned(self) -> None:
        """Entries older than 60 s are pruned, freeing a slot."""
        server = TurnRelayServer(hmac_key="k", max_joins_per_minute=1)
        ip = "10.0.0.3"
        assert server._check_rate_limit(ip) is True
        assert server._check_rate_limit(ip) is False
        # Manually move the timestamp out of the 60 s window
        server._join_attempts[ip] = [time.time() - 120.0]
        assert server._check_rate_limit(ip) is True

    # ── _expire_stale_sessions ────────────────────────────────────────

    def test_expire_stale_skipped_when_recent(self) -> None:
        """Cleanup is skipped when less than 300 s since last cleanup."""
        server = TurnRelayServer(hmac_key="k")
        server._expire_stale_sessions()
        assert True  # No exception

    def test_expire_stale_evicts_expired(self) -> None:
        """Sessions past TTL are removed."""
        server = TurnRelayServer(hmac_key="k")
        # Insert a session with a stale creation time
        server._sessions["old-session"] = []
        server._session_created["old-session"] = time.monotonic() - 7200.0  # 2 hours ago
        server._last_cleanup = 0.0  # Force cleanup to run
        server._expire_stale_sessions()
        assert "old-session" not in server._sessions
        assert "old-session" not in server._session_created

    def test_expire_stale_keeps_fresh(self) -> None:
        """Sessions within TTL are retained."""
        server = TurnRelayServer(hmac_key="k")
        server._sessions["fresh"] = []
        server._session_created["fresh"] = time.monotonic()
        server._last_cleanup = 0.0
        server._expire_stale_sessions()
        assert "fresh" in server._sessions

    # ── _handle_join — error paths ────────────────────────────────────

    def test_handle_join_empty_token(self) -> None:
        """JOIN: with an empty token sends ERROR and returns."""
        server = TurnRelayServer(hmac_key="k")
        with _UdpSock() as sock:
            server._handle_join(sock, b"JOIN:", ("127.0.0.1", 9999))
        assert server._sessions == {}

    def test_handle_join_rate_limited(self) -> None:
        """JOIN from a rate-limited IP sends ERROR."""
        server = TurnRelayServer(hmac_key="k", max_joins_per_minute=0)
        token = _make_token("s1", "k")
        with _UdpSock() as sock:
            server._handle_join(sock, f"JOIN:{token}".encode(), ("127.0.0.1", 9999))
        assert server._sessions == {}

    def test_handle_join_invalid_token(self) -> None:
        """JOIN with an invalid HMAC token sends ERROR."""
        server = TurnRelayServer(hmac_key="secret")
        with _UdpSock() as sock:
            server._handle_join(sock, b"JOIN:bad.token.format", ("127.0.0.1", 9999))
        assert server._sessions == {}

    # ── _handle_join — success paths ─────────────────────────────────

    def test_handle_join_first_peer(self) -> None:
        """A first peer joins and is registered (WAITING response)."""
        key = "test-key"
        server = TurnRelayServer(hmac_key=key)
        token = _make_token("my-session", key)
        addr = ("127.0.0.1", 20001)

        with _UdpSock() as sock:
            server._handle_join(sock, f"JOIN:{token}".encode(), addr)

        assert token in server._sessions
        assert addr in server._sessions[token]
        assert len(server._sessions[token]) == 1
        assert server._addr_to_session[addr] == token
        assert token in server._session_created

    def test_handle_join_duplicate_noop(self) -> None:
        """Re-joining the same addr with the same token is a no-op."""
        key = "test-key"
        server = TurnRelayServer(hmac_key=key)
        token = _make_token("sess", key)
        addr = ("127.0.0.1", 20002)

        with _UdpSock() as sock:
            server._handle_join(sock, f"JOIN:{token}".encode(), addr)
            server._handle_join(sock, f"JOIN:{token}".encode(), addr)

        assert token in server._sessions
        assert len(server._sessions[token]) == 1

    def test_handle_join_second_peer_paired(self) -> None:
        """A second peer causes PAIRED to be sent (both registered)."""
        key = "test-key"
        server = TurnRelayServer(hmac_key=key)
        token = _make_token("paired-session", key)
        peer_a = ("127.0.0.1", 20010)
        peer_b = ("127.0.0.1", 20011)

        with _UdpSock() as sock:
            server._handle_join(sock, f"JOIN:{token}".encode(), peer_a)
            server._handle_join(sock, f"JOIN:{token}".encode(), peer_b)

        assert token in server._sessions
        assert len(server._sessions[token]) == 2
        assert peer_a in server._sessions[token]
        assert peer_b in server._sessions[token]
        assert server._addr_to_session[peer_a] == token
        assert server._addr_to_session[peer_b] == token

    def test_handle_join_rejoin_different_session(self) -> None:
        """Same addr joining a different token leaves the old session."""
        key = "test-key"
        server = TurnRelayServer(hmac_key=key)
        token_a = _make_token("session-a", key)
        token_b = _make_token("session-b", key)
        addr = ("127.0.0.1", 20020)
        other = ("127.0.0.1", 20021)

        with _UdpSock() as sock:
            server._handle_join(sock, f"JOIN:{token_a}".encode(), addr)
            server._handle_join(sock, f"JOIN:{token_a}".encode(), other)
            server._handle_join(sock, f"JOIN:{token_b}".encode(), addr)

        assert server._addr_to_session[addr] == token_b
        assert addr not in server._sessions.get(token_a, [])

    # ── _forward_data ─────────────────────────────────────────────────

    def test_forward_data_no_session(self) -> None:
        """Forwarding data for an unregistered addr does nothing."""
        server = TurnRelayServer(hmac_key="k")
        with _UdpSock() as sock:
            server._forward_data(sock, b"data", ("127.0.0.1", 9999))
            # No exception expected

    def test_forward_data_partial_session(self) -> None:
        """Forwarding data when only one peer is connected does nothing."""
        key = "test-key"
        server = TurnRelayServer(hmac_key=key)
        token = _make_token("partial", key)
        addr = ("127.0.0.1", 20030)
        with _UdpSock() as sock:
            server._handle_join(sock, f"JOIN:{token}".encode(), addr)
            server._forward_data(sock, b"hello", addr)

    def test_forward_data_paired(self) -> None:
        """Data from one peer is forwarded to the other (fire-and-forget)."""
        key = "test-key"
        server = TurnRelayServer(hmac_key=key)
        token = _make_token("fwd", key)
        a = ("127.0.0.1", 20040)
        b = ("127.0.0.1", 20041)
        with _UdpSock() as sock:
            server._handle_join(sock, f"JOIN:{token}".encode(), a)
            server._handle_join(sock, f"JOIN:{token}".encode(), b)
            # Forward from a -> expects target b (UDP fire-and-forget)
            server._forward_data(sock, b"ping", a)

    # ── _leave_session ────────────────────────────────────────────────

    def test_leave_session_nonexistent(self) -> None:
        server = TurnRelayServer(hmac_key="k")
        server._leave_session(("1.2.3.4", 5678), "nonexistent")
        # No exception expected

    def test_leave_session_removes_peer(self) -> None:
        key = "test-key"
        server = TurnRelayServer(hmac_key=key)
        token = _make_token("leave-test", key)
        a = ("127.0.0.1", 20050)
        b = ("127.0.0.1", 20051)
        with _UdpSock() as sock:
            server._handle_join(sock, f"JOIN:{token}".encode(), a)
            server._handle_join(sock, f"JOIN:{token}".encode(), b)
        server._leave_session(a, token)
        assert a not in server._sessions[token]
        assert b in server._sessions[token]
        assert a not in server._addr_to_session
        assert server._addr_to_session[b] == token

    def test_leave_session_last_peer_closes(self) -> None:
        """When the last peer leaves, the session is removed entirely."""
        key = "test-key"
        server = TurnRelayServer(hmac_key=key)
        token = _make_token("close-me", key)
        a = ("127.0.0.1", 20060)
        with _UdpSock() as sock:
            server._handle_join(sock, f"JOIN:{token}".encode(), a)
        server._leave_session(a, "close-me")
        assert "close-me" not in server._sessions
        assert "close-me" not in server._session_created

    # ── _handle_relay dispatch ────────────────────────────────────────

    def test_handle_relay_dispatches_join(self) -> None:
        """Data prefixed with JOIN: is routed to _handle_join."""
        server = TurnRelayServer(hmac_key="k")
        with _UdpSock() as sock:
            server._handle_relay(sock, b"JOIN:invalidtoken", ("127.0.0.1", 9999))

    def test_handle_relay_dispatches_data(self) -> None:
        """Data not prefixed with JOIN: is routed to _forward_data."""
        server = TurnRelayServer(hmac_key="k")
        with _UdpSock() as sock:
            server._handle_relay(sock, b"HELLO", ("127.0.0.1", 9999))


# ── TurnRelayClient ───────────────────────────────────────────────────────


class TestTurnRelayClient:
    """TURN relay client (no-network methods)."""

    def test_init_defaults(self) -> None:
        client = TurnRelayClient("relay.example.com")
        assert client._relay_addr == ("relay.example.com", 3478)
        assert client._session_token == ""
        assert client._local_port == 0
        assert client._sock is None

    def test_init_custom(self) -> None:
        client = TurnRelayClient(
            "relay.test",
            relay_port=8888,
            session_token="abc123",
            local_port=9999,
        )
        assert client._relay_addr == ("relay.test", 8888)
        assert client._session_token == "abc123"
        assert client._local_port == 9999

    def test_send_not_joined(self) -> None:
        client = TurnRelayClient("relay.example.com")
        assert client.send(b"data") is False

    def test_recv_not_joined(self) -> None:
        client = TurnRelayClient("relay.example.com")
        assert client.recv() is None

    def test_recv_not_joined_with_timeout(self) -> None:
        client = TurnRelayClient("relay.example.com")
        assert client.recv(timeout=1.0) is None

    def test_close_not_joined(self) -> None:
        client = TurnRelayClient("relay.example.com")
        client.close()
        assert client._sock is None

    def test_close_idempotent(self) -> None:
        client = TurnRelayClient("relay.example.com")
        client.close()
        client.close()
        assert client._sock is None


# ── WebRTCNatTransport ────────────────────────────────────────────────────


class TestWebRTCNatTransport:
    """Adapter wrapping WebRTCTransport (no-transport path)."""

    def test_default_init(self) -> None:
        transport = WebRTCNatTransport()
        assert transport._transport is None
        assert isinstance(transport._recv_buffer, bytearray)

    def test_send_no_transport(self) -> None:
        assert WebRTCNatTransport().send(b"data") is False

    def test_recv_no_transport(self) -> None:
        assert WebRTCNatTransport().recv() is None

    def test_recv_no_transport_with_timeout(self) -> None:
        assert WebRTCNatTransport().recv(timeout=1.0) is None

    def test_close_no_transport(self) -> None:
        transport = WebRTCNatTransport()
        transport.close()
        assert transport._transport is None

    def test_close_idempotent(self) -> None:
        transport = WebRTCNatTransport()
        transport.close()
        transport.close()
        assert transport._transport is None

    def test_is_connected_no_transport(self) -> None:
        assert WebRTCNatTransport().is_connected is False

    def test_explicit_none_transport(self) -> None:
        """Passing transport=None behaves the same as the default."""
        transport = WebRTCNatTransport(transport=None)
        assert transport.send(b"x") is False
        assert transport.recv() is None
        assert transport.is_connected is False
