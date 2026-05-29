"""Comprehensive tests for ConnectionPool (core/connection_pool.py).

Covers:
- Init with defaults and custom config
- Get: create new, reuse pooled, validate dead, timeout
- Put: return to pool, close when full
- Remove: close all for host, no-op on unknown
- CloseAll: clear pool, handle concurrent access
- Concurrency: parallel get/put, race conditions
- Validation: max-age, select() check, getpeername()
- Metrics: hit rate, create count, eviction count
- Context manager: __enter__/__exit__
- Background purge: stale eviction
"""

import socket
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from distllm.core.connection_pool import (
    ConnectionPool,
    ConnectionPoolConfig,
    PooledConnection,
    PoolStats,
)


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def pool():
    """Fresh pool with short timeouts for testing."""
    p = ConnectionPool(
        max_size=5,
        connect_timeout=1.0,
        max_age_s=10.0,
        validate_timeout=0.1,
        purge_interval_s=0,  # no background purge in tests
    )
    yield p
    p.close_all()


@pytest.fixture
def mock_socket():
    """A mock socket that passes validation."""
    sock = MagicMock(spec=socket.socket)
    sock.getpeername.return_value = ("127.0.0.1", 8080)
    sock.fileno.return_value = 42
    return sock


# ── Init ───────────────────────────────────────────────────────────────


class TestConnectionPoolInit:
    def test_default_params(self):
        p = ConnectionPool()
        assert p._max_size == 10
        assert p._connect_timeout == 5.0
        assert p._max_age_s == 300.0

    def test_custom_params(self):
        p = ConnectionPool(max_size=20, connect_timeout=2.0, max_age_s=60.0)
        assert p._max_size == 20
        assert p._connect_timeout == 2.0
        assert p._max_age_s == 60.0

    def test_initial_state(self, pool):
        assert pool.total_pooled == 0
        assert pool.host_count == 0

    def test_config_dataclass(self):
        cfg = ConnectionPoolConfig(max_size=5, connect_timeout=1.0)
        assert cfg.max_size == 5


# ── Get ────────────────────────────────────────────────────────────────


class TestGetConnection:
    @patch("distllm.core.connection_pool.socket.create_connection")
    def test_get_creates_new(self, mock_create, pool, mock_socket):
        mock_create.return_value = mock_socket
        sock = pool.get("127.0.0.1", 8080)
        assert sock is mock_socket
        mock_create.assert_called_once_with(("127.0.0.1", 8080), timeout=1.0)

    @patch("distllm.core.connection_pool.select.select", return_value=([], [True], []))
    @patch("distllm.core.connection_pool.socket.create_connection")
    def test_get_reuses_pooled(self, mock_create, mock_select, pool, mock_socket):
        mock_create.return_value = mock_socket
        sock = pool.get("127.0.0.1", 8080)
        pool.put("127.0.0.1", 8080, sock)
        mock_create.reset_mock()

        sock2 = pool.get("127.0.0.1", 8080)
        assert sock2 is mock_socket
        mock_create.assert_not_called()

    @patch("distllm.core.connection_pool.socket.create_connection")
    def test_get_removes_dead_pooled(self, mock_create, pool):
        dead_sock = MagicMock(spec=socket.socket)
        dead_sock.getpeername.side_effect = OSError("Connection reset")
        dead_sock.fileno.return_value = 1

        fresh_sock = MagicMock(spec=socket.socket)
        fresh_sock.getpeername.return_value = ("127.0.0.1", 8080)
        fresh_sock.fileno.return_value = 2
        mock_create.return_value = fresh_sock

        pool._pool[("127.0.0.1", 8080)] = [
            PooledConnection(socket=dead_sock, created_at=time.time()),
        ]

        with patch("distllm.core.connection_pool.select.select", return_value=([], [True], [])):
            sock = pool.get("127.0.0.1", 8080)
        assert sock is fresh_sock

    @patch("distllm.core.connection_pool.socket.create_connection")
    def test_get_timeout_on_connect(self, mock_create, pool):
        mock_create.side_effect = OSError("Connection refused")
        with pytest.raises(OSError):
            pool.get("127.0.0.1", 9999)

    def test_get_after_close_raises(self, pool):
        pool.close_all()
        with pytest.raises(OSError, match="closed"):
            pool.get("127.0.0.1", 8080)


# ── Put ────────────────────────────────────────────────────────────────


class TestPutConnection:
    @patch("distllm.core.connection_pool.select.select", return_value=([], [True], []))
    @patch("distllm.core.connection_pool.socket.create_connection")
    def test_put_returns_to_pool(self, mock_create, mock_select, pool, mock_socket):
        mock_create.return_value = mock_socket
        sock = pool.get("127.0.0.1", 8080)
        pool.put("127.0.0.1", 8080, sock)
        assert pool.total_pooled == 1

    @patch("distllm.core.connection_pool.select.select", return_value=([], [True], []))
    @patch("distllm.core.connection_pool.socket.create_connection")
    def test_put_closes_when_full(self, mock_create, mock_select, pool):
        # Create 5 distinct sockets and put them all
        socks = []
        for i in range(5):
            s = MagicMock(spec=socket.socket)
            s.getpeername.return_value = ("127.0.0.1", 8080)
            s.fileno.return_value = i
            socks.append(s)
            pool.put("127.0.0.1", 8080, s)

        assert pool.total_pooled == 5

        # 6th socket should be closed since pool is full
        extra = MagicMock(spec=socket.socket)
        pool.put("127.0.0.1", 8080, extra)
        extra.close.assert_called_once()
        assert pool.total_pooled == 5

    def test_put_after_close_closes_socket(self, pool):
        pool.close_all()
        sock = MagicMock(spec=socket.socket)
        pool.put("127.0.0.1", 8080, sock)
        sock.close.assert_called_once()


# ── Remove ─────────────────────────────────────────────────────────────


class TestRemoveHost:
    def test_remove_closes_all(self, pool):
        socks = []
        for _ in range(3):
            s = MagicMock(spec=socket.socket)
            s.getpeername.return_value = ("127.0.0.1", 8080)
            s.fileno.return_value = len(socks)
            pool._pool.setdefault(("127.0.0.1", 8080), []).append(
                PooledConnection(socket=s)
            )
            socks.append(s)

        pool.remove("127.0.0.1", 8080)
        for s in socks:
            s.close.assert_called_once()
        assert pool.host_count == 0

    def test_remove_noop_on_unknown(self, pool):
        pool.remove("127.0.0.1", 9999)  # should not raise

    def test_remove_doesnt_affect_other_hosts(self, pool):
        s1 = MagicMock(spec=socket.socket)
        s2 = MagicMock(spec=socket.socket)
        pool._pool[("host-a", 80)] = [PooledConnection(socket=s1)]
        pool._pool[("host-b", 80)] = [PooledConnection(socket=s2)]

        pool.remove("host-a", 80)
        assert ("host-b", 80) in pool._pool
        s2.close.assert_not_called()


# ── CloseAll ───────────────────────────────────────────────────────────


class TestCloseAll:
    def test_close_all_clears_pool(self, pool):
        s = MagicMock(spec=socket.socket)
        pool._pool[("h", 1)] = [PooledConnection(socket=s)]
        pool.close_all()
        assert pool.total_pooled == 0
        s.close.assert_called_once()

    def test_close_all_sets_closed_flag(self, pool):
        pool.close_all()
        assert pool._closed is True

    def test_close_all_idempotent(self, pool):
        pool.close_all()
        pool.close_all()  # should not raise


# ── Concurrency ────────────────────────────────────────────────────────


class TestConcurrency:
    @patch("distllm.core.connection_pool.select.select", return_value=([], [True], []))
    @patch("distllm.core.connection_pool.socket.create_connection")
    def test_concurrent_get_put(self, mock_create, mock_select, pool):
        mock_create.return_value = MagicMock(spec=socket.socket)
        errors = []

        def worker():
            try:
                for _ in range(20):
                    sock = pool.get("127.0.0.1", 8080)
                    pool.put("127.0.0.1", 8080, sock)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    @patch("distllm.core.connection_pool.select.select", return_value=([], [True], []))
    @patch("distllm.core.connection_pool.socket.create_connection")
    def test_concurrent_get_same_host(self, mock_create, mock_select, pool):
        mock_create.return_value = MagicMock(spec=socket.socket)
        results = []
        lock = threading.Lock()

        def worker():
            sock = pool.get("127.0.0.1", 8080)
            with lock:
                results.append(id(sock))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10


# ── Validation ─────────────────────────────────────────────────────────


class TestValidation:
    def test_max_age_eviction(self, pool):
        old_sock = MagicMock(spec=socket.socket)
        old_sock.getpeername.return_value = ("h", 1)
        old_conn = PooledConnection(
            socket=old_sock,
            created_at=time.time() - 9999,  # very old
        )
        pool._pool[("h", 1)] = [old_conn]

        with patch("distllm.core.connection_pool.socket.create_connection") as mock:
            fresh = MagicMock(spec=socket.socket)
            mock.return_value = fresh
            sock = pool.get("h", 1)
            assert sock is fresh
        old_sock.close.assert_called_once()

    def test_select_validation(self, pool):
        """Connection that fails select() should be evicted."""
        bad_sock = MagicMock(spec=socket.socket)
        bad_sock.getpeername.return_value = ("h", 1)
        # select returns empty writable list — connection is dead
        with patch("distllm.core.connection_pool.select.select", return_value=([], [], [])):
            conn = PooledConnection(socket=bad_sock, created_at=time.time())
            assert pool._is_valid(conn, time.time()) is False


# ── Metrics ────────────────────────────────────────────────────────────


class TestMetrics:
    @patch("distllm.core.connection_pool.select.select", return_value=([], [True], []))
    @patch("distllm.core.connection_pool.socket.create_connection")
    def test_hit_rate(self, mock_create, mock_select, pool, mock_socket):
        mock_create.return_value = mock_socket

        # Miss
        pool.get("h", 1)
        pool.put("h", 1, mock_socket)

        # Hit
        pool.get("h", 1)

        stats = pool.stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_rate == 0.5

    @patch("distllm.core.connection_pool.socket.create_connection")
    def test_create_count(self, mock_create, pool, mock_socket):
        mock_create.return_value = mock_socket
        pool.get("h", 1)
        pool.get("h", 2)
        assert pool.stats().creates == 2


# ── Context Manager ────────────────────────────────────────────────────


class TestContextManager:
    def test_context_manager(self):
        with ConnectionPool(purge_interval_s=0) as p:
            assert p.total_pooled == 0
        assert p._closed is True


# ── Repr ───────────────────────────────────────────────────────────────


class TestRepr:
    def test_repr(self, pool):
        r = repr(pool)
        assert "ConnectionPool" in r
        assert "hosts=" in r


# ── Background Purge ───────────────────────────────────────────────────


class TestBackgroundPurge:
    def test_purge_evicts_stale(self):
        p = ConnectionPool(max_size=5, max_age_s=0.1, purge_interval_s=0)
        old_sock = MagicMock(spec=socket.socket)
        old_sock.getpeername.return_value = ("h", 1)
        p._pool[("h", 1)] = [PooledConnection(socket=old_sock, created_at=time.time() - 1.0)]

        evicted = p.purge_stale()
        assert evicted == 1
        assert p.total_pooled == 0
        old_sock.close.assert_called_once()
        p.close_all()
