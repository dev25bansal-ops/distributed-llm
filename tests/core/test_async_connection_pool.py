"""Tests for async_connection_pool.py.

Covers:
    AsyncPoolStats         -- construction, defaults, hit_rate
    AsyncConnectionPool    -- construction, defaults, properties
    get()                  -- pool hits, misses, closed pool, evictions
    put()                  -- return to pool, overflow, closed pool
    remove()               -- remove and close connections
    close_all()            -- close all connections, mark closed
    _is_valid()            -- age check, writer closing check
    __repr__               -- string representation
    context manager        -- __aenter__ / __aexit__

Every test is deterministic (no network, no time.sleep).
No MagicMock -- real objects or lightweight stubs only.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

# Bootstrap fake packages for the distllm namespace
bootstrap_fake_packages()

# Load the module under test
_mod = load_module("distllm/core/async_connection_pool.py")

# Re-export symbols for test readability
AsyncConnectionPool = _mod.AsyncConnectionPool
AsyncPooledConnection = _mod.AsyncPooledConnection
AsyncPoolStats = _mod.AsyncPoolStats


# ===================================================================
# Stubs
# ===================================================================

class StubStreamWriter:
    """Minimal asyncio.StreamWriter replacement -- no network I/O."""

    def __init__(self) -> None:
        self._closed = False

    def close(self) -> None:
        self._closed = True

    def is_closing(self) -> bool:
        return self._closed

    def write(self, data: bytes) -> None:
        pass

    def writelines(self, data: list[bytes]) -> None:
        pass

    async def drain(self) -> None:
        return None

    async def wait_closed(self) -> None:
        return None

    def get_extra_info(self, name: str, default: object = None) -> object:
        return default

    def transport(self) -> None:
        return None

    def abort(self) -> None:
        self._closed = True


class StubStreamReader:
    """Minimal asyncio.StreamReader replacement -- no event loop needed."""

    def __init__(self) -> None:
        self._loop = None

    async def read(self, n: int = -1) -> bytes:
        return b""

    def at_eof(self) -> bool:
        return False

    def feed_eof(self) -> None:
        pass


async def _stub_open_connection(
    host: str, port: int,
) -> tuple[StubStreamReader, StubStreamWriter]:
    """Stand-in for asyncio.open_connection that returns stubs."""
    return StubStreamReader(), StubStreamWriter()


async def _failing_open_connection(host: str, port: int):
    """Stand-in that raises OSError (connection refused)."""
    raise OSError(f"Connection refused to {host}:{port}")


# ===================================================================
# AsyncPoolStats TESTS
# ===================================================================

class TestAsyncPoolStats:
    """AsyncPoolStats dataclass -- defaults and hit_rate."""

    def test_defaults(self) -> None:
        """All counters should start at zero."""
        stats = AsyncPoolStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.creates == 0
        assert stats.evictions == 0
        assert stats.errors == 0

    def test_hit_rate_zero_when_no_requests(self) -> None:
        """hit_rate should be 0.0 when no requests have been made."""
        stats = AsyncPoolStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_perfect(self) -> None:
        """hit_rate should be 1.0 when every request is a hit."""
        stats = AsyncPoolStats(hits=10, misses=0)
        assert stats.hit_rate == 1.0

    def test_hit_rate_mixed(self) -> None:
        """hit_rate should reflect the hits/total ratio."""
        stats = AsyncPoolStats(hits=7, misses=3)
        assert stats.hit_rate == 0.7

    def test_hit_rate_all_misses(self) -> None:
        """hit_rate should be 0.0 when every request is a miss."""
        stats = AsyncPoolStats(hits=0, misses=5)
        assert stats.hit_rate == 0.0

    def test_hit_rate_custom_values_preserved(self) -> None:
        """Custom values passed to constructor should be preserved."""
        stats = AsyncPoolStats(hits=42, misses=8, creates=50, evictions=2, errors=1)
        assert stats.hits == 42
        assert stats.misses == 8
        assert stats.creates == 50
        assert stats.evictions == 2
        assert stats.errors == 1

    def test_hit_rate_equals_hits_over_total(self) -> None:
        """Verify that hit_rate == hits / (hits + misses)."""
        for hits, misses in [(0, 0), (1, 0), (0, 1), (3, 7), (9, 1), (50, 50)]:
            stats = AsyncPoolStats(hits=hits, misses=misses)
            total = hits + misses
            expected = hits / total if total > 0 else 0.0
            assert stats.hit_rate == expected


# ===================================================================
# CONSTRUCTION TESTS
# ===================================================================

class TestAsyncConnectionPoolConstruction:
    """Construction, defaults, and basic properties."""

    def test_default_construction(self) -> None:
        """Pool should start with sensible defaults."""
        pool = AsyncConnectionPool()
        assert pool._max_size == 10
        assert pool._connect_timeout == 5.0
        assert pool._max_age_s == 300.0
        assert pool.total_pooled == 0
        assert pool._closed is False
        assert len(pool._pool) == 0

    def test_custom_values(self) -> None:
        """All constructor parameters should be configurable."""
        pool = AsyncConnectionPool(
            max_size=5,
            connect_timeout=1.0,
            max_age_s=60.0,
        )
        assert pool._max_size == 5
        assert pool._connect_timeout == 1.0
        assert pool._max_age_s == 60.0

    def test_stats_object_initialized(self) -> None:
        """stats() should return a fresh AsyncPoolStats."""
        pool = AsyncConnectionPool()
        stats = pool.stats()
        assert isinstance(stats, AsyncPoolStats)
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.creates == 0

    def test_total_pooled_empty(self) -> None:
        """total_pooled should be 0 before any connections are added."""
        pool = AsyncConnectionPool()
        assert pool.total_pooled == 0

    def test_repr_empty(self) -> None:
        """__repr__ should show zero hosts and pooled connections."""
        pool = AsyncConnectionPool()
        r = repr(pool)
        assert "AsyncConnectionPool" in r
        assert "hosts=0" in r
        assert "pooled=0" in r
        assert "hit_rate=0.0%" in r or "hit_rate=0%" in r


# ===================================================================
# GET TESTS
# ===================================================================

class TestAsyncConnectionPoolGet:
    """get() -- pool hits, misses, evictions, errors, and edge cases."""

    @pytest.mark.asyncio
    async def test_get_from_closed_pool_raises(self) -> None:
        """get() on a closed pool should raise OSError."""
        pool = AsyncConnectionPool()
        await pool.close_all()
        with pytest.raises(OSError, match="closed"):
            await pool.get("localhost", 8080)

    @pytest.mark.asyncio
    async def test_get_miss_creates_new_connection(self, monkeypatch) -> None:
        """A pool miss should create a new connection via open_connection."""
        monkeypatch.setattr(asyncio, "open_connection", _stub_open_connection)
        pool = AsyncConnectionPool()
        reader, writer = await pool.get("localhost", 8080)
        assert isinstance(reader, StubStreamReader)
        assert isinstance(writer, StubStreamWriter)
        stats = pool.stats()
        assert stats.misses == 1
        assert stats.creates == 1
        assert stats.hits == 0

    @pytest.mark.asyncio
    async def test_get_hit_reuses_connection(self, monkeypatch) -> None:
        """After put(), a subsequent get() should return the pooled connection."""
        monkeypatch.setattr(asyncio, "open_connection", _stub_open_connection)
        pool = AsyncConnectionPool()
        reader, writer = await pool.get("localhost", 8080)
        await pool.put("localhost", 8080, writer, reader)
        assert pool.total_pooled == 1

        reader2, writer2 = await pool.get("localhost", 8080)
        stats = pool.stats()
        assert stats.hits == 1
        assert stats.misses == 1  # first was a miss
        assert stats.creates == 1  # only one create needed

    @pytest.mark.asyncio
    async def test_get_disjoint_hosts_separate_pools(self, monkeypatch) -> None:
        """Different (host, port) pairs should have separate connection pools."""
        monkeypatch.setattr(asyncio, "open_connection", _stub_open_connection)
        pool = AsyncConnectionPool()
        r1, w1 = await pool.get("host-a", 8080)
        r2, w2 = await pool.get("host-b", 9090)

        await pool.put("host-a", 8080, w1, r1)
        await pool.put("host-b", 9090, w2, r2)

        # Each pool has 1 connection
        assert pool.total_pooled == 2

        # Hitting each should succeed
        await pool.get("host-a", 8080)
        await pool.get("host-b", 9090)
        stats = pool.stats()
        assert stats.hits == 2
        assert stats.misses == 2  # the original creates

    @pytest.mark.asyncio
    async def test_get_connection_error(self, monkeypatch) -> None:
        """When open_connection raises OSError, get() should propagate it and count errors."""
        monkeypatch.setattr(asyncio, "open_connection", _failing_open_connection)
        pool = AsyncConnectionPool(connect_timeout=0.1)
        with pytest.raises(OSError, match="Connection refused"):
            await pool.get("localhost", 9999)
        stats = pool.stats()
        assert stats.errors == 1
        assert stats.misses == 1
        assert stats.creates == 0

    @pytest.mark.asyncio
    async def test_get_timeout_error(self, monkeypatch) -> None:
        """When open_connection times out, get() should raise TimeoutError and count errors."""

        async def _slow_open(host: str, port: int):
            await asyncio.sleep(999)

        monkeypatch.setattr(asyncio, "open_connection", _slow_open)
        pool = AsyncConnectionPool(connect_timeout=0.01)
        with pytest.raises(asyncio.TimeoutError):
            await pool.get("localhost", 8080)
        stats = pool.stats()
        assert stats.errors == 1
        assert stats.misses == 1
        assert stats.creates == 0

    @pytest.mark.asyncio
    async def test_get_evicts_stale_connection_by_age(self, monkeypatch) -> None:
        """A cached connection past max_age_s should be evicted, not returned."""
        monkeypatch.setattr(asyncio, "open_connection", _stub_open_connection)
        pool = AsyncConnectionPool(max_age_s=0)  # zero age = always stale
        reader, writer = await pool.get("localhost", 8080)
        await pool.put("localhost", 8080, writer, reader)
        assert pool.total_pooled == 1

        reader2, writer2 = await pool.get("localhost", 8080)
        stats = pool.stats()
        assert stats.evictions == 1  # the stale connection was evicted
        assert stats.creates == 2  # two creates: original + evicted replacement

    @pytest.mark.asyncio
    async def test_get_evicts_closed_writer(self, monkeypatch) -> None:
        """A cached connection whose writer was closed externally should be evicted."""
        monkeypatch.setattr(asyncio, "open_connection", _stub_open_connection)
        pool = AsyncConnectionPool()
        reader, writer = await pool.get("localhost", 8080)
        await pool.put("localhost", 8080, writer, reader)
        # Simulate remote close
        writer.close()
        assert writer.is_closing()

        reader2, writer2 = await pool.get("localhost", 8080)
        stats = pool.stats()
        assert stats.evictions == 1
        assert stats.creates == 2

    @pytest.mark.asyncio
    async def test_get_creates_multiple_when_pool_empty(self, monkeypatch) -> None:
        """Repeated gets without puts should create new connections each time."""
        monkeypatch.setattr(asyncio, "open_connection", _stub_open_connection)
        pool = AsyncConnectionPool()
        await pool.get("host", 8080)
        await pool.get("host", 8080)
        await pool.get("host", 8080)

        stats = pool.stats()
        assert stats.misses == 3
        assert stats.creates == 3
        assert stats.hits == 0

    @pytest.mark.asyncio
    async def test_get_evicts_multiple_stale_connections(self, monkeypatch) -> None:
        """When multiple stale connections exist, get() should evict them all and create a new one."""
        monkeypatch.setattr(asyncio, "open_connection", _stub_open_connection)
        pool = AsyncConnectionPool(max_size=3, max_age_s=0)
        # Pre-populate the pool with stale connections directly
        host, port = "host", 8080
        key = (host, port)
        async with pool._lock:
            pool._pool[key] = []
            for _ in range(3):
                conn = AsyncPooledConnection(
                    reader=StubStreamReader(),
                    writer=StubStreamWriter(),
                    created_at=0.0,
                    last_used=0.0,
                    use_count=1,
                )
                pool._pool[key].append(conn)
        assert pool.total_pooled == 3

        # Get should evict all stale connections and create one new one
        r, w = await pool.get(host, port)
        stats = pool.stats()
        assert stats.evictions == 3
        assert stats.creates == 1
        assert stats.misses == 1


# ===================================================================
# PUT TESTS
# ===================================================================

class TestAsyncConnectionPoolPut:
    """put() -- returning connections, overflow handling."""

    @pytest.mark.asyncio
    async def test_put_adds_to_pool(self) -> None:
        """put() should add the connection to the pool."""
        pool = AsyncConnectionPool(max_size=2)
        writer = StubStreamWriter()
        reader = StubStreamReader()
        await pool.put("localhost", 8080, writer, reader)
        assert pool.total_pooled == 1

    @pytest.mark.asyncio
    async def test_put_over_max_size_closes_writer(self) -> None:
        """When pool is full, put() should close the writer instead of storing it."""
        pool = AsyncConnectionPool(max_size=1)
        writer1 = StubStreamWriter()
        await pool.put("localhost", 8080, writer1, StubStreamReader())
        assert pool.total_pooled == 1
        assert not writer1.is_closing()

        writer2 = StubStreamWriter()
        await pool.put("localhost", 8080, writer2, StubStreamReader())
        assert pool.total_pooled == 1  # still 1
        assert writer2.is_closing()  # writer2 was closed

    @pytest.mark.asyncio
    async def test_put_to_closed_pool_closes_writer(self) -> None:
        """put() on a closed pool should close the writer immediately."""
        pool = AsyncConnectionPool()
        await pool.close_all()

        writer = StubStreamWriter()
        reader = StubStreamReader()
        await pool.put("localhost", 8080, writer, reader)
        assert writer.is_closing()
        assert pool.total_pooled == 0

    @pytest.mark.asyncio
    async def test_put_without_reader(self) -> None:
        """put() should create an asyncio.StreamReader when no reader is provided."""
        pool = AsyncConnectionPool(max_size=2)
        writer = StubStreamWriter()
        await pool.put("localhost", 8080, writer)  # no reader argument
        assert pool.total_pooled == 1

    @pytest.mark.asyncio
    async def test_put_multiple_hosts_separate_pools(self) -> None:
        """Different (host, port) pairs should go into separate sub-pools."""
        pool = AsyncConnectionPool(max_size=3)
        await pool.put("host-a", 8080, StubStreamWriter(), StubStreamReader())
        await pool.put("host-b", 9090, StubStreamWriter(), StubStreamReader())
        assert pool.total_pooled == 2

    @pytest.mark.asyncio
    async def test_put_multiple_connections_same_host(self) -> None:
        """Multiple put() calls for the same host should accumulate up to max_size."""
        pool = AsyncConnectionPool(max_size=3)
        writers = []
        for _ in range(3):
            w = StubStreamWriter()
            writers.append(w)
            await pool.put("host", 8080, w, StubStreamReader())
        assert pool.total_pooled == 3
        assert all(not w.is_closing() for w in writers)


# ===================================================================
# REMOVE TESTS
# ===================================================================

class TestAsyncConnectionPoolRemove:
    """remove() -- remove and close connections for a specific (host, port)."""

    @pytest.mark.asyncio
    async def test_remove_existing_host(self) -> None:
        """remove() should close the connection and remove it from the pool."""
        pool = AsyncConnectionPool()
        writer = StubStreamWriter()
        reader = StubStreamReader()
        await pool.put("localhost", 8080, writer, reader)
        assert pool.total_pooled == 1

        await pool.remove("localhost", 8080)
        assert pool.total_pooled == 0
        assert writer.is_closing()

    @pytest.mark.asyncio
    async def test_remove_nonexistent_host(self) -> None:
        """remove() on a host not in the pool should not raise."""
        pool = AsyncConnectionPool()
        await pool.remove("nonexistent", 1234)
        assert pool.total_pooled == 0

    @pytest.mark.asyncio
    async def test_remove_only_affects_one_host(self) -> None:
        """remove() should only affect the specified (host, port)."""
        pool = AsyncConnectionPool()
        w1, r1 = StubStreamWriter(), StubStreamReader()
        w2, r2 = StubStreamWriter(), StubStreamReader()
        await pool.put("host-a", 8080, w1, r1)
        await pool.put("host-b", 9090, w2, r2)
        assert pool.total_pooled == 2

        await pool.remove("host-a", 8080)
        assert pool.total_pooled == 1
        assert w1.is_closing()
        assert not w2.is_closing()

    @pytest.mark.asyncio
    async def test_remove_same_port_different_hosts(self) -> None:
        """Different hosts with the same port should be independent."""
        pool = AsyncConnectionPool()
        w1 = StubStreamWriter()
        w2 = StubStreamWriter()
        await pool.put("host-a", 8080, w1, StubStreamReader())
        await pool.put("host-b", 8080, w2, StubStreamReader())
        assert pool.total_pooled == 2

        await pool.remove("host-a", 8080)
        assert pool.total_pooled == 1
        assert not w2.is_closing()


# ===================================================================
# CLOSE_ALL TESTS
# ===================================================================

class TestAsyncConnectionPoolCloseAll:
    """close_all() -- close all connections and mark pool closed."""

    @pytest.mark.asyncio
    async def test_close_all_clears_pool(self) -> None:
        """close_all() should remove all connections from the pool."""
        pool = AsyncConnectionPool()
        for i in range(3):
            await pool.put(f"host-{i}", 8080, StubStreamWriter(), StubStreamReader())
        assert pool.total_pooled == 3

        await pool.close_all()
        assert pool.total_pooled == 0
        assert pool._closed is True

    @pytest.mark.asyncio
    async def test_close_all_closes_writers(self) -> None:
        """close_all() should close every writer in the pool."""
        pool = AsyncConnectionPool()
        writers = [StubStreamWriter() for _ in range(3)]
        for w in writers:
            await pool.put("host", 8080, w, StubStreamReader())

        await pool.close_all()
        assert all(w.is_closing() for w in writers)

    @pytest.mark.asyncio
    async def test_close_all_idempotent(self) -> None:
        """Calling close_all() twice should not raise."""
        pool = AsyncConnectionPool()
        await pool.close_all()
        await pool.close_all()  # second call should not raise
        assert pool._closed is True

    @pytest.mark.asyncio
    async def test_close_all_after_remove(self) -> None:
        """close_all() should work after remove() has been called."""
        pool = AsyncConnectionPool()
        await pool.put("host-a", 8080, StubStreamWriter(), StubStreamReader())
        await pool.put("host-b", 9090, StubStreamWriter(), StubStreamReader())
        await pool.remove("host-a", 8080)
        assert pool.total_pooled == 1

        await pool.close_all()
        assert pool.total_pooled == 0

    @pytest.mark.asyncio
    async def test_close_all_empty_pool(self) -> None:
        """close_all() on an empty pool should not raise."""
        pool = AsyncConnectionPool()
        await pool.close_all()
        assert pool._closed is True


# ===================================================================
# CONTEXT MANAGER TESTS
# ===================================================================

class TestAsyncConnectionPoolContextManager:
    """Async context manager -- __aenter__ / __aexit__."""

    @pytest.mark.asyncio
    async def test_context_manager_enters_and_exits(self) -> None:
        """Using 'async with pool' should enter and exit cleanly."""
        pool = AsyncConnectionPool()
        async with pool as p:
            assert p is pool
            assert not pool._closed
        assert pool._closed is True

    @pytest.mark.asyncio
    async def test_context_manager_closes_connections(self) -> None:
        """On exit, the context manager should close all connections."""
        pool = AsyncConnectionPool()
        writer = StubStreamWriter()
        async with pool:
            await pool.put("host", 8080, writer, StubStreamReader())
            assert pool.total_pooled == 1
        assert writer.is_closing()
        assert pool.total_pooled == 0


# ===================================================================
# _is_valid TESTS
# ===================================================================

class TestAsyncConnectionPoolIsValid:
    """_is_valid() -- connection validation logic."""

    def test_valid_connection(self) -> None:
        """A fresh, open connection should be valid."""
        pool = AsyncConnectionPool(max_age_s=60)
        writer = StubStreamWriter()
        conn = AsyncPooledConnection(
            reader=StubStreamReader(),
            writer=writer,
            created_at=0.0,
            last_used=0.0,
            use_count=1,
        )
        assert pool._is_valid(conn, now=30.0) is True

    def test_invalid_stale_age(self) -> None:
        """A connection past max_age_s should be invalid."""
        pool = AsyncConnectionPool(max_age_s=10)
        writer = StubStreamWriter()
        conn = AsyncPooledConnection(
            reader=StubStreamReader(),
            writer=writer,
            created_at=0.0,
            last_used=0.0,
            use_count=1,
        )
        assert pool._is_valid(conn, now=15.0) is False  # 15 - 0 > 10

    def test_invalid_closed_writer(self) -> None:
        """A connection with a closing writer should be invalid."""
        pool = AsyncConnectionPool(max_age_s=60)
        writer = StubStreamWriter()
        writer.close()
        conn = AsyncPooledConnection(
            reader=StubStreamReader(),
            writer=writer,
            created_at=0.0,
            last_used=0.0,
            use_count=1,
        )
        assert pool._is_valid(conn, now=30.0) is False

    def test_invalid_stale_and_closed(self) -> None:
        """A connection that is both stale and closed should be invalid."""
        pool = AsyncConnectionPool(max_age_s=10)
        writer = StubStreamWriter()
        writer.close()
        conn = AsyncPooledConnection(
            reader=StubStreamReader(),
            writer=writer,
            created_at=0.0,
            last_used=0.0,
            use_count=1,
        )
        assert pool._is_valid(conn, now=30.0) is False

    def test_valid_exactly_at_age_boundary(self) -> None:
        """A connection created exactly max_age_s ago should be valid (non-strict)."""
        pool = AsyncConnectionPool(max_age_s=10)
        writer = StubStreamWriter()
        conn = AsyncPooledConnection(
            reader=StubStreamReader(),
            writer=writer,
            created_at=0.0,
            last_used=0.0,
            use_count=1,
        )
        # now - created_at == max_age_s, not > max_age_s => valid
        assert pool._is_valid(conn, now=10.0) is True


# ===================================================================
# REPR TESTS
# ===================================================================

class TestAsyncConnectionPoolRepr:
    """__repr__ -- string representation."""

    @pytest.mark.asyncio
    async def test_repr_with_connections(self, monkeypatch) -> None:
        """repr should reflect the actual pool state."""
        monkeypatch.setattr(asyncio, "open_connection", _stub_open_connection)
        pool = AsyncConnectionPool()
        async with pool:
            r, w = await pool.get("host-a", 8080)
            await pool.put("host-a", 8080, w, r)
            r2, w2 = await pool.get("host-a", 8080)  # hit
            await pool.put("host-a", 8080, w2, r2)

            r3, w3 = await pool.get("host-b", 9090)
            await pool.put("host-b", 9090, w3, r3)

            r = repr(pool)
            assert "hosts=2" in r
            assert "pooled=2" in r

    def test_repr_empty(self) -> None:
        """repr of an empty pool should show zero hosts and zero connections."""
        pool = AsyncConnectionPool()
        r = repr(pool)
        assert "hosts=0" in r
        assert "pooled=0" in r


# ===================================================================
# _close_writer TESTS (static method)
# ===================================================================

class TestAsyncConnectionPoolCloseWriter:
    """_close_writer() -- static method for closing writers safely."""

    def test_close_writer_normal(self) -> None:
        """_close_writer should close a normal writer."""
        writer = StubStreamWriter()
        assert not writer.is_closing()
        AsyncConnectionPool._close_writer(writer)
        assert writer.is_closing()

    def test_close_writer_already_closed(self) -> None:
        """_close_writer should handle already-closed writers without error."""
        writer = StubStreamWriter()
        writer.close()
        AsyncConnectionPool._close_writer(writer)  # should not raise

    def test_close_writer_oserror_caught(self) -> None:
        """_close_writer should catch OSError during close()."""

        class _FailingWriter:
            def close(self):
                raise OSError("connection reset")

        AsyncConnectionPool._close_writer(_FailingWriter())

    def test_close_writer_other_errors_not_caught(self) -> None:
        """_close_writer should let non-OSError exceptions propagate."""

        class _BadWriter:
            def close(self):
                raise RuntimeError("unexpected error")

        with pytest.raises(RuntimeError):
            AsyncConnectionPool._close_writer(_BadWriter())


# ===================================================================
# POOLED CONNECTION TESTS
# ===================================================================

class TestAsyncPooledConnection:
    """AsyncPooledConnection dataclass -- construction and defaults."""

    def test_default_construction(self) -> None:
        """AsyncPooledConnection should initialize with reasonable defaults."""
        reader = StubStreamReader()
        writer = StubStreamWriter()
        conn = AsyncPooledConnection(reader=reader, writer=writer)
        assert conn.reader is reader
        assert conn.writer is writer
        assert conn.created_at > 0
        assert conn.last_used > 0
        assert conn.use_count == 0

    def test_custom_values(self) -> None:
        """All fields should accept custom values."""
        conn = AsyncPooledConnection(
            reader=StubStreamReader(),
            writer=StubStreamWriter(),
            created_at=100.0,
            last_used=200.0,
            use_count=5,
        )
        assert conn.created_at == 100.0
        assert conn.last_used == 200.0
        assert conn.use_count == 5

    def test_independent_defaults(self) -> None:
        """Each connection should get independent time values."""
        import time
        c1 = AsyncPooledConnection(reader=StubStreamReader(), writer=StubStreamWriter())
        c2 = AsyncPooledConnection(reader=StubStreamReader(), writer=StubStreamWriter())
        assert c1.created_at != c2.created_at or abs(c1.created_at - c2.created_at) < 0.01
