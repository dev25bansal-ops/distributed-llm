"""Async-native TCP connection pool for health checks.

Eliminates the sync/async split by providing an asyncio-compatible
pool that reuses connections across health check cycles.

Usage::

    async with AsyncConnectionPool(max_size=10) as pool:
        reader, writer = await pool.get("node-1", 8080)
        # ... use connection ...
        await pool.put("node-1", 8080, writer)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class AsyncPooledConnection:
    """A tracked async connection."""
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0


@dataclass
class AsyncPoolStats:
    """Async pool statistics."""
    hits: int = 0
    misses: int = 0
    creates: int = 0
    evictions: int = 0
    errors: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class AsyncConnectionPool:
    """Async TCP connection pool with validation and metrics.

    Args:
        max_size: Maximum idle connections per (host, port).
        connect_timeout: Timeout for creating new connections.
        max_age_s: Maximum connection age before eviction.
    """

    def __init__(
        self,
        max_size: int = 10,
        connect_timeout: float = 5.0,
        max_age_s: float = 300.0,
    ):
        self._max_size = max_size
        self._connect_timeout = connect_timeout
        self._max_age_s = max_age_s

        self._pool: dict[tuple[str, int], list[AsyncPooledConnection]] = {}
        self._lock = asyncio.Lock()
        self._stats = AsyncPoolStats()
        self._closed = False

    async def __aenter__(self) -> AsyncConnectionPool:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close_all()

    async def get(
        self, host: str, port: int,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Get a connection from the pool or create a new one."""
        if self._closed:
            raise OSError("AsyncConnectionPool is closed")

        key = (host, port)
        now = time.time()

        async with self._lock:
            if key in self._pool:
                while self._pool[key]:
                    conn = self._pool[key].pop()
                    if self._is_valid(conn, now):
                        conn.last_used = now
                        conn.use_count += 1
                        self._stats.hits += 1
                        return conn.reader, conn.writer
                    else:
                        self._close_writer(conn.writer)
                        self._stats.evictions += 1

        self._stats.misses += 1
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self._connect_timeout,
            )
            self._stats.creates += 1
            return reader, writer
        except (OSError, asyncio.TimeoutError) as e:
            self._stats.errors += 1
            raise

    async def put(
        self,
        host: str,
        port: int,
        writer: asyncio.StreamWriter,
        reader: asyncio.StreamReader | None = None,
    ) -> None:
        """Return a connection to the pool."""
        if self._closed:
            self._close_writer(writer)
            return

        key = (host, port)
        now = time.time()

        conn = AsyncPooledConnection(
            reader=reader or asyncio.StreamReader(),
            writer=writer,
            created_at=now,
            last_used=now,
            use_count=1,
        )

        async with self._lock:
            if key not in self._pool:
                self._pool[key] = []
            if len(self._pool[key]) < self._max_size:
                self._pool[key].append(conn)
            else:
                self._close_writer(writer)

    async def remove(self, host: str, port: int) -> None:
        """Remove and close all connections for a host:port."""
        key = (host, port)
        async with self._lock:
            conns = self._pool.pop(key, [])
            for conn in conns:
                self._close_writer(conn.writer)

    async def close_all(self) -> None:
        """Close all pooled connections."""
        self._closed = True
        async with self._lock:
            for conns in self._pool.values():
                for conn in conns:
                    self._close_writer(conn.writer)
            self._pool.clear()

    def _is_valid(self, conn: AsyncPooledConnection, now: float) -> bool:
        if now - conn.created_at > self._max_age_s:
            return False
        if conn.writer.is_closing():
            return False
        return True

    @staticmethod
    def _close_writer(writer: asyncio.StreamWriter) -> None:
        try:
            writer.close()
        except OSError:
            pass

    @property
    def total_pooled(self) -> int:
        return sum(len(c) for c in self._pool.values())

    def stats(self) -> AsyncPoolStats:
        return self._stats

    def __repr__(self) -> str:
        return (
            f"AsyncConnectionPool(hosts={len(self._pool)}, "
            f"pooled={self.total_pooled}, "
            f"hit_rate={self._stats.hit_rate:.1%})"
        )
