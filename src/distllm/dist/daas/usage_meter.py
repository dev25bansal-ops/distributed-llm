"""Per-tenant usage metering with token counting and SQLite persistence.

Tracks prompt tokens, completion tokens, request counts, and duration
for each tenant.  Supports reporting usage since a given timestamp and
persisting data to SQLite for durability.

Usage::

    meter = UsageMeter()
    meter.record_usage("tenant-a", prompt_tokens=150, completion_tokens=50, duration_ms=1200)

    report = meter.get_usage("tenant-a", since_timestamp=time.time() - 3600)
    all_usage = meter.get_all_usage()

    # With SQLite persistence:
    meter = UsageMeter(db_path="/data/usage.db")
    meter.record_usage("tenant-b", prompt_tokens=200, completion_tokens=80, duration_ms=900)
    meter.close()
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class UsageRecord:
    """A single usage record for one request."""

    tenant_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: float = 0.0
    timestamp: float = 0.0
    request_id: str = ""


@dataclass
class TenantUsage:
    """Aggregated usage for a tenant."""

    tenant_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_duration_ms / max(self.request_count, 1)

    @property
    def avg_prompt_tokens(self) -> float:
        return self.prompt_tokens / max(self.request_count, 1)

    @property
    def avg_completion_tokens(self) -> float:
        return self.completion_tokens / max(self.request_count, 1)


class UsageMeter:
    """Per-tenant usage meter with optional SQLite persistence.

    Thread-safe.  When *db_path* is provided, every ``record_usage()`` call
    writes to the SQLite database immediately.  The ``get_usage()`` and
    ``get_all_usage()`` methods always read from the in-memory aggregation,
    which is synchronized with the DB at construction time.

    The SQLite schema stores raw records so that arbitrary time-range queries
    are possible without pre-aggregating.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._lock = threading.Lock()

        # In-memory per-tenant rolling counters (lifetime totals).
        self._usage: dict[str, TenantUsage] = {}

        # Raw records for time-range queries (kept in memory, also in DB).
        self._records: list[UsageRecord] = []
        self._max_records: int = 100_000  # soft cap to bound memory

        # SQLite persistence.
        self._db_path: Path | None = None
        self._conn: sqlite3.Connection | None = None
        if db_path is not None:
            self._db_path = Path(db_path)
            self._open_db()
            self._load_from_db()

    # ── Database lifecycle ────────────────────────────────────────────

    def _open_db(self) -> None:
        """Open (or create) the SQLite database and ensure the schema exists."""
        if self._db_path is None:
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS usage_records (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id   TEXT NOT NULL,
                prompt_tokens    INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                duration_ms REAL NOT NULL DEFAULT 0.0,
                timestamp   REAL NOT NULL,
                request_id  TEXT DEFAULT ''
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_tenant_ts
            ON usage_records (tenant_id, timestamp)
        """)
        self._conn.commit()

    def _load_from_db(self) -> None:
        """Load existing DB records into in-memory aggregates."""
        if self._conn is None:
            return
        try:
            cursor = self._conn.execute(
                "SELECT tenant_id, prompt_tokens, completion_tokens, duration_ms "
                "FROM usage_records"
            )
            for tenant_id, pt, ct, dur in cursor:
                agg = self._usage.setdefault(
                    tenant_id,
                    TenantUsage(tenant_id=tenant_id),
                )
                agg.prompt_tokens += pt
                agg.completion_tokens += ct
                agg.total_tokens += pt + ct
                agg.request_count += 1
                agg.total_duration_ms += dur
        except sqlite3.Error as exc:
            logger.warning(f"Failed to load usage from DB: {exc}")

    def _query_window_db(
        self, tenant_id: str, since_timestamp: float
    ) -> tuple[int, int, float, int]:
        """Aggregate usage for *tenant_id* since *since_timestamp* from SQLite.

        Used when the in-memory record list is at its cap, so records that
        spilled to the DB are not silently dropped from time-window queries
        (F-024).  Returns (prompt_tokens, completion_tokens, duration_ms, count).
        """
        if self._conn is None:
            return (0, 0, 0.0, 0)
        try:
            cursor = self._conn.execute(
                "SELECT COALESCE(SUM(prompt_tokens), 0), "
                "COALESCE(SUM(completion_tokens), 0), "
                "COALESCE(SUM(duration_ms), 0), COUNT(*) "
                "FROM usage_records WHERE tenant_id = ? AND timestamp >= ?",
                (tenant_id, since_timestamp),
            )
            row = cursor.fetchone()
            if not row:
                return (0, 0, 0.0, 0)
            pt, ct, dur, cnt = row
            return (int(pt), int(ct), float(dur), int(cnt))
        except sqlite3.Error as exc:
            logger.warning(f"Failed to query usage window from DB: {exc}")
            return (0, 0, 0.0, 0)

    def close(self) -> None:
        """Close the SQLite connection, if open."""
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error as exc:
                logger.warning(f"Error closing usage DB: {exc}")
            finally:
                self._conn = None

    def __del__(self) -> None:
        self.close()

    # ── Recording ─────────────────────────────────────────────────────

    def record_usage(
        self,
        tenant_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: float,
        request_id: str = "",
    ) -> None:
        """Record usage for a single request from *tenant_id*.

        Updates in-memory aggregates and, if SQLite is configured, persists
        the raw record immediately.
        """
        if prompt_tokens < 0 or completion_tokens < 0 or duration_ms < 0:
            logger.warning(
                f"Negative usage values for {tenant_id}: "
                f"pt={prompt_tokens}, ct={completion_tokens}, dur={duration_ms}"
            )
            return

        now = time.time()
        total_tokens = prompt_tokens + completion_tokens

        with self._lock:
            agg = self._usage.setdefault(
                tenant_id,
                TenantUsage(tenant_id=tenant_id),
            )
            agg.prompt_tokens += prompt_tokens
            agg.completion_tokens += completion_tokens
            agg.total_tokens += total_tokens
            agg.request_count += 1
            agg.total_duration_ms += duration_ms

            # Keep a bounded list of raw records for time-range queries.
            if len(self._records) < self._max_records:
                self._records.append(
                    UsageRecord(
                        tenant_id=tenant_id,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        duration_ms=duration_ms,
                        timestamp=now,
                        request_id=request_id,
                    )
                )

        # Persist to SQLite (outside the lock to avoid holding it during I/O).
        if self._conn is not None:
            try:
                self._conn.execute(
                    "INSERT INTO usage_records "
                    "(tenant_id, prompt_tokens, completion_tokens, duration_ms, timestamp, request_id) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (tenant_id, prompt_tokens, completion_tokens, duration_ms, now, request_id),
                )
                self._conn.commit()
            except sqlite3.Error as exc:
                logger.warning(f"Failed to persist usage record: {exc}")

    # ── Querying ──────────────────────────────────────────────────────

    def get_usage(self, tenant_id: str, since_timestamp: float = 0.0) -> TenantUsage | None:
        """Return aggregated usage for *tenant_id* since *since_timestamp*.

        If *since_timestamp* is ``0.0`` (default), returns the lifetime
        aggregate.  Returns ``None`` if the tenant has no records.
        """
        with self._lock:
            if since_timestamp <= 0.0:
                return self._usage.get(tenant_id)

            # Aggregate from raw records in the time window.
            total_pt = 0
            total_ct = 0
            total_dur = 0.0
            count = 0

            records = self._records
            # If the in-memory list is at its cap, newer records may have
            # spilled to SQLite only — the DB is the authoritative source for
            # the window (it holds every record). Query it for the whole window
            # rather than under-counting from _records alone (F-024).
            if len(records) >= self._max_records and self._conn is not None:
                db_pt, db_ct, db_dur, db_count = self._query_window_db(
                    tenant_id, since_timestamp
                )
                if db_count == 0:
                    return None
                agg = TenantUsage(tenant_id=tenant_id)
                agg.prompt_tokens = db_pt
                agg.completion_tokens = db_ct
                agg.total_tokens = db_pt + db_ct
                agg.request_count = db_count
                agg.total_duration_ms = db_dur
                return agg

            # Fast path: records are roughly in insertion order (newest last).
            # Walk backwards and stop when records are older than the cutoff.
            for i in range(len(records) - 1, -1, -1):
                rec = records[i]
                if rec.tenant_id != tenant_id:
                    continue
                if rec.timestamp < since_timestamp:
                    break  # older records are before this one in insertion order
                total_pt += rec.prompt_tokens
                total_ct += rec.completion_tokens
                total_dur += rec.duration_ms
                count += 1

            if count == 0:
                return None

            agg = TenantUsage(tenant_id=tenant_id)
            agg.prompt_tokens = total_pt
            agg.completion_tokens = total_ct
            agg.total_tokens = total_pt + total_ct
            agg.request_count = count
            agg.total_duration_ms = total_dur
            return agg

    def get_all_usage(self) -> dict[str, TenantUsage]:
        """Return lifetime aggregated usage for all tenants.

        Returns a dict keyed by tenant ID.
        """
        with self._lock:
            return dict(self._usage)

    # ── Administrative ────────────────────────────────────────────────

    def reset(self, tenant_id: str | None = None) -> None:
        """Reset usage counters.

        If *tenant_id* is provided, only that tenant's counters are reset.
        Otherwise all counters are reset.
        """
        with self._lock:
            if tenant_id is not None:
                self._usage.pop(tenant_id, None)
                self._records = [r for r in self._records if r.tenant_id != tenant_id]
            else:
                self._usage.clear()
                self._records.clear()

        # Also reset in DB.
        if self._conn is not None:
            try:
                if tenant_id is not None:
                    self._conn.execute("DELETE FROM usage_records WHERE tenant_id = ?", (tenant_id,))
                else:
                    self._conn.execute("DELETE FROM usage_records")
                self._conn.commit()
            except sqlite3.Error as exc:
                logger.warning(f"Failed to reset usage DB: {exc}")

    def stats(self) -> dict[str, Any]:
        """Return summary statistics for the meter itself."""
        with self._lock:
            total_requests = sum(u.request_count for u in self._usage.values())
            total_prompt = sum(u.prompt_tokens for u in self._usage.values())
            total_completion = sum(u.completion_tokens for u in self._usage.values())
            return {
                "tenants": len(self._usage),
                "total_requests": total_requests,
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
                "in_memory_records": len(self._records),
                "db_path": str(self._db_path) if self._db_path else None,
            }
