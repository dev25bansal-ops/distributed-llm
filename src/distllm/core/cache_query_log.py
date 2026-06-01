"""N4: Audit trail for cache operations.

Writes structured JSONL logs for every cache lookup/store operation.
Enables offline analysis, pattern tuning, and debugging.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class CacheQueryLog:
    """A single cache operation log entry."""
    request_id: str
    timestamp: float = field(default_factory=time.time)
    operation: str = "lookup"  # "lookup", "store", "evict"
    tokens_prefix: str = ""  # First 8 tokens, truncated
    token_count: int = 0
    tier_hits: list[str] = field(default_factory=list)
    total_latency_ms: float = 0.0
    hit: bool = False
    match_length: int = 0
    metadata: dict = field(default_factory=dict)


class CacheQueryLogger:
    """Structured JSONL logger for cache operations.

    Writes one JSON line per operation for offline analysis.
    """

    def __init__(self, log_path: str | None = None, max_entries: int = 100000):
        self._log_path = Path(log_path) if log_path else None
        self._max_entries = max_entries
        self._entries: list[CacheQueryLog] = []
        self._file = None

        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self._log_path, "a")

    def log_lookup(
        self,
        request_id: str,
        tokens: list[int],
        hit: bool,
        match_length: int = 0,
        tier_hits: list[str] | None = None,
        latency_ms: float = 0.0,
        metadata: dict | None = None,
    ) -> None:
        """Log a cache lookup operation.

        Args:
            request_id: Unique request identifier.
            tokens: Token IDs (truncated to first 8 for privacy).
            hit: Whether the lookup was a cache hit.
            match_length: Length of the prefix match.
            tier_hits: Tiers that were hit.
            latency_ms: Lookup latency in milliseconds.
            metadata: Additional metadata.
        """
        entry = CacheQueryLog(
            request_id=request_id,
            operation="lookup",
            tokens_prefix=str(tokens[:8]),
            token_count=len(tokens),
            tier_hits=tier_hits or [],
            total_latency_ms=latency_ms,
            hit=hit,
            match_length=match_length,
            metadata=metadata or {},
        )
        self._write(entry)

    def log_store(
        self,
        request_id: str,
        tokens: list[int],
        tier: str = "local",
        latency_ms: float = 0.0,
        metadata: dict | None = None,
    ) -> None:
        """Log a cache store operation."""
        entry = CacheQueryLog(
            request_id=request_id,
            operation="store",
            tokens_prefix=str(tokens[:8]),
            token_count=len(tokens),
            tier_hits=[tier],
            total_latency_ms=latency_ms,
            hit=True,
            match_length=len(tokens),
            metadata=metadata or {},
        )
        self._write(entry)

    def log_evict(
        self,
        request_id: str,
        tokens: list[int],
        tier: str = "local",
        reason: str = "lru",
        metadata: dict | None = None,
    ) -> None:
        """Log a cache eviction."""
        entry = CacheQueryLog(
            request_id=request_id,
            operation="evict",
            tokens_prefix=str(tokens[:8]),
            token_count=len(tokens),
            tier_hits=[tier],
            hit=False,
            metadata={"reason": reason, **(metadata or {})},
        )
        self._write(entry)

    def _write(self, entry: CacheQueryLog) -> None:
        """Write a log entry."""
        self._entries.append(entry)

        # Trim if over limit
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

        # Write to file
        if self._file:
            try:
                line = json.dumps(asdict(entry), default=str)
                self._file.write(line + "\n")
                self._file.flush()
            except Exception as e:
                logger.debug(f"Failed to write cache log: {e}")

    def get_entries(
        self,
        operation: str | None = None,
        hit_only: bool = False,
        limit: int = 100,
    ) -> list[CacheQueryLog]:
        """Get log entries with optional filtering.

        Args:
            operation: Filter by operation type.
            hit_only: Only return hits.
            limit: Max entries to return.

        Returns:
            List of matching CacheQueryLog entries.
        """
        entries = self._entries
        if operation:
            entries = [e for e in entries if e.operation == operation]
        if hit_only:
            entries = [e for e in entries if e.hit]
        return entries[-limit:]

    def get_stats(self) -> dict:
        """Get aggregate statistics from the log."""
        if not self._entries:
            return {"total_operations": 0}

        lookups = [e for e in self._entries if e.operation == "lookup"]
        hits = [e for e in lookups if e.hit]

        return {
            "total_operations": len(self._entries),
            "lookups": len(lookups),
            "hits": len(hits),
            "hit_rate": len(hits) / max(len(lookups), 1),
            "avg_latency_ms": sum(e.total_latency_ms for e in lookups) / max(len(lookups), 1),
            "avg_match_length": sum(e.match_length for e in hits) / max(len(hits), 1),
        }

    def close(self) -> None:
        """Close the log file."""
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()
