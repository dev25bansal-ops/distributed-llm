"""LRU cache for draft model outputs keyed by prefix hash.

Avoids redundant draft inference when the same prefix has been seen
recently — a common pattern in speculative decoding where the same
prompt prefix appears across requests (e.g. shared system prompts,
chat templates, or repeated user inputs).

Each cache entry stores:
- The generated token IDs.
- The associated log-probabilities (optional).
- Metadata: hit count, creation time, and last access time.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class CachedDraftOutput:
    """A cached draft inference result."""

    tokens: list[int]
    logprobs: list[float] | None = None
    created_at: float = 0.0
    last_access: float = 0.0
    hit_count: int = 0

    @property
    def age_s(self) -> float:
        return time.time() - self.created_at


class DraftCache:
    """LRU cache for draft model outputs keyed by prefix hash.

    Thread-safe.  The cache evicts the least-recently-used entry when
    the maximum size is exceeded.

    Usage::

        cache = DraftCache(max_entries=5000)

        # Before running a draft:
        prefix_hash = hashlib.sha256(prompt_ids.tobytes()).hexdigest()
        cached = cache.get(prefix_hash)
        if cached is not None:
            # Use cached tokens directly, skip draft forward.
            ...

        # After running a draft:
        cache.put(prefix_hash, draft_tokens, draft_logprobs)
    """

    def __init__(self, max_entries: int = 10_000):
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._max_entries = max_entries
        self._store: OrderedDict[str, CachedDraftOutput] = OrderedDict()
        self._lock = threading.Lock()
        self._total_hits = 0
        self._total_misses = 0
        self._total_puts = 0
        self._total_evictions = 0

        logger.info(
            "DraftCache initialized (max_entries={})", max_entries,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, prefix_hash: str) -> CachedDraftOutput | None:
        """Look up a cached draft output by *prefix_hash*.

        Returns ``None`` if no entry exists.  On a cache hit the entry's
        ``last_access`` and ``hit_count`` are updated and the entry is
        moved to the end of the LRU order.
        """
        with self._lock:
            entry = self._store.get(prefix_hash)
            if entry is None:
                self._total_misses += 1
                return None

            # Move to end (most recently used).
            self._store.move_to_end(prefix_hash)
            entry.last_access = time.time()
            entry.hit_count += 1
            self._total_hits += 1
            # Return a shallow copy so callers cannot mutate the cache.
            return CachedDraftOutput(
                tokens=list(entry.tokens),
                logprobs=list(entry.logprobs) if entry.logprobs is not None else None,
                created_at=entry.created_at,
                last_access=entry.last_access,
                hit_count=entry.hit_count,
            )

    def put(
        self,
        prefix_hash: str,
        tokens: list[int],
        logprobs: list[float] | None = None,
    ) -> None:
        """Insert or update a cache entry.

        If the key already exists it is overwritten and moved to the
        most-recently-used position.  If the cache is at capacity the
        least-recently-used entry is evicted first.
        """
        with self._lock:
            now = time.time()

            if prefix_hash in self._store:
                # Update existing entry.
                entry = self._store[prefix_hash]
                entry.tokens = list(tokens)
                entry.logprobs = list(logprobs) if logprobs is not None else None
                entry.last_access = now
                self._store.move_to_end(prefix_hash)
                self._total_puts += 1
                return

            # Evict if at capacity.
            if len(self._store) >= self._max_entries:
                self._evict_one()

            self._store[prefix_hash] = CachedDraftOutput(
                tokens=list(tokens),
                logprobs=list(logprobs) if logprobs is not None else None,
                created_at=now,
                last_access=now,
                hit_count=0,
            )
            self._total_puts += 1

    def invalidate(self, prefix_hash: str) -> bool:
        """Remove a specific entry from the cache.

        Returns:
            True if the entry existed and was removed, False otherwise.
        """
        with self._lock:
            if prefix_hash in self._store:
                del self._store[prefix_hash]
                logger.debug("Invalidated cache entry {!r}", prefix_hash)
                return True
            return False

    def clear(self) -> int:
        """Remove all entries from the cache.

        Returns:
            The number of entries that were removed.
        """
        with self._lock:
            count = len(self._store)
            self._store.clear()
            if count:
                logger.info("Cleared {} entries from DraftCache", count)
            return count

    def evict_expired(self, max_age_s: float = 300.0) -> int:
        """Evict entries older than *max_age_s*.

        Returns:
            Number of entries evicted.
        """
        cutoff = time.time() - max_age_s
        with self._lock:
            stale = [
                k for k, v in self._store.items()
                if v.created_at < cutoff
            ]
            for k in stale:
                del self._store[k]
            if stale:
                logger.debug(
                    "Evicted {} expired entries (age > {}s)", len(stale), max_age_s,
                )
            return len(stale)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Current number of entries in the cache."""
        with self._lock:
            return len(self._store)

    @property
    def stats(self) -> dict[str, Any]:
        """Return cumulative cache statistics."""
        with self._lock:
            total_lookups = self._total_hits + self._total_misses
            return {
                "max_entries": self._max_entries,
                "current_entries": len(self._store),
                "total_hits": self._total_hits,
                "total_misses": self._total_misses,
                "total_puts": self._total_puts,
                "total_evictions": self._total_evictions,
                "hit_rate": round(
                    self._total_hits / max(total_lookups, 1), 4,
                ),
            }

    def entries(self) -> list[dict[str, Any]]:
        """Return metadata for all cache entries (no token data)."""
        with self._lock:
            return [
                {
                    "prefix_hash": k,
                    "tokens_len": len(v.tokens),
                    "has_logprobs": v.logprobs is not None,
                    "created_at": v.created_at,
                    "last_access": v.last_access,
                    "hit_count": v.hit_count,
                    "age_s": round(v.age_s, 1),
                }
                for k, v in self._store.items()
            ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_one(self) -> None:
        """Evict the least-recently-used entry.

        Assumes the lock is held.
        """
        if not self._store:
            return
        _key, _value = self._store.popitem(last=False)  # FIFO = LRU
        self._total_evictions += 1
