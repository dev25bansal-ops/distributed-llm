"""Cache index for P2P gossip-based cache lookups.

Re-exports CacheIndex from dist module and adds CacheIndexEntry for
gossip protocol integration.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from distllm.dist.cache import CacheIndex as _DistCacheIndex

__all__ = ["CacheIndex", "CacheIndexEntry"]


@dataclass
class CacheIndexEntry:
    """Metadata for a single cached KV entry in the gossip index."""
    key: str
    owner: str
    timestamp: float = field(default_factory=time.time)
    ttl: float = 3600.0
    num_tokens: int = 0
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.timestamp) > self.ttl


class CacheIndex(_DistCacheIndex):
    """Prefix-cache index that also STORES key→owner entries for gossip.

    Inherits token-hash helpers from the dist implementation and adds a
    thread-safe entry store used by :mod:`distllm.dist.p2p.gossip` to
    advertise/locate cached prefixes.
    """

    def __init__(self, default_ttl: float = 3600.0) -> None:
        super().__init__()
        self._entries: dict[str, CacheIndexEntry] = {}
        self._lock = threading.Lock()
        self.default_ttl = default_ttl

    def add(
        self, key: str, owner: str, num_tokens: int = 0, ttl: float | None = None,
    ) -> CacheIndexEntry:
        """Register (or refresh) ownership of *key*.

        ``ttl`` overrides the index default (e.g. the gossip protocol's
        cache_ttl so expiry matches the protocol's view).
        """
        entry = CacheIndexEntry(
            key=key, owner=owner, ttl=ttl if ttl is not None else self.default_ttl,
            num_tokens=num_tokens,
        )
        with self._lock:
            self._entries[key] = entry
        return entry

    def lookup(self, key: str) -> CacheIndexEntry | None:
        """Return the live entry for *key* (pruning it if expired)."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.is_expired:
                del self._entries[key]
                return None
            if entry is not None:
                entry.hit_count += 1
            return entry

    def remove(self, key: str) -> bool:
        with self._lock:
            return self._entries.pop(key, None) is not None

    def prune_expired(self) -> int:
        """Drop expired entries; returns how many were removed."""
        now = time.time()
        with self._lock:
            expired = [k for k, e in self._entries.items()
                       if (now - e.timestamp) > e.ttl]
            for k in expired:
                del self._entries[k]
            return len(expired)

    def all_entries(self) -> dict[str, CacheIndexEntry]:
        with self._lock:
            return dict(self._entries)
