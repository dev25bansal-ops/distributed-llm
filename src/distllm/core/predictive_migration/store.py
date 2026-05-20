from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContentEntry:
    """A content-addressed KV cache entry."""
    content_hash: str
    prefix_tokens: tuple[int, ...]
    kv_data: Any
    ref_count: int = 1
    size_bytes: int = 0
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)


class ContentAddressableStore:
    """KV cache store indexed by content hash (SHA-256 of token prefix).

    Deduplicates cache entries: identical prefixes share a single KV cache
    entry via reference counting. Integrates with RadixTreeCache for fast
    prefix lookup and with KVCacheTransfer for serialization.

    Usage:
        store = ContentAddressableStore()
        h = store.store([101, 205, ...], kv_tensors)
        data = store.get(h)
        match_len, data = store.lookup_by_prefix([101, 205, 309, ...])
    """

    def __init__(self, max_entries: int = 10000, default_ttl_secs: float = 3600.0):
        self._entries: dict[str, ContentEntry] = {}
        self._prefix_to_hash: dict[tuple[int, ...], str] = {}
        self._max_entries = max_entries
        self._default_ttl = default_ttl_secs
        self._lock = threading.Lock()
        self._total_lookups: int = 0
        self._total_hits: int = 0

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def store(
        self,
        prefix_tokens: list[int],
        kv_data: Any,
        ttl_secs: float | None = None,
    ) -> str:
        """Store KV cache data keyed by token prefix hash.

        If the same prefix is already stored, increments the reference
        count instead of duplicating data.

        Returns the content hash.
        """
        prefix_key = tuple(prefix_tokens)
        content_hash = self._hash_tokens(prefix_tokens)

        with self._lock:
            existing = self._entries.get(content_hash)
            if existing is not None:
                existing.ref_count += 1
                existing.last_access = time.time()
                return content_hash

            if len(self._entries) >= self._max_entries:
                self._evict_lru()

            entry = ContentEntry(
                content_hash=content_hash,
                prefix_tokens=prefix_key,
                kv_data=kv_data,
                size_bytes=self._estimate_size(kv_data),
            )
            self._entries[content_hash] = entry
            self._prefix_to_hash[prefix_key] = content_hash

        return content_hash

    def get(self, content_hash: str) -> Any:
        """Retrieve KV cache data by content hash."""
        with self._lock:
            self._total_lookups += 1
            entry = self._entries.get(content_hash)
            if entry is None:
                return None
            if self._is_expired(entry):
                self._entries.pop(content_hash, None)
                self._prefix_to_hash.pop(entry.prefix_tokens, None)
                return None
            entry.last_access = time.time()
            self._total_hits += 1
            return entry.kv_data

    def release(self, content_hash: str) -> bool:
        """Decrement reference count. If zero, evict the entry.

        Returns True if the entry was evicted.
        """
        with self._lock:
            entry = self._entries.get(content_hash)
            if entry is None:
                return False
            entry.ref_count -= 1
            if entry.ref_count <= 0:
                self._entries.pop(content_hash, None)
                self._prefix_to_hash.pop(entry.prefix_tokens, None)
                return True
            return False

    def lookup_by_prefix(
        self, token_ids: list[int]
    ) -> tuple[int, Any, str | None]:
        """Find the longest matching prefix and its KV cache data.

        Returns (match_len, kv_data, content_hash).
        match_len is 0 if no match found.
        """
        with self._lock:
            self._total_lookups += 1
            for length in range(len(token_ids), 0, -1):
                prefix_key = tuple(token_ids[:length])
                content_hash = self._prefix_to_hash.get(prefix_key)
                if content_hash:
                    entry = self._entries.get(content_hash)
                    if entry and not self._is_expired(entry):
                        entry.last_access = time.time()
                        self._total_hits += 1
                        return (length, entry.kv_data, content_hash)

            return (0, None, None)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def sweep_expired(self) -> int:
        now = time.time()
        expired = []
        with self._lock:
            for h, entry in self._entries.items():
                if now - entry.created_at > self._default_ttl:
                    expired.append(h)
            for h in expired:
                entry = self._entries.pop(h, None)
                if entry:
                    self._prefix_to_hash.pop(entry.prefix_tokens, None)
        return len(expired)

    def _evict_lru(self) -> None:
        if not self._entries:
            return
        oldest = min(
            self._entries.values(), key=lambda e: e.last_access
        )
        self._entries.pop(oldest.content_hash, None)
        self._prefix_to_hash.pop(oldest.prefix_tokens, None)

    def _is_expired(self, entry: ContentEntry) -> bool:
        return (time.time() - entry.created_at) > self._default_ttl

    def _hash_tokens(self, token_ids: list[int]) -> str:
        raw = ",".join(str(t) for t in token_ids)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def _estimate_size(self, kv_data: Any) -> int:
        if kv_data is None:
            return 0
        if isinstance(kv_data, (list, tuple)):
            total = 0
            for item in kv_data:
                if hasattr(item, "element_size") and hasattr(item, "numel"):
                    total += item.element_size() * item.numel()
                elif isinstance(item, (list, tuple)):
                    for t in item:
                        if hasattr(t, "element_size") and hasattr(t, "numel"):
                            total += t.element_size() * t.numel()
            return total
        return 0

    def get_entry(self, content_hash: str) -> ContentEntry | None:
        with self._lock:
            return self._entries.get(content_hash)

    def contains(self, prefix_tokens: list[int]) -> bool:
        return tuple(prefix_tokens) in self._prefix_to_hash

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._prefix_to_hash.clear()

    @property
    def hit_rate(self) -> float:
        if self._total_lookups == 0:
            return 0.0
        return self._total_hits / self._total_lookups

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total_size = sum(e.size_bytes for e in self._entries.values())
            total_refs = sum(e.ref_count for e in self._entries.values())
            return {
                "entries": len(self._entries),
                "total_refs": total_refs,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "max_entries": self._max_entries,
                "hit_rate": round(self.hit_rate, 4),
                "total_lookups": self._total_lookups,
                "total_hits": self._total_hits,
                "default_ttl_secs": self._default_ttl,
            }
