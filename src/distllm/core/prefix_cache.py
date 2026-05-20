"""Hash-based LRU prefix cache for token sequences with memory-based limits."""

import time
import threading
from collections import OrderedDict

from distllm.core.cache_eviction import TTLPolicy


# Default memory budget: 512 MiB
_DEFAULT_MEMORY_BUDGET_BYTES = 512 * 1024 * 1024


class PrefixCache:
    """Hash-based LRU prefix cache for token sequences.

    Stores precomputed KV data for common token prefixes (system prompts,
    conversation history). Returns the longest matching cached prefix so
    the caller can skip recomputing those tokens.

    Uses memory-based limits instead of a hard 1024-entry limit.
    Dynamic budget: tracks total bytes of stored KV data and evicts LRU
    entries when the budget is exceeded.

    In pipeline parallelism, each node maintains its own prefix cache
    since KV cache is split across machines.
    """

    _HASH_BASE = 31337
    _HASH_MOD = (1 << 61) - 1

    def __init__(
        self,
        max_entries: int = 0,
        min_prefix_len: int = 16,
        memory_budget_bytes: int = _DEFAULT_MEMORY_BUDGET_BYTES,
        paged_attention_mgr: object | None = None,
        ttl_policy: TTLPolicy | None = None,
    ):
        self.min_prefix_len = min_prefix_len
        self._paged_attention_mgr = paged_attention_mgr
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self._hits = 0
        self._misses = 0

        # Memory-based limits
        self._memory_budget = memory_budget_bytes
        self._total_memory_bytes = 0
        self._max_entries_soft = max_entries or 0  # 0 = unlimited soft cap

        # TTL-based eviction
        self._ttl_policy = ttl_policy
        self._lock = threading.Lock()

    @property
    def max_entries(self) -> int:
        return self._max_entries_soft or len(self._cache) + 1

    @max_entries.setter
    def max_entries(self, value: int) -> None:
        self._max_entries_soft = value

    def _estimate_entry_memory(self, kv_data: dict) -> int:
        """Estimate memory used by a prefix entry's KV data."""
        total = 0
        if isinstance(kv_data, dict):
            for v in kv_data.values():
                if hasattr(v, 'element_size') and hasattr(v, 'numel'):
                    total += v.element_size() * v.numel()
                elif isinstance(v, tuple) and v:
                    for t in v:
                        if hasattr(t, 'element_size') and hasattr(t, 'numel'):
                            total += t.element_size() * t.numel()
        elif isinstance(kv_data, list):
            for t in kv_data:
                if hasattr(t, 'element_size') and hasattr(t, 'numel'):
                    total += t.element_size() * t.numel()
        return total

    def _evict_until_fit(self, needed_bytes: int) -> None:
        """Evict expired entries first, then LRU entries until enough memory is free."""
        # First, evict expired TTL entries
        if self._ttl_policy:
            all_keys = list(self._cache.keys())
            expired = self._ttl_policy.get_expired_keys(all_keys)
            for key in expired:
                if key in self._cache:
                    entry = self._cache.pop(key)
                    entry_bytes = self._estimate_entry_memory(entry.get("kv_data", {}))
                    self._total_memory_bytes = max(0, self._total_memory_bytes - entry_bytes)
                    self._ttl_policy.remove(key)
                    if self._total_memory_bytes + needed_bytes <= self._memory_budget:
                        return

        # Fall back to LRU eviction
        while self._cache and (self._total_memory_bytes + needed_bytes > self._memory_budget):
            _key, entry = self._cache.popitem(last=False)
            entry_bytes = self._estimate_entry_memory(entry.get("kv_data", {}))
            self._total_memory_bytes = max(0, self._total_memory_bytes - entry_bytes)
            if self._ttl_policy:
                self._ttl_policy.remove(_key)

    def lookup(self, token_ids: list[int]) -> tuple[int, dict | None]:
        """Find the longest cached prefix."""
        if len(token_ids) < self.min_prefix_len:
            with self._lock:
                self._misses += 1
            return 0, None

        # Compute rolling hash for the full sequence (fast, no lock needed)
        hashes = []
        running_hash = 0
        for i, tok in enumerate(token_ids):
            running_hash = ((running_hash * self._HASH_BASE) + tok) % self._HASH_MOD
            length = i + 1
            if length >= self.min_prefix_len:
                hashes.append((length, running_hash))

        # Lock held only for dict lookups and the single matching compare
        for length, h in reversed(hashes):
            with self._lock:
                cached = self._cache.get(h)
            if cached is None:
                continue
            if self._ttl_policy and self._ttl_policy.is_expired(h):
                continue
            # Compare without allocating a slice
            cached_tokens = cached["tokens"]
            if len(cached_tokens) == length:
                match = True
                for j in range(length):
                    if cached_tokens[j] != token_ids[j]:
                        match = False
                        break
                if match:
                    with self._lock:
                        self._hits += 1
                        self._cache.move_to_end(h)
                        if self._ttl_policy:
                            self._ttl_policy.record_access(h)
                    return length, cached["kv_data"]

        with self._lock:
            self._misses += 1
        return 0, None

    def store(self, token_ids: list[int], kv_data: dict) -> None:
        """Cache a prefix's KV data."""
        with self._lock:
            if len(token_ids) < self.min_prefix_len:
                return

            h = 0
            for tok in token_ids:
                h = ((h * self._HASH_BASE) + tok) % self._HASH_MOD

            if h in self._cache:
                self._cache.move_to_end(h)
                self._cache[h]["kv_data"] = kv_data
                if self._ttl_policy:
                    self._ttl_policy.record_access(h)
                return

            while len(self._cache) >= self.max_entries:
                self._cache.popitem(last=False)

            self._cache[h] = {
                "tokens": list(token_ids),
                "kv_data": kv_data,
                "stored_at": time.time(),
            }
            entry_bytes = self._estimate_entry_memory(kv_data)
            self._total_memory_bytes += entry_bytes

            if self._ttl_policy:
                self._ttl_policy.record_access(h)

            self._evict_until_fit(0)

    def evict(self, token_ids: list[int]) -> bool:
        """Remove a specific prefix from the cache."""
        with self._lock:
            h = 0
            for tok in token_ids:
                h = ((h * self._HASH_BASE) + tok) % self._HASH_MOD
            if h in self._cache:
                del self._cache[h]
                return True
            return False

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            if self._ttl_policy:
                self._ttl_policy.clear()

    @property
    def hit_rate(self) -> float:
        with self._lock:
            total = self._hits + self._misses
            return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        with self._lock:
            return {
                "prefix_cache_entries": len(self._cache),
                "prefix_cache_max_entries": self.max_entries,
                "prefix_cache_hits": self._hits,
                "prefix_cache_misses": self._misses,
                "prefix_cache_hit_rate": round(self.hit_rate, 4),
                "prefix_cache_memory_bytes": self._total_memory_bytes,
                "prefix_cache_memory_budget": self._memory_budget,
                "prefix_cache_memory_util": round(
                    self._total_memory_bytes / max(self._memory_budget, 1), 4
                ),
            }

    def adjust_memory_budget(self, new_budget_bytes: int) -> None:
        with self._lock:
            self._memory_budget = new_budget_bytes
            self._evict_until_fit(0)
