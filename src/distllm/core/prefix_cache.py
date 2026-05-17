"""Hash-based LRU prefix cache for token sequences with memory-based limits."""

import time
import sys
from collections import OrderedDict
from typing import Any


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
        """Evict LRU entries until enough memory is free."""
        while self._cache and (self._total_memory_bytes + needed_bytes > self._memory_budget):
            _key, entry = self._cache.popitem(last=False)
            entry_bytes = self._estimate_entry_memory(entry.get("kv_data", {}))
            self._total_memory_bytes = max(0, self._total_memory_bytes - entry_bytes)

    def lookup(self, token_ids: list[int]) -> tuple[int, dict | None]:
        """Find the longest cached prefix.

        Uses incremental polynomial hash: computes hash(token_ids[:length])
        for each length from min_prefix_len to len(token_ids) in a single pass.

        Args:
            token_ids: Full sequence of token IDs.

        Returns:
            (matched_len, kv_data) where matched_len is the number of tokens
            that were found in the cache. Returns (0, None) on miss.
        """
        if len(token_ids) < self.min_prefix_len:
            self._misses += 1
            return 0, None

        # Compute rolling polynomial hash for each prefix length
        # Store (length, hash) pairs, then search from longest
        hashes = []
        running_hash = 0
        for i, tok in enumerate(token_ids):
            running_hash = ((running_hash * self._HASH_BASE) + tok) % self._HASH_MOD
            length = i + 1
            if length >= self.min_prefix_len:
                hashes.append((length, running_hash))

        # Search from longest prefix down to min_prefix_len
        for length, h in reversed(hashes):
            if h in self._cache:
                # Verify token match to handle hash collisions
                if self._cache[h]["tokens"] == token_ids[:length]:
                    self._hits += 1
                    self._cache.move_to_end(h)
                    return length, self._cache[h]["kv_data"]

        self._misses += 1
        return 0, None

    def store(self, token_ids: list[int], kv_data: dict) -> None:
        """Cache a prefix's KV data.

        Args:
            token_ids: Token sequence to cache.
            kv_data: Precomputed KV cache data (layer_idx -> (k, v) tensors).
        """
        if len(token_ids) < self.min_prefix_len:
            return

        # Compute polynomial hash for the full prefix
        h = 0
        for tok in token_ids:
            h = ((h * self._HASH_BASE) + tok) % self._HASH_MOD

        if h in self._cache:
            # Update existing entry
            self._cache.move_to_end(h)
            self._cache[h]["kv_data"] = kv_data
            return

        # Evict LRU if at capacity
        while len(self._cache) >= self.max_entries:
            self._cache.popitem(last=False)

        self._cache[h] = {
            "tokens": list(token_ids),
            "kv_data": kv_data,
            "stored_at": time.time(),
        }
        entry_bytes = self._estimate_entry_memory(kv_data)
        self._total_memory_bytes += entry_bytes
        self._evict_until_fit(0)

    def evict(self, token_ids: list[int]) -> bool:
        """Remove a specific prefix from the cache.

        Returns True if the entry was found and removed.
        """
        h = 0
        for tok in token_ids:
            h = ((h * self._HASH_BASE) + tok) % self._HASH_MOD
        if h in self._cache:
            del self._cache[h]
            return True
        return False

    def clear(self) -> None:
        """Remove all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
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
        self._memory_budget = new_budget_bytes
        self._evict_until_fit(0)
