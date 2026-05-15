"""Hash-based LRU prefix cache for token sequences."""

import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple


class PrefixCache:
    """Hash-based LRU prefix cache for token sequences.

    Stores precomputed KV data for common token prefixes (system prompts,
    conversation history). Returns the longest matching cached prefix so
    the caller can skip recomputing those tokens.

    Uses a rolling polynomial hash for O(n) lookup instead of O(n) SHA-256
    computations. Each prefix length gets one hash computed incrementally.

    In pipeline parallelism, each node maintains its own prefix cache
    since KV cache is split across machines.
    """

    # Polynomial hash parameters (large prime modulus, random-ish base)
    _HASH_BASE = 31337
    _HASH_MOD = (1 << 61) - 1  # Mersenne prime for fast modulo

    def __init__(self, max_entries: int = 1024, min_prefix_len: int = 16):
        self.max_entries = max_entries
        self.min_prefix_len = min_prefix_len
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def lookup(self, token_ids: List[int]) -> Tuple[int, Optional[dict]]:
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

    def store(self, token_ids: List[int], kv_data: dict) -> None:
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

    def evict(self, token_ids: List[int]) -> bool:
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
        }
