"""KV cache manager — manages caches for multiple concurrent requests.

Extracted from ``kv_cache.py`` to reduce that file below the 800-line ceiling.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict

import torch
from loguru import logger

from distllm.core.kv_cache import KVCache


class KVCacheManager:
    """Manages KV caches for multiple concurrent requests."""

    def __init__(self):
        self.caches: dict[str, KVCache] = {}
        self._metadata: dict[str, dict] = {}
        self._lock = threading.RLock()

        self._prefix_freq: dict[str, int] = defaultdict(int)
        self._request_prefix: dict[str, str] = {}

    def create(
        self,
        request_id: str,
        num_layers: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        device: str = "cpu",
        quant_bits: int = 0,
        prefix_hash: str = "",
    ) -> KVCache:
        """Create a new KV cache for a request."""
        cache = KVCache()
        cache.init_cache(num_layers, batch_size, num_heads, head_dim, device)
        if quant_bits > 0:
            cache.enable_quantization(quant_bits)
        with self._lock:
            self.caches[request_id] = cache
            self._metadata[request_id] = {
                "created_at": time.time(),
                "last_accessed": time.time(),
                "access_count": 0,
                "priority": 0,
                "prefix_hash": prefix_hash,
            }
            if prefix_hash:
                self._prefix_freq[prefix_hash] += 1
                self._request_prefix[request_id] = prefix_hash
        return cache

    def get(self, request_id: str) -> KVCache | None:
        """Get KV cache for a request."""
        with self._lock:
            cache = self.caches.get(request_id)
            if cache is not None and request_id in self._metadata:
                self._metadata[request_id]["last_accessed"] = time.time()
                self._metadata[request_id]["access_count"] += 1
            return cache

    def update(self, request_id: str, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Update KV cache for a request."""
        with self._lock:
            cache = self.caches.get(request_id)
            if cache is not None and request_id in self._metadata:
                self._metadata[request_id]["last_accessed"] = time.time()
        if cache is None:
            return None
        return cache.update(layer_idx, new_key, new_value)

    def delete(self, request_id: str):
        """Delete KV cache for a request."""
        with self._lock:
            if request_id in self.caches:
                self.caches[request_id].clear()
                del self.caches[request_id]
                self._metadata.pop(request_id, None)
                prefix_hash = self._request_prefix.pop(request_id, "")
                if prefix_hash:
                    self._prefix_freq[prefix_hash] = max(0, self._prefix_freq[prefix_hash] - 1)
                    if self._prefix_freq[prefix_hash] == 0:
                        del self._prefix_freq[prefix_hash]

    def clear_all(self):
        """Clear all caches."""
        with self._lock:
            for cache in self.caches.values():
                cache.clear()
            self.caches = {}
            self._metadata.clear()

    @property
    def active_requests(self) -> int:
        with self._lock:
            return len(self.caches)

    def total_memory_usage(self) -> int:
        """Total memory usage across all caches."""
        with self._lock:
            return sum(cache.memory_usage() for cache in self.caches.values())

    def eviction_score(self, request_id: str) -> float:
        """Compute eviction priority score for a cache."""
        with self._lock:
            if request_id not in self._metadata:
                return 0.0
            meta = self._metadata[request_id]
            now = time.time()
            age = now - meta["created_at"]
            idle = now - meta["last_accessed"]
            recency = max(0.0, 1.0 - idle / max(age, 1))
            frequency = min(1.0, meta["access_count"] / max(meta["access_count"] + 10, 1))
            cache = self.caches.get(request_id)
            mem = cache.memory_usage() if cache else 0
            total = sum(c.memory_usage() for c in self.caches.values())
            mem_pressure = mem / max(total, 1)
            prefix_hash = meta.get("prefix_hash", "")
            if prefix_hash and prefix_hash in self._prefix_freq:
                freq = self._prefix_freq[prefix_hash]
                reuse_score = min(1.0, freq / 10.0)
            else:
                reuse_score = 0.0
            return (recency + frequency + (1.0 - mem_pressure) + reuse_score) / 4

    def evict_lowest_score(self) -> str | None:
        """Evict the cache with the lowest eviction score."""
        with self._lock:
            if not self.caches:
                return None
            scores = {rid: self.eviction_score(rid) for rid in self.caches}
            victim = min(scores, key=scores.get)
            self.caches[victim].clear()
            del self.caches[victim]
            self._metadata.pop(victim, None)
            return victim
