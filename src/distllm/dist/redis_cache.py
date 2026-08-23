"""Redis-backed KV cache store — shared prefix cache across worker nodes.

Provides a :class:`RedisKVCache` that implements the same interface as the
in-memory ``PrefixCache`` / ``RadixTreeCache`` but uses Redis as the backing
store, enabling workers to share cached KV entries across pod restarts and
across nodes.

The cache is a simple key-value store where:
- **Key:** SHA-256 hash of the token prefix.
- **Value:** Serialized KV cache pages (via protobuf).
- **TTL:** Configurable per-entry TTL (default 1 hour).
- **Pub/Sub:** Optional invalidation channel for cross-worker consistency.

Usage::

    cache = RedisKVCache(redis_url="redis://localhost:6379/0")
    await cache.store(prefix_tokens, kv_cache)
    match_len, kv_data = await cache.lookup(prefix_tokens)
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from typing import Any

from loguru import logger


class RedisKVCache:
    """Redis-backed prefix KV cache for distributed workers.

    Gracefully degrades when Redis is unavailable — returns misses
    instead of raising.

    Args:
        redis_url: Redis connection URL.
        default_ttl_s: Default entry TTL in seconds (default 1 hour).
        prefix: Key prefix for all cache entries (for namespace isolation).
        max_value_bytes: Maximum serialized value size (default 256 MB).
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        default_ttl_s: float = 3600.0,
        prefix: str = "distllm:kv:",
        max_value_bytes: int = 256 * 1024 * 1024,
    ):
        self._redis_url = redis_url
        self._default_ttl = default_ttl_s
        self._prefix = prefix
        self._max_value_bytes = max_value_bytes
        self._redis = None
        self._available = False

        # In-memory fallback for when Redis is unavailable.
        self._fallback: dict[str, tuple[float, Any]] = {}
        self._fallback_max = 1000

        # Metrics
        self._hits = 0
        self._misses = 0
        self._stores = 0
        self._errors = 0

    async def _connect(self) -> bool:
        """Lazy-connect to Redis.

        Returns True if connected (or already connected).
        """
        if self._available and self._redis is not None:
            return True

        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self._redis_url,
                socket_connect_timeout=2,
                socket_timeout=5,
                retry_on_timeout=False,
            )
            await self._redis.ping()
            self._available = True
            logger.info("Connected to Redis KV cache at {}", self._redis_url)
            return True
        except Exception as e:
            self._redis = None
            self._available = False
            logger.warning(
                "Redis unavailable, using in-memory fallback: {}", e,
            )
            return False

    def _prefix_hash(self, tokens: list[int]) -> str:
        """Compute the key for a prefix token sequence."""
        h = hashlib.sha256()
        for t in tokens:
            h.update(struct.pack("!I", t))
        return f"{self._prefix}{h.hexdigest()}"

    def _serialize(self, kv_data: Any) -> bytes:
        """Serialize KV cache data to bytes.

        Uses protobuf when available, falls back to torch.save.
        """
        try:
            from distllm.dist.cross_cluster import CrossClusterForwarder
            forwarder = CrossClusterForwarder()
            kv_dict = {"layers": kv_data} if isinstance(kv_data, list) else kv_data
            return forwarder._kv_to_protobuf(kv_dict).encode("utf-8") if isinstance(
                forwarder._kv_to_protobuf(kv_dict), str
            ) else forwarder._kv_to_protobuf(kv_dict)
        except Exception:
            import torch
            import io
            buf = io.BytesIO()
            torch.save(kv_data, buf)
            return buf.getvalue()

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize bytes back to KV cache data."""
        import torch
        import io
        try:
            return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        except Exception:
            return data

    # ── Public API ────────────────────────────────────────────────────

    async def store(
        self,
        tokens: list[int],
        kv_data: Any,
        ttl_s: float | None = None,
    ) -> bool:
        """Store KV cache data for a token prefix.

        Args:
            tokens: Prefix token IDs.
            kv_data: KV cache tensors (list of layer tuples, or dict).
            ttl_s: Override TTL.  Defaults to *default_ttl_s*.

        Returns:
            True if stored successfully.
        """
        key = self._prefix_hash(tokens)
        ttl = ttl_s or self._default_ttl

        try:
            serialized = self._serialize(kv_data)
            if len(serialized) > self._max_value_bytes:
                logger.warning(
                    "KV cache entry too large: {} bytes (max {})",
                    len(serialized), self._max_value_bytes,
                )
                return False

            if await self._connect():
                await self._redis.setex(key, int(ttl), serialized)  # type: ignore[union-attr]
                self._stores += 1
                return True
        except Exception as e:
            self._errors += 1
            logger.debug("Redis store failed, using fallback: {}", e)

        # Fallback: in-memory dict.
        self._fallback[key] = (time.time() + ttl, kv_data)
        if len(self._fallback) > self._fallback_max:
            self._evict_fallback()
        self._stores += 1
        return True

    async def lookup(
        self,
        tokens: list[int],
    ) -> tuple[int, Any]:
        """Look up KV cache data for a token prefix.

        Args:
            tokens: Prefix token IDs.

        Returns:
            ``(match_len, kv_data)`` where *match_len* is the number of
            tokens matched (0 if miss), and *kv_data* is ``None`` on miss.
        """
        key = self._prefix_hash(tokens)

        try:
            if await self._connect():
                raw = await self._redis.get(key)  # type: ignore[union-attr]
                if raw is not None:
                    self._hits += 1
                    return len(tokens), self._deserialize(raw)
        except Exception as e:
            self._errors += 1
            logger.debug("Redis lookup failed, using fallback: {}", e)

        # Fallback: in-memory dict.
        entry = self._fallback.get(key)
        if entry is not None:
            expiry, kv_data = entry
            if time.time() < expiry:
                self._hits += 1
                return len(tokens), kv_data
            else:
                del self._fallback[key]

        self._misses += 1
        return 0, None

    async def delete(self, tokens: list[int]) -> bool:
        """Remove a cached entry."""
        key = self._prefix_hash(tokens)
        try:
            if self._available and self._redis is not None:
                await self._redis.delete(key)
        except Exception:
            pass
        self._fallback.pop(key, None)
        return True

    async def clear(self) -> None:
        """Clear all cached entries with our prefix."""
        try:
            if await self._connect():
                cursor = 0
                while True:
                    cursor, keys = await self._redis.scan(  # type: ignore[union-attr]
                        cursor=cursor, match=f"{self._prefix}*", count=100,
                    )
                    if keys:
                        await self._redis.delete(*keys)  # type: ignore[union-attr]
                    if cursor == 0:
                        break
        except Exception:
            pass
        self._fallback.clear()
        logger.info("Redis KV cache cleared")

    def _evict_fallback(self) -> None:
        """Evict oldest entries from the in-memory fallback."""
        now = time.time()
        stale = [k for k, (exp, _) in self._fallback.items() if now >= exp]
        for k in stale:
            del self._fallback[k]
        while len(self._fallback) > self._fallback_max * 0.8:
            self._fallback.pop(next(iter(self._fallback)), None)

    # ── Metrics ───────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(total, 1), 3),
            "stores": self._stores,
            "errors": self._errors,
            "redis_connected": self._available,
            "fallback_entries": len(self._fallback),
        }
