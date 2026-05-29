"""Redis-backed distributed prompt cache.

Provides RedisPromptCache for storing and retrieving prompt prefix KV cache
references across a distributed cluster via Redis.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

try:
    import redis
except ImportError:
    redis = None  # type: ignore[assignment]


@dataclass
class CachedPrompt:
    """A cached prompt entry stored in Redis."""
    prefix_hash: str
    tokens: list[int] = field(default_factory=list)
    kv_cache_ref: str = ""
    token_count: int = 0
    created_at: float = field(default_factory=time.time)
    ttl: float = 3600.0

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl


class RedisPromptCache:
    """Redis-backed distributed prompt cache.

    Stores prompt prefix hashes with references to KV cache locations
    (node:cache:key) for cross-node cache sharing.
    """

    HASH_PREFIX = "distllm:prompt:"
    INDEX_KEY = "distllm:index:"
    STATS_KEY = "distllm:stats"

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        max_entries: int = 10000,
        ttl_seconds: float = 3600.0,
    ):
        self._url = url
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._client: Any = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to Redis. Returns True on success."""
        if redis is None:
            logger.warning("redis package not installed")
            return False
        try:
            self._client = redis.from_url(self._url)
            self._client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self._url}")
            return True
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._connected = False

    def is_connected(self) -> bool:
        """Check if connected to Redis."""
        if not self._connected or self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            self._connected = False
            return False

    def _hash_tokens(self, token_ids: list[int]) -> str:
        """Compute SHA-256 hash of token sequence, returned as hex string."""
        h = hashlib.sha256()
        for tok in token_ids:
            h.update(tok.to_bytes(4, "little", signed=True))
        return h.hexdigest()

    def store(self, token_ids: list[int], kv_cache_ref: str = "") -> str:
        """Store a prompt prefix in Redis.

        Returns:
            The prefix hash (16-char hex prefix).
        """
        if not self.is_connected():
            return ""

        prefix_hash = self._hash_tokens(token_ids)[:16]
        key = f"{self.HASH_PREFIX}{prefix_hash}"

        entry = CachedPrompt(
            prefix_hash=prefix_hash,
            tokens=token_ids,
            kv_cache_ref=kv_cache_ref,
            token_count=len(token_ids),
        )

        try:
            self._client.setex(key, int(self._ttl), json.dumps({
                "prefix_hash": entry.prefix_hash,
                "kv_cache_ref": entry.kv_cache_ref,
                "token_count": entry.token_count,
                "created_at": entry.created_at,
            }))

            # Add to sorted set index (score = timestamp for LRU ordering)
            self._client.zadd(self.INDEX_KEY, {prefix_hash: time.time()})

            # Evict if over limit
            self._evict_if_needed()

            logger.debug(f"Stored prompt prefix {prefix_hash} ({len(token_ids)} tokens)")
            return prefix_hash

        except Exception as e:
            logger.warning(f"Failed to store prompt in Redis: {e}")
            return ""

    def lookup(self, token_ids: list[int]) -> CachedPrompt | None:
        """Lookup a prompt prefix in Redis.

        Returns:
            CachedPrompt if found and not expired, None otherwise.
        """
        if not self.is_connected():
            return None

        prefix_hash = self._hash_tokens(token_ids)[:16]
        key = f"{self.HASH_PREFIX}{prefix_hash}"

        try:
            raw = self._client.get(key)
            if raw is None:
                self._client.hincrby(self.STATS_KEY, "misses", 1)
                return None

            data = json.loads(raw)
            prompt = CachedPrompt(
                prefix_hash=data["prefix_hash"],
                kv_cache_ref=data["kv_cache_ref"],
                token_count=data["token_count"],
                created_at=data["created_at"],
            )

            # Update access index
            self._client.zadd(self.INDEX_KEY, {prefix_hash: time.time()})
            self._client.hincrby(self.STATS_KEY, "hits", 1)

            return prompt

        except Exception as e:
            logger.warning(f"Redis lookup failed: {e}")
            return None

    def lookup_prefix(self, token_ids: list[int]) -> tuple[int, str]:
        """Find the longest matching prefix by trying progressively shorter prefixes.

        Returns:
            (match_length, kv_cache_ref) or (0, "") if no match.
        """
        if not self.is_connected():
            return 0, ""

        # Try from longest to shortest prefix
        best_len = 0
        best_ref = ""

        for length in range(len(token_ids), 0, -1):
            prefix_hash = self._hash_tokens(token_ids[:length])[:16]
            key = f"{self.HASH_PREFIX}{prefix_hash}"

            try:
                raw = self._client.get(key)
                if raw is not None:
                    data = json.loads(raw)
                    if data["token_count"] > best_len:
                        best_len = data["token_count"]
                        best_ref = data["kv_cache_ref"]
            except Exception:
                continue

        return best_len, best_ref

    def delete(self, token_ids: list[int]) -> bool:
        """Delete a prompt prefix from Redis."""
        if not self.is_connected():
            return False

        prefix_hash = self._hash_tokens(token_ids)[:16]
        key = f"{self.HASH_PREFIX}{prefix_hash}"

        try:
            self._client.delete(key)
            self._client.zrem(self.INDEX_KEY, prefix_hash)
            return True
        except Exception as e:
            logger.warning(f"Redis delete failed: {e}")
            return False

    def clear(self) -> int:
        """Clear all prompt cache entries. Returns number of entries removed."""
        if not self.is_connected():
            return 0

        try:
            keys = self._client.keys(f"{self.HASH_PREFIX}*")
            count = 0
            if keys:
                self._client.delete(*keys)
                count = len(keys)
            self._client.delete(self.INDEX_KEY)
            return count
        except Exception as e:
            logger.warning(f"Redis clear failed: {e}")
            return 0

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if over max_entries limit."""
        try:
            total = self._client.zcard(self.INDEX_KEY)
            if total is not None and total > self._max_entries:
                excess = total - self._max_entries
                old_hashes = self._client.zrange(self.INDEX_KEY, 0, excess - 1)
                for h in old_hashes:
                    self._client.delete(f"{self.HASH_PREFIX}{h}")
                    self._client.zrem(self.INDEX_KEY, h)
        except Exception as e:
            logger.debug(f"Eviction check failed: {e}")

    def stats(self) -> dict:
        """Return cache statistics."""
        if not self.is_connected():
            return {"connected": False}

        try:
            total_entries = self._client.zcard(self.INDEX_KEY) or 0
            hits = int(self._client.hget(self.STATS_KEY, "hits") or 0)
            misses = int(self._client.hget(self.STATS_KEY, "misses") or 0)
            total = hits + misses
            return {
                "connected": True,
                "total_entries": total_entries,
                "hits": hits,
                "misses": misses,
                "hit_rate": hits / total if total > 0 else 0.0,
            }
        except Exception as e:
            logger.warning(f"Redis stats failed: {e}")
            return {"connected": False}
