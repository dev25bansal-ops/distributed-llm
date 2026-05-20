"""Redis-backed cross-session prompt cache for distributed LLM inference.

Provides a shared, persistent cache layer across coordinators, nodes,
and sessions. Supports KV cache storage, prompt prefix indexing,
and cross-coordinator cache sharing.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field

from loguru import logger

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None


@dataclass
class CachedPrompt:
    """A cached prompt entry."""
    prefix_hash: str
    tokens: list[int]
    kv_cache_ref: str | None = None
    hit_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    ttl: float = 3600  # 1 hour default

    def is_expired(self) -> bool:
        return time.time() > self.created_at + self.ttl


class RedisPromptCache:
    """Redis-backed distributed prompt cache.

    Stores prompt prefix hashes and their associated KV cache references
    in Redis, enabling cross-coordinator and cross-session cache sharing.

    Usage:
        cache = RedisPromptCache("redis://localhost:6379/0")
        cache.connect()
        cache.store(tokens, kv_ref="node1:cache:abc")
        result = cache.lookup(tokens)
    """

    # Key prefixes
    PROMPT_PREFIX = "distllm:prompt:"
    INDEX_PREFIX = "distllm:index:"
    STATS_PREFIX = "distllm:stats"

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        ttl: float = 3600,
        max_entries: int = 100000,
    ):
        if not REDIS_AVAILABLE:
            raise ImportError("redis package required: pip install redis")

        self._url = url
        self._ttl = ttl
        self._max_entries = max_entries
        self._client: redis.Redis | None = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to Redis server."""
        try:
            self._client = redis.from_url(
                self._url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
            )
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
        if self._client:
            self._client.close()
            self._connected = False

    def _hash_tokens(self, tokens: list[int]) -> str:
        """Compute a deterministic hash for token sequence."""
        h = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:16]
        return h

    def store(
        self,
        tokens: list[int],
        kv_cache_ref: str | None = None,
        ttl: float | None = None,
    ) -> str:
        """Store a prompt in the cache.

        Args:
            tokens: Token ID sequence.
            kv_cache_ref: Reference to the KV cache (e.g., node_id:cache_key).
            ttl: Optional TTL override.

        Returns:
            Prefix hash of the stored prompt.
        """
        if not self._connected:
            return ""

        prefix_hash = self._hash_tokens(tokens)
        entry = CachedPrompt(
            prefix_hash=prefix_hash,
            tokens=tokens,
            kv_cache_ref=kv_cache_ref,
            ttl=ttl or self._ttl,
        )

        try:
            # Store the entry
            key = f"{self.PROMPT_PREFIX}{prefix_hash}"
            self._client.setex(
                key,
                int(entry.ttl),
                json.dumps({
                    "prefix_hash": entry.prefix_hash,
                    "kv_cache_ref": entry.kv_cache_ref,
                    "token_count": len(entry.tokens),
                    "created_at": entry.created_at,
                }),
            )

            # Add to sorted set for LRU eviction
            score = time.time()
            self._client.zadd(self.INDEX_PREFIX, {prefix_hash: score})

            # Enforce max entries
            self._evict_if_needed()

            # Increment stats
            self._client.hincrby(self.STATS_PREFIX, "total_stored", 1)

            return prefix_hash
        except Exception as e:
            logger.warning(f"Redis store failed: {e}")
            return ""

    def lookup(self, tokens: list[int]) -> CachedPrompt | None:
        """Look up a prompt in the cache.

        Args:
            tokens: Token ID sequence.

        Returns:
            CachedPrompt if found, None otherwise.
        """
        if not self._connected:
            return None

        prefix_hash = self._hash_tokens(tokens)
        key = f"{self.PROMPT_PREFIX}{prefix_hash}"

        try:
            data = self._client.get(key)
            if data is None:
                self._client.hincrby(self.STATS_PREFIX, "misses", 1)
                return None

            entry_data = json.loads(data)
            self._client.hincrby(self.STATS_PREFIX, "hits", 1)
            self._client.hincrby(self.STATS_PREFIX, "total_lookups", 1)

            # Update access time and hit count
            self._client.zadd(self.INDEX_PREFIX, {prefix_hash: time.time()})

            return CachedPrompt(
                prefix_hash=prefix_hash,
                tokens=tokens,
                kv_cache_ref=entry_data.get("kv_cache_ref"),
                hit_count=int(self._client.hget(self.STATS_PREFIX, f"hits:{prefix_hash}") or 0),
                last_accessed=time.time(),
            )
        except Exception as e:
            logger.warning(f"Redis lookup failed: {e}")
            return None

    def lookup_prefix(self, tokens: list[int]) -> tuple[int, str | None]:
        """Find the longest matching prefix in the cache.

        Similar to PrefixCache lookup_prefix but uses Redis backing.

        Args:
            tokens: Full token sequence.

        Returns:
            Tuple of (matched_length, kv_cache_ref).
        """
        if not self._connected:
            return 0, None

        # Try progressively longer prefixes (step by 16 tokens)
        step = 16
        best_match = 0
        best_ref = None

        for i in range(step, len(tokens) + 1, step):
            prefix = tokens[:i]
            result = self.lookup(prefix)
            if result is not None:
                best_match = i
                best_ref = result.kv_cache_ref
            else:
                break  # No point checking longer prefixes

        return best_match, best_ref

    def delete(self, tokens: list[int]) -> bool:
        """Delete a prompt from the cache."""
        if not self._connected:
            return False

        prefix_hash = self._hash_tokens(tokens)
        try:
            self._client.delete(f"{self.PROMPT_PREFIX}{prefix_hash}")
            self._client.zrem(self.INDEX_PREFIX, prefix_hash)
            return True
        except Exception:
            return False

    def clear(self) -> int:
        """Clear all cached entries."""
        if not self._connected:
            return 0

        try:
            keys = self._client.keys(f"{self.PROMPT_PREFIX}*")
            if keys:
                self._client.delete(*keys)
            self._client.delete(self.INDEX_PREFIX)
            count = len(keys)
            logger.info(f"Cleared {count} entries from Redis cache")
            return count
        except Exception as e:
            logger.warning(f"Redis clear failed: {e}")
            return 0

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache exceeds max_entries."""
        if not self._connected:
            return

        try:
            current_count = self._client.zcard(self.INDEX_PREFIX)
            if current_count > self._max_entries:
                to_evict = current_count - self._max_entries
                # Remove oldest entries (lowest scores)
                oldest = self._client.zrange(self.INDEX_PREFIX, 0, to_evict - 1)
                if oldest:
                    self._client.zrem(self.INDEX_PREFIX, *oldest)
                    for key in oldest:
                        self._client.delete(f"{self.PROMPT_PREFIX}{key}")
                    logger.info(f"Evicted {len(oldest)} entries from Redis cache")
        except Exception as e:
            logger.warning(f"Redis eviction failed: {e}")

    def stats(self) -> dict:
        """Get cache statistics."""
        if not self._connected:
            return {"connected": False}

        try:
            hits = int(self._client.hget(self.STATS_PREFIX, "hits") or 0)
            misses = int(self._client.hget(self.STATS_PREFIX, "misses") or 0)
            total = hits + misses
            return {
                "connected": True,
                "total_entries": self._client.zcard(self.INDEX_PREFIX),
                "max_entries": self._max_entries,
                "hits": hits,
                "misses": misses,
                "hit_rate": hits / max(total, 1),
                "total_stored": int(self._client.hget(self.STATS_PREFIX, "total_stored") or 0),
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}

    def is_connected(self) -> bool:
        """Check if connected to Redis."""
        if not self._connected or not self._client:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            return False
