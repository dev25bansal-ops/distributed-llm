"""Redis-backed distributed rate limiter for multi-instance deployments.

Replaces the in-memory ``_RateLimiter`` in ``middleware.py`` when a Redis
URL is configured.  All API server instances share the same rate limit
state, preventing an attacker from rotating across instances.

Usage::

    from distllm.api.redis_rate_limiter import RedisRateLimiter
    rate_limiter = RedisRateLimiter(redis_url="redis://localhost:6379")
    if await rate_limiter.is_rate_limited(client_ip):
        ...

Env var ``DISTLLM_REDIS_URL`` enables the Redis backend.
"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Any

from loguru import logger


class RedisRateLimiter:
    """Sliding window rate limiter backed by Redis.

    Each IP address gets a Redis key with a sorted set of timestamps.
    Expired entries are removed via ``ZREMRANGEBYSCORE`` on every check.

    Args:
        redis_url: Redis connection URL.  Falls back to in-memory if None.
        max_attempts: Max failed attempts before rate limiting.
        window_seconds: Sliding window duration.
        key_prefix: Redis key prefix for rate limit entries.
        max_tracked_ips: Maximum IPs to track in memory fallback (LRU eviction).
    """

    def __init__(
        self,
        redis_url: str | None = None,
        max_attempts: int = 30,
        window_seconds: int = 60,
        key_prefix: str = "distllm:ratelimit:",
        max_tracked_ips: int = 10000,
    ):
        self._redis_url = redis_url or os.environ.get("DISTLLM_REDIS_URL")
        self._max_attempts = max_attempts
        self._window = window_seconds
        self._key_prefix = key_prefix
        self._max_tracked_ips = max_tracked_ips
        self._redis: Any = None
        # Sliding window: deque per IP with maxlen = max_attempts + 1
        # Automatically evicts oldest entries when full
        self._local: dict[str, deque[float]] = {}

    async def _get_redis(self) -> Any:
        """Lazy-init Redis connection."""
        if self._redis is not None:
            return self._redis
        if not self._redis_url:
            return None
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
            await self._redis.ping()
            logger.info("Redis rate limiter connected")
        except Exception as e:
            logger.warning(f"Redis rate limiter unavailable ({e}), falling back to in-memory")
            self._redis = None
        return self._redis

    def _local_key(self, ip: str) -> str:
        return f"{self._key_prefix}{ip}"

    async def is_rate_limited(self, ip: str) -> bool:
        redis = await self._get_redis()
        if redis:
            return await self._redis_check(ip, redis)
        return self._local_check(ip)

    async def _redis_check(self, ip: str, redis: Any) -> bool:
        key = self._local_key(ip)
        now = time.time()
        cutoff = now - self._window
        pipe = redis.pipeline()
        pipe.zremrangebyscore(key, 0, cutoff)
        pipe.zcard(key)
        _, count = await pipe.execute()
        return count >= self._max_attempts

    def _local_check(self, ip: str) -> bool:
        now = time.time()
        cutoff = now - self._window
        timestamps = self._local.get(ip)
        if timestamps is None:
            return False
        # Prune expired entries from the left of the deque
        while timestamps and timestamps[0] <= cutoff:
            timestamps.popleft()
        if not timestamps:
            del self._local[ip]
            return False
        # Evict oldest IPs if over cap
        if len(self._local) > self._max_tracked_ips:
            oldest_ip = min(self._local, key=lambda k: self._local[k][0] if self._local[k] else now)
            del self._local[oldest_ip]
        return len(timestamps) >= self._max_attempts

    async def retry_after(self, ip: str) -> int:
        redis = await self._get_redis()
        if redis:
            return await self._redis_retry_after(ip, redis)
        return self._local_retry_after(ip)

    async def _redis_retry_after(self, ip: str, redis: Any) -> int:
        key = self._local_key(ip)
        now = time.time()
        timestamps = await redis.zrangebyscore(key, now - self._window, now)
        if len(timestamps) >= self._max_attempts:
            oldest = float(timestamps[0])
            return max(1, int(self._window - (now - oldest)))
        return 0

    def _local_retry_after(self, ip: str) -> int:
        now = time.time()
        timestamps = self._local.get(ip)
        if timestamps and len(timestamps) >= self._max_attempts:
            oldest = timestamps[0]
            return max(1, int(self._window - (now - oldest)))
        return 0

    async def record_attempt(self, ip: str) -> None:
        redis = await self._get_redis()
        if redis:
            await self._redis_record(ip, redis)
        else:
            self._local_record(ip)

    async def _redis_record(self, ip: str, redis: Any) -> None:
        key = self._local_key(ip)
        now = time.time()
        pipe = redis.pipeline()
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, self._window * 2)
        await pipe.execute()

    def _local_record(self, ip: str) -> None:
        if ip not in self._local:
            self._local[ip] = deque(maxlen=self._max_attempts + 1)
        self._local[ip].append(time.time())
        # Evict stale IPs periodically
        if len(self._local) > self._max_tracked_ips:
            now = time.time()
            cutoff = now - self._window * 2
            stale = [k for k, v in self._local.items() if not v or v[-1] < cutoff]
            for k in stale:
                del self._local[k]
            # If still over cap, evict oldest
            if len(self._local) > self._max_tracked_ips:
                oldest_ip = min(self._local, key=lambda k: self._local[k][0] if self._local[k] else now)
                del self._local[oldest_ip]
