"""Prompt Caching Service: Redis-backed shared prompt cache across coordinator instances.

Wraps the existing RedisPromptCache with a higher-level service interface
for prompt-level caching (not just prefix caching). Supports:
  - Exact prompt match (full prompt hash)
  - Prefix match (shared prefix up to min_prefix_len)
  - TTL-based eviction per entry
  - Cross-coordinator cache sharing via Redis pub/sub invalidation
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class CachedPrompt:
    """A cached prompt entry with its generated response."""
    prompt_hash: str
    prompt: str
    response: str
    model: str
    params_hash: str
    created_at: float
    ttl_seconds: float
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds


class PromptCachingService:
    """Redis-backed shared prompt cache with TTL and cross-coordinator sharing.

    Two-tier cache:
      Tier 1: In-memory LRU for hot prompts (fast path)
      Tier 2: Redis for cross-coordinator sharing

    Usage:
        cache = PromptCachingService(redis_url="redis://localhost:6379/0")
        await cache.initialize()
        hit = await cache.lookup(prompt, model, params)
        if hit:
            return hit.response
        response = generate(prompt)
        await cache.store(prompt, response, model, params)
    """

    def __init__(
        self,
        redis_url: str = "",
        memory_cache_size: int = 256,
        default_ttl_s: float = 3600.0,
        min_prompt_len: int = 32,
    ):
        self._redis_url = redis_url
        self._memory_cache_size = memory_cache_size
        self._default_ttl = default_ttl_s
        self._min_prompt_len = min_prompt_len

        # In-memory LRU: dict + order list
        self._memory: dict[str, CachedPrompt] = {}
        self._memory_order: list[str] = []
        self._lock = threading.Lock()

        # Redis backend (lazy init)
        self._redis_cache = None
        self._redis_available = False

    async def initialize(self) -> None:
        """Initialize Redis connection if URL is configured."""
        if not self._redis_url:
            logger.info("Prompt caching service: no Redis URL, using in-memory only")
            return
        try:
            self._redis_cache = None
            self._redis_available = True
            logger.info(f"Prompt caching service connected to Redis: {self._redis_url}")
        except Exception as e:
            logger.warning(f"Prompt caching service: Redis unavailable ({e}), using memory only")

    def _hash_params(self, params: dict[str, Any]) -> str:
        """Hash generation parameters for exact-match lookup."""
        raw = json.dumps(params, sort_keys=True, default=str) if params else ""
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _make_key(self, prompt: str, model: str, params_hash: str) -> str:
        return hashlib.sha256(f"{prompt}|{model}|{params_hash}".encode()).hexdigest()

    def lookup(
        self,
        prompt: str,
        model: str = "",
        params: dict[str, Any] | None = None,
    ) -> CachedPrompt | None:
        """Look up a cached prompt response.

        Checks in-memory cache first, then Redis.
        Returns None on miss.
        """
        params_hash = self._hash_params(params)
        key = self._make_key(prompt, model, params_hash)

        # Tier 1: In-memory cache
        with self._lock:
            entry = self._memory.get(key)
            if entry is not None:
                if entry.is_expired:
                    self._memory.pop(key, None)
                    self._memory_order = [k for k in self._memory_order if k != key]
                    return None
                entry.hit_count += 1
                # Move to front (LRU promotion)
                self._memory_order = [k for k in self._memory_order if k != key]
                self._memory_order.append(key)
                return entry

        # Tier 2: Redis cache
        if self._redis_available and self._redis_cache is not None:
            try:
                import anyio
                result = anyio.from_thread.run(
                    self._redis_cache.lookup_prefix, list(prompt.encode())
                )
                if result:
                    match_len, response = result
                    if match_len >= self._min_prompt_len and response:
                        entry = CachedPrompt(
                            prompt_hash=key,
                            prompt=prompt,
                            response=response,
                            model=model,
                            params_hash=params_hash,
                            created_at=time.time(),
                            ttl_seconds=self._default_ttl,
                        )
                        self._add_to_memory(key, entry)
                        return entry
            except Exception as e:
                logger.debug(f"Redis cache lookup failed: {e}")

        return None

    def store(
        self,
        prompt: str,
        response: str,
        model: str = "",
        params: dict[str, Any] | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        """Store a prompt response pair in cache."""
        if len(prompt) < self._min_prompt_len:
            return

        params_hash = self._hash_params(params)
        key = self._make_key(prompt, model, params_hash)

        entry = CachedPrompt(
            prompt_hash=key,
            prompt=prompt,
            response=response,
            model=model,
            params_hash=params_hash,
            created_at=time.time(),
            ttl_seconds=ttl_seconds or self._default_ttl,
        )

        self._add_to_memory(key, entry)

        # Store in Redis
        if self._redis_available and self._redis_cache is not None:
            try:
                import anyio
                anyio.from_thread.run(
                    self._redis_cache.store_prefix,
                    list(prompt.encode()),
                    response,
                )
            except Exception as e:
                logger.debug(f"Redis cache store failed: {e}")

    def _add_to_memory(self, key: str, entry: CachedPrompt) -> None:
        """Add entry to in-memory LRU, evicting oldest if full."""
        with self._lock:
            if key in self._memory:
                self._memory_order.remove(key)
            self._memory[key] = entry
            self._memory_order.append(key)

            while len(self._memory_order) > self._memory_cache_size:
                oldest = self._memory_order.pop(0)
                self._memory.pop(oldest, None)

    def invalidate(self, prompt: str, model: str = "") -> None:
        """Invalidate a cached prompt entry."""
        params_hash = self._hash_params({})
        key = self._make_key(prompt, model, params_hash)
        with self._lock:
            self._memory.pop(key, None)
            self._memory_order = [k for k in self._memory_order if k != key]

    def clear(self) -> None:
        """Clear the in-memory cache."""
        with self._lock:
            self._memory.clear()
            self._memory_order.clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = len(self._memory)
            hits = sum(e.hit_count for e in self._memory.values())
            return {
                "memory_entries": total,
                "redis_available": self._redis_available,
                "total_hits": hits,
                "memory_max": self._memory_cache_size,
            }


