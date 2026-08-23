"""Zero-mock tests for RedisKVCache — in-memory fallback path.

No Redis server required. All tests exercise the local fallback dict that
RedisKVCache uses when the Redis connection is unavailable.
"""

from __future__ import annotations

import time

import pytest

from distllm.dist.redis_cache import RedisKVCache


class TestRedisKVCache:
    """Public API of RedisKVCache (store, lookup, delete, clear, stats)."""

    # --- constructor ---

    def test_init_defaults(self):
        cache = RedisKVCache()
        assert cache._redis_url == "redis://localhost:6379/0"
        assert cache._default_ttl == 3600.0
        assert cache._prefix == "distllm:kv:"
        assert cache._max_value_bytes == 256 * 1024 * 1024
        assert cache._available is False
        assert cache._redis is None

    def test_init_custom_parameters(self):
        cache = RedisKVCache(
            redis_url="redis://other:6380/1",
            default_ttl_s=60.0,
            prefix="custom:",
            max_value_bytes=1024,
        )
        assert cache._redis_url == "redis://other:6380/1"
        assert cache._default_ttl == 60.0
        assert cache._prefix == "custom:"
        assert cache._max_value_bytes == 1024

    # --- _prefix_hash ---

    def test_prefix_hash_deterministic(self):
        cache = RedisKVCache()
        assert cache._prefix_hash([1, 2, 3]) == cache._prefix_hash([1, 2, 3])

    def test_prefix_hash_differs_on_different_tokens(self):
        cache = RedisKVCache()
        assert cache._prefix_hash([1, 2, 3]) != cache._prefix_hash([1, 2, 4])

    def test_prefix_hash_empty_tokens(self):
        cache = RedisKVCache()
        h = cache._prefix_hash([])
        assert isinstance(h, str)
        assert h.startswith("distllm:kv:")

    def test_prefix_hash_custom_prefix(self):
        cache = RedisKVCache(prefix="test:kv:")
        h = cache._prefix_hash([42])
        assert h.startswith("test:kv:")

    # --- store ---

    @pytest.mark.asyncio
    async def test_store_basic(self):
        cache = RedisKVCache()
        assert await cache.store([1, 2, 3], {"k": "v"}) is True
        s = cache.stats()
        assert s["stores"] == 1
        assert s["fallback_entries"] == 1

    @pytest.mark.asyncio
    async def test_store_empty_tokens(self):
        cache = RedisKVCache()
        assert await cache.store([], {"data": "x"}) is True
        assert cache.stats()["fallback_entries"] == 1

    @pytest.mark.asyncio
    async def test_store_default_ttl(self):
        cache = RedisKVCache(default_ttl_s=3600.0)
        await cache.store([1], {"k": "v"})
        key = cache._prefix_hash([1])
        expiry, _ = cache._fallback[key]
        assert expiry > time.time() + 3590.0

    @pytest.mark.asyncio
    async def test_store_custom_ttl(self):
        cache = RedisKVCache()
        await cache.store([1], {"k": "v"}, ttl_s=99.0)
        key = cache._prefix_hash([1])
        expiry, _ = cache._fallback[key]
        assert expiry > time.time() + 90.0

    @pytest.mark.asyncio
    async def test_store_multiple_entries(self):
        cache = RedisKVCache()
        await cache.store([1], {"a": 1})
        await cache.store([2], {"b": 2})
        await cache.store([3], {"c": 3})
        s = cache.stats()
        assert s["stores"] == 3
        assert s["fallback_entries"] == 3

    @pytest.mark.asyncio
    async def test_store_overwrite_existing_key(self):
        cache = RedisKVCache()
        await cache.store([1, 2], {"k": "old"})
        await cache.store([1, 2], {"k": "new"})
        s = cache.stats()
        assert s["stores"] == 2
        assert s["fallback_entries"] == 1
        key = cache._prefix_hash([1, 2])
        assert cache._fallback[key][1] == {"k": "new"}

    @pytest.mark.asyncio
    async def test_store_overflow_triggers_eviction(self):
        cache = RedisKVCache()
        cache._fallback_max = 5
        for i in range(20):
            await cache.store([i], {"idx": i})
        assert len(cache._fallback) <= 5

    # --- lookup ---

    @pytest.mark.asyncio
    async def test_lookup_hit(self):
        cache = RedisKVCache()
        await cache.store([10, 20, 30], {"k": "v"})
        match_len, kv_data = await cache.lookup([10, 20, 30])
        assert match_len == 3
        assert kv_data == {"k": "v"}

    @pytest.mark.asyncio
    async def test_lookup_miss_empty_cache(self):
        cache = RedisKVCache()
        match_len, kv_data = await cache.lookup([1, 2, 3])
        assert match_len == 0
        assert kv_data is None

    @pytest.mark.asyncio
    async def test_lookup_miss_wrong_tokens(self):
        cache = RedisKVCache()
        await cache.store([1, 2, 3], {"k": "v"})
        match_len, kv_data = await cache.lookup([9, 9, 9])
        assert match_len == 0
        assert kv_data is None

    @pytest.mark.asyncio
    async def test_lookup_empty_tokens_on_empty_cache(self):
        cache = RedisKVCache()
        match_len, kv_data = await cache.lookup([])
        assert match_len == 0
        assert kv_data is None

    @pytest.mark.asyncio
    async def test_lookup_expired_entry_returns_miss(self):
        cache = RedisKVCache()
        key = cache._prefix_hash([1, 2])
        cache._fallback[key] = (time.time() - 10, {"k": "v"})
        match_len, kv_data = await cache.lookup([1, 2])
        assert match_len == 0
        assert kv_data is None

    @pytest.mark.asyncio
    async def test_lookup_tracks_hits_and_misses(self):
        cache = RedisKVCache()
        await cache.store([1], {"k": "v"})
        await cache.lookup([1])  # hit
        await cache.lookup([2])  # miss
        s = cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["hit_rate"] == 0.5

    # --- delete ---

    @pytest.mark.asyncio
    async def test_delete_existing(self):
        cache = RedisKVCache()
        await cache.store([10, 20], {"k": "v"})
        assert await cache.delete([10, 20]) is True
        match_len, kv_data = await cache.lookup([10, 20])
        assert match_len == 0
        assert kv_data is None

    @pytest.mark.asyncio
    async def test_delete_non_existent(self):
        cache = RedisKVCache()
        assert await cache.delete([99, 100]) is True

    @pytest.mark.asyncio
    async def test_delete_empty_tokens(self):
        cache = RedisKVCache()
        assert await cache.delete([]) is True

    @pytest.mark.asyncio
    async def test_delete_removes_from_fallback(self):
        cache = RedisKVCache()
        await cache.store([1], {"k": "v"})
        key = cache._prefix_hash([1])
        assert key in cache._fallback
        await cache.delete([1])
        assert key not in cache._fallback

    # --- clear ---

    @pytest.mark.asyncio
    async def test_clear_empty_cache(self):
        cache = RedisKVCache()
        await cache.clear()
        assert cache.stats()["fallback_entries"] == 0

    @pytest.mark.asyncio
    async def test_clear_removes_all(self):
        cache = RedisKVCache()
        await cache.store([1], {"a": 1})
        await cache.store([2], {"b": 2})
        await cache.store([3], {"c": 3})
        await cache.clear()
        assert cache.stats()["fallback_entries"] == 0
        assert (await cache.lookup([1]))[1] is None
        assert (await cache.lookup([2]))[1] is None
        assert (await cache.lookup([3]))[1] is None

    # --- stats ---

    def test_stats_initial_values(self):
        cache = RedisKVCache()
        s = cache.stats()
        assert s["hits"] == 0
        assert s["misses"] == 0
        assert s["hit_rate"] == 0.0
        assert s["stores"] == 0
        assert s["errors"] == 0
        assert s["redis_connected"] is False
        assert s["fallback_entries"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_operations(self):
        cache = RedisKVCache()
        await cache.store([1], {"k": "v"})
        await cache.lookup([1])  # hit
        await cache.lookup([2])  # miss
        s = cache.stats()
        assert s["stores"] == 1
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["hit_rate"] == 0.5
        assert s["fallback_entries"] == 1

    # --- _connect gracefully degrades ---

    @pytest.mark.asyncio
    async def test_connect_returns_false_without_redis(self):
        cache = RedisKVCache()
        connected = await cache._connect()
        assert connected is False
        assert cache._available is False
        assert cache._redis is None

    # --- round-trip with multiple entries ---

    @pytest.mark.asyncio
    async def test_multiple_store_and_lookup(self):
        cache = RedisKVCache()
        entries = {
            (1, 2, 3): "alpha",
            (4, 5, 6, 7): "beta",
            (8,): "gamma",
        }
        for tokens, val in entries.items():
            await cache.store(list(tokens), {"label": val})
        for tokens, val in entries.items():
            match_len, kv_data = await cache.lookup(list(tokens))
            assert match_len == len(tokens)
            assert kv_data == {"label": val}
        s = cache.stats()
        assert s["hits"] == len(entries)
        assert s["misses"] == 0
