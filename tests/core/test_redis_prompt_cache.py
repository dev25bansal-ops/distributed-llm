"""Tests for RedisPromptCache with mocked Redis client.

The ``redis`` package is an optional dependency (the module fails closed
when it is absent), so these tests inject a fake module-level ``redis``
namespace rather than requiring the real package.
"""

import time
import types
from unittest.mock import MagicMock

import pytest

from distllm.core.redis_prompt_cache import RedisPromptCache, CachedPrompt


@pytest.fixture
def mock_redis(monkeypatch):
    import distllm.core.redis_prompt_cache as rpc

    fake_redis = types.ModuleType("fake_redis")
    client = MagicMock()
    client.ping.return_value = True
    client.setex.return_value = True
    client.zadd.return_value = 1
    client.zcard.return_value = 0
    client.hincrby.return_value = 1
    client.get.return_value = None
    client.delete.return_value = True
    client.zrem.return_value = 1
    client.keys.return_value = []
    fake_redis.ConnectionPool = MagicMock()
    fake_redis.ConnectionPool.from_url.return_value = MagicMock()
    fake_redis.Redis = MagicMock(return_value=client)
    monkeypatch.setattr(rpc, "redis", fake_redis)
    return client


@pytest.fixture
def cache(mock_redis):
    c = RedisPromptCache(url="redis://localhost:6379/0")
    c.connect()
    assert c._connected
    return c


class TestRedisPromptCache:
    def test_connect_success(self, mock_redis):
        c = RedisPromptCache(url="redis://localhost:6379/0")
        result = c.connect()
        assert result is True
        assert c._connected is True
        mock_redis.ping.assert_called_once()

    def test_connect_failure(self, mock_redis):
        mock_redis.ping.side_effect = ConnectionError("refused")
        c = RedisPromptCache(url="redis://localhost:6379/0")
        result = c.connect()
        assert result is False
        assert c._connected is False

    def test_disconnect_resets_connected(self, cache, mock_redis):
        cache.disconnect()
        assert cache._connected is False
        mock_redis.close.assert_called_once()

    def test_store_returns_hash(self, cache, mock_redis):
        h = cache.store([1, 2, 3], kv_cache_ref="node1:cache:abc")
        assert len(h) == 16
        mock_redis.setex.assert_called_once()
        mock_redis.zadd.assert_called_once()

    def test_store_sets_correct_key(self, cache, mock_redis):
        cache.store([1, 2, 3], kv_cache_ref="node1:cache:abc")
        args, _ = mock_redis.setex.call_args
        key = args[0]
        assert key.startswith("distllm:prompt:")
        assert len(key) > len("distllm:prompt:")

    def test_store_sets_ttl(self, cache, mock_redis):
        cache.store([1, 2, 3], kv_cache_ref="node1:cache:abc")
        args, _ = mock_redis.setex.call_args
        ttl = args[1]
        assert ttl == 3600

    def test_lookup_hit_returns_cached_prompt(self, cache, mock_redis):
        mock_redis.get.return_value = (
            '{"prefix_hash":"abc","kv_cache_ref":"node1:x","token_count":3,"created_at":100.0}'
        )
        mock_redis.hget.return_value = 5
        result = cache.lookup([1, 2, 3])
        assert result is not None
        assert isinstance(result, CachedPrompt)
        assert result.kv_cache_ref == "node1:x"

    def test_lookup_miss_returns_none(self, cache, mock_redis):
        mock_redis.get.return_value = None
        result = cache.lookup([9, 9, 9])
        assert result is None
        mock_redis.hincrby.assert_called_with("distllm:stats", "misses", 1)

    def test_lookup_hit_increments_hits(self, cache, mock_redis):
        mock_redis.get.return_value = (
            '{"prefix_hash":"abc","kv_cache_ref":"node1:x","token_count":3,"created_at":100.0}'
        )
        cache.lookup([1, 2, 3])
        mock_redis.hincrby.assert_any_call("distllm:stats", "hits", 1)

    def test_lookup_updates_index(self, cache, mock_redis):
        mock_redis.get.return_value = (
            '{"prefix_hash":"abc","kv_cache_ref":"node1:x","token_count":3,"created_at":100.0}'
        )
        cache.lookup([1, 2, 3])
        mock_redis.zadd.assert_called_once()
        args, _ = mock_redis.zadd.call_args
        assert args[0] == "distllm:index:"

    def test_store_lookup_roundtrip(self, cache, mock_redis):
        h = cache.store([10, 20, 30], kv_cache_ref="node2:cache:xyz")
        mock_redis.get.return_value = (
            '{"prefix_hash":"' + h + '","kv_cache_ref":"node2:cache:xyz","token_count":3,"created_at":200.0}'
        )
        result = cache.lookup([10, 20, 30])
        assert result is not None
        assert result.kv_cache_ref == "node2:cache:xyz"
        assert result.prefix_hash == h

    def test_lookup_prefix_finds_longest_match(self, cache, mock_redis):
        mock_redis.get.side_effect = [
            '{"prefix_hash":"p1","kv_cache_ref":"node1:x","token_count":16,"created_at":100.0}',
            '{"prefix_hash":"p2","kv_cache_ref":"node1:y","token_count":32,"created_at":200.0}',
            None,
        ]
        tokens = list(range(64))
        match_len, ref = cache.lookup_prefix(tokens)
        assert match_len == 32
        assert ref == "node1:y"

    def test_delete_removes_key(self, cache, mock_redis):
        result = cache.delete([1, 2, 3])
        assert result is True
        mock_redis.delete.assert_called_once()
        mock_redis.zrem.assert_called_once()

    def test_delete_not_connected(self, mock_redis):
        c = RedisPromptCache(url="redis://localhost:6379/0")
        result = c.delete([1, 2, 3])
        assert result is False

    def test_clear_deletes_all_keys(self, cache, mock_redis):
        mock_redis.keys.return_value = ["distllm:prompt:a", "distllm:prompt:b"]
        count = cache.clear()
        assert count == 2
        assert mock_redis.delete.call_count == 2
        mock_redis.delete.assert_any_call("distllm:prompt:a", "distllm:prompt:b")
        mock_redis.delete.assert_any_call("distllm:index:")

    def test_clear_connected_no_keys(self, cache, mock_redis):
        mock_redis.keys.return_value = []
        count = cache.clear()
        assert count == 0

    def test_is_connected_returns_true(self, cache, mock_redis):
        assert cache.is_connected() is True
        mock_redis.ping.assert_called()

    def test_is_connected_ping_fails(self, cache, mock_redis):
        mock_redis.ping.side_effect = ConnectionError("down")
        assert cache.is_connected() is False

    def test_is_connected_not_connected(self, mock_redis):
        c = RedisPromptCache(url="redis://localhost:6379/0")
        assert c.is_connected() is False

    def test_stats_not_connected(self, mock_redis):
        c = RedisPromptCache(url="redis://localhost:6379/0")
        stats = c.stats()
        assert stats == {"connected": False}

    def test_stats_connected(self, cache, mock_redis):
        mock_redis.zcard.return_value = 5
        mock_redis.hget.side_effect = lambda key, field: {"hits": 10, "misses": 2, "total_stored": 15}.get(field, 0)
        stats = cache.stats()
        assert stats["connected"] is True
        assert stats["total_entries"] == 5
        assert stats["hits"] == 10
        assert stats["misses"] == 2
        assert stats["hit_rate"] == pytest.approx(10 / 12)

    def test_evict_if_needed_under_limit(self, cache, mock_redis):
        mock_redis.zcard.return_value = 50
        cache._max_entries = 100
        cache._evict_if_needed()
        mock_redis.zrange.assert_not_called()

    def test_evict_if_needed_over_limit(self, cache, mock_redis):
        mock_redis.zcard.return_value = 150
        mock_redis.zrange.return_value = ["old_hash"]
        cache._max_entries = 100
        cache._evict_if_needed()
        mock_redis.zrange.assert_called_once_with("distllm:index:", 0, 49)
        mock_redis.zrem.assert_called_once()
        mock_redis.delete.assert_called()

    def test_store_not_connected(self, mock_redis):
        c = RedisPromptCache(url="redis://localhost:6379/0")
        result = c.store([1, 2, 3])
        assert result == ""

    def test_lookup_not_connected(self, mock_redis):
        c = RedisPromptCache(url="redis://localhost:6379/0")
        result = c.lookup([1, 2, 3])
        assert result is None

    def test_cached_prompt_is_expired(self):
        p = CachedPrompt(prefix_hash="h", tokens=[1], created_at=0, ttl=1)
        assert p.is_expired()

    def test_cached_prompt_not_expired(self):
        p = CachedPrompt(prefix_hash="h", tokens=[1], ttl=3600)
        assert not p.is_expired()
