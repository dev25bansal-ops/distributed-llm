"""Tests for PromptCachingService — in-memory fallback when Redis is unavailable."""

from unittest.mock import MagicMock, patch

import pytest

from distllm.core.prompt_caching_service import PromptCachingService


@pytest.fixture
def service():
    return PromptCachingService(redis_url="", memory_cache_size=4, default_ttl_s=3600, min_prompt_len=4)


class TestPromptCachingServiceMemoryOnly:
    """Redis unavailable: falls back to in-memory cache."""

    def test_init_no_redis(self):
        s = PromptCachingService(redis_url="")
        assert s._redis_url == ""
        assert s._redis_cache is None
        assert s._redis_available is False

    def test_initialize_no_redis_url(self, service):
        assert service._redis_available is False
        assert service._redis_cache is None

    def test_store_and_lookup_hit(self, service):
        service.store("hello world", "response text", model="gpt-4")
        result = service.lookup("hello world", model="gpt-4")
        assert result is not None
        assert result.response == "response text"
        assert result.prompt == "hello world"

    def test_lookup_miss(self, service):
        result = service.lookup("unknown prompt", model="gpt-4")
        assert result is None

    def test_store_below_min_length_skips(self, service):
        service.store("ab", "short", model="gpt-4")
        result = service.lookup("ab", model="gpt-4")
        assert result is None

    def test_lookup_different_model_miss(self, service):
        service.store("hello world", "response-1", model="gpt-4")
        result = service.lookup("hello world", model="gpt-3")
        assert result is None

    def test_lookup_with_params(self, service):
        service.store("hello world", "resp", model="gpt-4", params={"temp": 0.7})
        result = service.lookup("hello world", model="gpt-4", params={"temp": 0.7})
        assert result is not None

    def test_lookup_with_different_params_miss(self, service):
        service.store("hello world", "resp", model="gpt-4", params={"temp": 0.7})
        result = service.lookup("hello world", model="gpt-4", params={"temp": 1.0})
        assert result is None

    def test_memory_lru_eviction(self, service):
        for i in range(6):
            service.store(f"prompt number {i}", f"response_{i}", model="m")
        result = service.lookup("prompt number 0", model="m")
        assert result is None

    def test_lru_recent_survives(self, service):
        for i in range(5):
            service.store(f"prompt number {i}", f"response_{i}", model="m")
        result = service.lookup("prompt number 4", model="m")
        assert result is not None

    def test_invalidate_removes_entry(self, service):
        service.store("hello world", "response text", model="gpt-4")
        service.invalidate("hello world", model="gpt-4")
        result = service.lookup("hello world", model="gpt-4")
        assert result is None

    def test_clear(self, service):
        service.store("prompt a", "resp a", model="m")
        service.store("prompt b", "resp b", model="m")
        service.clear()
        assert service.lookup("prompt a", model="m") is None
        assert service.lookup("prompt b", model="m") is None

    def test_stats(self, service):
        service.store("prompt a", "resp a", model="m")
        service.lookup("prompt a", model="m")
        stats = service.stats()
        assert stats["memory_entries"] == 1
        assert stats["redis_available"] is False
        assert stats["total_hits"] >= 1

    def test_expired_entry_not_returned(self, service):
        service._default_ttl = 0.0
        import time
        service.store("hello world", "response text", model="gpt-4")
        result = service.lookup("hello world", model="gpt-4")
        assert result is None

    def test_multiple_stores_same_key_updates(self, service):
        service.store("hello world", "v1", model="gpt-4")
        service.store("hello world", "v2", model="gpt-4")
        result = service.lookup("hello world", model="gpt-4")
        assert result.response == "v2"
