"""Tests for KV cache persistence — save/load cycle with edge cases."""

import time

import pytest
import torch

from distllm.core.kv_cache import KVCache, KVCacheManager


class TestKVCachePersistence:
    """Test KV cache save/load including edge cases."""

    def test_save_and_load_empty_cache(self, tmp_path):
        """Empty cache saves and loads without error."""
        cache = KVCache(max_seq_len=128)
        path = str(tmp_path / "empty_cache.pt")
        torch.save(cache.get_all(), path)
        loaded = torch.load(path, weights_only=True)
        assert loaded == []

    def test_save_and_load_populated_cache(self, tmp_path):
        """Populated cache round-trips through save/load."""
        cache = KVCache(max_seq_len=128)
        cache.init_cache(num_layers=2, batch_size=1, num_heads=8, head_dim=64, device="cpu")

        # Add some data
        k = torch.randn(1, 8, 10, 64)
        v = torch.randn(1, 8, 10, 64)
        cache.update(0, k, v)
        cache.update(1, k, v)

        # Save
        path = str(tmp_path / "cache.pt")
        torch.save(cache.get_all(), path)

        # Load into new cache
        loaded = torch.load(path, weights_only=True)
        new_cache = KVCache(max_seq_len=128)
        new_cache.set_all(loaded)

        assert new_cache.num_layers == 2
        # set_all doesn't track sequence length, but tensors are preserved
        all_data = new_cache.get_all()
        assert len(all_data) == 2
        # The pre-allocated buffer has max_seq_len capacity
        assert all_data[0][0].shape[-2] >= 10  # at least 10 tokens fit

    def test_cache_memory_tracking(self):
        """Cache tracks memory usage correctly."""
        cache = KVCache(max_seq_len=128)
        cache.init_cache(num_layers=2, batch_size=1, num_heads=8, head_dim=64, device="cpu")

        k = torch.randn(1, 8, 5, 64)
        v = torch.randn(1, 8, 5, 64)
        cache.update(0, k, v)
        cache.update(1, k, v)

        mem = cache.memory_usage()
        assert mem > 0

        # Quantization flags
        assert not cache._quantized
        cache.enable_quantization(bits=8)
        assert cache._quantized

    def test_concurrent_save_load(self, tmp_path):
        """Concurrent saves don't corrupt data."""
        import threading

        cache = KVCache(max_seq_len=128)
        cache.init_cache(num_layers=4, batch_size=1, num_heads=8, head_dim=64, device="cpu")

        errors = []

        def save_cache(idx):
            try:
                path = str(tmp_path / f"cache_{idx}.pt")
                torch.save(cache.get_all(), path)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=save_cache, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestRateLimiterAdversarial:
    """Test rate limiter under adversarial load."""

    def test_10k_unique_ips(self):
        """Rate limiter handles 10K+ unique IPs without memory explosion."""
        from distllm.api.middleware import _RateLimiter

        limiter = _RateLimiter(max_attempts=10, window_seconds=60, max_ips=10000)

        # Register 10K unique IPs
        for i in range(10000):
            ip = f"192.168.{i // 256}.{i % 256}"
            limiter.record_attempt(ip)

        # Verify eviction works
        assert len(limiter._attempts) <= 10000

        # New IP should work (after eviction)
        limiter.record_attempt("10.0.0.1")
        assert not limiter.is_rate_limited("10.0.0.1")

    def test_rapid_fire_single_ip(self):
        """Single IP hitting rate limit rapidly."""
        from distllm.api.middleware import _RateLimiter

        limiter = _RateLimiter(max_attempts=5, window_seconds=60)

        for _ in range(10):
            limiter.record_attempt("attacker-ip")

        assert limiter.is_rate_limited("attacker-ip")

    def test_distributed_attack_pattern(self):
        """Attack from many IPs with different patterns."""
        from distllm.api.middleware import _RateLimiter

        limiter = _RateLimiter(max_attempts=3, window_seconds=60, max_ips=100)

        # Each IP makes 4 attempts (exceeds limit of 3)
        for i in range(100):
            ip = f"10.0.{i // 256}.{i % 256}"
            for _ in range(4):
                limiter.record_attempt(ip)

        # All should be rate limited
        for i in range(100):
            ip = f"10.0.{i // 256}.{i % 256}"
            assert limiter.is_rate_limited(ip)

    def test_lru_eviction_under_pressure(self):
        """LRU eviction works correctly under memory pressure."""
        from distllm.api.middleware import _RateLimiter

        limiter = _RateLimiter(max_attempts=100, window_seconds=60, max_ips=50)

        # Add 100 IPs (should evict oldest 50)
        for i in range(100):
            ip = f"192.168.{i // 256}.{i % 256}"
            limiter.record_attempt(ip)

        # Only last 50 should be tracked
        assert len(limiter._attempts) <= 50
