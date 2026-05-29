"""Performance benchmarks for cache system.

Measures P50/P99 latency, throughput, and scalability.
"""

import time

import pytest

from distllm.core.cache_manager import CacheManager, RollingHash, _rolling_prefix_hashes
from distllm.dist.prefix_cache import PrefixCache
from distllm.dist.cache import CacheIndex


class TestCacheLookupLatency:
    """Benchmark lookup latency for different cache implementations."""

    def test_prefix_cache_lookup_latency(self):
        """Measure PrefixCache lookup latency."""
        cache = PrefixCache(min_prefix_len=16, max_entries=1000)

        # Pre-populate
        for i in range(100):
            tokens = [i] + list(range(100, 116))
            cache.store(tokens, {"data": i})

        # Benchmark
        latencies = []
        for i in range(1000):
            tokens = [i % 100] + list(range(100, 116))
            start = time.perf_counter()
            cache.lookup(tokens)
            latencies.append((time.perf_counter() - start) * 1000)

        sorted_lat = sorted(latencies)
        p50 = sorted_lat[len(sorted_lat) // 2]
        p99 = sorted_lat[int(len(sorted_lat) * 0.99)]

        assert p50 < 1.0, f"P50 latency too high: {p50:.3f}ms"
        assert p99 < 10.0, f"P99 latency too high: {p99:.3f}ms"

    def test_cache_index_lookup_latency(self):
        """Measure CacheIndex lookup latency."""
        idx = CacheIndex()

        # Pre-populate
        for i in range(1000):
            idx.store(f"hash_{i}", f"node_{i % 10}", f"ref_{i}")

        # Benchmark
        latencies = []
        for i in range(10000):
            start = time.perf_counter()
            idx.lookup(f"hash_{i % 1000}")
            latencies.append((time.perf_counter() - start) * 1000)

        sorted_lat = sorted(latencies)
        p50 = sorted_lat[len(sorted_lat) // 2]
        p99 = sorted_lat[int(len(sorted_lat) * 0.99)]

        assert p50 < 0.1, f"P50 latency too high: {p50:.4f}ms"
        assert p99 < 1.0, f"P99 latency too high: {p99:.4f}ms"

    def test_rolling_hash_throughput(self):
        """Measure RollingHash extend throughput."""
        h = RollingHash()
        tokens = list(range(1000))

        start = time.perf_counter()
        for t in tokens:
            h.extend(t)
        elapsed = time.perf_counter() - start

        ops_per_sec = len(tokens) / elapsed
        assert ops_per_sec > 100000, f"Throughput too low: {ops_per_sec:.0f} ops/sec"

    def test_rolling_prefix_hashes_throughput(self):
        """Measure _rolling_prefix_hashes throughput."""
        tokens = list(range(512))

        start = time.perf_counter()
        for _ in range(100):
            _rolling_prefix_hashes(tokens, 512)
        elapsed = time.perf_counter() - start

        ops_per_sec = 100 / elapsed
        assert ops_per_sec > 100, f"Throughput too low: {ops_per_sec:.0f} ops/sec"


class TestCacheScalability:
    """Test cache performance at different scales."""

    @pytest.mark.parametrize("num_entries", [10, 100, 1000])
    def test_prefix_cache_scalability(self, num_entries):
        """PrefixCache should scale linearly with entries."""
        cache = PrefixCache(min_prefix_len=16, max_entries=num_entries + 100)

        # Pre-populate
        for i in range(num_entries):
            tokens = [i] + list(range(100, 116))
            cache.store(tokens, {"data": i})

        # Benchmark
        start = time.perf_counter()
        for i in range(1000):
            tokens = [i % num_entries] + list(range(100, 116))
            cache.lookup(tokens)
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time regardless of scale
        assert elapsed < 5.0, f"Too slow at {num_entries} entries: {elapsed:.2f}s"

    @pytest.mark.parametrize("num_entries", [10, 100, 1000])
    def test_cache_index_scalability(self, num_entries):
        """CacheIndex should scale linearly with entries."""
        idx = CacheIndex()

        for i in range(num_entries):
            idx.store(f"hash_{i}", f"node_{i % 10}", f"ref_{i}")

        start = time.perf_counter()
        for i in range(10000):
            idx.lookup(f"hash_{i % num_entries}")
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0, f"Too slow at {num_entries} entries: {elapsed:.2f}s"


class TestCacheMemoryEfficiency:
    """Test memory usage of cache implementations."""

    def test_prefix_cache_memory_within_budget(self):
        """PrefixCache should stay within memory budget."""
        budget = 10000  # 10KB
        cache = PrefixCache(min_prefix_len=1, memory_budget_bytes=budget)

        for i in range(100):
            tokens = [i, 1, 2, 3, 4, 5]
            cache.store(tokens, {"data": "x" * 50})

        stats = cache.stats()
        assert stats["prefix_cache_memory_bytes"] <= budget
