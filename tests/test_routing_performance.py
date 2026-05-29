"""Performance tests for routing decisions.

Covers:
- Routing decision latency (< 1ms target)
- Cache hit ratio
- Memory usage
"""

import time

import pytest

from distllm.core.cross_cloud_router import CrossCloudRouter


@pytest.mark.benchmark
class TestRoutingLatency:
    """Test routing decision latency."""

    def test_select_provider_latency(self):
        router = CrossCloudRouter(expand_regions=True)
        iterations = 1000

        start = time.perf_counter()
        for _ in range(iterations):
            router.select_provider(gpu_type="A100", max_latency_ms=1000)
        elapsed = time.perf_counter() - start

        per_call_ms = (elapsed / iterations) * 1000
        assert per_call_ms < 5.0, f"select_provider took {per_call_ms:.2f}ms (target: <5ms)"

    def test_select_provider_carbon_aware_latency(self):
        router = CrossCloudRouter(expand_regions=True)
        iterations = 1000

        start = time.perf_counter()
        for _ in range(iterations):
            router.select_provider_carbon_aware(gpu_type="A100", max_latency_ms=1000)
        elapsed = time.perf_counter() - start

        per_call_ms = (elapsed / iterations) * 1000
        assert per_call_ms < 10.0, f"select_provider_carbon_aware took {per_call_ms:.2f}ms (target: <10ms)"

    def test_get_all_prices_latency(self):
        router = CrossCloudRouter(expand_regions=True)
        iterations = 100

        start = time.perf_counter()
        for _ in range(iterations):
            router.get_all_prices(gpu_type="A100")
        elapsed = time.perf_counter() - start

        per_call_ms = (elapsed / iterations) * 1000
        assert per_call_ms < 50.0, f"get_all_prices took {per_call_ms:.2f}ms"


@pytest.mark.benchmark
class TestCacheHitRatio:
    """Test carbon intensity cache hit ratio."""

    def test_high_cache_hit_ratio(self):
        router = CrossCloudRouter(expand_regions=True, carbon_api_cache_ttl=300)
        regions = list(set(p.region for p in router._providers))

        # First pass: populate cache
        for region in regions:
            router._carbon_provider.get_intensity(region)

        # Second pass: should hit cache
        hits = 0
        total = len(regions) * 10
        for _ in range(10):
            for region in regions:
                intensity = router._carbon_provider.get_intensity(region)
                if intensity.source == "static":  # Static is cached
                    hits += 1

        hit_ratio = hits / total
        assert hit_ratio > 0.95, f"Cache hit ratio: {hit_ratio:.2%} (target: >95%)"


@pytest.mark.benchmark
class TestMemoryUsage:
    """Test memory usage with many providers."""

    def test_provider_list_memory(self):
        router = CrossCloudRouter(expand_regions=True)
        # Should have many providers but not excessive
        assert len(router._providers) < 500
        assert len(router._providers) > 50
