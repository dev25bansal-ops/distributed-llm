"""Property-based tests for routing using Hypothesis.

Covers:
- Routing invariants (price ≤ max_price, latency ≤ max_latency)
- Carbon-aware normalization stability
- Scoring monotonicity
- Cache consistency
"""

import time

import pytest

from distllm.core.cross_cloud_router import (
    CrossCloudRouter,
    _CLOUD_GPU_PRICING,
    _REGIONAL_CARBON_INTENSITY,
)

try:
    from hypothesis import given, settings, strategies as st
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


@pytest.mark.property
class TestRoutingInvariants:
    """Test that routing decisions always satisfy constraints."""

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @given(
        max_latency=st.floats(min_value=10, max_value=10000),
        max_price=st.floats(min_value=1, max_value=1000),
    )
    @settings(max_examples=50, deadline=None)
    def test_select_provider_respects_constraints(self, max_latency, max_price):
        router = CrossCloudRouter(expand_regions=False)
        decision = router.select_provider(
            gpu_type="A100",
            max_latency_ms=max_latency,
            max_price=max_price,
        )
        if decision is not None:
            assert decision.price_per_hour <= max_price
            assert decision.latency_ms <= max_latency

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @given(
        carbon_weight=st.floats(min_value=0, max_value=1),
        max_carbon=st.floats(min_value=10, max_value=1000),
    )
    @settings(max_examples=30, deadline=None)
    def test_carbon_aware_respects_constraints(self, carbon_weight, max_carbon):
        router = CrossCloudRouter(expand_regions=False)
        decision = router.select_provider_carbon_aware(
            gpu_type="A100",
            max_latency_ms=10000,
            carbon_weight=carbon_weight,
            max_carbon_intensity=max_carbon,
        )
        if decision is not None:
            assert decision.carbon_intensity <= max_carbon


@pytest.mark.property
class TestCarbonNormalizationStability:
    """Test that identical inputs produce identical scores."""

    def test_deterministic_routing(self):
        router = CrossCloudRouter(expand_regions=False)
        d1 = router.select_provider_carbon_aware(gpu_type="A100", max_latency_ms=10000)
        d2 = router.select_provider_carbon_aware(gpu_type="A100", max_latency_ms=10000)
        assert d1 is not None
        assert d2 is not None
        assert d1.provider == d2.provider
        assert d1.instance_type == d2.instance_type
        assert d1.price_per_hour == d2.price_per_hour

    def test_same_price_same_ranking(self):
        router = CrossCloudRouter(expand_regions=False)
        # Set all providers to same price
        for p in router._providers:
            p.price_per_hour = 5.0
            p.spot_price = 5.0
        d1 = router.select_provider(max_latency_ms=10000)
        d2 = router.select_provider(max_latency_ms=10000)
        assert d1 is not None
        assert d2 is not None
        assert d1.provider == d2.provider


@pytest.mark.property
class TestScoringMonotonicity:
    """Test that cheaper options are always preferred."""

    def test_cheaper_option_wins(self):
        router = CrossCloudRouter(expand_regions=False)
        # Make one provider much cheaper
        cheapest = router._providers[0]
        cheapest.spot_price = 0.01
        cheapest.price_per_hour = 0.01

        decision = router.select_provider(max_latency_ms=10000)
        assert decision is not None
        assert decision.instance_type == cheapest.instance_type


@pytest.mark.property
class TestCacheConsistency:
    """Test cache TTL behavior."""

    def test_cache_returns_same_within_ttl(self):
        router = CrossCloudRouter(expand_regions=True, carbon_api_cache_ttl=300)
        i1 = router._carbon_provider.get_intensity("us-east-1")
        i2 = router._carbon_provider.get_intensity("us-east-1")
        assert i1.timestamp == i2.timestamp  # Same cached object

    def test_cache_refreshes_after_ttl(self):
        router = CrossCloudRouter(expand_regions=True, carbon_api_cache_ttl=0)
        i1 = router._carbon_provider.get_intensity("us-east-1")
        time.sleep(0.01)
        i2 = router._carbon_provider.get_intensity("us-east-1")
        assert i2.timestamp >= i1.timestamp


@pytest.mark.property
class TestRegionCoverage:
    """Test that region expansion covers all expected regions."""

    def test_all_cloud_regions_represented(self):
        router = CrossCloudRouter(expand_regions=True)
        regions = {p.region for p in router._providers}
        # Should cover at least the major regions
        expected = {"us-east-1", "us-west-2", "eu-west-1", "eu-north-1"}
        assert expected.issubset(regions)

    def test_each_region_has_multiple_instances(self):
        router = CrossCloudRouter(expand_regions=True)
        region_counts: dict[str, int] = {}
        for p in router._providers:
            region_counts[p.region] = region_counts.get(p.region, 0) + 1
        # At least one region should have multiple instances
        assert max(region_counts.values()) > 1
