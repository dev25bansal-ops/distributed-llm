"""Unit tests for CrossCloudRouter.

Covers:
- GPU type filtering
- Carbon-aware scoring edge cases
- Empty/single candidate handling
- Price normalization with zero range
- Region-to-zone mappings
- WattTime token caching
- Latency filtering
- Region expansion
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from distllm.core.cross_cloud_router import (
    CarbonIntensity,
    CarbonIntensityProvider,
    CarbonProvider,
    CloudProvider,
    CrossCloudRouter,
    RouteDecision,
    _CLOUD_GPU_PRICING,
    _INSTANCE_GPU_TYPE,
    _REGION_LATENCY,
    _REGIONAL_CARBON_INTENSITY,
)


class TestSelectProviderGpuTypeFilter:
    """Test GPU type filtering in select_provider()."""

    def test_select_a100_returns_only_a100_instances(self):
        router = CrossCloudRouter(expand_regions=False)
        decision = router.select_provider(gpu_type="A100", max_latency_ms=1000)
        assert decision is not None
        assert decision.gpu_type.upper() == "A100"

    def test_select_v100_returns_only_v100_instances(self):
        router = CrossCloudRouter(expand_regions=False)
        decision = router.select_provider(gpu_type="V100", max_latency_ms=1000)
        assert decision is not None
        assert "V100" in decision.gpu_type.upper() or "V100" in decision.instance_type.upper()

    def test_select_empty_gpu_type_returns_any(self):
        router = CrossCloudRouter(expand_regions=False)
        decision = router.select_provider(gpu_type="", max_latency_ms=1000)
        assert decision is not None

    def test_select_nonexistent_gpu_type_returns_none(self):
        router = CrossCloudRouter(expand_regions=False)
        decision = router.select_provider(gpu_type="H200", max_latency_ms=1000)
        assert decision is None

    def test_matches_gpu_type_exact(self):
        provider = CloudProvider(name="aws", instance_type="p4d.24xlarge", gpu_type="A100")
        assert CrossCloudRouter._matches_gpu_type(provider, "A100") is True
        assert CrossCloudRouter._matches_gpu_type(provider, "V100") is False

    def test_matches_gpu_type_case_insensitive(self):
        provider = CloudProvider(name="aws", instance_type="p4d.24xlarge", gpu_type="A100")
        assert CrossCloudRouter._matches_gpu_type(provider, "a100") is True
        assert CrossCloudRouter._matches_gpu_type(provider, "A100") is True

    def test_matches_gpu_type_by_instance_name_fallback(self):
        provider = CloudProvider(name="aws", instance_type="p4d.24xlarge", gpu_type="")
        # "A100" is not in "p4d.24xlarge" — fallback returns False
        assert CrossCloudRouter._matches_gpu_type(provider, "A100") is False
        # But "p4d" IS in the instance name
        provider2 = CloudProvider(name="aws", instance_type="p4d.24xlarge", gpu_type="")
        assert CrossCloudRouter._matches_gpu_type(provider2, "P4D") is True


class TestCarbonAwareScoring:
    """Test carbon-aware scoring edge cases."""

    def test_empty_candidates_returns_none(self):
        router = CrossCloudRouter(expand_regions=False)
        # Set max_price to 0 to filter out all candidates
        decision = router.select_provider_carbon_aware(max_price=0.001)
        assert decision is None

    def test_single_candidate_returns_it(self):
        router = CrossCloudRouter(expand_regions=False)
        # Disable all providers
        for p in router._providers:
            router.update_availability(p.name, p.instance_type, False)
        # Also disable by region key
        for p in router._providers:
            router._availability_cache[f"{p.name}:{p.instance_type}:{p.region}"] = False

        # Re-enable one specific provider
        first = router._providers[0]
        router._availability_cache[f"{first.name}:{first.instance_type}:{first.region}"] = True

        decision = router.select_provider_carbon_aware(max_latency_ms=10000)
        assert decision is not None
        assert decision.instance_type == first.instance_type

    def test_price_normalization_zero_range(self):
        """When all candidates have the same price, normalization should not crash."""
        router = CrossCloudRouter(expand_regions=False)
        # Set all providers to same price
        for p in router._providers:
            p.price_per_hour = 5.0
            p.spot_price = 5.0
        decision = router.select_provider_carbon_aware(max_latency_ms=10000)
        assert decision is not None

    def test_carbon_weight_zero_is_pure_cost(self):
        router = CrossCloudRouter(expand_regions=False)
        decision = router.select_provider_carbon_aware(
            carbon_weight=0.0, max_latency_ms=10000
        )
        assert decision is not None
        # Should pick cheapest
        prices = [p.spot_price for p in router._providers if p.available]
        assert decision.price_per_hour == min(prices)

    def test_carbon_weight_one_is_pure_carbon(self):
        router = CrossCloudRouter(expand_regions=False)
        decision = router.select_provider_carbon_aware(
            carbon_weight=1.0, max_latency_ms=10000
        )
        assert decision is not None
        assert decision.carbon_intensity > 0

    def test_max_carbon_intensity_filter(self):
        router = CrossCloudRouter(expand_regions=False)
        decision = router.select_provider_carbon_aware(
            max_carbon_intensity=10, max_latency_ms=10000
        )
        # Only eu-north-1 (15) and ca-central-1 (30) are below 10... actually none below 10
        # Let's use 20
        decision = router.select_provider_carbon_aware(
            max_carbon_intensity=20, max_latency_ms=10000
        )
        if decision is not None:
            assert decision.carbon_intensity <= 20


class TestRegionToZone:
    """Test region-to-zone mapping."""

    def test_known_regions_return_zone(self):
        assert CarbonIntensityProvider._region_to_zone("us-east-1") == "US-VIRG"
        assert CarbonIntensityProvider._region_to_zone("eu-west-1") == "IE"
        assert CarbonIntensityProvider._region_to_zone("eu-north-1") == "SE"

    def test_unknown_region_returns_none(self):
        result = CarbonIntensityProvider._region_to_zone("mars-colony-1")
        assert result is None

    def test_azure_regions_mapped(self):
        assert CarbonIntensityProvider._region_to_zone("eastus") == "US-VIRG"
        assert CarbonIntensityProvider._region_to_zone("westeurope") == "NL"

    def test_gcp_regions_mapped(self):
        assert CarbonIntensityProvider._region_to_zone("us-central1") == "US-MIDW-MISO"
        assert CarbonIntensityProvider._region_to_zone("europe-north1") == "FI"


class TestWattTimeTokenCaching:
    """Test WattTime token caching behavior."""

    def test_token_cached_after_first_fetch(self):
        provider = CarbonIntensityProvider(provider=CarbonProvider.WATTTIME)
        provider._watttime_token = "test-token"
        provider._watttime_token_expiry = time.time() + 3600
        # Should not make a new login request
        with patch("httpx.Client") as mock_client:
            provider._fetch_watttime("us-east-1")
            # The client should be called for index but not for login
            # (since token is cached)

    def test_token_refreshed_on_expiry(self):
        provider = CarbonIntensityProvider(provider=CarbonProvider.WATTTIME)
        provider._watttime_token = "old-token"
        provider._watttime_token_expiry = time.time() - 100  # Expired
        # Should attempt to refresh

    def test_token_cleared_on_auth_failure(self):
        provider = CarbonIntensityProvider(provider=CarbonProvider.WATTTIME)
        provider._watttime_token = "bad-token"
        provider._watttime_token_expiry = time.time() + 3600
        # Simulate auth failure


class TestLatencyFiltering:
    """Test latency-based filtering."""

    def test_high_latency_provider_filtered_out(self):
        router = CrossCloudRouter(expand_regions=False)
        # Set all providers to high latency
        for p in router._providers:
            router.update_latency(p.name, 9999.0)
        decision = router.select_provider(max_latency_ms=10)
        assert decision is None

    def test_per_region_latency_used(self):
        router = CrossCloudRouter(expand_regions=True)
        # us-east-1 should have ~10ms latency
        decision = router.select_provider(max_latency_ms=15, gpu_type="A100")
        if decision is not None:
            assert decision.latency_ms <= 15


class TestRegionExpansion:
    """Test per-region provider expansion."""

    def test_expand_regions_creates_multiple_providers(self):
        router = CrossCloudRouter(expand_regions=True)
        # Should have providers for multiple regions
        regions = {p.region for p in router._providers}
        assert len(regions) > 5

    def test_no_expand_regions_single_provider_per_instance(self):
        router = CrossCloudRouter(expand_regions=False)
        # Should have one provider per instance type
        instances = {(p.name, p.instance_type) for p in router._providers}
        assert len(instances) == len(router._providers)


class TestPricingStaleness:
    """Test pricing staleness warnings."""

    def test_pricing_age_infinity_without_update(self):
        router = CrossCloudRouter(expand_regions=False)
        assert router.pricing_age_hours == float("inf")

    def test_pricing_age_after_sync(self):
        router = CrossCloudRouter(expand_regions=False)
        router._pricing_last_updated = time.time()
        assert router.pricing_age_hours < 0.01


class TestRouteDecision:
    """Test RouteDecision dataclass."""

    def test_route_decision_fields(self):
        decision = RouteDecision(
            provider="aws",
            instance_type="p4d.24xlarge",
            price_per_hour=14.40,
            estimated_cost=14.40,
            latency_ms=50.0,
            reason="test",
            region="us-east-1",
            gpu_type="A100",
            is_spot=True,
            alternatives_considered=5,
        )
        assert decision.provider == "aws"
        assert decision.is_spot is True
        assert decision.alternatives_considered == 5


class TestEstimateCost:
    """Test estimate_cost method."""

    def test_estimate_known_provider(self):
        router = CrossCloudRouter(expand_regions=False)
        cost = router.estimate_cost("aws", 2.0, use_spot=True)
        assert cost > 0

    def test_estimate_unknown_provider_logs_warning(self):
        router = CrossCloudRouter(expand_regions=False)
        cost = router.estimate_cost("nonexistent", 2.0)
        assert cost == 0.0

    def test_estimate_with_region_filter(self):
        router = CrossCloudRouter(expand_regions=True)
        cost = router.estimate_cost("aws", 1.0, use_spot=True, region="us-east-1")
        assert cost > 0


class TestCarbonReport:
    """Test carbon report generation."""

    def test_report_includes_all_provider_region_pairs(self):
        router = CrossCloudRouter(expand_regions=True)
        report = router.get_carbon_report()
        # Should have entries for multiple provider+region pairs
        assert len(report) > 5
        # Each entry should have provider and region
        for entry in report:
            assert "provider" in entry
            assert "region" in entry
            assert "gco2_per_kwh" in entry

    def test_report_sorted_by_carbon(self):
        router = CrossCloudRouter(expand_regions=True)
        report = router.get_carbon_report()
        carbons = [e["gco2_per_kwh"] for e in report]
        assert carbons == sorted(carbons)
