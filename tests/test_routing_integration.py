"""Integration tests for end-to-end routing flow.

Covers:
- CrossCloudRouter → CostTracker → Marketplace pipeline
- Carbon API fallback chain
- Pricing refresh lifecycle
- Unified router cloud+peer merge
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from distllm.core.cross_cloud_router import (
    CarbonIntensityProvider,
    CarbonProvider,
    CrossCloudRouter,
    _REGIONAL_CARBON_INTENSITY,
)
from distllm.core.cost_tracker import CostTracker
from distllm.dist.marketplace import Marketplace


class TestEndToEndRoutingFlow:
    """Test full routing → cost → marketplace pipeline."""

    def test_route_then_estimate_cost(self):
        router = CrossCloudRouter(expand_regions=False)
        decision = router.select_provider(gpu_type="A100", max_latency_ms=1000)
        assert decision is not None

        tracker = CostTracker()
        estimate = tracker.estimate_cost(
            input_tokens=100, output_tokens=200,
            gpu_type="A100-80GB",
        )
        assert estimate.estimated_cost_usd > 0

    def test_route_then_marketplace_match(self):
        router = CrossCloudRouter(expand_regions=False)
        marketplace = Marketplace()

        # Import cloud prices
        prices = router.get_all_prices(gpu_type="A100")
        cloud_prices = [
            {
                "provider": p["provider"],
                "instance_type": p["instance_type"],
                "region": p["region"],
                "gpu_type": p.get("gpu_type", "A100"),
                "gpu_count": p.get("gpu_count", 1),
                "gpu_memory_gb": p.get("gpu_memory_gb", 80.0),
                "on_demand_price": p["price_per_hour"],
                "spot_price": p["spot_price"],
                "carbon_intensity": p.get("carbon_gco2_kwh", 0),
            }
            for p in prices
        ]
        marketplace.add_cloud_listings(cloud_prices)

        # Also add a peer listing
        marketplace.create_listing("peer-1", "A100", 80 * 1024**3, 3.0, region="us-east-1")

        # Post a job and match
        job = marketplace.post_job("user-1", "Llama-70B", 40 * 1024**3, 20.0)
        matched = marketplace.match_job(job.job_id)
        assert matched is not None


class TestCarbonAPIFallback:
    """Test carbon API fallback chain."""

    def test_static_fallback_no_env_vars(self):
        provider = CarbonIntensityProvider(provider=CarbonProvider.ELECTRICITYMAP)
        # No ELECTRICITYMAP_AUTH_TOKEN set
        intensity = provider.get_intensity("us-east-1")
        assert intensity.source == "static"
        assert intensity.gco2_per_kwh > 0

    def test_static_data_covers_all_regions(self):
        provider = CarbonIntensityProvider(provider=CarbonProvider.STATIC)
        for region in _REGIONAL_CARBON_INTENSITY:
            intensity = provider.get_intensity(region)
            assert intensity.gco2_per_kwh > 0
            assert intensity.source == "static"

    def test_cache_hit_on_second_call(self):
        provider = CarbonIntensityProvider(provider=CarbonProvider.STATIC, cache_ttl=300)
        first = provider.get_intensity("us-east-1")
        second = provider.get_intensity("us-east-1")
        assert first.timestamp == second.timestamp  # Same cached object

    def test_cache_refresh_after_ttl(self):
        provider = CarbonIntensityProvider(provider=CarbonProvider.STATIC, cache_ttl=0)
        first = provider.get_intensity("us-east-1")
        time.sleep(0.01)
        second = provider.get_intensity("us-east-1")
        assert second.timestamp >= first.timestamp


class TestPricingRefreshLifecycle:
    """Test pricing refresh lifecycle."""

    def test_sync_live_pricing_updates_providers(self):
        router = CrossCloudRouter(expand_regions=False)
        mock_manager = MagicMock()
        mock_manager.get_all_pricing.return_value = [
            MagicMock(
                provider="aws", instance_type="p4d.24xlarge", region="us-east-1",
                on_demand_price=99.99, spot_price=49.99,
                gpu_type="A100", gpu_count=8, gpu_memory_gb=80.0,
            ),
        ]
        router._pricing_manager = mock_manager
        updated = router.sync_live_pricing()
        assert updated > 0
        # Verify the price was updated
        aws_providers = [p for p in router._providers if p.name == "aws" and p.instance_type == "p4d.24xlarge"]
        assert any(p.price_per_hour == 99.99 for p in aws_providers)

    def test_stale_pricing_warning(self):
        router = CrossCloudRouter(expand_regions=False)
        router._pricing_last_updated = time.time() - 100000  # Very old
        router._check_pricing_staleness()
        assert router._pricing_stale_warning_issued is True


class TestUnifiedRouterMerge:
    """Test unified router merging cloud + peer options."""

    def test_unified_router_returns_both_sources(self):
        from distllm.core.unified_router import ComputeSource, UnifiedRouter

        router = UnifiedRouter()
        router.set_cloud_options([
            {"provider": "aws", "instance_type": "p4d.24xlarge", "region": "us-east-1",
             "gpu_type": "A100", "gpu_memory_gb": 80.0, "on_demand_price": 32.77,
             "spot_price": 14.40, "gpu_count": 8},
        ])
        router.set_peer_options([
            MagicMock(
                listing_id="peer-1", provider_id="peer-1", provider_name="peer-1",
                gpu_name="A100", gpu_memory_bytes=80 * 1024**3, gpu_count=1,
                price_per_hour=3.0, region="us-east-1", latency_ms=50.0,
                reputation_score=0.8, uptime_pct=99.0, is_available=True,
                max_concurrent_jobs=1, current_jobs=0,
            ),
        ])

        decision = router.route(gpu_type="A100", max_price=50.0, max_latency_ms=10000)
        assert decision is not None
        # Peer should be cheaper
        assert decision.selected.source == ComputeSource.PEER
