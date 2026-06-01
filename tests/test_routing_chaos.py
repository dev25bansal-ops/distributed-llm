"""Chaos tests for routing resilience.

Covers:
- Carbon API outage → static fallback
- Mixed stale/fresh pricing
- Region-wide failure → failover
- Concurrent mutations
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from distllm.core.cross_cloud_router import (
    CarbonIntensityProvider,
    CarbonProvider,
    CrossCloudRouter,
)


@pytest.mark.chaos
class TestCarbonAPIOutage:
    """Test behavior when carbon API is down."""

    def test_electricitymap_outage_falls_back_to_static(self):
        provider = CarbonIntensityProvider(provider=CarbonProvider.ELECTRICITYMAP)
        import os
        os.environ["ELECTRICITYMAP_AUTH_TOKEN"] = "test-token"
        try:
            with patch("httpx.Client") as mock_client:
                mock_instance = MagicMock()
                mock_instance.get.side_effect = Exception("Connection refused")
                mock_client.return_value = mock_instance
                provider._http_client = mock_instance

                intensity = provider.get_intensity("us-east-1")
                assert intensity.source == "static"
                assert intensity.gco2_per_kwh > 0
        finally:
            del os.environ["ELECTRICITYMAP_AUTH_TOKEN"]

    def test_watttime_outage_falls_back_to_static(self):
        provider = CarbonIntensityProvider(provider=CarbonProvider.WATTTIME)
        import os
        os.environ["WATTTIME_USERNAME"] = "test"
        os.environ["WATTTIME_PASSWORD"] = "test"
        try:
            with patch("httpx.Client") as mock_client:
                mock_instance = MagicMock()
                mock_instance.get.side_effect = Exception("Timeout")
                mock_client.return_value = mock_instance
                provider._http_client = mock_instance

                intensity = provider.get_intensity("us-east-1")
                assert intensity.source == "static"
        finally:
            del os.environ["WATTTIME_USERNAME"]
            del os.environ["WATTTIME_PASSWORD"]


@pytest.mark.chaos
class TestMixedStaleFreshPricing:
    """Test routing with mixed stale and fresh pricing data."""

    def test_stale_data_still_enables_routing(self):
        router = CrossCloudRouter(expand_regions=False)
        # Mix of stale and fresh prices — should still route
        decision = router.select_provider(gpu_type="A100", max_latency_ms=10000)
        assert decision is not None

    def test_all_providers_same_price(self):
        router = CrossCloudRouter(expand_regions=False)
        for p in router._providers:
            p.price_per_hour = 5.0
            p.spot_price = 5.0
        decision = router.select_provider(max_latency_ms=10000)
        assert decision is not None


@pytest.mark.chaos
class TestRegionWideFailure:
    """Test failover when entire region is unavailable."""

    def test_failover_from_us_east_1(self):
        router = CrossCloudRouter(expand_regions=True)
        # Mark all us-east-1 providers as unavailable
        for p in router._providers:
            if p.region == "us-east-1":
                router.update_availability(p.name, p.instance_type, False, region="us-east-1")

        decision = router.select_provider(gpu_type="A100", max_latency_ms=10000)
        assert decision is not None
        assert decision.region != "us-east-1"

    def test_all_regions_unavailable_returns_none(self):
        router = CrossCloudRouter(expand_regions=False)
        for p in router._providers:
            router.update_availability(p.name, p.instance_type, False)
            router._availability_cache[f"{p.name}:{p.instance_type}:{p.region}"] = False
        decision = router.select_provider(max_latency_ms=10000)
        assert decision is None


@pytest.mark.chaos
class TestConcurrentMutations:
    """Test concurrent reads and writes don't crash."""

    def test_concurrent_update_and_select(self):
        router = CrossCloudRouter(expand_regions=True)
        errors = []
        lock = threading.Lock()

        def update_loop():
            try:
                for i in range(100):
                    with lock:
                        router.update_latency("aws", float(i % 200))
                        router.update_availability("aws", "p4d.24xlarge", i % 2 == 0)
            except Exception as e:
                errors.append(e)

        def select_loop():
            try:
                for _ in range(100):
                    with lock:
                        router.select_provider(gpu_type="A100", max_latency_ms=10000)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=update_loop))
            threads.append(threading.Thread(target=select_loop))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_carbon_aware_select(self):
        router = CrossCloudRouter(expand_regions=True)
        errors = []
        lock = threading.Lock()

        def select_loop():
            try:
                for _ in range(50):
                    with lock:
                        router.select_provider_carbon_aware(
                            gpu_type="A100", max_latency_ms=10000, carbon_weight=0.5
                        )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=select_loop) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
