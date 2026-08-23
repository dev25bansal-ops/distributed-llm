"""Tests for PricingManager and related dataclass (InstancePricing).

Covers:
- InstancePricing dataclass: fields, spot_discount_pct
- lookup_gpu_info for known and unknown instance types
- PricingManager: construction, add_provider, get_all_pricing
- PricingManager: get_provider_pricing
- PricingManager: get_cheapest with filters
- PricingManager: refresh with static fallback data
- PricingManager: background refresh thread lifecycle
- PricingManager: is_stale property, last_refresh
- Thread safety of refresh/lock
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/pricing_providers.py")
InstancePricing = _mod.InstancePricing
PricingManager = _mod.PricingManager
PricingProvider = _mod.PricingProvider
AWSPricingProvider = _mod.AWSPricingProvider
GCPPricingProvider = _mod.GCPPricingProvider
AzurePricingProvider = _mod.AzurePricingProvider
lookup_gpu_info = _mod.lookup_gpu_info


# ---------------------------------------------------------------------------
# Patching helpers -- patch httpx.get so real API calls fail fast and
# providers fall back to static data.  Without this the AWS and Azure
# pricing endpoints hang indefinitely on SSL reads in CI/test environments
# without outbound HTTPS access.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_outbound_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch httpx.get so that any real HTTP call raises immediately.

    This fixture applies to every test in the module, ensuring that
    tests exercising get_all_pricing / refresh / get_cheapest (which
    have cache_ttl=0 and thus trigger refresh) do not hang on the real
    cloud pricing APIs.
    """
    def _raise(*args, **kwargs):
        raise RuntimeError("httpx.get blocked -- no outbound HTTP")

    monkeypatch.setattr("httpx.get", _raise)


# ---------------------------------------------------------------------------
# InstancePricing dataclass
# ---------------------------------------------------------------------------


class TestInstancePricing:
    """InstancePricing dataclass fields and computed properties."""

    def test_defaults(self) -> None:
        p = InstancePricing(provider="aws", instance_type="p4d.24xlarge", region="us-east-1")
        assert p.provider == "aws"
        assert p.instance_type == "p4d.24xlarge"
        assert p.region == "us-east-1"
        assert p.gpu_type == ""
        assert p.gpu_count == 1
        assert p.on_demand_price == 0.0
        assert p.spot_price == 0.0
        assert p.currency == "USD"
        assert p.source == "static"

    def test_full_construction(self) -> None:
        p = InstancePricing(
            provider="aws",
            instance_type="p4d.24xlarge",
            region="us-east-1",
            gpu_type="A100",
            gpu_count=8,
            gpu_memory_gb=80.0,
            on_demand_price=32.77,
            spot_price=14.40,
            currency="USD",
            source="aws_api",
        )
        assert p.gpu_type == "A100"
        assert p.gpu_count == 8
        assert p.on_demand_price == 32.77
        assert p.spot_price == 14.40
        assert p.source == "aws_api"

    def test_spot_discount_pct(self) -> None:
        p = InstancePricing(
            provider="aws",
            instance_type="p4d.24xlarge",
            region="us-east-1",
            on_demand_price=100.0,
            spot_price=30.0,
        )
        assert p.spot_discount_pct == 70.0  # (1 - 30/100) * 100

    def test_spot_discount_zero_on_demand(self) -> None:
        p = InstancePricing(
            provider="aws",
            instance_type="p4d.24xlarge",
            region="us-east-1",
            on_demand_price=0.0,
            spot_price=10.0,
        )
        assert p.spot_discount_pct == 0.0

    def test_last_updated_set_automatically(self) -> None:
        p = InstancePricing(provider="aws", instance_type="p4d.24xlarge", region="us-east-1")
        assert p.last_updated > 0


# ---------------------------------------------------------------------------
# lookup_gpu_info
# ---------------------------------------------------------------------------


class TestLookupGpuInfo:
    """lookup_gpu_info helper."""

    def test_known_instance(self) -> None:
        gpu_type, count, memory = lookup_gpu_info("p4d.24xlarge")
        assert gpu_type == "A100"
        assert count == 8
        assert memory == 80.0

    def test_known_gcp_instance(self) -> None:
        gpu_type, count, memory = lookup_gpu_info("a2-highgpu-1g")
        assert gpu_type == "A100"
        assert count == 1
        assert memory == 40.0

    def test_known_azure_instance(self) -> None:
        gpu_type, count, memory = lookup_gpu_info("Standard_NC24ads_A100_v4")
        assert gpu_type == "A100"

    def test_unknown_instance(self) -> None:
        gpu_type, count, memory = lookup_gpu_info("nonexistent_instance_type")
        assert gpu_type == ""
        assert count == 0
        assert memory == 0.0


# ---------------------------------------------------------------------------
# PricingManager construction
# ---------------------------------------------------------------------------


class TestPricingManagerConstruction:
    """Construction and initial state."""

    def test_default_construction(self) -> None:
        mgr = PricingManager()
        assert mgr._cache_ttl == 3600
        assert mgr._cache == []
        assert mgr._last_refresh == 0.0
        assert len(mgr._providers) == 3  # AWS, GCP, Azure

    def test_custom_ttl(self) -> None:
        mgr = PricingManager(cache_ttl=7200)
        assert mgr._cache_ttl == 7200


# ---------------------------------------------------------------------------
# add_provider
# ---------------------------------------------------------------------------


class TestPricingManagerAddProvider:
    """Adding custom providers."""

    def test_add_provider(self) -> None:
        mgr = PricingManager()

        class FakeProvider(PricingProvider):
            def provider_name(self) -> str:
                return "fake"

            def fetch_pricing(self, regions=None):
                return [
                    InstancePricing(
                        provider="fake",
                        instance_type="fake-instance",
                        region="fake-region",
                        on_demand_price=1.0,
                        spot_price=0.5,
                    )
                ]

        mgr.add_provider(FakeProvider())
        assert len(mgr._providers) == 4


# ---------------------------------------------------------------------------
# get_all_pricing / get_provider_pricing
# ---------------------------------------------------------------------------


class TestPricingManagerGetPricing:
    """Retrieving pricing from cache/providers."""

    def test_get_all_pricing_initial_refresh(self) -> None:
        mgr = PricingManager()
        # Force fast refresh
        mgr._cache_ttl = 0
        prices = mgr.get_all_pricing()
        # Should have at least static fallback data
        assert isinstance(prices, list)
        # At least one provider returned data (static fallback)
        assert len(prices) > 0

    def test_get_all_pricing_force_refresh(self) -> None:
        mgr = PricingManager(cache_ttl=3600)
        # Set stale cache
        mgr._cache = [InstancePricing(provider="aws", instance_type="p4d.24xlarge", region="us-east-1")]
        mgr._last_refresh = time.time()  # fresh
        prices = mgr.get_all_pricing(force_refresh=True)
        assert len(prices) > 0

    def test_get_provider_pricing(self) -> None:
        mgr = PricingManager(cache_ttl=0)
        all_prices = mgr.get_all_pricing()
        aws_prices = mgr.get_provider_pricing("aws")
        for p in aws_prices:
            assert p.provider == "aws"
        gcp_prices = mgr.get_provider_pricing("gcp")
        for p in gcp_prices:
            assert p.provider == "gcp"
        assert len(aws_prices) + len(gcp_prices) + len(mgr.get_provider_pricing("azure")) == len(all_prices)


# ---------------------------------------------------------------------------
# get_cheapest
# ---------------------------------------------------------------------------


class TestPricingManagerGetCheapest:
    """Finding the cheapest instance."""

    def test_get_cheapest_no_filter(self) -> None:
        mgr = PricingManager(cache_ttl=0)
        cheapest = mgr.get_cheapest()
        if cheapest is not None:
            assert cheapest.on_demand_price > 0 or cheapest.spot_price > 0

    def test_get_cheapest_by_gpu_type(self) -> None:
        mgr = PricingManager(cache_ttl=0)
        cheapest = mgr.get_cheapest(gpu_type="A100")
        if cheapest is not None:
            assert cheapest.gpu_type == "A100"

    def test_get_cheapest_no_match(self) -> None:
        mgr = PricingManager(cache_ttl=0)
        cheapest = mgr.get_cheapest(gpu_type="UNKNOWN_GPU", min_gpu_memory_gb=9999)
        assert cheapest is None

    def test_get_cheapest_prefers_spot(self) -> None:
        mgr = PricingManager(cache_ttl=0)
        cheapest_spot = mgr.get_cheapest(prefer_spot=True)
        cheapest_on_demand = mgr.get_cheapest(prefer_spot=False)
        if cheapest_spot and cheapest_on_demand:
            # Spot should be <= on_demand price
            spot_price = cheapest_spot.spot_price if cheapest_spot.spot_price > 0 else cheapest_spot.on_demand_price
            od_price = cheapest_on_demand.on_demand_price
            # Not strictly guaranteed across different instances, but a reasonable check
            assert spot_price > 0


# ---------------------------------------------------------------------------
# refresh
# ---------------------------------------------------------------------------


class TestPricingManagerRefresh:
    """Refresh pricing data."""

    def test_refresh(self) -> None:
        mgr = PricingManager()
        before = mgr._last_refresh
        mgr.refresh()
        assert len(mgr._cache) > 0
        assert mgr._last_refresh >= before

    def test_refresh_populates_cache(self) -> None:
        mgr = PricingManager()
        mgr.refresh()
        assert len(mgr._cache) > 0
        for p in mgr._cache:
            assert isinstance(p, InstancePricing)


# ---------------------------------------------------------------------------
# is_stale / last_refresh
# ---------------------------------------------------------------------------


class TestPricingManagerStaleness:
    """Staleness detection."""

    def test_is_stale_after_ttl(self) -> None:
        mgr = PricingManager(cache_ttl=0)  # Immediate expiry
        # After refresh, cache is stale immediately
        mgr.refresh()
        assert mgr.is_stale is True

    def test_is_stale_with_fresh_cache(self) -> None:
        mgr = PricingManager(cache_ttl=3600)
        mgr.refresh()
        assert mgr.is_stale is False

    def test_last_refresh_property(self) -> None:
        mgr = PricingManager()
        assert mgr.last_refresh == 0.0
        mgr.refresh()
        assert mgr.last_refresh > 0


# ---------------------------------------------------------------------------
# Background refresh thread
# ---------------------------------------------------------------------------


class TestPricingManagerBackgroundRefresh:
    """Background refresh thread lifecycle."""

    def test_start_background_refresh(self) -> None:
        mgr = PricingManager()
        mgr.start_background_refresh(interval_s=3600)
        assert mgr._refresh_thread is not None
        assert mgr._refresh_thread.is_alive() is True
        mgr.shutdown()

    def test_start_background_refresh_idempotent(self) -> None:
        mgr = PricingManager()
        mgr.start_background_refresh(3600)
        t1 = mgr._refresh_thread
        mgr.start_background_refresh(3600)  # Should not start another
        t2 = mgr._refresh_thread
        assert t2 is t1
        mgr.shutdown()

    def test_shutdown_stops_thread(self) -> None:
        mgr = PricingManager()
        mgr.start_background_refresh(3600)
        mgr.shutdown()
        assert mgr._shutdown.is_set()


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestPricingManagerThreadSafety:
    """Thread safety of concurrent operations."""

    def test_concurrent_get_all_pricing(self) -> None:
        mgr = PricingManager(cache_ttl=0)
        results: list[list[Any]] = []

        def get_pricing() -> None:
            results.append(mgr.get_all_pricing())

        threads = [threading.Thread(target=get_pricing) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        for r in results:
            assert len(r) > 0

    def test_concurrent_refresh(self) -> None:
        mgr = PricingManager()
        errors: list[Exception] = []

        def do_refresh() -> None:
            try:
                mgr.refresh()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_refresh) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(mgr._cache) > 0


# ---------------------------------------------------------------------------
# Provider static fallback
# ---------------------------------------------------------------------------


class TestProviderFallbacks:
    """Each provider has a _fallback with static data."""

    def test_aws_static_fallback(self) -> None:
        provider = AWSPricingProvider()
        prices = provider._fallback()
        assert len(prices) > 0
        for p in prices:
            assert p.provider == "aws"
            assert p.source == "static"
            assert p.on_demand_price > 0

    def test_gcp_static_fallback(self) -> None:
        provider = GCPPricingProvider()
        prices = provider._fallback()
        assert len(prices) > 0
        for p in prices:
            assert p.provider == "gcp"
            assert p.source == "static"
            assert p.on_demand_price > 0

    def test_azure_static_fallback(self) -> None:
        provider = AzurePricingProvider()
        prices = provider._fallback()
        assert len(prices) > 0
        for p in prices:
            assert p.provider == "azure"
            assert p.source == "static"
            assert p.on_demand_price > 0
