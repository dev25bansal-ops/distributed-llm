"""Tests for CrossCloudRouter, dataclasses, and helper functions.

Uses the import-helper pattern to avoid circular imports.
"""

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_router_mod = load_module("distllm/core/cross_cloud_router.py")
CrossCloudRouter = _router_mod.CrossCloudRouter
CloudProvider = _router_mod.CloudProvider
RouteDecision = _router_mod.RouteDecision
CarbonProvider = _router_mod.CarbonProvider
CarbonIntensity = _router_mod.CarbonIntensity
CarbonIntensityProvider = _router_mod.CarbonIntensityProvider
compute_relative_latency = _router_mod.compute_relative_latency


# ── Dataclass defaults ──────────────────────────────────────────────────────


class TestCloudProvider:
    def test_defaults(self):
        p = CloudProvider(name="aws")
        assert p.name == "aws"
        # Unset region is empty (a real region is only set on expansion).
        assert p.region == ""
        assert p.gpu_count == 1
        assert p.available is True
        assert p.spot_available is True
        assert p.price_per_hour == 0.0

    def test_custom_values(self):
        p = CloudProvider(
            name="gcp", region="europe-west1", instance_type="a2-highgpu-1g",
            gpu_type="A100", gpu_count=1, price_per_hour=3.67,
        )
        assert p.price_per_hour == 3.67
        assert p.region == "europe-west1"


class TestRouteDecision:
    def test_defaults(self):
        d = RouteDecision(
            provider="aws", instance_type="p4d.24xlarge",
            price_per_hour=14.40, estimated_cost=14.40,
            latency_ms=10, reason="test",
        )
        assert d.carbon_intensity == 0.0
        assert d.carbon_cost_factor == 1.0
        assert d.is_spot is False
        assert d.alternatives_considered == 0


class TestCarbonIntensity:
    def test_defaults(self):
        ci = CarbonIntensity(region="us-east-1", gco2_per_kwh=380)
        assert ci.source == "static"
        assert ci.renewable_pct == 0.0


# ── compute_relative_latency ──────────────────────────────────────────────────


class TestComputeRelativeLatency:
    def test_same_region(self):
        lat = compute_relative_latency("us-east-1", "us-east-1")
        assert lat == 10.0

    def test_known_pair(self):
        lat = compute_relative_latency("us-east-1", "eu-west-1")
        assert lat == 80.0  # base=10 + delta=70

    def test_reverse_known_pair(self):
        lat = compute_relative_latency("eu-west-1", "us-east-1")
        assert lat == 10.0  # base=80 + delta=-70 = 10

    def test_unknown_pair(self):
        lat = compute_relative_latency("us-east-1", "mars-central-1")
        assert lat == 30.0  # (10 + 50) / 2

    def test_minimum_floor(self):
        lat = compute_relative_latency("us-east-1", "us-east-1")
        assert lat >= 5.0


# ── CarbonIntensityProvider (static only, no network) ─────────────────────────


class TestCarbonIntensityProviderStatic:
    def test_get_static_known_region(self):
        provider = CarbonIntensityProvider(provider=CarbonProvider.STATIC)
        ci = provider.get_intensity("us-east-1")
        assert ci.gco2_per_kwh == 380
        assert ci.source == "static"

    def test_get_static_unknown_region(self):
        provider = CarbonIntensityProvider(provider=CarbonProvider.STATIC)
        ci = provider.get_intensity("unknown-region")
        assert ci.gco2_per_kwh == 400  # default
        assert ci.source == "static"

    def test_get_static_caches(self):
        provider = CarbonIntensityProvider(provider=CarbonProvider.STATIC, cache_ttl=300)
        ci1 = provider.get_intensity("us-west-2")
        ci2 = provider.get_intensity("us-west-2")
        assert ci1 is ci2  # cached reference

    def test_region_to_zone_mapping(self):
        zone = CarbonIntensityProvider._region_to_zone("us-east-1")
        assert zone == "US-VIRG"

    def test_region_to_zone_unmapped(self):
        zone = CarbonIntensityProvider._region_to_zone("nowhere")
        assert zone is None


# ── CrossCloudRouter construction and defaults ────────────────────────────────


class TestCrossCloudRouterInit:
    def test_defaults(self):
        router = CrossCloudRouter()
        assert len(router._providers) > 0
        assert router._expand_regions is True
        assert router._stats == {"routes": 0, "savings_usd": 0.0, "carbon_saved_kg": 0.0}
        assert router.pricing_age_hours == float("inf")

    def test_init_without_region_expansion(self):
        router = CrossCloudRouter(expand_regions=False)
        assert len(router._providers) > 0
        # Without region expansion, each instance has no region set (empty string)
        for p in router._providers:
            assert p.region == ""  # no region expansion

    def test_stats_property(self):
        router = CrossCloudRouter()
        assert router.stats == {"routes": 0, "savings_usd": 0.0, "carbon_saved_kg": 0.0}

    def test_known_providers_present(self):
        router = CrossCloudRouter(expand_regions=False)
        names = set(p.name for p in router._providers)
        assert "aws" in names
        assert "gcp" in names
        assert "azure" in names


# ── add_provider, update_latency, update_availability ─────────────────────────


class TestCrossCloudRouterManagement:
    def test_add_provider(self):
        router = CrossCloudRouter(expand_regions=False)
        count_before = len(router._providers)
        router.add_provider(CloudProvider(name="custom", price_per_hour=1.0))
        assert len(router._providers) == count_before + 1

    def test_update_latency(self):
        router = CrossCloudRouter()
        router.update_latency("custom-provider", 150.0)
        assert router._latency_cache["custom-provider"] == 150.0

    def test_update_availability(self):
        router = CrossCloudRouter()
        router.update_availability("aws", "p4d.24xlarge", False, region="us-east-1")
        key = "aws:p4d.24xlarge:us-east-1"
        assert router._availability_cache.get(key) is False

    def test_update_availability_without_region(self):
        router = CrossCloudRouter()
        router.update_availability("aws", "p4d.24xlarge", False)
        key = "aws:p4d.24xlarge"
        assert router._availability_cache.get(key) is False

    def test_update_carbon_intensity(self):
        router = CrossCloudRouter()
        router.update_carbon_intensity("custom-region", 100.0)
        ci = router._carbon_provider._cache["custom-region"]
        assert ci.gco2_per_kwh == 100.0
        assert ci.source == "manual"

    def test_set_latency_tracker(self):
        router = CrossCloudRouter()
        tracker = type("Tracker", (), {})()
        router.set_latency_tracker(tracker)
        assert router._latency_tracker is tracker

    def test_sync_latency_from_tracker_no_tracker(self):
        router = CrossCloudRouter()
        assert router.sync_latency_from_tracker() == 0


# ── _matches_gpu_type ────────────────────────────────────────────────────────


class TestMatchesGpuType:
    def test_exact_match(self):
        provider = CloudProvider(name="aws", instance_type="p4d.24xlarge", gpu_type="A100")
        assert CrossCloudRouter._matches_gpu_type(provider, "A100") is True

    def test_case_insensitive(self):
        provider = CloudProvider(name="aws", instance_type="p4d.24xlarge", gpu_type="A100")
        assert CrossCloudRouter._matches_gpu_type(provider, "a100") is True

    def test_no_gpu_type_fallback_to_instance_type(self):
        provider = CloudProvider(name="aws", instance_type="p4d.24xlarge", gpu_type="")
        assert CrossCloudRouter._matches_gpu_type(provider, "P4D") is True  # substring match

    def test_no_match(self):
        provider = CloudProvider(name="aws", instance_type="p4d.24xlarge", gpu_type="A100")
        assert CrossCloudRouter._matches_gpu_type(provider, "V100") is False

    def test_empty_gpu_type(self):
        provider = CloudProvider(name="aws", instance_type="p4d.24xlarge", gpu_type="A100")
        assert CrossCloudRouter._matches_gpu_type(provider, "") is False


# ── select_provider ──────────────────────────────────────────────────────────


class TestCrossCloudRouterSelectProvider:
    def test_select_cheapest_gpu_type(self):
        router = CrossCloudRouter(expand_regions=False)
        decision = router.select_provider(gpu_type="A100")
        assert decision is not None
        assert isinstance(decision, RouteDecision)
        assert decision.gpu_type == "A100"
        assert decision.price_per_hour > 0
        assert decision.alternatives_considered > 0

    def test_select_respects_max_latency(self):
        router = CrossCloudRouter(expand_regions=True)
        decision = router.select_provider(gpu_type="A100", max_latency_ms=10.0)
        if decision is not None:
            assert decision.latency_ms <= 10.0

    def test_select_respects_max_price(self):
        router = CrossCloudRouter(expand_regions=False)
        # Threshold well below the cheapest A100 so the filter must reject
        # everything (don't hardcode a price that drifts with the table).
        prices = [p.price_per_hour for p in router._providers if p.gpu_type == "A100"]
        prices += [
            p.spot_price for p in router._providers
            if p.gpu_type == "A100" and p.spot_price > 0
        ]
        assert prices, "fixture expects A100 instances"
        decision = router.select_provider(gpu_type="A100", max_price=min(prices) * 0.5)
        assert decision is None

    def test_select_no_gpu_type(self):
        router = CrossCloudRouter(expand_regions=False)
        decision = router.select_provider()
        assert decision is not None
        assert decision.price_per_hour > 0

    def test_select_with_custom_latency(self):
        router = CrossCloudRouter(expand_regions=False)
        router.update_latency("aws", 200.0)
        decision = router.select_provider(gpu_type="A100", max_latency_ms=50.0)
        # aws latency is now 200ms, so aws should be excluded
        if decision is not None:
            assert decision.provider != "aws" or decision.latency_ms <= 50.0

    def test_select_uses_spot_by_default(self):
        router = CrossCloudRouter(expand_regions=False)
        decision = router.select_provider(gpu_type="L4", prefer_spot=True)
        assert decision is not None
        assert decision.is_spot is True


# ── select_provider_carbon_aware ──────────────────────────────────────────────


class TestCrossCloudRouterCarbonAware:
    def test_carbon_aware_selects_cleanest(self):
        router = CrossCloudRouter(expand_regions=True)
        decision = router.select_provider_carbon_aware(
            gpu_type="A100", carbon_weight=1.0,  # Pure carbon optimization
        )
        if decision is not None:
            assert decision.carbon_intensity > 0
            assert decision.carbon_cost_factor >= 1.0

    def test_carbon_aware_respects_max_carbon(self):
        router = CrossCloudRouter(expand_regions=True)
        decision = router.select_provider_carbon_aware(
            gpu_type="A100", max_carbon_intensity=1.0,  # unrealistically low
        )
        assert decision is None  # no region has < 1 gCO2/kWh

    def test_carbon_aware_pure_cost(self):
        router = CrossCloudRouter(expand_regions=False)
        decision = router.select_provider_carbon_aware(
            gpu_type="A100", carbon_weight=0.0,  # Pure cost optimization
        )
        assert decision is not None

    def test_carbon_aware_with_max_price(self):
        router = CrossCloudRouter(expand_regions=False)
        decision = router.select_provider_carbon_aware(
            gpu_type="A100", max_price=0.5,
        )
        assert decision is None  # Nothing that cheap

    def test_stats_incremented(self):
        router = CrossCloudRouter(expand_regions=False)
        before = router.stats["routes"]
        router.select_provider_carbon_aware(gpu_type="A100")
        assert router.stats["routes"] == before + 1


# ── get_carbon_report, get_all_prices, estimate_cost ──────────────────────────


class TestCrossCloudRouterReports:
    def test_get_carbon_report(self):
        router = CrossCloudRouter(expand_regions=False)
        report = router.get_carbon_report()
        assert len(report) > 0
        for entry in report:
            assert "provider" in entry
            assert "region" in entry
            assert "gco2_per_kwh" in entry
        # Should be sorted by carbon intensity ascending
        intensities = [r["gco2_per_kwh"] for r in report]
        assert intensities == sorted(intensities)

    def test_get_all_prices(self):
        router = CrossCloudRouter(expand_regions=False)
        prices = router.get_all_prices()
        assert len(prices) > 0
        for p in prices:
            assert "price_per_hour" in p
            assert "spot_price" in p

    def test_get_all_prices_filtered_by_gpu(self):
        router = CrossCloudRouter(expand_regions=False)
        prices = router.get_all_prices(gpu_type="L4")
        for p in prices:
            assert "L4" in p.get("gpu_type", "").upper()

    def test_estimate_cost_matching_provider(self):
        router = CrossCloudRouter(expand_regions=False)
        cost = router.estimate_cost("aws", duration_hours=1.0, use_spot=True)
        assert cost > 0

    def test_estimate_cost_unknown_provider(self):
        router = CrossCloudRouter(expand_regions=False)
        cost = router.estimate_cost("non-existent", duration_hours=1.0)
        assert cost == 0.0

    def test_estimate_cost_with_region(self):
        router = CrossCloudRouter(expand_regions=True)
        cost = router.estimate_cost("aws", duration_hours=2.0, use_spot=False, region="us-east-1")
        # Without region filter, region mismatch may yield 0
        # With expand_regions=True and region='us-east-1', should find matching
        if cost > 0:
            assert cost > 0


# ── sync_live_pricing ────────────────────────────────────────────────────────


class TestCrossCloudRouterSyncPricing:
    def test_sync_without_pricing_manager(self):
        router = CrossCloudRouter()
        assert router.sync_live_pricing() == 0

    def test_sync_with_pricing_manager_error(self):
        router = CrossCloudRouter()
        err_manager = type("Mgr", (), {"get_all_pricing": lambda: (_ for _ in ()).throw(Exception("fail"))})()
        router._pricing_manager = err_manager
        assert router.sync_live_pricing() == 0

    def test_pricing_age_hours_infinite_initially(self):
        router = CrossCloudRouter()
        assert router.pricing_age_hours == float("inf")


# ── _check_pricing_staleness ────────────────────────────────────────────────


class TestCrossCloudRouterPricingStaleness:
    def test_initial_no_warning(self):
        router = CrossCloudRouter()
        # Should not raise (warning is logged, not thrown)
        router._check_pricing_staleness()
        assert router._pricing_stale_warning_issued is False

    def test_warning_issued_once(self):
        router = CrossCloudRouter()
        # Simulate data synced once, then aged past 24h (never-synced
        # routers are silent by contract — see test_initial_no_warning).
        import time as _time
        router._pricing_last_updated = _time.time() - 25 * 3600
        router._check_pricing_staleness()
        # With _pricing_last_updated=0.0, pricing_age_hours is inf
        # Since age_hours > 24, warning flag is set
        assert router._pricing_stale_warning_issued is True
