"""Tests for distllm.dist.cloud_selector."""
from __future__ import annotations

import pytest

from distllm.dist.cloud_selector import CloudRegionSelector, RegionOffer


class TestRegionOffer:
    """Tests for the RegionOffer dataclass."""

    def test_default_values(self) -> None:
        """spot_available defaults to True, latency_ms and carbon_intensity to 0."""
        offer = RegionOffer(
            provider="aws",
            region="us-east-1",
            gpu_type="A100",
            gpu_count=8,
            price_per_hour=32.44,
            spot_price_per_hour=9.73,
        )
        assert offer.spot_available is True
        assert offer.latency_ms == 0.0
        assert offer.carbon_intensity == 0.0

    def test_all_fields_stored(self) -> None:
        """All dataclass fields are stored and accessible."""
        offer = RegionOffer(
            provider="gcp",
            region="us-central1",
            gpu_type="H100",
            gpu_count=4,
            price_per_hour=37.84,
            spot_price_per_hour=11.35,
            spot_available=False,
            latency_ms=15.2,
            carbon_intensity=420.0,
        )
        assert offer.provider == "gcp"
        assert offer.region == "us-central1"
        assert offer.gpu_type == "H100"
        assert offer.gpu_count == 4
        assert offer.price_per_hour == 37.84
        assert offer.spot_price_per_hour == 11.35
        assert offer.spot_available is False
        assert offer.latency_ms == 15.2
        assert offer.carbon_intensity == 420.0

    def test_is_cost_effective_positive(self) -> None:
        """is_cost_effective is True when price_per_hour > 0."""
        offer = RegionOffer(
            provider="aws", region="us-east-1", gpu_type="A100",
            gpu_count=8, price_per_hour=1.0, spot_price_per_hour=0.5,
        )
        assert offer.is_cost_effective is True

    def test_is_cost_effective_zero(self) -> None:
        """is_cost_effective is False when price_per_hour is 0."""
        offer = RegionOffer(
            provider="aws", region="us-east-1", gpu_type="A100",
            gpu_count=8, price_per_hour=0.0, spot_price_per_hour=0.0,
        )
        assert offer.is_cost_effective is False

    def test_is_cost_effective_negative(self) -> None:
        """is_cost_effective is False when price_per_hour is negative."""
        offer = RegionOffer(
            provider="aws", region="us-east-1", gpu_type="A100",
            gpu_count=8, price_per_hour=-1.0, spot_price_per_hour=-0.5,
        )
        assert offer.is_cost_effective is False


class TestStaticPricing:
    """Tests for static fallback pricing data and helper methods."""

    @pytest.mark.parametrize("provider", ["aws", "gcp", "azure"])
    def test_static_offers_known_provider(self, provider: str) -> None:
        """Static offers for known providers return RegionOffer instances."""
        offers = CloudRegionSelector._static_offers(provider)
        assert len(offers) > 0
        assert all(isinstance(o, RegionOffer) for o in offers)
        assert all(o.provider == provider for o in offers)

    def test_static_offers_unknown_provider(self) -> None:
        """Unknown providers yield an empty list."""
        assert CloudRegionSelector._static_offers("unknown") == []

    @pytest.mark.parametrize(
        ("gpu_type", "required_gb", "expected"),
        [
            ("A100", 80.0, True),
            ("A100", 40.0, True),
            ("A100-40GB", 40.0, True),
            ("A100-40GB", 41.0, False),
            ("H100", 80.0, True),
            ("H100", 81.0, False),
            ("H200", 141.0, True),
            ("H200", 142.0, False),
            ("L40S", 48.0, True),
            ("L40S", 49.0, False),
            ("L4", 24.0, True),
            ("L4", 25.0, False),
            ("V100", 32.0, True),
            ("V100", 33.0, False),
            ("T4", 16.0, True),
            ("T4", 17.0, False),
        ],
    )
    def test_gpu_matches(
        self, gpu_type: str, required_gb: float, expected: bool,
    ) -> None:
        """_gpu_matches correctly checks GPU memory sufficiency."""
        assert CloudRegionSelector._gpu_matches(gpu_type, required_gb) is expected

    def test_gpu_matches_unknown_type_defaults_80gb(self) -> None:
        """Unknown GPU types default to 80 GB available memory."""
        assert CloudRegionSelector._gpu_matches("UnknownGPU", 80.0) is True
        assert CloudRegionSelector._gpu_matches("UnknownGPU", 80.1) is False

    def test_gpu_matches_zero_requirement(self) -> None:
        """Zero required memory always matches."""
        assert CloudRegionSelector._gpu_matches("T4", 0.0) is True


class TestCloudRegionSelectorInit:
    """Tests for CloudRegionSelector construction."""

    def test_default_init(self) -> None:
        """Default constructor sets expected values."""
        selector = CloudRegionSelector()
        assert selector._providers == ["aws", "gcp", "azure"]
        assert selector._prefer_spot is True
        assert selector._cache_ttl_s == 3600.0
        assert selector._api_keys == {}
        assert selector._cached_offers is None

    def test_custom_providers(self) -> None:
        """Custom provider list is stored."""
        selector = CloudRegionSelector(providers=["aws"])
        assert selector._providers == ["aws"]

    def test_providers_none_defaults_all(self) -> None:
        """None providers defaults to all three cloud providers."""
        selector = CloudRegionSelector(providers=None)
        assert selector._providers == ["aws", "gcp", "azure"]

    def test_providers_empty_list_defaults_all(self) -> None:
        """Empty provider list defaults to all three cloud providers."""
        selector = CloudRegionSelector(providers=[])
        assert selector._providers == ["aws", "gcp", "azure"]

    def test_prefer_spot_false(self) -> None:
        """prefer_spot=False is stored correctly."""
        selector = CloudRegionSelector(prefer_spot=False)
        assert selector._prefer_spot is False

    def test_custom_cache_ttl(self) -> None:
        """Custom cache TTL is stored."""
        selector = CloudRegionSelector(cache_ttl_s=60.0)
        assert selector._cache_ttl_s == 60.0

    def test_api_keys_stored(self) -> None:
        """API keys dict is stored."""
        selector = CloudRegionSelector(api_keys={"aws": "ak", "gcp": "gk"})
        assert selector._api_keys == {"aws": "ak", "gcp": "gk"}


class TestFindCheapestRegion:
    """Tests for CloudRegionSelector.find_cheapest_region."""

    def test_default_params_finds_gcp_us_central1_a100_spot(self) -> None:
        """With default params, GCP us-central1 A100 (cheapest spot at $8.86) wins."""
        selector = CloudRegionSelector()
        best = selector.find_cheapest_region(required_gpu_memory_gb=80.0)
        assert best is not None
        assert best.provider == "gcp"
        assert best.region == "us-central1"
        assert best.gpu_type == "A100"
        assert best.price_per_hour == 29.52
        assert best.spot_price_per_hour == 8.86

    def test_prefer_spot_false_uses_on_demand_sort(self) -> None:
        """prefer_spot=False sorts by on-demand price; GCP us-central1 A100 still cheapest."""
        selector = CloudRegionSelector(prefer_spot=False)
        best = selector.find_cheapest_region(required_gpu_memory_gb=80.0)
        assert best is not None
        assert best.provider == "gcp"
        assert best.region == "us-central1"
        assert best.gpu_type == "A100"
        assert best.price_per_hour == 29.52

    def test_min_gpu_count_zero_matches_all(self) -> None:
        """min_gpu_count=0 includes all offers."""
        selector = CloudRegionSelector()
        best = selector.find_cheapest_region(
            required_gpu_memory_gb=80.0, min_gpu_count=0,
        )
        assert best is not None

    def test_min_gpu_count_too_high_returns_none(self) -> None:
        """min_gpu_count above all available GPU counts returns None."""
        selector = CloudRegionSelector()
        best = selector.find_cheapest_region(
            required_gpu_memory_gb=80.0, min_gpu_count=9,
        )
        assert best is None

    def test_max_price_excludes_all_returns_none(self) -> None:
        """max_price_per_hour below cheapest price returns None."""
        selector = CloudRegionSelector()
        best = selector.find_cheapest_region(
            required_gpu_memory_gb=80.0, max_price_per_hour=1.0,
        )
        assert best is None

    def test_lower_memory_requirement_selects_l40s(self) -> None:
        """At 40 GB requirement L40S (48 GB, $5.62 spot) is cheapest."""
        selector = CloudRegionSelector()
        best = selector.find_cheapest_region(required_gpu_memory_gb=40.0)
        assert best is not None
        assert best.gpu_type == "L40S"
        assert best.region == "us-east-1"
        assert best.provider == "aws"

    def test_model_name_is_accepted_but_ignored(self) -> None:
        """model_name parameter is accepted without error (currently unused)."""
        selector = CloudRegionSelector()
        best = selector.find_cheapest_region(
            model_name="llama-70b", required_gpu_memory_gb=80.0,
        )
        assert best is not None

    def test_unmet_memory_requirement_returns_none(self) -> None:
        """required_gpu_memory_gb above all known GPU memory returns None."""
        selector = CloudRegionSelector()
        best = selector.find_cheapest_region(required_gpu_memory_gb=999.0)
        assert best is None

    def test_returns_region_offer_instance(self) -> None:
        """Return type is RegionOffer."""
        selector = CloudRegionSelector()
        best = selector.find_cheapest_region()
        assert isinstance(best, RegionOffer)

    def test_infinite_max_price_includes_all(self) -> None:
        """float('inf') max_price does not exclude anything."""
        selector = CloudRegionSelector()
        best_on = selector.find_cheapest_region(
            required_gpu_memory_gb=80.0, max_price_per_hour=float("inf"),
        )
        best_off = selector.find_cheapest_region(
            required_gpu_memory_gb=80.0, max_price_per_hour=1e12,
        )
        # Both should find the same result
        assert best_on is not None
        assert best_off is not None
        assert best_on.provider == best_off.provider

    def test_no_providers_configured_returns_none(self) -> None:
        """A selector with an unknown provider returns None."""
        selector = CloudRegionSelector(providers=["does-not-exist"])
        assert selector.find_cheapest_region() is None


class TestListRegions:
    """Tests for CloudRegionSelector.list_regions."""

    def test_80gb_lists_only_a100_and_h100(self) -> None:
        """At 80 GB requirement only A100 and H100 pass the filter."""
        selector = CloudRegionSelector()
        regions = selector.list_regions(required_gpu_memory_gb=80.0)
        assert all(r["gpu_type"] in ("A100", "H100") for r in regions)
        # 5 AWS + 3 GCP + 2 Azure = 10
        assert len(regions) == 10

    def test_40gb_includes_l40s(self) -> None:
        """At 40 GB requirement L40S (48 GB) is also included."""
        selector = CloudRegionSelector()
        regions = selector.list_regions(required_gpu_memory_gb=40.0)
        assert any(r["gpu_type"] == "L40S" for r in regions)
        # 6 AWS + 3 GCP + 2 Azure = 11
        assert len(regions) == 11

    def test_very_high_memory_returns_empty(self) -> None:
        """Impossibly high memory requirement returns an empty list."""
        selector = CloudRegionSelector()
        regions = selector.list_regions(required_gpu_memory_gb=999.0)
        assert regions == []

    def test_zero_memory_returns_all(self) -> None:
        """Zero memory requirement includes all offers."""
        selector = CloudRegionSelector()
        regions = selector.list_regions(required_gpu_memory_gb=0.0)
        assert len(regions) == 11  # same as 40 GB since no sub-48GB types exist

    def test_returned_dict_has_expected_keys(self) -> None:
        """Each entry has the expected keys."""
        selector = CloudRegionSelector()
        regions = selector.list_regions()
        assert len(regions) > 0
        expected = {"provider", "region", "gpu_type", "gpu_count",
                    "price_per_hour", "spot_price"}
        for r in regions:
            assert set(r.keys()) == expected


class TestProviderFiltering:
    """Tests with specific provider selections."""

    def test_aws_only(self) -> None:
        """AWS-only selector finds cheapest AWS A100 spot at us-east-1."""
        selector = CloudRegionSelector(providers=["aws"])
        best = selector.find_cheapest_region(required_gpu_memory_gb=80.0)
        assert best is not None
        assert best.provider == "aws"
        assert best.region == "us-east-1"
        assert best.gpu_type == "A100"

    def test_gcp_only(self) -> None:
        """GCP-only selector finds cheapest GCP A100 at us-central1."""
        selector = CloudRegionSelector(providers=["gcp"])
        best = selector.find_cheapest_region(required_gpu_memory_gb=80.0)
        assert best is not None
        assert best.provider == "gcp"
        assert best.region == "us-central1"
        assert best.gpu_type == "A100"

    def test_azure_only(self) -> None:
        """Azure-only selector finds cheapest Azure A100 at eastus."""
        selector = CloudRegionSelector(providers=["azure"])
        best = selector.find_cheapest_region(required_gpu_memory_gb=80.0)
        assert best is not None
        assert best.provider == "azure"
        assert best.region == "eastus"
        assert best.gpu_type == "A100"

    def test_nonexistent_provider_returns_none(self) -> None:
        """Selector with only unknown providers returns None."""
        selector = CloudRegionSelector(providers=["nonexistent"])
        assert selector.find_cheapest_region() is None

    def test_mixed_known_and_unknown_providers(self) -> None:
        """Unknown providers do not interfere with known ones."""
        selector = CloudRegionSelector(providers=["nonexistent", "gcp"])
        best = selector.find_cheapest_region(required_gpu_memory_gb=80.0)
        assert best is not None
        assert best.provider == "gcp"


class TestCaching:
    """Tests for internal pricing data caching."""

    def test_consecutive_calls_return_same_offer_object(self) -> None:
        """The same RegionOffer object is returned on repeated calls (cache hit)."""
        selector = CloudRegionSelector()
        offer1 = selector.find_cheapest_region()
        offer2 = selector.find_cheapest_region()
        assert offer1 is offer2

    def test_different_selectors_have_independent_caches(self) -> None:
        """Two selectors produce distinct RegionOffer objects."""
        s1 = CloudRegionSelector()
        s2 = CloudRegionSelector()
        o1 = s1.find_cheapest_region()
        o2 = s2.find_cheapest_region()
        assert o1 is not o2

    def test_get_offers_returns_populated_list(self) -> None:
        """_get_offers returns a non-empty list of RegionOffer instances."""
        selector = CloudRegionSelector()
        offers = selector._get_offers()
        assert isinstance(offers, list)
        assert len(offers) > 0
        assert all(isinstance(o, RegionOffer) for o in offers)
