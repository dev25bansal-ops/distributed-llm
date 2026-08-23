"""Tests for distllm.dist.marketplace module.

Covers the full public API: enums, dataclasses, Marketplace, and CloudCostModel.
Deterministic -- no time.sleep, no threading, no network, no GPU.
"""

import time

import pytest

from distllm.dist.marketplace import (
    CloudCostModel,
    GPUListing,
    JobStatus,
    ListingStatus,
    Marketplace,
    MarketplaceJob,
    ProviderEarnings,
)


# ── Enum Tests ──────────────────────────────────────────────────────────────


class TestListingStatus:
    """ListingStatus enum values and construction."""

    def test_members(self) -> None:
        assert ListingStatus.ACTIVE.value == "active"
        assert ListingStatus.BUSY.value == "busy"
        assert ListingStatus.OFFLINE.value == "offline"
        assert ListingStatus.PAUSED.value == "paused"

    def test_from_string(self) -> None:
        assert ListingStatus("active") is ListingStatus.ACTIVE
        assert ListingStatus("busy") is ListingStatus.BUSY
        assert ListingStatus("offline") is ListingStatus.OFFLINE
        assert ListingStatus("paused") is ListingStatus.PAUSED

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            ListingStatus("unknown")

    def test_all_enum_members_covered(self) -> None:
        # All four statuses expected
        assert len(ListingStatus) == 4


class TestJobStatus:
    """JobStatus enum values and construction."""

    def test_members(self) -> None:
        assert JobStatus.OPEN.value == "open"
        assert JobStatus.MATCHED.value == "matched"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_from_string(self) -> None:
        assert JobStatus("open") is JobStatus.OPEN
        assert JobStatus("matched") is JobStatus.MATCHED
        assert JobStatus("running") is JobStatus.RUNNING
        assert JobStatus("completed") is JobStatus.COMPLETED
        assert JobStatus("failed") is JobStatus.FAILED
        assert JobStatus("cancelled") is JobStatus.CANCELLED

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            JobStatus("unknown")


# ── GPUListing Dataclass ────────────────────────────────────────────────────


class TestGPUListing:
    """GPUListing dataclass construction, properties, and edge cases."""

    def test_minimal_constructor(self) -> None:
        listing = GPUListing(listing_id="lid-1", provider_id="p1")
        assert listing.listing_id == "lid-1"
        assert listing.provider_id == "p1"
        assert listing.provider_name == ""
        assert listing.gpu_memory_bytes == 0
        assert listing.gpu_count == 1
        assert listing.price_per_hour == 0.0
        assert listing.status is ListingStatus.ACTIVE
        assert listing.supported_dtypes == ["float16"]
        assert listing.max_batch_size == 8
        assert listing.supports_streaming is True
        assert listing.carbon_intensity == 0.0
        assert listing.reputation_score == 0.5
        assert listing.uptime_pct == 100.0
        assert listing.source == "peer"
        assert listing.tags == []

    def test_full_constructor(self) -> None:
        listing = GPUListing(
            listing_id="lid-2",
            provider_id="p1",
            provider_name="MyGPU",
            gpu_name="A100",
            gpu_memory_bytes=80 * 1024**3,
            gpu_count=4,
            price_per_hour=12.50,
            region="us-east-1",
            carbon_intensity=50.0,
            renewable_pct=75.0,
            reputation_score=0.95,
            tags=["premium", "low-latency"],
        )
        assert listing.provider_name == "MyGPU"
        assert listing.gpu_name == "A100"
        assert listing.gpu_memory_bytes == 80 * 1024**3
        assert listing.gpu_count == 4
        assert listing.price_per_hour == 12.50
        assert listing.region == "us-east-1"
        assert listing.tags == ["premium", "low-latency"]

    def test_constructor_zero_values(self) -> None:
        listing = GPUListing(
            listing_id="lid-3",
            provider_id="p1",
            gpu_memory_bytes=0,
            price_per_hour=0.0,
            reputation_score=0.0,
        )
        assert listing.gpu_memory_bytes == 0
        assert listing.price_per_hour == 0.0
        assert listing.reputation_score == 0.0

    def test_constructor_negative_price(self) -> None:
        listing = GPUListing(
            listing_id="lid-4",
            provider_id="p1",
            price_per_hour=-1.0,
        )
        assert listing.price_per_hour == -1.0

    def test_is_available_true_by_default(self) -> None:
        listing = GPUListing(listing_id="lid-1", provider_id="p1")
        assert listing.is_available is True

    def test_is_available_false_when_busy(self) -> None:
        listing = GPUListing(
            listing_id="lid-1", provider_id="p1",
            status=ListingStatus.BUSY,
        )
        assert listing.is_available is False

    def test_is_available_false_when_offline(self) -> None:
        listing = GPUListing(
            listing_id="lid-1", provider_id="p1",
            status=ListingStatus.OFFLINE,
        )
        assert listing.is_available is False

    def test_is_available_false_when_paused(self) -> None:
        listing = GPUListing(
            listing_id="lid-1", provider_id="p1",
            status=ListingStatus.PAUSED,
        )
        assert listing.is_available is False

    def test_is_available_false_when_slots_full(self) -> None:
        listing = GPUListing(
            listing_id="lid-1", provider_id="p1",
            max_concurrent_jobs=3,
            current_jobs=3,
        )
        assert listing.is_available is False

    def test_is_available_true_with_available_slots(self) -> None:
        listing = GPUListing(
            listing_id="lid-1", provider_id="p1",
            max_concurrent_jobs=3,
            current_jobs=2,
        )
        assert listing.is_available is True

    def test_is_available_active_with_exact_slots(self) -> None:
        listing = GPUListing(
            listing_id="lid-1", provider_id="p1",
            max_concurrent_jobs=1,
            current_jobs=0,
        )
        assert listing.is_available is True
        listing.current_jobs = 1
        assert listing.is_available is False

    def test_effective_score_zero_price_maxes_price_score(self) -> None:
        listing = GPUListing(
            listing_id="lid-1", provider_id="p1",
            price_per_hour=0.0,
            gpu_memory_bytes=80 * 1024**3,
        )
        # price_score = 1.0 / 0.01 = 100.0, rep = 0.5, perf = 0.8
        # effective_score = 100*0.3 + 0.5*0.5 + 0.8*0.2 = 30.41
        assert listing.effective_score == pytest.approx(30.41, rel=1e-3)

    def test_effective_score_free_cheaper_than_free(self) -> None:
        """Two free listings differ only by reputation/memory."""
        free_good = GPUListing(
            listing_id="1", provider_id="p1",
            price_per_hour=0.0, reputation_score=0.9,
            gpu_memory_bytes=80 * 1024**3,
        )
        free_bad = GPUListing(
            listing_id="2", provider_id="p2",
            price_per_hour=0.0, reputation_score=0.1,
            gpu_memory_bytes=16 * 1024**3,
        )
        assert free_good.effective_score > free_bad.effective_score

    def test_effective_score_zero_memory(self) -> None:
        listing = GPUListing(
            listing_id="lid-1", provider_id="p1",
            price_per_hour=5.0,
            gpu_memory_bytes=0,
        )
        assert listing.effective_score > 0

    def test_effective_score_zero_reputation(self) -> None:
        listing = GPUListing(
            listing_id="lid-1", provider_id="p1",
            price_per_hour=5.0,
            gpu_memory_bytes=80 * 1024**3,
            reputation_score=0.0,
        )
        score = listing.effective_score
        assert score >= 0

    def test_effective_score_prefers_lower_price(self) -> None:
        cheap = GPUListing(
            listing_id="1", provider_id="p1",
            price_per_hour=1.0, reputation_score=0.5,
            gpu_memory_bytes=80 * 1024**3,
        )
        expensive = GPUListing(
            listing_id="2", provider_id="p2",
            price_per_hour=100.0, reputation_score=0.5,
            gpu_memory_bytes=80 * 1024**3,
        )
        assert cheap.effective_score > expensive.effective_score

    def test_effective_score_prefers_higher_reputation(self) -> None:
        trusted = GPUListing(
            listing_id="1", provider_id="p1",
            price_per_hour=5.0, reputation_score=0.9,
            gpu_memory_bytes=80 * 1024**3,
        )
        untrusted = GPUListing(
            listing_id="2", provider_id="p2",
            price_per_hour=5.0, reputation_score=0.1,
            gpu_memory_bytes=80 * 1024**3,
        )
        assert trusted.effective_score > untrusted.effective_score

    def test_effective_score_prefers_more_gpu_memory(self) -> None:
        big = GPUListing(
            listing_id="1", provider_id="p1",
            price_per_hour=5.0, reputation_score=0.5,
            gpu_memory_bytes=80 * 1024**3,
        )
        small = GPUListing(
            listing_id="2", provider_id="p2",
            price_per_hour=5.0, reputation_score=0.5,
            gpu_memory_bytes=16 * 1024**3,
        )
        assert big.effective_score > small.effective_score

    def test_effective_score_tie(self) -> None:
        a = GPUListing(
            listing_id="1", provider_id="p1",
            price_per_hour=5.0, reputation_score=0.5,
            gpu_memory_bytes=80 * 1024**3,
        )
        b = GPUListing(
            listing_id="2", provider_id="p2",
            price_per_hour=5.0, reputation_score=0.5,
            gpu_memory_bytes=80 * 1024**3,
        )
        assert a.effective_score == b.effective_score

    def test_created_at_is_set(self) -> None:
        before = time.time()
        listing = GPUListing(listing_id="lid-1", provider_id="p1")
        after = time.time()
        assert before <= listing.created_at <= after

    def test_mutable_defaults_are_independent(self) -> None:
        a = GPUListing(listing_id="a", provider_id="p1", tags=["x"])
        b = GPUListing(listing_id="b", provider_id="p1")
        # Ensure b's default is not contaminated by a's explicit tags
        assert b.tags == []
        # Ensure custom supported_dtypes propagates
        c = GPUListing(
            listing_id="c", provider_id="p1",
            supported_dtypes=["float32", "bfloat16"],
        )
        assert c.supported_dtypes == ["float32", "bfloat16"]


# ── MarketplaceJob Dataclass ────────────────────────────────────────────────


class TestMarketplaceJob:
    """MarketplaceJob dataclass construction, properties, and edge cases."""

    def test_minimal_constructor(self) -> None:
        job = MarketplaceJob(job_id="j-1", requester_id="u1")
        assert job.job_id == "j-1"
        assert job.requester_id == "u1"
        assert job.model_name == ""
        assert job.min_gpu_memory_bytes == 0
        assert job.max_price_per_hour == 0.0
        assert job.status is JobStatus.OPEN
        assert job.priority == 2
        assert job.matched_listing_id == ""
        assert job.tags == []

    def test_full_constructor(self) -> None:
        job = MarketplaceJob(
            job_id="j-2",
            requester_id="u1",
            model_name="Llama-70B",
            min_gpu_memory_bytes=40 * 1024**3,
            max_price_per_hour=10.0,
            priority=0,
            preferred_regions=["us-east-1"],
            tags=["urgent"],
        )
        assert job.model_name == "Llama-70B"
        assert job.priority == 0
        assert job.preferred_regions == ["us-east-1"]

    def test_constructor_boundary_values(self) -> None:
        job = MarketplaceJob(
            job_id="j-3",
            requester_id="u1",
            min_gpu_memory_bytes=0,
            max_price_per_hour=0.0,
            min_reputation=0.0,
            priority=3,
            max_latency_ms=0.0,
            min_uptime_pct=0.0,
        )
        assert job.max_price_per_hour == 0.0
        assert job.min_reputation == 0.0

    def test_duration_hours_not_started(self) -> None:
        job = MarketplaceJob(job_id="j-1", requester_id="u1")
        assert job.duration_hours == 0.0

    def test_duration_hours_completed(self) -> None:
        job = MarketplaceJob(
            job_id="j-1", requester_id="u1",
            started_at=1000.0, completed_at=4600.0,
        )
        assert job.duration_hours == 1.0

    def test_duration_hours_partial(self) -> None:
        job = MarketplaceJob(
            job_id="j-1", requester_id="u1",
            started_at=1000.0, completed_at=2800.0,
        )
        # (2800 - 1000) / 3600 = 1800 / 3600 = 0.5
        assert job.duration_hours == 0.5

    def test_duration_hours_zero_interval(self) -> None:
        job = MarketplaceJob(
            job_id="j-1", requester_id="u1",
            started_at=100.0, completed_at=100.0,
        )
        assert job.duration_hours == 0.0

    def test_default_preferred_regions_empty(self) -> None:
        job = MarketplaceJob(job_id="j-1", requester_id="u1")
        assert job.preferred_regions == []

    def test_default_min_reputation(self) -> None:
        job = MarketplaceJob(job_id="j-1", requester_id="u1")
        assert job.min_reputation == 0.3

    def test_default_max_latency(self) -> None:
        job = MarketplaceJob(job_id="j-1", requester_id="u1")
        assert job.max_latency_ms == 5000.0

    def test_created_at_is_set(self) -> None:
        before = time.time()
        job = MarketplaceJob(job_id="j-1", requester_id="u1")
        after = time.time()
        assert before <= job.created_at <= after


# ── ProviderEarnings Dataclass ──────────────────────────────────────────────


class TestProviderEarnings:
    """ProviderEarnings dataclass construction and defaults."""

    def test_minimal_constructor(self) -> None:
        e = ProviderEarnings(provider_id="p1")
        assert e.provider_id == "p1"
        assert e.total_earnings == 0.0
        assert e.total_gpu_hours == 0.0
        assert e.total_tokens_served == 0
        assert e.total_jobs == 0
        assert e.current_month_earnings == 0.0
        assert e.pending_payout == 0.0
        assert e.last_payout_at == 0.0

    def test_full_constructor(self) -> None:
        e = ProviderEarnings(
            provider_id="p1",
            total_earnings=5000.0,
            total_gpu_hours=100.0,
            total_tokens_served=1_000_000,
            total_jobs=50,
            current_month_earnings=250.0,
            pending_payout=100.0,
            last_payout_at=1_700_000_000.0,
        )
        assert e.total_earnings == 5000.0
        assert e.total_gpu_hours == 100.0
        assert e.total_jobs == 50


# ── Marketplace ─────────────────────────────────────────────────────────────


class TestMarketplaceInit:
    """Marketplace constructor (no backend, no event bus)."""

    def test_default_constructor(self) -> None:
        mp = Marketplace()
        assert mp._listings == {}
        assert mp._jobs == {}
        assert mp._earnings == {}
        assert mp._event_bus is None
        assert mp._backend is None


class TestMarketplaceCreateListing:
    """Marketplace.create_listing()."""

    def test_basic(self) -> None:
        mp = Marketplace()
        listing = mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        assert listing.listing_id.startswith("gpu-")
        assert listing.provider_id == "p1"
        assert listing.gpu_name == "A100"
        assert listing.gpu_memory_bytes == 80 * 1024**3
        assert listing.price_per_hour == 5.0

    def test_with_extra_kwargs(self) -> None:
        mp = Marketplace()
        listing = mp.create_listing(
            "p1", "A100", 80 * 1024**3, 5.0,
            region="eu-west-1",
            carbon_intensity=50.0,
            supports_lora=True,
        )
        assert listing.region == "eu-west-1"
        assert listing.supports_lora is True

    def test_zero_price(self) -> None:
        mp = Marketplace()
        listing = mp.create_listing("p1", "A100", 80 * 1024**3, 0.0)
        assert listing.price_per_hour == 0.0

    def test_zero_memory(self) -> None:
        mp = Marketplace()
        listing = mp.create_listing("p1", "T4", 0, 2.0)
        assert listing.gpu_memory_bytes == 0

    def test_negative_price(self) -> None:
        mp = Marketplace()
        listing = mp.create_listing("p1", "A100", 80 * 1024**3, -5.0)
        assert listing.price_per_hour == -5.0

    def test_empty_provider_id(self) -> None:
        mp = Marketplace()
        listing = mp.create_listing("", "A100", 80 * 1024**3, 5.0)
        assert listing.provider_id == ""

    def test_unique_ids(self) -> None:
        mp = Marketplace()
        l1 = mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        l2 = mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        assert l1.listing_id != l2.listing_id

    def test_creates_earnings_entry(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        earnings = mp.get_provider_earnings("p1")
        assert earnings is not None
        assert earnings.provider_id == "p1"

    def test_reuses_earnings_entry_for_same_provider(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        mp.create_listing("p1", "H100", 80 * 1024**3, 10.0)
        earnings = mp.get_provider_earnings("p1")
        assert earnings is not None
        assert earnings.total_jobs == 0  # no jobs completed yet


class TestMarketplaceUpdateListing:
    """Marketplace.update_listing()."""

    def test_update_price(self) -> None:
        mp = Marketplace()
        listing = mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        updated = mp.update_listing(listing.listing_id, price_per_hour=3.0)
        assert updated is not None
        assert updated.price_per_hour == 3.0
        # Verify in-place mutation
        assert listing.price_per_hour == 3.0

    def test_update_status(self) -> None:
        mp = Marketplace()
        listing = mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        mp.update_listing(listing.listing_id, status=ListingStatus.PAUSED)
        fetched = mp.get_listing(listing.listing_id)
        assert fetched is not None
        assert fetched.status is ListingStatus.PAUSED

    def test_update_nonexistent_returns_none(self) -> None:
        mp = Marketplace()
        assert mp.update_listing("nonexistent", price_per_hour=3.0) is None

    def test_update_empty_updates(self) -> None:
        mp = Marketplace()
        listing = mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        updated = mp.update_listing(listing.listing_id)
        assert updated is not None
        assert updated.price_per_hour == 5.0

    def test_update_unknown_field_ignored(self) -> None:
        mp = Marketplace()
        listing = mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        updated = mp.update_listing(
            listing.listing_id, not_a_real_field="value",
        )
        assert updated is not None

    def test_update_multiple_fields(self) -> None:
        mp = Marketplace()
        listing = mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        mp.update_listing(
            listing.listing_id,
            price_per_hour=2.0,
            gpu_name="A100-80GB",
            region="us-west-2",
        )
        assert listing.price_per_hour == 2.0
        assert listing.gpu_name == "A100-80GB"
        assert listing.region == "us-west-2"


class TestMarketplaceRemoveListing:
    """Marketplace.remove_listing()."""

    def test_remove_existing(self) -> None:
        mp = Marketplace()
        listing = mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        assert mp.remove_listing(listing.listing_id) is True
        assert mp.get_listing(listing.listing_id) is None

    def test_remove_nonexistent(self) -> None:
        mp = Marketplace()
        assert mp.remove_listing("nonexistent") is False

    def test_remove_twice(self) -> None:
        mp = Marketplace()
        listing = mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        assert mp.remove_listing(listing.listing_id) is True
        assert mp.remove_listing(listing.listing_id) is False

    def test_remove_does_not_affect_others(self) -> None:
        mp = Marketplace()
        l1 = mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        l2 = mp.create_listing("p2", "T4", 16 * 1024**3, 2.0)
        mp.remove_listing(l1.listing_id)
        remaining = mp.list_listings()
        assert len(remaining) == 1
        assert remaining[0].listing_id == l2.listing_id


class TestMarketplaceGetListing:
    """Marketplace.get_listing()."""

    def test_existing(self) -> None:
        mp = Marketplace()
        listing = mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        assert mp.get_listing(listing.listing_id) is listing

    def test_nonexistent(self) -> None:
        mp = Marketplace()
        assert mp.get_listing("nonexistent") is None


class TestMarketplaceListListings:
    """Marketplace.list_listings()."""

    def test_empty(self) -> None:
        mp = Marketplace()
        assert mp.list_listings() == []

    def test_all_listings(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        mp.create_listing("p2", "V100", 16 * 1024**3, 2.0)
        assert len(mp.list_listings()) == 2

    def test_filter_by_status(self) -> None:
        mp = Marketplace()
        l1 = mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        mp.update_listing(l1.listing_id, status=ListingStatus.PAUSED)
        mp.create_listing("p2", "V100", 16 * 1024**3, 2.0)

        active = mp.list_listings(status=ListingStatus.ACTIVE)
        assert len(active) == 1
        paused = mp.list_listings(status=ListingStatus.PAUSED)
        assert len(paused) == 1

    def test_filter_by_min_gpu_memory(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        mp.create_listing("p2", "T4", 16 * 1024**3, 2.0)

        result = mp.list_listings(min_gpu_memory=40 * 1024**3)
        assert len(result) == 1
        assert result[0].gpu_name == "A100"

        result = mp.list_listings(min_gpu_memory=0)
        assert len(result) == 2

    def test_filter_by_max_price(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 10.0)
        mp.create_listing("p2", "T4", 16 * 1024**3, 2.0)

        result = mp.list_listings(max_price=5.0)
        assert len(result) == 1
        assert result[0].price_per_hour == 2.0

        result = mp.list_listings(max_price=0.0)  # 0 = no filter
        assert len(result) == 2

    def test_filter_by_region(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0, region="us-east-1")
        mp.create_listing("p2", "A100", 80 * 1024**3, 5.0, region="eu-west-1")

        result = mp.list_listings(region="us-east-1")
        assert len(result) == 1

        result = mp.list_listings(region="")
        assert len(result) == 2

    def test_filter_by_source(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0, source="peer")
        mp.create_listing("p2", "A100", 80 * 1024**3, 5.0, source="cloud")

        peer = mp.list_listings(source="peer")
        assert len(peer) == 1
        cloud = mp.list_listings(source="cloud")
        assert len(cloud) == 1

    def test_filter_by_carbon(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0, carbon_intensity=50.0)
        mp.create_listing("p2", "A100", 80 * 1024**3, 5.0, carbon_intensity=500.0)

        clean = mp.list_listings(max_carbon_gco2_kwh=100.0)
        assert len(clean) == 1
        assert clean[0].carbon_intensity == 50.0

        no_filter = mp.list_listings(max_carbon_gco2_kwh=0.0)
        assert len(no_filter) == 2

    def test_carbon_filter_skips_zero_intensity(self) -> None:
        """Listings with carbon_intensity=0 are excluded when carbon filter > 0."""
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0, carbon_intensity=0.0)
        mp.create_listing("p2", "A100", 80 * 1024**3, 5.0, carbon_intensity=50.0)

        result = mp.list_listings(max_carbon_gco2_kwh=100.0)
        assert len(result) == 1
        assert result[0].carbon_intensity == 50.0

    def test_sorted_by_effective_score_descending(self) -> None:
        mp = Marketplace()
        mp.create_listing("p-cheap", "A100", 80 * 1024**3, 1.0, reputation_score=0.9)
        mp.create_listing("p-expensive", "A100", 80 * 1024**3, 100.0, reputation_score=0.1)

        result = mp.list_listings()
        assert len(result) == 2
        assert result[0].effective_score >= result[1].effective_score

    def test_combined_filters(self) -> None:
        mp = Marketplace()
        mp.create_listing(
            "p1", "A100", 80 * 1024**3, 5.0,
            region="us-east-1", source="peer",
        )
        mp.create_listing(
            "p2", "T4", 16 * 1024**3, 2.0,
            region="us-east-1", source="peer",
        )
        mp.create_listing(
            "p3", "A100", 80 * 1024**3, 5.0,
            region="eu-west-1", source="peer",
        )

        result = mp.list_listings(
            region="us-east-1",
            min_gpu_memory=40 * 1024**3,
        )
        assert len(result) == 1
        assert result[0].gpu_name == "A100"


class TestMarketplacePostJob:
    """Marketplace.post_job()."""

    def test_basic(self) -> None:
        mp = Marketplace()
        job = mp.post_job("u1", "Llama-70B", 40 * 1024**3, 10.0)
        assert job.job_id.startswith("job-")
        assert job.requester_id == "u1"
        assert job.model_name == "Llama-70B"
        assert job.status is JobStatus.OPEN

    def test_minimal(self) -> None:
        mp = Marketplace()
        job = mp.post_job("u1", "model")
        assert job.min_gpu_memory_bytes == 0
        assert job.max_price_per_hour == 0.0

    def test_with_extra_kwargs(self) -> None:
        mp = Marketplace()
        job = mp.post_job(
            "u1", "model", 40 * 1024**3, 10.0,
            requires_quantization=True,
            priority=0,
            preferred_regions=["us-east-1"],
        )
        assert job.requires_quantization is True
        assert job.priority == 0

    def test_unique_ids(self) -> None:
        mp = Marketplace()
        j1 = mp.post_job("u1", "model")
        j2 = mp.post_job("u1", "model")
        assert j1.job_id != j2.job_id

    def test_empty_model_name(self) -> None:
        mp = Marketplace()
        job = mp.post_job("u1", "")
        assert job.model_name == ""

    def test_zero_budget(self) -> None:
        mp = Marketplace()
        job = mp.post_job("u1", "model", max_price_per_hour=0.0)
        assert job.max_price_per_hour == 0.0

    def test_boundary_priority(self) -> None:
        mp = Marketplace()
        j_high = mp.post_job("u1", "model", priority=0)
        j_low = mp.post_job("u1", "model", priority=3)
        assert j_high.priority == 0
        assert j_low.priority == 3


class TestMarketplaceMatchJob:
    """Marketplace.match_job()."""

    def test_nonexistent_job(self) -> None:
        mp = Marketplace()
        assert mp.match_job("nonexistent") is None

    def test_no_available_listings(self) -> None:
        mp = Marketplace()
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        assert mp.match_job(job.job_id) is None

    def test_already_matched_job(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        mp.match_job(job.job_id)
        assert mp.match_job(job.job_id) is None

    def test_already_running_job(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        mp.match_job(job.job_id)
        mp.start_job(job.job_id)
        assert mp.match_job(job.job_id) is None

    def test_insufficient_gpu_memory(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "T4", 16 * 1024**3, 2.0)
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        assert mp.match_job(job.job_id) is None

    def test_exceeds_budget(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 20.0)
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        assert mp.match_job(job.job_id) is None

    def test_zero_budget_no_price_filter(self) -> None:
        """When max_price_per_hour is 0, the price filter is skipped."""
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 100.0)
        job = mp.post_job("u1", "model", 40 * 1024**3, max_price_per_hour=0.0)
        assert mp.match_job(job.job_id) is not None

    def test_below_min_reputation(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0, reputation_score=0.1)
        job = mp.post_job(
            "u1", "model", 40 * 1024**3, 10.0, min_reputation=0.5,
        )
        assert mp.match_job(job.job_id) is None

    def test_below_min_uptime(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0, uptime_pct=95.0)
        job = mp.post_job(
            "u1", "model", 40 * 1024**3, 10.0, min_uptime_pct=99.0,
        )
        assert mp.match_job(job.job_id) is None

    def test_region_mismatch(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0, region="us-east-1")
        job = mp.post_job(
            "u1", "model", 40 * 1024**3, 10.0,
            preferred_regions=["eu-west-1"],
        )
        assert mp.match_job(job.job_id) is None

    def test_region_match(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0, region="us-east-1")
        job = mp.post_job(
            "u1", "model", 40 * 1024**3, 10.0,
            preferred_regions=["us-east-1"],
        )
        assert mp.match_job(job.job_id) is not None

    def test_carbon_filter(self) -> None:
        mp = Marketplace()
        mp.create_listing(
            "p1", "A100", 80 * 1024**3, 5.0, carbon_intensity=500.0,
        )
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        assert mp.match_job(job.job_id, max_carbon_gco2_kwh=100.0) is None

    def test_best_score_selected(self) -> None:
        mp = Marketplace()
        mp.create_listing(
            "p-low", "A100", 80 * 1024**3, 5.0, reputation_score=0.1,
        )
        mp.create_listing(
            "p-high", "A100", 80 * 1024**3, 5.0, reputation_score=0.9,
        )
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        result = mp.match_job(job.job_id)
        assert result is not None
        assert result.provider_id == "p-high"

    def test_sets_job_status_matched(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        mp.match_job(job.job_id)
        assert job.status is JobStatus.MATCHED
        assert job.matched_listing_id != ""
        assert job.matched_provider_id != ""

    def test_sets_listing_busy_when_full(self) -> None:
        mp = Marketplace()
        listing = mp.create_listing(
            "p1", "A100", 80 * 1024**3, 5.0, max_concurrent_jobs=1,
        )
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        mp.match_job(job.job_id)
        assert listing.status is ListingStatus.BUSY

    def test_listing_remains_active_with_slots(self) -> None:
        mp = Marketplace()
        listing = mp.create_listing(
            "p1", "A100", 80 * 1024**3, 5.0, max_concurrent_jobs=2,
        )
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        mp.match_job(job.job_id)
        assert listing.status is ListingStatus.ACTIVE
        assert listing.current_jobs == 1

    def test_all_filters_combined_pass(self) -> None:
        mp = Marketplace()
        mp.create_listing(
            "p1", "A100", 80 * 1024**3, 5.0,
            reputation_score=0.8,
            uptime_pct=99.5,
            region="us-east-1",
            carbon_intensity=50.0,
        )
        job = mp.post_job(
            "u1", "model", 40 * 1024**3, 10.0,
            min_reputation=0.5,
            min_uptime_pct=99.0,
            preferred_regions=["us-east-1"],
        )
        result = mp.match_job(job.job_id, max_carbon_gco2_kwh=100.0)
        assert result is not None
        assert result.provider_id == "p1"


class TestMarketplaceStartJob:
    """Marketplace.start_job()."""

    def test_start_matched_job(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        mp.match_job(job.job_id)
        assert mp.start_job(job.job_id) is True
        assert job.status is JobStatus.RUNNING
        assert job.started_at > 0

    def test_start_from_open_fails(self) -> None:
        mp = Marketplace()
        job = mp.post_job("u1", "model")
        assert mp.start_job(job.job_id) is False

    def test_start_from_running_fails(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        mp.match_job(job.job_id)
        mp.start_job(job.job_id)
        assert mp.start_job(job.job_id) is False

    def test_start_from_completed_fails(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        mp.match_job(job.job_id)
        mp.start_job(job.job_id)
        mp.complete_job(job.job_id)
        assert mp.start_job(job.job_id) is False

    def test_start_from_cancelled_fails(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        mp.match_job(job.job_id)
        mp.cancel_job(job.job_id)
        assert mp.start_job(job.job_id) is False

    def test_start_from_failed_fails(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        mp.match_job(job.job_id)
        mp.fail_job(job.job_id)
        assert mp.start_job(job.job_id) is False

    def test_start_nonexistent(self) -> None:
        mp = Marketplace()
        assert mp.start_job("nonexistent") is False


class TestMarketplaceCompleteJob:
    """Marketplace.complete_job()."""

    def test_complete_running_job(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        mp.match_job(job.job_id)
        mp.start_job(job.job_id)
        assert mp.complete_job(job.job_id, tokens_generated=1000) is True
        assert job.status is JobStatus.COMPLETED
        assert job.tokens_generated == 1000
        assert job.completed_at > 0

    def test_complete_with_zero_tokens(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        mp.match_job(job.job_id)
        mp.start_job(job.job_id)
        assert mp.complete_job(job.job_id, tokens_generated=0) is True
        assert job.tokens_generated == 0

    def test_complete_not_running_fails(self) -> None:
        mp = Marketplace()
        job = mp.post_job("u1", "model")
        assert mp.complete_job(job.job_id) is False

        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        mp.match_job(job.job_id)
        assert mp.complete_job(job.job_id) is False

    def test_complete_nonexistent(self) -> None:
        mp = Marketplace()
        assert mp.complete_job("nonexistent") is False

    def test_complete_frees_listing_slot(self) -> None:
        mp = Marketplace()
        listing = mp.create_listing(
            "p1", "A100", 80 * 1024**3, 5.0, max_concurrent_jobs=1,
        )
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        mp.match_job(job.job_id)
        mp.start_job(job.job_id)
        mp.complete_job(job.job_id)
        assert listing.status is ListingStatus.ACTIVE
        assert listing.current_jobs == 0
        assert listing.total_jobs_completed == 1

    def test_complete_updates_earnings(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        mp.match_job(job.job_id)
        mp.start_job(job.job_id)
        mp.complete_job(job.job_id, tokens_generated=5000)

        earnings = mp.get_provider_earnings("p1")
        assert earnings is not None
        assert earnings.total_jobs == 1
        assert earnings.total_tokens_served == 5000
        assert earnings.total_earnings > 0


class TestMarketplaceCancelJob:
    """Marketplace.cancel_job()."""

    def test_cancel_open_job(self) -> None:
        mp = Marketplace()
        job = mp.post_job("u1", "model")
        assert mp.cancel_job(job.job_id) is True
        assert job.status is JobStatus.CANCELLED

    def test_cancel_matched_job(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        mp.match_job(job.job_id)
        assert mp.cancel_job(job.job_id) is True
        assert job.status is JobStatus.CANCELLED

    def test_cancel_running_job(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        mp.match_job(job.job_id)
        mp.start_job(job.job_id)
        assert mp.cancel_job(job.job_id) is True

    def test_cancel_completed_job_fails(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        mp.match_job(job.job_id)
        mp.start_job(job.job_id)
        mp.complete_job(job.job_id)
        assert mp.cancel_job(job.job_id) is False

    def test_cancel_cancelled_job_fails(self) -> None:
        mp = Marketplace()
        job = mp.post_job("u1", "model")
        mp.cancel_job(job.job_id)
        assert mp.cancel_job(job.job_id) is False

    def test_cancel_nonexistent(self) -> None:
        mp = Marketplace()
        assert mp.cancel_job("nonexistent") is False

    def test_cancel_frees_listing(self) -> None:
        mp = Marketplace()
        listing = mp.create_listing(
            "p1", "A100", 80 * 1024**3, 5.0, max_concurrent_jobs=1,
        )
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        mp.match_job(job.job_id)
        assert listing.status is ListingStatus.BUSY
        mp.cancel_job(job.job_id)
        assert listing.status is ListingStatus.ACTIVE
        assert listing.current_jobs == 0


class TestMarketplaceFailJob:
    """Marketplace.fail_job()."""

    def test_fail_running_job(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        mp.match_job(job.job_id)
        mp.start_job(job.job_id)
        assert mp.fail_job(job.job_id, error="OOM error") is True
        assert job.status is JobStatus.FAILED

    def test_fail_open_job(self) -> None:
        mp = Marketplace()
        job = mp.post_job("u1", "model")
        assert mp.fail_job(job.job_id) is True
        assert job.status is JobStatus.FAILED

    def test_fail_matched_job(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        mp.match_job(job.job_id)
        assert mp.fail_job(job.job_id) is True
        assert job.status is JobStatus.FAILED

    def test_fail_already_failed_fails(self) -> None:
        mp = Marketplace()
        job = mp.post_job("u1", "model")
        mp.fail_job(job.job_id)
        assert mp.fail_job(job.job_id) is False

    def test_fail_completed_job_fails(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        mp.match_job(job.job_id)
        mp.start_job(job.job_id)
        mp.complete_job(job.job_id)
        assert mp.fail_job(job.job_id) is False

    def test_fail_cancelled_job_fails(self) -> None:
        mp = Marketplace()
        job = mp.post_job("u1", "model")
        mp.cancel_job(job.job_id)
        assert mp.fail_job(job.job_id) is False

    def test_fail_with_error_message(self) -> None:
        mp = Marketplace()
        job = mp.post_job("u1", "model")
        mp.fail_job(job.job_id, error="GPU out of memory")
        assert job.status is JobStatus.FAILED

    def test_fail_nonexistent(self) -> None:
        mp = Marketplace()
        assert mp.fail_job("nonexistent") is False

    def test_fail_frees_listing(self) -> None:
        mp = Marketplace()
        listing = mp.create_listing(
            "p1", "A100", 80 * 1024**3, 5.0, max_concurrent_jobs=1,
        )
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        mp.match_job(job.job_id)
        assert listing.status is ListingStatus.BUSY
        mp.fail_job(job.job_id)
        assert listing.status is ListingStatus.ACTIVE
        assert listing.current_jobs == 0


class TestMarketplaceGetJob:
    """Marketplace.get_job()."""

    def test_existing(self) -> None:
        mp = Marketplace()
        job = mp.post_job("u1", "model")
        assert mp.get_job(job.job_id) is job

    def test_nonexistent(self) -> None:
        mp = Marketplace()
        assert mp.get_job("nonexistent") is None

    def test_after_state_change(self) -> None:
        mp = Marketplace()
        job = mp.post_job("u1", "model")
        mp.cancel_job(job.job_id)
        fetched = mp.get_job(job.job_id)
        assert fetched is not None
        assert fetched.status is JobStatus.CANCELLED


class TestMarketplaceListJobs:
    """Marketplace.list_jobs()."""

    def test_empty(self) -> None:
        mp = Marketplace()
        assert mp.list_jobs() == []

    def test_all(self) -> None:
        mp = Marketplace()
        mp.post_job("u1", "model-a")
        mp.post_job("u2", "model-b")
        assert len(mp.list_jobs()) == 2

    def test_filter_by_requester(self) -> None:
        mp = Marketplace()
        mp.post_job("u1", "model-a")
        mp.post_job("u2", "model-b")
        result = mp.list_jobs(requester_id="u1")
        assert len(result) == 1
        assert result[0].requester_id == "u1"

    def test_filter_by_requester_empty(self) -> None:
        mp = Marketplace()
        mp.post_job("u1", "model-a")
        result = mp.list_jobs(requester_id="u-nonexistent")
        assert result == []

    def test_filter_by_status(self) -> None:
        mp = Marketplace()
        j1 = mp.post_job("u1", "model-a")
        mp.post_job("u2", "model-b")

        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        mp.match_job(j1.job_id)

        open_jobs = mp.list_jobs(status=JobStatus.OPEN)
        matched_jobs = mp.list_jobs(status=JobStatus.MATCHED)

        assert len(open_jobs) == 1
        assert len(matched_jobs) == 1

    def test_combined_filter(self) -> None:
        mp = Marketplace()
        mp.post_job("u1", "model-a")
        j2 = mp.post_job("u1", "model-b")

        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        mp.match_job(j2.job_id)

        results = mp.list_jobs(requester_id="u1", status=JobStatus.MATCHED)
        assert len(results) == 1
        assert results[0].model_name == "model-b"

    def test_sorted_by_created_at_desc(self) -> None:
        mp = Marketplace()
        j1 = mp.post_job("u1", "model-a")
        j2 = mp.post_job("u2", "model-b")
        result = mp.list_jobs()
        assert len(result) >= 2
        assert result[0].created_at >= result[1].created_at


class TestMarketplaceGetProviderEarnings:
    """Marketplace.get_provider_earnings()."""

    def test_existing_after_listing(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        earnings = mp.get_provider_earnings("p1")
        assert earnings is not None
        assert earnings.provider_id == "p1"

    def test_nonexistent(self) -> None:
        mp = Marketplace()
        assert mp.get_provider_earnings("nonexistent") is None


class TestMarketplaceGetMarketplaceStats:
    """Marketplace.get_marketplace_stats()."""

    def test_empty_marketplace(self) -> None:
        mp = Marketplace()
        stats = mp.get_marketplace_stats()
        assert stats["total_listings"] == 0
        assert stats["active_listings"] == 0
        assert stats["total_jobs"] == 0
        assert stats["open_jobs"] == 0
        assert stats["running_jobs"] == 0
        assert stats["completed_jobs"] == 0
        assert stats["total_volume_usd"] == 0.0
        assert stats["total_tokens_served"] == 0
        assert stats["avg_price_per_hour"] == 0.0

    def test_with_listings_only(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 10.0)
        mp.create_listing("p2", "T4", 16 * 1024**3, 2.0)
        stats = mp.get_marketplace_stats()
        assert stats["total_listings"] == 2
        assert stats["active_listings"] == 2
        assert stats["total_jobs"] == 0
        assert stats["avg_price_per_hour"] == 6.0  # (10 + 2) / 2

    def test_with_jobs(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        job = mp.post_job("u1", "model", 40 * 1024**3, 10.0)
        mp.match_job(job.job_id)
        mp.start_job(job.job_id)
        mp.complete_job(job.job_id, tokens_generated=500)

        stats = mp.get_marketplace_stats()
        assert stats["total_jobs"] == 1
        assert stats["completed_jobs"] == 1
        assert stats["total_tokens_served"] == 500
        assert stats["total_volume_usd"] > 0


class TestMarketplaceAddCloudListings:
    """Marketplace.add_cloud_listings()."""

    def test_empty(self) -> None:
        mp = Marketplace()
        assert mp.add_cloud_listings([]) == 0

    def test_single_listing(self) -> None:
        mp = Marketplace()
        cloud = [
            {
                "provider": "aws",
                "instance_type": "p4d.24xlarge",
                "region": "us-east-1",
                "gpu_type": "A100",
                "gpu_count": 8,
                "gpu_memory_gb": 80.0,
                "on_demand_price": 32.77,
                "spot_price": 14.40,
                "carbon_intensity": 380,
            },
        ]
        assert mp.add_cloud_listings(cloud) == 1
        listings = mp.list_listings(source="cloud")
        assert len(listings) == 1
        assert listings[0].price_per_hour == 14.40  # spot price wins

    def test_uses_on_demand_when_spot_zero(self) -> None:
        mp = Marketplace()
        cloud = [
            {
                "provider": "aws",
                "instance_type": "p4d.24xlarge",
                "region": "us-east-1",
                "gpu_type": "A100",
                "gpu_count": 8,
                "gpu_memory_gb": 80.0,
                "on_demand_price": 32.77,
                "spot_price": 0.0,
            },
        ]
        mp.add_cloud_listings(cloud)
        listings = mp.list_listings()
        assert listings[0].price_per_hour == 32.77

    def test_sets_source_to_cloud(self) -> None:
        mp = Marketplace()
        cloud = [
            {
                "provider": "gcp",
                "instance_type": "a2-highgpu-1g",
                "region": "us-central1",
                "gpu_type": "A100",
                "gpu_count": 1,
                "gpu_memory_gb": 40.0,
                "on_demand_price": 3.67,
                "spot_price": 0.92,
            },
        ]
        assert mp.add_cloud_listings(cloud) == 1
        assert mp.list_listings()[0].source == "cloud"

    def test_sets_reputation_and_uptime(self) -> None:
        mp = Marketplace()
        cloud = [
            {
                "provider": "aws",
                "instance_type": "p4d.24xlarge",
                "region": "us-east-1",
                "gpu_type": "A100",
                "gpu_count": 8,
                "gpu_memory_gb": 80.0,
                "on_demand_price": 32.77,
                "spot_price": 14.40,
            },
        ]
        mp.add_cloud_listings(cloud)
        listing = mp.list_listings()[0]
        assert listing.reputation_score == 1.0
        assert listing.uptime_pct == 99.9

    def test_multiple_listings(self) -> None:
        mp = Marketplace()
        cloud = [
            {
                "provider": "aws",
                "instance_type": "p4d.24xlarge",
                "region": "us-east-1",
                "gpu_type": "A100",
                "gpu_count": 8,
                "gpu_memory_gb": 80.0,
                "on_demand_price": 32.77,
                "spot_price": 14.40,
            },
            {
                "provider": "gcp",
                "instance_type": "a2-highgpu-1g",
                "region": "us-central1",
                "gpu_type": "A100",
                "gpu_count": 1,
                "gpu_memory_gb": 40.0,
                "on_demand_price": 3.67,
                "spot_price": 0.92,
            },
        ]
        assert mp.add_cloud_listings(cloud) == 2
        assert len(mp.list_listings()) == 2


class TestMarketplaceGetUnifiedListings:
    """Marketplace.get_unified_listings()."""

    def test_empty(self) -> None:
        mp = Marketplace()
        assert mp.get_unified_listings() == []

    def test_returns_peer_and_cloud(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        mp.add_cloud_listings([
            {
                "provider": "aws",
                "instance_type": "p4d.24xlarge",
                "region": "us-east-1",
                "gpu_type": "A100",
                "gpu_count": 8,
                "gpu_memory_gb": 80.0,
                "on_demand_price": 32.77,
                "spot_price": 14.40,
            },
        ])
        result = mp.get_unified_listings()
        assert len(result) == 2

    def test_applies_filters(self) -> None:
        mp = Marketplace()
        mp.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        mp.create_listing("p2", "T4", 16 * 1024**3, 2.0)
        result = mp.get_unified_listings(min_gpu_memory=40 * 1024**3)
        assert len(result) == 1
        assert result[0].gpu_name == "A100"

    def test_carbon_filter(self) -> None:
        mp = Marketplace()
        mp.create_listing(
            "p1", "A100", 80 * 1024**3, 5.0, carbon_intensity=50.0,
        )
        mp.create_listing(
            "p2", "A100", 80 * 1024**3, 5.0, carbon_intensity=500.0,
        )
        result = mp.get_unified_listings(max_carbon_gco2_kwh=100.0)
        assert len(result) == 1


# ── CloudCostModel ──────────────────────────────────────────────────────────


class TestCloudCostModel:
    """CloudCostModel.from_pricing_data()."""

    def test_empty_input(self) -> None:
        assert CloudCostModel.from_pricing_data([]) == []

    def test_with_dicts(self) -> None:
        prices = [
            {
                "provider": "aws",
                "instance_type": "p4d.24xlarge",
                "region": "us-east-1",
                "gpu_type": "A100",
                "gpu_count": 8,
                "gpu_memory_gb": 80.0,
                "on_demand_price": 32.77,
                "spot_price": 14.40,
                "carbon_intensity": 380,
            },
        ]
        result = CloudCostModel.from_pricing_data(prices)
        assert len(result) == 1
        assert result[0]["provider"] == "aws"
        assert result[0]["spot_price"] == 14.40

    def test_with_objects_with_dict(self) -> None:
        """Objects with __dict__ should be converted like dicts."""

        class FakePricing:
            def __init__(self) -> None:
                self.provider = "gcp"
                self.instance_type = "a2-highgpu-1g"
                self.region = "us-central1"
                self.gpu_type = "A100"
                self.gpu_count = 1
                self.gpu_memory_gb = 40.0
                self.on_demand_price = 3.67
                self.spot_price = 0.92
                self.carbon_intensity = 450

        result = CloudCostModel.from_pricing_data([FakePricing()])
        assert len(result) == 1
        assert result[0]["provider"] == "gcp"
        assert result[0]["spot_price"] == 0.92

    def test_skips_unrecognized_types(self) -> None:
        result = CloudCostModel.from_pricing_data([42, "string", None, 3.14])
        assert result == []

    def test_mixed_valid_and_invalid(self) -> None:
        prices = [
            {"provider": "aws", "instance_type": "p4d", "region": "ue1"},
            42,
            {"provider": "gcp", "instance_type": "a2", "region": "uc1"},
        ]
        result = CloudCostModel.from_pricing_data(prices)
        assert len(result) == 2

    def test_minimal_dict(self) -> None:
        prices = [{"provider": "aws", "instance_type": "x", "region": "y"}]
        result = CloudCostModel.from_pricing_data(prices)
        assert len(result) == 1
        assert result[0]["gpu_memory_gb"] == 0.0
        assert result[0]["spot_price"] == 0.0
        assert result[0]["gpu_type"] == ""

    def test_fallback_on_demand_price(self) -> None:
        """When price_per_hour is provided but on_demand_price is not."""
        prices = [
            {
                "provider": "aws",
                "instance_type": "x",
                "region": "y",
                "price_per_hour": 5.0,
            },
        ]
        result = CloudCostModel.from_pricing_data(prices)
        assert result[0]["on_demand_price"] == 5.0
