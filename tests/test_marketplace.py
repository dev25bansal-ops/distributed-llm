"""Unit tests for Marketplace.

Covers:
- Match job concurrent access
- Earnings calculation
- Listing status transitions
- Effective score bounds
- Carbon filtering
- Cloud listing import
"""

import threading
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


@pytest.fixture
def marketplace():
    return Marketplace()


@pytest.fixture
def sample_listing(marketplace):
    return marketplace.create_listing(
        provider_id="provider-1",
        gpu_name="A100",
        gpu_memory_bytes=80 * 1024**3,
        price_per_hour=5.0,
        gpu_count=1,
        region="us-east-1",
    )


@pytest.fixture
def sample_job(marketplace):
    return marketplace.post_job(
        requester_id="user-1",
        model_name="Llama-70B",
        min_gpu_memory_bytes=40 * 1024**3,
        max_price_per_hour=10.0,
    )


class TestMatchJobConcurrentAccess:
    """Test concurrent job matching doesn't cause races."""

    def test_concurrent_match_same_job(self, marketplace):
        marketplace.create_listing("p1", "A100", 80 * 1024**3, 5.0, region="us-east-1")
        marketplace.create_listing("p2", "A100", 80 * 1024**3, 6.0, region="us-west-2")
        job = marketplace.post_job("user-1", "model", 40 * 1024**3, 10.0)

        results = []
        errors = []

        def match():
            try:
                result = marketplace.match_job(job.job_id)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=match) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Only one should succeed (job goes to MATCHED status)
        matched = [r for r in results if r is not None]
        assert len(matched) == 1

    def test_concurrent_listing_creation(self, marketplace):
        errors = []

        def create(i):
            try:
                marketplace.create_listing(f"p{i}", "A100", 80 * 1024**3, float(i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(marketplace.list_listings()) == 50


class TestEarningsCalculation:
    """Test provider earnings tracking."""

    def test_earnings_on_job_completion(self, marketplace):
        listing = marketplace.create_listing("p1", "A100", 80 * 1024**3, 10.0)
        job = marketplace.post_job("user-1", "model", 40 * 1024**3, 15.0)
        marketplace.match_job(job.job_id)
        marketplace.start_job(job.job_id)

        # Simulate 1 hour of work
        time.sleep(0.1)
        marketplace.complete_job(job.job_id, tokens_generated=1000)

        earnings = marketplace.get_provider_earnings("p1")
        assert earnings is not None
        assert earnings.total_tokens_served == 1000
        assert earnings.total_jobs == 1
        assert earnings.total_earnings > 0

    def test_earnings_accumulate(self, marketplace):
        marketplace.create_listing("p1", "A100", 80 * 1024**3, 10.0)

        for i in range(3):
            job = marketplace.post_job(f"user-{i}", "model", 40 * 1024**3, 15.0)
            marketplace.match_job(job.job_id)
            marketplace.start_job(job.job_id)
            time.sleep(0.05)
            marketplace.complete_job(job.job_id, tokens_generated=500)

        earnings = marketplace.get_provider_earnings("p1")
        assert earnings.total_jobs == 3
        assert earnings.total_tokens_served == 1500


class TestListingStatusTransitions:
    """Test listing status state machine."""

    def test_active_to_busy_on_full(self, marketplace):
        listing = marketplace.create_listing(
            "p1", "A100", 80 * 1024**3, 5.0, max_concurrent_jobs=1
        )
        assert listing.status == ListingStatus.ACTIVE

        job = marketplace.post_job("user-1", "model", 40 * 1024**3, 10.0)
        marketplace.match_job(job.job_id)
        assert listing.status == ListingStatus.BUSY

    def test_busy_to_active_on_job_complete(self, marketplace):
        listing = marketplace.create_listing(
            "p1", "A100", 80 * 1024**3, 5.0, max_concurrent_jobs=1
        )
        job = marketplace.post_job("user-1", "model", 40 * 1024**3, 10.0)
        marketplace.match_job(job.job_id)
        assert listing.status == ListingStatus.BUSY

        marketplace.start_job(job.job_id)
        marketplace.complete_job(job.job_id, tokens_generated=100)
        assert listing.status == ListingStatus.ACTIVE

    def test_busy_to_active_on_job_cancel(self, marketplace):
        listing = marketplace.create_listing(
            "p1", "A100", 80 * 1024**3, 5.0, max_concurrent_jobs=1
        )
        job = marketplace.post_job("user-1", "model", 40 * 1024**3, 10.0)
        marketplace.match_job(job.job_id)
        assert listing.status == ListingStatus.BUSY

        marketplace.cancel_job(job.job_id)
        assert listing.status == ListingStatus.ACTIVE

    def test_is_available_property(self):
        listing = GPUListing(
            listing_id="test", provider_id="p1",
            status=ListingStatus.ACTIVE, max_concurrent_jobs=2, current_jobs=0,
        )
        assert listing.is_available is True

        listing.current_jobs = 2
        assert listing.is_available is False

        listing.status = ListingStatus.BUSY
        assert listing.is_available is False


class TestEffectiveScoreBounds:
    """Test effective_score stays in reasonable bounds."""

    def test_score_positive(self, marketplace):
        listing = marketplace.create_listing("p1", "A100", 80 * 1024**3, 0.01)
        assert listing.effective_score > 0

    def test_score_not_nan(self, marketplace):
        listing = marketplace.create_listing("p1", "A100", 80 * 1024**3, 0.0)
        assert listing.effective_score == listing.effective_score  # Not NaN

    def test_higher_reputation_higher_score(self):
        l1 = GPUListing(listing_id="1", provider_id="p1", price_per_hour=5.0,
                        reputation_score=0.9, gpu_memory_bytes=80 * 1024**3)
        l2 = GPUListing(listing_id="2", provider_id="p2", price_per_hour=5.0,
                        reputation_score=0.1, gpu_memory_bytes=80 * 1024**3)
        assert l1.effective_score > l2.effective_score

    def test_cheaper_higher_score(self):
        l1 = GPUListing(listing_id="1", provider_id="p1", price_per_hour=1.0,
                        reputation_score=0.5, gpu_memory_bytes=80 * 1024**3)
        l2 = GPUListing(listing_id="2", provider_id="p2", price_per_hour=10.0,
                        reputation_score=0.5, gpu_memory_bytes=80 * 1024**3)
        assert l1.effective_score > l2.effective_score


class TestCarbonFiltering:
    """Test carbon intensity filtering in marketplace."""

    def test_list_listings_carbon_filter(self, marketplace):
        marketplace.create_listing("p1", "A100", 80 * 1024**3, 5.0,
                                   carbon_intensity=50.0, region="eu-west-1")
        marketplace.create_listing("p2", "A100", 80 * 1024**3, 5.0,
                                   carbon_intensity=500.0, region="us-east-1")

        clean = marketplace.list_listings(max_carbon_gco2_kwh=100)
        assert len(clean) == 1
        assert clean[0].carbon_intensity == 50.0

    def test_match_job_carbon_filter(self, marketplace):
        marketplace.create_listing("p1", "A100", 80 * 1024**3, 5.0,
                                   carbon_intensity=50.0, region="eu-west-1")
        marketplace.create_listing("p2", "A100", 80 * 1024**3, 5.0,
                                   carbon_intensity=500.0, region="us-east-1")
        job = marketplace.post_job("user-1", "model", 40 * 1024**3, 10.0)

        matched = marketplace.match_job(job.job_id, max_carbon_gco2_kwh=100)
        assert matched is not None
        assert matched.carbon_intensity <= 100


class TestCloudListingImport:
    """Test cloud listing import via add_cloud_listings."""

    def test_import_cloud_listings(self, marketplace):
        cloud_prices = [
            {"provider": "aws", "instance_type": "p4d.24xlarge", "region": "us-east-1",
             "gpu_type": "A100", "gpu_count": 8, "gpu_memory_gb": 80.0,
             "on_demand_price": 32.77, "spot_price": 14.40, "carbon_intensity": 380},
            {"provider": "gcp", "instance_type": "a2-highgpu-1g", "region": "us-central1",
             "gpu_type": "A100", "gpu_count": 1, "gpu_memory_gb": 40.0,
             "on_demand_price": 3.67, "spot_price": 0.92, "carbon_intensity": 450},
        ]
        created = marketplace.add_cloud_listings(cloud_prices)
        assert created == 2

        all_listings = marketplace.list_listings()
        assert len(all_listings) == 2
        cloud_only = marketplace.list_listings(source="cloud")
        assert len(cloud_only) == 2

    def test_cloud_cost_model_conversion(self):
        from distllm.core.pricing_providers import InstancePricing
        prices = [
            InstancePricing(provider="aws", instance_type="p4d.24xlarge",
                            region="us-east-1", gpu_type="A100", gpu_count=8,
                            gpu_memory_gb=80.0, on_demand_price=32.77,
                            spot_price=14.40),
        ]
        converted = CloudCostModel.from_pricing_data(prices)
        assert len(converted) == 1
        assert converted[0]["provider"] == "aws"
        assert converted[0]["spot_price"] == 14.40


class TestMarketplaceStats:
    """Test marketplace statistics."""

    def test_stats_empty(self, marketplace):
        stats = marketplace.get_marketplace_stats()
        assert stats["total_listings"] == 0
        assert stats["total_jobs"] == 0

    def test_stats_with_data(self, marketplace):
        marketplace.create_listing("p1", "A100", 80 * 1024**3, 5.0)
        marketplace.create_listing("p2", "V100", 16 * 1024**3, 2.0)
        marketplace.post_job("user-1", "model", 40 * 1024**3, 10.0)

        stats = marketplace.get_marketplace_stats()
        assert stats["total_listings"] == 2
        assert stats["active_listings"] == 2
        assert stats["total_jobs"] == 1
        assert stats["open_jobs"] == 1
