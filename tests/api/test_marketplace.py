"""Marketplace API route tests: GPU listing, job posting, auto-matching, earnings, and stats.

All tests use real objects (no MagicMock) and the real Marketplace class
with an in-memory backend.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from distllm.api.routes.marketplace import router
from distllm.dist.marketplace import (
    JobStatus,
    Marketplace,
)

# ---------------------------------------------------------------------------
# Test auth key
# ---------------------------------------------------------------------------

TEST_ADMIN_KEY = "test-admin-key-12345"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    """Minimal FastAPI app with marketplace router and test auth middleware.

    Only the marketplace router is included (no other routes, no production
    middleware).  A lightweight ``BaseHTTPMiddleware`` validates the
    ``Authorization`` header and sets ``request.state.api_key_role`` so
    that ``require_role("admin")`` dependencies pass.
    """
    _app = FastAPI()

    class _TestAuthMiddleware(BaseHTTPMiddleware):
        """Simulate AuthMiddleware: validate Bearer token, set api_key_role."""

        async def dispatch(self, request: Request, call_next):
            auth = request.headers.get("Authorization", "")
            if auth.startswith("Bearer ") and auth[7:] == TEST_ADMIN_KEY:
                request.state.api_key_role = "admin"
                request.state.api_key_id = "test-key-id"
            return await call_next(request)

    _app.add_middleware(_TestAuthMiddleware)
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    """TestClient bound to the minimal test app."""
    return TestClient(app)


@pytest.fixture
def marketplace():
    """Return a real ``Marketplace`` instance (in-memory, no backend)."""
    return Marketplace()


@pytest.fixture
def admin_headers():
    """Valid admin ``Authorization`` header for admin-required endpoints."""
    return {"Authorization": f"Bearer {TEST_ADMIN_KEY}"}


@pytest.fixture
def _setup_marketplace(marketplace):
    """Inject a ``Marketplace`` into the shared ``AppState``.

    The marketplace routes call ``g.get("marketplace")`` which delegates to
    ``AppState``.  This fixture places a fresh instance there before every
    test that needs it, and cleans up afterwards.
    """
    from distllm.api.api_state import _state

    _state.marketplace = marketplace
    yield
    _state.marketplace = None


# ===================================================================
# POST /v1/marketplace/listings
# ===================================================================


class TestCreateListing:
    """Create a new GPU listing."""

    ENDPOINT = "/v1/marketplace/listings"
    PAYLOAD = {
        "provider_id": "provider-1",
        "gpu_name": "NVIDIA A100 80GB",
        "gpu_memory_bytes": 85_899_345_920,
        "price_per_hour": 2.50,
        "gpu_count": 1,
        "region": "us-east",
        "supported_models": ["llama-3-8b"],
        "max_batch_size": 16,
        "supports_streaming": True,
        "supports_quantization": False,
        "supports_lora": True,
        "tags": ["fast", "reliable"],
    }

    def test_no_auth(self, client, _setup_marketplace):
        """Returns 401 when no auth header is provided."""
        resp = client.post(self.ENDPOINT, json=self.PAYLOAD)
        assert resp.status_code == 401

    def test_no_marketplace(self, client, admin_headers):
        """Returns 503 when marketplace is not available."""
        resp = client.post(self.ENDPOINT, json=self.PAYLOAD, headers=admin_headers)
        assert resp.status_code == 503
        assert resp.json()["detail"] == "Marketplace not available"

    def test_success(self, client, marketplace, _setup_marketplace, admin_headers):
        """Creates a listing and returns its full details."""
        resp = client.post(self.ENDPOINT, json=self.PAYLOAD, headers=admin_headers)

        assert resp.status_code == 200
        data = resp.json()
        assert data["provider_id"] == "provider-1"
        assert data["gpu_name"] == "NVIDIA A100 80GB"
        assert data["gpu_memory_bytes"] == 85_899_345_920
        assert data["price_per_hour"] == 2.5
        assert data["gpu_count"] == 1
        assert data["region"] == "us-east"
        assert data["status"] == "active"
        assert data["is_available"] is True
        assert data["reputation_score"] == 0.5
        assert data["total_jobs_completed"] == 0
        assert len(data["listing_id"]) > 0

        # Verify it was actually stored in the marketplace
        assert marketplace.get_listing(data["listing_id"]) is not None


# ===================================================================
# GET /v1/marketplace/listings
# ===================================================================


class TestListListings:
    """List GPU listings with optional filters."""

    ENDPOINT = "/v1/marketplace/listings"

    def test_no_marketplace(self, client):
        """Returns 503 when marketplace is not available."""
        resp = client.get(self.ENDPOINT)
        assert resp.status_code == 503

    def test_empty(self, client, _setup_marketplace):
        """Returns an empty list when no listings exist."""
        resp = client.get(self.ENDPOINT)
        assert resp.status_code == 200
        assert resp.json() == []

    def test_all_listings(self, client, marketplace, _setup_marketplace):
        """Returns all listings when no filters are applied."""
        marketplace.create_listing(
            provider_id="p1", gpu_name="A100",
            gpu_memory_bytes=80_000_000_000, price_per_hour=2.0,
        )
        marketplace.create_listing(
            provider_id="p2", gpu_name="V100",
            gpu_memory_bytes=32_000_000_000, price_per_hour=1.0,
        )
        resp = client.get(self.ENDPOINT)
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_filter_min_gpu_memory(self, client, marketplace, _setup_marketplace):
        """Filters by ``min_gpu_memory`` query parameter."""
        marketplace.create_listing(
            provider_id="p1", gpu_name="A100",
            gpu_memory_bytes=80_000_000_000, price_per_hour=2.0,
            region="us-east",
        )
        marketplace.create_listing(
            provider_id="p2", gpu_name="V100",
            gpu_memory_bytes=32_000_000_000, price_per_hour=1.0,
            region="eu-west",
        )
        resp = client.get(self.ENDPOINT, params={"min_gpu_memory": 64_000_000_000})
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["gpu_name"] == "A100"

    def test_filter_region(self, client, marketplace, _setup_marketplace):
        """Filters by ``region`` query parameter."""
        marketplace.create_listing(
            provider_id="p1", gpu_name="A100",
            gpu_memory_bytes=80_000_000_000, price_per_hour=2.0,
            region="us-east",
        )
        marketplace.create_listing(
            provider_id="p2", gpu_name="V100",
            gpu_memory_bytes=32_000_000_000, price_per_hour=1.0,
            region="eu-west",
        )
        resp = client.get(self.ENDPOINT, params={"region": "eu-west"})
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["gpu_name"] == "V100"


# ===================================================================
# GET /v1/marketplace/listings/{listing_id}
# ===================================================================


class TestGetListing:
    """Get a specific GPU listing."""

    ENDPOINT = "/v1/marketplace/listings"

    def test_no_marketplace(self, client):
        """Returns 503 when marketplace is not available."""
        resp = client.get(f"{self.ENDPOINT}/nonexistent")
        assert resp.status_code == 503

    def test_not_found(self, client, _setup_marketplace):
        """Returns 404 when the listing does not exist."""
        resp = client.get(f"{self.ENDPOINT}/nonexistent")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Listing not found"

    def test_success(self, client, marketplace, _setup_marketplace):
        """Returns the full listing details."""
        listing = marketplace.create_listing(
            provider_id="p1", gpu_name="A100",
            gpu_memory_bytes=80_000_000_000, price_per_hour=2.0,
            region="us-east",
        )
        resp = client.get(f"{self.ENDPOINT}/{listing.listing_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["listing_id"] == listing.listing_id
        assert data["gpu_name"] == "A100"
        assert data["is_available"] is True


# ===================================================================
# DELETE /v1/marketplace/listings/{listing_id}
# ===================================================================


class TestRemoveListing:
    """Remove a GPU listing."""

    ENDPOINT = "/v1/marketplace/listings"

    def test_no_auth(self, client, _setup_marketplace):
        """Returns 401 without auth."""
        resp = client.delete(f"{self.ENDPOINT}/some-id")
        assert resp.status_code == 401

    def test_no_marketplace(self, client, admin_headers):
        """Returns 503 when marketplace is not available."""
        resp = client.delete(f"{self.ENDPOINT}/some-id", headers=admin_headers)
        assert resp.status_code == 503

    def test_not_found(self, client, _setup_marketplace, admin_headers):
        """Returns 404 when the listing does not exist."""
        resp = client.delete(f"{self.ENDPOINT}/nonexistent", headers=admin_headers)
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Listing not found"

    def test_success(self, client, marketplace, _setup_marketplace, admin_headers):
        """Removes the listing and returns a confirmation."""
        listing = marketplace.create_listing(
            provider_id="p1", gpu_name="A100",
            gpu_memory_bytes=80_000_000_000, price_per_hour=2.0,
        )
        resp = client.delete(
            f"{self.ENDPOINT}/{listing.listing_id}", headers=admin_headers
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "removed"

        # Verify it is actually gone
        assert marketplace.get_listing(listing.listing_id) is None


# ===================================================================
# POST /v1/marketplace/jobs
# ===================================================================


class TestPostJob:
    """Post a compute job to the marketplace."""

    ENDPOINT = "/v1/marketplace/jobs"
    PAYLOAD = {
        "requester_id": "requester-1",
        "model_name": "llama-3-8b",
        "min_gpu_memory_bytes": 16_000_000_000,
        "max_price_per_hour": 5.0,
        "max_latency_ms": 2000,
        "min_reputation": 0.3,
        "preferred_regions": ["us-east"],
        "priority": 2,
        "requires_streaming": True,
        "requires_quantization": False,
        "requires_lora": False,
        "max_budget_total": 50.0,
    }

    def test_no_auth(self, client, _setup_marketplace):
        """Returns 401 without auth."""
        resp = client.post(self.ENDPOINT, json=self.PAYLOAD)
        assert resp.status_code == 401

    def test_no_marketplace(self, client, admin_headers):
        """Returns 503 when marketplace is not available."""
        resp = client.post(
            self.ENDPOINT, json=self.PAYLOAD, headers=admin_headers
        )
        assert resp.status_code == 503

    def test_success_no_match(self, client, marketplace, _setup_marketplace, admin_headers):
        """Posts a job that does not match any listing (remains open)."""
        resp = client.post(
            self.ENDPOINT, json=self.PAYLOAD, headers=admin_headers
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["requester_id"] == "requester-1"
        assert data["model_name"] == "llama-3-8b"
        assert data["status"] == "open"
        assert data["matched_listing_id"] == ""
        assert data["matched_provider_id"] == ""

    def test_success_with_auto_match(self, client, marketplace, _setup_marketplace, admin_headers):
        """Posts a job that auto-matches to a compatible listing."""
        marketplace.create_listing(
            provider_id="p1", gpu_name="A100",
            gpu_memory_bytes=80_000_000_000, price_per_hour=2.0,
            region="us-east",
        )
        resp = client.post(
            self.ENDPOINT, json=self.PAYLOAD, headers=admin_headers
        )
        assert resp.status_code == 200
        data = resp.json()
        # The route auto-matches AND starts the job, so status is "running"
        assert data["status"] == "running"
        assert len(data["matched_listing_id"]) > 0
        assert data["matched_provider_id"] == "p1"


# ===================================================================
# GET /v1/marketplace/jobs
# ===================================================================


class TestListJobs:
    """List marketplace jobs."""

    ENDPOINT = "/v1/marketplace/jobs"

    def test_no_marketplace(self, client):
        """Returns 503 when marketplace is not available."""
        resp = client.get(self.ENDPOINT)
        assert resp.status_code == 503

    def test_empty(self, client, _setup_marketplace):
        """Returns an empty list when no jobs exist."""
        resp = client.get(self.ENDPOINT)
        assert resp.status_code == 200
        assert resp.json() == []

    def test_all_jobs(self, client, marketplace, _setup_marketplace):
        """Returns all jobs in the system."""
        marketplace.post_job(requester_id="r1", model_name="llama-3-8b")
        marketplace.post_job(requester_id="r2", model_name="gpt-3")
        resp = client.get(self.ENDPOINT)
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_filter_by_requester(self, client, marketplace, _setup_marketplace):
        """Filters jobs by ``requester_id`` query parameter."""
        marketplace.post_job(requester_id="r1", model_name="m1")
        marketplace.post_job(requester_id="r2", model_name="m2")
        resp = client.get(self.ENDPOINT, params={"requester_id": "r1"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["requester_id"] == "r1"

    def test_filter_by_status(self, client, marketplace, _setup_marketplace):
        """Filters jobs by ``status`` query parameter."""
        job = marketplace.post_job(requester_id="r1", model_name="m1")
        # Create a listing so the job can be matched and started
        marketplace.create_listing(
            provider_id="p1", gpu_name="A100",
            gpu_memory_bytes=80_000_000_000, price_per_hour=1.0,
            region="us-east",
        )
        marketplace.match_job(job.job_id)
        marketplace.start_job(job.job_id)

        resp = client.get(self.ENDPOINT, params={"status": "running"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["job_id"] == job.job_id

    def test_invalid_status(self, client, _setup_marketplace):
        """Returns 400 for an invalid status string."""
        resp = client.get(self.ENDPOINT, params={"status": "bogus_status"})
        assert resp.status_code == 400
        assert "Invalid status" in resp.json()["detail"]


# ===================================================================
# POST /v1/marketplace/jobs/{job_id}/complete
# ===================================================================


class TestCompleteJob:
    """Mark a marketplace job as completed."""

    ENDPOINT = "/v1/marketplace/jobs"

    def test_no_auth(self, client, _setup_marketplace):
        """Returns 401 without auth."""
        resp = client.post(f"{self.ENDPOINT}/some-id/complete")
        assert resp.status_code == 401

    def test_no_marketplace(self, client, admin_headers):
        """Returns 503 when marketplace is not available."""
        resp = client.post(
            f"{self.ENDPOINT}/some-id/complete", headers=admin_headers
        )
        assert resp.status_code == 503

    def test_cannot_complete(self, client, marketplace, _setup_marketplace, admin_headers):
        """Returns 400 when the job is not in a completable state."""
        # An open job cannot be completed
        job = marketplace.post_job(requester_id="r1", model_name="m1")
        resp = client.post(
            f"{self.ENDPOINT}/{job.job_id}/complete", headers=admin_headers
        )
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Cannot complete job"

    def test_success(self, client, marketplace, _setup_marketplace, admin_headers):
        """Completes a running job and records tokens."""
        marketplace.create_listing(
            provider_id="p1", gpu_name="A100",
            gpu_memory_bytes=80_000_000_000, price_per_hour=2.0,
            region="us-east",
        )
        job = marketplace.post_job(requester_id="r1", model_name="m1")
        marketplace.match_job(job.job_id)
        marketplace.start_job(job.job_id)

        resp = client.post(
            f"{self.ENDPOINT}/{job.job_id}/complete",
            params={"tokens_generated": 1500},
            headers=admin_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"
        assert resp.json()["job_id"] == job.job_id


# ===================================================================
# POST /v1/marketplace/jobs/{job_id}/cancel
# ===================================================================


class TestCancelJob:
    """Cancel a marketplace job."""

    ENDPOINT = "/v1/marketplace/jobs"

    def test_no_auth(self, client, _setup_marketplace):
        """Returns 401 without auth."""
        resp = client.post(f"{self.ENDPOINT}/some-id/cancel")
        assert resp.status_code == 401

    def test_no_marketplace(self, client, admin_headers):
        """Returns 503 when marketplace is not available."""
        resp = client.post(
            f"{self.ENDPOINT}/some-id/cancel", headers=admin_headers
        )
        assert resp.status_code == 503

    def test_cannot_cancel(self, client, marketplace, _setup_marketplace, admin_headers):
        """Returns 400 when the job is already completed."""
        marketplace.create_listing(
            provider_id="p1", gpu_name="A100",
            gpu_memory_bytes=80_000_000_000, price_per_hour=2.0,
        )
        job = marketplace.post_job(requester_id="r1", model_name="m1")
        marketplace.match_job(job.job_id)
        marketplace.start_job(job.job_id)
        marketplace.complete_job(job.job_id)

        resp = client.post(
            f"{self.ENDPOINT}/{job.job_id}/cancel", headers=admin_headers
        )
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Cannot cancel job"

    def test_success(self, client, marketplace, _setup_marketplace, admin_headers):
        """Cancels an open job."""
        job = marketplace.post_job(requester_id="r1", model_name="m1")
        resp = client.post(
            f"{self.ENDPOINT}/{job.job_id}/cancel", headers=admin_headers
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"
        assert resp.json()["job_id"] == job.job_id


# ===================================================================
# GET /v1/marketplace/providers/{provider_id}/earnings
# ===================================================================


class TestGetProviderEarnings:
    """Get earnings summary for a GPU provider."""

    ENDPOINT = "/v1/marketplace/providers"

    def test_no_marketplace(self, client):
        """Returns 503 when marketplace is not available."""
        resp = client.get(f"{self.ENDPOINT}/p1/earnings")
        assert resp.status_code == 503

    def test_not_found(self, client, _setup_marketplace):
        """Returns 404 for an unknown provider."""
        resp = client.get(f"{self.ENDPOINT}/unknown/earnings")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Provider not found"

    def test_success_initial(self, client, marketplace, _setup_marketplace):
        """Returns initial (empty) earnings for a provider with a listing."""
        marketplace.create_listing(
            provider_id="p1", gpu_name="A100",
            gpu_memory_bytes=80_000_000_000, price_per_hour=2.0,
        )
        resp = client.get(f"{self.ENDPOINT}/p1/earnings")
        assert resp.status_code == 200
        data = resp.json()
        assert data["provider_id"] == "p1"
        assert data["total_earnings"] == 0.0
        assert data["total_jobs"] == 0
        assert data["current_month_earnings"] == 0.0

    def test_success_with_earnings(self, client, marketplace, _setup_marketplace):
        """Returns accumulated earnings after completed jobs."""
        marketplace.create_listing(
            provider_id="p1", gpu_name="A100",
            gpu_memory_bytes=80_000_000_000, price_per_hour=2.0,
        )
        job = marketplace.post_job(requester_id="r1", model_name="m1")
        marketplace.match_job(job.job_id)
        marketplace.start_job(job.job_id)
        marketplace.complete_job(job.job_id, tokens_generated=500)

        resp = client.get(f"{self.ENDPOINT}/p1/earnings")
        assert resp.status_code == 200
        data = resp.json()
        assert data["provider_id"] == "p1"
        assert data["total_jobs"] >= 1
        assert data["current_month_earnings"] >= 0


# ===================================================================
# GET /v1/marketplace/stats
# ===================================================================


class TestGetMarketplaceStats:
    """Get overall marketplace statistics."""

    ENDPOINT = "/v1/marketplace/stats"

    def test_no_marketplace(self, client):
        """Returns 503 when marketplace is not available."""
        resp = client.get(self.ENDPOINT)
        assert resp.status_code == 503

    def test_empty_marketplace(self, client, _setup_marketplace):
        """Returns zeroed stats when no data exists."""
        resp = client.get(self.ENDPOINT)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_listings"] == 0
        assert data["active_listings"] == 0
        assert data["total_jobs"] == 0
        assert data["open_jobs"] == 0
        assert data["running_jobs"] == 0
        assert data["completed_jobs"] == 0
        assert data["total_volume_usd"] == 0.0
        assert data["total_tokens_served"] == 0
        assert data["avg_price_per_hour"] == 0.0

    def test_with_data(self, client, marketplace, _setup_marketplace):
        """Returns aggregate stats reflecting the current state."""
        marketplace.create_listing(
            provider_id="p1", gpu_name="A100",
            gpu_memory_bytes=80_000_000_000, price_per_hour=2.0,
        )
        marketplace.create_listing(
            provider_id="p2", gpu_name="V100",
            gpu_memory_bytes=32_000_000_000, price_per_hour=1.0,
        )
        marketplace.post_job(requester_id="r1", model_name="llama-3-8b")

        resp = client.get(self.ENDPOINT)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_listings"] == 2
        assert data["active_listings"] == 2
        assert data["total_jobs"] == 1
        assert data["open_jobs"] == 1
        assert data["avg_price_per_hour"] == 1.5
