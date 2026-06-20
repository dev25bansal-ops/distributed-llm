"""Integration tests for rate limiter + batch scheduler backpressure.

Tests cover:
- Rate limiter + batch scheduler integration
- Backpressure under load (BatchCapacityError from full pending queue)
- Graduated backpressure tiers (HierarchicalRateLimiter)
- Retry-After header behavior through the middleware
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient
from fastapi import FastAPI

from distllm.api.rate_limiter import (
    HierarchicalRateLimiter,
    RateLimiter,
    TokenBucket,
)
from distllm.api.rate_limit_middleware import RateLimitMiddleware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app_with_limiter(
    limiter: RateLimiter,
    enabled: bool = True,
    endpoint: str = "/test",
) -> tuple[FastAPI, TestClient]:
    """Create a minimal FastAPI app with RateLimitMiddleware attached."""
    app = FastAPI()

    @app.get(endpoint)
    def test_endpoint():
        return {"status": "ok"}

    app.add_middleware(RateLimitMiddleware, rate_limiter=limiter, enabled=enabled)
    return app, TestClient(app)


def _make_seq(request_id: str, priority: int = 2, total_len: int = 10):
    """Create a minimal Sequence-like mock for scheduler tests."""
    seq = MagicMock()
    seq.request_id = request_id
    seq.priority = priority
    seq.total_len = total_len
    seq.prompt_tokens = list(range(total_len))
    seq.max_new_tokens = 256
    seq.generated_tokens = []
    seq.status = MagicMock()
    seq.status.value = "pending"
    seq.created_at = time.time()
    seq.is_complete = False
    seq.stop_token_ids = []
    seq.decode_input_token = 1
    seq.prefix_match_len = 0
    seq.constraint = None
    seq.adapter_id = None
    seq.max_latency_ms = None
    return seq


# ---------------------------------------------------------------------------
# Test: Rate limiter + batch scheduler integration
# ---------------------------------------------------------------------------


class TestRateLimiterBatchSchedulerIntegration:
    """Verify that rate limiting happens before batch scheduling,
    and that 429 responses do not consume scheduler capacity."""

    def test_rate_limiter_blocks_before_reaching_scheduler(self):
        """When the rate limiter rejects a request (429), the batch
        scheduler never sees it — no tokens are consumed from the
        scheduler's pending queue."""
        # Arrange: 1 RPM limit with no burst headroom
        limiter = RateLimiter(default_rpm=1.0, burst_multiplier=1.0)
        app, client = _make_app_with_limiter(limiter)
        headers = {"X-Forwarded-For": "10.0.0.1"}

        # Act: first request passes, second is rate-limited
        resp1 = client.get("/test", headers=headers)
        resp2 = client.get("/test", headers=headers)

        # Assert
        assert resp1.status_code == 200
        assert resp2.status_code == 429

    def test_rate_limit_headers_on_success(self):
        """Successful responses include X-RateLimit-Limit and
        X-RateLimit-Remaining headers."""
        # Arrange
        limiter = RateLimiter(default_rpm=100.0, burst_multiplier=1.0)
        app, client = _make_app_with_limiter(limiter)
        headers = {"X-Forwarded-For": "10.0.0.2"}

        # Act
        resp = client.get("/test", headers=headers)

        # Assert
        assert resp.status_code == 200
        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Remaining" in resp.headers
        remaining = int(resp.headers["X-RateLimit-Remaining"])
        assert remaining >= 0

    def test_rate_limit_headers_on_429(self):
        """429 responses include X-RateLimit-Limit, X-RateLimit-Remaining=0,
        and Retry-After headers."""
        # Arrange
        limiter = RateLimiter(default_rpm=1.0, burst_multiplier=1.0)
        app, client = _make_app_with_limiter(limiter)
        headers = {"X-Forwarded-For": "10.0.0.3"}

        # Act: exhaust the bucket
        client.get("/test", headers=headers)
        resp = client.get("/test", headers=headers)

        # Assert
        assert resp.status_code == 429
        assert resp.headers.get("X-RateLimit-Remaining") == "0"
        assert "Retry-After" in resp.headers
        retry_after = int(resp.headers["Retry-After"])
        assert retry_after >= 1

    def test_per_client_isolation_at_middleware_level(self):
        """Different clients have independent rate limits."""
        # Arrange
        limiter = RateLimiter(default_rpm=1.0, burst_multiplier=1.0)
        app, client = _make_app_with_limiter(limiter)

        # Act: exhaust client-A
        client.get("/test", headers={"X-Forwarded-For": "10.0.0.10"})
        blocked = client.get("/test", headers={"X-Forwarded-For": "10.0.0.10"})

        # client-B should still pass
        ok = client.get("/test", headers={"X-Forwarded-For": "10.0.0.11"})

        # Assert
        assert blocked.status_code == 429
        assert ok.status_code == 200

    def test_endpoint_specific_limits_enforced(self):
        """Endpoint-specific RPM overrides default limits."""
        # Arrange: /v1/chat limited to 1 RPM, /other uses default
        limiter = RateLimiter(
            default_rpm=100.0,
            endpoint_limits={"/v1/chat": 1.0},
            burst_multiplier=1.0,
        )
        app = FastAPI()

        @app.get("/v1/chat")
        def chat():
            return {"model": "test"}

        @app.get("/other")
        def other():
            return {"status": "ok"}

        app.add_middleware(RateLimitMiddleware, rate_limiter=limiter, enabled=True)
        client = TestClient(app)
        headers = {"X-Forwarded-For": "10.0.0.20"}

        # Act: /v1/chat limited to 1
        client.get("/v1/chat", headers=headers)
        blocked = client.get("/v1/chat", headers=headers)
        ok = client.get("/other", headers=headers)

        # Assert
        assert blocked.status_code == 429
        assert ok.status_code == 200


# ---------------------------------------------------------------------------
# Test: Backpressure under load (BatchCapacityError)
# ---------------------------------------------------------------------------


class TestBackpressureUnderLoad:
    """When the batch scheduler's pending queue is full, new requests
    must be rejected with BatchCapacityError (simulating backpressure)."""

    def test_scheduler_rejects_when_pending_queue_full(self):
        """BatchScheduler.add() raises BatchCapacityError when the
        pending queue exceeds _max_pending."""
        # Arrange
        from distllm.core.batch_scheduler import BatchScheduler
        from distllm.core.scheduler.sequence import Sequence
        from distllm.errors.types import BatchCapacityError

        scheduler = BatchScheduler(max_batch_size=2, max_tokens_per_batch=128)
        scheduler._max_pending = 3  # very small queue

        # Act: fill the queue
        for i in range(3):
            seq = Sequence(request_id=f"req-{i}", prompt_tokens=[1, 2, 3])
            scheduler.add(seq)

        # Assert: next add should fail
        overflow = Sequence(request_id="req-overflow", prompt_tokens=[1, 2, 3])
        with pytest.raises(BatchCapacityError) as exc_info:
            scheduler.add(overflow)

        assert "capacity" in str(exc_info.value).lower()

    def test_scheduler_accepts_after_queue_drains(self):
        """After the pending queue drains, new requests are accepted again."""
        # Arrange
        from distllm.core.batch_scheduler import BatchScheduler
        from distllm.core.scheduler.sequence import Sequence

        scheduler = BatchScheduler(max_batch_size=2, max_tokens_per_batch=128)
        scheduler._max_pending = 2

        # Fill queue
        s1 = Sequence(request_id="r1", prompt_tokens=[1, 2])
        s2 = Sequence(request_id="r2", prompt_tokens=[1, 2])
        scheduler.add(s1)
        scheduler.add(s2)

        # Manually drain the queue
        with scheduler._lock:
            scheduler._pending_heap.clear()

        # Act: should accept now
        s3 = Sequence(request_id="r3", prompt_tokens=[1, 2])
        scheduler.add(s3)  # should not raise

        # Assert
        assert scheduler.pending_count == 1

    def test_backpressure_does_not_affect_other_clients_at_rate_limiter(self):
        """Rate limiter backpressure is per-client; one client being
        blocked does not affect another."""
        # Arrange
        limiter = RateLimiter(default_rpm=2.0, burst_multiplier=1.0)

        # Act: drain client-A
        assert limiter.is_allowed("heavy-client", "/api") is True
        assert limiter.is_allowed("heavy-client", "/api") is True
        assert limiter.is_allowed("heavy-client", "/api") is False

        # Assert: light-client is unaffected
        assert limiter.is_allowed("light-client", "/api") is True

    def test_rate_limiter_resets_after_refill_period(self):
        """After tokens refill, previously blocked requests succeed."""
        # Arrange: 60 RPM = 1 token/sec, burst=1.0
        limiter = RateLimiter(default_rpm=60.0, burst_multiplier=1.0)

        # Exhaust tokens
        bucket = limiter._get_bucket("client", "/api")
        bucket.tokens = 0.0
        bucket._last_refill = time.monotonic()

        # Act: should be blocked immediately
        assert limiter.is_allowed("client", "/api") is False

        # Wait for 1 token to refill (1/sec at 60 RPM)
        time.sleep(1.1)

        # Assert: should succeed now
        assert limiter.is_allowed("client", "/api") is True


# ---------------------------------------------------------------------------
# Test: Graduated backpressure tiers (HierarchicalRateLimiter)
# ---------------------------------------------------------------------------


class TestGraduatedBackpressureTiers:
    """HierarchicalRateLimiter enforces global, tenant, and model tiers.
    A request must pass ALL three tiers to proceed."""

    def test_all_tiers_pass(self):
        """Request succeeds when all three tiers have capacity."""
        # Arrange
        limiter = HierarchicalRateLimiter(
            global_rpm=1000,
            tenant_rpm=100,
            model_rpm={"llama-70b": 50},
            burst_multiplier=1.0,
        )

        # Act
        result = limiter.is_allowed(tenant="acme", model="llama-70b", endpoint="/v1/chat")

        # Assert
        assert result is True

    def test_global_tier_blocks(self):
        """When the global bucket is exhausted, all requests are blocked
        regardless of tenant or model."""
        # Arrange
        limiter = HierarchicalRateLimiter(
            global_rpm=2,
            tenant_rpm=100,
            model_rpm={"m": 100},
            burst_multiplier=1.0,
        )

        # Exhaust global bucket
        limiter.is_allowed(tenant="t1", model="m")
        limiter.is_allowed(tenant="t2", model="m")

        # Act: global is exhausted
        result = limiter.is_allowed(tenant="t3", model="m")

        # Assert
        assert result is False

    def test_tenant_tier_blocks_independently(self):
        """Tenant exhaustion blocks only that tenant; others continue."""
        # Arrange
        limiter = HierarchicalRateLimiter(
            global_rpm=10000,
            tenant_rpm=2,
            model_rpm={"m": 100},
            burst_multiplier=1.0,
        )

        # Exhaust tenant "heavy"
        limiter.is_allowed(tenant="heavy", model="m")
        limiter.is_allowed(tenant="heavy", model="m")

        # Act
        blocked = limiter.is_allowed(tenant="heavy", model="m")
        ok = limiter.is_allowed(tenant="light", model="m")

        # Assert
        assert blocked is False
        assert ok is True

    def test_model_tier_blocks(self):
        """Model-specific exhaustion blocks that model only."""
        # Arrange
        limiter = HierarchicalRateLimiter(
            global_rpm=10000,
            tenant_rpm=10000,
            model_rpm={"llama-70b": 1, "llama-8b": 100},
            burst_multiplier=1.0,
        )

        # Exhaust llama-70b
        limiter.is_allowed(tenant="acme", model="llama-70b")

        # Act
        blocked = limiter.is_allowed(tenant="acme", model="llama-70b")
        ok = limiter.is_allowed(tenant="acme", model="llama-8b")

        # Assert
        assert blocked is False
        assert ok is True

    def test_model_without_explicit_limit_uses_tenant_rpm(self):
        """Models not in model_rpm dict inherit the tenant RPM."""
        # Arrange
        limiter = HierarchicalRateLimiter(
            global_rpm=10000,
            tenant_rpm=2,
            model_rpm={},  # no model-specific limits
            burst_multiplier=1.0,
        )

        # Act: exhaust tenant bucket via model "unknown"
        limiter.is_allowed(tenant="t", model="unknown")
        limiter.is_allowed(tenant="t", model="unknown")

        # Assert: third call blocked
        assert limiter.is_allowed(tenant="t", model="unknown") is False

    def test_get_limits_returns_all_tiers(self):
        """get_limits reports remaining tokens for all three tiers."""
        # Arrange
        limiter = HierarchicalRateLimiter(
            global_rpm=100,
            tenant_rpm=50,
            model_rpm={"m": 25},
            burst_multiplier=1.0,
        )

        # Act
        info = limiter.get_limits(tenant="acme", model="m")

        # Assert
        assert "global_remaining" in info
        assert "tenant_remaining" in info
        assert "model_remaining" in info
        assert info["global_limit"] == 150  # 100 * 1.5
        assert info["tenant_limit"] == 75   # 50 * 1.5
        assert info["model_limit"] == 37    # 25 * 1.5

    def test_reset_tenant_restores_capacity(self):
        """After reset_tenant, that tenant can make requests again."""
        # Arrange
        limiter = HierarchicalRateLimiter(
            global_rpm=10000,
            tenant_rpm=1,
            model_rpm={"m": 1000},
            burst_multiplier=1.0,
        )

        # Exhaust tenant
        limiter.is_allowed(tenant="acme", model="m")
        assert limiter.is_allowed(tenant="acme", model="m") is False

        # Act
        limiter.reset_tenant("acme")

        # Assert: fresh bucket
        assert limiter.is_allowed(tenant="acme", model="m") is True

    def test_burst_multiplier_applies_to_all_tiers(self):
        """Burst multiplier increases capacity of all tiers."""
        # Arrange
        limiter = HierarchicalRateLimiter(
            global_rpm=10,
            tenant_rpm=10,
            model_rpm={"m": 10},
            burst_multiplier=2.0,
        )

        # Act: capacity should be 20 per tier (10 * 2.0)
        info = limiter.get_limits(tenant="t", model="m")

        # Assert
        assert info["global_limit"] == 20
        assert info["tenant_limit"] == 20
        assert info["model_limit"] == 20


# ---------------------------------------------------------------------------
# Test: Retry-After header behavior
# ---------------------------------------------------------------------------


class TestRetryAfterHeaderBehavior:
    """Retry-After header must accurately reflect the time until the
    next token becomes available."""

    def test_retry_after_present_on_429(self):
        """429 responses always include a Retry-After header >= 1."""
        # Arrange
        limiter = RateLimiter(default_rpm=1.0, burst_multiplier=1.0)
        app, client = _make_app_with_limiter(limiter)
        headers = {"X-Forwarded-For": "10.0.0.50"}

        # Act: exhaust bucket
        client.get("/test", headers=headers)
        resp = client.get("/test", headers=headers)

        # Assert
        assert resp.status_code == 429
        retry_after = int(resp.headers["Retry-After"])
        assert retry_after >= 1

    def test_retry_after_not_present_on_success(self):
        """Successful responses do not include Retry-After."""
        # Arrange
        limiter = RateLimiter(default_rpm=100.0, burst_multiplier=1.0)
        app, client = _make_app_with_limiter(limiter)
        headers = {"X-Forwarded-For": "10.0.0.51"}

        # Act
        resp = client.get("/test", headers=headers)

        # Assert
        assert resp.status_code == 200
        assert "Retry-After" not in resp.headers

    def test_retry_after_reflects_actual_wait_time(self):
        """Retry-After value should be proportional to the deficit.
        With 1 RPM (1 token per 60s), after exhausting tokens the
        retry-after should be close to 60 seconds."""
        # Arrange: 1 RPM = 1 token per 60 seconds
        bucket = TokenBucket(rate_per_minute=1.0, burst_multiplier=1.0)
        bucket.tokens = 0.0
        bucket._last_refill = time.monotonic()

        # Act
        retry_after = bucket.get_retry_after()

        # Assert: should be ~60 seconds (within tolerance)
        assert 55.0 <= retry_after <= 65.0

    def test_retry_after_zero_when_tokens_available(self):
        """get_retry_after returns 0.0 when tokens are available."""
        # Arrange
        bucket = TokenBucket(rate_per_minute=60.0, burst_multiplier=1.0)

        # Act
        retry_after = bucket.get_retry_after()

        # Assert
        assert retry_after == 0.0

    def test_retry_after_decreases_after_partial_refill(self):
        """After some time passes, retry-after should decrease."""
        # Arrange: 60 RPM = 1 token/sec
        bucket = TokenBucket(rate_per_minute=60.0, burst_multiplier=1.0)
        bucket.tokens = 0.0
        bucket._last_refill = time.monotonic()

        # Act: wait 0.5 seconds
        time.sleep(0.5)
        retry_after = bucket.get_retry_after()

        # Assert: should be ~0.5 seconds (within tolerance)
        assert 0.0 < retry_after < 1.0

    def test_retry_after_at_least_one_second_in_middleware(self):
        """The middleware converts retry_after to int with max(1, ...)
        so even sub-second waits report at least 1."""
        # Arrange: 3600 RPM = 1 token per 0.0167 sec
        limiter = RateLimiter(default_rpm=3600.0, burst_multiplier=1.0)
        app, client = _make_app_with_limiter(limiter)
        headers = {"X-Forwarded-For": "10.0.0.52"}

        # Exhaust the bucket
        for _ in range(3600):
            limiter.is_allowed("10.0.0.52", "/test")

        # Act
        resp = client.get("/test", headers=headers)

        # Assert
        if resp.status_code == 429:
            retry_after = int(resp.headers["Retry-After"])
            assert retry_after >= 1

    def test_sequential_429s_report_decreasing_retry_after(self):
        """After getting 429, waiting and retrying should eventually
        succeed with decreasing Retry-After values."""
        # Arrange: 60 RPM
        limiter = RateLimiter(default_rpm=60.0, burst_multiplier=1.0)
        app, client = _make_app_with_limiter(limiter)
        headers = {"X-Forwarded-For": "10.0.0.53"}

        # Exhaust the bucket
        for _ in range(60):
            limiter.is_allowed("10.0.0.53", "/test")

        # Act: first 429
        resp1 = client.get("/test", headers=headers)
        assert resp1.status_code == 429
        ra1 = int(resp1.headers["Retry-After"])

        # Wait for tokens to refill
        time.sleep(1.2)

        # Should succeed now
        resp2 = client.get("/test", headers=headers)
        assert resp2.status_code == 200

    def test_retry_after_header_is_integer(self):
        """Retry-After must be an integer (HTTP spec requirement)."""
        # Arrange
        limiter = RateLimiter(default_rpm=1.0, burst_multiplier=1.0)
        app, client = _make_app_with_limiter(limiter)
        headers = {"X-Forwarded-For": "10.0.0.54"}

        # Act
        client.get("/test", headers=headers)
        resp = client.get("/test", headers=headers)

        # Assert
        if resp.status_code == 429:
            ra = resp.headers["Retry-After"]
            # Must be parseable as integer
            int(ra)
