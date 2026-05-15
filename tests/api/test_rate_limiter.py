"""Tests for Feature 22: API Rate Limiter."""

import time

import pytest

from distllm.api.rate_limiter import TokenBucket, RateLimiter


class TestTokenBucket:
    def test_initial_tokens_equals_burst_size(self):
        bucket = TokenBucket(rate_per_minute=60.0, burst_multiplier=1.5)
        assert bucket.tokens == 90.0  # 60 * 1.5

    def test_consume_reduces_tokens(self):
        bucket = TokenBucket(rate_per_minute=60.0, burst_multiplier=1.0)
        assert bucket.consume() is True
        assert bucket.tokens < 60.0

    def test_consume_returns_false_when_empty(self):
        bucket = TokenBucket(rate_per_minute=1.0, burst_multiplier=0.01)
        # Drain all tokens
        for _ in range(100):
            bucket.consume()
        assert bucket.consume() is False

    def test_refill_over_time(self):
        bucket = TokenBucket(rate_per_minute=60.0, burst_multiplier=1.0)
        # Consume all tokens
        for _ in range(60):
            bucket.consume()
        assert bucket.tokens < 1.0

        # Wait a bit for refill (1 token per second at 60 RPM)
        time.sleep(0.15)
        bucket._refill()
        assert bucket.tokens > 0

    def test_get_remaining(self):
        bucket = TokenBucket(rate_per_minute=60.0, burst_multiplier=1.0)
        remaining = bucket.get_remaining()
        assert remaining == 60

    def test_get_retry_after_when_available(self):
        bucket = TokenBucket(rate_per_minute=60.0, burst_multiplier=1.0)
        assert bucket.get_retry_after() == 0.0

    def test_burst_size_calculation(self):
        bucket = TokenBucket(rate_per_minute=100.0, burst_multiplier=2.0)
        assert bucket.burst_size == 200
        assert bucket.max_tokens == 200.0


class TestRateLimiter:
    def test_default_allows_requests(self):
        limiter = RateLimiter(default_rpm=60.0)
        assert limiter.is_allowed("client-1", "/v1/chat/completions") is True

    def test_per_client_isolation(self):
        limiter = RateLimiter(default_rpm=60.0, burst_multiplier=1.0)
        # Drain client-1's bucket (60 tokens)
        for _ in range(60):
            limiter.is_allowed("client-1", "/api")
        # client-1 should be blocked
        assert limiter.is_allowed("client-1", "/api") is False
        # client-2 should still have tokens
        assert limiter.is_allowed("client-2", "/api") is True

    def test_endpoint_specific_limits(self):
        limiter = RateLimiter(
            default_rpm=60.0,
            endpoint_limits={"/v1/chat/completions": 2.0},
            burst_multiplier=1.0,
        )
        # Use up chat completion tokens
        limiter.is_allowed("client-1", "/v1/chat/completions")
        limiter.is_allowed("client-1", "/v1/chat/completions")
        assert limiter.is_allowed("client-1", "/v1/chat/completions") is False
        # Other endpoint should still work
        assert limiter.is_allowed("client-1", "/other") is True

    def test_get_limits(self):
        limiter = RateLimiter(default_rpm=60.0, burst_multiplier=1.0)
        limit, remaining, retry_after = limiter.get_limits("client-1", "/api")
        assert limit == 60
        assert remaining == 60
        assert retry_after == 0.0

    def test_reset_client(self):
        limiter = RateLimiter(default_rpm=60.0, burst_multiplier=1.0)
        # Drain all tokens
        for _ in range(90):
            limiter.is_allowed("client-1", "/api")
        assert limiter.is_allowed("client-1", "/api") is False
        # Reset should give fresh tokens
        limiter.reset_client("client-1")
        assert limiter.is_allowed("client-1", "/api") is True

    def test_reset_all(self):
        limiter = RateLimiter(default_rpm=60.0, burst_multiplier=1.0)
        # Drain both clients
        for _ in range(90):
            limiter.is_allowed("c1", "/api")
            limiter.is_allowed("c2", "/api")
        limiter.reset_all()
        # Should have fresh tokens
        assert limiter.is_allowed("c1", "/api") is True
        assert limiter.is_allowed("c2", "/api") is True

    def test_rate_limit_exhaustion(self):
        limiter = RateLimiter(default_rpm=5.0, burst_multiplier=1.0)
        # 5 requests should be allowed
        for _ in range(5):
            assert limiter.is_allowed("client-1", "/api") is True
        # 6th should be blocked
        assert limiter.is_allowed("client-1", "/api") is False


class TestRateLimitMiddleware:
    def test_middleware_disabled_passes_through(self):
        from starlette.testclient import TestClient
        from fastapi import FastAPI
        from distllm.api.rate_limiter import RateLimiter
        from distllm.api.rate_limit_middleware import RateLimitMiddleware

        app = FastAPI()

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        limiter = RateLimiter(default_rpm=1.0, burst_multiplier=0.0)
        app.add_middleware(RateLimitMiddleware, rate_limiter=limiter, enabled=False)

        client = TestClient(app)
        response = client.get("/test")
        assert response.status_code == 200

    def test_middleware_allows_when_within_limit(self):
        from starlette.testclient import TestClient
        from fastapi import FastAPI
        from distllm.api.rate_limiter import RateLimiter
        from distllm.api.rate_limit_middleware import RateLimitMiddleware

        app = FastAPI()

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        limiter = RateLimiter(default_rpm=60.0, burst_multiplier=1.0)
        app.add_middleware(RateLimitMiddleware, rate_limiter=limiter, enabled=True)

        client = TestClient(app)
        response = client.get("/test")
        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers

    def test_middleware_returns_429_when_exhausted(self):
        from starlette.testclient import TestClient
        from fastapi import FastAPI
        from distllm.api.rate_limiter import RateLimiter
        from distllm.api.rate_limit_middleware import RateLimitMiddleware

        app = FastAPI()

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        # Small bucket: 2 RPM with burst=1.0 gives 2 tokens total
        limiter = RateLimiter(default_rpm=2.0, burst_multiplier=1.0)
        app.add_middleware(RateLimitMiddleware, rate_limiter=limiter, enabled=True)

        client = TestClient(app)
        # Send requests with consistent client ID via X-Forwarded-For
        headers = {"X-Forwarded-For": "1.2.3.4"}
        response = None
        # First 2 should succeed, 3rd should be 429
        for _ in range(5):
            response = client.get("/test", headers=headers)
            if response.status_code == 429:
                break
        assert response is not None
        assert response.status_code == 429
        data = response.json()
        assert data["error"] == "rate_limit_exceeded"
        assert "retry_after" in data

    def test_middleware_skips_health_and_metrics(self):
        from starlette.testclient import TestClient
        from fastapi import FastAPI
        from distllm.api.rate_limiter import RateLimiter
        from distllm.api.rate_limit_middleware import RateLimitMiddleware

        app = FastAPI()

        @app.get("/health")
        def health():
            return {"status": "ok"}

        @app.get("/metrics")
        def metrics():
            return "metrics"

        # Zero rate limit
        limiter = RateLimiter(default_rpm=0.001, burst_multiplier=0.0)
        app.add_middleware(RateLimitMiddleware, rate_limiter=limiter, enabled=True)

        client = TestClient(app)
        # Health and metrics should still work
        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 200
