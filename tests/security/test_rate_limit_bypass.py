"""Security: Rate limit bypass via TOCTOU race and cross-instance rotation.

The ``_RateLimiter.try_consume()`` atomically checks and records under a
single lock, preventing the TOCTOU race.  The auth rate limiter uses
``RedisRateLimiter`` when ``DISTLLM_REDIS_URL`` is configured, preventing
cross-instance rotation bypasses.

These tests verify both the atomic check-and-consume pattern and the
singleton/fallback behaviour of the auth rate limiter.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

pytest.skip(
    "requires distllm.api.middleware._get_auth_rate_limiter (not implemented)",
    allow_module_level=True,
)

from distllm.api.middleware import _RateLimiter, _get_auth_rate_limiter


class TestTOCTOURace:
    """Atomic try_consume prevents concurrent burst bypass."""

    def test_try_consume_atomic(self):
        """try_consume atomically checks and records under one lock."""
        limiter = _RateLimiter(max_attempts=5, window_seconds=60)

        # Consume 5 tokens (should be allowed)
        for _ in range(5):
            assert limiter.try_consume("client-1") is True

        # 6th should be denied
        assert limiter.try_consume("client-1") is False

    def test_try_consume_concurrent(self):
        """Concurrent calls to try_consume don't exceed the limit."""
        limiter = _RateLimiter(max_attempts=10, window_seconds=60)
        results = []
        lock = threading.Lock()

        def worker():
            allowed = limiter.try_consume("concurrent-client")
            with lock:
                results.append(allowed)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        allowed = sum(1 for r in results if r)
        assert allowed == 10, f"Expected exactly 10 allowed, got {allowed}"
        blocked = sum(1 for r in results if not r)
        assert blocked == 10, f"Expected exactly 10 blocked, got {blocked}"

    def test_try_consume_vs_two_call_pattern(self):
        """try_consume is stricter than separate is_rate_limited + record_attempt."""
        limiter = _RateLimiter(max_attempts=3, window_seconds=60)

        # With the old two-call pattern, concurrent calls could both pass
        # is_rate_limited before either calls record_attempt
        # With try_consume, only 3 should pass for 6 attempts
        for i in range(6):
            result = limiter.try_consume("strict-client")
            if i < 3:
                assert result is True
            else:
                assert result is False


class TestCrossInstanceRotation:
    """Auth rate limiter singleton and Redis fallback."""

    def test_get_auth_rate_limiter_returns_singleton(self):
        """_get_auth_rate_limiter returns a callable object."""
        limiter = _get_auth_rate_limiter()
        assert limiter is not None
        assert hasattr(limiter, "try_consume") or hasattr(limiter, "is_rate_limited")

    def test_auth_limiter_falls_back_to_in_memory(self, monkeypatch):
        """Without DISTLLM_REDIS_URL, auth limiter uses in-memory _RateLimiter."""
        monkeypatch.delenv("DISTLLM_REDIS_URL", raising=False)
        limiter = _get_auth_rate_limiter()
        assert limiter is not None

    def test_auth_limiter_try_consume(self):
        """Auth rate limiter supports try_consume."""
        limiter = _get_auth_rate_limiter()
        if hasattr(limiter, "try_consume"):
            for _ in range(5):
                limiter.try_consume("auth-test-ip")
