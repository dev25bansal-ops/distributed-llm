"""Rate limiter sliding window tests.

Covers in-memory fallback behavior, window expiry, retry-after computation,
and property-based monotonic rate limiting invariants.
"""

import asyncio
import socket
import struct
import threading
import time
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import numpy as np

try:
    from hypothesis import given, strategies as st, settings as hp_settings
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


from tests.comprehensive.conftest import _load_module

# Load clean modules
_redis_limiter = _load_module("distllm/api/redis_rate_limiter.py")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Rate Limiter Sliding Window
# ═══════════════════════════════════════════════════════════════════════════

class TestRateLimiterSlidingWindow:
    """Sliding window rate limiter: in-memory fallback behavior."""

    @pytest.fixture
    def limiter(self):
        return _redis_limiter.RedisRateLimiter(
            redis_url=None, max_attempts=5, window_seconds=60
        )

    @pytest.mark.asyncio
    async def test_not_limited_initially(self, limiter):
        assert not await limiter.is_rate_limited("1.2.3.4")

    @pytest.mark.asyncio
    async def test_limited_after_max_attempts(self, limiter):
        for _ in range(5):
            await limiter.record_attempt("1.2.3.4")
        assert await limiter.is_rate_limited("1.2.3.4")

    @pytest.mark.asyncio
    async def test_not_limited_below_max(self, limiter):
        for _ in range(3):
            await limiter.record_attempt("1.2.3.4")
        assert not await limiter.is_rate_limited("1.2.3.4")

    @pytest.mark.asyncio
    async def test_different_ips_independent(self, limiter):
        for _ in range(5):
            await limiter.record_attempt("1.2.3.4")
        assert not await limiter.is_rate_limited("5.6.7.8")

    @pytest.mark.asyncio
    async def test_window_expires_old_entries(self, limiter):
        limiter._window = 0.01
        for _ in range(5):
            await limiter.record_attempt("1.2.3.4")
        assert await limiter.is_rate_limited("1.2.3.4")
        await _async_sleep(0.02)
        assert not await limiter.is_rate_limited("1.2.3.4")

    @pytest.mark.asyncio
    async def test_retry_after_returns_zero_when_not_limited(self, limiter):
        retry = await limiter.retry_after("1.2.3.4")
        assert retry == 0

    @pytest.mark.asyncio
    async def test_retry_after_positive_when_limited(self, limiter):
        limiter._window = 10
        for _ in range(5):
            await limiter.record_attempt("1.2.3.4")
        retry = await limiter.retry_after("1.2.3.4")
        assert retry >= 1

    @pytest.mark.asyncio
    async def test_window_duration_affects_limiting(self, limiter):
        limiter_short = _redis_limiter.RedisRateLimiter(
            redis_url=None, max_attempts=5, window_seconds=0.5
        )
        for _ in range(5):
            await limiter_short.record_attempt("1.2.3.4")
        assert await limiter_short.is_rate_limited("1.2.3.4")
        await _async_sleep(0.6)
        assert not await limiter_short.is_rate_limited("1.2.3.4")

    @pytest.mark.asyncio
    async def test_max_attempts_threshold(self, limiter):
        low = _redis_limiter.RedisRateLimiter(
            redis_url=None, max_attempts=1, window_seconds=60
        )
        await low.record_attempt("1.2.3.4")
        assert await low.is_rate_limited("1.2.3.4")

    @pytest.mark.asyncio
    async def test_local_key_format(self, limiter):
        key = limiter._local_key("1.2.3.4")
        assert key == "distllm:ratelimit:1.2.3.4"

    @pytest.mark.asyncio
    async def test_retry_after_still_limited_after_some_expiry(self, limiter):
        limiter._window = 10
        now = time.time()
        limiter._local["1.2.3.4"] = [now - 8, now - 7, now - 6, now - 5, now - 4]
        retry = await limiter.retry_after("1.2.3.4")
        assert retry >= 1

    @pytest.mark.asyncio
    async def test_rate_limit_then_recover(self, limiter):
        limiter._window = 0.5
        for _ in range(5):
            await limiter.record_attempt("1.2.3.4")
        assert await limiter.is_rate_limited("1.2.3.4")
        await _async_sleep(0.6)
        assert not await limiter.is_rate_limited("1.2.3.4")

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @hp_settings(max_examples=50)
    @given(
        max_attempts=st.integers(min_value=1, max_value=20),
        attempts_made=st.integers(min_value=0, max_value=50),
    )
    @pytest.mark.asyncio
    async def test_rate_limit_monotonic_property(self, max_attempts, attempts_made):
        limiter = _redis_limiter.RedisRateLimiter(
            redis_url=None, max_attempts=max_attempts, window_seconds=60
        )
        for _ in range(attempts_made):
            await limiter.record_attempt("1.2.3.4")
        limited = await limiter.is_rate_limited("1.2.3.4")
        if attempts_made >= max_attempts:
            assert limited
        else:
            assert not limited


async def _async_sleep(seconds):
    """Async sleep helper compatible with Python 3.14."""
    await _sleep_raw(seconds)


async def _sleep_raw(delay):
    """asyncio sleep wrapper."""
    import asyncio
    await asyncio.sleep(delay)
