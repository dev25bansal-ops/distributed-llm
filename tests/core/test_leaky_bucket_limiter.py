"""Tests for LeakyBucketRateLimiter -- rate limiting with leaky bucket algorithm.

Covers:
- Construction with default rate/burst
- LeakyBucket try_add and _drip
- check passes within limits
- check blocks when over limit
- check_many returns dict
- set_limit changes rate/burst
- reset_key restores bucket
- remaining capacity
- stats report

No MagicMock -- real time and counters.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/leaky_bucket_limiter.py")
LeakyBucket = _mod.LeakyBucket
LeakyBucketRateLimiter = _mod.LeakyBucketRateLimiter


class TestLeakyBucketConstruction:
    """LeakyBucket dataclass construction."""

    def test_default_construction(self) -> None:
        bucket = LeakyBucket(key="test", rate_per_sec=10.0, burst=20)
        assert bucket.key == "test"
        assert bucket.rate_per_sec == 10.0
        assert bucket.burst == 20
        assert bucket.water == 0.0
        assert bucket.consecutive_violations == 0

    def test_last_drip_is_set(self) -> None:
        bucket = LeakyBucket(key="test", rate_per_sec=5.0, burst=10)
        assert bucket.last_drip > 0


class TestLeakyBucketDrip:
    """Drip mechanic."""

    def test_drip_drains_water(self) -> None:
        bucket = LeakyBucket(key="test", rate_per_sec=10.0, burst=20)
        bucket.water = 10.0
        bucket.last_drip = time.time() - 1.0  # 1 second ago
        bucket._drip()
        # Should have drained ~10 tokens (10/sec * 1s)
        assert bucket.water < 10.0
        assert bucket.water >= 0.0

    def test_drip_never_negative(self) -> None:
        bucket = LeakyBucket(key="test", rate_per_sec=100.0, burst=10)
        bucket.water = 5.0
        bucket.last_drip = time.time() - 10.0  # 10 seconds ago
        bucket._drip()
        assert bucket.water == 0.0


class TestLeakyBucketTryAdd:
    """Adding tokens."""

    def test_try_add_accepts_within_burst(self) -> None:
        bucket = LeakyBucket(key="test", rate_per_sec=10.0, burst=5)
        assert bucket.try_add(3.0) is True
        assert bucket.water == 3.0

    def test_try_add_rejects_exceeding_burst(self) -> None:
        bucket = LeakyBucket(key="test", rate_per_sec=10.0, burst=5)
        bucket.water = 4.0
        bucket.last_drip = time.time()  # fresh timestamp so no drip
        assert bucket.try_add(2.0) is False
        # Water may drip slightly if time has passed; check it's <= 4 but unchanged by try_add
        assert bucket.water <= 4.0
        assert bucket.consecutive_violations >= 1

    def test_try_add_drips_first(self) -> None:
        bucket = LeakyBucket(key="test", rate_per_sec=100.0, burst=10)
        bucket.water = 10.0
        bucket.last_drip = time.time() - 1.0  # 1 sec = 100 tokens drained
        # After drip, water should be 0, should accept
        assert bucket.try_add(5.0) is True


class TestLeakyBucketWaitTime:
    """Wait time estimation."""

    def test_wait_time_zero_when_below_burst(self) -> None:
        bucket = LeakyBucket(key="test", rate_per_sec=10.0, burst=5)
        assert bucket.wait_time() == 0.0

    def test_wait_time_positive_when_full(self) -> None:
        bucket = LeakyBucket(key="test", rate_per_sec=10.0, burst=5)
        bucket.water = 5.0
        # excess = 5 + 1 - 5 = 1 token / 10 rps = 0.1s
        assert bucket.wait_time() == pytest.approx(0.1, rel=0.5)


class TestLeakyBucketRateLimiterConstruction:
    """Rate limiter construction."""

    def test_default_construction(self) -> None:
        limiter = LeakyBucketRateLimiter()
        assert limiter._default_rate == 10.0
        assert limiter._default_burst == 20
        assert limiter._enable_backoff is True
        assert limiter._buckets == {}

    def test_custom_limits(self) -> None:
        limiter = LeakyBucketRateLimiter(default_rate=5.0, default_burst=10, enable_backoff=False)
        assert limiter._default_rate == 5.0
        assert limiter._default_burst == 10
        assert limiter._enable_backoff is False


class TestLeakyBucketRateLimiterCheck:
    """Check method."""

    def test_check_allows_within_limits(self) -> None:
        limiter = LeakyBucketRateLimiter(default_rate=100.0, default_burst=50, enable_backoff=False)
        assert limiter.check("user:1") is True

    def test_check_blocks_over_limit(self) -> None:
        limiter = LeakyBucketRateLimiter(default_rate=1.0, default_burst=2, enable_backoff=False)
        assert limiter.check("user:1", tokens=2.0) is True
        assert limiter.check("user:1", tokens=1.0) is False

    def test_check_many(self) -> None:
        limiter = LeakyBucketRateLimiter(default_rate=100.0, default_burst=10, enable_backoff=False)
        results = limiter.check_many(["user:1", "user:2"])
        assert len(results) == 2
        assert results["user:1"] is True

    def test_remaining_capacity(self) -> None:
        limiter = LeakyBucketRateLimiter(default_rate=100.0, default_burst=10, enable_backoff=False)
        assert limiter.remaining("user:1") == 10.0
        limiter.check("user:1", tokens=3.0)
        remaining = limiter.remaining("user:1")
        # Drip may have run between check and remaining call, so water <= 3.0
        # remaining = burst - water = 10 - (up to 3.0)
        assert remaining >= 6.9 and remaining <= 10.0


class TestLeakyBucketRateLimiterSetLimit:
    """Custom limits."""

    def test_set_limit_updates_bucket(self) -> None:
        limiter = LeakyBucketRateLimiter(default_rate=10.0, default_burst=20, enable_backoff=False)
        limiter.set_limit("user:admin", rate=100.0, burst=200)
        assert limiter._custom_limits["user:admin"] == (100.0, 200)
        # First check creates bucket with custom limits
        limiter.check("user:admin")
        assert limiter._buckets["user:admin"].rate_per_sec == 100.0
        assert limiter._buckets["user:admin"].burst == 200

    def test_set_limit_on_existing_bucket(self) -> None:
        limiter = LeakyBucketRateLimiter(default_rate=10.0, default_burst=20, enable_backoff=False)
        limiter.check("user:1")
        limiter.set_limit("user:1", rate=50.0, burst=100)
        assert limiter._buckets["user:1"].burst == 100

    def test_reset_key(self) -> None:
        limiter = LeakyBucketRateLimiter(default_rate=10.0, default_burst=20, enable_backoff=False)
        limiter.check("user:1", tokens=15.0)
        limiter.reset_key("user:1")
        assert limiter._buckets["user:1"].water == 0.0


class TestLeakyBucketRateLimiterStats:
    """Statistics."""

    def test_stats_default(self) -> None:
        limiter = LeakyBucketRateLimiter(enable_backoff=False)
        s = limiter.stats()
        assert s["total_keys"] == 0
        assert s["active_keys"] == 0
        assert s["violators_in_backoff"] == 0

    def test_stats_after_checks(self) -> None:
        limiter = LeakyBucketRateLimiter(default_rate=10.0, default_burst=5, enable_backoff=False)
        limiter.check("user:1")
        s = limiter.stats()
        assert s["total_keys"] >= 1
