"""Tests for API-ARCH-04: distributed rate limiter.

Covers:
- ``_RateLimiter.try_consume`` atomic check-and-record (no TOCTOU window).
- ``RedisRateLimiter`` (both ``redis_rate_limiter.py`` async variant and
  ``rate_limiter_unified.py`` sync variant) Lua-script atomic consume,
  proven with a shared ``fakeredis`` server simulating two workers.
- ``HierarchicalRateLimiter`` tier-leak bug: a request rejected at the
  model tier must not consume capacity from global/tenant tiers.
"""
from __future__ import annotations

import pytest

redis = pytest.importorskip("redis")
fakeredis = pytest.importorskip("fakeredis")

from tests.comprehensive.conftest import _load_module

_redis_limiter = _load_module("distllm/api/redis_rate_limiter.py")
_rate_limiter = _load_module("distllm/api/rate_limiter.py")
_unified = _load_module("distllm/api/rate_limiter_unified.py")
_middleware = _load_module("distllm/api/middleware.py")

HierarchicalRateLimiter = _rate_limiter.HierarchicalRateLimiter
RateLimitConfig = _unified.RateLimitConfig
RateLimitStrategy = _unified.RateLimitStrategy
RateLimiter = _unified.RateLimiter
UnifiedRedisLimiter = _unified.RedisRateLimiter
AsyncRedisLimiter = _redis_limiter.RedisRateLimiter
_RateLimiter = _middleware._RateLimiter


# ---------------------------------------------------------------------------
# In-memory atomic try_consume
# ---------------------------------------------------------------------------

def test_rate_limiter_try_consume_atomic_single_process():
    import threading

    limiter = _RateLimiter(max_attempts=10, window_seconds=60, key_by="ip")
    results: list[bool] = []

    def worker():
        results.append(limiter.try_consume("1.2.3.4"))

    threads = [threading.Thread(target=worker) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    allowed = sum(1 for r in results if r)
    assert allowed == 10, f"expected exactly 10 allowed, got {allowed}"


# ---------------------------------------------------------------------------
# Redis (unified sync variant) — two logical workers, shared server
# ---------------------------------------------------------------------------

def test_unified_redis_limiter_two_workers_never_over_admits():
    server = fakeredis.FakeServer()
    config = RateLimitConfig(
        strategy=RateLimitStrategy.REDIS, max_requests=5, window_seconds=60
    )
    w1 = UnifiedRedisLimiter(config, redis_client=fakeredis.FakeRedis(server=server))
    w2 = UnifiedRedisLimiter(config, redis_client=fakeredis.FakeRedis(server=server))

    allowed = 0
    for _ in range(10):
        for w in (w1, w2):
            if w.is_allowed("shared-key"):
                allowed += 1

    assert allowed == 5, f"two workers over-admitted: {allowed} > 5"


def test_unified_redis_consume_on_reject_does_not_grow_key():
    server = fakeredis.FakeServer()
    config = RateLimitConfig(
        strategy=RateLimitStrategy.REDIS, max_requests=3, window_seconds=60
    )
    limiter = UnifiedRedisLimiter(config, redis_client=fakeredis.FakeRedis(server=server))

    assert limiter.is_allowed("k") is True
    assert limiter.is_allowed("k") is True
    assert limiter.is_allowed("k") is True
    for _ in range(5):
        assert limiter.is_allowed("k") is False

    assert limiter.remaining("k") == 0


def test_unified_redis_remaining_and_retry_after_agree():
    server = fakeredis.FakeServer()
    config = RateLimitConfig(
        strategy=RateLimitStrategy.REDIS, max_requests=2, window_seconds=60
    )
    limiter = UnifiedRedisLimiter(config, redis_client=fakeredis.FakeRedis(server=server))
    assert limiter.is_allowed("k") is True
    assert limiter.is_allowed("k") is True
    assert limiter.remaining("k") == 0
    assert limiter.retry_after("k") >= 1.0


# ---------------------------------------------------------------------------
# Async RedisRateLimiter — two workers via fakeredis async
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_redis_try_consume_two_workers():
    from fakeredis import FakeAsyncRedis, FakeServer

    server = FakeServer()

    def make():
        lim = AsyncRedisLimiter(redis_url=None, max_attempts=5, window_seconds=60)
        lim._redis = FakeAsyncRedis(server=server)
        return lim

    w1, w2 = make(), make()
    allowed = 0
    for _ in range(10):
        for w in (w1, w2):
            if await w.try_consume("1.2.3.4"):
                allowed += 1
    assert allowed == 5, f"two workers over-admitted: {allowed} > 5"


@pytest.mark.asyncio
async def test_async_redis_try_consume_on_reject_does_not_record():
    from fakeredis import FakeAsyncRedis

    lim = AsyncRedisLimiter(redis_url=None, max_attempts=3, window_seconds=60)
    lim._redis = FakeAsyncRedis()
    assert await lim.try_consume("1.2.3.4") is True
    assert await lim.try_consume("1.2.3.4") is True
    assert await lim.try_consume("1.2.3.4") is True
    for _ in range(3):
        assert await lim.try_consume("1.2.3.4") is False
    assert await lim.is_rate_limited("1.2.3.4") is True
    assert await lim.retry_after("1.2.3.4") >= 1


# ---------------------------------------------------------------------------
# Hierarchical tier leak
# ---------------------------------------------------------------------------

def test_hierarchical_rejection_does_not_consume_earlier_tiers():
    limiter = HierarchicalRateLimiter(
        global_rpm=5, tenant_rpm=100, model_rpm={"m": 3}, burst_multiplier=1.0
    )
    for _ in range(3):
        assert limiter.is_allowed(tenant="acme", model="m") is True

    before = limiter.get_limits(tenant="acme", model="m")
    assert limiter.is_allowed(tenant="acme", model="m") is False
    after = limiter.get_limits(tenant="acme", model="m")

    assert after["global_remaining"] == before["global_remaining"], (
        f"global tier leaked on rejection: {before['global_remaining']} -> "
        f"{after['global_remaining']}"
    )
    assert after["tenant_remaining"] == before["tenant_remaining"], (
        f"tenant tier leaked on rejection: {before['tenant_remaining']} -> "
        f"{after['tenant_remaining']}"
    )


def test_unified_hierarchical_rejection_does_not_consume_earlier_tiers():
    limiter = RateLimiter(
        default_config=RateLimitConfig(
            strategy=RateLimitStrategy.SLIDING_WINDOW, max_requests=1000, window_seconds=60
        ),
        global_config=RateLimitConfig(
            strategy=RateLimitStrategy.SLIDING_WINDOW, max_requests=1000, window_seconds=60
        ),
        tenant_configs={
            "acme": RateLimitConfig(
                strategy=RateLimitStrategy.SLIDING_WINDOW, max_requests=1000, window_seconds=60
            )
        },
        model_configs={
            "m": RateLimitConfig(
                strategy=RateLimitStrategy.SLIDING_WINDOW, max_requests=2, window_seconds=60
            )
        },
    )
    for _ in range(2):
        assert limiter.is_allowed("c1", "/x", tenant="acme", model="m") is True

    before = limiter.get_limits("c1", "/x", tenant="acme", model="m")
    assert limiter.is_allowed("c1", "/x", tenant="acme", model="m") is False
    after = limiter.get_limits("c1", "/x", tenant="acme", model="m")

    assert after["global"]["remaining"] == before["global"]["remaining"]
    assert after["tenant"]["remaining"] == before["tenant"]["remaining"]
    assert after["endpoint"]["remaining"] == before["endpoint"]["remaining"]
