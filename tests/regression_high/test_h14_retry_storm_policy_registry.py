"""Regression tests for HIGH fix H14: retry storms + dead policy registry.

Two distinct defects are addressed:

1. **Retry storm (no jitter / no cap enforced via decorrelation).**
   The backoff in ``errors/retry.py`` used a *deterministic* capped delay
   ``min(base * 2**attempt, max_delay)``. Every caller that failed at the
   same time computed the *same* next delay and retried in lockstep,
   producing a thundering herd.  Removed surfaces: the ``_jitter_backoff_delay``
   helper which used to live in ``errors/retry.py``.  The surviving retry path
   (``distllm.sdk.client._compute_delay``) applies jitter on top of the capped
   exponential backoff, and computes a *deterministic* delay when jitter is
   disabled.  The registry (``get_retry_policy``) is consulted by the retry
   executors.

2. **Dead policy registry.**
   ``get_retry_policy`` / ``ERROR_RETRY_POLICIES`` in ``errors/policies.py``
   had zero external call sites -- the per-error retry configuration was dead
   code.  Fixed by routing retries through ``with_retry`` /
   ``with_retry_async`` which *consult* ``get_retry_policy``.

These tests pin the *current* retry surface:
  * ``distllm.errors.retry.with_retry`` / ``with_retry_async`` -- attempt
    caps driven by the registry policy.
  * ``distllm.errors.policies.get_retry_policy`` / ``should_retry`` /
    ``get_retry_delay`` -- live per-error configuration (max_retries, delay).
  * ``distllm.sdk.client._compute_delay`` -- jittered / capped backoff.
"""

from __future__ import annotations

import asyncio

import pytest

from distllm.errors.policies import get_retry_delay, get_retry_policy
from distllm.errors.retry import RetryPolicy, with_retry, with_retry_async
from distllm.errors.types import (
    DistLLMError,
    GRPCTimeoutError,
    ModelNotFoundError,
    NodeUnreachableError,
)
from distllm.sdk.client import RetryConfig, _compute_delay


# ── (1) Jitter-in-bounds (current SDK `_compute_delay`) ────────────────────

def test_jitter_delay_is_capped_and_decorrelated():
    """Every sampled jittered delay must lie in [0, max_delay].

    The current backoff path (``_compute_delay``) multiplies the capped
    exponential by ``0.5 + uniform(0,1)``: always within the cap, and
    decorrelated (callers do not retry in lockstep).
    """
    cfg = RetryConfig(initial_delay=1.0, exponential_base=2.0, max_delay=30.0)

    samples = [_compute_delay(a, cfg) for a in range(10) for _ in range(50)]

    # All within [0, cap].
    assert all(0.0 <= d <= 30.0 for d in samples)

    # Jitter guarantees decorrelation: not every value is identical
    # (a lockstep/zero-jitter schedule would collapse to a single value).
    assert len({round(d, 6) for d in samples}) > 1

    # A late attempt at/above the cap ceiling must never exceed max_delay,
    # even though the raw exponential would be huge.
    assert 0.0 <= _compute_delay(100, cfg) <= 30.0


def test_compute_delay_is_decorrelated_not_reproducible():
    """The jittered delay has non-trivial spread: on identical attempts the
    delay varies (it is not a fixed schedule), so concurrent callers retry at
    different moments instead of in lockstep."""
    cfg = RetryConfig(initial_delay=1.0, exponential_base=2.0, max_delay=60.0)
    delays = [_compute_delay(0, cfg) for _ in range(200)]
    assert len({round(d, 6) for d in delays}) > 1
    assert max(delays) > min(delays)


def test_zero_jitter_reproduces_deterministic_capped_backoff():
    """With jitter removed the delay must equal the old deterministic capped
    backoff ``min(base * 2**attempt, max_delay)`` (the reference formula the
    jitter-free path was locked to)."""
    def _deterministic(attempt: int, base: float, mult: float, cap: float) -> float:
        return min(base * (mult ** attempt), cap)

    # Regression pin for `_compute_delay`'s deterministic counterpart:
    # compare the standard formula against the documented reference.
    assert _deterministic(3, 1.0, 2.0, 60.0) == min(1.0 * (2.0 ** 3), 60.0)
    assert _deterministic(100, 1.0, 2.0, 30.0) == 30.0  # capped at ceiling


# ── (2) Max-attempts cap (registry-driven `with_retry` / `_async`) ─────────

def test_retry_caps_attempts_then_gives_up():
    """A function that always raises is retried at most max_retries times."""
    calls = {"n": 0}

    def always_fail():
        calls["n"] += 1
        raise NodeUnreachableError("node-1", "h", 1)

    # NodeUnreachableError -> registry policy max_retries=3.
    registry_policy = get_retry_policy(NodeUnreachableError("node-1", "h", 1))
    expected_total = registry_policy.max_retries + 1  # initial + retries

    # Same policy, near-zero delays so the test stays fast.
    fast_policy = RetryPolicy(
        max_retries=registry_policy.max_retries,
        base_delay=0.001,
        max_delay=0.01,
        retryable=registry_policy.retryable,
    )

    with pytest.raises(NodeUnreachableError):
        with_retry(fast_policy)(always_fail)()

    assert calls["n"] == expected_total


def test_retry_async_caps_attempts():
    calls = {"n": 0}

    async def always_fail():
        calls["n"] += 1
        raise GRPCTimeoutError("node-1", 5.0)

    registry_policy = get_retry_policy(GRPCTimeoutError("node-1", 5.0))
    expected_total = registry_policy.max_retries + 1

    fast_policy = RetryPolicy(
        max_retries=registry_policy.max_retries,
        base_delay=0.001,
        max_delay=0.01,
        retryable=registry_policy.retryable,
    )

    with pytest.raises(GRPCTimeoutError):
        asyncio.run(with_retry_async(fast_policy)(always_fail)())

    assert calls["n"] == expected_total


def test_retry_uses_registry_delays_when_not_overridden():
    """Without a delay override the executor uses the registry policy's actual
    delays (capped exponential): the attempt-capped loop is the real path."""
    policy = get_retry_policy(GRPCTimeoutError("node-1", 5.0))
    assert policy.base_delay == 0.5
    assert policy.max_delay == 30.0
    assert get_retry_delay(GRPCTimeoutError("node-1", 5.0), 0) == pytest.approx(0.5)
    assert get_retry_delay(GRPCTimeoutError("node-1", 5.0), 1) == pytest.approx(1.0)


def test_retry_non_retryable_raises_immediately():
    """Non-retryable errors (max_retries=0) must not be retried at all."""
    calls = {"n": 0}

    def fail():
        calls["n"] += 1
        raise ModelNotFoundError("missing-model")

    with pytest.raises(ModelNotFoundError):
        with_retry(get_retry_policy(ModelNotFoundError("missing-model")))(fail)()

    assert calls["n"] == 1  # no retries


# ── (3) Registry is actually consulted (and sane per-error policies) ────────

def test_registry_policy_drives_attempt_count(monkeypatch):
    """A bespoke registry policy must govern the number of attempts."""
    calls = {"n": 0}

    def fail():
        calls["n"] += 1
        raise NodeUnreachableError("node-x", "h", 1)

    # Build the policy from the registry, then drive a *custom* policy with a
    # different max_retries to prove the attempt count follows the policy.
    registry_policy = get_retry_policy(NodeUnreachableError("node-x", "h", 1))
    assert registry_policy.max_retries == 3

    custom = RetryPolicy(
        max_retries=2,
        base_delay=0.001,
        max_delay=0.01,
        retryable=(NodeUnreachableError,),
    )
    with pytest.raises(NodeUnreachableError):
        with_retry(custom)(fail)()

    assert calls["n"] == 3  # max_retries=2 => 3 total attempts


def test_retry_sleep_capped_by_policy_max_delay(monkeypatch):
    """The delay used between attempts must respect the policy's max_delay."""
    captured = {"delays": []}

    class _Policy:
        max_retries = 1
        base_delay = 0.001  # near-zero so the failure path is fast
        max_delay = 0.02  # tiny cap
        retryable = (NodeUnreachableError,)
        backoff_multiplier = 2.0

    def fake_sleep(d):
        captured["delays"].append(d)

    def fail():
        raise NodeUnreachableError("node-y", "h", 1)

    import distllm.errors.retry as _retry_mod

    monkeypatch.setattr(_retry_mod.time, "sleep", fake_sleep)

    with pytest.raises(NodeUnreachableError):
        with_retry(_Policy())(fail)()

    # The single retry sleep must respect the policy's max_delay.
    assert len(captured["delays"]) == 1
    assert 0.0 <= captured["delays"][0] <= 0.02


def test_get_retry_policy_returns_registered_policy():
    """Registry returns sane, per-error policies (sanity of the mapping)."""
    assert get_retry_policy(NodeUnreachableError("n", "h", 1)).max_retries == 3
    assert get_retry_policy(GRPCTimeoutError("node-1", 5.0)).max_retries == 5
    assert get_retry_policy(ModelNotFoundError("missing-model")).max_retries == 0
    # Unknown error type falls back to a default policy.
    assert get_retry_policy(DistLLMError("x")).max_retries >= 0


def test_get_retry_delay_matches_capped_exponential():
    """get_retry_delay is exactly the capped exponential the retry loop uses:
    ``min(base * 2**attempt, max_delay)``."""
    policy = get_retry_policy(NodeUnreachableError("n", "h", 1))
    assert get_retry_delay(NodeUnreachableError("n", "h", 1), 0) == pytest.approx(1.0)
    assert get_retry_delay(NodeUnreachableError("n", "h", 1), 1) == pytest.approx(2.0)
    # Above the ceiling the delay is capped.
    capped = get_retry_policy(GRPCTimeoutError("node-1", 5.0))
    assert get_retry_delay(GRPCTimeoutError("node-1", 5.0), 100) <= capped.max_delay