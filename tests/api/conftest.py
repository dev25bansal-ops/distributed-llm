"""Shared fixtures for API tests.

Clears the dedup cache and circuit breaker between tests to prevent
cross-test state pollution that causes spurious failures.
"""

import pytest


@pytest.fixture(autouse=True)
def _reset_middleware_state():
    """Clear request dedup cache and circuit breaker before each test.

    The dedup middleware caches responses based on request body fingerprint,
    and the circuit breaker tracks failure counts globally. Without resetting
    these, a failing test can poison all subsequent tests.
    """
    # Reset dedup cache
    from distllm.api.dedup import _cache
    _cache._cache.clear()
    _cache._results.clear()
    _cache._in_flight.clear()
    _cache._wait_events.clear()

    # Reset circuit breaker
    from distllm.api.circuit_breaker_middleware import _breaker, CircuitState
    _breaker._state = CircuitState.CLOSED
    _breaker._failures = 0
    _breaker._successes = 0
    _breaker._last_failure_time = 0.0
    _breaker._half_open_calls = 0
    _breaker._recent_results.clear()

    yield
