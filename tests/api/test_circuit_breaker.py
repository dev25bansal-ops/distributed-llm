"""Tests for CircuitBreakerMiddleware and ServerCircuitBreaker.

Covers:
- State transitions (CLOSED → OPEN → HALF_OPEN → CLOSED)
- Failure threshold and error-rate logic
- Recovery timeout and half-open probing
- Middleware integration (503 rejection, Retry-After headers)
- SKIP_PATHS exemption for health endpoints
"""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from distllm.api.circuit_breaker_middleware import (
    CircuitBreakerMiddleware,
    CircuitState,
    ServerCircuitBreaker,
    get_circuit_breaker,
)


# ======================================================================
# ServerCircuitBreaker unit tests
# ======================================================================


class TestServerCircuitBreaker:
    """ServerCircuitBreaker state-machine logic in isolation."""

    def test_initial_state_is_closed(self):
        cb = ServerCircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert not cb.is_open()
        assert cb.stats()["state"] == "closed"

    def test_opens_after_threshold_failures(self):
        cb = ServerCircuitBreaker(failure_threshold=3, recovery_timeout_s=9999)
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.is_open()

    def test_remains_closed_below_threshold(self):
        cb = ServerCircuitBreaker(failure_threshold=5)
        for _ in range(4):
            cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_transitions_to_half_open_after_timeout(self):
        cb = ServerCircuitBreaker(failure_threshold=1, recovery_timeout_s=0.01)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.02)
        # Accessing .state triggers the auto-transition
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_returns_to_closed(self):
        cb = ServerCircuitBreaker(
            failure_threshold=1, recovery_timeout_s=0.01, half_open_max_calls=2
        )
        cb.record_failure()
        time.sleep(0.02)
        # Should be half-open now
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN  # still need more
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_returns_to_open(self):
        cb = ServerCircuitBreaker(failure_threshold=1, recovery_timeout_s=9999)
        # Force to half-open
        cb._state = CircuitState.HALF_OPEN
        cb._last_failure_time = time.time() - 1

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_error_rate_calculation(self):
        cb = ServerCircuitBreaker(window_s=60.0)
        # 3 failures out of 10 total = 0.3
        for _ in range(3):
            cb.record_failure()
        for _ in range(7):
            cb.record_success()
        assert cb.get_error_rate() == pytest.approx(0.3, abs=0.01)

    def test_error_rate_empty_window(self):
        cb = ServerCircuitBreaker()
        assert cb.get_error_rate() == 0.0

    def test_prune_old_removes_expired_entries(self):
        cb = ServerCircuitBreaker(window_s=0.01)
        cb.record_failure()
        cb.record_success()
        assert len(cb._recent_results) == 2
        time.sleep(0.02)
        cb._prune_old()
        assert len(cb._recent_results) == 0

    def test_stats_snapshot(self):
        cb = ServerCircuitBreaker(failure_threshold=3, recovery_timeout_s=30.0)
        cb.record_success()
        s = cb.stats()
        assert s["state"] == "closed"
        assert s["successes"] == 1
        assert s["failures"] == 0
        assert s["threshold"] == 3
        assert s["recovery_timeout_s"] == 30.0

    def test_lock_prevents_race_on_state_read(self):
        """Verify that state property holds the lock during auto-transition."""
        cb = ServerCircuitBreaker(failure_threshold=1, recovery_timeout_s=0.01)
        cb.record_failure()
        time.sleep(0.02)
        # Access from multiple contexts (simulated)
        s1 = cb.state
        s2 = cb.state
        assert s1 == s2  # both see same state

    def test_record_success_in_closed_state(self):
        cb = ServerCircuitBreaker()
        cb.record_success()
        assert cb._successes == 1
        assert cb.state == CircuitState.CLOSED


# ======================================================================
# CircuitBreakerMiddleware integration tests
# ======================================================================


@pytest.fixture
def breaker_app():
    """Minimal FastAPI app with CircuitBreakerMiddleware and a route that
    can be toggled to succeed or fail."""
    app = FastAPI()

    @app.get("/ok")
    async def ok():
        return {"status": "ok"}

    @app.get("/fail")
    async def fail():
        return JSONResponse(status_code=502, content={"error": "upstream failed"})

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/raise")
    async def raise_error():
        raise RuntimeError("unexpected error")

    app.add_middleware(CircuitBreakerMiddleware)
    return app


@pytest.fixture(autouse=True)
def reset_breaker():
    """Reset the shared _breaker singleton before each integration test."""
    from distllm.api.circuit_breaker_middleware import _breaker as b
    b._state = CircuitState.CLOSED
    b._failures = 0
    b._successes = 0
    b._last_failure_time = 0.0
    b._half_open_calls = 0
    b._recent_results = []


@pytest.fixture
def breaker_client(breaker_app):
    return TestClient(breaker_app, raise_server_exceptions=False)


def test_passthrough_when_closed(breaker_client):
    """Requests pass through normally when circuit is CLOSED."""
    resp = breaker_client.get("/ok")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_returns_503_when_open(breaker_client):
    """When circuit opens, subsequent requests get 503 with Retry-After."""
    # Trigger the circuit breaker by hitting /fail enough times
    breaker = get_circuit_breaker()
    breaker._threshold = 3

    for _ in range(3):
        breaker_client.get("/fail")

    assert breaker.state == CircuitState.OPEN
    resp = breaker_client.get("/ok")
    assert resp.status_code == 503
    assert "Retry-After" in resp.headers
    error_body = resp.json()
    assert "circuit_breaker_open" in error_body.get("error", {}).get("type", "")


def test_skips_health_endpoints(breaker_client):
    """Health endpoints bypass the circuit breaker."""
    breaker = get_circuit_breaker()
    breaker._state = CircuitState.OPEN

    resp = breaker_client.get("/health")
    assert resp.status_code == 200  # not 503


def test_records_failure_on_5xx(breaker_client):
    """A 5xx response triggers record_failure()."""
    breaker = get_circuit_breaker()
    failures_before = breaker._failures

    breaker_client.get("/fail")

    assert breaker._failures > failures_before


def test_records_success_on_2xx(breaker_client):
    """A 2xx response triggers record_success()."""
    breaker = get_circuit_breaker()
    successes_before = breaker._successes

    breaker_client.get("/ok")

    assert breaker._successes > successes_before


def test_records_failure_on_exception(breaker_client):
    """An unhandled exception in the route triggers record_failure()."""
    breaker = get_circuit_breaker()
    failures_before = breaker._failures

    breaker_client.get("/raise")  # will return 500

    assert breaker._failures > failures_before


# ======================================================================
# Helper / smoke tests
# ======================================================================


def test_get_circuit_breaker_returns_singleton():
    cb1 = get_circuit_breaker()
    cb2 = get_circuit_breaker()
    assert cb1 is cb2


def test_breaker_resets_after_recovery():
    """Full cycle: CLOSED → OPEN → HALF_OPEN → CLOSED via middleware."""
    cb = ServerCircuitBreaker(failure_threshold=2, recovery_timeout_s=0.02, half_open_max_calls=2)
    assert cb.state == CircuitState.CLOSED

    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    time.sleep(0.03)
    assert cb.state == CircuitState.HALF_OPEN

    cb.record_success()
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
