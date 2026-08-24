"""Server-side circuit breaker middleware for distributed LLM.

Protects downstream services (backends, external APIs) from cascade
failures by opening the circuit when the backend looks unhealthy, rejecting
requests fast with 503, and automatically probing for recovery.

State machine::

                  failure_threshold exceeded (or windowed
                  error rate >= error_rate_threshold)
        ┌────────┐ ─────────────────────────────────────► ┌────────┐
        │ CLOSED │                                        │  OPEN  │
        └────────◄─┐                                        └───┬────┘
           ▲       │ any success resets     recovery_timeout_s  │
           │       │ the consecutive failure elapsed (checked    │
           │  ┌────┴┐ streak              lazily)             ▼
           │  │HALF ├────────────────────────────────► cooldown
           └──┤OPEN │   any probe failure re-opens
              └─────┘
        HALF_OPEN: up to ``half_open_max_calls`` concurrent probes;
        ``success_threshold`` clean probes in a row close the circuit.

Usage::

    from distllm.api.circuit_breaker_middleware import CircuitBreakerMiddleware
    app.add_middleware(CircuitBreakerMiddleware)
"""

from __future__ import annotations

import math
import threading
import time
from enum import Enum
from typing import Callable

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class CircuitState(str, Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Rejecting requests (fast-fail 503)
    HALF_OPEN = "half_open"  # Testing if service recovered


class ServerCircuitBreaker:
    """Server-side circuit breaker for protecting downstream services.

    CLOSED
        Requests are admitted freely. Consecutive failures are counted and
        any success resets the streak to zero, so isolated historical
        failures can never accumulate into a trip. The circuit also opens
        if the sliding-window error rate reaches ``error_rate_threshold``
        once at least ``min_window_samples`` results are inside the window
        (catches sustained ~50% brownouts that never form a long streak).

    OPEN
        All requests are rejected until ``recovery_timeout_s`` seconds have
        elapsed since the last failure. The transition to HALF_OPEN happens
        lazily on the next observation.

    HALF_OPEN
        At most ``half_open_max_calls`` concurrent probe requests are
        admitted (excess gets 503). ``success_threshold`` consecutive clean
        probes close the circuit; any probe failure immediately re-opens it
        with a fresh cooldown.

    All mutating operations are guarded by a lock and safe to call
    concurrently from multiple threads/tasks.

    Args:
        failure_threshold: Consecutive failures in CLOSED before opening.
        recovery_timeout_s: Cooldown in OPEN before probing (HALF_OPEN).
        half_open_max_calls: Max concurrent probe requests in HALF_OPEN.
        error_rate_threshold: Windowed failure ratio (0.0-1.0) that opens
            the circuit once the window holds enough samples.
        window_s: Sliding-window length for the error-rate rule.
        success_threshold: Clean probes needed in HALF_OPEN to close.
            Defaults to ``half_open_max_calls``.
        min_window_samples: Minimum results inside the window before the
            error-rate rule may trip (protects low-volume traffic from
            being tripped by a couple of unlucky calls).
        time_fn: Clock returning seconds; injectable for tests. Defaults
            to ``time.time``.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 30.0,
        half_open_max_calls: int = 3,
        error_rate_threshold: float = 0.5,
        window_s: float = 60.0,
        success_threshold: int | None = None,
        min_window_samples: int = 10,
        time_fn: Callable[[], float] = time.time,
    ):
        self._threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_s
        self._half_open_max = half_open_max_calls
        self._error_rate_threshold = error_rate_threshold
        self._window = window_s
        self._success_threshold = (
            success_threshold if success_threshold is not None else half_open_max_calls
        )
        self._min_window_samples = min_window_samples
        self._time_fn = time_fn

        self._state = CircuitState.CLOSED
        self._failures = 0            # consecutive failures (CLOSED only)
        self._successes = 0           # lifetime successful calls (metric)
        self._last_failure_time = 0.0
        self._half_open_calls = 0     # probe slots currently in flight
        self._half_open_successes = 0  # clean probes since entering HALF_OPEN
        self._recent_results: list[tuple[float, bool]] = []  # (timestamp, success)
        self._total_trips = 0         # times the circuit has opened
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """Current circuit state, lazily applying the OPEN cooldown."""
        with self._lock:
            return self._check_recovery_locked()

    def is_open(self) -> bool:
        """Check if circuit is open (should reject requests)."""
        return self.state == CircuitState.OPEN

    def allow_request(self) -> bool:
        """Admission gate for incoming requests.

        CLOSED: always admitted. OPEN: rejected until the cooldown elapses.
        HALF_OPEN: admitted only while free probe slots remain (at most
        ``half_open_max_calls`` concurrent probes); admitted probes occupy
        a slot until their result is recorded.
        """
        with self._lock:
            self._check_recovery_locked()
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self._half_open_max:
                    self._half_open_calls += 1
                    return True
                return False
            return False  # OPEN

    def retry_after_seconds(self) -> int:
        """Seconds (>= 1) until the circuit may accept probes again."""
        with self._lock:
            remaining = self._recovery_timeout - (self._time_fn() - self._last_failure_time)
            return max(1, math.ceil(remaining))

    # ------------------------------------------------------------------
    # Result recording
    # ------------------------------------------------------------------

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            now = self._time_fn()
            self._successes += 1
            self._recent_results.append((now, True))
            self._prune_old(now)

            if self._state == CircuitState.HALF_OPEN:
                self._release_probe_slot()
                self._half_open_successes += 1
                if self._half_open_successes >= self._success_threshold:
                    self._transition_to_locked(
                        CircuitState.CLOSED, reason="service recovered"
                    )
            elif self._state == CircuitState.CLOSED:
                # A healthy call breaks the failure streak — failures must
                # never accumulate across unrelated good periods.
                self._failures = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            now = self._time_fn()
            self._failures += 1
            self._last_failure_time = now
            self._recent_results.append((now, False))
            self._prune_old(now)

            if self._state == CircuitState.HALF_OPEN:
                self._release_probe_slot()
                self._transition_to_locked(CircuitState.OPEN, reason="recovery failed")
            elif self._state == CircuitState.CLOSED:
                if self._failures >= self._threshold or self._should_trip_on_rate():
                    self._transition_to_locked(CircuitState.OPEN)

    def reset(self) -> None:
        """Manually reset the circuit breaker to a pristine CLOSED state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._last_failure_time = 0.0
            self._half_open_calls = 0
            self._half_open_successes = 0
            self._recent_results.clear()
            self._total_trips = 0

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_error_rate(self) -> float:
        """Current failure ratio among results inside the window."""
        with self._lock:
            return self._error_rate_locked()

    def stats(self) -> dict:
        with self._lock:
            state = self._check_recovery_locked()
            return {
                "state": state.value,
                "failures": self._failures,
                "successes": self._successes,
                "error_rate": round(self._error_rate_locked(), 3),
                "threshold": self._threshold,
                "recovery_timeout_s": self._recovery_timeout,
                "half_open_max_calls": self._half_open_max,
                "success_threshold": self._success_threshold,
                "half_open_calls": self._half_open_calls,
                "half_open_successes": self._half_open_successes,
                "total_trips": self._total_trips,
                "window_s": self._window,
                "error_rate_threshold": self._error_rate_threshold,
                "min_window_samples": self._min_window_samples,
            }

    # ------------------------------------------------------------------
    # Internals — caller must hold self._lock
    # ------------------------------------------------------------------

    def _check_recovery_locked(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if self._time_fn() - self._last_failure_time >= self._recovery_timeout:
                self._transition_to_locked(
                    CircuitState.HALF_OPEN, reason="recovery timeout elapsed"
                )
        return self._state

    def _transition_to_locked(self, new_state: CircuitState, reason: str = "") -> None:
        previous = self._state
        self._state = new_state
        if new_state == CircuitState.OPEN:
            self._total_trips += 1
            logger.warning(
                f"Circuit breaker: {previous.value} → OPEN ({reason or 'threshold reached'}, "
                f"failures={self._failures}/{self._threshold})"
            )
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._half_open_successes = 0
            logger.info(f"Circuit breaker: OPEN → HALF_OPEN ({reason})")
        elif new_state == CircuitState.CLOSED:
            self._failures = 0
            self._half_open_calls = 0
            self._half_open_successes = 0
            logger.info(f"Circuit breaker: {previous.value} → CLOSED ({reason})")

    def _release_probe_slot(self) -> None:
        # Called when an admitted HALF_OPEN probe completes. Guard against
        # going negative if the slot counter was reset by a transition in
        # between (e.g. another probe re-opened the circuit).
        self._half_open_calls = max(0, self._half_open_calls - 1)

    def _should_trip_on_rate(self) -> bool:
        n = len(self._recent_results)
        if n < max(1, self._min_window_samples):
            return False
        failures = sum(1 for _, ok in self._recent_results if not ok)
        return (failures / n) >= self._error_rate_threshold

    def _error_rate_locked(self) -> float:
        self._prune_old(self._time_fn())
        if not self._recent_results:
            return 0.0
        failures = sum(1 for _, ok in self._recent_results if not ok)
        return failures / len(self._recent_results)

    def _prune_old(self, now: float | None = None) -> None:
        if now is None:
            now = self._time_fn()
        cutoff = now - self._window
        self._recent_results = [(t, s) for t, s in self._recent_results if t > cutoff]


# Global circuit breaker instance
_breaker = ServerCircuitBreaker()


class CircuitBreakerMiddleware(BaseHTTPMiddleware):
    """Middleware that rejects requests when circuit breaker is open.

    Returns 503 with Retry-After header when the circuit is open (or when
    HALF_OPEN probe capacity is exhausted). Passes through normally when
    closed; health/metrics paths always bypass the breaker.
    """

    SKIP_PATHS = {"/health", "/ready", "/live", "/metrics"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        if not _breaker.allow_request():
            retry_after = _breaker.retry_after_seconds()
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "Service temporarily unavailable (circuit breaker open)",
                        "type": "circuit_breaker_open",
                        "retry_after": retry_after,
                    }
                },
                headers={"Retry-After": str(retry_after)},
            )

        try:
            response = await call_next(request)
            if response.status_code < 500:
                _breaker.record_success()
            else:
                _breaker.record_failure()
            return response
        except Exception:
            _breaker.record_failure()
            raise


def get_circuit_breaker() -> ServerCircuitBreaker:
    """Get the global circuit breaker instance."""
    return _breaker
