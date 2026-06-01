"""Server-side circuit breaker middleware for distributed LLM.

Protects downstream services (backends, external APIs) from cascade
failures by opening the circuit when error rates exceed thresholds.

Usage::

    from distllm.api.circuit_breaker_middleware import CircuitBreakerMiddleware
    app.add_middleware(CircuitBreakerMiddleware)
"""

from __future__ import annotations

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
    OPEN = "open"          # Rejecting requests
    HALF_OPEN = "half_open" # Testing if service recovered


class ServerCircuitBreaker:
    """Server-side circuit breaker for protecting downstream services.

    Opens when error rate exceeds threshold, rejects requests with 503,
    and automatically retries after a cooldown period.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 30.0,
        half_open_max_calls: int = 3,
        error_rate_threshold: float = 0.5,
        window_s: float = 60.0,
    ):
        self._threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_s
        self._half_open_max = half_open_max_calls
        self._error_rate_threshold = error_rate_threshold
        self._window = window_s

        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._recent_results: list[tuple[float, bool]] = []  # (timestamp, success)
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout elapsed
                if time.time() - self._last_failure_time >= self._recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info("Circuit breaker: OPEN → HALF_OPEN (recovery timeout elapsed)")
            return self._state

    def record_success(self) -> None:
        with self._lock:
            self._successes += 1
            self._recent_results.append((time.time(), True))
            self._prune_old()

            if self._state == CircuitState.HALF_OPEN:
                if self._successes >= self._half_open_max:
                    self._state = CircuitState.CLOSED
                    self._failures = 0
                    logger.info("Circuit breaker: HALF_OPEN → CLOSED (service recovered)")

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()
            self._recent_results.append((time.time(), False))
            self._prune_old()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning("Circuit breaker: HALF_OPEN → OPEN (recovery failed)")
            elif self._state == CircuitState.CLOSED:
                # Check if we should open
                if self._failures >= self._threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit breaker: CLOSED → OPEN "
                        f"(failures={self._failures}/{self._threshold})"
                    )

    def is_open(self) -> bool:
        """Check if circuit is open (should reject requests)."""
        return self.state == CircuitState.OPEN

    def _prune_old(self) -> None:
        cutoff = time.time() - self._window
        self._recent_results = [(t, s) for t, s in self._recent_results if t > cutoff]

    def get_error_rate(self) -> float:
        """Get current error rate in the window."""
        with self._lock:
            self._prune_old()
            if not self._recent_results:
                return 0.0
            failures = sum(1 for _, s in self._recent_results if not s)
            return failures / len(self._recent_results)

    def stats(self) -> dict:
        return {
            "state": self.state.value,
            "failures": self._failures,
            "successes": self._successes,
            "error_rate": round(self.get_error_rate(), 3),
            "threshold": self._threshold,
            "recovery_timeout_s": self._recovery_timeout,
        }


# Global circuit breaker instance
_breaker = ServerCircuitBreaker()


class CircuitBreakerMiddleware(BaseHTTPMiddleware):
    """Middleware that rejects requests when circuit breaker is open.

    Returns 503 with Retry-After header when the circuit is open.
    Passes through normally when closed or half-open.
    """

    SKIP_PATHS = {"/health", "/ready", "/live", "/metrics"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        if _breaker.is_open():
            retry_after = int(_breaker._recovery_timeout)
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
        except Exception as e:
            _breaker.record_failure()
            raise


def get_circuit_breaker() -> ServerCircuitBreaker:
    """Get the global circuit breaker instance."""
    return _breaker
