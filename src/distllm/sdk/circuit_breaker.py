"""Circuit breaker pattern for SDK client.

Implements the circuit breaker state machine (closed -> open -> half-open -> closed)
to prevent cascading failures when the backend is unhealthy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Failing fast, rejecting requests
    HALF_OPEN = "half_open"    # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for the circuit breaker."""
    failure_threshold: int = 5          # Consecutive failures before opening
    recovery_timeout: float = 30.0      # Seconds before transitioning to half-open
    success_threshold: int = 3          # Successes in half-open to close
    half_open_max_calls: int = 3        # Max test calls allowed in half-open


class CircuitBreakerError(Exception):
    """Raised when a request is rejected because the circuit is open."""

    def __init__(self, message: str = "Circuit breaker is open", state: CircuitState = CircuitState.OPEN):
        super().__init__(message)
        self.state = state


class CircuitBreaker:
    """Circuit breaker state machine.

    Usage:
        cb = CircuitBreaker()
        if not cb.can_execute():
            raise CircuitBreakerError()
        try:
            result = call_backend()
            cb.record_success()
        except Exception:
            cb.record_failure()
            raise
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self._config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        self._last_state_change = time.monotonic()

    @property
    def state(self) -> CircuitState:
        """Current circuit state, with automatic transition check for OPEN -> HALF_OPEN."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self._config.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN)
        return self._state

    def can_execute(self) -> bool:
        """Returns True if the call should proceed, False if circuit is open."""
        state = self.state  # Triggers state transition check
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            if self._half_open_calls < self._config.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False
        return False  # OPEN

    def record_success(self) -> None:
        """Record a successful call."""
        self._total_successes += 1
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self._config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0  # Reset consecutive failure count

    def record_failure(self) -> None:
        """Record a failed call."""
        self._total_failures += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open immediately re-opens
            self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self._config.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        self._transition_to(CircuitState.CLOSED)
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._total_failures = 0
        self._total_successes = 0

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        self._state = new_state
        self._last_state_change = time.monotonic()
        if new_state == CircuitState.HALF_OPEN:
            self._success_count = 0
            self._half_open_calls = 0
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0

    def get_metrics(self) -> dict:
        """Return current metrics for monitoring."""
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
            "half_open_calls": self._half_open_calls,
            "last_state_change": self._last_state_change,
        }
