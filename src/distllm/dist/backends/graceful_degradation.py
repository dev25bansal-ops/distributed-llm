"""Automatic backend fallback and circuit-breaker for graceful degradation.

Provides a three-tier fallback chain (primary -> secondary -> CPU) with
a per-backend circuit breaker that prevents cascading failures.  Tracks
degradation events and exposes recovery-check logic.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class CircuitBreakerState:
    """Tracks circuit-breaker state for a single backend."""

    failures: int = 0
    last_failure_time: float = 0.0
    state: str = "closed"  # "closed" | "open" | "half-open"
    opened_at: float = 0.0


@dataclass
class DegradationEvent:
    """Record of a single degradation event."""

    backend_id: str
    from_backend: str  # e.g. "primary" | "secondary"
    to_backend: str
    reason: str
    timestamp: float = 0.0
    duration_s: float = 0.0


@dataclass
class DegradationStats:
    """Aggregated degradation statistics."""

    total_degradations: int = 0
    total_recoveries: int = 0
    current_degraded: int = 0
    circuit_breaker_open: int = 0
    average_recovery_time_s: float = 0.0


# ---------------------------------------------------------------------------
# GracefulDegradationHandler
# ---------------------------------------------------------------------------


class GracefulDegradationHandler:
    """Automatic backend fallback with circuit breaker.

    Implements a staged degradation chain:

    1. Try the **primary** backend.
    2. On failure, try a **secondary** backend (lower quality but available).
    3. On secondary failure, try a **CPU fallback** backend.
    4. If all fail, return a **degraded response** instead of an error.

    A configurable circuit breaker (default: 5 failures / 30 s cooldown)
    protects each backend tier so that repeatedly failing backends are
    skipped until they have a chance to recover.

    Parameters
    ----------
    backends:
        Ordered fallback chain per backend group.  Structure:
        ``{"backend-a": ["backend-b", "backend-cpu"]}``
    fallback_providers:
        Optional mapping of backend_id -> fallback strategy for custom
        fallback resolution (e.g. model-based fallback).
    failure_threshold:
        Number of consecutive failures before the circuit breaker opens
        (default 5).
    cooldown_s:
        Seconds the circuit breaker stays open before transitioning to
        half-open (default 30).
    half_open_max_retries:
        Number of probes allowed in half-open state before the breaker
        either closes (on success) or re-opens (default 3).
    on_degradation:
        Called when a degradation event occurs:
        ``fn(event: DegradationEvent)``.
    on_recovery:
        Called when a backend recovers:
        ``fn(backend_id: str)``.
    """

    def __init__(
        self,
        backends: dict[str, list[str]] | None = None,
        fallback_providers: dict[str, Callable[[str], str | None]] | None = None,
        failure_threshold: int = 5,
        cooldown_s: float = 30.0,
        half_open_max_retries: int = 3,
        on_degradation: Callable[[DegradationEvent], None] | None = None,
        on_recovery: Callable[[str], None] | None = None,
    ) -> None:
        self._fallback_chain: dict[str, list[str]] = {}
        """backend_id -> ordered fallback list."""
        if backends:
            self._fallback_chain.update(backends)

        self._fallback_providers: dict[str, Callable[[str], str | None]] = {}
        if fallback_providers:
            self._fallback_providers.update(fallback_providers)

        self._failure_threshold = failure_threshold
        self._cooldown_s = cooldown_s
        self._half_open_max_retries = half_open_max_retries
        self._on_degradation = on_degradation
        self._on_recovery = on_recovery

        # Per-backend circuit breakers
        self._breakers: dict[str, CircuitBreakerState] = {}

        # Track which tier each backend is currently running at
        self._active_tier: dict[str, str] = {}
        """backend_id -> current active tier name (e.g. "primary")."""

        self._lock = threading.Lock()

        self._degradation_events: list[DegradationEvent] = []
        self._recovery_count: int = 0
        self._total_degradation_time_s: float = 0.0
        self._current_degradation_start: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Backend management
    # ------------------------------------------------------------------

    def register_backend(
        self,
        backend_id: str,
        fallback_chain: list[str] | None = None,
    ) -> None:
        """Register a backend for degradation handling.

        Args:
            backend_id: Primary backend identifier.
            fallback_chain: Ordered list of fallback backends to try
                when *backend_id* fails.  If omitted, uses an empty chain
                (no fallback).
        """
        with self._lock:
            if fallback_chain is not None:
                self._fallback_chain[backend_id] = list(fallback_chain)
            elif backend_id not in self._fallback_chain:
                self._fallback_chain[backend_id] = []
            if backend_id not in self._breakers:
                self._breakers[backend_id] = CircuitBreakerState()
            if backend_id not in self._active_tier:
                self._active_tier[backend_id] = "primary"

    def unregister_backend(self, backend_id: str) -> None:
        """Remove a backend from degradation handling."""
        with self._lock:
            self._fallback_chain.pop(backend_id, None)
            self._breakers.pop(backend_id, None)
            self._active_tier.pop(backend_id, None)

    def set_fallback_chain(self, backend_id: str, chain: list[str]) -> None:
        """Set the fallback chain for *backend_id*."""
        with self._lock:
            self._fallback_chain[backend_id] = list(chain)

    # ------------------------------------------------------------------
    # Degrade / recover
    # ------------------------------------------------------------------

    def degrade(self, backend_id: str) -> str | None:
        """Recommend the next fallback for *backend_id*.

        Returns the most viable fallback backend ID, considering circuit
        breaker state.  Returns ``None`` when the fallback chain is
        exhausted or all fallbacks have open breakers.

        The returned ID is one of:
          - A secondary backend (first non-breaker item in the chain)
          - A CPU fallback backend
          - ``None`` when all tiers are exhausted
        """
        with self._lock:
            chain = list(self._fallback_chain.get(backend_id, []))

            # Try each fallback in order
            for fallback_id in chain:
                if self._is_breaker_open(fallback_id):
                    continue  # skip backends with open breakers
                # Also check if the fallback itself is degraded
                cb = self._breakers.get(fallback_id)
                if cb and cb.state == "half-open" and cb.failures >= self._half_open_max_retries:
                    continue

                # Record the degradation event
                now = time.time()
                old_tier = self._active_tier.get(backend_id, "primary")
                self._active_tier[backend_id] = fallback_id
                self._current_degradation_start[backend_id] = now

                event = DegradationEvent(
                    backend_id=backend_id,
                    from_backend=old_tier,
                    to_backend=fallback_id,
                    reason=f"Primary backend '{backend_id}' failed, falling back to '{fallback_id}'",
                    timestamp=now,
                )
                self._degradation_events.append(event)
                self._total_degradation_time_s = 0.0  # tracked per-event

                logger.warning(
                    f"Degradation: {backend_id} -> {fallback_id} "
                    f"(failures={cb.failures if cb else 0})"
                )
                if self._on_degradation:
                    self._on_degradation(event)

                return fallback_id

            # All fallbacks exhausted
            logger.error(
                f"All fallback backends exhausted for '{backend_id}'"
            )
            return None

    def try_primary(self, backend_id: str) -> str | None:
        """Check if the primary backend is ready to accept traffic again.

        Returns ``"primary"`` if the backend can go back to primary duty,
        or ``None`` if the circuit breaker is still open.
        """
        if self._is_breaker_open(backend_id):
            return None
        return "primary"

    def recovery_check(self, backend_id: str) -> bool:
        """Check whether *backend_id* is ready for traffic after a failure.

        A backend is considered ready when:
          1. The circuit breaker is closed or half-open with available
             retries.
          2. It is not currently tracked as actively degraded (tier
             transition already happened).

        Returns ``True`` when the backend should be re-enabled for routing.
        """
        with self._lock:
            cb = self._breakers.get(backend_id)
            if cb is None:
                return True  # unknown backend — assume ready

            if cb.state == "closed":
                return True

            if cb.state == "half-open" and cb.failures < self._half_open_max_retries:
                return True

            # Check cooldown expiry for open breakers
            if cb.state == "open":
                elapsed = time.time() - cb.opened_at
                if elapsed >= self._cooldown_s:
                    # Transition to half-open
                    cb.state = "half-open"
                    cb.failures = 0
                    logger.info(
                        f"Circuit breaker '{backend_id}' transitioned to half-open"
                    )
                    return True

            return False

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    def record_failure(self, backend_id: str) -> None:
        """Record a failure for *backend_id* and update breaker state."""
        with self._lock:
            cb = self._breakers.setdefault(
                backend_id,
                CircuitBreakerState(),
            )
            cb.failures += 1
            cb.last_failure_time = time.time()

            if cb.state == "closed" and cb.failures >= self._failure_threshold:
                cb.state = "open"
                cb.opened_at = time.time()
                logger.warning(
                    f"Circuit breaker OPEN for '{backend_id}' "
                    f"({cb.failures} failures, {self._cooldown_s}s cooldown)"
                )

            elif cb.state == "half-open":
                if cb.failures >= self._half_open_max_retries:
                    cb.state = "open"
                    cb.opened_at = time.time()
                    logger.warning(
                        f"Circuit breaker re-OPEN for '{backend_id}' "
                        f"(half-open retries exhausted)"
                    )

    def record_success(self, backend_id: str) -> None:
        """Record a success, potentially closing the circuit breaker."""
        with self._lock:
            cb = self._breakers.get(backend_id)
            if cb is None:
                return

            if cb.state in ("open", "half-open"):
                old_state = cb.state
                cb.state = "closed"
                cb.failures = 0
                self._recovery_count += 1

                # Track recovery duration
                start = self._current_degradation_start.pop(backend_id, None)
                if start is not None:
                    duration = time.time() - start
                    self._total_degradation_time_s += duration

                self._active_tier[backend_id] = "primary"
                logger.info(
                    f"Circuit breaker CLOSED for '{backend_id}' "
                    f"(was {old_state})"
                )
                if self._on_recovery:
                    self._on_recovery(backend_id)

    def _is_breaker_open(self, backend_id: str) -> bool:
        """Check whether the circuit breaker for *backend_id* is open."""
        cb = self._breakers.get(backend_id)
        if cb is None:
            return False
        if cb.state == "closed":
            return False
        if cb.state == "open":
            elapsed = time.time() - cb.opened_at
            if elapsed >= self._cooldown_s:
                # Auto-transition to half-open
                cb.state = "half-open"
                cb.failures = 0
                logger.info(
                    f"Circuit breaker '{backend_id}' transitioned to half-open "
                    f"(cooldown expired)"
                )
                return False
            return True
        # half-open
        return cb.failures >= self._half_open_max_retries

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return degradation and circuit-breaker statistics."""
        with self._lock:
            open_count = sum(
                1 for cb in self._breakers.values() if cb.state == "open"
            )
            degraded = len(self._current_degradation_start)
            avg_recovery = (
                self._total_degradation_time_s / max(self._recovery_count, 1)
            )

            return {
                "total_degradations": len(self._degradation_events),
                "total_recoveries": self._recovery_count,
                "current_degraded": degraded,
                "circuit_breaker_open": open_count,
                "circuit_breaker_half_open": sum(
                    1 for cb in self._breakers.values() if cb.state == "half-open"
                ),
                "average_recovery_time_s": round(avg_recovery, 2),
                "failure_threshold": self._failure_threshold,
                "cooldown_s": self._cooldown_s,
                "events": [
                    {
                        "backend_id": e.backend_id,
                        "from": e.from_backend,
                        "to": e.to_backend,
                        "reason": e.reason,
                        "timestamp": e.timestamp,
                    }
                    for e in self._degradation_events[-20:]  # keep last 20
                ],
            }

    def get_breaker_state(self, backend_id: str) -> str | None:
        """Return the current circuit-breaker state for *backend_id*."""
        cb = self._breakers.get(backend_id)
        return cb.state if cb is not None else None

    def get_active_tier(self, backend_id: str) -> str:
        """Return the current active tier for *backend_id*."""
        return self._active_tier.get(backend_id, "primary")
