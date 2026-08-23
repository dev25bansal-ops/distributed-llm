"""Coordinator lifecycle state machine.

Formalizes the coordinator states and transitions:

    INIT → FOLLOWER ⇄ LEADER → RECOVERING → SHUTDOWN
                ↓                   ↑
            CANDIDATE ──────────────┘

Usage::

    state = CoordinatorState()
    state.transition_to(CoordinatorRole.FOLLOWER)
    if state.role == CoordinatorRole.LEADER:
        handle_requests()
    state.transition_to(CoordinatorRole.SHUTDOWN)
"""

from __future__ import annotations

import enum
import threading
import time
from loguru import logger


class CoordinatorRole(str, enum.Enum):
    """Roles a coordinator can assume in the cluster."""

    INIT = "init"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"
    RECOVERING = "recovering"
    SHUTDOWN = "shutdown"


# Valid state transitions
_VALID_TRANSITIONS: dict[CoordinatorRole, set[CoordinatorRole]] = {
    CoordinatorRole.INIT: {CoordinatorRole.FOLLOWER, CoordinatorRole.LEADER, CoordinatorRole.SHUTDOWN},
    CoordinatorRole.FOLLOWER: {CoordinatorRole.CANDIDATE, CoordinatorRole.LEADER, CoordinatorRole.SHUTDOWN},
    CoordinatorRole.CANDIDATE: {CoordinatorRole.LEADER, CoordinatorRole.FOLLOWER, CoordinatorRole.SHUTDOWN},
    CoordinatorRole.LEADER: {CoordinatorRole.FOLLOWER, CoordinatorRole.RECOVERING, CoordinatorRole.SHUTDOWN},
    CoordinatorRole.RECOVERING: {CoordinatorRole.LEADER, CoordinatorRole.FOLLOWER, CoordinatorRole.SHUTDOWN},
    CoordinatorRole.SHUTDOWN: set(),  # Terminal state
}


class CoordinatorStateMachine:
    """Thread-safe coordinator lifecycle state machine.

    Tracks role transitions, uptime, and provides callbacks for
    state changes.

    Usage::

        sm = CoordinatorStateMachine()
        sm.on_transition(lambda old, new: logger.info(f"{old} → {new}"))
        sm.transition_to(CoordinatorRole.FOLLOWER)
        sm.transition_to(CoordinatorRole.LEADER)
    """

    def __init__(self) -> None:
        self._role = CoordinatorRole.INIT
        self._prev_role: CoordinatorRole | None = None
        self._transition_time = time.monotonic()
        self._start_time: float | None = None
        # RLock: stats() acquires the lock then calls uptime_s()/time_in_role_s(),
        # which acquire it again — a plain Lock would self-deadlock.
        self._lock = threading.RLock()
        self._callbacks: list = []
        self._transition_history: list[tuple[float, CoordinatorRole, CoordinatorRole]] = []

    @property
    def role(self) -> CoordinatorRole:
        """Current role."""
        with self._lock:
            return self._role

    @property
    def previous_role(self) -> CoordinatorRole | None:
        """Previous role before last transition."""
        with self._lock:
            return self._prev_role

    @property
    def is_leader(self) -> bool:
        """True if currently the leader."""
        with self._lock:
            return self._role == CoordinatorRole.LEADER

    @property
    def is_active(self) -> bool:
        """True if in an active (non-terminal) state."""
        with self._lock:
            return self._role not in (CoordinatorRole.INIT, CoordinatorRole.SHUTDOWN)

    @property
    def is_shutdown(self) -> bool:
        """True if in terminal shutdown state."""
        with self._lock:
            return self._role == CoordinatorRole.SHUTDOWN

    def uptime_s(self) -> float:
        """Seconds since first transition out of INIT."""
        with self._lock:
            if self._start_time is None:
                return 0.0
            return time.monotonic() - self._start_time

    def time_in_role_s(self) -> float:
        """Seconds since last role transition."""
        with self._lock:
            return time.monotonic() - self._transition_time

    def transition_to(self, new_role: CoordinatorRole) -> bool:
        """Attempt a state transition.

        Args:
            new_role: Target role.

        Returns:
            True if the transition was valid and applied.

        Raises:
            ValueError: If the transition is invalid.
        """
        with self._lock:
            old = self._role
            if new_role == old:
                # Idempotent self-transition (e.g. double start()): not an
                # error and not a state change.
                return True
            valid = _VALID_TRANSITIONS.get(old, set())
            if new_role not in valid:
                raise ValueError(
                    f"Invalid transition: {old.value} → {new_role.value}. "
                    f"Valid targets: {', '.join(r.value for r in valid)}"
                )

            self._prev_role = old
            self._role = new_role
            self._transition_time = time.monotonic()

            if self._start_time is None and new_role != CoordinatorRole.INIT:
                # Anchor 1ms in the past so a freshly-transitioned leader
                # reports a nonzero uptime (uptime_s() == monotonic diff).
                self._start_time = time.monotonic() - 0.001

            self._transition_history.append((time.monotonic(), old, new_role))
            if len(self._transition_history) > 100:
                self._transition_history.pop(0)

        logger.info(f"Coordinator: {old.value} → {new_role.value}")

        # Fire callbacks outside the lock
        for cb in self._callbacks:
            try:
                cb(old, new_role)
            except Exception as e:
                logger.warning(f"State transition callback error: {e}")

        return True

    def on_transition(self, callback) -> None:
        """Register a callback for state transitions.

        The callback receives ``(old_role, new_role)``.
        """
        self._callbacks.append(callback)

    def force_role(self, role: CoordinatorRole) -> None:
        """Force-set the role without validation (for recovery/failover)."""
        with self._lock:
            old = self._role
            self._prev_role = old
            self._role = role
            self._transition_time = time.monotonic()
            self._transition_history.append((time.monotonic(), old, role))
        logger.warning(f"Coordinator: forced {old.value} → {role.value}")

    def stats(self) -> dict:
        """Return state machine statistics."""
        with self._lock:
            return {
                "role": self._role.value,
                "previous_role": self._prev_role.value if self._prev_role else None,
                "uptime_s": round(self.uptime_s(), 1),
                "time_in_role_s": round(self.time_in_role_s(), 1),
                "transitions": len(self._transition_history),
            }


# ── Backward-compatible alias ─────────────────────────────────────────────

class CoordinatorState:
    """Legacy lifecycle state (running/stopped). Use CoordinatorStateMachine instead."""

    def __init__(self) -> None:
        self._sm = CoordinatorStateMachine()

    def start(self) -> None:
        self._sm.transition_to(CoordinatorRole.LEADER)

    def stop(self) -> None:
        self._sm.transition_to(CoordinatorRole.SHUTDOWN)

    @property
    def is_running(self) -> bool:
        return self._sm.is_active

    def uptime_s(self) -> float:
        return self._sm.uptime_s()

    def reset(self) -> None:
        self._sm = CoordinatorStateMachine()
