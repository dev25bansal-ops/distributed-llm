"""Dynamic speculation length controller.

Adjusts the number of speculative candidates based on recent acceptance
rates using a proportional controller.  Easy inputs get more candidates
(higher throughput), hard inputs get fewer candidates (less wasted work).

Usage::

    controller = DynamicSpeculationController(
        initial_candidates=5,
        min_candidates=1,
        max_candidates=10,
        target_acceptance_rate=0.7,
    )

    # After each verification step:
    controller.update(accepted_count, num_draft)
    next_num_candidates = controller.current
"""

from __future__ import annotations

from collections import deque


class DynamicSpeculationController:
    """Proportional controller for speculation length.

    Maintains a sliding window of recent acceptance rates and adjusts
    the candidate count proportionally::

        acceptance_rate = accepted / proposed  (sliding window average)
        error = target_rate - acceptance_rate
        adjustment = round(Kp * error * current)
        new_candidates = clamp(current + adjustment, min, max)

    Args:
        initial_candidates: Starting number of draft candidates.
        min_candidates: Minimum candidates (never go below this).
        max_candidates: Maximum candidates (never exceed this).
        target_acceptance_rate: Desired acceptance rate (0.0-1.0).
            Higher = more aggressive (more candidates on easy inputs).
        window_size: Number of recent steps to average over.
        Kp: Proportional gain.  Higher = faster adaptation but more
            oscillation.  Typical range: 0.1-0.5.
        adaptation_delay: Number of steps to wait before adapting
            (lets the window warm up).
    """

    def __init__(
        self,
        initial_candidates: int = 5,
        min_candidates: int = 1,
        max_candidates: int = 10,
        target_acceptance_rate: float = 0.7,
        window_size: int = 10,
        Kp: float = 0.3,
        adaptation_delay: int = 5,
    ):
        self._current = initial_candidates
        self._min = min_candidates
        self._max = max_candidates
        self._target = target_acceptance_rate
        self._Kp = Kp
        self._delay = adaptation_delay

        # Sliding window of (accepted, proposed) pairs
        self._window: deque[tuple[int, int]] = deque(maxlen=window_size)
        self._steps = 0

    @property
    def current(self) -> int:
        """Current recommended number of candidates."""
        return self._current

    def update(self, accepted: int, proposed: int) -> int:
        """Update the sliding window and adjust candidates.

        Args:
            accepted: Number of accepted draft tokens in this step.
            proposed: Number of proposed draft tokens in this step.

        Returns:
            The updated candidate count (same as ``self.current``).
        """
        self._steps += 1
        self._window.append((accepted, proposed))

        # Don't adapt until the window has enough data
        if self._steps < self._delay:
            return self._current

        # Compute sliding window acceptance rate
        total_accepted = sum(a for a, _ in self._window)
        total_proposed = sum(p for _, p in self._window)
        rate = total_accepted / max(total_proposed, 1)

        # Proportional control with minimum step.
        # When acceptance is HIGH (rate > target), the input is easy —
        # increase candidates for more speedup.
        # When acceptance is LOW (rate < target), the input is hard —
        # decrease candidates to avoid wasted work.
        error = rate - self._target
        adjustment = round(self._Kp * error * self._current)
        # Ensure at least 1 unit of adjustment when error is significant
        if abs(error) > 0.1 and adjustment == 0:
            adjustment = 1 if error > 0 else -1

        # Apply adjustment with bounds
        self._current = max(self._min, min(self._max, self._current + adjustment))
        return self._current

    def reset(self, candidates: int | None = None) -> None:
        """Reset the controller to its initial state.

        Args:
            candidates: Optional new initial value (default: keeps current).
        """
        self._window.clear()
        self._steps = 0
        if candidates is not None:
            self._current = candidates

    @property
    def acceptance_rate(self) -> float:
        """Current sliding-window acceptance rate (0.0-1.0)."""
        if not self._window:
            return 0.0
        total_accepted = sum(a for a, _ in self._window)
        total_proposed = sum(p for _, p in self._window)
        return total_accepted / max(total_proposed, 1)
