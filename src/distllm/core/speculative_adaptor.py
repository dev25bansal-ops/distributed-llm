"""Speculative adaptor for adaptive num_assistant_tokens.

Uses a PID controller to dynamically adjust the number of assistant
tokens based on acceptance rate, targeting an optimal rate of 0.6.
"""

from __future__ import annotations

from loguru import logger


class SpeculativeAdaptor:
    """Adapts num_assistant_tokens based on acceptance rate feedback.

    PID controller with:
    - Target acceptance rate: 0.6 (sweet spot for speedup)
    - High acceptance (>0.7) -> increase tokens
    - Low acceptance (<0.5) -> decrease tokens
    - Very low (<0.15) -> disable speculative decoding

    Parameters:
        base_tokens: Starting number of assistant tokens.
        min_tokens: Minimum tokens (1 = always try at least 1).
        max_tokens: Maximum tokens allowed.
        target_rate: Target acceptance rate for optimal speedup.
        kp: Proportional gain for the PID controller.
        ki: Integral gain (prevents steady-state error).
        kd: Derivative gain (dampens oscillations).
    """

    def __init__(
        self,
        base_tokens: int = 5,
        min_tokens: int = 1,
        max_tokens: int = 10,
        target_rate: float = 0.6,
        kp: float = 2.0,
        ki: float = 0.1,
        kd: float = 0.5,
    ) -> None:
        self.base_tokens = base_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.target_rate = target_rate
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self._current_tokens = base_tokens
        self._integral = 0.0
        self._prev_error = 0.0
        self._disabled = False

    def adapt(self, current_acceptance_rate: float) -> int:
        """Adjust num_assistant_tokens based on current acceptance rate.

        Args:
            current_acceptance_rate: Latest acceptance rate (0.0-1.0).

        Returns:
            New num_assistant_tokens value.
        """
        # Auto-disable if acceptance rate too low
        if current_acceptance_rate < 0.15:
            if not self._disabled:
                logger.warning(
                    f"Speculative decoding disabled: acceptance rate {current_acceptance_rate:.2f} < 0.15"
                )
                self._disabled = True
            return 0

        self._disabled = False

        # PID controller
        error = current_acceptance_rate - self.target_rate
        self._integral += error
        derivative = error - self._prev_error
        self._prev_error = error

        adjustment = (
            self.kp * error
            + self.ki * self._integral
            + self.kd * derivative
        )

        # Scale adjustment to token count (round to nearest int)
        delta = int(round(adjustment))
        new_tokens = self._current_tokens + delta
        new_tokens = max(self.min_tokens, min(self.max_tokens, new_tokens))

        if new_tokens != self._current_tokens:
            logger.debug(
                f"Speculative adaptor: {self._current_tokens} -> {new_tokens} tokens "
                f"(acceptance={current_acceptance_rate:.2f}, error={error:.2f})"
            )

        self._current_tokens = new_tokens
        return new_tokens

    def reset(self) -> None:
        """Reset adaptor state to initial values."""
        self._current_tokens = self.base_tokens
        self._integral = 0.0
        self._prev_error = 0.0
        self._disabled = False

    @property
    def current_tokens(self) -> int:
        return self._current_tokens

    @property
    def is_disabled(self) -> bool:
        return self._disabled
