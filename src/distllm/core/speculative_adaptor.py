"""Speculative Adaptor — adaptive candidate count based on acceptance rate.

Dynamically adjusts the number of draft tokens based on real-time
acceptance rate feedback.  When the draft model is performing well
(high acceptance), more candidates are generated to maximize throughput.
When performance drops, fewer candidates are generated to avoid wasting
target model compute on verification.

Usage::

    adaptor = SpeculativeAdaptor(base_tokens=5, target_rate=0.6)

    new_tokens = adaptor.adapt(acceptance_rate=0.8)  # Increase
    new_tokens = adaptor.adapt(acceptance_rate=0.2)  # Decrease
    new_tokens = adaptor.adapt(acceptance_rate=0.05) # Disable (returns 0)
"""

from __future__ import annotations

from loguru import logger


class SpeculativeAdaptor:
    """Adapts speculative decoding candidate count based on acceptance rate.

    Args:
        base_tokens: Initial number of draft tokens.
        target_rate: Acceptance rate threshold for adjustments.
        min_tokens: Minimum draft tokens (below this, speculation stops).
        max_tokens: Maximum draft tokens.
        disable_threshold: Acceptance rate below this disables speculation.
        adjustment_step: How many tokens to add/remove per adjustment.
    """

    def __init__(
        self,
        base_tokens: int = 5,
        target_rate: float = 0.6,
        min_tokens: int = 1,
        max_tokens: int = 10,
        disable_threshold: float = 0.15,
        adjustment_step: int = 1,
    ) -> None:
        self._base_tokens = base_tokens
        self._current_tokens = base_tokens
        self._target_rate = target_rate
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens
        self._disable_threshold = disable_threshold
        self._step = adjustment_step
        self._disabled = False

    def adapt(self, acceptance_rate: float) -> int:
        """Adjust candidate count based on the latest acceptance rate.

        Returns the new number of draft tokens (0 if speculation is disabled).
        """
        if self._disabled:
            return 0

        # Disable if acceptance is critically low
        if acceptance_rate < self._disable_threshold:
            self._disabled = True
            self._current_tokens = 0
            logger.debug(
                "Speculative decoding disabled: acceptance_rate={:.3f} < threshold={:.3f}",
                acceptance_rate, self._disable_threshold,
            )
            return 0

        # Increase if above target
        if acceptance_rate > self._target_rate + 0.1:
            self._current_tokens = min(
                self._current_tokens + self._step, self._max_tokens,
            )
        # Decrease if below target
        elif acceptance_rate < self._target_rate - 0.1:
            self._current_tokens = max(
                self._current_tokens - self._step, self._min_tokens,
            )

        return self._current_tokens

    def reset(self) -> None:
        """Reset to base configuration."""
        self._current_tokens = self._base_tokens
        self._disabled = False

    @property
    def current_tokens(self) -> int:
        return self._current_tokens

    @property
    def is_disabled(self) -> bool:
        return self._disabled

    @property
    def base_tokens(self) -> int:
        return self._base_tokens
