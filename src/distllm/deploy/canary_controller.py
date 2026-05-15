"""Canary controller for managing deployments."""

from typing import Callable, Dict, Optional

from loguru import logger

from distllm.deploy.traffic_splitter import TrafficSplitter
from distllm.deploy.rollout_strategy import RolloutStrategy, RolloutState


class CanaryController:
    """Manages canary deployment lifecycle: promotion, rollback, monitoring.

    Flow:
    1. Deploy canary with initial weight (e.g., 5%)
    2. Monitor error rate, latency, throughput for analysis window
    3. If metrics pass thresholds, increase weight by configured step
    4. Repeat until 100% or rollback triggered
    """

    def __init__(
        self,
        stable_version: str = "stable",
        canary_version: str = "canary",
        rollback_threshold: float = 0.05,
        strategy: Optional[RolloutStrategy] = None,
    ):
        self.stable_version = stable_version
        self.canary_version = canary_version
        self.rollback_threshold = rollback_threshold

        self.strategy = strategy or RolloutStrategy()
        self.splitter = TrafficSplitter(
            stable_version=stable_version,
            canary_version=canary_version,
        )
        self._rollout_state: Optional[RolloutState] = None
        self._on_rollback: Optional[Callable] = None
        self._on_promotion: Optional[Callable] = None

    def set_rollback_callback(self, callback: Callable) -> None:
        """Set callback invoked when rollback is triggered."""
        self._on_rollback = callback

    def set_promotion_callback(self, callback: Callable) -> None:
        """Set callback invoked when a stage is promoted."""
        self._on_promotion = callback

    @property
    def is_active(self) -> bool:
        """Check if a canary deployment is active."""
        return self._rollout_state is not None and not self._rollout_state.is_complete

    @property
    def rollout_state(self) -> Optional[RolloutState]:
        return self._rollout_state

    def start_canary(self, canary_version: str) -> RolloutState:
        """Start a canary deployment.

        Args:
            canary_version: Version string for the canary.

        Returns:
            The created RolloutState.
        """
        if self.is_active:
            raise RuntimeError("Canary deployment already active")

        self.canary_version = canary_version
        self.splitter.canary_version = canary_version
        self._rollout_state = self.strategy.create_rollout(canary_version)

        # Start at first stage
        first_stage = self.strategy.get_current_stage(self._rollout_state)
        if first_stage:
            self.splitter.set_canary_pct(first_stage.weight_pct)
            self._rollout_state.current_weight_pct = first_stage.weight_pct

        logger.info(f"[Canary] Started canary deployment '{canary_version}' at {first_stage.weight_pct}%")
        return self._rollout_state

    def check_and_advance(self, current_error_rate: float, current_p99_ms: float, stable_p99_ms: float) -> str:
        """Check metrics and decide whether to advance or rollback.

        Args:
            current_error_rate: Current error rate for canary.
            current_p99_ms: Current p99 latency for canary.
            stable_p99_ms: Current p99 latency for stable.

        Returns:
            "advanced", "rolled_back", or "waiting"
        """
        if not self.is_active or self._rollout_state is None:
            return "waiting"

        # Check if we should rollback
        if self.strategy.should_rollback(
            self._rollout_state,
            current_error_rate,
            current_p99_ms,
            stable_p99_ms,
        ):
            self._trigger_rollback()
            return "rolled_back"

        # Check if analysis window has elapsed
        import time
        stage = self.strategy.get_current_stage(self._rollout_state)
        if stage is None:
            return "waiting"

        elapsed = time.time() - self._rollout_state.stage_start_time
        if elapsed >= stage.analysis_duration_s:
            # Advance to next stage
            if self.strategy.advance_stage(self._rollout_state):
                self.splitter.set_canary_pct(self._rollout_state.current_weight_pct)
                if self._on_promotion:
                    self._on_promotion(self._rollout_state)
                logger.info(
                    f"[Canary] Advanced to {self._rollout_state.current_weight_pct}% canary"
                )
                return "advanced"
            else:
                logger.info("[Canary] Rollout complete, 100% traffic on canary")
                return "advanced"

        return "waiting"

    def _trigger_rollback(self) -> None:
        """Execute a canary rollback."""
        if self._rollout_state is None:
            return

        self.strategy.trigger_rollback(self._rollout_state)
        self.splitter.set_canary_pct(0.0)

        if self._on_rollback:
            self._on_rollback(self._rollout_state)

        logger.warning(f"[Canary] Rolled back canary '{self.canary_version}'")

    def get_canary_version(self, request_id: str) -> str:
        """Get which version should handle a specific request."""
        if not self.is_active:
            return self.stable_version
        return self.splitter.select_version(request_id)

    def abort_canary(self) -> None:
        """Manually abort the canary deployment."""
        if self._rollout_state:
            self._trigger_rollback()
            self._rollout_state = None
