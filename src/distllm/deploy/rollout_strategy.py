"""Rollout strategy for progressive canary deployments."""

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class RolloutStage:
    """A single stage in a canary rollout."""
    weight_pct: float
    analysis_duration_s: int = 300
    max_error_rate: float = 0.05
    max_latency_p99_multiplier: float = 2.0  # compared to stable p99
    min_throughput: float = 0.0


@dataclass
class RolloutState:
    """Current state of a canary rollout."""
    canary_version: str
    current_stage_index: int = 0
    current_weight_pct: float = 0.0
    stage_start_time: float = 0.0
    is_rolling_back: bool = False
    is_complete: bool = False
    error_count: int = 0
    total_requests: int = 0
    avg_latency_ms: float = 0.0


class RolloutStrategy:
    """Manages progressive canary rollout stages."""

    DEFAULT_STAGES = [
        RolloutStage(weight_pct=5, analysis_duration_s=300),
        RolloutStage(weight_pct=25, analysis_duration_s=600),
        RolloutStage(weight_pct=50, analysis_duration_s=600),
        RolloutStage(weight_pct=75, analysis_duration_s=300),
        RolloutStage(weight_pct=100, analysis_duration_s=300),
    ]

    def __init__(self, stages: list[RolloutStage] | None = None):
        self.stages = stages or self.DEFAULT_STAGES

    def create_rollout(self, canary_version: str) -> RolloutState:
        """Create a new rollout state."""
        import time
        return RolloutState(
            canary_version=canary_version,
            current_stage_index=0,
            current_weight_pct=0.0,
            stage_start_time=time.time(),
        )

    def get_next_stage(self, state: RolloutState) -> RolloutStage | None:
        """Get the next rollout stage.

        Returns:
            Next RolloutStage, or None if rollout is complete.
        """
        next_index = state.current_stage_index + 1
        if next_index >= len(self.stages):
            return None
        return self.stages[next_index]

    def get_current_stage(self, state: RolloutState) -> RolloutStage | None:
        """Get the current rollout stage."""
        if state.current_stage_index >= len(self.stages):
            return None
        return self.stages[state.current_stage_index]

    def advance_stage(self, state: RolloutState) -> bool:
        """Advance to the next stage.

        Returns:
            True if advanced, False if rollout is complete.
        """
        next_stage = self.get_next_stage(state)
        if next_stage is None:
            state.is_complete = True
            state.current_weight_pct = 100.0
            return False

        import time
        state.current_stage_index += 1
        state.current_weight_pct = next_stage.weight_pct
        state.stage_start_time = time.time()
        return True

    def should_rollback(
        self,
        state: RolloutState,
        current_error_rate: float,
        current_p99_latency_ms: float,
        stable_p99_latency_ms: float,
    ) -> bool:
        """Check if rollout should be rolled back.

        Args:
            state: Current rollout state.
            current_error_rate: Current error rate for canary.
            current_p99_latency_ms: Current p99 latency for canary.
            stable_p99_latency_ms: Current p99 latency for stable.

        Returns:
            True if rollback is needed.
        """
        stage = self.get_current_stage(state)
        if stage is None:
            return False

        # Check error rate threshold
        if current_error_rate > stage.max_error_rate:
            return True

        # Check latency threshold
        if stable_p99_latency_ms > 0:
            latency_ratio = current_p99_latency_ms / stable_p99_latency_ms
            if latency_ratio > stage.max_latency_p99_multiplier:
                return True

        return False

    def trigger_rollback(self, state: RolloutState) -> None:
        """Mark rollout for rollback."""
        state.is_rolling_back = True
        state.current_weight_pct = 0.0

    def record_request(self, state: RolloutState, was_error: bool, latency_ms: float) -> None:
        """Record a request for analysis."""
        state.total_requests += 1
        if was_error:
            state.error_count += 1
        # Running average latency
        n = state.total_requests
        state.avg_latency_ms = state.avg_latency_ms + (latency_ms - state.avg_latency_ms) / n

    def get_analysis(self, state: RolloutState) -> dict:
        """Get current analysis metrics."""
        error_rate = state.error_count / state.total_requests if state.total_requests > 0 else 0.0
        return {
            "error_rate": error_rate,
            "total_requests": state.total_requests,
            "avg_latency_ms": state.avg_latency_ms,
            "current_weight_pct": state.current_weight_pct,
            "stage_index": state.current_stage_index,
            "is_complete": state.is_complete,
            "is_rolling_back": state.is_rolling_back,
        }
