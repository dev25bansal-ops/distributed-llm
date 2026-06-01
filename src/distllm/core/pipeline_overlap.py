"""1F1B (One-Forward-One-Backward) pipeline overlap strategy.

Implements the 1F1B scheduling pattern for pipeline-parallel inference
that overlaps forward and backward passes across pipeline stages to
reduce bubble time and improve throughput by 2-3x.

The 1F1B pattern:
1. Warmup phase: Fill the pipeline with forward passes
2. Steady state: Alternate 1 forward + 1 backward per stage
3. Cooldown phase: Drain remaining backward passes

This eliminates the pipeline bubble that occurs in sequential scheduling
where all stages wait for the slowest stage.

Usage::

    scheduler = OneFOneBScheduler(num_stages=4, micro_batch_size=4)
    schedule = scheduler.schedule(num_micro_batches=16)
    # schedule contains the execution order for each stage
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from loguru import logger


class PipelineAction(Enum):
    """Actions a pipeline stage can take."""
    FORWARD = auto()
    BACKWARD = auto()
    IDLE = auto()
    SEND = auto()
    RECV = auto()


@dataclass
class ScheduleStep:
    """A single step in the pipeline schedule."""
    step_id: int
    stage_id: int
    action: PipelineAction
    micro_batch_id: int | None = None


@dataclass
class PipelineSchedule:
    """Complete pipeline schedule for all stages."""
    steps: list[ScheduleStep] = field(default_factory=list)
    num_stages: int = 0
    num_micro_batches: int = 0
    total_steps: int = 0
    bubble_ratio: float = 0.0

    def get_stage_schedule(self, stage_id: int) -> list[ScheduleStep]:
        """Get the schedule for a specific stage."""
        return [s for s in self.steps if s.stage_id == stage_id]

    def summary(self) -> str:
        return (
            f"PipelineSchedule(stages={self.num_stages}, "
            f"micro_batches={self.num_micro_batches}, "
            f"total_steps={self.total_steps}, "
            f"bubble={self.bubble_ratio:.1%})"
        )


class OneFOneBScheduler:
    """Implements 1F1B (One-Forward-One-Backward) pipeline scheduling.

    The 1F1B pattern reduces pipeline bubbles by interleaving forward
    and backward passes. After the warmup phase, each stage alternates
    between forward and backward, keeping all stages busy.

    Args:
        num_stages: Number of pipeline stages (nodes).
        micro_batch_size: Number of micro-batches to split the batch into.
    """

    def __init__(self, num_stages: int = 4, micro_batch_size: int = 4):
        self._num_stages = num_stages
        self._micro_batch_size = micro_batch_size

    def schedule(self, num_micro_batches: int | None = None) -> PipelineSchedule:
        """Generate a 1F1B schedule for the given number of micro-batches.

        Args:
            num_micro_batches: Number of micro-batches. Defaults to micro_batch_size.

        Returns:
            PipelineSchedule with the execution order for each stage.
        """
        n = num_micro_batches or self._micro_batch_size
        s = self._num_stages

        steps: list[ScheduleStep] = []
        step_id = 0

        # Phase 1: Warmup — fill the pipeline with forward passes
        # Each stage starts with s-1 forward passes
        for i in range(s - 1):
            for stage in range(s):
                mb = i - stage
                if 0 <= mb < n:
                    steps.append(ScheduleStep(
                        step_id=step_id,
                        stage_id=stage,
                        action=PipelineAction.FORWARD,
                        micro_batch_id=mb,
                    ))
                    step_id += 1

        # Phase 2: Steady state — alternate 1 forward + 1 backward
        for i in range(n):
            # Forward pass
            for stage in range(s):
                mb = i
                if mb < n:
                    steps.append(ScheduleStep(
                        step_id=step_id,
                        stage_id=stage,
                        action=PipelineAction.FORWARD,
                        micro_batch_id=mb,
                    ))
                    step_id += 1

            # Backward pass
            for stage in range(s - 1, -1, -1):
                mb = i - (s - 1 - stage)
                if 0 <= mb < n:
                    steps.append(ScheduleStep(
                        step_id=step_id,
                        stage_id=stage,
                        action=PipelineAction.BACKWARD,
                        micro_batch_id=mb,
                    ))
                    step_id += 1

        # Phase 3: Cooldown — drain remaining backward passes
        for i in range(s - 1):
            for stage in range(s - 1, -1, -1):
                mb = n - (s - 1 - stage) + i
                if 0 <= mb < n:
                    steps.append(ScheduleStep(
                        step_id=step_id,
                        stage_id=stage,
                        action=PipelineAction.BACKWARD,
                        micro_batch_id=mb,
                    ))
                    step_id += 1

        # Calculate bubble ratio
        total_ops = step_id
        useful_ops = n * s * 2  # n micro-batches * s stages * (forward + backward)
        bubble_ratio = 1.0 - (useful_ops / max(total_ops, 1))

        schedule = PipelineSchedule(
            steps=steps,
            num_stages=s,
            num_micro_batches=n,
            total_steps=step_id,
            bubble_ratio=max(0, bubble_ratio),
        )

        logger.debug(schedule.summary())
        return schedule

    def get_warmup_steps(self, stage_id: int) -> int:
        """Return the number of warmup forward passes for a stage."""
        return self._num_stages - 1 - stage_id

    def get_cooldown_steps(self, stage_id: int) -> int:
        """Return the number of cooldown backward passes for a stage."""
        return stage_id
