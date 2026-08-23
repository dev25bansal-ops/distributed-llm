"""Tests for OneFOneBScheduler, PipelineSchedule, ScheduleStep, PipelineAction.

No mocks -- pure algorithmic tests for the 1F1B pipeline scheduling.
"""

from __future__ import annotations

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/pipeline_overlap.py")
OneFOneBScheduler = _mod.OneFOneBScheduler
PipelineSchedule = _mod.PipelineSchedule
ScheduleStep = _mod.ScheduleStep
PipelineAction = _mod.PipelineAction


class TestPipelineAction:
    """PipelineAction enum."""

    def test_values(self) -> None:
        assert PipelineAction.FORWARD.name == "FORWARD"
        assert PipelineAction.BACKWARD.name == "BACKWARD"
        assert PipelineAction.IDLE.name == "IDLE"
        assert PipelineAction.SEND.name == "SEND"
        assert PipelineAction.RECV.name == "RECV"

    def test_auto_values(self) -> None:
        """Each action gets a unique auto() value."""
        values = {a.value for a in PipelineAction}
        assert len(values) == 5


class TestScheduleStep:
    """ScheduleStep dataclass."""

    def test_creation(self) -> None:
        step = ScheduleStep(step_id=0, stage_id=1, action=PipelineAction.FORWARD, micro_batch_id=2)
        assert step.step_id == 0
        assert step.stage_id == 1
        assert step.action == PipelineAction.FORWARD
        assert step.micro_batch_id == 2

    def test_micro_batch_id_defaults_none(self) -> None:
        step = ScheduleStep(step_id=3, stage_id=0, action=PipelineAction.IDLE)
        assert step.micro_batch_id is None


class TestPipelineSchedule:
    """PipelineSchedule dataclass and helpers."""

    def test_default_construction(self) -> None:
        sched = PipelineSchedule()
        assert sched.steps == []
        assert sched.num_stages == 0
        assert sched.num_micro_batches == 0
        assert sched.total_steps == 0
        assert sched.bubble_ratio == 0.0

    def test_get_stage_schedule_filters_by_stage(self) -> None:
        steps = [
            ScheduleStep(0, 0, PipelineAction.FORWARD, 0),
            ScheduleStep(1, 1, PipelineAction.FORWARD, 0),
            ScheduleStep(2, 0, PipelineAction.BACKWARD, 0),
        ]
        sched = PipelineSchedule(steps=steps, num_stages=2, num_micro_batches=1, total_steps=3, bubble_ratio=0.0)
        stage0 = sched.get_stage_schedule(0)
        assert len(stage0) == 2
        assert all(s.stage_id == 0 for s in stage0)
        stage1 = sched.get_stage_schedule(1)
        assert len(stage1) == 1

    def test_summary_format(self) -> None:
        sched = PipelineSchedule(num_stages=4, num_micro_batches=8, total_steps=80, bubble_ratio=0.25)
        summary = sched.summary()
        assert "stages=4" in summary
        assert "micro_batches=8" in summary
        assert "total_steps=80" in summary
        assert "25.0%" in summary or "25%" in summary


class TestOneFOneBScheduler:
    """OneFOneBScheduler -- construction and schedule generation."""

    def test_default_construction(self) -> None:
        sched = OneFOneBScheduler()
        assert sched._num_stages == 4
        assert sched._micro_batch_size == 4

    def test_custom_construction(self) -> None:
        sched = OneFOneBScheduler(num_stages=8, micro_batch_size=16)
        assert sched._num_stages == 8
        assert sched._micro_batch_size == 16

    def test_schedule_uses_micro_batch_size_default(self) -> None:
        sched = OneFOneBScheduler(num_stages=2, micro_batch_size=4)
        schedule = sched.schedule()
        assert schedule.num_micro_batches == 4

    def test_schedule_single_stage(self) -> None:
        """With 1 stage, no pipeline bubble, just forward+backward pairs."""
        sched = OneFOneBScheduler(num_stages=1, micro_batch_size=3)
        schedule = sched.schedule(num_micro_batches=3)
        assert schedule.num_stages == 1
        assert schedule.num_micro_batches == 3
        # Single stage: no warmup, no cooldown
        assert schedule.total_steps == 6  # 3 forward + 3 backward

    def test_schedule_two_stages(self) -> None:
        sched = OneFOneBScheduler(num_stages=2, micro_batch_size=2)
        schedule = sched.schedule(num_micro_batches=2)
        assert schedule.num_stages == 2
        assert schedule.num_micro_batches == 2
        # Number of steps should be reasonable
        assert schedule.total_steps > 0

    def test_schedule_steps_have_valid_actions(self) -> None:
        sched = OneFOneBScheduler(num_stages=4, micro_batch_size=8)
        schedule = sched.schedule(num_micro_batches=8)
        valid_actions = {PipelineAction.FORWARD, PipelineAction.BACKWARD}
        for step in schedule.steps:
            assert step.action in valid_actions
            assert step.micro_batch_id is not None
            assert 0 <= step.micro_batch_id < 8
            assert 0 <= step.stage_id < 4

    def test_schedule_bubble_ratio_lower_with_more_batches(self) -> None:
        sched = OneFOneBScheduler(num_stages=4)
        sched_small = sched.schedule(num_micro_batches=4)
        sched_large = sched.schedule(num_micro_batches=16)
        # More micro-batches -> lower bubble ratio
        assert sched_large.bubble_ratio < sched_small.bubble_ratio

    def test_schedule_bubble_ratio_is_between_0_and_1(self) -> None:
        sched = OneFOneBScheduler(num_stages=4)
        for n in [2, 4, 8, 16, 32]:
            schedule = sched.schedule(num_micro_batches=n)
            assert 0.0 <= schedule.bubble_ratio <= 1.0

    def test_warmup_steps_decrease_with_stage(self) -> None:
        sched = OneFOneBScheduler(num_stages=4)
        assert sched.get_warmup_steps(0) == 3  # stage 0: 4-1-0 = 3
        assert sched.get_warmup_steps(1) == 2
        assert sched.get_warmup_steps(2) == 1
        assert sched.get_warmup_steps(3) == 0

    def test_cooldown_steps_increase_with_stage(self) -> None:
        sched = OneFOneBScheduler(num_stages=4)
        assert sched.get_cooldown_steps(0) == 0
        assert sched.get_cooldown_steps(1) == 1
        assert sched.get_cooldown_steps(2) == 2
        assert sched.get_cooldown_steps(3) == 3

    def test_schedule_4_stages_4_batches(self) -> None:
        """Known schedule: 4 stages, 4 micro-batches -> 38 steps, ~15.8% bubble."""
        sched = OneFOneBScheduler(num_stages=4, micro_batch_size=4)
        schedule = sched.schedule(num_micro_batches=4)
        assert schedule.total_steps == 38
        assert schedule.bubble_ratio == pytest.approx(0.1579, 0.01)

    def test_every_micro_batch_is_forwarded_and_backwarded(self) -> None:
        """Each micro-batch must have exactly one forward and one backward per stage."""
        sched = OneFOneBScheduler(num_stages=3, micro_batch_size=2)
        schedule = sched.schedule(num_micro_batches=2)
        for stage in range(3):
            fwds = {
                s.micro_batch_id
                for s in schedule.steps
                if s.stage_id == stage and s.action == PipelineAction.FORWARD
            }
            bwd = {
                s.micro_batch_id
                for s in schedule.steps
                if s.stage_id == stage and s.action == PipelineAction.BACKWARD
            }
            assert fwds == {0, 1}
            assert bwd == {0, 1}

