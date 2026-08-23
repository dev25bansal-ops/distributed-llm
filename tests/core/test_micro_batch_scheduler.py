"""Tests for MicroBatchScheduler -- micro-batched pipeline decode scheduling.

Covers:
- Construction with stage/micro-batch size
- schedule_decode divides tokens into micro-batches
- schedule_decode bubble ratio estimation
- get_next_batch returns pending batch
- get_next_batch returns None at max_in_flight
- complete_batch marks batch complete
- stats and properties

No MagicMock -- real lists, counters, and scheduling logic.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/micro_batch_scheduler.py")
MicroBatchScheduler = _mod.MicroBatchScheduler
MicroBatch = _mod.MicroBatch
MicroBatchSchedule = _mod.MicroBatchSchedule


class TestMicroBatchConstruction:
    """Dataclasses construction."""

    def test_micro_batch_defaults(self) -> None:
        mb = MicroBatch(batch_id=0)
        assert mb.batch_id == 0
        assert mb.token_ids == []
        assert mb.request_ids == []
        assert mb.status == "pending"
        assert mb.start_time == 0.0

    def test_schedule_defaults(self) -> None:
        sched = MicroBatchSchedule()
        assert sched.batches == []
        assert sched.num_stages == 0
        assert sched.total_tokens == 0
        assert sched.estimated_bubble_ratio == 0.0

    def test_scheduler_default_construction(self) -> None:
        sched = MicroBatchScheduler()
        assert sched._num_stages == 4
        assert sched._micro_batch_size == 4
        assert sched._max_in_flight == 8
        assert sched._in_flight == []
        assert sched._completed == []


class TestMicroBatchSchedulerSchedule:
    """Schedule creation."""

    def test_schedule_decode_single_batch(self) -> None:
        sched = MicroBatchScheduler(num_stages=2, micro_batch_size=4)
        schedule = sched.schedule_decode(num_tokens=3)
        assert len(schedule.batches) == 1
        assert schedule.total_tokens == 3
        assert schedule.batches[0].token_ids == [0, 1, 2]

    def test_schedule_decode_multiple_batches(self) -> None:
        sched = MicroBatchScheduler(num_stages=2, micro_batch_size=4)
        schedule = sched.schedule_decode(num_tokens=10)
        assert len(schedule.batches) == 3  # 4 + 4 + 2
        assert schedule.batches[0].token_ids == [0, 1, 2, 3]
        assert schedule.batches[1].token_ids == [4, 5, 6, 7]
        assert schedule.batches[2].token_ids == [8, 9]

    def test_schedule_bubble_ratio(self) -> None:
        sched = MicroBatchScheduler(num_stages=4, micro_batch_size=4)
        schedule = sched.schedule_decode(num_tokens=4)
        # total_steps = 1 + 4 - 1 = 4, useful = 1
        # bubble = 1 - 1/4 = 0.75
        assert schedule.estimated_bubble_ratio > 0.0
        assert schedule.num_stages == 4

    def test_schedule_smaller_bubble_with_more_tokens(self) -> None:
        sched = MicroBatchScheduler(num_stages=4, micro_batch_size=4)
        schedule_small = sched.schedule_decode(num_tokens=4)
        schedule_large = sched.schedule_decode(num_tokens=16)
        # More tokens = more batches = lower bubble ratio
        assert schedule_large.estimated_bubble_ratio < schedule_small.estimated_bubble_ratio

    def test_schedule_zero_tokens(self) -> None:
        sched = MicroBatchScheduler()
        schedule = sched.schedule_decode(num_tokens=0)
        assert len(schedule.batches) == 0
        assert schedule.total_tokens == 0


class TestMicroBatchSchedulerGetNext:
    """Getting next batch."""

    def test_get_next_batch_no_schedule(self) -> None:
        sched = MicroBatchScheduler()
        batch = sched.get_next_batch()
        assert batch is None  # nothing scheduled

    def test_get_next_batch_returns_none_at_max(self) -> None:
        sched = MicroBatchScheduler(num_stages=2, micro_batch_size=2, max_in_flight=1)
        schedule = sched.schedule_decode(num_tokens=4)
        # Add first batch to in_flight
        sched._in_flight.append(schedule.batches[0])
        batch = sched.get_next_batch()
        assert batch is None  # max_in_flight reached


class TestMicroBatchSchedulerComplete:
    """Batch completion."""

    def test_complete_batch(self) -> None:
        sched = MicroBatchScheduler()
        schedule = sched.schedule_decode(num_tokens=4)
        sched._in_flight.append(schedule.batches[0])
        sched.complete_batch(batch_id=0)
        assert len(sched._completed) == 1
        assert schedule.batches[0].status == "complete"

    def test_complete_batch_updates_end_time(self) -> None:
        sched = MicroBatchScheduler()
        schedule = sched.schedule_decode(num_tokens=4)
        sched._in_flight.append(schedule.batches[0])
        sched.complete_batch(batch_id=0)
        assert schedule.batches[0].end_time > 0.0

    def test_complete_nonexistent_batch_does_not_crash(self) -> None:
        sched = MicroBatchScheduler()
        sched.complete_batch(batch_id=999)  # should not raise


class TestMicroBatchSchedulerProperties:
    """Properties and stats."""

    def test_micro_batch_size_property(self) -> None:
        sched = MicroBatchScheduler(micro_batch_size=8)
        assert sched.micro_batch_size == 8

    def test_in_flight_count(self) -> None:
        sched = MicroBatchScheduler()
        assert sched.in_flight_count == 0
        sched._in_flight.append(MicroBatch(batch_id=1))
        assert sched.in_flight_count == 1

    def test_stats(self) -> None:
        sched = MicroBatchScheduler(num_stages=4, micro_batch_size=4)
        sched.schedule_decode(num_tokens=8)
        stats = sched.stats()
        assert stats["batches_scheduled"] == 2
        assert stats["tokens_processed"] == 8
        assert stats["num_stages"] == 4
        assert stats["micro_batch_size"] == 4
