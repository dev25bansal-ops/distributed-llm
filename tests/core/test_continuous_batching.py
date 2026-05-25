"""Continuous batching: merge, complete, and iteration-level scheduling tests.

Scenarios:
  - Existing active + new pending → merged batch
  - Finished sequence → removed from batch
  - Prefill vs decode iteration scheduling
  - Sarathi-Serve adaptive pressure budget
"""

import math

import pytest
import torch

from distllm.core.batch_scheduler import (
    BatchScheduler,
    DecodePressureTracker,
    IterationBudget,
    ScheduledBatch,
    Sequence,
    SequenceStatus,
)


class TestContinuousBatchingMerge:
    """Existing active sequences + new pending requests → merged batch."""

    def test_merge_active_with_pending(self):
        sched = BatchScheduler(max_batch_size=8, max_tokens_per_batch=512)
        for i in range(3):
            sched.add(Sequence(request_id=f"req-{i}", prompt_tokens=[i] * 8))
        batch1 = sched.schedule()
        assert batch1 is not None and batch1.batch_size == 3

        for i in range(3, 6):
            sched.add(Sequence(request_id=f"req-{i}", prompt_tokens=[i] * 8))
        batch2 = sched.schedule()
        assert batch2 is not None and batch2.batch_size == 6
        assert all(f"req-{i}" in batch2.request_ids for i in range(6))

    def test_merge_respects_max_batch(self):
        sched = BatchScheduler(max_batch_size=4, max_tokens_per_batch=10000)
        sched._adapt_prefill_budget = False
        for i in range(3):
            sched.add(Sequence(request_id=f"active-{i}", prompt_tokens=[i] * 10))
        sched.schedule()
        for i in range(5):
            sched.add(Sequence(request_id=f"new-{i}", prompt_tokens=[i] * 10))
        batch = sched.schedule()
        assert batch.batch_size == 4
        assert sched.pending_count == 4

    def test_merge_keeps_active_unchanged(self):
        sched = BatchScheduler(max_batch_size=8, max_tokens_per_batch=10000)
        sched._adapt_prefill_budget = False
        sched.add(Sequence(request_id="stay", prompt_tokens=[1] * 50, max_new_tokens=10))
        sched.schedule()
        sched.add(Sequence(request_id="new", prompt_tokens=[2] * 10))
        batch = sched.schedule()
        assert "stay" in batch.request_ids
        assert "new" in batch.request_ids

    def test_merge_empty_pending_returns_only_active(self):
        sched = BatchScheduler(max_batch_size=8, max_tokens_per_batch=10000)
        sched.add(Sequence(request_id="req-1", prompt_tokens=[1] * 10))
        sched.schedule()
        batch = sched.schedule()
        assert batch is not None
        assert batch.batch_size == 1
        assert batch.request_ids == ["req-1"]


class TestContinuousBatchingComplete:
    """Finished sequence → removed from batch."""

    def test_step_marks_done_and_removes_from_active(self):
        sched = BatchScheduler(max_batch_size=4, max_tokens_per_batch=256)
        seq = Sequence(request_id="req-1", prompt_tokens=[1], max_new_tokens=1)
        sched.active["req-1"] = seq
        batch = ScheduledBatch(
            sequences=[seq],
            input_ids=torch.tensor([[1]]),
            seq_lengths=[1],
            position_offsets=[0],
            is_prefill=[True],
            request_ids=["req-1"],
        )
        sched.step(batch, torch.tensor([42]))
        assert seq.status == SequenceStatus.DONE
        assert "req-1" not in sched.active

    def test_one_completes_others_stay(self):
        sched = BatchScheduler(max_batch_size=8, max_tokens_per_batch=10000)
        seq1 = Sequence(request_id="done", prompt_tokens=[1], max_new_tokens=1)
        seq2 = Sequence(request_id="stay", prompt_tokens=[2], max_new_tokens=10)
        sched.active["done"] = seq1
        sched.active["stay"] = seq2
        batch = ScheduledBatch(
            sequences=[seq1, seq2],
            input_ids=torch.tensor([[1], [2]]),
            seq_lengths=[1, 1],
            position_offsets=[0, 0],
            is_prefill=[True, True],
            request_ids=["done", "stay"],
        )
        sched.step(batch, torch.tensor([42, 43]))
        assert "done" not in sched.active
        assert "stay" in sched.active

    def test_next_schedule_excludes_completed(self):
        sched = BatchScheduler(max_batch_size=8, max_tokens_per_batch=10000)
        seq = Sequence(request_id="done", prompt_tokens=[1], max_new_tokens=1)
        sched.active["done"] = seq
        step_batch = ScheduledBatch(
            sequences=[seq],
            input_ids=torch.tensor([[1]]),
            seq_lengths=[1],
            position_offsets=[0],
            is_prefill=[True],
            request_ids=["done"],
        )
        sched.step(step_batch, torch.tensor([42]))
        next_batch = sched.schedule()
        assert next_batch is None or "done" not in next_batch.request_ids

    def test_stop_token_removes_from_active(self):
        sched = BatchScheduler(max_batch_size=4, max_tokens_per_batch=256)
        seq = Sequence(request_id="req-1", prompt_tokens=[1], max_new_tokens=10, stop_token_ids=[0])
        sched.active["req-1"] = seq
        batch = ScheduledBatch(
            sequences=[seq],
            input_ids=torch.tensor([[1]]),
            seq_lengths=[1],
            position_offsets=[0],
            is_prefill=[True],
            request_ids=["req-1"],
        )
        sched.step(batch, torch.tensor([0]))
        assert seq.status == SequenceStatus.DONE
        assert "req-1" not in sched.active

    def test_completion_then_new_pending_promoted(self):
        sched = BatchScheduler(max_batch_size=4, max_tokens_per_batch=10000)
        seq = Sequence(request_id="done", prompt_tokens=[1], max_new_tokens=1)
        sched.active["done"] = seq
        step_batch = ScheduledBatch(
            sequences=[seq],
            input_ids=torch.tensor([[1]]),
            seq_lengths=[1],
            position_offsets=[0],
            is_prefill=[True],
            request_ids=["done"],
        )
        sched.step(step_batch, torch.tensor([42]))
        sched.add(Sequence(request_id="replacement", prompt_tokens=[2] * 10))
        next_batch = sched.schedule()
        assert next_batch is not None
        assert "replacement" in next_batch.request_ids
        assert "done" not in next_batch.request_ids


class TestIterationLevelScheduling:
    """Prefill vs decode scheduling, chunked prefill, and iteration budgets."""

    def test_schedule_iteration_basic(self):
        sched = BatchScheduler(max_batch_size=8, max_tokens_per_batch=10000)
        sched.add(Sequence(request_id="req-1", prompt_tokens=[1] * 50))
        batch = sched.schedule_iteration()
        assert batch is not None
        assert batch.batch_size == 1

    def test_schedule_iteration_respects_budget(self):
        budget = IterationBudget(
            max_prefill_tokens=4,
            max_decode_tokens=1,
            max_batch_size=4,
            max_total_tokens=32,
            enable_chunked_prefill=True,
        )
        sched = BatchScheduler(max_batch_size=4, max_tokens_per_batch=32)
        sched.set_iteration_budget(budget)
        sched.add(Sequence(request_id="big", prompt_tokens=[1] * 20))
        batch = sched.schedule_iteration()
        assert batch is not None
        prefill_tokens = batch.seq_lengths[0]
        assert prefill_tokens <= budget.max_prefill_tokens

    def test_iteration_tracks_count(self):
        sched = BatchScheduler(max_batch_size=8, max_tokens_per_batch=10000)
        sched.add(Sequence(request_id="req-1", prompt_tokens=[1] * 10))
        assert sched._iteration_count == 0
        sched.schedule_iteration()
        assert sched._iteration_count == 1
        sched.schedule_iteration()
        assert sched._iteration_count == 2

    def test_decode_always_included(self):
        budget = IterationBudget(
            max_prefill_tokens=1,
            max_decode_tokens=4,
            max_batch_size=4,
            max_total_tokens=32,
            enable_chunked_prefill=True,
        )
        sched = BatchScheduler(max_batch_size=4, max_tokens_per_batch=32)
        sched.set_iteration_budget(budget)
        sched.active["decode1"] = Sequence(
            request_id="decode1", prompt_tokens=[1] * 5, generated_tokens=[10], status=SequenceStatus.DECODING
        )
        batch = sched.schedule_iteration(budget)
        assert batch is not None
        assert "decode1" in batch.request_ids

    def test_chunked_prefill_splits_large_prompt(self):
        budget = IterationBudget(
            max_prefill_tokens=4,
            max_decode_tokens=4,
            max_batch_size=4,
            max_total_tokens=100,
            enable_chunked_prefill=True,
        )
        sched = BatchScheduler(max_batch_size=4, max_tokens_per_batch=100)
        sched.set_iteration_budget(budget)
        sched.add(Sequence(request_id="big", prompt_tokens=[1] * 20))
        batch1 = sched.schedule_iteration(budget)
        assert batch1 is not None
        assert "big" in batch1.request_ids
        cinfo = sched._chunked_prefill.get("big")
        assert cinfo is not None
        assert not cinfo.is_complete
        assert cinfo.chunks_remaining > 0

    def test_chunked_prefill_completes_across_iterations(self):
        budget = IterationBudget(
            max_prefill_tokens=4,
            max_decode_tokens=4,
            max_batch_size=4,
            max_total_tokens=100,
            enable_chunked_prefill=True,
        )
        sched = BatchScheduler(max_batch_size=4, max_tokens_per_batch=100)
        sched.set_iteration_budget(budget)
        sched.add(Sequence(request_id="big", prompt_tokens=[1] * 10))
        iterations = 0
        while "big" in sched._chunked_prefill or sched.pending_count > 0:
            batch = sched.schedule_iteration(budget)
            if batch is None:
                break
            iterations += 1
            for seq in batch.sequences:
                if seq.request_id == "big" and seq.status == SequenceStatus.PREFILLING:
                    pass
        assert iterations > 1

    def test_mixed_prefill_decode_batch(self):
        sched = BatchScheduler(max_batch_size=8, max_tokens_per_batch=10000)
        sched.active["decode1"] = Sequence(
            request_id="decode1", prompt_tokens=[1] * 5, generated_tokens=[10], status=SequenceStatus.DECODING
        )
        sched.add(Sequence(request_id="new-prefill", prompt_tokens=[2] * 10))
        batch = sched.schedule_iteration()
        assert batch is not None
        assert len(batch.request_ids) == 2
        assert "decode1" in batch.request_ids
        assert "new-prefill" in batch.request_ids

    def test_prefill_chunk_advances_tokens_processed(self):
        budget = IterationBudget(
            max_prefill_tokens=3,
            max_decode_tokens=4,
            max_batch_size=4,
            max_total_tokens=100,
            enable_chunked_prefill=True,
        )
        sched = BatchScheduler(max_batch_size=4, max_tokens_per_batch=100)
        sched.set_iteration_budget(budget)
        sched.add(Sequence(request_id="seq", prompt_tokens=[1] * 9))
        batch1 = sched.schedule_iteration(budget)
        assert sched._chunked_prefill["seq"].tokens_processed == 3
        batch2 = sched.schedule_iteration(budget)
        assert sched._chunked_prefill["seq"].tokens_processed == 6


class TestSarathiBudget:
    """Sarathi-Serve adaptive decode-first budget computation."""

    def test_no_pressure_default_budget(self):
        tracker = DecodePressureTracker(window_size=5, target_ms_per_token=8.0)
        assert tracker.pressure == 0.0

    def test_pressure_increases_with_latency(self):
        tracker = DecodePressureTracker(window_size=5, target_ms_per_token=10.0)
        for _ in range(5):
            tracker.record_decode_step(4, 80.0)
        assert tracker.pressure > 0.0

    def test_pressure_capped_at_one(self):
        tracker = DecodePressureTracker(window_size=3, target_ms_per_token=10.0)
        for _ in range(3):
            tracker.record_decode_step(1, 100.0)
        assert tracker.pressure <= 1.0

    def test_pressure_zero_with_no_samples(self):
        tracker = DecodePressureTracker()
        assert tracker.pressure == 0.0
        assert tracker.avg_ms_per_token == 0.0

    def test_avg_ms_per_token(self):
        tracker = DecodePressureTracker(window_size=3)
        tracker.record_decode_step(2, 20.0)
        assert tracker.avg_ms_per_token == 10.0

    def test_sarathi_budget_high_pressure_throttles_prefill(self):
        sched = BatchScheduler(max_batch_size=8, max_tokens_per_batch=10000)
        sched._pressure_tracker._decode_latencies = []
        for _ in range(10):
            sched._pressure_tracker.record_decode_step(1, 100.0)
        budget = IterationBudget(
            max_prefill_tokens=4096,
            max_decode_tokens=512,
            max_batch_size=8,
            max_total_tokens=32768,
        )
        sarathi = sched._compute_sarathi_budget(budget)
        assert sarathi.max_prefill_tokens < budget.max_prefill_tokens

    def test_sarathi_budget_low_pressure_relaxes(self):
        sched = BatchScheduler(max_batch_size=8, max_tokens_per_batch=10000)
        sched._pressure_tracker._decode_latencies = []
        for _ in range(10):
            sched._pressure_tracker.record_decode_step(4, 5.0)
        budget = IterationBudget(
            max_prefill_tokens=4096,
            max_decode_tokens=512,
            max_batch_size=8,
            max_total_tokens=32768,
        )
        sched.active["d1"] = Sequence(
            request_id="d1", prompt_tokens=[1], generated_tokens=[2], status=SequenceStatus.DECODING
        )
        sarathi = sched._compute_sarathi_budget(budget)
        assert sarathi.max_decode_tokens >= 1

    def test_sarathi_guarantees_decode_slots(self):
        sched = BatchScheduler(max_batch_size=4, max_tokens_per_batch=10000)
        for i in range(3):
            sched.active[f"d{i}"] = Sequence(
                request_id=f"d{i}", prompt_tokens=[1], generated_tokens=[2], status=SequenceStatus.DECODING
            )
        budget = IterationBudget(
            max_prefill_tokens=4096,
            max_decode_tokens=2,
            max_batch_size=4,
            max_total_tokens=32768,
        )
        sched._pressure_tracker = DecodePressureTracker(window_size=5, target_ms_per_token=10.0)
        for _ in range(5):
            sched._pressure_tracker.record_decode_step(1, 9.0)
        sarathi = sched._compute_sarathi_budget(budget)
        assert sarathi.decode_slots >= 1
        assert sarathi.decode_slots <= budget.max_batch_size
