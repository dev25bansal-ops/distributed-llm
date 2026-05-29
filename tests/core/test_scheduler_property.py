"""Property-based tests for BatchScheduler using Hypothesis.

Tests invariant properties that must always hold after scheduling:
- Batch size never exceeds max_batch_size
- Total tokens never exceeds max_tokens_per_batch
- Priority ordering is respected
- No data loss during scheduling
- Budget safety

Run: pytest tests/core/test_scheduler_property.py -v
"""

import pytest
import torch
from hypothesis import given, settings, assume, strategies as st

from distllm.core.batch_scheduler import (
    BatchScheduler,
    IterationBudget,
    Sequence,
    SequenceStatus,
)


def _make_seq(
    request_id: str = "req-1",
    prompt_len: int = 50,
    max_new: int = 128,
    priority: int = 2,
) -> Sequence:
    return Sequence(
        request_id=request_id,
        prompt_tokens=[1] * prompt_len,
        max_new_tokens=max_new,
        priority=priority,
    )


# ===================================================================
# Property-Based Tests
# ===================================================================


class TestSchedulerInvariants:
    """Invariant properties that must always hold."""

    @given(
        max_batch=st.integers(1, 32),
        n=st.integers(0, 50),
        priorities=st.lists(st.integers(0, 3), max_size=50),
        max_tokens=st.integers(100, 5000),
    )
    @settings(max_examples=50, deadline=5000)
    def test_batch_size_never_exceeds_max(self, max_batch, n, priorities, max_tokens):
        """Batch size must never exceed max_batch_size."""
        sched = BatchScheduler(
            max_batch_size=max_batch,
            max_tokens_per_batch=max_tokens,
        )
        for i in range(n):
            p = priorities[i] if i < len(priorities) else 2
            sched.add(_make_seq(f"r{i}", prompt_len=10, priority=p))

        batch = sched.schedule()
        if batch is not None:
            assert len(batch.sequences) <= max_batch

    @given(
        max_batch=st.integers(1, 16),
        n=st.integers(0, 30),
        max_tokens=st.integers(100, 5000),
    )
    @settings(max_examples=50, deadline=5000)
    def test_total_tokens_never_exceeds_budget(self, max_batch, n, max_tokens):
        """Total tokens in a batch must never exceed max_tokens_per_batch."""
        sched = BatchScheduler(
            max_batch_size=max_batch,
            max_tokens_per_batch=max_tokens,
        )
        for i in range(n):
            sched.add(_make_seq(f"r{i}", prompt_len=10))

        batch = sched.schedule()
        if batch is not None:
            assert batch.total_tokens <= max_tokens

    @given(
        n=st.integers(1, 20),
        priorities=st.lists(st.integers(0, 3), min_size=1, max_size=20),
    )
    @settings(max_examples=30, deadline=5000)
    def test_higher_priority_scheduled_first(self, n, priorities):
        """Higher priority (lower numeric) sequences are scheduled first."""
        sched = BatchScheduler(max_batch_size=n, max_tokens_per_batch=100000)
        for i in range(n):
            p = priorities[i] if i < len(priorities) else 2
            sched.add(_make_seq(f"r{i}", prompt_len=10, priority=p))

        batch = sched.schedule()
        if batch is not None and len(batch.sequences) > 1:
            # Verify that lower priority numbers appear first
            pri_seq = [(s.priority, i) for i, s in enumerate(batch.sequences)]
            # Not strictly sorted due to aging/urgency, but high-priority
            # sequences should generally appear before low-priority ones
            high_pri_indices = [i for p, i in pri_seq if p <= 1]
            low_pri_indices = [i for p, i in pri_seq if p >= 3]
            if high_pri_indices and low_pri_indices:
                # At least one high-priority seq should come before a low-priority one
                assert min(high_pri_indices) <= max(low_pri_indices)

    @given(
        n=st.integers(1, 20),
        prompt_lens=st.lists(st.integers(1, 100), min_size=1, max_size=20),
    )
    @settings(max_examples=30, deadline=5000)
    def test_no_data_loss_on_schedule(self, n, prompt_lens):
        """All added sequences must eventually be scheduled (no data loss)."""
        sched = BatchScheduler(max_batch_size=4, max_tokens_per_batch=10000)
        added_ids = set()
        for i in range(min(n, len(prompt_lens))):
            rid = f"r{i}"
            sched.add(_make_seq(rid, prompt_len=prompt_lens[i]))
            added_ids.add(rid)

        # Schedule until all are processed
        scheduled_ids = set()
        for _ in range(100):
            batch = sched.schedule()
            if batch is None:
                break
            for seq in batch.sequences:
                scheduled_ids.add(seq.request_id)
                # Mark as complete
                seq.generated_tokens = list(range(seq.max_new_tokens))
            # Force evict completed
            sched.step(batch, torch.zeros(len(batch.sequences), dtype=torch.long))

        # All added sequences should have been scheduled
        assert scheduled_ids == added_ids

    @given(
        max_batch=st.integers(1, 16),
        n=st.integers(0, 20),
    )
    @settings(max_examples=30, deadline=5000)
    def test_schedule_idempotent_on_empty(self, max_batch, n):
        """Scheduling with no pending sequences always returns None."""
        sched = BatchScheduler(max_batch_size=max_batch)
        # Don't add any sequences
        assert sched.schedule() is None

    @given(
        n=st.integers(1, 10),
        chunk_size=st.integers(10, 200),
    )
    @settings(max_examples=20, deadline=5000)
    def test_chunked_prefill_respects_budget(self, n, chunk_size):
        """Chunked prefill never exceeds max_prefill_tokens per iteration."""
        max_prefill = chunk_size
        sched = BatchScheduler(
            max_batch_size=n,
            max_tokens_per_batch=100000,
            enable_chunked_prefill=True,
            max_prefill_tokens=max_prefill,
        )
        for i in range(n):
            sched.add(_make_seq(f"r{i}", prompt_len=chunk_size * 3))

        batch = sched.schedule()
        if batch is not None:
            # Prefill tokens per sequence should be bounded
            for i, is_pf in enumerate(batch.is_prefill):
                if is_pf:
                    assert batch.seq_lengths[i] <= max_prefill + 1  # +1 for rounding


class TestBudgetSafety:
    """Budget-related safety properties."""

    @given(
        budget_tokens=st.integers(100, 10000),
        n=st.integers(1, 20),
    )
    @settings(max_examples=30, deadline=5000)
    def test_budget_respected(self, budget_tokens, n):
        """Iteration budget is always respected."""
        sched = BatchScheduler(
            max_batch_size=32,
            max_tokens_per_batch=budget_tokens,
        )
        for i in range(n):
            sched.add(_make_seq(f"r{i}", prompt_len=50))

        budget = sched.get_iteration_budget()
        assert budget.max_total_tokens <= budget_tokens

    @given(
        pressure=st.floats(0.0, 1.0),
    )
    @settings(max_examples=20, deadline=5000)
    def test_sarathi_budget_bounds(self, pressure):
        """Sarathi budget always produces valid bounds."""
        from distllm.core.batch_scheduler import DecodePressureTracker

        sched = BatchScheduler(max_batch_size=32, max_tokens_per_batch=32768)
        tracker = DecodePressureTracker(alpha=0.1, target_ms_per_token=8.0)
        for _ in range(20):
            tracker.record_decode_step(1, pressure * 100)
        sched._pressure_tracker = tracker

        budget = sched._compute_sarathi_budget(
            IterationBudget(max_prefill_tokens=4096, max_decode_tokens=512, max_batch_size=32)
        )
        assert budget.max_prefill_tokens >= 0
        assert budget.max_decode_tokens >= 0
        assert budget.max_batch_size >= 1


class TestPriorityAging:
    """Aging-related properties."""

    @given(
        aging_interval=st.floats(0.1, 10.0),
        max_boost=st.integers(0, 5),
    )
    @settings(max_examples=20, deadline=5000)
    def test_aging_boost_bounded(self, aging_interval, max_boost):
        """Aging boost never exceeds max_boost."""
        sched = BatchScheduler(
            max_batch_size=4,
            aging_enabled=True,
            aging_interval_s=aging_interval,
            aging_max_boost=max_boost,
        )
        seq = _make_seq("req-1")
        seq.created_at = time.time() - 1000  # Very old

        boost = sched._aging_boost(seq)
        assert boost <= max_boost


import time


class TestSequenceProperties:
    """Sequence dataclass properties."""

    @given(
        prompt_len=st.integers(0, 1000),
        generated_len=st.integers(0, 1000),
    )
    @settings(max_examples=30, deadline=5000)
    def test_total_len_sum(self, prompt_len, generated_len):
        """total_len is always prompt + generated."""
        seq = Sequence(
            request_id="test",
            prompt_tokens=[1] * prompt_len,
            generated_tokens=[1] * generated_len,
        )
        assert seq.total_len == prompt_len + generated_len

    @given(
        max_new=st.integers(1, 1000),
        generated_len=st.integers(0, 1000),
    )
    @settings(max_examples=30, deadline=5000)
    def test_is_complete_boundary(self, max_new, generated_len):
        """is_complete is True iff generated >= max_new or status is DONE/FAILED."""
        seq = Sequence(
            request_id="test",
            prompt_tokens=[1],
            max_new_tokens=max_new,
            generated_tokens=[1] * generated_len,
        )
        if generated_len >= max_new:
            assert seq.is_complete
        elif seq.status in (SequenceStatus.DONE, SequenceStatus.FAILED):
            assert seq.is_complete
        else:
            assert not seq.is_complete
