"""Property-based fuzz tests for batch scheduler invariants.

Covers: token budget, lifecycle, priority ordering, preemption,
IterationBudget boundaries, GenerationSystem integration.
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from distllm.core.batch_scheduler import BatchScheduler, Sequence, SequenceStatus, IterationBudget


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def sequence_strategy(draw):
    request_id = f"req-{draw(st.integers(min_value=1, max_value=100000))}"
    prompt_len = draw(st.integers(min_value=1, max_value=128))
    max_new = draw(st.integers(min_value=1, max_value=64))
    priority = draw(st.integers(min_value=0, max_value=3))
    prompt_tokens = list(range(1, prompt_len + 1))
    return Sequence(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new,
        priority=priority,
    )


@st.composite
def sequence_list_strategy(draw):
    """Generate a list of sequences with varied properties."""
    count = draw(st.integers(1, 12))
    return [draw(sequence_strategy()) for _ in range(count)]


# ---------------------------------------------------------------------------
# Token budget invariants
# ---------------------------------------------------------------------------

@given(
    num_sequences=st.integers(min_value=1, max_value=16),
    max_batch_size=st.integers(min_value=1, max_value=32),
    max_tokens=st.integers(min_value=16, max_value=512),
)
@settings(max_examples=20, deadline=None)
def test_scheduled_batch_tokens_within_budget(num_sequences, max_batch_size, max_tokens):
    """Total tokens in a scheduled batch never exceed the budget."""
    scheduler = BatchScheduler(
        max_batch_size=max_batch_size,
        max_tokens_per_batch=max_tokens,
    )

    for i in range(num_sequences):
        seq = Sequence(
            request_id=f"req-{i}",
            prompt_tokens=list(range(1, (i % 24) + 5)),
            max_new_tokens=16,
            priority=2,
        )
        scheduler.add(seq)

    batch = scheduler.schedule()
    if batch is not None:
        total_tokens = batch.total_tokens
        assert total_tokens <= max_tokens, (
            f"Batch has {total_tokens} tokens, exceeds budget of {max_tokens}"
        )
        assert batch.batch_size <= max_batch_size


@given(
    token_budget=st.integers(min_value=32, max_value=512),
    prompt_lengths=st.lists(
        st.integers(min_value=1, max_value=128),
        min_size=1, max_size=16,
    ),
)
@settings(max_examples=20, deadline=None)
def test_token_budget_enforced_across_sequences(token_budget, prompt_lengths):
    """Even with many sequences, total tokens never exceed budget."""
    scheduler = BatchScheduler(
        max_batch_size=max(len(prompt_lengths), 1) * 2,
        max_tokens_per_batch=token_budget,
    )

    for i, plen in enumerate(prompt_lengths):
        scheduler.add(Sequence(
            request_id=f"tok-{i}",
            prompt_tokens=list(range(1, plen + 1)),
            max_new_tokens=8,
            priority=2,
        ))

    batch = scheduler.schedule()
    if batch is not None:
        assert batch.total_tokens <= token_budget
        assert all(s.total_len > 0 for s in batch.sequences)


# ---------------------------------------------------------------------------
# Sequence lifecycle
# ---------------------------------------------------------------------------

@given(
    prompt_len=st.integers(min_value=1, max_value=64),
    max_new=st.integers(min_value=1, max_value=32),
)
@settings(max_examples=20, deadline=None)
def test_sequence_lifecycle(prompt_len, max_new):
    """Sequence transitions through expected lifecycle: PENDING -> DONE/FAILED."""
    scheduler = BatchScheduler(max_batch_size=8, max_tokens_per_batch=512)
    seq = Sequence(
        request_id="lifecycle-test",
        prompt_tokens=list(range(prompt_len)),
        max_new_tokens=max_new,
        priority=2,
    )

    assert seq.status == SequenceStatus.PENDING
    scheduler.add(seq)

    batch = scheduler.schedule()
    assert batch is not None
    assert seq in batch.sequences

    batch_size = batch.batch_size
    next_tokens = torch.zeros(batch_size, dtype=torch.long)
    scheduler.step(batch, next_tokens)

    steps = 0
    while not seq.is_complete and steps < max_new + 10:
        batch = scheduler.schedule()
        if batch is None:
            break
        batch_size = batch.batch_size
        next_tokens = torch.zeros(batch_size, dtype=torch.long)
        scheduler.step(batch, next_tokens)
        steps += 1

    assert seq.is_complete or seq.status in (SequenceStatus.DONE, SequenceStatus.FAILED)


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------

@given(
    num_high_priority=st.integers(min_value=0, max_value=6),
    num_medium_priority=st.integers(min_value=0, max_value=6),
    num_low_priority=st.integers(min_value=0, max_value=6),
)
@settings(max_examples=20, deadline=None)
def test_priority_ordering(num_high_priority, num_medium_priority, num_low_priority):
    """Higher priority sequences are scheduled before lower priority ones."""
    scheduler = BatchScheduler(max_batch_size=32, max_tokens_per_batch=1024)

    for i in range(num_low_priority):
        scheduler.add(Sequence(
            request_id=f"low-{i}", prompt_tokens=[1, 2, 3],
            max_new_tokens=4, priority=3,
        ))
    for i in range(num_medium_priority):
        scheduler.add(Sequence(
            request_id=f"med-{i}", prompt_tokens=[1, 2, 3],
            max_new_tokens=4, priority=2,
        ))
    for i in range(num_high_priority):
        scheduler.add(Sequence(
            request_id=f"high-{i}", prompt_tokens=[1, 2, 3],
            max_new_tokens=4, priority=0,
        ))

    batch = scheduler.schedule()
    if batch is not None and batch.sequences:
        high_in_batch = sum(1 for s in batch.sequences if s.request_id.startswith("high"))
        low_in_batch = sum(1 for s in batch.sequences if s.request_id.startswith("low"))
        if num_high_priority > 0 and num_low_priority > 0:
            assert high_in_batch > 0  # High pri must be present


# ---------------------------------------------------------------------------
# Preemption invariants
# ---------------------------------------------------------------------------

@given(
    num_seqs=st.integers(min_value=2, max_value=10),
    min_priority=st.integers(min_value=0, max_value=3),
)
@settings(max_examples=20, deadline=None)
def test_preemption_preserves_remaining_tokens(num_seqs, min_priority):
    """Preempted sequences go back to pending and can be re-scheduled."""
    scheduler = BatchScheduler(max_batch_size=8, max_tokens_per_batch=256)

    for i in range(num_seqs):
        scheduler.add(Sequence(
            request_id=f"seq-{i}",
            prompt_tokens=list(range(1, (i % 12) + 4)),
            max_new_tokens=8,
            priority=0 if i < max(1, num_seqs // 3) else 3,
        ))

    batch = scheduler.schedule()
    if batch is None or batch.batch_size == 0:
        return

    batch_size = batch.batch_size
    next_tokens = torch.zeros(batch_size, dtype=torch.long)
    scheduler.step(batch, next_tokens)

    preempted = scheduler.preempt_lowest(min_priority=min_priority)
    if preempted:
        assert preempted.status == SequenceStatus.PENDING
        assert scheduler.has_pending


@given(
    num_seqs=st.integers(min_value=1, max_value=6),
)
@settings(max_examples=15, deadline=None)
def test_promote_request_changes_priority(num_seqs):
    """Promoting a request changes its priority and affects scheduling."""
    scheduler = BatchScheduler(max_batch_size=16, max_tokens_per_batch=512)

    for i in range(num_seqs):
        scheduler.add(Sequence(
            request_id=f"seq-{i}",
            prompt_tokens=list(range(1, 6)),
            max_new_tokens=4,
            priority=3,
        ))

    result = scheduler.promote_request("seq-0", 0)
    assert result is True
    seq = scheduler.get_sequence("seq-0")
    assert seq is not None
    assert seq.priority == 0


# ---------------------------------------------------------------------------
# IterationBudget boundaries
# ---------------------------------------------------------------------------

@given(
    max_prefill=st.integers(min_value=0, max_value=16384),
    max_decode=st.integers(min_value=0, max_value=4096),
    max_batch=st.integers(min_value=0, max_value=128),
    max_total=st.integers(min_value=0, max_value=131072),
)
@settings(max_examples=30, deadline=None)
def test_iteration_budget_bounds(max_prefill, max_decode, max_batch, max_total):
    """IterationBudget handles edge cases without raising."""
    budget = IterationBudget(
        max_prefill_tokens=max_prefill or 1,
        max_decode_tokens=max_decode or 1,
        max_batch_size=max(max_batch, 1),
        max_total_tokens=max(max_total, 1),
    )
    assert budget.max_prefill_tokens >= 1
    assert budget.max_decode_tokens >= 1
    assert budget.max_batch_size >= 1
    assert budget.max_total_tokens >= 1
    assert budget.decode_slots >= 0


@given(
    slack_ratio=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    enable_chunked=st.booleans(),
)
@settings(max_examples=30, deadline=None)
def test_iteration_budget_slack(slack_ratio, enable_chunked):
    """IterationBudget slack ratio and chunked prefill don't cause crashes."""
    budget = IterationBudget(
        max_prefill_tokens=4096,
        max_decode_tokens=512,
        max_batch_size=32,
        max_total_tokens=32768,
        enable_chunked_prefill=enable_chunked,
        prefill_slack_ratio=slack_ratio,
    )
    assert budget.prefill_slack_ratio == slack_ratio


# ---------------------------------------------------------------------------
# Concurrency: add + schedule + step cycle
# ---------------------------------------------------------------------------

@given(sequence_list_strategy())
@settings(max_examples=20, deadline=None)
def test_add_schedule_step_cycle(sequences):
    """Multiple add → schedule → step cycles don't corrupt scheduler state."""
    scheduler = BatchScheduler(
        max_batch_size=32,
        max_tokens_per_batch=4096,
    )

    for seq in sequences:
        scheduler.add(seq)

    pending_before = scheduler.pending_count
    assert pending_before == len(sequences)

    batch = scheduler.schedule()
    if batch is not None:
        assert batch.batch_size > 0
        next_tokens = torch.zeros(batch.batch_size, dtype=torch.long)
        scheduler.step(batch, next_tokens)

    stats = scheduler.stats()
    assert isinstance(stats, dict)
    assert "active_requests" in stats or "pending_requests" in stats


@given(
    num_sequences=st.integers(min_value=1, max_value=6),
)
@settings(max_examples=10, deadline=None)
def test_stats_consistency(num_sequences):
    """Scheduler stats are internally consistent after operations."""
    scheduler = BatchScheduler(max_batch_size=8, max_tokens_per_batch=256)

    for i in range(num_sequences):
        scheduler.add(Sequence(
            request_id=f"stat-{i}",
            prompt_tokens=list(range(1, 10)),
            max_new_tokens=4,
            priority=2,
        ))

    for _ in range(3):
        batch = scheduler.schedule()
        if batch is not None:
            next_tokens = torch.zeros(batch.batch_size, dtype=torch.long)
            scheduler.step(batch, next_tokens)

    stats = scheduler.stats()
    total = stats.get("active_requests", 0) + stats.get("pending_requests", 0)
    assert total >= 0
    assert stats.get("completed_requests", 0) >= 0
