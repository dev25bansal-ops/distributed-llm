"""Tests for SchedulingPolicy protocol and implementations.

Tests:
- SchedulingPolicy protocol compliance
- DefaultPolicy passthrough
- SarathiPolicy pressure adaptation
- CompositePolicy chaining
- Policy switching

Run: pytest tests/core/test_scheduler_policy.py -v
"""

import pytest
import torch

from distllm.core.batch_scheduler import (
    BatchScheduler,
    DecodePressureTracker,
    IterationBudget,
    ScheduledBatch,
    Sequence,
)
from distllm.core.advanced_scheduling import (
    SchedulingPolicy,
    DefaultPolicy,
    SarathiPolicy,
    CompositePolicy,
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
# SchedulingPolicy Protocol Tests
# ===================================================================


class TestSchedulingPolicyProtocol:
    def test_default_policy_is_protocol(self):
        policy = DefaultPolicy()
        assert isinstance(policy, SchedulingPolicy)

    def test_sarathi_policy_is_protocol(self):
        policy = SarathiPolicy()
        assert isinstance(policy, SchedulingPolicy)

    def test_composite_policy_is_protocol(self):
        policy = CompositePolicy()
        assert isinstance(policy, SchedulingPolicy)


# ===================================================================
# DefaultPolicy Tests
# ===================================================================


class TestDefaultPolicy:
    def test_compute_budget_passthrough(self):
        policy = DefaultPolicy()
        budget = IterationBudget(max_prefill_tokens=4096, max_batch_size=32)
        result = policy.compute_budget(budget)
        assert result.max_prefill_tokens == 4096
        assert result.max_batch_size == 32

    def test_on_before_schedule_passthrough(self):
        policy = DefaultPolicy()
        seqs = [_make_seq("a"), _make_seq("b")]
        result = policy.on_before_schedule(seqs)
        assert len(result) == 2


# ===================================================================
# SarathiPolicy Tests
# ===================================================================


class TestSarathiPolicy:
    def test_no_pressure_tracker_passthrough(self):
        policy = SarathiPolicy()
        budget = IterationBudget(max_prefill_tokens=4096)
        result = policy.compute_budget(budget)
        assert result.max_prefill_tokens == 4096

    def test_low_pressure_increases_prefill(self):
        tracker = DecodePressureTracker(alpha=0.1, target_ms_per_token=8.0)
        # Low pressure: per_token << target
        for _ in range(20):
            tracker.record_decode_step(1, 2.0)  # 2ms << 8ms target

        policy = SarathiPolicy(pressure_tracker=tracker)
        budget = IterationBudget(
            max_prefill_tokens=4096,
            max_decode_tokens=512,
            max_batch_size=32,
            max_total_tokens=32768,
        )
        result = policy.compute_budget(budget)
        # Low pressure should relax decode slots
        assert result.max_decode_tokens <= 512

    def test_high_pressure_throttles_prefill(self):
        tracker = DecodePressureTracker(alpha=0.1, target_ms_per_token=8.0)
        # High pressure: per_token >> target
        for _ in range(20):
            tracker.record_decode_step(1, 50.0)  # 50ms >> 8ms target

        policy = SarathiPolicy(pressure_tracker=tracker)
        budget = IterationBudget(
            max_prefill_tokens=4096,
            max_decode_tokens=512,
            max_batch_size=32,
            max_total_tokens=32768,
        )
        result = policy.compute_budget(budget)
        # High pressure should reduce prefill
        assert result.max_prefill_tokens < 4096

    def test_severe_pressure_limits_batch(self):
        tracker = DecodePressureTracker(alpha=0.1, target_ms_per_token=8.0)
        for _ in range(20):
            tracker.record_decode_step(1, 100.0)  # Very high pressure

        policy = SarathiPolicy(pressure_tracker=tracker)
        budget = IterationBudget(
            max_prefill_tokens=4096,
            max_decode_tokens=512,
            max_batch_size=32,
            max_total_tokens=32768,
        )
        result = policy.compute_budget(budget)
        assert result.max_batch_size <= 32

    def test_on_before_schedule_passthrough(self):
        policy = SarathiPolicy()
        seqs = [_make_seq("a")]
        result = policy.on_before_schedule(seqs)
        assert len(result) == 1


# ===================================================================
# CompositePolicy Tests
# ===================================================================


class TestCompositePolicy:
    def test_empty_composite(self):
        policy = CompositePolicy()
        budget = IterationBudget(max_prefill_tokens=4096)
        result = policy.compute_budget(budget)
        assert result.max_prefill_tokens == 4096

    def test_single_policy_composite(self):
        default = DefaultPolicy()
        policy = CompositePolicy(policies=[default])
        budget = IterationBudget(max_prefill_tokens=4096)
        result = policy.compute_budget(budget)
        assert result.max_prefill_tokens == 4096

    def test_chained_policies(self):
        """Policies are applied in order."""
        tracker = DecodePressureTracker(alpha=0.1, target_ms_per_token=8.0)
        for _ in range(20):
            tracker.record_decode_step(1, 50.0)  # High pressure

        sarathi = SarathiPolicy(pressure_tracker=tracker)
        default = DefaultPolicy()

        policy = CompositePolicy(policies=[default, sarathi])
        budget = IterationBudget(
            max_prefill_tokens=4096,
            max_decode_tokens=512,
            max_batch_size=32,
            max_total_tokens=32768,
        )
        result = policy.compute_budget(budget)
        # Sarathi should have reduced prefill
        assert result.max_prefill_tokens < 4096

    def test_on_before_schedule_chained(self):
        policy1 = DefaultPolicy()
        policy2 = DefaultPolicy()
        composite = CompositePolicy(policies=[policy1, policy2])

        seqs = [_make_seq("a")]
        result = composite.on_before_schedule(seqs)
        assert len(result) == 1


# ===================================================================
# Policy Integration with BatchScheduler Tests
# ===================================================================


class TestPolicyIntegration:
    def test_set_scheduling_policy(self):
        sched = BatchScheduler(max_batch_size=4)
        policy = DefaultPolicy()
        sched.set_scheduling_policy(policy)
        assert sched._scheduling_policy is policy

    def test_set_scheduling_policy_none(self):
        sched = BatchScheduler(max_batch_size=4)
        sched.set_scheduling_policy(DefaultPolicy())
        sched.set_scheduling_policy(None)
        assert sched._scheduling_policy is None

    def test_schedule_uses_policy(self):
        """When a policy is set, it's used for budget computation."""
        sched = BatchScheduler(max_batch_size=4, max_tokens_per_batch=4096)
        policy = DefaultPolicy()
        sched.set_scheduling_policy(policy)

        sched.add(_make_seq("req-1", prompt_len=50))
        batch = sched.schedule()
        assert batch is not None

    def test_sarathi_skipped_when_policy_set(self):
        """When a policy is set, Sarathi-Serve is skipped."""
        sched = BatchScheduler(
            max_batch_size=4,
            max_tokens_per_batch=4096,
            enable_chunked_prefill=True,
        )
        # Set a custom policy
        sched.set_scheduling_policy(DefaultPolicy())
        # Disable Sarathi
        sched._adapt_prefill_budget = False

        sched.add(_make_seq("req-1", prompt_len=50))
        batch = sched.schedule()
        assert batch is not None

    def test_wan_overrides_sarathi(self):
        """WAN mode disables Sarathi pressure adaptation."""
        sched = BatchScheduler(max_batch_size=4, max_tokens_per_batch=4096)
        sched.set_wan_mode(enabled=True, rtt_threshold_ms=10)
        sched.set_node_capabilities({
            "wan": _make_node("wan", latency_ms=50),
        })

        # Sarathi should be disabled
        budget = IterationBudget(max_prefill_tokens=4096, max_decode_tokens=512)
        result = sched._compute_sarathi_budget(budget)
        assert result.max_prefill_tokens == 4096  # Unchanged


def _make_node(node_id: str, latency_ms: float = 1.0):
    from distllm.core.advanced_scheduling import NodeCapabilityInfo, DeviceClass
    return NodeCapabilityInfo(
        node_id=node_id,
        gpu_name="RTX-4090",
        device_class=DeviceClass.MID_RANGE_GPU,
        total_memory_bytes=24 * 1024**3,
        free_memory_bytes=16 * 1024**3,
        measured_latency_ms=latency_ms,
    )
