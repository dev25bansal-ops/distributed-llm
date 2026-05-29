"""Tests for IterationScheduler — tenant SLA tracking and GPU isolation.

Run: pytest tests/core/test_iteration_scheduler.py -v
"""

import time

import pytest
import torch

from distllm.core.batch_scheduler import (
    BatchScheduler,
    ScheduledBatch,
    Sequence,
    SequenceStatus,
)
from distllm.dist.scheduling.iteration import (
    IterationScheduler,
    TenantSLA,
    TenantBudget,
    GPUIsolationConfig,
    SLASchedulingPolicy,
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
# IterationScheduler Tests
# ===================================================================


class TestIterationScheduler:
    def test_init(self):
        sched = IterationScheduler(max_batch_size=8, max_tokens_per_batch=4096)
        assert sched.max_batch_size == 8
        assert sched.prefill_chunk_size == 256
        assert sched.decode_priority is True

    def test_schedule_returns_scheduled_batch(self):
        """schedule() returns ScheduledBatch, not torch.Tensor."""
        sched = IterationScheduler(max_batch_size=4, max_tokens_per_batch=4096)
        sched.add(_make_seq("req-1", prompt_len=50))
        result = sched.schedule()
        assert result is not None
        assert isinstance(result, ScheduledBatch)

    def test_schedule_empty_returns_none(self):
        sched = IterationScheduler(max_batch_size=4)
        assert sched.schedule() is None

    def test_add_with_tenant_id(self):
        sched = IterationScheduler(max_batch_size=4)
        seq = _make_seq("req-1")
        sched.add(seq, tenant_id="tenant-a")
        assert "req-1" in sched._seq_tenants
        assert sched._seq_tenants["req-1"] == "tenant-a"

    def test_set_tenant_sla(self):
        sched = IterationScheduler(max_batch_size=4)
        sla = TenantSLA(tenant_id="prod", target_ttft_ms=100, deadline_ms=2000)
        sched.set_tenant_sla(sla)
        assert "prod" in sched.sla_tracker._tenant_slas

    def test_set_tenant_budget(self):
        sched = IterationScheduler(max_batch_size=4)
        sched.set_tenant_budget("prod", max_tokens_per_minute=50000)
        assert "prod" in sched._tenant_budgets
        assert sched._tenant_budgets["prod"].max_tokens_per_minute == 50000

    def test_schedule_with_sla_boost(self):
        """SLA boosts are applied to pending requests."""
        sched = IterationScheduler(max_batch_size=4)
        sla = TenantSLA(tenant_id="prod", target_ttft_ms=100, deadline_ms=2000)
        sched.set_tenant_sla(sla)

        seq = _make_seq("req-1", priority=3)
        sched.add(seq, tenant_id="prod")

        # Simulate SLA violation
        sched.sla_tracker._request_start_times["req-1"] = time.time() - 10  # 10s ago
        sched.sla_tracker._request_first_token_at["req-1"] = None

        batch = sched.schedule()
        assert batch is not None

    def test_step_records_sla_metrics(self):
        """step() records SLA metrics for sequences."""
        sched = IterationScheduler(max_batch_size=4)
        seq = _make_seq("req-1", prompt_len=50, max_new=5)
        sched.add(seq, tenant_id="prod")

        batch = sched.schedule()
        assert batch is not None

        sched.step(batch, torch.tensor([42] * len(batch.sequences)))
        metrics = sched.get_sla_metrics("req-1")
        assert "token_count" in metrics

    def test_stats_includes_sla(self):
        sched = IterationScheduler(max_batch_size=4)
        stats = sched.stats()
        assert "sla" in stats
        assert "tenant_budgets" in stats
        assert "prefill_chunk_size" in stats

    def test_on_before_schedule(self):
        """on_before_schedule is compatible with SchedulingPolicy protocol."""
        sched = IterationScheduler(max_batch_size=4)
        sla = TenantSLA(tenant_id="prod", target_ttft_ms=100)
        sched.set_tenant_sla(sla)

        seq = _make_seq("req-1", priority=3)
        sched.add(seq, tenant_id="prod")

        # on_before_schedule should return sequences
        result = sched.on_before_schedule([seq])
        assert len(result) == 1


# ===================================================================
# TenantBudget Tests
# ===================================================================


class TestTenantBudget:
    def test_can_spend_within_budget(self):
        budget = TenantBudget(tenant_id="t1", max_tokens_per_minute=1000)
        assert budget.can_spend(500)

    def test_cannot_spend_over_budget(self):
        budget = TenantBudget(tenant_id="t1", max_tokens_per_minute=1000)
        budget.spend(800)
        assert not budget.can_spend(300)

    def test_spend_throttles_at_limit(self):
        budget = TenantBudget(tenant_id="t1", max_tokens_per_minute=1000)
        budget.spend(1000)
        assert budget._is_throttled
        assert not budget.can_spend(1)

    def test_utilization(self):
        budget = TenantBudget(tenant_id="t1", max_tokens_per_minute=1000)
        budget.spend(500)
        assert budget.utilization == 0.5


# ===================================================================
# TenantSLA Tests
# ===================================================================


class TestTenantSLA:
    def test_default_values(self):
        sla = TenantSLA(tenant_id="t1")
        assert sla.target_ttft_ms == 200.0
        assert sla.target_tpot_ms == 50.0
        assert sla.deadline_ms == 5000.0


# ===================================================================
# GPUIsolationConfig Tests
# ===================================================================


class TestGPUIsolationConfig:
    def test_default_mode(self):
        config = GPUIsolationConfig()
        assert config.mode == "none"

    def test_mps_mode(self):
        config = GPUIsolationConfig(mode="mps", mps_active_thread_percentage=50)
        assert config.mode == "mps"
        assert config.mps_active_thread_percentage == 50

    def test_repr(self):
        config = GPUIsolationConfig(mode="mig", mig_profile="1g.5gb")
        r = repr(config)
        assert "mig" in r
        assert "1g.5gb" in r


# ===================================================================
# SLASchedulingPolicy Tests
# ===================================================================


class TestSLASchedulingPolicy:
    def test_init(self):
        policy = SLASchedulingPolicy()
        assert policy.stats()["sla"]["active_requests"] == 0

    def test_set_tenant_sla(self):
        policy = SLASchedulingPolicy()
        sla = TenantSLA(tenant_id="prod", target_ttft_ms=100)
        policy.set_tenant_sla(sla)
        assert "prod" in policy._sla_tracker._tenant_slas

    def test_set_tenant_budget(self):
        policy = SLASchedulingPolicy()
        policy.set_tenant_budget("prod", max_tokens_per_minute=50000)
        assert "prod" in policy._tenant_budgets

    def test_compute_budget_passthrough(self):
        from distllm.core.batch_scheduler import IterationBudget
        policy = SLASchedulingPolicy()
        budget = IterationBudget(max_prefill_tokens=4096)
        result = policy.compute_budget(budget)
        assert result.max_prefill_tokens == 4096

    def test_register_and_complete_request(self):
        policy = SLASchedulingPolicy()
        policy.register_request("req-1", tenant_id="prod")
        assert policy.stats()["sla"]["active_requests"] == 1

        policy.complete_request("req-1")
        assert policy.stats()["sla"]["active_requests"] == 0

    def test_stats(self):
        policy = SLASchedulingPolicy()
        policy.register_request("req-1", tenant_id="prod")
        stats = policy.stats()
        assert "sla" in stats
        assert "tenant_budgets" in stats
