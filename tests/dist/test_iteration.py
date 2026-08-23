"""Tests for distllm.dist.scheduling.iteration -- real objects, zero mocks.

Covers: TenantSLA, TenantBudget, SLATracker, GPUIsolationConfig,
IterationScheduler, and SLASchedulingPolicy.
"""

from __future__ import annotations

import os
import time

import pytest
import torch

from distllm.dist.scheduling.iteration import (
    GPUIsolationConfig,
    IterationScheduler,
    SLASchedulingPolicy,
    SLATracker,
    TenantBudget,
    TenantSLA,
)
from distllm.core.scheduler.budget import IterationBudget
from distllm.core.scheduler.sequence import (
    ScheduledBatch,
    Sequence,
    SequenceStatus,
)


# ---------------------------------------------------------------------------
# TenantSLA
# ---------------------------------------------------------------------------


class TestTenantSLA:
    """TenantSLA dataclass -- defaults and custom values."""

    def test_default_values(self) -> None:
        sla = TenantSLA(tenant_id="prod")
        assert sla.tenant_id == "prod"
        assert sla.target_ttft_ms == 200.0
        assert sla.target_tpot_ms == 50.0
        assert sla.deadline_ms == 5000.0
        assert sla.min_throughput_toks_per_s == 10.0
        assert sla.priority_boost_factor == 1.5

    def test_custom_values(self) -> None:
        sla = TenantSLA(
            tenant_id="prod",
            target_ttft_ms=100.0,
            target_tpot_ms=30.0,
            deadline_ms=2000.0,
            min_throughput_toks_per_s=5.0,
            priority_boost_factor=2.0,
        )
        assert sla.tenant_id == "prod"
        assert sla.target_ttft_ms == 100.0
        assert sla.target_tpot_ms == 30.0
        assert sla.deadline_ms == 2000.0
        assert sla.min_throughput_toks_per_s == 5.0
        assert sla.priority_boost_factor == 2.0


# ---------------------------------------------------------------------------
# TenantBudget
# ---------------------------------------------------------------------------


class TestTenantBudget:
    """TenantBudget rate-limit logic."""

    def test_default_construction(self) -> None:
        budget = TenantBudget(tenant_id="test")
        assert budget.tenant_id == "test"
        assert budget.max_tokens_per_minute == 1000.0
        assert budget.tokens_used == 0.0
        assert not budget._is_throttled

    def test_custom_max_tokens(self) -> None:
        budget = TenantBudget(tenant_id="test", max_tokens_per_minute=5000.0)
        assert budget.max_tokens_per_minute == 5000.0

    def test_can_spend_within_limit(self) -> None:
        budget = TenantBudget(tenant_id="test", max_tokens_per_minute=1000.0)
        assert budget.can_spend(500)

    def test_can_spend_at_limit(self) -> None:
        budget = TenantBudget(tenant_id="test", max_tokens_per_minute=100.0)
        assert budget.can_spend(100)

    def test_cannot_spend_exceed_limit(self) -> None:
        budget = TenantBudget(tenant_id="test", max_tokens_per_minute=100.0)
        budget.spend(100)
        assert not budget.can_spend(1)

    def test_spend_updates_usage(self) -> None:
        budget = TenantBudget(tenant_id="test")
        budget.spend(50)
        assert budget.tokens_used == 50.0

    def test_spend_multiple(self) -> None:
        budget = TenantBudget(tenant_id="test", max_tokens_per_minute=100.0)
        budget.spend(30)
        budget.spend(40)
        budget.spend(20)
        assert budget.tokens_used == 90.0
        assert budget.can_spend(10)
        assert not budget.can_spend(11)

    def test_zero_max_tokens_cannot_spend(self) -> None:
        budget = TenantBudget(tenant_id="test", max_tokens_per_minute=0.0)
        assert not budget.can_spend(1)

    def test_zero_max_tokens_utilization(self) -> None:
        budget = TenantBudget(tenant_id="test", max_tokens_per_minute=0.0)
        assert budget.utilization == 0.0  # avoids ZeroDivisionError

    def test_utilization_partial(self) -> None:
        budget = TenantBudget(tenant_id="test", max_tokens_per_minute=200.0)
        budget.spend(50)
        assert budget.utilization == 0.25

    def test_utilization_full(self) -> None:
        budget = TenantBudget(tenant_id="test", max_tokens_per_minute=100.0)
        budget.spend(100)
        assert budget.utilization == 1.0

    def test_spend_zero_no_change(self) -> None:
        budget = TenantBudget(tenant_id="test")
        budget.spend(0)
        assert budget.tokens_used == 0.0
        assert not budget._is_throttled

    def test_throttled_flag(self) -> None:
        budget = TenantBudget(tenant_id="test", max_tokens_per_minute=50.0)
        budget.spend(50)
        assert budget._is_throttled
        assert not budget.can_spend(1)

    def test_window_reset(self) -> None:
        """Force window reset by setting window_start far in the past."""
        budget = TenantBudget(tenant_id="test", max_tokens_per_minute=100.0)
        budget.spend(100)
        assert budget._is_throttled
        budget.window_start = time.time() - 120.0
        assert budget.can_spend(50)
        assert not budget._is_throttled
        assert budget.tokens_used == 0.0


# ---------------------------------------------------------------------------
# SLATracker
# ---------------------------------------------------------------------------


class TestSLATracker:
    """SLATracker -- request lifecycle, priority boosts, metrics."""

    def test_stats_empty(self) -> None:
        tracker = SLATracker()
        assert tracker.stats() == {"active_requests": 0, "tenants_tracked": 0}

    def test_register_request(self) -> None:
        tracker = SLATracker()
        tracker.register_request("req-1")
        assert tracker.stats()["active_requests"] == 1

    def test_register_request_with_tenant(self) -> None:
        tracker = SLATracker()
        tracker.register_request("req-1", tenant_id="tenant-a")
        assert tracker.stats()["active_requests"] == 1

    def test_complete_request_removes(self) -> None:
        tracker = SLATracker()
        tracker.register_request("req-1")
        tracker.complete_request("req-1")
        assert tracker.stats()["active_requests"] == 0

    def test_complete_request_unknown(self) -> None:
        tracker = SLATracker()
        tracker.complete_request("nonexistent")  # must not raise

    def test_record_first_token(self) -> None:
        tracker = SLATracker()
        tracker.register_request("req-1")
        tracker.record_first_token("req-1")  # must not raise

    def test_record_token(self) -> None:
        tracker = SLATracker()
        tracker.register_request("req-1")
        tracker.record_token("req-1")
        assert tracker.stats()["active_requests"] == 1

    def test_priority_boost_unknown_request(self) -> None:
        tracker = SLATracker()
        assert tracker.get_priority_boost("nonexistent", base_priority=2) == 2

    def test_priority_boost_no_tenant_sla(self) -> None:
        tracker = SLATracker()
        tracker.register_request("req-1")
        assert tracker.get_priority_boost("req-1", base_priority=2) == 2

    def test_priority_boost_sla_different_tenant(self) -> None:
        tracker = SLATracker()
        tracker.set_tenant_sla(TenantSLA(tenant_id="tenant-a"))
        tracker.register_request("req-1")  # no tenant_id -- not linked
        assert tracker.get_priority_boost("req-1", base_priority=2) == 2

    def test_priority_boost_within_sla(self) -> None:
        tracker = SLATracker()
        tracker.set_tenant_sla(
            TenantSLA(tenant_id="tenant-a", target_ttft_ms=100000.0)
        )
        tracker.register_request("req-1", tenant_id="tenant-a")
        # elapsed is near 0, well under target_ttft_ms
        assert tracker.get_priority_boost("req-1", base_priority=2) == 2

    def test_priority_boost_base_zero(self) -> None:
        tracker = SLATracker()
        assert tracker.get_priority_boost("nonexistent", base_priority=0) == 0

    def test_set_tenant_sla(self) -> None:
        tracker = SLATracker()
        sla = TenantSLA(tenant_id="prod", target_ttft_ms=100.0)
        tracker.set_tenant_sla(sla)
        assert tracker.stats()["tenants_tracked"] == 1

    def test_request_metrics_unknown(self) -> None:
        tracker = SLATracker()
        metrics = tracker.get_request_metrics("nonexistent")
        assert metrics["ttft_ms"] is None
        assert metrics["token_count"] == 0

    def test_request_metrics_new(self) -> None:
        tracker = SLATracker()
        tracker.register_request("req-1")
        metrics = tracker.get_request_metrics("req-1")
        assert "ttft_ms" in metrics
        assert "tpot_ms" in metrics
        assert "total_ms" in metrics
        assert metrics["token_count"] == 0

    def test_request_metrics_after_tokens(self) -> None:
        tracker = SLATracker()
        tracker.register_request("req-1")
        tracker.record_first_token("req-1")
        tracker.record_token("req-1")
        metrics = tracker.get_request_metrics("req-1")
        assert metrics["token_count"] == 1

    def test_multiple_requests(self) -> None:
        tracker = SLATracker()
        tracker.register_request("req-1")
        tracker.register_request("req-2")
        assert tracker.stats()["active_requests"] == 2

    def test_multiple_tenants(self) -> None:
        tracker = SLATracker()
        tracker.set_tenant_sla(TenantSLA(tenant_id="tenant-a"))
        tracker.set_tenant_sla(TenantSLA(tenant_id="tenant-b"))
        assert tracker.stats()["tenants_tracked"] == 2

    def test_stats_after_complete(self) -> None:
        tracker = SLATracker()
        tracker.register_request("req-1")
        tracker.register_request("req-2")
        tracker.complete_request("req-1")
        assert tracker.stats()["active_requests"] == 1


# ---------------------------------------------------------------------------
# GPUIsolationConfig
# ---------------------------------------------------------------------------


class TestGPUIsolationConfig:
    """GPUIsolationConfig -- construction, repr, apply."""

    def test_default_config(self) -> None:
        config = GPUIsolationConfig()
        assert config.mode == "none"
        assert config.mig_profile == ""
        assert config.mps_active_thread_percentage == 100
        assert config.gpu_memory_limit_mb == 0

    def test_mps_mode(self) -> None:
        config = GPUIsolationConfig(mode="mps", mps_active_thread_percentage=75)
        assert config.mode == "mps"
        assert config.mps_active_thread_percentage == 75

    def test_mig_mode(self) -> None:
        config = GPUIsolationConfig(mode="mig", mig_profile="1g.10gb")
        assert config.mode == "mig"
        assert config.mig_profile == "1g.10gb"

    def test_memory_limit(self) -> None:
        config = GPUIsolationConfig(gpu_memory_limit_mb=4096)
        assert config.gpu_memory_limit_mb == 4096

    def test_repr_default(self) -> None:
        config = GPUIsolationConfig()
        assert "mode=none" in repr(config)

    def test_repr_mig(self) -> None:
        config = GPUIsolationConfig(mode="mig", mig_profile="1g.10gb")
        parts = repr(config)
        assert "mode=mig" in parts
        assert "mig=1g.10gb" in parts

    def test_repr_memory_limit(self) -> None:
        config = GPUIsolationConfig(gpu_memory_limit_mb=2048)
        assert "mem_limit=2048MB" in repr(config)

    def test_repr_all(self) -> None:
        config = GPUIsolationConfig(
            mode="mig", mig_profile="1g.10gb", gpu_memory_limit_mb=4096
        )
        parts = repr(config)
        assert "mode=mig" in parts
        assert "mig=1g.10gb" in parts
        assert "mem_limit=4096MB" in parts

    def test_apply_none_sets_nothing(self) -> None:
        for key in ["CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY"]:
            os.environ.pop(key, None)
        config = GPUIsolationConfig()
        config.apply()
        assert "CUDA_MPS_PIPE_DIRECTORY" not in os.environ
        assert "CUDA_MPS_LOG_DIRECTORY" not in os.environ

    def test_apply_mps_sets_env(self) -> None:
        config = GPUIsolationConfig(mode="mps", mps_active_thread_percentage=75)
        config.apply()
        assert os.environ.get("CUDA_MPS_PIPE_DIRECTORY") == "/tmp/mps_pipe"
        assert os.environ.get("CUDA_MPS_LOG_DIRECTORY") == "/tmp/mps_log"
        assert os.environ.get("MPS_ACTIVE_THREAD_PERCENTAGE") == "75"
        for key in [
            "CUDA_MPS_PIPE_DIRECTORY",
            "CUDA_MPS_LOG_DIRECTORY",
            "MPS_ACTIVE_THREAD_PERCENTAGE",
        ]:
            os.environ.pop(key, None)

    def test_apply_mps_100_pct_omits_var(self) -> None:
        config = GPUIsolationConfig(mode="mps", mps_active_thread_percentage=100)
        config.apply()
        assert os.environ.get("CUDA_MPS_PIPE_DIRECTORY") == "/tmp/mps_pipe"
        assert "MPS_ACTIVE_THREAD_PERCENTAGE" not in os.environ
        for key in ["CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY"]:
            os.environ.pop(key, None)

    def test_apply_memory_limit_sets_alloc_conf(self) -> None:
        cvd = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        config = GPUIsolationConfig(gpu_memory_limit_mb=4096)
        config.apply()
        assert "PYTORCH_CUDA_ALLOC_CONF" in os.environ
        assert "max_split_size_mb:4096" in os.environ["PYTORCH_CUDA_ALLOC_CONF"]
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        if cvd is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = cvd

    def test_apply_mig_does_not_raise(self) -> None:
        config = GPUIsolationConfig(mode="mig")
        config.apply()  # must not raise


# ---------------------------------------------------------------------------
# IterationScheduler
# ---------------------------------------------------------------------------


class TestIterationScheduler:
    """IterationScheduler -- construction, add, schedule, step, stats."""

    def test_init_defaults(self) -> None:
        scheduler = IterationScheduler()
        assert scheduler.max_batch_size == 32
        assert scheduler.prefill_chunk_size == 256
        assert scheduler.decode_priority is True
        assert scheduler.sla_tracker is not None
        assert scheduler.sla_tracker.stats()["active_requests"] == 0

    def test_init_custom(self) -> None:
        scheduler = IterationScheduler(
            max_batch_size=64,
            max_tokens_per_batch=8192,
            prefill_chunk_size=512,
            decode_priority=False,
        )
        assert scheduler.max_batch_size == 64
        assert scheduler.prefill_chunk_size == 512
        assert scheduler.decode_priority is False

    def test_set_tenant_sla(self) -> None:
        scheduler = IterationScheduler()
        scheduler.set_tenant_sla(TenantSLA(tenant_id="prod", target_ttft_ms=100.0))
        stats = scheduler.stats()
        assert stats["sla"]["tenants_tracked"] == 1

    def test_set_tenant_budget(self) -> None:
        scheduler = IterationScheduler()
        scheduler.set_tenant_budget("tenant-a", max_tokens_per_minute=5000.0)
        stats = scheduler.stats()
        assert "tenant-a" in stats["tenant_budgets"]
        assert stats["tenant_budgets"]["tenant-a"]["throttled"] is False
        assert stats["tenant_budgets"]["tenant-a"]["utilization"] == 0.0

    def test_add_sequence(self) -> None:
        scheduler = IterationScheduler()
        seq = Sequence(request_id="req-1", prompt_tokens=[1, 2, 3])
        scheduler.add(seq)
        assert scheduler.has_pending

    def test_add_sequence_with_tenant(self) -> None:
        scheduler = IterationScheduler()
        seq = Sequence(request_id="req-1", prompt_tokens=[1, 2, 3])
        scheduler.add(seq, tenant_id="prod")
        assert scheduler.has_pending

    def test_add_empty_prompt(self) -> None:
        scheduler = IterationScheduler()
        seq = Sequence(request_id="req-1")
        scheduler.add(seq)
        assert scheduler.has_pending

    def test_schedule_no_work(self) -> None:
        scheduler = IterationScheduler()
        assert scheduler.schedule() is None

    def test_schedule_with_pending(self) -> None:
        scheduler = IterationScheduler()
        seq = Sequence(request_id="req-1", prompt_tokens=[1, 2, 3])
        scheduler.add(seq)
        batch = scheduler.schedule()
        assert batch is not None
        assert isinstance(batch, ScheduledBatch)
        assert len(batch.sequences) == 1
        assert batch.sequences[0].request_id == "req-1"

    def test_schedule_moves_to_active(self) -> None:
        scheduler = IterationScheduler()
        seq = Sequence(request_id="req-1", prompt_tokens=[1, 2, 3])
        scheduler.add(seq)
        scheduler.schedule()
        assert "req-1" in scheduler.active
        assert scheduler.active["req-1"].status == SequenceStatus.PREFILLING

    def test_schedule_batch_has_input_ids(self) -> None:
        scheduler = IterationScheduler()
        seq = Sequence(request_id="req-1", prompt_tokens=[1, 2, 3])
        scheduler.add(seq)
        batch = scheduler.schedule()
        assert batch is not None
        assert isinstance(batch.input_ids, torch.Tensor)
        assert batch.input_ids.numel() == 3

    def test_schedule_priority_unchanged_without_sla(self) -> None:
        scheduler = IterationScheduler()
        seq = Sequence(request_id="req-1", prompt_tokens=[1, 2, 3], priority=2)
        scheduler.add(seq)
        scheduler.schedule()
        assert scheduler.active["req-1"].priority == 2

    def test_step_generates_token(self) -> None:
        scheduler = IterationScheduler()
        seq = Sequence(request_id="req-1", prompt_tokens=[1, 2, 3])
        scheduler.add(seq)
        batch = scheduler.schedule()
        assert batch is not None
        scheduler.step(batch, torch.tensor([42]))
        assert seq.generated_tokens == [42]

    def test_step_transitions_to_decoding(self) -> None:
        scheduler = IterationScheduler()
        seq = Sequence(request_id="req-1", prompt_tokens=[1, 2, 3])
        scheduler.add(seq)
        batch = scheduler.schedule()
        assert batch is not None
        scheduler.step(batch, torch.tensor([42]))
        assert seq.status == SequenceStatus.DECODING

    def test_step_completes_short_sequence(self) -> None:
        scheduler = IterationScheduler()
        seq = Sequence(
            request_id="req-1", prompt_tokens=[1, 2, 3], max_new_tokens=1
        )
        scheduler.add(seq)
        batch = scheduler.schedule()
        assert batch is not None
        scheduler.step(batch, torch.tensor([42]))
        assert seq.is_complete
        assert len(seq.generated_tokens) == 1

    def test_step_keeps_long_sequence_alive(self) -> None:
        scheduler = IterationScheduler()
        seq = Sequence(
            request_id="req-1", prompt_tokens=[1, 2, 3], max_new_tokens=10
        )
        scheduler.add(seq)
        batch = scheduler.schedule()
        assert batch is not None
        scheduler.step(batch, torch.tensor([42]))
        assert not seq.is_complete
        assert seq.status == SequenceStatus.DECODING

    def test_step_multiple_sequences(self) -> None:
        scheduler = IterationScheduler()
        seq1 = Sequence(request_id="req-1", prompt_tokens=[1, 2, 3])
        seq2 = Sequence(request_id="req-2", prompt_tokens=[4, 5, 6])
        scheduler.add(seq1)
        scheduler.add(seq2)
        batch = scheduler.schedule()
        assert batch is not None
        assert len(batch.sequences) == 2
        scheduler.step(batch, torch.tensor([42, 99]))
        assert seq1.generated_tokens == [42]
        assert seq2.generated_tokens == [99]

    def test_schedule_removes_completed(self) -> None:
        scheduler = IterationScheduler()
        seq = Sequence(
            request_id="req-1", prompt_tokens=[1, 2, 3], max_new_tokens=1
        )
        scheduler.add(seq)
        batch = scheduler.schedule()
        assert batch is not None
        scheduler.step(batch, torch.tensor([42]))
        # Sequence should have been removed from active by parent step()
        assert "req-1" not in scheduler.active

    def test_sla_tracking_during_step(self) -> None:
        scheduler = IterationScheduler()
        seq = Sequence(request_id="req-1", prompt_tokens=[1, 2, 3])
        scheduler.add(seq)
        batch = scheduler.schedule()
        assert batch is not None
        scheduler.step(batch, torch.tensor([42]))
        metrics = scheduler.get_sla_metrics("req-1")
        assert metrics["token_count"] == 1

    def test_get_sla_metrics(self) -> None:
        scheduler = IterationScheduler()
        scheduler.add(Sequence(request_id="req-1", prompt_tokens=[1, 2, 3]))
        metrics = scheduler.get_sla_metrics("req-1")
        assert "ttft_ms" in metrics
        assert metrics["token_count"] == 0

    def test_get_sla_metrics_unknown(self) -> None:
        scheduler = IterationScheduler()
        metrics = scheduler.get_sla_metrics("nonexistent")
        assert metrics["ttft_ms"] is None

    def test_on_before_schedule(self) -> None:
        scheduler = IterationScheduler()
        seq = Sequence(request_id="req-1")
        result = scheduler.on_before_schedule([seq])
        assert len(result) == 1
        assert result[0].request_id == "req-1"

    def test_stats_structure(self) -> None:
        scheduler = IterationScheduler()
        stats = scheduler.stats()
        assert "active_requests" in stats
        assert "pending_requests" in stats
        assert "prefill_chunk_size" in stats
        assert "decode_priority" in stats
        assert "sla" in stats
        assert "tenant_budgets" in stats
        assert stats["prefill_chunk_size"] == 256
        assert stats["decode_priority"] is True

    def test_stats_with_tenant(self) -> None:
        scheduler = IterationScheduler()
        scheduler.set_tenant_sla(TenantSLA(tenant_id="prod"))
        scheduler.set_tenant_budget("prod", max_tokens_per_minute=1000.0)
        stats = scheduler.stats()
        assert stats["sla"]["tenants_tracked"] == 1
        assert "prod" in stats["tenant_budgets"]

    def test_stats_after_work(self) -> None:
        scheduler = IterationScheduler()
        scheduler.add(Sequence(request_id="req-1", prompt_tokens=[1, 2, 3]))
        stats_before = scheduler.stats()
        assert stats_before["pending_requests"] == 1
        scheduler.schedule()
        stats_after = scheduler.stats()
        assert stats_after["active_requests"] == 1


# ---------------------------------------------------------------------------
# SLASchedulingPolicy
# ---------------------------------------------------------------------------


class TestSLASchedulingPolicy:
    """Standalone SLA scheduling policy."""

    def test_init(self) -> None:
        policy = SLASchedulingPolicy()
        stats = policy.stats()
        assert stats["sla"]["active_requests"] == 0
        assert stats["sla"]["tenants_tracked"] == 0

    def test_set_tenant_sla(self) -> None:
        policy = SLASchedulingPolicy()
        policy.set_tenant_sla(TenantSLA(tenant_id="prod"))
        assert policy.stats()["sla"]["tenants_tracked"] == 1

    def test_set_tenant_budget(self) -> None:
        policy = SLASchedulingPolicy()
        policy.set_tenant_budget("tenant-a", max_tokens_per_minute=2000.0)
        stats = policy.stats()
        assert "tenant-a" in stats["tenant_budgets"]
        assert not stats["tenant_budgets"]["tenant-a"]["throttled"]

    def test_compute_budget_returns_same(self) -> None:
        policy = SLASchedulingPolicy()
        budget = IterationBudget(max_prefill_tokens=1024)
        result = policy.compute_budget(budget)
        assert result is budget

    def test_register_request(self) -> None:
        policy = SLASchedulingPolicy()
        policy.register_request("req-1", tenant_id="prod")
        assert policy.stats()["sla"]["active_requests"] == 1

    def test_register_request_no_tenant(self) -> None:
        policy = SLASchedulingPolicy()
        policy.register_request("req-1")
        assert policy.stats()["sla"]["active_requests"] == 1

    def test_complete_request(self) -> None:
        policy = SLASchedulingPolicy()
        policy.register_request("req-1")
        policy.complete_request("req-1")
        assert policy.stats()["sla"]["active_requests"] == 0

    def test_get_request_metrics(self) -> None:
        policy = SLASchedulingPolicy()
        policy.register_request("req-1")
        metrics = policy.get_request_metrics("req-1")
        assert "ttft_ms" in metrics
        assert metrics["token_count"] == 0

    def test_get_request_metrics_unknown(self) -> None:
        policy = SLASchedulingPolicy()
        metrics = policy.get_request_metrics("nonexistent")
        assert metrics["ttft_ms"] is None

    def test_on_before_schedule_returns_sequences(self) -> None:
        policy = SLASchedulingPolicy()
        seq1 = Sequence(request_id="req-1")
        seq2 = Sequence(request_id="req-2")
        result = policy.on_before_schedule([seq1, seq2])
        assert len(result) == 2
        assert result[0].request_id == "req-1"
        assert result[1].request_id == "req-2"

    def test_stats_structure(self) -> None:
        policy = SLASchedulingPolicy()
        stats = policy.stats()
        assert "sla" in stats
        assert "tenant_budgets" in stats
        assert stats["sla"]["active_requests"] == 0

    def test_stats_after_work(self) -> None:
        policy = SLASchedulingPolicy()
        policy.register_request("req-1", tenant_id="prod")
        policy.set_tenant_sla(TenantSLA(tenant_id="prod"))
        policy.set_tenant_budget("prod")
        stats = policy.stats()
        assert stats["sla"]["active_requests"] == 1
        assert stats["sla"]["tenants_tracked"] == 1
        assert "prod" in stats["tenant_budgets"]
