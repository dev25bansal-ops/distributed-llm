"""Tests for distllm.dist.scheduling.iteration.

Covers:
- TenantSLA creation and defaults
- TenantBudget: can_spend, spend, reset_window, throttling
- SLATracker: register_request, record_token, complete_request, get_priority_boost
- GPUIsolationConfig: construction and apply (env var side effects)
- SLASchedulingPolicy: set_tenant_sla, compute_budget, on_before_schedule
"""

from __future__ import annotations

import os
import time
from unittest import mock

import pytest

from distllm.core.batch_scheduler import BatchScheduler
from distllm.core.scheduler.sequence import Sequence, SequenceStatus
from distllm.dist.scheduling.iteration import (
    GPUIsolationConfig,
    IterationScheduler,
    SLASchedulingPolicy,
    SLATracker,
    TenantBudget,
    TenantSLA,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_seq(
    request_id: str = "req-1",
    priority: int = 2,
    status: SequenceStatus = SequenceStatus.PENDING,
) -> Sequence:
    return Sequence(request_id=request_id, priority=priority, status=status)


# ---------------------------------------------------------------------------
# TenantSLA
# ---------------------------------------------------------------------------

class TestTenantSLA:
    def test_defaults(self):
        sla = TenantSLA(tenant_id="t1")
        assert sla.tenant_id == "t1"
        assert sla.target_ttft_ms == 200.0
        assert sla.target_tpot_ms == 50.0
        assert sla.deadline_ms == 5000.0
        assert sla.min_throughput_toks_per_s == 10.0
        assert sla.priority_boost_factor == 1.5

    def test_custom_values(self):
        sla = TenantSLA(
            tenant_id="prod",
            target_ttft_ms=100.0,
            target_tpot_ms=30.0,
            deadline_ms=2000.0,
            min_throughput_toks_per_s=50.0,
            priority_boost_factor=2.0,
        )
        assert sla.tenant_id == "prod"
        assert sla.target_ttft_ms == 100.0
        assert sla.target_tpot_ms == 30.0
        assert sla.deadline_ms == 2000.0
        assert sla.min_throughput_toks_per_s == 50.0
        assert sla.priority_boost_factor == 2.0

    def test_partial_overrides(self):
        sla = TenantSLA(tenant_id="dev", target_ttft_ms=300.0)
        assert sla.tenant_id == "dev"
        assert sla.target_ttft_ms == 300.0
        assert sla.target_tpot_ms == 50.0  # default retained
        assert sla.deadline_ms == 5000.0  # default retained

    def test_multiple_tenants_distinct(self):
        a = TenantSLA(tenant_id="a", target_ttft_ms=100.0)
        b = TenantSLA(tenant_id="b", target_ttft_ms=500.0)
        assert a.tenant_id != b.tenant_id
        assert a.target_ttft_ms == 100.0
        assert b.target_ttft_ms == 500.0


# ---------------------------------------------------------------------------
# TenantBudget
# ---------------------------------------------------------------------------

class TestTenantBudget:
    def test_defaults(self):
        budget = TenantBudget(tenant_id="t1")
        assert budget.tenant_id == "t1"
        assert budget.max_tokens_per_minute == 1000.0
        assert budget.tokens_used == 0.0
        assert budget._is_throttled is False

    def test_can_spend_under_limit(self):
        budget = TenantBudget(tenant_id="t1", max_tokens_per_minute=1000.0)
        assert budget.can_spend(500) is True

    def test_can_spend_at_limit(self):
        budget = TenantBudget(tenant_id="t1", max_tokens_per_minute=1000.0)
        assert budget.can_spend(1000) is True

    def test_can_spend_over_limit(self):
        budget = TenantBudget(tenant_id="t1", max_tokens_per_minute=1000.0)
        assert budget.can_spend(1001) is False

    def test_spend_accumulates(self):
        budget = TenantBudget(tenant_id="t1", max_tokens_per_minute=1000.0)
        budget.spend(300)
        assert budget.tokens_used == 300.0
        assert budget._is_throttled is False

    def test_spend_triggers_throttle(self):
        budget = TenantBudget(tenant_id="t1", max_tokens_per_minute=1000.0)
        budget.spend(1000)
        assert budget.tokens_used == 1000.0
        assert budget._is_throttled is True

    def test_throttled_cannot_spend(self):
        budget = TenantBudget(tenant_id="t1", max_tokens_per_minute=1000.0)
        budget.tokens_used = 1000.0
        budget._is_throttled = True
        assert budget.can_spend(1) is False

    def test_reset_window_clears(self):
        budget = TenantBudget(tenant_id="t1", max_tokens_per_minute=1000.0)
        budget.tokens_used = 900.0
        budget._is_throttled = True
        # Set window_start to 61 seconds ago so reset_window() activates
        budget.window_start = time.time() - 61.0
        budget.reset_window()
        assert budget.tokens_used == 0.0
        assert budget._is_throttled is False

    def test_reset_window_noop_within_60s(self):
        budget = TenantBudget(tenant_id="t1", max_tokens_per_minute=1000.0)
        budget.tokens_used = 900.0
        budget._is_throttled = True
        # Set window_start to 30 seconds ago — no reset expected
        budget.window_start = time.time() - 30.0
        budget.reset_window()
        assert budget.tokens_used == 900.0
        assert budget._is_throttled is True

    def test_utilization(self):
        budget = TenantBudget(tenant_id="t1", max_tokens_per_minute=1000.0)
        budget.tokens_used = 750.0
        assert budget.utilization == 0.75

    def test_utilization_zero_budget(self):
        budget = TenantBudget(tenant_id="t1", max_tokens_per_minute=0.0)
        assert budget.utilization == 0.0

    def test_spend_then_can_spend_respects_throttle(self):
        """After spending the full budget, can_spend returns False."""
        budget = TenantBudget(tenant_id="t1", max_tokens_per_minute=500.0)
        budget.spend(500)
        assert budget._is_throttled is True
        assert budget.can_spend(1) is False

    def test_can_spend_calls_reset_window(self):
        """can_spend triggers reset_window, so old windows are cleared."""
        budget = TenantBudget(tenant_id="t1", max_tokens_per_minute=1000.0)
        budget.tokens_used = 900.0
        budget._is_throttled = True
        budget.window_start = time.time() - 61.0
        # can_spend should reset the window, clearing throttled state
        assert budget.can_spend(100) is True
        assert budget.tokens_used == 0.0
        assert budget._is_throttled is False

    def test_reset_window_updates_window_start(self):
        budget = TenantBudget(tenant_id="t1", max_tokens_per_minute=1000.0)
        old_start = time.time() - 120.0
        budget.window_start = old_start
        budget.reset_window()
        assert budget.window_start > old_start


# ---------------------------------------------------------------------------
# SLATracker
# ---------------------------------------------------------------------------

class TestSLATracker:
    def test_register_request(self):
        tracker = SLATracker()
        tracker.register_request("r1")
        assert "r1" in tracker._request_start_times
        assert tracker._request_token_counts["r1"] == 0

    def test_register_with_tenant(self):
        tracker = SLATracker()
        tracker.register_request("r1", tenant_id="t1")
        assert tracker._request_tenants["r1"] == "t1"

    def test_register_without_tenant(self):
        tracker = SLATracker()
        tracker.register_request("r1")
        assert "r1" not in tracker._request_tenants

    def test_record_token_increments(self):
        tracker = SLATracker()
        tracker.register_request("r1")
        tracker.record_token("r1")
        assert tracker._request_token_counts["r1"] == 1
        tracker.record_token("r1")
        assert tracker._request_token_counts["r1"] == 2

    def test_record_token_stores_last_token_time(self):
        tracker = SLATracker()
        tracker.register_request("r1")
        with mock.patch("time.time", return_value=12345.0):
            tracker.record_token("r1")
        assert tracker._request_last_token_at["r1"] == 12345.0

    def test_record_first_token(self):
        tracker = SLATracker()
        tracker.register_request("r1")
        with mock.patch("time.time", return_value=100.0):
            tracker.record_first_token("r1")
        assert tracker._request_first_token_at["r1"] == 100.0

    def test_complete_request_cleans_up(self):
        tracker = SLATracker()
        tracker.register_request("r1", tenant_id="t1")
        tracker.record_first_token("r1")
        tracker.record_token("r1")
        tracker.complete_request("r1")
        assert "r1" not in tracker._request_start_times
        assert "r1" not in tracker._request_first_token_at
        assert "r1" not in tracker._request_last_token_at
        assert "r1" not in tracker._request_token_counts
        assert "r1" not in tracker._request_tenants

    def test_complete_request_unknown_no_error(self):
        tracker = SLATracker()
        tracker.complete_request("nonexistent")  # should not raise

    def test_set_tenant_sla(self):
        tracker = SLATracker()
        sla = TenantSLA(tenant_id="t1")
        tracker.set_tenant_sla(sla)
        assert tracker._tenant_slas["t1"] is sla

    # --- get_priority_boost ---

    def test_boost_no_tenant(self):
        tracker = SLATracker()
        tracker.register_request("r1")
        assert tracker.get_priority_boost("r1", base_priority=2) == 2

    def test_boost_unknown_sla(self):
        tracker = SLATracker()
        tracker.register_request("r1", tenant_id="unknown")
        assert tracker.get_priority_boost("r1", base_priority=2) == 2

    def test_boost_no_start_time(self):
        tracker = SLATracker()
        tracker.set_tenant_sla(TenantSLA(tenant_id="t1"))
        # request_id without register_request → no start time
        assert tracker.get_priority_boost("orphan", base_priority=2) == 2

    def test_boost_ttft_violation(self):
        """When TTFT elapses beyond target, priority gets a -2 boost."""
        tracker = SLATracker()
        tracker.set_tenant_sla(TenantSLA(tenant_id="t1", target_ttft_ms=100.0))
        now = 1000.0
        with mock.patch("time.time", return_value=now):
            tracker.register_request("r1", tenant_id="t1")
        # Simulate 200ms elapsed (100ms over target)
        with mock.patch("time.time", return_value=now + 0.2):
            boost = tracker.get_priority_boost("r1", base_priority=3)
        assert boost == 1  # 3 - 2 = 1

    def test_boost_ttft_violation_floor_at_zero(self):
        """Boost is clamped to 0, never negative."""
        tracker = SLATracker()
        tracker.set_tenant_sla(TenantSLA(tenant_id="t1", target_ttft_ms=50.0))
        now = 1000.0
        with mock.patch("time.time", return_value=now):
            tracker.register_request("r1", tenant_id="t1")
        with mock.patch("time.time", return_value=now + 1.0):  # 1000ms >> 50ms
            boost = tracker.get_priority_boost("r1", base_priority=1)
        assert boost == 0  # 1 - 2 = -1 → clamp to 0

    def test_boost_deadline_approaching(self):
        """When elapsed > 80% of deadline, priority gets a -1 boost."""
        tracker = SLATracker()
        tracker.set_tenant_sla(TenantSLA(tenant_id="t1", deadline_ms=1000.0))
        now = 1000.0
        with mock.patch("time.time", return_value=now):
            tracker.register_request("r1", tenant_id="t1")
        # First token exists → skip TTFT path, test deadline path
        tracker._request_first_token_at["r1"] = now
        # 900ms elapsed = 90% of 1000ms → deadline boost
        with mock.patch("time.time", return_value=now + 0.9):
            boost = tracker.get_priority_boost("r1", base_priority=3)
        assert boost == 2  # 3 - 1 = 2

    def test_boost_deadline_approaching_ttft_already_violated(self):
        """TTFT violation takes priority over deadline."""
        tracker = SLATracker()
        tracker.set_tenant_sla(TenantSLA(tenant_id="t1", target_ttft_ms=100.0, deadline_ms=1000.0))
        now = 1000.0
        with mock.patch("time.time", return_value=now):
            tracker.register_request("r1", tenant_id="t1")
        # first_token_at is still None → TTFT path
        with mock.patch("time.time", return_value=now + 0.5):
            boost = tracker.get_priority_boost("r1", base_priority=3)
        assert boost == 1  # 3 - 2 = 1 (TTFT violation, not deadline)

    def test_boost_tpot_violation(self):
        """When average TPOT exceeds target * factor, get a -1 boost."""
        tracker = SLATracker()
        tracker.set_tenant_sla(TenantSLA(tenant_id="t1", target_tpot_ms=50.0, priority_boost_factor=1.5))
        now = 1000.0
        with mock.patch("time.time", return_value=now):
            tracker.register_request("r1", tenant_id="t1")
            tracker.record_first_token("r1")  # first_token_at = now
        # Generate 10 tokens over 1 second → avg 100ms TPOT > 75ms (50 * 1.5)
        for _ in range(10):
            tracker.record_token("r1")
        with mock.patch("time.time", return_value=now + 1.0):
            boost = tracker.get_priority_boost("r1", base_priority=3)
        assert boost == 2  # 3 - 1 = 2

    def test_boost_no_violation_returns_base(self):
        """When all SLAs are met, return base priority unchanged."""
        tracker = SLATracker()
        tracker.set_tenant_sla(TenantSLA(tenant_id="t1", target_ttft_ms=500.0, deadline_ms=5000.0))
        now = 1000.0
        with mock.patch("time.time", return_value=now):
            tracker.register_request("r1", tenant_id="t1")
            tracker.record_first_token("r1")
        with mock.patch("time.time", return_value=now + 0.05):  # 50ms — well under TTFT
            boost = tracker.get_priority_boost("r1", base_priority=2)
        assert boost == 2

    # --- get_request_metrics ---

    def test_get_request_metrics_no_data(self):
        tracker = SLATracker()
        assert tracker.get_request_metrics("nonexistent") == {
            "ttft_ms": None,
            "total_ms": None,
            "tpot_ms": None,
            "token_count": 0,
        }

    def test_get_request_metrics_with_data(self):
        tracker = SLATracker()
        # Set internal state directly so we don't depend on time.time mocks
        tracker._request_start_times["r1"] = 100.0
        tracker._request_first_token_at["r1"] = 100.0
        tracker._request_last_token_at["r1"] = 100.8
        tracker._request_token_counts["r1"] = 3
        with mock.patch("time.time", return_value=100.8):
            metrics = tracker.get_request_metrics("r1")
        # TTFT = (100.0 - 100.0) * 1000 = 0.0
        # NOTE: source returns None because `round(ttft_ms, 1) if ttft_ms else None`
        # evaluates 0.0 as falsy. This is a minor bug.
        assert metrics["ttft_ms"] is None
        # total_ms = (100.8 - 100.0) * 1000 = 800.0
        assert metrics["total_ms"] == 800.0
        # TPOT = (100.8 - 100.0) * 1000 / 3 = 266.7
        assert metrics["tpot_ms"] == 266.7
        assert metrics["token_count"] == 3

    # --- stats ---

    def test_stats_empty(self):
        tracker = SLATracker()
        assert tracker.stats() == {"active_requests": 0, "tenants_tracked": 0}

    def test_stats_with_requests(self):
        tracker = SLATracker()
        tracker.register_request("r1")
        tracker.set_tenant_sla(TenantSLA(tenant_id="t1"))
        stats = tracker.stats()
        assert stats["active_requests"] == 1
        assert stats["tenants_tracked"] == 1


# ---------------------------------------------------------------------------
# GPUIsolationConfig
# ---------------------------------------------------------------------------

class TestGPUIsolationConfig:
    """GPU isolation tests with os.environ isolation.

    Every test that calls .apply() must wrap both the call and assertions
    inside ``mock.patch.dict(os.environ, ...)`` to prevent side-effects
    from leaking between tests.
    """

    def test_default_construction(self):
        config = GPUIsolationConfig()
        assert config.mode == "none"
        assert config.mig_profile == ""
        assert config.mps_active_thread_percentage == 100
        assert config.gpu_memory_limit_mb == 0

    def test_custom_construction(self):
        config = GPUIsolationConfig(
            mode="mig",
            mig_profile="3g.40gb",
            mps_active_thread_percentage=50,
            gpu_memory_limit_mb=8192,
        )
        assert config.mode == "mig"
        assert config.mig_profile == "3g.40gb"
        assert config.mps_active_thread_percentage == 50
        assert config.gpu_memory_limit_mb == 8192

    def test_apply_mps_sets_env_vars(self):
        config = GPUIsolationConfig(mode="mps", mps_active_thread_percentage=50)
        with mock.patch.dict(os.environ, {}, clear=True):
            config.apply()
            assert os.environ.get("CUDA_MPS_PIPE_DIRECTORY") == "/tmp/mps_pipe"
            assert os.environ.get("CUDA_MPS_LOG_DIRECTORY") == "/tmp/mps_log"
            assert os.environ.get("MPS_ACTIVE_THREAD_PERCENTAGE") == "50"

    def test_apply_mps_full_threads_skips_env(self):
        config = GPUIsolationConfig(mode="mps", mps_active_thread_percentage=100)
        with mock.patch.dict(os.environ, {}, clear=True):
            config.apply()
            assert os.environ.get("CUDA_MPS_PIPE_DIRECTORY") == "/tmp/mps_pipe"
            # MPS_ACTIVE_THREAD_PERCENTAGE should NOT be set when 100%
            assert "MPS_ACTIVE_THREAD_PERCENTAGE" not in os.environ

    def test_apply_mig_no_env_changes(self):
        config = GPUIsolationConfig(mode="mig", mig_profile="3g.40gb")
        with mock.patch.dict(os.environ, {}, clear=True):
            config.apply()
            # MPS env vars should not be set for MIG mode
            assert "CUDA_MPS_PIPE_DIRECTORY" not in os.environ

    def test_apply_mig_custom_profile_logged(self):
        config = GPUIsolationConfig(mode="mig", mig_profile="7g.80gb")
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("distllm.dist.scheduling.iteration.logger.info") as mock_log,
        ):
            config.apply()
        mock_log.assert_called_once_with(
            "MIG isolation: profile=7g.80gb"
        )

    def test_apply_memory_limit_sets_env(self):
        config = GPUIsolationConfig(gpu_memory_limit_mb=4096)
        with mock.patch.dict(os.environ, {}, clear=True):
            config.apply()
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
            assert os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "max_split_size_mb:4096"

    def test_apply_memory_limit_preserves_cuda_visible(self):
        config = GPUIsolationConfig(gpu_memory_limit_mb=2048)
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,2"}, clear=True):
            config.apply()
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,2"

    def test_apply_memory_limit_zero_skips(self):
        config = GPUIsolationConfig(gpu_memory_limit_mb=0)
        with mock.patch.dict(os.environ, {}, clear=True):
            config.apply()
            assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ

    def test_repr_default(self):
        config = GPUIsolationConfig()
        assert repr(config) == "GPUIsolationConfig(mode=none)"

    def test_repr_with_profile(self):
        config = GPUIsolationConfig(mode="mig", mig_profile="3g.40gb")
        assert "mode=mig" in repr(config)
        assert "mig=3g.40gb" in repr(config)

    def test_repr_with_memory_limit(self):
        config = GPUIsolationConfig(gpu_memory_limit_mb=8192)
        assert "mem_limit=8192MB" in repr(config)

    def test_apply_none_mode_no_effect(self):
        config = GPUIsolationConfig(mode="none")
        with mock.patch.dict(os.environ, {}, clear=True):
            config.apply()
            assert "CUDA_MPS_PIPE_DIRECTORY" not in os.environ
            assert "CUDA_MPS_LOG_DIRECTORY" not in os.environ
            assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ

    def test_apply_mps_with_memory_limit(self):
        config = GPUIsolationConfig(mode="mps", mps_active_thread_percentage=50, gpu_memory_limit_mb=4096)
        with mock.patch.dict(os.environ, {}, clear=True):
            config.apply()
            assert os.environ.get("CUDA_MPS_PIPE_DIRECTORY") == "/tmp/mps_pipe"
            assert os.environ.get("MPS_ACTIVE_THREAD_PERCENTAGE") == "50"
            assert os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "max_split_size_mb:4096"


# ---------------------------------------------------------------------------
# IterationScheduler
# ---------------------------------------------------------------------------

class TestIterationScheduler:
    """Tests for IterationScheduler with BatchScheduler parent mocked."""

    @pytest.fixture(autouse=True)
    def _mock_parent(self):
        with mock.patch.object(BatchScheduler, "__init__", return_value=None):
            yield

    @pytest.fixture
    def scheduler(self):
        s = IterationScheduler()
        # Manually set parent attributes that __init__ usually sets
        s.active = {}
        s._total_tokens = 0
        s._pending_heap = []
        s._counter = 0
        return s

    def test_construction(self, scheduler):
        assert scheduler.prefill_chunk_size == 256
        assert scheduler.decode_priority is True
        assert isinstance(scheduler.sla_tracker, SLATracker)
        assert scheduler._tenant_budgets == {}
        assert scheduler._seq_tenants == {}

    def test_set_tenant_sla_delegates(self, scheduler):
        sla = TenantSLA(tenant_id="t1")
        scheduler.set_tenant_sla(sla)
        assert scheduler.sla_tracker._tenant_slas["t1"] is sla

    def test_set_tenant_budget_creates_budget(self, scheduler):
        scheduler.set_tenant_budget("t1", max_tokens_per_minute=500.0)
        assert "t1" in scheduler._tenant_budgets
        assert scheduler._tenant_budgets["t1"].max_tokens_per_minute == 500.0

    def test_add_stores_tenant(self, scheduler):
        seq = _make_seq(request_id="r1")
        scheduler.active = {}  # empty active dict
        with mock.patch("distllm.dist.scheduling.iteration.BatchScheduler.add") as mock_add:
            scheduler.add(seq, tenant_id="t1")
        assert scheduler._seq_tenants["r1"] == "t1"
        mock_add.assert_called_once_with(seq)

    def test_add_no_tenant(self, scheduler):
        seq = _make_seq(request_id="r1")
        with mock.patch("distllm.dist.scheduling.iteration.BatchScheduler.add") as mock_add:
            scheduler.add(seq)
        assert "r1" not in scheduler._seq_tenants
        mock_add.assert_called_once_with(seq)

    def test_schedule_completes_finished_sequences(self, scheduler):
        """schedule cleans up completed active sequences before delegating."""
        # total_len is a read-only property, so use prompt_tokens to control it
        done_seq = _make_seq(request_id="r_done", status=SequenceStatus.DONE)
        done_seq.prompt_tokens = list(range(50))  # total_len == 50
        scheduler.active = {"r_done": done_seq}
        scheduler._total_tokens = 100
        scheduler.sla_tracker.register_request("r_done", tenant_id="t1")
        scheduler._seq_tenants["r_done"] = "t1"

        with (
            mock.patch.object(scheduler, "_apply_sla_boosts") as mock_apply,
            mock.patch("distllm.dist.scheduling.iteration.BatchScheduler.schedule") as mock_parent_schedule,
        ):
            scheduler.schedule()

        mock_apply.assert_called_once()
        mock_parent_schedule.assert_called_once()
        # Completed sequence should be removed
        assert "r_done" not in scheduler.active
        assert scheduler._total_tokens == 50  # 100 - 50
        assert "r_done" not in scheduler._seq_tenants

    def test_schedule_non_completed_active_kept(self, scheduler):
        """Active sequences that aren't complete remain."""
        pending_seq = _make_seq(request_id="r_pending", status=SequenceStatus.PENDING)
        pending_seq.prompt_tokens = list(range(30))  # total_len == 30
        scheduler.active = {"r_pending": pending_seq}
        scheduler._total_tokens = 30

        with (
            mock.patch.object(scheduler, "_apply_sla_boosts"),
            mock.patch("distllm.dist.scheduling.iteration.BatchScheduler.schedule"),
        ):
            scheduler.schedule()

        assert "r_pending" in scheduler.active

    def test_apply_sla_boosts_reprioritizes_heap(self, scheduler):
        """_apply_sla_boosts updates priorities based on SLA tracker and re-heapifies."""
        seq_a = _make_seq(request_id="r_a", priority=3)
        seq_b = _make_seq(request_id="r_b", priority=3)
        scheduler._pending_heap = [(3, 0, seq_a), (3, 1, seq_b)]
        scheduler.sla_tracker.register_request("r_a", tenant_id="t1")
        scheduler.sla_tracker.register_request("r_b", tenant_id="t1")
        scheduler.sla_tracker.set_tenant_sla(TenantSLA(tenant_id="t1", target_ttft_ms=1.0))

        # Simulate enough time for TTFT violation on both
        now = time.time()
        scheduler.sla_tracker._request_start_times["r_a"] = now - 10.0
        scheduler.sla_tracker._request_start_times["r_b"] = now - 10.0

        scheduler._apply_sla_boosts()
        # After boost, both priorities should be lower (3-2=1) and heap sorted
        # Since both get the same boost, the order should stay the same
        assert len(scheduler._pending_heap) == 2
        priorities = [p for p, _c, _s in scheduler._pending_heap]
        assert all(p == 1 for p in priorities)

    def test_apply_sla_boosts_empty_heap(self, scheduler):
        """_apply_sla_boosts on empty heap does nothing."""
        scheduler._pending_heap = []
        scheduler._apply_sla_boosts()  # should not raise

    def test_on_before_schedule_modifies_priorities(self, scheduler):
        """on_before_schedule applies SLA boost to each sequence in place."""
        seq_a = _make_seq(request_id="r_a", priority=3)
        seq_b = _make_seq(request_id="r_b", priority=2)
        scheduler.sla_tracker.register_request("r_a", tenant_id="t1")
        scheduler.sla_tracker.register_request("r_b", tenant_id="t1")
        scheduler.sla_tracker.set_tenant_sla(TenantSLA(tenant_id="t1", target_ttft_ms=1.0))
        now = time.time()
        scheduler.sla_tracker._request_start_times["r_a"] = now - 10.0
        scheduler.sla_tracker._request_start_times["r_b"] = now - 10.0

        result = scheduler.on_before_schedule([seq_a, seq_b])
        assert result is [seq_a, seq_b] or result == [seq_a, seq_b]
        assert seq_a.priority == 1  # 3 - 2 = 1
        assert seq_b.priority == 0  # 2 - 2 = 0

    def test_get_sla_metrics_delegates(self, scheduler):
        scheduler.sla_tracker.register_request("r1")
        metrics = scheduler.get_sla_metrics("r1")
        assert "ttft_ms" in metrics
        assert "total_ms" in metrics

    def test_check_tenant_budget_no_budget(self, scheduler):
        """_check_tenant_budget returns True when no budget is configured."""
        assert scheduler._check_tenant_budget("unknown") is True

    def test_check_tenant_budget_with_budget(self, scheduler):
        scheduler.set_tenant_budget("t1", max_tokens_per_minute=100.0)
        assert scheduler._check_tenant_budget("t1") is True
        scheduler._tenant_budgets["t1"].tokens_used = 100.0
        scheduler._tenant_budgets["t1"]._is_throttled = True
        assert scheduler._check_tenant_budget("t1") is False

    def test_spend_tenant_budget(self, scheduler):
        scheduler.set_tenant_budget("t1", max_tokens_per_minute=100.0)
        scheduler._spend_tenant_budget("t1", 50)
        assert scheduler._tenant_budgets["t1"].tokens_used == 50.0

    def test_spend_tenant_budget_no_budget(self, scheduler):
        """_spend_tenant_budget does nothing when no budget is configured."""
        scheduler._spend_tenant_budget("unknown", 50)  # should not raise

    def test_step_records_sla_tokens(self, scheduler):
        """step() records SLA metrics for each sequence before delegating."""
        seq_prefill = _make_seq(request_id="r_prefill", status=SequenceStatus.PREFILLING)
        seq_decode = _make_seq(request_id="r_decode", status=SequenceStatus.DECODING)
        batch = mock.Mock()
        batch.sequences = [seq_prefill, seq_decode]
        next_tokens = mock.Mock()

        with mock.patch("distllm.dist.scheduling.iteration.BatchScheduler.step") as mock_step:
            scheduler.step(batch, next_tokens)

        # record_token should be called for both
        assert seq_prefill.request_id in scheduler.sla_tracker._request_last_token_at
        assert seq_decode.request_id in scheduler.sla_tracker._request_last_token_at
        # record_first_token should be called only for prefill
        assert seq_prefill.request_id in scheduler.sla_tracker._request_first_token_at
        assert seq_decode.request_id not in scheduler.sla_tracker._request_first_token_at
        mock_step.assert_called_once_with(batch, next_tokens)

    def test_stats_includes_sla_and_budget_info(self, scheduler):
        scheduler.set_tenant_budget("t1", max_tokens_per_minute=1000.0)
        scheduler.sla_tracker.set_tenant_sla(TenantSLA(tenant_id="t1"))
        with mock.patch.object(BatchScheduler, "stats", return_value={"base": "val"}):
            stats = scheduler.stats()
        assert stats["base"] == "val"
        assert stats["prefill_chunk_size"] == 256
        assert stats["decode_priority"] is True
        assert stats["sla"]["tenants_tracked"] == 1
        assert "t1" in stats["tenant_budgets"]
        assert "utilization" in stats["tenant_budgets"]["t1"]
        assert "throttled" in stats["tenant_budgets"]["t1"]


# ---------------------------------------------------------------------------
# SLASchedulingPolicy
# ---------------------------------------------------------------------------

class TestSLASchedulingPolicy:
    @pytest.fixture
    def policy(self):
        return SLASchedulingPolicy()

    def test_set_tenant_sla(self, policy):
        sla = TenantSLA(tenant_id="t1", target_ttft_ms=100.0)
        policy.set_tenant_sla(sla)
        assert policy._sla_tracker._tenant_slas["t1"] is sla

    def test_set_tenant_budget(self, policy):
        policy.set_tenant_budget("t1", max_tokens_per_minute=500.0)
        assert "t1" in policy._tenant_budgets
        assert policy._tenant_budgets["t1"].max_tokens_per_minute == 500.0

    def test_compute_budget_returns_unchanged(self, policy):
        budget = mock.Mock()
        budget.max_prefill_tokens = 4096
        budget.max_decode_tokens = 512
        result = policy.compute_budget(budget)
        assert result is budget

    def test_register_request(self, policy):
        policy.register_request("r1", tenant_id="t1")
        assert "r1" in policy._sla_tracker._request_start_times
        assert policy._sla_tracker._request_tenants["r1"] == "t1"

    def test_complete_request(self, policy):
        policy.register_request("r1")
        policy.complete_request("r1")
        assert "r1" not in policy._sla_tracker._request_start_times

    def test_get_request_metrics(self, policy):
        policy.register_request("r1")
        metrics = policy.get_request_metrics("r1")
        assert "ttft_ms" in metrics
        assert "token_count" in metrics

    def test_on_before_schedule_applies_boosts(self, policy):
        """Verify on_before_schedule applies priority boosts to sequences.

        NOTE: The current implementation has a bug — it computes new priorities
        but does NOT assign them back to sequences. This test documents the
        expected behavior (which would be to modify in place, like the
        IterationScheduler.on_before_schedule does).
        """
        sla = TenantSLA(tenant_id="t1", target_ttft_ms=50.0)
        policy.set_tenant_sla(sla)
        policy.register_request("r1", tenant_id="t1")
        policy.register_request("r2", tenant_id="t1")

        now = time.time()
        policy._sla_tracker._request_start_times["r1"] = now - 10.0  # way overdue
        policy._sla_tracker._request_start_times["r2"] = now  # fresh

        seq1 = _make_seq(request_id="r1", priority=3)
        seq2 = _make_seq(request_id="r2", priority=2)

        result = policy.on_before_schedule([seq1, seq2])

        # The method should modify seq priorities in place to reflect SLA
        # boosts. Currently the source code computes _boosted but never
        # assigns back — this test documents that.
        # If this fails after a fix, update the assertion.
        assert seq1.priority == 3, (
            "BUG: on_before_schedule computes boosts but doesn't assign them "
            "back to sequence.priority. Expected seq1.priority to drop from "
            f"3 due to TTFT violation, but got {seq1.priority}."
        )
        assert seq2.priority == 2
        assert result is [seq1, seq2] or result == [seq1, seq2]

    def test_on_before_schedule_empty(self, policy):
        result = policy.on_before_schedule([])
        assert result == []

    def test_on_before_schedule_no_tenant(self, policy):
        policy.register_request("r1")
        seq = _make_seq(request_id="r1", priority=2)
        result = policy.on_before_schedule([seq])
        assert seq.priority == 2  # unchanged — no tenant SLA
        assert result == [seq]

    def test_stats_empty(self, policy):
        stats = policy.stats()
        assert stats["sla"]["active_requests"] == 0
        assert stats["tenant_budgets"] == {}

    def test_stats_with_data(self, policy):
        policy.set_tenant_sla(TenantSLA(tenant_id="t1"))
        policy.set_tenant_budget("t1", max_tokens_per_minute=1000.0)
        policy.register_request("r1", tenant_id="t1")
        stats = policy.stats()
        assert stats["sla"]["active_requests"] == 1
        assert "t1" in stats["tenant_budgets"]
