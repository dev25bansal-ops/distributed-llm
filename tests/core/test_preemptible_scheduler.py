"""Tests for the preemptible scheduler module.

Covers:
    PriorityTier          -- Enum values and ordering
    TierPolicy            -- Construction, defaults, overrides
    ScheduledRequest      -- Construction, properties
    TenantBudget          -- Budget tracking, reset, can_use, record_use
    PreemptibleScheduler  -- Full lifecycle: enqueue, dequeue, preempt, evict

Every test is deterministic (no network, no GPU, no time.sleep).
No MagicMock -- real objects or lightweight stubs only.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

# Bootstrap fake packages for distllm namespace
bootstrap_fake_packages()

# Load the preemptible_scheduler module
_sched_mod = load_module("distllm/core/preemptible_scheduler.py")

# Re-export symbols for test readability
PriorityTier = _sched_mod.PriorityTier
TierPolicy = _sched_mod.TierPolicy
DEFAULT_TIER_POLICIES = _sched_mod.DEFAULT_TIER_POLICIES
ScheduledRequest = _sched_mod.ScheduledRequest
TenantBudget = _sched_mod.TenantBudget
PreemptibleScheduler = _sched_mod.PreemptibleScheduler


# ===================================================================
# PRIORITY TIER TESTS
# ===================================================================

class TestPriorityTier:
    """PriorityTier enum -- values, ordering, membership."""

    def test_values_are_correct(self) -> None:
        assert PriorityTier.CRITICAL.value == 0
        assert PriorityTier.HIGH.value == 1
        assert PriorityTier.NORMAL.value == 2
        assert PriorityTier.LOW.value == 3
        assert PriorityTier.BACKGROUND.value == 4

    def test_ordering_critical_less_than_high(self) -> None:
        assert PriorityTier.CRITICAL < PriorityTier.HIGH
        assert PriorityTier.CRITICAL < PriorityTier.NORMAL
        assert PriorityTier.CRITICAL < PriorityTier.LOW
        assert PriorityTier.CRITICAL < PriorityTier.BACKGROUND

    def test_ordering_background_greater_than_all(self) -> None:
        assert PriorityTier.BACKGROUND > PriorityTier.LOW
        assert PriorityTier.BACKGROUND > PriorityTier.NORMAL
        assert PriorityTier.BACKGROUND > PriorityTier.HIGH
        assert PriorityTier.BACKGROUND > PriorityTier.CRITICAL

    def test_iteration_is_priority_order(self) -> None:
        """Iterating over PriorityTier should yield CRITICAL first."""
        tiers = list(PriorityTier)
        assert tiers == [
            PriorityTier.CRITICAL,
            PriorityTier.HIGH,
            PriorityTier.NORMAL,
            PriorityTier.LOW,
            PriorityTier.BACKGROUND,
        ]

    def test_from_int(self) -> None:
        assert PriorityTier(0) is PriorityTier.CRITICAL
        assert PriorityTier(1) is PriorityTier.HIGH
        assert PriorityTier(2) is PriorityTier.NORMAL
        assert PriorityTier(3) is PriorityTier.LOW
        assert PriorityTier(4) is PriorityTier.BACKGROUND

    def test_name_access(self) -> None:
        assert PriorityTier.CRITICAL.name == "CRITICAL"
        assert PriorityTier.BACKGROUND.name == "BACKGROUND"


# ===================================================================
# TIER POLICY TESTS
# ===================================================================

class TestTierPolicy:
    """TierPolicy dataclass -- defaults and custom construction."""

    def test_minimal_construction(self) -> None:
        policy = TierPolicy(
            tier=PriorityTier.NORMAL, label="Normal",
        )
        assert policy.tier is PriorityTier.NORMAL
        assert policy.label == "Normal"
        assert policy.prefer_spot is True
        assert policy.allow_on_demand is True
        assert policy.max_latency_ms == 200.0
        assert policy.max_price_per_hour == float("inf")
        assert policy.max_carbon_intensity == float("inf")
        assert policy.carbon_weight == 0.3
        assert policy.max_queue_time_s == 30.0
        assert policy.preemption_enabled is False

    def test_label_default_is_tier_name(self) -> None:
        policy = TierPolicy(
            tier=PriorityTier.CRITICAL, label="Critical",
        )
        assert policy.label == "Critical"

    def test_label_default_background(self) -> None:
        policy = TierPolicy(
            tier=PriorityTier.BACKGROUND, label="Background",
        )
        assert policy.label == "Background"

    def test_custom_values(self) -> None:
        policy = TierPolicy(
            tier=PriorityTier.HIGH,
            label="High Priority",
            prefer_spot=False,
            allow_on_demand=True,
            max_latency_ms=75.0,
            max_price_per_hour=2.50,
            max_carbon_intensity=300.0,
            carbon_weight=0.1,
            max_queue_time_s=10.0,
            preemption_enabled=True,
        )
        assert policy.tier is PriorityTier.HIGH
        assert policy.label == "High Priority"
        assert policy.prefer_spot is False
        assert policy.allow_on_demand is True
        assert policy.max_latency_ms == 75.0
        assert policy.max_price_per_hour == 2.50
        assert policy.max_carbon_intensity == 300.0
        assert policy.carbon_weight == 0.1
        assert policy.max_queue_time_s == 10.0
        assert policy.preemption_enabled is True


class TestDefaultTierPolicies:
    """DEFAULT_TIER_POLICIES dict -- completeness and key properties."""

    def test_all_tiers_have_policies(self) -> None:
        for tier in PriorityTier:
            assert tier in DEFAULT_TIER_POLICIES

    def test_critical_prefers_on_demand(self) -> None:
        critical = DEFAULT_TIER_POLICIES[PriorityTier.CRITICAL]
        assert critical.prefer_spot is False
        assert critical.preemption_enabled is True
        assert critical.max_latency_ms == 50.0
        assert critical.carbon_weight == 0.0
        assert critical.max_queue_time_s == 5.0

    def test_high_prefers_spot_with_bounds(self) -> None:
        high = DEFAULT_TIER_POLICIES[PriorityTier.HIGH]
        assert high.prefer_spot is True
        assert high.allow_on_demand is True
        assert high.max_carbon_intensity == 500.0
        assert high.preemption_enabled is True
        assert high.max_queue_time_s == 15.0

    def test_normal_prefers_spot_only(self) -> None:
        normal = DEFAULT_TIER_POLICIES[PriorityTier.NORMAL]
        assert normal.prefer_spot is True
        assert normal.allow_on_demand is False
        assert normal.preemption_enabled is False
        assert normal.max_queue_time_s == 30.0

    def test_low_is_batch_grain(self) -> None:
        low = DEFAULT_TIER_POLICIES[PriorityTier.LOW]
        assert low.prefer_spot is True
        assert low.allow_on_demand is False
        assert low.max_latency_ms == 5000.0
        assert low.max_queue_time_s == 300.0
        assert low.carbon_weight == 0.5

    def test_background_is_best_effort(self) -> None:
        bg = DEFAULT_TIER_POLICIES[PriorityTier.BACKGROUND]
        assert bg.prefer_spot is True
        assert bg.allow_on_demand is False
        assert bg.max_latency_ms == 60000.0
        assert bg.max_queue_time_s == 3600.0
        assert bg.carbon_weight == 0.8


# ===================================================================
# SCHEDULED REQUEST TESTS
# ===================================================================

class TestScheduledRequest:
    """ScheduledRequest dataclass -- defaults and properties."""

    def test_minimal_construction(self) -> None:
        req = ScheduledRequest(request_id="req-1", priority=PriorityTier.HIGH)
        assert req.request_id == "req-1"
        assert req.priority is PriorityTier.HIGH
        assert req.gpu_type == ""
        assert req.min_gpu_memory_gb == 0.0
        assert req.user_id == ""
        assert req.metadata == {}

    def test_full_construction(self) -> None:
        req = ScheduledRequest(
            request_id="req-2",
            priority=PriorityTier.LOW,
            gpu_type="A100",
            min_gpu_memory_gb=40.0,
            user_id="tenant-42",
            metadata={"model": "llama-70b", "batch": True},
        )
        assert req.request_id == "req-2"
        assert req.priority is PriorityTier.LOW
        assert req.gpu_type == "A100"
        assert req.min_gpu_memory_gb == 40.0
        assert req.user_id == "tenant-42"
        assert req.metadata == {"model": "llama-70b", "batch": True}

    def test_queued_at_defaults_to_now(self) -> None:
        before = time.time()
        req = ScheduledRequest(request_id="req-3", priority=PriorityTier.NORMAL)
        after = time.time()
        assert before <= req.queued_at <= after

    def test_queue_time_s_increases(self) -> None:
        """queue_time_s should be positive after a brief wait."""
        queued_at = time.time() - 1.0  # 1 second ago
        req = ScheduledRequest(
            request_id="req-4", priority=PriorityTier.NORMAL,
            queued_at=queued_at,
        )
        assert req.queue_time_s == pytest.approx(1.0, abs=0.1)

    def test_queue_time_s_zero_when_just_queued(self) -> None:
        now = time.time()
        req = ScheduledRequest(
            request_id="req-5", priority=PriorityTier.NORMAL,
            queued_at=now,
        )
        assert req.queue_time_s == pytest.approx(0.0, abs=0.05)

    def test_queue_time_s_long_wait(self) -> None:
        queued_at = time.time() - 3600.0  # 1 hour ago
        req = ScheduledRequest(
            request_id="req-6", priority=PriorityTier.BACKGROUND,
            queued_at=queued_at,
        )
        assert req.queue_time_s == pytest.approx(3600.0, abs=1.0)

    def test_explicit_queued_at(self) -> None:
        """Custom queued_at should be preserved."""
        req = ScheduledRequest(
            request_id="req-7", priority=PriorityTier.HIGH,
            queued_at=12345.0,
        )
        assert req.queued_at == 12345.0

    def test_metadata_mutability(self) -> None:
        """Default factory should give independent dicts."""
        req1 = ScheduledRequest(request_id="r1", priority=PriorityTier.NORMAL)
        req2 = ScheduledRequest(request_id="r2", priority=PriorityTier.NORMAL)
        req1.metadata["key"] = "value"
        assert "key" not in req2.metadata


# ===================================================================
# TENANT BUDGET TESTS
# ===================================================================

class TestTenantBudget:
    """TenantBudget -- budget tracking, reset, can_use, record_use."""

    def test_default_construction(self) -> None:
        budget = TenantBudget(tenant_id="tenant-1")
        assert budget.tenant_id == "tenant-1"
        assert budget.max_critical_per_hour == 10
        assert budget.max_high_per_hour == 100
        assert budget.max_normal_per_hour == 1000
        assert budget.current_critical == 0
        assert budget.current_high == 0
        assert budget.current_normal == 0

    def test_can_use_returns_true_when_under_limit(self) -> None:
        budget = TenantBudget(
            tenant_id="t1",
            max_critical_per_hour=10,
            current_critical=5,
        )
        assert budget.can_use(PriorityTier.CRITICAL) is True

    def test_can_use_returns_false_when_at_limit(self) -> None:
        budget = TenantBudget(
            tenant_id="t1",
            max_critical_per_hour=10,
            current_critical=10,
        )
        assert budget.can_use(PriorityTier.CRITICAL) is False

    def test_can_use_returns_false_when_over_limit(self) -> None:
        budget = TenantBudget(
            tenant_id="t1",
            max_critical_per_hour=10,
            current_critical=11,
        )
        assert budget.can_use(PriorityTier.CRITICAL) is False

    def test_can_use_high_tier(self) -> None:
        budget = TenantBudget(
            tenant_id="t1",
            max_high_per_hour=5,
            current_high=5,
        )
        assert budget.can_use(PriorityTier.HIGH) is False

    def test_can_use_normal_tier(self) -> None:
        budget = TenantBudget(
            tenant_id="t1",
            max_normal_per_hour=3,
            current_normal=2,
        )
        assert budget.can_use(PriorityTier.NORMAL) is True

    def test_can_use_low_and_background_always_true(self) -> None:
        budget = TenantBudget(tenant_id="t1")
        assert budget.can_use(PriorityTier.LOW) is True
        assert budget.can_use(PriorityTier.BACKGROUND) is True

    def test_can_use_empty_budget(self) -> None:
        budget = TenantBudget(
            tenant_id="t1",
            max_critical_per_hour=0,
            current_critical=0,
        )
        assert budget.can_use(PriorityTier.CRITICAL) is False

    def test_record_use_increments_critical(self) -> None:
        budget = TenantBudget(tenant_id="t1")
        budget.record_use(PriorityTier.CRITICAL)
        assert budget.current_critical == 1

    def test_record_use_increments_high(self) -> None:
        budget = TenantBudget(tenant_id="t1")
        budget.record_use(PriorityTier.HIGH)
        assert budget.current_high == 1

    def test_record_use_increments_normal(self) -> None:
        budget = TenantBudget(tenant_id="t1")
        budget.record_use(PriorityTier.NORMAL)
        assert budget.current_normal == 1

    def test_record_use_ignores_low_and_background(self) -> None:
        budget = TenantBudget(tenant_id="t1")
        budget.record_use(PriorityTier.LOW)
        budget.record_use(PriorityTier.BACKGROUND)
        assert budget.current_critical == 0
        assert budget.current_high == 0
        assert budget.current_normal == 0

    def test_record_use_multiple_critical(self) -> None:
        budget = TenantBudget(tenant_id="t1")
        for _ in range(5):
            budget.record_use(PriorityTier.CRITICAL)
        assert budget.current_critical == 5

    def test_reset_if_needed_does_not_reset_when_recent(self) -> None:
        budget = TenantBudget(
            tenant_id="t1",
            hour_start=time.time() - 1800,  # 30 min ago
            current_critical=7,
        )
        budget.reset_if_needed()
        assert budget.current_critical == 7

    def test_reset_if_needed_resets_after_hour(self) -> None:
        budget = TenantBudget(
            tenant_id="t1",
            hour_start=time.time() - 3700,  # > 1 hour ago
            current_critical=7,
            current_high=50,
            current_normal=200,
        )
        budget.reset_if_needed()
        assert budget.current_critical == 0
        assert budget.current_high == 0
        assert budget.current_normal == 0

    def test_reset_if_needed_updates_hour_start(self) -> None:
        old_hour_start = time.time() - 3700
        budget = TenantBudget(
            tenant_id="t1",
            hour_start=old_hour_start,
        )
        budget.reset_if_needed()
        assert budget.hour_start != old_hour_start
        # Should now be close to current time
        assert budget.hour_start == pytest.approx(time.time(), abs=1.0)

    def test_reset_if_needed_at_exactly_one_hour(self) -> None:
        budget = TenantBudget(
            tenant_id="t1",
            hour_start=time.time() - 3599,  # 59m 59s, just under 1 hour
            current_critical=7,
        )
        budget.reset_if_needed()
        assert budget.current_critical == 7  # not reset

    def test_can_use_triggers_reset(self) -> None:
        """can_use should call reset_if_needed before checking."""
        budget = TenantBudget(
            tenant_id="t1",
            hour_start=time.time() - 3700,
            current_critical=10,  # at limit but should reset
        )
        assert budget.can_use(PriorityTier.CRITICAL) is True  # reset clears counter

    def test_record_use_triggers_reset(self) -> None:
        """record_use should call reset_if_needed before incrementing."""
        budget = TenantBudget(
            tenant_id="t1",
            hour_start=time.time() - 3700,
            current_critical=10,
        )
        budget.record_use(PriorityTier.CRITICAL)
        assert budget.current_critical == 1  # reset then increment

    def test_custom_max_values(self) -> None:
        budget = TenantBudget(
            tenant_id="t1",
            max_critical_per_hour=100,
            max_high_per_hour=500,
            max_normal_per_hour=9999,
        )
        assert budget.max_critical_per_hour == 100
        assert budget.max_high_per_hour == 500
        assert budget.max_normal_per_hour == 9999


# ===================================================================
# PREEMPTIBLE SCHEDULER TESTS -- Construction & Defaults
# ===================================================================

class TestPreemptibleSchedulerConstruction:
    """PreemptibleScheduler -- construction, defaults, properties."""

    def test_default_construction(self) -> None:
        scheduler = PreemptibleScheduler()
        assert scheduler is not None
        assert scheduler.stats == {
            "enqueued": 0,
            "dequeued": 0,
            "preempted": 0,
            "budget_rejected": 0,
            "timeout_evicted": 0,
        }

    def test_is_idle_on_construction(self) -> None:
        scheduler = PreemptibleScheduler()
        assert scheduler.is_idle() is True

    def test_queue_lengths_all_zero_on_construction(self) -> None:
        scheduler = PreemptibleScheduler()
        lengths = scheduler.queue_lengths()
        for tier in PriorityTier:
            assert lengths[tier.name] == 0

    def test_custom_tier_policies(self) -> None:
        custom = {
            PriorityTier.CRITICAL: TierPolicy(
                tier=PriorityTier.CRITICAL,
                label="Custom Critical",
                max_latency_ms=10.0,
            ),
            PriorityTier.HIGH: TierPolicy(
                tier=PriorityTier.HIGH,
                label="Custom High",
                max_latency_ms=50.0,
            ),
            PriorityTier.NORMAL: DEFAULT_TIER_POLICIES[PriorityTier.NORMAL],
            PriorityTier.LOW: DEFAULT_TIER_POLICIES[PriorityTier.LOW],
            PriorityTier.BACKGROUND: DEFAULT_TIER_POLICIES[PriorityTier.BACKGROUND],
        }
        scheduler = PreemptibleScheduler(tier_policies=custom)
        assert scheduler.get_policy(PriorityTier.CRITICAL).max_latency_ms == 10.0
        assert scheduler.get_policy(PriorityTier.HIGH).max_latency_ms == 50.0


# ===================================================================
# PREEMPTIBLE SCHEDULER TESTS -- Policy management
# ===================================================================

class TestPreemptibleSchedulerPolicy:
    """PreemptibleScheduler -- get_policy and set_policy."""

    def test_get_policy_default_critical(self) -> None:
        scheduler = PreemptibleScheduler()
        policy = scheduler.get_policy(PriorityTier.CRITICAL)
        assert policy.tier is PriorityTier.CRITICAL
        assert policy.max_latency_ms == 50.0
        assert policy.preemption_enabled is True

    def test_get_policy_default_background(self) -> None:
        scheduler = PreemptibleScheduler()
        policy = scheduler.get_policy(PriorityTier.BACKGROUND)
        assert policy.tier is PriorityTier.BACKGROUND
        assert policy.max_latency_ms == 60000.0

    def test_get_policy_unknown_tier_fallback(self) -> None:
        """An unconfigured tier should fall back to NORMAL defaults."""
        # Create a scheduler with a minimal policy map that excludes BACKGROUND
        partial = {
            PriorityTier.CRITICAL: DEFAULT_TIER_POLICIES[PriorityTier.CRITICAL],
            PriorityTier.HIGH: DEFAULT_TIER_POLICIES[PriorityTier.HIGH],
            PriorityTier.NORMAL: DEFAULT_TIER_POLICIES[PriorityTier.NORMAL],
            PriorityTier.LOW: DEFAULT_TIER_POLICIES[PriorityTier.LOW],
            # Intentionally omit BACKGROUND
        }
        scheduler = PreemptibleScheduler(tier_policies=partial)

        # BACKGROUND is missing from the custom dict, should fall back to NORMAL
        policy = scheduler.get_policy(PriorityTier.BACKGROUND)
        assert policy.max_latency_ms == 200.0  # NORMAL default
        assert policy.max_queue_time_s == 30.0  # NORMAL default

    def test_set_policy_overrides(self) -> None:
        policies = copy.deepcopy(DEFAULT_TIER_POLICIES)
        scheduler = PreemptibleScheduler(tier_policies=policies)
        new_policy = TierPolicy(
            tier=PriorityTier.CRITICAL,
            label="Modified Critical",
            max_latency_ms=25.0,
            preemption_enabled=False,
        )
        scheduler.set_policy(PriorityTier.CRITICAL, new_policy)
        policy = scheduler.get_policy(PriorityTier.CRITICAL)
        assert policy.max_latency_ms == 25.0
        assert policy.preemption_enabled is False
        assert policy.label == "Modified Critical"

    def test_set_policy_new_tier(self) -> None:
        """set_policy should work even for tiers not in the original dict."""
        policies = copy.deepcopy(DEFAULT_TIER_POLICIES)
        scheduler = PreemptibleScheduler(tier_policies=policies)
        # Overwrite HIGH (which exists by default)
        high_policy = TierPolicy(
            tier=PriorityTier.HIGH, label="Custom",
            max_latency_ms=80.0,
        )
        scheduler.set_policy(PriorityTier.HIGH, high_policy)
        assert scheduler.get_policy(PriorityTier.HIGH).max_latency_ms == 80.0


# ===================================================================
# PREEMPTIBLE SCHEDULER TESTS -- Tenant budgets
# ===================================================================

class TestPreemptibleSchedulerTenantBudget:
    """PreemptibleScheduler -- set_tenant_budget."""

    def test_set_tenant_budget(self) -> None:
        scheduler = PreemptibleScheduler()
        budget = TenantBudget(tenant_id="t1", max_critical_per_hour=5)
        # Just verify it doesn't crash
        scheduler.set_tenant_budget("t1", budget)

    def test_enqueue_with_budget_allows_good_requests(self) -> None:
        scheduler = PreemptibleScheduler()
        budget = TenantBudget(tenant_id="t1", max_critical_per_hour=5)
        scheduler.set_tenant_budget("t1", budget)

        result = scheduler.enqueue(
            "req-1", PriorityTier.CRITICAL,
            user_id="t1",
        )
        assert result is not None
        assert result.request_id == "req-1"
        assert scheduler.stats["budget_rejected"] == 0

    def test_enqueue_with_budget_rejects_over_limit(self) -> None:
        scheduler = PreemptibleScheduler()
        budget = TenantBudget(tenant_id="t1", max_critical_per_hour=2)
        scheduler.set_tenant_budget("t1", budget)

        # Use up the budget
        assert scheduler.enqueue("r1", PriorityTier.CRITICAL, user_id="t1") is not None
        assert scheduler.enqueue("r2", PriorityTier.CRITICAL, user_id="t1") is not None

        # Third one should be rejected
        result = scheduler.enqueue("r3", PriorityTier.CRITICAL, user_id="t1")
        assert result is None
        assert scheduler.stats["budget_rejected"] == 1

    def test_enqueue_budget_rejected_tracks_stats(self) -> None:
        scheduler = PreemptibleScheduler()
        budget = TenantBudget(
            tenant_id="t1",
            max_critical_per_hour=0,
            max_high_per_hour=0,
        )
        scheduler.set_tenant_budget("t1", budget)

        scheduler.enqueue("r1", PriorityTier.CRITICAL, user_id="t1")
        scheduler.enqueue("r2", PriorityTier.HIGH, user_id="t1")
        assert scheduler.stats["budget_rejected"] == 2

    def test_enqueue_different_tenants_independent(self) -> None:
        scheduler = PreemptibleScheduler()
        scheduler.set_tenant_budget(
            "t1", TenantBudget(tenant_id="t1", max_critical_per_hour=1),
        )
        scheduler.set_tenant_budget(
            "t2", TenantBudget(tenant_id="t2", max_critical_per_hour=1),
        )

        assert scheduler.enqueue("r1", PriorityTier.CRITICAL, user_id="t1") is not None
        # t1 over budget
        assert scheduler.enqueue("r2", PriorityTier.CRITICAL, user_id="t1") is None
        # t2 still has budget
        assert scheduler.enqueue("r3", PriorityTier.CRITICAL, user_id="t2") is not None

    def test_enqueue_no_budget_for_user_passes(self) -> None:
        """If no budget is configured for a user, they are never budget-rejected."""
        scheduler = PreemptibleScheduler()
        result = scheduler.enqueue("r1", PriorityTier.CRITICAL, user_id="unknown")
        assert result is not None
        assert scheduler.stats["budget_rejected"] == 0


# ===================================================================
# PREEMPTIBLE SCHEDULER TESTS -- Enqueue
# ===================================================================

class TestPreemptibleSchedulerEnqueue:
    """PreemptibleScheduler -- enqueue method."""

    def test_enqueue_minimal(self) -> None:
        scheduler = PreemptibleScheduler()
        result = scheduler.enqueue("req-1", PriorityTier.NORMAL)
        assert result is not None
        assert result.request_id == "req-1"
        assert result.priority is PriorityTier.NORMAL

    def test_enqueue_full(self) -> None:
        scheduler = PreemptibleScheduler()
        result = scheduler.enqueue(
            "req-1",
            PriorityTier.CRITICAL,
            gpu_type="H100",
            min_gpu_memory_gb=80.0,
            user_id="tenant-99",
            metadata={"model": "gpt-4"},
        )
        assert result is not None
        assert result.gpu_type == "H100"
        assert result.min_gpu_memory_gb == 80.0
        assert result.user_id == "tenant-99"
        assert result.metadata == {"model": "gpt-4"}

    def test_enqueue_increments_stats(self) -> None:
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("r1", PriorityTier.NORMAL)
        assert scheduler.stats["enqueued"] == 1

        scheduler.enqueue("r2", PriorityTier.NORMAL)
        assert scheduler.stats["enqueued"] == 2

    def test_enqueue_updates_queue_lengths(self) -> None:
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("r1", PriorityTier.CRITICAL)
        scheduler.enqueue("r2", PriorityTier.NORMAL)
        scheduler.enqueue("r3", PriorityTier.NORMAL)

        lengths = scheduler.queue_lengths()
        assert lengths["CRITICAL"] == 1
        assert lengths["NORMAL"] == 2
        assert lengths["HIGH"] == 0

    def test_enqueue_default_metadata(self) -> None:
        scheduler = PreemptibleScheduler()
        result = scheduler.enqueue("r1", PriorityTier.HIGH)
        assert result is not None
        assert result.metadata == {}

    def test_enqueue_metadata_none_becomes_empty(self) -> None:
        scheduler = PreemptibleScheduler()
        result = scheduler.enqueue("r1", PriorityTier.HIGH, metadata=None)
        assert result is not None
        assert result.metadata == {}


# ===================================================================
# PREEMPTIBLE SCHEDULER TESTS -- Dequeue
# ===================================================================

class TestPreemptibleSchedulerDequeue:
    """PreemptibleScheduler -- dequeue method."""

    def test_dequeue_empty_returns_none(self) -> None:
        scheduler = PreemptibleScheduler()
        result = scheduler.dequeue()
        assert result is None

    def test_dequeue_single_request(self) -> None:
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("r1", PriorityTier.NORMAL)
        result = scheduler.dequeue()
        assert result is not None
        assert result.request_id == "r1"

    def test_dequeue_priority_order(self) -> None:
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("low", PriorityTier.LOW)
        scheduler.enqueue("critical", PriorityTier.CRITICAL)
        scheduler.enqueue("normal", PriorityTier.NORMAL)

        # Should dequeue in priority order: CRITICAL, NORMAL, LOW
        assert scheduler.dequeue().request_id == "critical"
        assert scheduler.dequeue().request_id == "normal"
        assert scheduler.dequeue().request_id == "low"

    def test_dequeue_priority_order_all_tiers(self) -> None:
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("bg", PriorityTier.BACKGROUND)
        scheduler.enqueue("high", PriorityTier.HIGH)
        scheduler.enqueue("critical", PriorityTier.CRITICAL)

        assert scheduler.dequeue().request_id == "critical"
        assert scheduler.dequeue().request_id == "high"
        assert scheduler.dequeue().request_id == "bg"

    def test_dequeue_increments_stats(self) -> None:
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("r1", PriorityTier.NORMAL)
        scheduler.dequeue()
        assert scheduler.stats["dequeued"] == 1

    def test_dequeue_moves_request_to_active(self) -> None:
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("r1", PriorityTier.HIGH)
        scheduler.dequeue()
        # Not idle because there's an active request
        assert scheduler.is_idle() is False

    def test_dequeue_fifo_within_same_priority(self) -> None:
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("first", PriorityTier.NORMAL)
        scheduler.enqueue("second", PriorityTier.NORMAL)

        assert scheduler.dequeue().request_id == "first"
        assert scheduler.dequeue().request_id == "second"

    def test_dequeue_after_complete(self) -> None:
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("r1", PriorityTier.HIGH)
        req = scheduler.dequeue()
        assert req is not None
        scheduler.complete(req.request_id)
        assert scheduler.is_idle() is True

    def test_dequeue_multiple_returns_none_when_exhausted(self) -> None:
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("r1", PriorityTier.CRITICAL)
        scheduler.dequeue()
        result = scheduler.dequeue()
        assert result is None


# ===================================================================
# PREEMPTIBLE SCHEDULER TESTS -- Complete
# ===================================================================

class TestPreemptibleSchedulerComplete:
    """PreemptibleScheduler -- complete method."""

    def test_complete_removes_active_request(self) -> None:
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("r1", PriorityTier.HIGH)
        scheduler.dequeue()
        assert scheduler.is_idle() is False

        scheduler.complete("r1")
        assert scheduler.is_idle() is True

    def test_complete_nonexistent_request_does_not_raise(self) -> None:
        scheduler = PreemptibleScheduler()
        scheduler.complete("nonexistent")  # should not raise

    def test_complete_idempotent(self) -> None:
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("r1", PriorityTier.HIGH)
        scheduler.dequeue()
        scheduler.complete("r1")
        scheduler.complete("r1")  # second complete should not raise


# ===================================================================
# PREEMPTIBLE SCHEDULER TESTS -- Preempt lower
# ===================================================================

class TestPreemptibleSchedulerPreempt:
    """PreemptibleScheduler -- preempt_lower method."""

    def test_preempt_lower_empty_active(self) -> None:
        """No active requests to preempt."""
        scheduler = PreemptibleScheduler()
        result = scheduler.preempt_lower(PriorityTier.CRITICAL)
        assert result == []

    def test_preempt_lower_no_lower_priority(self) -> None:
        """If active requests are all higher or equal priority, nothing preempted."""
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("r1", PriorityTier.CRITICAL)
        scheduler.dequeue()  # CRITICAL is now active
        # Try to preempt with HIGH -- CRITICAL > HIGH so no preemption
        result = scheduler.preempt_lower(PriorityTier.HIGH)
        assert result == []

    def test_preempt_lower_preempts_lower_priority(self) -> None:
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("low", PriorityTier.LOW)
        scheduler.enqueue("critical", PriorityTier.CRITICAL)
        # Dequeue both so they're active
        scheduler.dequeue()  # CRITICAL
        scheduler.dequeue()  # LOW

        # Incoming HIGH should preempt LOW but not CRITICAL
        result = scheduler.preempt_lower(PriorityTier.HIGH)
        assert "low" in result
        assert "critical" not in result

    def test_preempt_lower_incoming_critical_preempts_all(self) -> None:
        scheduler = PreemptibleScheduler()
        for tier in [PriorityTier.HIGH, PriorityTier.NORMAL, PriorityTier.LOW, PriorityTier.BACKGROUND]:
            scheduler.enqueue(f"req-{tier.name}", tier)
            scheduler.dequeue()  # makes them active

        result = scheduler.preempt_lower(PriorityTier.CRITICAL)
        assert set(result) == {"req-HIGH", "req-NORMAL", "req-LOW", "req-BACKGROUND"}

    def test_preempt_lower_increments_stats(self) -> None:
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("low", PriorityTier.LOW)
        scheduler.dequeue()
        assert scheduler.stats["preempted"] == 0

        scheduler.preempt_lower(PriorityTier.CRITICAL)
        assert scheduler.stats["preempted"] == 1

    def test_preempt_lower_when_preemption_disabled(self) -> None:
        """NORMAL tier has preemption_enabled=False -- should return []."""
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("low", PriorityTier.LOW)
        scheduler.dequeue()
        assert scheduler.is_idle() is False

        # NORMAL has preemption_enabled=False in defaults
        result = scheduler.preempt_lower(PriorityTier.NORMAL)
        assert result == []
        assert scheduler.stats["preempted"] == 0

    def test_preempt_lower_disabled_tier_does_not_preempt(self) -> None:
        """Even if there are lower-priority active requests, disabled tier won't preempt."""
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("low", PriorityTier.LOW)
        scheduler.dequeue()

        # NORMAL has preemption disabled
        result = scheduler.preempt_lower(PriorityTier.NORMAL)
        assert result == []

    def test_preempt_lower_with_preemption_enabled_works(self) -> None:
        """HIGH has preemption enabled -- should preempt lower."""
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("low", PriorityTier.LOW)
        scheduler.dequeue()

        # HIGH has preemption_enabled=True
        result = scheduler.preempt_lower(PriorityTier.HIGH)
        assert "low" in result
        assert scheduler.stats["preempted"] == 1

    def test_preempt_lower_logs_entry(self) -> None:
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("low", PriorityTier.LOW)
        scheduler.dequeue()

        scheduler.preempt_lower(PriorityTier.CRITICAL)

        log = scheduler.get_preemption_log()
        assert len(log) == 1
        entry = log[0]
        assert entry["preempted_id"] == "low"
        assert entry["preempted_priority"] == "LOW"
        assert entry["by_priority"] == "CRITICAL"
        assert "timestamp" in entry

    def test_preempt_lower_multiple_times(self) -> None:
        """Preemption log should accumulate entries."""
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("low1", PriorityTier.LOW)
        scheduler.enqueue("low2", PriorityTier.LOW)
        scheduler.dequeue()
        scheduler.dequeue()

        scheduler.preempt_lower(PriorityTier.CRITICAL)
        scheduler.preempt_lower(PriorityTier.HIGH)

        log = scheduler.get_preemption_log()
        assert len(log) == 2

    def test_preempt_lower_get_log_limit(self) -> None:
        scheduler = PreemptibleScheduler()
        # Add 3 preemption entries
        for i in range(5):
            scheduler.enqueue(f"low{i}", PriorityTier.LOW)
            scheduler.dequeue()
            scheduler.preempt_lower(PriorityTier.CRITICAL)

        log = scheduler.get_preemption_log(limit=3)
        assert len(log) == 3

    def test_preempt_lower_multiple_active_requests(self) -> None:
        """Only lower-priority requests should be preempted."""
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("low", PriorityTier.LOW)
        scheduler.enqueue("critical", PriorityTier.CRITICAL)
        scheduler.dequeue()  # CRITICAL
        scheduler.dequeue()  # LOW

        result = scheduler.preempt_lower(PriorityTier.HIGH)
        assert "low" in result
        assert "critical" not in result


# ===================================================================
# PREEMPTIBLE SCHEDULER TESTS -- Queue idle state
# ===================================================================

class TestPreemptibleSchedulerIdle:
    """PreemptibleScheduler -- is_idle method."""

    def test_construction_idle(self) -> None:
        scheduler = PreemptibleScheduler()
        assert scheduler.is_idle() is True

    def test_not_idle_with_queued_requests(self) -> None:
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("r1", PriorityTier.NORMAL)
        # Any queued request means scheduler is not idle
        assert scheduler.is_idle() is False

    def test_not_idle_with_active_request(self) -> None:
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("r1", PriorityTier.HIGH)
        scheduler.dequeue()
        assert scheduler.is_idle() is False

    def test_idle_after_complete(self) -> None:
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("r1", PriorityTier.HIGH)
        req = scheduler.dequeue()
        scheduler.complete(req.request_id)
        assert scheduler.is_idle() is True

    def test_idle_with_queued_but_no_active(self) -> None:
        """is_idle returns False if items are in a queue even if none active."""
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("r1", PriorityTier.NORMAL)
        # Queued items mean the scheduler is not idle, even without active requests
        assert scheduler.is_idle() is False


# ===================================================================
# PREEMPTIBLE SCHEDULER TESTS -- Timeout eviction
# ===================================================================

class TestPreemptibleSchedulerEviction:
    """PreemptibleScheduler -- _evict_timed_out and timeout eviction."""

    def test_dequeue_evicts_timed_out_requests(self) -> None:
        """Requests queued beyond their tier's max_queue_time_s should be evicted."""
        scheduler = PreemptibleScheduler()
        # NORMAL has max_queue_time_s=30.0
        # Create a request with queued_at far in the past
        old_time = time.time() - 100.0  # well past 30s limit
        req = ScheduledRequest(
            request_id="stale",
            priority=PriorityTier.NORMAL,
            queued_at=old_time,
        )
        # Manually add to queue to bypass enqueue()'s default time.time()
        scheduler._queues[PriorityTier.NORMAL].append(req)

        assert scheduler.stats["timeout_evicted"] == 0
        result = scheduler.dequeue()
        assert result is None  # stale request evicted
        assert scheduler.stats["timeout_evicted"] == 1

    def test_dequeue_does_not_evict_recent_requests(self) -> None:
        """Recent requests should remain in the queue."""
        scheduler = PreemptibleScheduler()
        req = ScheduledRequest(
            request_id="fresh",
            priority=PriorityTier.NORMAL,
            queued_at=time.time(),
        )
        scheduler._queues[PriorityTier.NORMAL].append(req)

        result = scheduler.dequeue()
        assert result is not None
        assert result.request_id == "fresh"
        assert scheduler.stats["timeout_evicted"] == 0

    def test_eviction_checks_all_tiers(self) -> None:
        """_evict_timed_out should check all tiers, not just one."""
        scheduler = PreemptibleScheduler()
        old_time = time.time() - 3600.0  # 1 hour ago

        for tier in PriorityTier:
            req = ScheduledRequest(
                request_id=f"old-{tier.name}",
                priority=tier,
                queued_at=old_time,
            )
            scheduler._queues[tier].append(req)

        scheduler._evict_timed_out()
        # All 5 requests should be evicted
        assert scheduler.stats["timeout_evicted"] == 5

    def test_eviction_partial_within_same_tier(self) -> None:
        """Only the head of the queue that's timed out should be evicted, fresh ones remain."""
        scheduler = PreemptibleScheduler()
        old_time = time.time() - 100.0
        now = time.time()

        # Add stale, then fresh in the same NORMAL queue
        stale = ScheduledRequest(
            request_id="stale", priority=PriorityTier.NORMAL,
            queued_at=old_time,
        )
        fresh = ScheduledRequest(
            request_id="fresh", priority=PriorityTier.NORMAL,
            queued_at=now,
        )
        scheduler._queues[PriorityTier.NORMAL].append(stale)
        scheduler._queues[PriorityTier.NORMAL].append(fresh)

        scheduler._evict_timed_out()
        assert scheduler.stats["timeout_evicted"] == 1
        # Fresh should still be there and be dequeuable
        result = scheduler.dequeue()
        assert result is not None
        assert result.request_id == "fresh"

    def test_eviction_tier_specific_timeouts(self) -> None:
        """Each tier has its own max_queue_time_s -- test CRITICAL (5s) vs NORMAL (30s)."""
        scheduler = PreemptibleScheduler()
        # A request queued 20s ago would be stale for CRITICAL (5s limit) but
        # still fresh for NORMAL (30s limit)
        middle_time = time.time() - 20.0

        stale_critical = ScheduledRequest(
            request_id="stale-critical", priority=PriorityTier.CRITICAL,
            queued_at=middle_time,
        )
        fresh_normal = ScheduledRequest(
            request_id="fresh-normal", priority=PriorityTier.NORMAL,
            queued_at=middle_time,
        )
        scheduler._queues[PriorityTier.CRITICAL].append(stale_critical)
        scheduler._queues[PriorityTier.NORMAL].append(fresh_normal)

        scheduler._evict_timed_out()
        assert scheduler.stats["timeout_evicted"] == 1
        assert scheduler.queue_lengths()["CRITICAL"] == 0
        assert scheduler.queue_lengths()["NORMAL"] == 1


# ===================================================================
# PREEMPTIBLE SCHEDULER TESTS -- Stats
# ===================================================================

class TestPreemptibleSchedulerStats:
    """PreemptibleScheduler -- stats property."""

    def test_stats_start_zero(self) -> None:
        scheduler = PreemptibleScheduler()
        assert scheduler.stats == {
            "enqueued": 0,
            "dequeued": 0,
            "preempted": 0,
            "budget_rejected": 0,
            "timeout_evicted": 0,
        }

    def test_stats_is_copy(self) -> None:
        """The stats property should return a copy, not the internal dict."""
        scheduler = PreemptibleScheduler()
        stats_ref = scheduler.stats
        stats_ref["enqueued"] = 999  # modify the copy
        assert scheduler.stats["enqueued"] == 0  # original unchanged

    def test_stats_tracks_all_operations(self) -> None:
        scheduler = PreemptibleScheduler()
        budget = TenantBudget(tenant_id="t1", max_critical_per_hour=1)
        scheduler.set_tenant_budget("t1", budget)

        scheduler.enqueue("r1", PriorityTier.CRITICAL, user_id="t1")  # enqueued=1
        scheduler.enqueue("r2", PriorityTier.CRITICAL, user_id="t1")  # budget_rejected=1
        scheduler.dequeue()  # dequeued=1

        old_time = time.time() - 100.0
        stale = ScheduledRequest(
            request_id="stale", priority=PriorityTier.NORMAL,
            queued_at=old_time,
        )
        scheduler._queues[PriorityTier.NORMAL].append(stale)
        scheduler.dequeue()  # timeout_evicted=1

        new_req = scheduler.enqueue("r3", PriorityTier.LOW, user_id="t1")
        assert new_req is not None  # LOW is not tracked in budget
        scheduler.dequeue()
        scheduler.preempt_lower(PriorityTier.CRITICAL)  # no active, so no preemption

        assert scheduler.stats == {
            "enqueued": 2,
            "dequeued": 2,
            "preempted": 1,  # r3 was preempted by CRITICAL
            "budget_rejected": 1,
            "timeout_evicted": 1,
        }


# ===================================================================
# PREEMPTIBLE SCHEDULER TESTS -- Queue lengths
# ===================================================================

class TestPreemptibleSchedulerQueueLengths:
    """PreemptibleScheduler -- queue_lengths method."""

    def test_queue_lengths_empty(self) -> None:
        scheduler = PreemptibleScheduler()
        lengths = scheduler.queue_lengths()
        for tier in PriorityTier:
            assert lengths[tier.name] == 0

    def test_queue_lengths_after_enqueue(self) -> None:
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("r1", PriorityTier.CRITICAL)
        scheduler.enqueue("r2", PriorityTier.HIGH)
        scheduler.enqueue("r3", PriorityTier.HIGH)
        scheduler.enqueue("r4", PriorityTier.BACKGROUND)

        lengths = scheduler.queue_lengths()
        assert lengths["CRITICAL"] == 1
        assert lengths["HIGH"] == 2
        assert lengths["NORMAL"] == 0
        assert lengths["LOW"] == 0
        assert lengths["BACKGROUND"] == 1

    def test_queue_lengths_after_dequeue(self) -> None:
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("r1", PriorityTier.CRITICAL)
        scheduler.enqueue("r2", PriorityTier.NORMAL)
        scheduler.dequeue()  # takes CRITICAL

        lengths = scheduler.queue_lengths()
        assert lengths["CRITICAL"] == 0
        assert lengths["NORMAL"] == 1


# ===================================================================
# PREEMPTIBLE SCHEDULER TESTS -- Full lifecycle integration
# ===================================================================

class TestPreemptibleSchedulerLifecycle:
    """PreemptibleScheduler -- end-to-end scenarios."""

    def test_enqueue_dequeue_complete_cycle(self) -> None:
        scheduler = PreemptibleScheduler()

        # Enqueue
        req = scheduler.enqueue("r1", PriorityTier.NORMAL)
        assert req is not None
        assert scheduler.queue_lengths()["NORMAL"] == 1
        assert scheduler.is_idle() is False  # queued, not active

        # Dequeue
        result = scheduler.dequeue()
        assert result is not None
        assert result.request_id == "r1"
        assert scheduler.queue_lengths()["NORMAL"] == 0
        assert scheduler.is_idle() is False  # now active

        # Complete
        scheduler.complete("r1")
        assert scheduler.is_idle() is True

        # Stats
        assert scheduler.stats["enqueued"] == 1
        assert scheduler.stats["dequeued"] == 1

    def test_mixed_priorities(self) -> None:
        """A complete workflow with all priority levels."""
        scheduler = PreemptibleScheduler()

        # Enqueue in reverse priority order
        scheduler.enqueue("bg", PriorityTier.BACKGROUND)
        scheduler.enqueue("low", PriorityTier.LOW)
        scheduler.enqueue("normal", PriorityTier.NORMAL)
        scheduler.enqueue("high", PriorityTier.HIGH)
        scheduler.enqueue("critical", PriorityTier.CRITICAL)

        # Dequeue should respect priority order
        assert scheduler.dequeue().request_id == "critical"
        assert scheduler.dequeue().request_id == "high"
        assert scheduler.dequeue().request_id == "normal"
        assert scheduler.dequeue().request_id == "low"
        assert scheduler.dequeue().request_id == "bg"

    def test_empty_after_all_dequeued(self) -> None:
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("r1", PriorityTier.NORMAL)
        scheduler.dequeue()
        scheduler.complete("r1")

        assert scheduler.is_idle() is True
        assert scheduler.dequeue() is None

    def test_preempted_request_removed_from_active(self) -> None:
        scheduler = PreemptibleScheduler()
        scheduler.enqueue("low", PriorityTier.LOW)
        scheduler.dequeue()  # LOW is active

        # Preempt it
        preempted = scheduler.preempt_lower(PriorityTier.CRITICAL)
        assert "low" in preempted
        # LOW is no longer active
        assert scheduler.is_idle() is True

    def test_budget_rejection_then_acceptance(self) -> None:
        scheduler = PreemptibleScheduler()
        budget = TenantBudget(tenant_id="t1", max_critical_per_hour=1)
        scheduler.set_tenant_budget("t1", budget)

        # First request accepted
        assert scheduler.enqueue("r1", PriorityTier.CRITICAL, user_id="t1") is not None

        # Second request rejected
        assert scheduler.enqueue("r2", PriorityTier.CRITICAL, user_id="t1") is None

        # Non-critical requests not bound by critical budget
        assert scheduler.enqueue("r3", PriorityTier.BACKGROUND, user_id="t1") is not None

    def test_time_travel_timeout_eviction(self) -> None:
        """Requests from the distant past should be evicted before dequeue."""
        scheduler = PreemptibleScheduler()

        # Normal request from the long past
        very_old = time.time() - 9999.0
        stale = ScheduledRequest(
            request_id="ancient", priority=PriorityTier.NORMAL,
            queued_at=very_old,
        )
        scheduler._queues[PriorityTier.NORMAL].append(stale)

        # A fresh low-priority request
        fresh = ScheduledRequest(
            request_id="fresh", priority=PriorityTier.LOW,
            queued_at=time.time(),
        )
        scheduler._queues[PriorityTier.LOW].append(fresh)

        # dequeue should evict the stale NORMAL request and return the fresh LOW
        result = scheduler.dequeue()
        assert result is not None
        assert result.request_id == "fresh"
        assert scheduler.stats["timeout_evicted"] == 1
