"""Tests for live per-tenant DP budget metering.

Covers:
- TenantPrivacyBudget: snapshot, record_query, spent/remaining/exhausted,
  fail-closed behavior once the ε budget is spent
- PrivacyBudgetMeter: registration, meter, record_query, all_snapshots
"""

from __future__ import annotations

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

# Load differential_privacy first so privacy_budget.py's module-level
# ``from distllm.core.differential_privacy import ...`` resolves against the
# real source module (distllm.core is a fake package).
_dp = load_module("distllm/core/differential_privacy.py")
_privacy = load_module("distllm/core/privacy_budget.py")

TenantPrivacyBudget = _privacy.TenantPrivacyBudget
PrivacyBudgetMeter = _privacy.PrivacyBudgetMeter
DifferentialPrivacyConfig = _dp.DifferentialPrivacyConfig


# ── TenantPrivacyBudget ───────────────────────────────────────────────────────


class TestTenantPrivacyBudget:
    def test_default_budget(self):
        b = TenantPrivacyBudget(tenant_id="t1")
        assert b.queries == 0
        snap = b.snapshot()
        assert snap["tenant_id"] == "t1"
        assert snap["num_queries"] == 0
        assert snap["spent_epsilon"] == 0.0
        assert snap["remaining_epsilon"] == pytest.approx(5.0)
        assert snap["exhausted"] is False

    def test_record_query_increments_and_spends(self):
        b = TenantPrivacyBudget(tenant_id="t1", epsilon_limit=5.0)
        before = b.spent_epsilon()
        snap = b.record_query()
        assert b.queries == 1
        assert snap["num_queries"] == 1
        assert snap["spent_epsilon"] > before

    def test_remaining_decreases_after_queries(self):
        b = TenantPrivacyBudget(tenant_id="t1", epsilon_limit=5.0)
        before = b.remaining()
        b.record_query()
        b.record_query()
        after = b.remaining()
        assert after < before

    def test_advanced_composition_grows_sublinearly(self):
        # Composed ε = ε * sqrt(2k * ln(1.25/δ)); two queries cost less than
        # twice one query under advanced composition.
        b = TenantPrivacyBudget(tenant_id="t1", epsilon_limit=100.0)
        one = b.record_query()["spent_epsilon"]
        b.record_query()
        two = b.spent_epsilon()
        assert two < 2 * one

    def test_exhausted_after_spending_limit(self):
        # Tiny ε limit so a single query exhausts the budget.
        b = TenantPrivacyBudget(tenant_id="t1", epsilon_limit=0.0001)
        assert not b.is_exhausted()
        b.record_query()
        assert b.is_exhausted()

    def test_record_query_fails_closed_when_exhausted(self):
        b = TenantPrivacyBudget(tenant_id="t1", epsilon_limit=0.0)
        with pytest.raises(RuntimeError):
            b.record_query()

    def test_snapshot_clamps_remaining_at_zero(self):
        b = TenantPrivacyBudget(tenant_id="t1", epsilon_limit=0.0001)
        b.record_query()
        snap = b.snapshot()
        assert snap["remaining_epsilon"] == 0.0
        assert snap["exhausted"] is True

    def test_snapshot_exposes_noise_scale_from_config(self):
        cfg = DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5)
        b = TenantPrivacyBudget(tenant_id="t1", config=cfg)
        snap = b.snapshot()
        assert snap["noise_multiplier"] == pytest.approx(cfg.sigma)


# ── PrivacyBudgetMeter ────────────────────────────────────────────────────────


class TestPrivacyBudgetMeter:
    def test_register_and_get(self):
        meter = PrivacyBudgetMeter()
        budget = meter.register_tenant("t1")
        assert meter.get("t1") is budget

    def test_register_is_idempotent(self):
        meter = PrivacyBudgetMeter()
        first = meter.register_tenant("t1", epsilon_limit=1.0)
        second = meter.register_tenant("t1", epsilon_limit=99.0)
        assert first is second

    def test_meter_registers_on_first_use(self):
        meter = PrivacyBudgetMeter()
        snap = meter.meter("new-tenant")
        assert snap["tenant_id"] == "new-tenant"
        assert snap["num_queries"] == 0

    def test_record_query_tracks_per_tenant(self):
        meter = PrivacyBudgetMeter()
        meter.record_query("a")
        meter.record_query("a")
        meter.record_query("b")
        assert meter.get("a").queries == 2
        assert meter.get("b").queries == 1

    def test_all_snapshots(self):
        meter = PrivacyBudgetMeter()
        meter.record_query("a")
        meter.record_query("a")
        meter.record_query("b")
        snaps = meter.all_snapshots()
        assert snaps["a"]["num_queries"] == 2
        assert snaps["b"]["num_queries"] == 1

    def test_fail_closed_at_meter_level(self):
        meter = PrivacyBudgetMeter(default_epsilon_limit=0.0)
        with pytest.raises(RuntimeError):
            meter.record_query("t1")
