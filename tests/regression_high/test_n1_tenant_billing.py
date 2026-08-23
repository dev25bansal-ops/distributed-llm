"""Regression tests for N1 — multi-tenant billing / quotas.

Covers:
  1. quota enforcement — allow under limit, Deny/429 over limit
  2. usage aggregates into per-tenant invoice line items
  3. per-tenant isolation — tenant A usage never affects tenant B
  4. tier limits differ (free < pro < enterprise)

Enforcement + aggregation are REAL; Stripe export remains the E12 stub.
"""

from __future__ import annotations

import time

import pytest

from distllm.core.metering import MeteringStore, reset_metering_store
from distllm.core.tenant_billing import (
    TIER_PLANS,
    AllowDeny,
    TenantBillingManager,
    TierPlan,
    get_tenant_billing_manager,
    reset_tenant_billing_manager,
)


@pytest.fixture
def mgr() -> TenantBillingManager:
    """Fresh manager with an isolated in-memory metering store."""
    reset_metering_store()
    reset_tenant_billing_manager()
    store = MeteringStore()  # no backend => memory-only, isolated per test
    return TenantBillingManager(store=store)


# ── 1. Quota enforcement (allow under / deny 429 over) ─────────────────────

def test_requests_per_min_allow_then_deny_429(mgr: TenantBillingManager):
    mgr.set_custom_plan("t1", TierPlan("tiny", requests_per_min=3,
                                       tokens_per_day=0, monthly_cost_cap_usd=0.0))
    # First 3 allowed.
    for i in range(3):
        d = mgr.check("t1")
        assert d.allowed, f"request {i} should be allowed"
        assert d.status_code == 200
    # 4th denied with 429.
    d = mgr.check("t1")
    assert not d.allowed
    assert d.status_code == 429
    assert "rate limit" in d.reason


def test_tokens_per_day_enforced(mgr: TenantBillingManager):
    mgr.set_custom_plan("t1", TierPlan("cap", requests_per_min=0,
                                       tokens_per_day=1000, monthly_cost_cap_usd=0.0))
    # Under limit: allowed.
    assert mgr.check("t1", requested_tokens=500).allowed
    # Record 900 tokens of usage.
    mgr.record_usage("t1", tokens_in=600, tokens_out=300, model_name="m")
    # Another 500 would push to 1400 > 1000 -> deny.
    d = mgr.check("t1", requested_tokens=500)
    assert not d.allowed
    assert d.status_code == 429
    assert "daily token limit" in d.reason
    # A small request that still fits is allowed.
    assert mgr.check("t1", requested_tokens=50).allowed


def test_monthly_cost_cap_enforced(mgr: TenantBillingManager):
    mgr.set_custom_plan("t1", TierPlan("costcap", requests_per_min=0,
                                       tokens_per_day=0, monthly_cost_cap_usd=1.0))
    # Record usage with explicit cost of $0.90.
    mgr.record_usage("t1", tokens_in=100, tokens_out=100, model_name="m",
                     cost_usd=0.90, compute_s=1.0)
    # A request costing $0.20 -> 1.10 > 1.0 -> deny.
    d = mgr.check("t1", request_cost=0.20)
    assert not d.allowed
    assert d.status_code == 429
    assert "monthly cost cap" in d.reason
    # A tiny request that fits is allowed.
    assert mgr.check("t1", request_cost=0.05).allowed


def test_allowdeny_truthiness():
    assert bool(AllowDeny.allow()) is True
    assert bool(AllowDeny.deny("x")) is False
    assert AllowDeny.deny("x").status_code == 429


# ── 2. Usage aggregates into per-tenant invoice line items ─────────────────

def test_invoice_line_items_aggregate_per_model(mgr: TenantBillingManager):
    mgr.set_tier("acme", "pro")
    mgr.record_usage("acme", tokens_in=100, tokens_out=50, model_name="llama-70b",
                     cost_usd=0.10, compute_s=2.0)
    mgr.record_usage("acme", tokens_in=200, tokens_out=100, model_name="llama-70b",
                     cost_usd=0.20, compute_s=3.0)
    mgr.record_usage("acme", tokens_in=10, tokens_out=5, model_name="mistral-7b",
                     cost_usd=0.01, compute_s=0.5)

    items = mgr.aggregate_line_items("acme")
    by_model = {it["model_name"]: it for it in items}
    assert set(by_model) == {"llama-70b", "mistral-7b"}

    llama = by_model["llama-70b"]
    assert llama["requests"] == 2
    assert llama["tokens_in"] == 300
    assert llama["tokens_out"] == 150
    assert llama["total_tokens"] == 450
    assert llama["cost_usd"] == pytest.approx(0.30)

    invoice = mgr.build_invoice("acme")
    assert invoice["tenant_id"] == "acme"
    assert invoice["tier"] == "pro"
    assert invoice["line_item_count"] == 3          # E12 per-record line items
    assert len(invoice["line_items_by_model"]) == 2  # N1 per-model aggregation
    assert invoice["subtotal_usd"] == pytest.approx(0.31)
    # Stripe stays a stub (E12).
    assert invoice["mode"] == "stub"


# ── 3. Per-tenant isolation ────────────────────────────────────────────────

def test_per_tenant_isolation(mgr: TenantBillingManager):
    mgr.set_custom_plan("A", TierPlan("a", requests_per_min=0,
                                      tokens_per_day=1000, monthly_cost_cap_usd=0.0))
    mgr.set_custom_plan("B", TierPlan("b", requests_per_min=0,
                                      tokens_per_day=1000, monthly_cost_cap_usd=0.0))
    # Exhaust tenant A's token budget.
    mgr.record_usage("A", tokens_in=900, tokens_out=100, model_name="m")
    assert not mgr.check("A", requested_tokens=200).allowed  # A over limit
    # Tenant B is entirely unaffected.
    assert mgr.check("B", requested_tokens=500).allowed
    assert mgr.aggregate_line_items("B") == []
    assert mgr.tenant_summary("B")["tokens_month"] == 0
    # A's invoice has usage; B's does not.
    assert mgr.build_invoice("A")["total_tokens"] == 1000
    assert mgr.build_invoice("B")["total_tokens"] == 0


def test_rate_window_isolation(mgr: TenantBillingManager):
    plan = TierPlan("r", requests_per_min=2, tokens_per_day=0, monthly_cost_cap_usd=0.0)
    mgr.set_custom_plan("A", plan)
    mgr.set_custom_plan("B", plan)
    assert mgr.check("A").allowed
    assert mgr.check("A").allowed
    assert not mgr.check("A").allowed  # A exhausted
    # B still has its own full window.
    assert mgr.check("B").allowed
    assert mgr.check("B").allowed


# ── 4. Tier limits differ (free < pro < enterprise) ───────────────────────

def test_tier_limits_strictly_increasing():
    free = TIER_PLANS["free"]
    pro = TIER_PLANS["pro"]
    ent = TIER_PLANS["enterprise"]
    assert free.requests_per_min < pro.requests_per_min < ent.requests_per_min
    assert free.tokens_per_day < pro.tokens_per_day < ent.tokens_per_day
    assert free.monthly_cost_cap_usd < pro.monthly_cost_cap_usd < ent.monthly_cost_cap_usd


def test_tier_assignment_changes_enforcement(mgr: TenantBillingManager):
    # A request of 200k tokens: denied under free (100k/day), allowed under pro.
    mgr.set_tier("t", "free")
    assert not mgr.check("t", requested_tokens=200_000).allowed
    mgr.set_tier("t", "pro")
    assert mgr.check("t", requested_tokens=200_000).allowed


def test_default_tier_is_free(mgr: TenantBillingManager):
    assert mgr.get_tier("never-seen") == "free"
    assert mgr.get_plan("never-seen").name == "free"


def test_unknown_tier_rejected(mgr: TenantBillingManager):
    with pytest.raises(ValueError):
        mgr.set_tier("t", "platinum")


def test_singleton_reset():
    reset_tenant_billing_manager()
    a = get_tenant_billing_manager()
    b = get_tenant_billing_manager()
    assert a is b
    reset_tenant_billing_manager()
    c = get_tenant_billing_manager()
    assert c is not a
