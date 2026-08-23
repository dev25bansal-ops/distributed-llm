"""Multi-tenant enterprise billing / quotas (Task N1).

This module builds a cohesive :class:`TenantBillingManager` on top of the
existing E12 metering layer (:mod:`distllm.core.metering`) and the cost
tracker (:mod:`distllm.core.cost_tracker`).  It does **not** duplicate their
logic — it reuses:

* :class:`~distllm.core.metering.MeteringStore` / :class:`UsageRecord` as the
  source of truth for per-tenant usage (tokens / compute / cost).
* :class:`~distllm.core.cost_tracker.CostTracker` to turn token counts into a
  canonical USD cost + GPU-seconds estimate.
* :class:`~distllm.core.metering.BillingExporter` for invoice serialization
  (the Stripe export path remains the E12 **stub**).

What is REAL here (not stubbed):

* **Tiered plans** — ``free`` / ``pro`` / ``enterprise`` with increasing
  limits (requests/min, tokens/day, monthly cost cap).
* **Quota enforcement** — :meth:`TenantBillingManager.check` returns an
  :class:`AllowDeny` decision.  ``quota_middleware`` uses it to return HTTP
  429 with a clear body when a tenant exceeds its plan.
* **Usage aggregation → invoice** — E12 ``UsageRecord`` objects are grouped
  per tenant into invoice line items (per model), with real subtotals.
* **Per-tenant isolation** — every counter/quota/tally is keyed by
  ``tenant_id``; tenant A's usage never affects tenant B.

Only the final Stripe *submission* is a stub (inherited from E12).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from distllm.core.cost_tracker import get_cost_tracker
from distllm.core.metering import (
    BillingExporter,
    MeteringStore,
    UsageRecord,
    get_metering_store,
)


# ── Tier plans ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TierPlan:
    """A subscription tier with per-tenant limits.

    A limit of ``0`` means "unlimited" for that dimension.

    Attributes:
        name: Plan name (``free`` / ``pro`` / ``enterprise``).
        requests_per_min: Max requests allowed in any rolling 60s window.
        tokens_per_day: Max total tokens (in + out) per UTC calendar day.
        monthly_cost_cap_usd: Max accrued USD cost per UTC calendar month.
    """

    name: str
    requests_per_min: int
    tokens_per_day: int
    monthly_cost_cap_usd: float


# Ordered from least to most permissive: free < pro < enterprise.
TIER_PLANS: dict[str, TierPlan] = {
    "free": TierPlan(
        name="free",
        requests_per_min=20,
        tokens_per_day=100_000,
        monthly_cost_cap_usd=10.0,
    ),
    "pro": TierPlan(
        name="pro",
        requests_per_min=300,
        tokens_per_day=5_000_000,
        monthly_cost_cap_usd=1_000.0,
    ),
    "enterprise": TierPlan(
        name="enterprise",
        requests_per_min=5_000,
        tokens_per_day=500_000_000,
        monthly_cost_cap_usd=100_000.0,
    ),
}

DEFAULT_TIER = "free"


# ── Decision object ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AllowDeny:
    """Result of a quota check.

    ``allowed`` is ``True`` for *Allow* and ``False`` for *Deny*.  On deny,
    ``reason`` is a human-readable explanation and ``status_code`` is 429.
    """

    allowed: bool
    reason: str = "ok"
    status_code: int = 200

    def __bool__(self) -> bool:  # allow `if decision:` truthiness
        return self.allowed

    @classmethod
    def allow(cls, reason: str = "ok") -> "AllowDeny":
        return cls(allowed=True, reason=reason, status_code=200)

    @classmethod
    def deny(cls, reason: str) -> "AllowDeny":
        return cls(allowed=False, reason=reason, status_code=429)


# ── Time-window helpers ────────────────────────────────────────────────────

def _day_bounds(ts: float | None = None) -> tuple[float, float]:
    """Return [start, end) epoch seconds for the UTC day containing ``ts``."""
    now = datetime.fromtimestamp(ts if ts is not None else time.time(), tz=timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_ts = start.timestamp()
    return start_ts, start_ts + 86400.0


def _month_bounds(ts: float | None = None) -> tuple[float, float]:
    """Return [start, end) epoch seconds for the UTC month containing ``ts``."""
    now = datetime.fromtimestamp(ts if ts is not None else time.time(), tz=timezone.utc)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if start.month == 12:
        nxt = start.replace(year=start.year + 1, month=1)
    else:
        nxt = start.replace(month=start.month + 1)
    return start.timestamp(), nxt.timestamp()


def _month_label(ts: float | None = None) -> str:
    now = datetime.fromtimestamp(ts if ts is not None else time.time(), tz=timezone.utc)
    return now.strftime("%Y-%m")


# ── TenantBillingManager ───────────────────────────────────────────────────

class TenantBillingManager:
    """Per-tenant quotas, 429 enforcement, and invoice aggregation.

    Reuses the E12 :class:`MeteringStore` as the authoritative usage store and
    the cost tracker for canonical cost math.  All state is keyed by
    ``tenant_id`` for strict per-tenant isolation.

    Usage::

        mgr = TenantBillingManager()
        mgr.set_tier("acme", "pro")
        decision = mgr.check("acme", requested_tokens=1000)
        if decision.allowed:
            mgr.record_usage("acme", tokens_in=800, tokens_out=200,
                             model_name="llama-70b")
        invoice = mgr.build_invoice("acme")
    """

    def __init__(
        self,
        store: MeteringStore | None = None,
        cost_tracker: Any | None = None,
        exporter: BillingExporter | None = None,
    ) -> None:
        # Authoritative usage store (E12).  Falls back to the singleton.
        self._store = store or get_metering_store()
        # Canonical cost math (reused, not re-created).
        self._tracker = cost_tracker or get_cost_tracker()
        self._exporter = exporter or BillingExporter()

        self._lock = threading.RLock()
        # tenant_id -> tier name
        self._tiers: dict[str, str] = {}
        # tenant_id -> optional custom plan override (takes precedence over tier)
        self._custom_plans: dict[str, TierPlan] = {}
        # tenant_id -> rolling request timestamps (for requests/min)
        self._req_windows: dict[str, list[float]] = {}

    # ── Tier / plan management ─────────────────────────────────────────────

    def set_tier(self, tenant_id: str, tier: str) -> None:
        """Assign a tenant to a named tier (``free`` / ``pro`` / ``enterprise``)."""
        if tier not in TIER_PLANS:
            raise ValueError(f"unknown tier {tier!r}; valid: {sorted(TIER_PLANS)}")
        with self._lock:
            self._tiers[tenant_id] = tier
            self._custom_plans.pop(tenant_id, None)

    def set_custom_plan(self, tenant_id: str, plan: TierPlan) -> None:
        """Override a tenant's limits with a bespoke plan."""
        with self._lock:
            self._custom_plans[tenant_id] = plan

    def get_plan(self, tenant_id: str) -> TierPlan:
        """Resolve the effective plan for a tenant (custom > tier > default)."""
        with self._lock:
            custom = self._custom_plans.get(tenant_id)
            if custom is not None:
                return custom
            tier = self._tiers.get(tenant_id, DEFAULT_TIER)
        return TIER_PLANS[tier]

    def get_tier(self, tenant_id: str) -> str:
        with self._lock:
            return self._tiers.get(tenant_id, DEFAULT_TIER)

    # ── Enforcement ────────────────────────────────────────────────────────

    def check(
        self,
        tenant_id: str,
        request_cost: float = 0.0,
        requested_tokens: int = 0,
        *,
        consume: bool = True,
    ) -> AllowDeny:
        """Decide whether a request from ``tenant_id`` is allowed.

        Enforces, in order: requests/min, tokens/day, monthly cost cap.  Each
        check compares the tenant's *current* usage (aggregated from the E12
        store) plus the incoming request against the tenant's plan.

        Args:
            tenant_id: The tenant making the request.
            request_cost: Estimated USD cost of the incoming request (added to
                the month-to-date cost before comparing to the cap).
            requested_tokens: Total tokens (in + out) the incoming request will
                consume (added to today's tokens before comparing to the cap).
            consume: When ``True`` (default) and the request is allowed, record
                a timestamp in the rolling requests/min window.  Set ``False``
                for a dry-run check.

        Returns:
            An :class:`AllowDeny` decision (status 200 allow / 429 deny).
        """
        plan = self.get_plan(tenant_id)
        now = time.time()

        # 1) requests/min — rolling 60s window, isolated per tenant.
        if plan.requests_per_min > 0:
            with self._lock:
                window = self._req_windows.setdefault(tenant_id, [])
                window[:] = [t for t in window if now - t < 60.0]
                if len(window) >= plan.requests_per_min:
                    return AllowDeny.deny(
                        f"rate limit {plan.requests_per_min} requests/min exceeded "
                        f"(tier={plan.name})"
                    )

        # 2) tokens/day — aggregate today's tokens from the E12 store.
        if plan.tokens_per_day > 0:
            today_tokens = self._tokens_today(tenant_id, now)
            if today_tokens + max(0, requested_tokens) > plan.tokens_per_day:
                return AllowDeny.deny(
                    f"daily token limit {plan.tokens_per_day} exceeded "
                    f"(used={today_tokens}, tier={plan.name})"
                )

        # 3) monthly cost cap — aggregate month-to-date cost from the store.
        if plan.monthly_cost_cap_usd > 0:
            month_cost = self._cost_this_month(tenant_id, now)
            if month_cost + max(0.0, request_cost) > plan.monthly_cost_cap_usd:
                return AllowDeny.deny(
                    f"monthly cost cap ${plan.monthly_cost_cap_usd:.2f} exceeded "
                    f"(used=${month_cost:.4f}, tier={plan.name})"
                )

        if consume and plan.requests_per_min > 0:
            with self._lock:
                self._req_windows.setdefault(tenant_id, []).append(now)

        return AllowDeny.allow(f"within {plan.name} plan limits")

    def estimate_request_cost(
        self, input_tokens: int, output_tokens: int, model_name: str = ""
    ) -> tuple[float, float]:
        """Return ``(cost_usd, compute_s)`` for a request via the cost tracker."""
        est = self._tracker.estimate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_name=model_name,
        )
        return float(est.estimated_cost_usd), float(est.estimated_gpu_seconds)

    # ── Recording ──────────────────────────────────────────────────────────

    def record_usage(
        self,
        tenant_id: str,
        *,
        tokens_in: int,
        tokens_out: int,
        model_name: str = "",
        endpoint: str = "",
        request_id: str = "",
        cost_usd: float | None = None,
        compute_s: float | None = None,
        timestamp: float | None = None,
    ) -> UsageRecord:
        """Record a completed request into the E12 store for this tenant.

        Cost / compute are taken from the cost tracker unless explicitly
        supplied, so figures match the rest of the platform.
        """
        if cost_usd is None or compute_s is None:
            est_cost, est_compute = self.estimate_request_cost(
                tokens_in, tokens_out, model_name
            )
            cost_usd = est_cost if cost_usd is None else cost_usd
            compute_s = est_compute if compute_s is None else compute_s

        return self._store.record_request(
            tenant_id=tenant_id,
            tokens_in=int(tokens_in),
            tokens_out=int(tokens_out),
            compute_s=float(compute_s),
            cost_usd=float(cost_usd),
            model_name=model_name,
            endpoint=endpoint,
            request_id=request_id,
            timestamp=timestamp,
        )

    # ── Aggregation → invoice ──────────────────────────────────────────────

    def _records_in_month(self, tenant_id: str, ts: float | None = None) -> list[UsageRecord]:
        start, end = _month_bounds(ts)
        return [
            r
            for r in self._store.records_for_tenant(tenant_id)
            if start <= r.timestamp < end
        ]

    def _tokens_today(self, tenant_id: str, ts: float | None = None) -> int:
        start, end = _day_bounds(ts)
        return sum(
            r.total_tokens
            for r in self._store.records_for_tenant(tenant_id)
            if start <= r.timestamp < end
        )

    def _cost_this_month(self, tenant_id: str, ts: float | None = None) -> float:
        return round(sum(r.cost_usd for r in self._records_in_month(tenant_id, ts)), 8)

    def aggregate_line_items(
        self, tenant_id: str, ts: float | None = None
    ) -> list[dict[str, Any]]:
        """Aggregate a tenant's month-to-date usage into per-model line items."""
        buckets: dict[str, dict[str, Any]] = {}
        for r in self._records_in_month(tenant_id, ts):
            key = r.model_name or "unknown"
            b = buckets.setdefault(
                key,
                {
                    "model_name": key,
                    "requests": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "total_tokens": 0,
                    "compute_seconds": 0.0,
                    "cost_usd": 0.0,
                },
            )
            b["requests"] += 1
            b["tokens_in"] += r.tokens_in
            b["tokens_out"] += r.tokens_out
            b["total_tokens"] += r.total_tokens
            b["compute_seconds"] += r.compute_s
            b["cost_usd"] += r.cost_usd
        for b in buckets.values():
            b["compute_seconds"] = round(b["compute_seconds"], 8)
            b["cost_usd"] = round(b["cost_usd"], 8)
        return sorted(buckets.values(), key=lambda x: x["model_name"])

    def build_invoice(
        self, tenant_id: str, period: str | None = None, ts: float | None = None
    ) -> dict[str, Any]:
        """Build a per-tenant invoice for the current UTC month.

        Reuses E12's :class:`BillingExporter` for the base document (Stripe
        export stays a stub) and augments it with tier metadata and the
        per-model aggregated line items produced by :meth:`aggregate_line_items`.
        """
        period = period or _month_label(ts)
        records = self._records_in_month(tenant_id, ts)
        invoice = self._exporter.export_invoice(tenant_id, records, period=period)

        plan = self.get_plan(tenant_id)
        invoice["tier"] = plan.name
        invoice["plan_limits"] = {
            "requests_per_min": plan.requests_per_min,
            "tokens_per_day": plan.tokens_per_day,
            "monthly_cost_cap_usd": plan.monthly_cost_cap_usd,
        }
        invoice["line_items_by_model"] = self.aggregate_line_items(tenant_id, ts)
        return invoice

    def tenant_summary(self, tenant_id: str, ts: float | None = None) -> dict[str, Any]:
        """Return a compact per-tenant usage summary for the current month."""
        plan = self.get_plan(tenant_id)
        records = self._records_in_month(tenant_id, ts)
        return {
            "tenant_id": tenant_id,
            "tier": plan.name,
            "requests": len(records),
            "tokens_today": self._tokens_today(tenant_id, ts),
            "tokens_month": sum(r.total_tokens for r in records),
            "cost_month_usd": round(sum(r.cost_usd for r in records), 8),
            "monthly_cost_cap_usd": plan.monthly_cost_cap_usd,
        }

    def reset_rate_windows(self) -> None:
        """Clear rolling requests/min windows (used by tests)."""
        with self._lock:
            self._req_windows.clear()


# ── Module-level singleton ─────────────────────────────────────────────────

_manager: TenantBillingManager | None = None
_manager_lock = threading.Lock()


def get_tenant_billing_manager() -> TenantBillingManager:
    """Get or create the module-level :class:`TenantBillingManager`."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = TenantBillingManager()
    return _manager


def reset_tenant_billing_manager() -> None:
    """Reset the singleton (used by tests to get a clean manager)."""
    global _manager
    with _manager_lock:
        _manager = None
