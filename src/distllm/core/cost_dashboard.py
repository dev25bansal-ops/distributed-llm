"""Cost dashboard for per-request cost tracking and budget management.

Tracks inference costs per user, per model, and per time period.
Provides budget alerts and monthly projections.

Usage::

    dashboard = CostDashboard()
    dashboard.record_cost(user_id="user-1", model="llama-3-70b", tokens=1000, cost_usd=0.05)
    report = dashboard.get_report(user_id="user-1")
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class CostRecord:
    """A single cost record."""
    user_id: str
    model: str
    tokens: int
    cost_usd: float
    timestamp: float = field(default_factory=time.time)
    request_id: str = ""


@dataclass
class BudgetAlert:
    """A budget alert."""
    user_id: str
    threshold_pct: float
    current_spend: float
    budget_usd: float
    message: str
    timestamp: float = field(default_factory=time.time)


class CostDashboard:
    """Per-request cost tracking with budget management."""

    def __init__(
        self,
        default_budget_usd: float = 100.0,
        alert_thresholds: list[float] | None = None,
    ):
        self._default_budget = default_budget_usd
        self._alert_thresholds = alert_thresholds or [0.5, 0.75, 0.9, 1.0]
        self._records: list[CostRecord] = []
        self._budgets: dict[str, float] = {}  # user_id -> budget
        self._alerted: dict[str, set[float]] = {}  # user_id -> set of threshold %
        self._lock = threading.Lock()

    def set_budget(self, user_id: str, budget_usd: float) -> None:
        """Set monthly budget for a user."""
        with self._lock:
            self._budgets[user_id] = budget_usd

    def record_cost(
        self,
        user_id: str,
        model: str,
        tokens: int,
        cost_usd: float,
        request_id: str = "",
    ) -> BudgetAlert | None:
        """Record a cost and check budget alerts.

        Returns:
            BudgetAlert if a threshold was crossed, else None.
        """
        record = CostRecord(
            user_id=user_id,
            model=model,
            tokens=tokens,
            cost_usd=cost_usd,
            request_id=request_id,
        )

        with self._lock:
            self._records.append(record)

            # Check budget
            budget = self._budgets.get(user_id, self._default_budget)
            monthly_spend = self._get_monthly_spend(user_id)
            ratio = monthly_spend / budget if budget > 0 else 0

            # Check alert thresholds
            alerted = self._alerted.get(user_id, set())
            for threshold in self._alert_thresholds:
                if ratio >= threshold and threshold not in alerted:
                    if user_id not in self._alerted:
                        self._alerted[user_id] = set()
                    self._alerted[user_id].add(threshold)

                    alert = BudgetAlert(
                        user_id=user_id,
                        threshold_pct=threshold,
                        current_spend=monthly_spend,
                        budget_usd=budget,
                        message=f"User {user_id} has spent {ratio:.0%} of ${budget:.2f} budget (${monthly_spend:.2f})",
                    )
                    logger.warning(f"Budget alert: {alert.message}")
                    return alert

        return None

    def get_report(self, user_id: str | None = None) -> dict:
        """Generate a cost report.

        Args:
            user_id: If provided, report for this user. Otherwise, aggregate.

        Returns:
            Cost report dict.
        """
        with self._lock:
            if user_id:
                records = [r for r in self._records if r.user_id == user_id]
            else:
                records = self._records

            total_cost = sum(r.cost_usd for r in records)
            total_tokens = sum(r.tokens for r in records)

            # Per-model breakdown
            by_model: dict[str, float] = {}
            for r in records:
                by_model[r.model] = by_model.get(r.model, 0) + r.cost_usd

            # Monthly spend
            monthly = self._get_monthly_spend(user_id) if user_id else total_cost
            budget = self._budgets.get(user_id, self._default_budget) if user_id else 0

            return {
                "total_cost_usd": round(total_cost, 4),
                "total_tokens": total_tokens,
                "monthly_spend_usd": round(monthly, 4),
                "budget_usd": budget,
                "budget_used_pct": round(monthly / budget * 100, 1) if budget > 0 else 0,
                "by_model": {k: round(v, 4) for k, v in by_model.items()},
                "record_count": len(records),
            }

    def get_projection(self, user_id: str) -> dict:
        """Project monthly cost based on current usage rate."""
        with self._lock:
            monthly = self._get_monthly_spend(user_id)
            now = time.time()
            month_start = now - (30 * 24 * 3600)  # Approximate

            # Find earliest record this month
            user_records = [r for r in self._records if r.user_id == user_id and r.timestamp > month_start]
            if not user_records:
                return {"projected_usd": 0, "days_remaining": 30}

            earliest = min(r.timestamp for r in user_records)
            elapsed_days = max((now - earliest) / 86400, 1)
            daily_rate = monthly / elapsed_days
            days_in_month = 30
            projected = daily_rate * days_in_month

            budget = self._budgets.get(user_id, self._default_budget)
            days_until_budget = budget / daily_rate if daily_rate > 0 else float("inf")

            return {
                "projected_usd": round(projected, 2),
                "daily_rate_usd": round(daily_rate, 4),
                "days_until_budget_exhausted": round(days_until_budget, 1),
                "budget_usd": budget,
            }

    def _get_monthly_spend(self, user_id: str | None) -> float:
        """Get spend for the current month."""
        now = time.time()
        month_start = now - (30 * 24 * 3600)
        return sum(
            r.cost_usd
            for r in self._records
            if r.timestamp > month_start and (user_id is None or r.user_id == user_id)
        )

    def reset_alerts(self, user_id: str) -> None:
        """Reset alert thresholds for a user (e.g., at month start)."""
        with self._lock:
            self._alerted.pop(user_id, None)

    def get_forecast(self, user_id: str | None = None) -> dict:
        """Generate a detailed cost forecast with trend analysis.

        Returns:
            Dict with projected costs, daily breakdown, trend, and confidence.
        """
        with self._lock:
            now = time.time()
            month_start = now - (30 * 24 * 3600)

            records = [
                r for r in self._records
                if r.timestamp > month_start and (user_id is None or r.user_id == user_id)
            ]

            if not records:
                return {
                    "projected_monthly_usd": 0.0,
                    "current_spend_usd": 0.0,
                    "daily_average_usd": 0.0,
                    "trend": "stable",
                    "confidence": "low",
                    "days_with_data": 0,
                    "by_model": {},
                }

            # Daily breakdown
            daily_costs: dict[str, float] = {}
            for r in records:
                day = time.strftime("%Y-%m-%d", time.localtime(r.timestamp))
                daily_costs[day] = daily_costs.get(day, 0) + r.cost_usd

            days_with_data = len(daily_costs)
            total_spend = sum(r.cost_usd for r in records)
            daily_avg = total_spend / max(days_with_data, 1)

            # Trend: compare first half vs second half
            sorted_days = sorted(daily_costs.items())
            mid = len(sorted_days) // 2
            if mid > 0:
                first_half = sum(v for _, v in sorted_days[:mid]) / mid
                second_half = sum(v for _, v in sorted_days[mid:]) / max(len(sorted_days) - mid, 1)
                if second_half > first_half * 1.2:
                    trend = "increasing"
                elif second_half < first_half * 0.8:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            # Confidence based on data points
            if days_with_data >= 14:
                confidence = "high"
            elif days_with_data >= 7:
                confidence = "medium"
            else:
                confidence = "low"

            # Project to 30 days
            projected = daily_avg * 30

            # By model breakdown
            by_model: dict[str, float] = {}
            for r in records:
                by_model[r.model] = by_model.get(r.model, 0) + r.cost_usd

            budget = self._budgets.get(user_id or "", self._default_budget)

            return {
                "projected_monthly_usd": round(projected, 2),
                "current_spend_usd": round(total_spend, 4),
                "daily_average_usd": round(daily_avg, 4),
                "daily_breakdown": {k: round(v, 4) for k, v in sorted_days},
                "trend": trend,
                "confidence": confidence,
                "days_with_data": days_with_data,
                "budget_usd": budget,
                "budget_used_pct": round(total_spend / budget * 100, 1) if budget > 0 else 0,
                "by_model": {k: round(v, 4) for k, v in sorted(by_model.items(), key=lambda x: -x[1])},
            }


class SLAManager:
    """Define and enforce latency/throughput SLOs with automatic remediation."""

    def __init__(self):
        self._slos: dict[str, dict] = {}  # name -> SLO config
        self._violations: list[dict] = []
        self._lock = threading.Lock()

    def define_slo(
        self,
        name: str,
        metric: str,
        threshold: float,
        window_seconds: float = 60.0,
        violation_threshold: int = 3,
        remediation: str = "alert",
    ) -> None:
        """Define an SLO.

        Args:
            name: SLO name.
            metric: Metric to track (latency_p99, throughput, error_rate).
            threshold: Threshold value.
            window_seconds: Evaluation window.
            violation_threshold: Number of violations before remediation.
            remediation: Remediation action (alert, scale, restart).
        """
        with self._lock:
            self._slos[name] = {
                "metric": metric,
                "threshold": threshold,
                "window_seconds": window_seconds,
                "violation_threshold": violation_threshold,
                "remediation": remediation,
                "violations": 0,
                "last_check": time.time(),
            }

    def evaluate(self, name: str, current_value: float) -> dict | None:
        """Evaluate an SLO against a current metric value.

        Returns:
            Violation dict if SLO is breached, else None.
        """
        with self._lock:
            slo = self._slos.get(name)
            if slo is None:
                return None

            breached = current_value > slo["threshold"]
            if breached:
                slo["violations"] += 1
                if slo["violations"] >= slo["violation_threshold"]:
                    violation = {
                        "slo_name": name,
                        "metric": slo["metric"],
                        "threshold": slo["threshold"],
                        "actual": current_value,
                        "violations": slo["violations"],
                        "remediation": slo["remediation"],
                        "timestamp": time.time(),
                    }
                    self._violations.append(violation)
                    slo["violations"] = 0  # Reset
                    return violation
            else:
                slo["violations"] = 0

            return None

    def get_status(self) -> dict:
        """Get status of all SLOs."""
        with self._lock:
            return {
                name: {
                    "metric": slo["metric"],
                    "threshold": slo["threshold"],
                    "violations": slo["violations"],
                    "remediation": slo["remediation"],
                }
                for name, slo in self._slos.items()
            }

    def get_violations(self, limit: int = 50) -> list[dict]:
        """Get recent SLO violations."""
        with self._lock:
            return self._violations[-limit:]
