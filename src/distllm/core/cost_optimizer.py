"""Real-time cost optimizer with spot pricing and ROI dashboard.

Ties together arbitrage engine, cost tracker, and pricing providers
to provide:
- Real-time cloud spot price monitoring
- Automatic migration to cheapest provider
- Cost budget alerts with escalation
- ROI dashboard per model

Usage::

    optimizer = CostOptimizer(
        cost_tracker=get_cost_tracker(),
        pricing_manager=PricingManager(),
    )
    optimizer.start()
    report = optimizer.get_roi_report()
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger


@dataclass
class ModelROI:
    """ROI metrics for a single model."""
    model_name: str
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    total_requests: int = 0
    avg_cost_per_token: float = 0.0
    avg_cost_per_request: float = 0.0
    throughput_tok_s: float = 0.0
    cloud_equivalent_cost: float = 0.0
    savings_vs_cloud: float = 0.0
    savings_pct: float = 0.0
    uptime_pct: float = 100.0


@dataclass
class CostAlert:
    """A cost optimization alert."""
    alert_type: str  # "budget_exceeded", "price_spike", "migration_opportunity"
    severity: str    # "info", "warning", "critical"
    model: str = ""
    message: str = ""
    current_cost: float = 0.0
    threshold: float = 0.0
    recommendation: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProviderCost:
    """Cost comparison across providers."""
    provider: str
    region: str
    instance_type: str
    spot_price: float
    on_demand_price: float
    estimated_monthly: float
    availability: float  # 0-1


class CostOptimizer:
    """Real-time cost optimization with spot pricing and ROI tracking.

    Monitors:
    - Per-model cost and ROI
    - Spot pricing across providers
    - Budget utilization and alerts
    - Migration opportunities
    """

    def __init__(
        self,
        cost_tracker: Any = None,
        pricing_manager: Any = None,
        arbitrage_engine: Any = None,
        budget_alert_callback: Callable[[CostAlert], None] | None = None,
        check_interval_s: float = 60.0,
        cloud_cost_per_1k_tokens: float = 0.0,
    ):
        """Initialize the cost optimizer.

        Args:
            cost_tracker: Optional cost tracker instance.
            pricing_manager: Optional pricing manager instance.
            arbitrage_engine: Optional arbitrage engine instance.
            budget_alert_callback: Optional callback for budget alerts.
            check_interval_s: Interval between optimization checks.
            cloud_cost_per_1k_tokens: USD cost per 1k tokens for the
                equivalent cloud/hosted inference (used to estimate
                ``cloud_equivalent_cost`` and savings in ROI reports).
                ``0.0`` disables cloud-cost estimation.
        """
        self._cost_tracker = cost_tracker
        self._pricing = pricing_manager
        self._arbitrage = arbitrage_engine
        self._on_alert = budget_alert_callback
        self._check_interval = check_interval_s
        self._cloud_rate_per_1k = cloud_cost_per_1k_tokens

        self._model_costs: dict[str, dict] = {}  # model -> cost data
        self._alerts: list[CostAlert] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stats = {
            "checks": 0,
            "alerts_generated": 0,
            "migrations_recommended": 0,
            "total_savings": 0.0,
        }

    def start(self) -> None:
        """Start the cost optimizer background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._optimization_loop,
            daemon=True,
            name="cost-optimizer",
        )
        self._thread.start()
        logger.info("Cost optimizer started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=self._check_interval * 2)

    def record_model_cost(
        self,
        model_name: str,
        cost_usd: float,
        tokens: int,
        requests: int = 1,
    ) -> None:
        """Record cost data for a model.

        When a ``cloud_cost_per_1k_tokens`` rate is configured, the
        cloud-equivalent cost is estimated from the recorded token count so
        ROI reports can compare self-hosted vs hosted inference.
        """
        with self._lock:
            if model_name not in self._model_costs:
                self._model_costs[model_name] = {
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "total_requests": 0,
                    "cloud_cost": 0.0,
                }
            mc = self._model_costs[model_name]
            mc["total_cost"] += cost_usd
            mc["total_tokens"] += tokens
            mc["total_requests"] += requests
            if self._cloud_rate_per_1k > 0:
                mc["cloud_cost"] += (tokens / 1000) * self._cloud_rate_per_1k

    def get_roi_report(self) -> dict:
        """Generate ROI report for all tracked models."""
        with self._lock:
            models = []
            total_cost = 0.0
            total_savings = 0.0

            for model_name, data in self._model_costs.items():
                cost = data["total_cost"]
                tokens = data["total_tokens"]
                requests = data["total_requests"]
                cloud_cost = data.get("cloud_cost", 0)

                roi = ModelROI(
                    model_name=model_name,
                    total_cost_usd=round(cost, 4),
                    total_tokens=tokens,
                    total_requests=requests,
                    avg_cost_per_token=round(cost / max(tokens, 1), 6),
                    avg_cost_per_request=round(cost / max(requests, 1), 4),
                    cloud_equivalent_cost=round(cloud_cost, 4),
                    savings_vs_cloud=round(cloud_cost - cost, 4),
                    savings_pct=round(
                        ((cloud_cost - cost) / max(cloud_cost, 0.001)) * 100, 1
                    ),
                )
                models.append(roi.__dict__)
                total_cost += cost
                total_savings += max(cloud_cost - cost, 0)

            return {
                "models": models,
                "total_cost_usd": round(total_cost, 4),
                "total_savings_usd": round(total_savings, 4),
                "total_tokens": sum(d["total_tokens"] for d in self._model_costs.values()),
                "total_requests": sum(d["total_requests"] for d in self._model_costs.values()),
                "alerts": len(self._alerts),
                "recent_alerts": [
                    {
                        "type": a.alert_type,
                        "severity": a.severity,
                        "message": a.message,
                        "recommendation": a.recommendation,
                        "timestamp": a.timestamp,
                    }
                    for a in self._alerts[-10:]
                ],
            }

    def check_budgets(self, budgets: dict[str, float]) -> list[CostAlert]:
        """Check model costs against budgets and generate alerts."""
        alerts = []
        with self._lock:
            for model_name, data in self._model_costs.items():
                budget = budgets.get(model_name, 0)
                if budget <= 0:
                    continue

                cost = data["total_cost"]
                utilization = cost / budget

                if utilization >= 1.0:
                    alert = CostAlert(
                        alert_type="budget_exceeded",
                        severity="critical",
                        model=model_name,
                        message=f"Budget exceeded for {model_name}: ${cost:.2f} / ${budget:.2f}",
                        current_cost=cost,
                        threshold=budget,
                        recommendation="Reduce usage, switch to smaller model, or increase budget",
                    )
                    alerts.append(alert)
                elif utilization >= 0.9:
                    alert = CostAlert(
                        alert_type="budget_warning",
                        severity="warning",
                        model=model_name,
                        message=f"Budget 90% used for {model_name}: ${cost:.2f} / ${budget:.2f}",
                        current_cost=cost,
                        threshold=budget,
                        recommendation="Consider switching to smaller model for non-critical requests",
                    )
                    alerts.append(alert)

        for alert in alerts:
            self._alerts.append(alert)
            self._stats["alerts_generated"] += 1
            if self._on_alert:
                try:
                    self._on_alert(alert)
                except Exception as e:
                    logger.warning(f"Alert callback failed: {e}")

        return alerts

    def get_migration_recommendations(self) -> list[dict]:
        """Get cost-saving migration recommendations from arbitrage engine."""
        if self._arbitrage is None:
            return []

        try:
            opportunities = self._arbitrage.detect_opportunities()
            recommendations = []
            for opp in opportunities:
                d = opp.to_dict()
                recommendations.append({
                    "type": d.get("type", ""),
                    "current_provider": opp.current_provider,
                    "recommended_provider": opp.recommended_provider,
                    "estimated_savings": d.get("savings_per_hour", 0),
                    "risk": d.get("migration_risk", "low"),
                })
                self._stats["migrations_recommended"] += 1
            return recommendations
        except Exception as e:
            logger.debug(f"Migration check failed: {e}")
            return []

    def _optimization_loop(self) -> None:
        """Background loop for cost optimization checks."""
        while self._running:
            try:
                self._stats["checks"] += 1

                # Update spot pricing if available
                if self._pricing:
                    try:
                        self._pricing.refresh()
                    except Exception:
                        pass

                # Check for arbitrage opportunities
                if self._arbitrage:
                    self.get_migration_recommendations()

            except Exception as e:
                logger.debug(f"Cost optimization check failed: {e}")

            deadline = time.time() + self._check_interval
            while self._running and time.time() < deadline:
                time.sleep(1.0)

    def stats(self) -> dict:
        with self._lock:
            return {
                **self._stats,
                "tracked_models": len(self._model_costs),
                "total_alerts": len(self._alerts),
                "optimizer_running": self._running,
            }
