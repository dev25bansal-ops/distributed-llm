"""Budget scheduler for cost-constrained cloud inference.

Ensures total spot instance spend stays within a configurable
per-hour budget. Automatically scales down or switches to cheaper
providers when approaching budget limits.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from loguru import logger

from distllm.cloud.spot_provider import (
    CloudProvider,
    SpotPrice,
    SpotProvider,
)
from distllm.cloud.preemption_predictor import PreemptionPredictor


class BudgetAction(str, Enum):
    SCALE_DOWN = "scale_down"
    SWITCH_PROVIDER = "switch_provider"
    SWITCH_INSTANCE = "switch_instance"
    HALT_PROVISIONING = "halt_provisioning"


@dataclass
class BudgetDecision:
    """A decision made by the budget scheduler."""
    action: BudgetAction
    reason: str
    estimated_savings: float = 0.0
    current_cost: float = 0.0
    budget_limit: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class BudgetScheduler:
    """Enforces per-hour budget across all cloud providers.

    Strategy:
    1. Track accumulated spend per instance
    2. When approaching budget limits, take cost-saving actions
    3. Prefer switching providers over terminating workloads
    4. Use preemption risk as a secondary signal for cost decisions

    Args:
        budget_per_hour: Maximum spend per hour in USD.
        warning_threshold: Fraction of budget that triggers warnings (default 0.8).
        critical_threshold: Fraction that forces cost-saving actions (default 0.9).
        check_interval_s: How often to re-evaluate budget (seconds).
        predictor: PreemptionPredictor instance for risk-aware decisions.
        on_decision: Optional callback fired on each BudgetDecision.
    """

    def __init__(
        self,
        budget_per_hour: float = 0.0,
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.9,
        check_interval_s: float = 60.0,
        predictor: PreemptionPredictor | None = None,
        on_decision: Callable[[BudgetDecision], None] | None = None,
    ):
        self.budget_per_hour = budget_per_hour
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.check_interval_s = check_interval_s
        self.predictor = predictor or PreemptionPredictor()
        self.on_decision = on_decision

        self._instances: dict[str, dict[str, Any]] = {}
        self._spend_tracker: dict[str, float] = {}
        self._decisions: list[BudgetDecision] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def register_instance(
        self,
        instance_id: str,
        provider: CloudProvider,
        instance_type: str,
        region: str,
        price_per_hour: float,
    ) -> None:
        """Register a running spot instance for budget tracking."""
        with self._lock:
            self._instances[instance_id] = {
                "instance_id": instance_id,
                "provider": provider,
                "instance_type": instance_type,
                "region": region,
                "price_per_hour": price_per_hour,
                "launched_at": time.time(),
            }
            logger.debug(
                f"Registered {instance_id} ({provider.value}/{instance_type}) "
                f"at ${price_per_hour:.4f}/hr"
            )

    def unregister_instance(self, instance_id: str) -> None:
        """Remove a terminated instance from budget tracking."""
        with self._lock:
            instance = self._instances.pop(instance_id, None)
            if instance:
                elapsed = time.time() - instance["launched_at"]
                cost = (elapsed / 3600.0) * instance["price_per_hour"]
                self._spend_tracker[instance_id] = (
                    self._spend_tracker.get(instance_id, 0.0) + cost
                )
                logger.debug(
                    f"Unregistered {instance_id}, accrued ${cost:.4f}"
                )

    def current_spend(self) -> float:
        """Calculate current hourly spend across all instances."""
        with self._lock:
            total = sum(
                inst["price_per_hour"]
                for inst in self._instances.values()
            )
        return total

    def accumulated_spend(self) -> float:
        """Return total accumulated spend (historical + current)."""
        with self._lock:
            return sum(self._spend_tracker.values()) + self.current_spend()

    def _evaluate(self) -> list[BudgetDecision]:
        """Evaluate budget and return actions to take."""
        decisions: list[BudgetDecision] = []
        if self.budget_per_hour <= 0:
            return decisions

        current = self.current_spend()
        ratio = current / self.budget_per_hour if self.budget_per_hour > 0 else 0.0

        if ratio >= self.critical_threshold:
            decision = self._find_cost_savings()
            if decision:
                decisions.append(decision)

        elif ratio >= self.warning_threshold:
            decision = BudgetDecision(
                action=BudgetAction.SWITCH_PROVIDER,
                reason=f"Budget warning: ${current:.2f}/hr ({ratio*100:.0f}% of limit)",
                current_cost=current,
                budget_limit=self.budget_per_hour,
            )
            decisions.append(decision)
            logger.warning(decision.reason)

        return decisions

    def _find_cost_savings(self) -> BudgetDecision | None:
        """Find the best action to reduce costs when over budget."""
        if not self._instances:
            return None

        current = self.current_spend()
        overage = current - self.budget_per_hour * self.critical_threshold

        # Sort by price (highest first)
        sorted_instances = sorted(
            self._instances.values(),
            key=lambda i: i["price_per_hour"],
            reverse=True,
        )

        # Check if we can switch the most expensive instance to a cheaper type
        most_expensive = sorted_instances[0]
        cheaper_exists = any(
            inst["price_per_hour"] < most_expensive["price_per_hour"] * 0.8
            for inst in sorted_instances[1:]
        )

        if cheaper_exists:
            return BudgetDecision(
                action=BudgetAction.SWITCH_INSTANCE,
                reason=f"Switching {most_expensive['instance_id']} to cheaper instance",
                estimated_savings=most_expensive["price_per_hour"] * 0.3,
                current_cost=current,
                budget_limit=self.budget_per_hour,
            )

        # Scale down the least critical instance
        candidates = sorted(
            self._instances.values(),
            key=lambda i: (
                self.predictor.predict(
                    i["provider"], i["instance_type"], i["region"]
                ).risk_score
                if self.predictor
                else 0.5
            ),
            reverse=True,
        )

        if candidates:
            target = candidates[0]
            return BudgetDecision(
                action=BudgetAction.SCALE_DOWN,
                reason=f"Budget exceeded: terminating {target['instance_id']}",
                estimated_savings=target["price_per_hour"],
                current_cost=current,
                budget_limit=self.budget_per_hour,
            )

        return None

    def get_decisions(self, limit: int = 20) -> list[BudgetDecision]:
        return self._decisions[-limit:]

    def start(self) -> None:
        """Start the background budget evaluation loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(
            f"Budget scheduler started: ${self.budget_per_hour:.2f}/hr limit"
        )

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        while self._running:
            decisions = self._evaluate()
            for decision in decisions:
                self._decisions.append(decision)
                if self.on_decision:
                    try:
                        self.on_decision(decision)
                    except Exception as e:
                        logger.error(f"Budget decision callback failed: {e}")
            time.sleep(self.check_interval_s)
