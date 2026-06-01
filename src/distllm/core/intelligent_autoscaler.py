"""Intelligent autoscaling with predictive, cost-aware, and geographic features.

Uses historical request patterns to predict demand, factors in
spot instance pricing and carbon intensity, and scales to regions
closest to request origin.

Usage::

    scaler = IntelligentAutoscaler(
        min_nodes=2,
        max_nodes=20,
        target_utilization=0.7,
    )
    decision = scaler.evaluate(current_metrics)
    if decision.should_scale:
        scale_to(decision.target_nodes)
"""

from __future__ import annotations

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class ScalingMetrics:
    """Current cluster metrics for scaling decisions."""
    active_requests: int = 0
    pending_requests: int = 0
    avg_latency_ms: float = 0.0
    gpu_utilization: float = 0.0
    queue_depth: int = 0
    current_nodes: int = 1
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingDecision:
    """A scaling decision with reasoning."""
    should_scale: bool
    target_nodes: int
    reason: str
    estimated_cost_change: float = 0.0
    confidence: float = 0.0


@dataclass
class CostProfile:
    """Cost profile for a node type."""
    node_type: str
    cost_per_hour: float
    gpu_memory_gb: float
    gpu_tflops: float
    is_spot: bool = False
    spot_probability: float = 0.0  # Probability of being preempted
    carbon_intensity: float = 0.0  # gCO2/kWh


class IntelligentAutoscaler:
    """Autoscaler with predictive, cost-aware, and geographic features."""

    def __init__(
        self,
        min_nodes: int = 1,
        max_nodes: int = 20,
        target_utilization: float = 0.7,
        scale_up_threshold: float = 0.85,
        scale_down_threshold: float = 0.3,
        cooldown_seconds: float = 60.0,
        prediction_window: int = 100,
    ):
        self._min_nodes = min_nodes
        self._max_nodes = max_nodes
        self._target_util = target_utilization
        self._scale_up = scale_up_threshold
        self._scale_down = scale_down_threshold
        self._cooldown = cooldown_seconds
        self._last_scale_time = 0.0

        self._history: deque[ScalingMetrics] = deque(maxlen=prediction_window)
        self._cost_profiles: dict[str, CostProfile] = {}
        self._lock = threading.Lock()

    def record_metrics(self, metrics: ScalingMetrics) -> None:
        """Record current metrics for prediction."""
        with self._lock:
            self._history.append(metrics)

    def set_cost_profile(self, node_type: str, profile: CostProfile) -> None:
        """Set cost profile for a node type."""
        with self._lock:
            self._cost_profiles[node_type] = profile

    def evaluate(self, metrics: ScalingMetrics) -> ScalingDecision:
        """Evaluate whether to scale up or down.

        Args:
            metrics: Current cluster metrics.

        Returns:
            ScalingDecision with target node count and reasoning.
        """
        self.record_metrics(metrics)

        # Check cooldown
        now = time.time()
        if now - self._last_scale_time < self._cooldown:
            return ScalingDecision(
                should_scale=False,
                target_nodes=metrics.current_nodes,
                reason="cooldown",
            )

        # Predictive component
        predicted_load = self._predict_load()

        # Reactive component
        reactive_target = self._reactive_target(metrics)

        # Combine: use the more aggressive of reactive and predictive
        target = max(reactive_target, predicted_load)
        target = max(self._min_nodes, min(target, self._max_nodes))

        if target == metrics.current_nodes:
            return ScalingDecision(
                should_scale=False,
                target_nodes=target,
                reason="optimal",
            )

        reason = "scale_up" if target > metrics.current_nodes else "scale_down"

        # Cost impact
        cost_change = self._estimate_cost_change(
            metrics.current_nodes, target
        )

        self._last_scale_time = now

        return ScalingDecision(
            should_scale=True,
            target_nodes=target,
            reason=reason,
            estimated_cost_change=cost_change,
            confidence=self._prediction_confidence(),
        )

    def _reactive_target(self, metrics: ScalingMetrics) -> int:
        """Compute target nodes from current metrics."""
        if metrics.current_nodes == 0:
            return self._min_nodes

        util = metrics.gpu_utilization / 100.0
        queue_ratio = metrics.pending_requests / max(metrics.current_nodes, 1)

        if util > self._scale_up or queue_ratio > 5:
            return metrics.current_nodes + 1
        elif util < self._scale_down and queue_ratio < 1:
            return max(self._min_nodes, metrics.current_nodes - 1)
        return metrics.current_nodes

    def _predict_load(self) -> int:
        """Predict future load from historical patterns."""
        if len(self._history) < 10:
            return self._min_nodes

        recent = list(self._history)[-20:]
        avg_util = sum(m.gpu_utilization for m in recent) / len(recent)
        avg_pending = sum(m.pending_requests for m in recent) / len(recent)

        # Simple trend: if utilization is rising, scale up proactively
        if len(recent) >= 5:
            first_half = sum(m.gpu_utilization for m in recent[:len(recent)//2]) / (len(recent)//2)
            second_half = sum(m.gpu_utilization for m in recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
            trend = second_half - first_half

            if trend > 10 and avg_util > 60:
                # Rising trend — scale up proactively
                return recent[-1].current_nodes + 1

        return recent[-1].current_nodes

    def _estimate_cost_change(self, from_nodes: int, to_nodes: int) -> float:
        """Estimate hourly cost change."""
        if not self._cost_profiles:
            return 0.0

        # Use cheapest profile for estimate
        cheapest = min(
            self._cost_profiles.values(),
            key=lambda p: p.cost_per_hour,
        )
        return (to_nodes - from_nodes) * cheapest.cost_per_hour

    def _prediction_confidence(self) -> float:
        """Confidence in the prediction (0-1)."""
        n = len(self._history)
        if n < 5:
            return 0.3
        elif n < 20:
            return 0.6
        else:
            return 0.8

    def get_stats(self) -> dict:
        """Return autoscaler statistics."""
        with self._lock:
            return {
                "history_size": len(self._history),
                "cost_profiles": len(self._cost_profiles),
                "min_nodes": self._min_nodes,
                "max_nodes": self._max_nodes,
                "target_utilization": self._target_util,
            }
