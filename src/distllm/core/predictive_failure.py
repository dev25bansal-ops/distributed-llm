"""Predictive failure detection — detect GPU health issues before they cause failures.

Monitors GPU signals (ECC errors, thermal throttling, memory fragmentation,
clock speed reduction) and returns a failure probability score.

Usage::

    detector = PredictiveFailureDetector()
    prob = detector.check_gpu_health("node-0", {
        "ecc_uncorrectable": 0,
        "thermal_throttle": True,
        "memory_fragmentation_pct": 45,
        "clock_throttle_pct": 10,
    })
    if prob > 0.7:
        logger.warning(f"Node likely to fail soon (prob={prob:.2f})")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GPUSignal:
    """A single GPU health signal with weight and threshold."""
    name: str
    weight: float
    threshold: float
    description: str = ""


# Default signal configuration
DEFAULT_SIGNALS = [
    GPUSignal("ecc_uncorrectable", 0.95, 0, "Any uncorrectable ECC error = imminent failure"),
    GPUSignal("thermal_throttle", 0.40, 0, "Thermal throttling active"),
    GPUSignal("memory_fragmentation_pct", 0.30, 80, "Memory fragmentation above 80%"),
    GPUSignal("clock_throttle_pct", 0.25, 20, "Clock speed reduced by >20%"),
    GPUSignal("power_throttle", 0.20, 0, "Power limit throttling"),
    GPUSignal("xid_errors", 0.35, 1, "XID errors (GPU hardware errors)"),
]


@dataclass
class FailurePrediction:
    """Result of a predictive failure check."""
    node_id: str
    failure_probability: float  # 0.0-1.0
    signals: list[tuple[str, float]]  # (signal_name, contribution)
    recommendation: str  # "ok", "monitor", "preemptive_drain", "immediate_drain"
    timestamp: float = field(default_factory=time.time)


class PredictiveFailureDetector:
    """Detects early warning signs of node failure.

    Uses configurable weighted signals to compute a failure probability.
    No ML dependencies — pure threshold-based heuristics.
    """

    def __init__(self, signals: list[GPUSignal] | None = None):
        self._signals = signals or DEFAULT_SIGNALS
        self._history: dict[str, list[FailurePrediction]] = {}
        self._max_history = 50

    def check_gpu_health(
        self, node_id: str, gpu_data: dict[str, Any]
    ) -> FailurePrediction:
        """Evaluate GPU health signals and return failure probability.

        Args:
            node_id: Node identifier.
            gpu_data: Dict of GPU metrics (ecc_uncorrectable, thermal_throttle, etc.)

        Returns:
            FailurePrediction with probability 0.0-1.0 and recommendation.
        """
        # ECC uncorrectable errors are an immediate red flag
        if gpu_data.get("ecc_uncorrectable", 0) > 0:
            pred = FailurePrediction(
                node_id=node_id,
                failure_probability=0.95,
                signals=[("ecc_uncorrectable", 0.95)],
                recommendation="immediate_drain",
            )
            self._record(node_id, pred)
            return pred

        total_weight = 0.0
        weighted_sum = 0.0
        active_signals = []

        for sig in self._signals:
            value = gpu_data.get(sig.name)
            if value is None:
                continue

            # Determine if signal is active
            triggered = False
            if isinstance(value, bool):
                triggered = value
            elif isinstance(value, (int, float)):
                triggered = value > sig.threshold

            if triggered:
                # Scale contribution by how far above threshold
                if isinstance(value, (int, float)) and sig.threshold > 0:
                    scale = min(value / sig.threshold, 3.0) / 3.0
                    contribution = sig.weight * (0.5 + 0.5 * scale)
                else:
                    contribution = sig.weight
                weighted_sum += contribution
                total_weight += sig.weight
                active_signals.append((sig.name, contribution))

        # Normalize to 0.0-1.0
        if total_weight > 0:
            probability = min(weighted_sum / max(total_weight, 1.0), 1.0)
        else:
            probability = 0.0

        # Determine recommendation
        if probability >= 0.8:
            recommendation = "immediate_drain"
        elif probability >= 0.5:
            recommendation = "preemptive_drain"
        elif probability >= 0.2:
            recommendation = "monitor"
        else:
            recommendation = "ok"

        pred = FailurePrediction(
            node_id=node_id,
            failure_probability=round(probability, 3),
            signals=active_signals,
            recommendation=recommendation,
        )
        self._record(node_id, pred)
        return pred

    def _record(self, node_id: str, pred: FailurePrediction) -> None:
        if node_id not in self._history:
            self._history[node_id] = []
        self._history[node_id].append(pred)
        if len(self._history[node_id]) > self._max_history:
            self._history[node_id] = self._history[node_id][-self._max_history:]

    def get_history(self, node_id: str) -> list[FailurePrediction]:
        return list(self._history.get(node_id, []))

    def get_trending_nodes(self, threshold: float = 0.3) -> list[str]:
        """Return node IDs with recent average probability above threshold."""
        trending = []
        for node_id, preds in self._history.items():
            if len(preds) < 3:
                continue
            recent = preds[-5:]
            avg = sum(p.failure_probability for p in recent) / len(recent)
            if avg >= threshold:
                trending.append(node_id)
        return trending
