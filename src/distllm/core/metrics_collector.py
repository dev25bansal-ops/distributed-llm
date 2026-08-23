"""Metrics collection from all subsystems."""

from typing import Any

from distllm.dist.latency import LatencyTracker
from distllm.dist.straggler import StragglerDetector
from distllm.dist.recovery import NodeRecoveryManager


class MetricsCollector:
    """Collects and aggregates metrics from latency trackers, straggler
    detectors, and recovery managers."""

    def __init__(
        self,
        latency_tracker: LatencyTracker | None = None,
        straggler_detector: StragglerDetector | None = None,
        recovery_manager: NodeRecoveryManager | None = None,
    ):
        self._latency_tracker = latency_tracker
        self._straggler_detector = straggler_detector
        self._recovery_manager = recovery_manager
        self._counters: dict[str, float] = {}

    def record(self, name: str, value: float = 1.0) -> None:
        """Accumulate a named counter (RequestHandler.record_metric calls this)."""
        self._counters[name] = self._counters.get(name, 0.0) + value

    def collect(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        if self._latency_tracker:
            metrics["latency"] = self._latency_tracker.get_all_avg()
        if self._straggler_detector:
            metrics["straggler"] = self._straggler_detector.stats()
        if self._recovery_manager:
            metrics["recovery"] = self._recovery_manager.get_metrics()
        if self._counters:
            metrics["counters"] = dict(self._counters)
        return metrics
