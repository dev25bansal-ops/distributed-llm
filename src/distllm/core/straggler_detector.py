"""Detect and mitigate slow nodes in the pipeline.

Monitors per-node latency, detects stragglers using statistical analysis,
and triggers mitigation actions. Standalone module that integrates with
the existing Rebalancer for layer reassignment.

Detection methods:
- Threshold-based: node exceeds p95 latency of others
- MAD-based: median absolute deviation outlier detection
- Trend-based: sustained latency increase over time
- Skip rate: tokens/second below expected throughput
"""

from __future__ import annotations

import math
import statistics
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from loguru import logger


class DetectionMethod(Enum):
    THRESHOLD = "threshold"       # Node exceeds p95 of others
    MAD = "mad"                   # Median absolute deviation
    TREND = "trend"               # Sustained increase over baseline
    THROUGHPUT = "throughput"     # Below expected tokens/second


class StragglerSeverity(Enum):
    NONE = "none"
    MILD = "mild"           # 1.5-2x slower
    MODERATE = "moderate"   # 2-3x slower
    SEVERE = "severe"       # 3x+ slower


@dataclass
class NodeTiming:
    node_id: str
    latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    throughputs: deque = field(default_factory=lambda: deque(maxlen=100))
    last_seen: float = 0.0
    consecutive_slow: int = 0
    is_straggler: bool = False
    severity: StragglerSeverity = StragglerSeverity.NONE
    baseline_latency: float = 0.0
    baseline_throughput: float = 0.0

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / max(len(self.latencies), 1)

    @property
    def avg_throughput(self) -> float:
        return sum(self.throughputs) / max(len(self.throughputs), 1)

    @property
    def p95_latency(self) -> float:
        if len(self.latencies) < 2:
            return self.avg_latency
        sorted_lats = sorted(self.latencies)
        idx = int(len(sorted_lats) * 0.95)
        return sorted_lats[min(idx, len(sorted_lats) - 1)]


@dataclass
class StragglerReport:
    node_id: str
    severity: StragglerSeverity
    avg_latency: float
    p95_latency: float
    baseline_latency: float
    slowdown_factor: float
    detection_method: DetectionMethod
    consecutive_detections: int
    recommended_action: str


class StragglerDetector:
    """Detects and mitigates slow nodes in the pipeline.

    Usage:
        detector = StragglerDetector(
            on_straggler_cb=lambda report: print(report),
        )

        # During inference:
        detector.record_latency("node_1", 45.0)
        detector.record_throughput("node_1", 120.0)

        if detector.check():
            for report in detector.get_reports():
                print(report)
    """

    def __init__(
        self,
        on_straggler_cb: Callable[[StragglerReport], None] | None = None,
        detection_method: DetectionMethod = DetectionMethod.MAD,
        slow_threshold_ms: float = 100.0,
        consecutive_threshold: int = 3,
        window_size: int = 50,
        mad_threshold: float = 2.0,
        check_interval_s: float = 10.0,
    ):
        self._on_straggler = on_straggler_cb
        self._detection_method = detection_method
        self._slow_threshold = slow_threshold_ms
        self._consecutive_threshold = consecutive_threshold
        self._window_size = window_size
        self._mad_threshold = mad_threshold
        self._check_interval = check_interval_s

        self._nodes: dict[str, NodeTiming] = {}
        self._lock = threading.Lock()
        self._last_check = 0.0
        self._total_checks = 0
        self._total_detections = 0

    def record_latency(self, node_id: str, latency_ms: float) -> None:
        """Record a single forward latency for a node."""
        with self._lock:
            if node_id not in self._nodes:
                self._nodes[node_id] = NodeTiming(node_id=node_id)
            node = self._nodes[node_id]
            node.latencies.append(latency_ms)
            node.last_seen = time.time()

            # Update baseline after first 10 samples
            if len(node.latencies) == 10:
                node.baseline_latency = statistics.median(node.latencies)

    def record_throughput(self, node_id: str, tokens_per_second: float) -> None:
        """Record throughput for a node."""
        with self._lock:
            if node_id not in self._nodes:
                self._nodes[node_id] = NodeTiming(node_id=node_id)
            node = self._nodes[node_id]
            node.throughputs.append(tokens_per_second)

            if len(node.throughputs) == 10:
                node.baseline_throughput = statistics.median(node.throughputs)

    def record_batch(
        self,
        node_id: str,
        latency_ms: float,
        tokens_generated: int = 0,
        batch_size: int = 0,
    ) -> None:
        """Record a full batch timing for a node."""
        self.record_latency(node_id, latency_ms)
        if tokens_generated > 0 and latency_ms > 0:
            self.record_throughput(node_id, tokens_generated / (latency_ms / 1000.0))

    def check(self) -> list[StragglerReport]:
        """Run straggler detection on all tracked nodes.

        Returns list of StragglerReport for nodes flagged as stragglers.
        """
        now = time.time()
        if now - self._last_check < self._check_interval:
            return []

        self._last_check = now
        self._total_checks += 1
        reports: list[StragglerReport] = []

        with self._lock:
            if len(self._nodes) < 2:
                return []

            all_latencies = []
            for node in self._nodes.values():
                if len(node.latencies) >= 5:
                    all_latencies.extend(node.latencies)

            if not all_latencies:
                return []

            median_all = statistics.median(all_latencies)
            p95_all = sorted(all_latencies)[int(len(all_latencies) * 0.95)]

            for node_id, node in self._nodes.items():
                if len(node.latencies) < 5:
                    continue

                is_slow = False
                method = self._detection_method
                severity = StragglerSeverity.NONE

                if method == DetectionMethod.THRESHOLD:
                    is_slow = node.avg_latency > p95_all * 1.5
                elif method == DetectionMethod.MAD:
                    devs = [abs(l - median_all) for l in all_latencies]
                    mad = statistics.median(devs) if devs else 0
                    if mad > 0:
                        is_slow = abs(node.avg_latency - median_all) / mad > self._mad_threshold
                elif method == DetectionMethod.TREND:
                    if node.baseline_latency > 0:
                        is_slow = node.avg_latency > node.baseline_latency * 1.5
                elif method == DetectionMethod.THROUGHPUT:
                    if node.baseline_throughput > 0:
                        is_slow = node.avg_throughput < node.baseline_throughput * 0.5

                if is_slow:
                    slowdown = node.avg_latency / max(node.baseline_latency, median_all, 1)
                    if slowdown > 3.0:
                        severity = StragglerSeverity.SEVERE
                    elif slowdown > 2.0:
                        severity = StragglerSeverity.MODERATE
                    elif slowdown > 1.5:
                        severity = StragglerSeverity.MILD

                    node.consecutive_slow += 1
                else:
                    node.consecutive_slow = 0

                node.is_straggler = node.consecutive_slow >= self._consecutive_threshold
                node.severity = severity if node.is_straggler else StragglerSeverity.NONE

                if node.is_straggler:
                    action = self._recommend_action(severity)
                    report = StragglerReport(
                        node_id=node_id,
                        severity=severity,
                        avg_latency=round(node.avg_latency, 2),
                        p95_latency=round(node.p95_latency, 2),
                        baseline_latency=round(node.baseline_latency, 2),
                        slowdown_factor=round(node.avg_latency / max(node.baseline_latency, 1), 2),
                        detection_method=method,
                        consecutive_detections=node.consecutive_slow,
                        recommended_action=action,
                    )
                    reports.append(report)
                    self._total_detections += 1

        for report in reports:
            logger.warning(
                f"Straggler detected: {report.node_id} "
                f"({report.severity.value}, {report.slowdown_factor}x slower, "
                f"action: {report.recommended_action})"
            )
            if self._on_straggler:
                try:
                    self._on_straggler(report)
                except Exception as e:
                    logger.error(f"Straggler callback failed: {e}")

        return reports

    def _recommend_action(self, severity: StragglerSeverity) -> str:
        if severity == StragglerSeverity.SEVERE:
            return "reassign_layers"
        elif severity == StragglerSeverity.MODERATE:
            return "reduce_batch"
        return "monitor_only"

    def get_reports(self) -> list[StragglerReport]:
        self.check()
        reports = []
        with self._lock:
            for node in self._nodes.values():
                if node.is_straggler:
                    reports.append(StragglerReport(
                        node_id=node.node_id,
                        severity=node.severity,
                        avg_latency=round(node.avg_latency, 2),
                        p95_latency=round(node.p95_latency, 2),
                        baseline_latency=round(node.baseline_latency, 2),
                        slowdown_factor=round(node.avg_latency / max(node.baseline_latency, 1), 2),
                        detection_method=self._detection_method,
                        consecutive_detections=node.consecutive_slow,
                        recommended_action="",
                    ))
        return reports

    def clear_node(self, node_id: str) -> None:
        with self._lock:
            self._nodes.pop(node_id, None)

    def reset_all(self) -> None:
        with self._lock:
            self._nodes.clear()
            self._total_checks = 0
            self._total_detections = 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            active = len(self._nodes)
            stragglers = sum(1 for n in self._nodes.values() if n.is_straggler)
            return {
                "active_nodes": active,
                "straggler_nodes": stragglers,
                "detection_method": self._detection_method.value,
                "total_checks": self._total_checks,
                "total_detections": self._total_detections,
                "nodes": {
                    nid: {
                        "avg_latency": round(node.avg_latency, 2),
                        "p95_latency": round(node.p95_latency, 2),
                        "avg_throughput": round(node.avg_throughput, 1),
                        "consecutive_slow": node.consecutive_slow,
                        "is_straggler": node.is_straggler,
                        "severity": node.severity.value,
                    }
                    for nid, node in self._nodes.items()
                },
            }
