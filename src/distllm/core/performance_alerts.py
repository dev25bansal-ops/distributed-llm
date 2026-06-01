"""Performance degradation alerts for distributed LLM inference.

Monitors throughput, latency, and error rates against baselines
and triggers alerts when metrics degrade beyond thresholds.

Usage::

    alerts = PerformanceAlertManager()
    alerts.set_baseline("throughput", 50.0)  # 50 tokens/sec
    alerts.record("throughput", 45.0)
    # alerts.check() returns alert if below threshold
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from loguru import logger


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceAlert:
    """A performance degradation alert."""
    metric: str
    severity: AlertSeverity
    baseline: float
    current: float
    degradation_pct: float
    message: str
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False


@dataclass
class MetricBaseline:
    """Baseline configuration for a metric."""
    name: str
    baseline_value: float
    warning_threshold: float = 0.8   # 80% of baseline
    critical_threshold: float = 0.5  # 50% of baseline
    higher_is_better: bool = True    # False for latency/error rate
    window_size: int = 10            # Number of recent values to average


class PerformanceAlertManager:
    """Monitors performance metrics and triggers degradation alerts.

    Tracks metrics against configurable baselines and generates
    alerts when performance drops below thresholds.
    """

    def __init__(
        self,
        on_alert: Callable[[PerformanceAlert], None] | None = None,
    ):
        self._baselines: dict[str, MetricBaseline] = {}
        self._values: dict[str, deque[float]] = {}
        self._alerts: list[PerformanceAlert] = []
        self._on_alert = on_alert
        self._lock = threading.Lock()

    def set_baseline(
        self,
        metric: str,
        baseline_value: float,
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.5,
        higher_is_better: bool = True,
        window_size: int = 10,
    ) -> None:
        """Set a baseline for a metric.

        Args:
            metric: Metric name (e.g., "throughput", "latency_p99").
            baseline_value: Expected normal value.
            warning_threshold: Fraction of baseline that triggers warning.
            critical_threshold: Fraction of baseline that triggers critical alert.
            higher_is_better: True if higher values are better (throughput).
            window_size: Number of recent values to average.
        """
        with self._lock:
            self._baselines[metric] = MetricBaseline(
                name=metric,
                baseline_value=baseline_value,
                warning_threshold=warning_threshold,
                critical_threshold=critical_threshold,
                higher_is_better=higher_is_better,
                window_size=window_size,
            )
            self._values[metric] = deque(maxlen=window_size)

    def record(self, metric: str, value: float) -> None:
        """Record a metric value."""
        with self._lock:
            if metric not in self._values:
                self._values[metric] = deque(maxlen=10)
            self._values[metric].append(value)

    def check(self) -> list[PerformanceAlert]:
        """Check all metrics against baselines and return new alerts.

        Returns:
            List of new PerformanceAlert objects.
        """
        new_alerts = []

        with self._lock:
            for metric, baseline in self._baselines.items():
                values = self._values.get(metric, deque())
                if len(values) < 3:  # Need at least 3 values
                    continue

                avg_value = sum(values) / len(values)
                baseline_val = baseline.baseline_value

                if baseline_val == 0:
                    continue

                if baseline.higher_is_better:
                    ratio = avg_value / baseline_val
                else:
                    # For latency/errors, lower is better
                    ratio = baseline_val / avg_value if avg_value > 0 else 1.0

                severity = None
                if ratio <= baseline.critical_threshold:
                    severity = AlertSeverity.CRITICAL
                elif ratio <= baseline.warning_threshold:
                    severity = AlertSeverity.WARNING

                if severity is not None:
                    degradation_pct = (1 - ratio) * 100
                    alert = PerformanceAlert(
                        metric=metric,
                        severity=severity,
                        baseline=baseline_val,
                        current=avg_value,
                        degradation_pct=degradation_pct,
                        message=(
                            f"{metric} degraded by {degradation_pct:.0f}%: "
                            f"baseline={baseline_val:.1f}, current={avg_value:.1f}"
                        ),
                    )
                    new_alerts.append(alert)
                    self._alerts.append(alert)

                    if self._on_alert:
                        try:
                            self._on_alert(alert)
                        except Exception as e:
                            logger.warning(f"Alert callback failed: {e}")

                    logger.warning(f"Performance alert: {alert.message}")

        return new_alerts

    def get_alerts(
        self,
        severity: AlertSeverity | None = None,
        limit: int = 50,
    ) -> list[PerformanceAlert]:
        """Get recent alerts, optionally filtered by severity."""
        with self._lock:
            alerts = list(self._alerts)
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            return alerts[-limit:]

    def acknowledge(self, index: int) -> bool:
        """Acknowledge an alert by index."""
        with self._lock:
            if 0 <= index < len(self._alerts):
                self._alerts[index].acknowledged = True
                return True
            return False

    def stats(self) -> dict[str, Any]:
        """Return alert statistics."""
        with self._lock:
            return {
                "baselines": len(self._baselines),
                "total_alerts": len(self._alerts),
                "unacknowledged": sum(1 for a in self._alerts if not a.acknowledged),
                "critical": sum(1 for a in self._alerts if a.severity == AlertSeverity.CRITICAL),
                "warning": sum(1 for a in self._alerts if a.severity == AlertSeverity.WARNING),
            }
