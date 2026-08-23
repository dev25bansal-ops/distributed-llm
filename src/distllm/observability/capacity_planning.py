"""Capacity planning, what-if analysis, HPA/VPA config, Grafana dashboard,
and GPU billing for distributed-llm.

Provides six components:

* **CapacityPlanner** — tracks GPU / memory / request-rate / token-throughput
  time-series data, computes growth trends via linear regression, and detects
  scaling triggers (>80 % utilisation for >1 hour).
  ``plan()`` returns actionable scaling recommendations.

* **WhatIfCalculator** — uses the linear model from historical data to answer
  "what if" questions: adding GPUs, upgrading GPU types, or changing batch
  sizes.

* **HPAConfig** — generates Kubernetes HPA and VPA YAML manifests for a
  service, driven by CPU, memory, and requests-per-second metrics.

* **CapacityDashboard** — generates a complete, importable Grafana dashboard
  JSON with panels for usage trends (7d / 30d / 90d), forecast, scaling
  recommendations, and GPU-hour cost projection.

* **GPUBilling** — tracks GPU usage sessions per node, computes GPU-hours per
  period, calculates cost against configurable rates, and produces CSV / JSON
  reports.

* **CapacityConfigurator** — top-level orchestrator that wires the five
  components together.  ``start()`` begins periodic data collection,
  ``stop()`` halts it, and ``generate_report()`` writes a comprehensive
  capacity-planning report to disk.
"""

from __future__ import annotations

import csv
import dataclasses
import io
import json
import math
import threading
import warnings
from collections import deque
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, IO, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Optional yaml — no-op fallback when not installed
# ---------------------------------------------------------------------------

try:
    from yaml import dump as yaml_dump

    YAML_AVAILABLE = True
except ImportError:  # pragma: no cover
    YAML_AVAILABLE = False

    def yaml_dump(*args: Any, **kwargs: Any) -> str:  # type: ignore[misc]
        """Fallback no-op for pyyaml.dump."""
        return "# pyyaml not available\n"


import numpy as np

# ===================================================================
# Constants
# ===================================================================

_HOUR_SECONDS = 3600
_DAY_SECONDS = 86400
_DEFAULT_MAX_SNAPSHOT_AGE_DAYS = 90
_UTILIZATION_TRIGGER_THRESHOLD = 0.80
_UTILIZATION_TRIGGER_WINDOW = timedelta(hours=1)
_GRAFANA_PANEL_H = 8
_GRAFANA_PANEL_W = 8
_GRAFANA_ROW_GAP = 1

GPU_PERFORMANCE_INDEX: Dict[str, float] = {
    "A100": 1.0,
    "A100-80GB": 1.05,
    "V100": 0.6,
    "T4": 0.3,
    "L4": 0.45,
    "H100": 1.8,
    "H100-80GB": 1.9,
    "H200": 2.0,
    "RTX-3090": 0.5,
    "RTX-4090": 0.7,
    "A10G": 0.55,
    "A6000": 0.65,
    "unknown": 0.5,
}


# ===================================================================
# Custom exceptions
# ===================================================================


class CapacityPlanningError(Exception):
    """Base exception for capacity-planning module errors."""


class InsufficientDataError(CapacityPlanningError):
    """Raised when there is not enough historical data for an operation."""


class InvalidConfigurationError(CapacityPlanningError):
    """Raised when configuration parameters are invalid."""


# ===================================================================
# Data models
# ===================================================================


@dataclasses.dataclass(frozen=True)
class GPUSnapshot:
    """A single GPU utilisation measurement.

    Attributes:
        node_id: Identifies the worker node.
        gpu_type: GPU model string (e.g. ``"A100"``, ``"H100"``).
        gpu_utilization: GPU compute utilisation as a fraction in [0, 1].
        memory_used: GPU memory used in bytes.
        memory_total: Total GPU memory in bytes.
        timestamp: When the measurement was taken.
    """

    node_id: str
    gpu_type: str
    gpu_utilization: float
    memory_used: int
    memory_total: int
    timestamp: datetime


@dataclasses.dataclass(frozen=True)
class PerformanceSnapshot:
    """Aggregate cluster performance at a point in time.

    Attributes:
        timestamp: When the snapshot was captured.
        request_rate: In-flight request rate (req/s).
        token_throughput: Token generation throughput (tokens/s).
        avg_latency_ms: Average request latency in milliseconds.
        active_nodes: Number of nodes currently serving.
        gpu_snapshots: Per-GPU utilisation data.
    """

    timestamp: datetime
    request_rate: float
    token_throughput: float
    avg_latency_ms: float
    active_nodes: int
    gpu_snapshots: Tuple[GPUSnapshot, ...]


@dataclasses.dataclass(frozen=True)
class TrendResult:
    """Linear regression result for a metric over time.

    Attributes:
        slope: Slope of the fitted line (units per second).
        intercept: Intercept value at epoch 0.
        r_squared: Coefficient of determination in [0, 1].
        forecast_7d: Projected value 7 days from the latest data point.
        forecast_30d: Projected value 30 days from the latest data point.
        forecast_90d: Projected value 90 days from the latest data point.
    """

    slope: float
    intercept: float
    r_squared: float
    forecast_7d: float
    forecast_30d: float
    forecast_90d: float


class RecommendationPriority(str, Enum):
    """Priority level for a scaling recommendation."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecommendationType(str, Enum):
    """Category of scaling recommendation."""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    UPGRADE_GPU = "upgrade_gpu"
    OPTIMIZE_BATCH = "optimize_batch"
    ADD_NODE = "add_node"
    REPLACE_NODE = "replace_node"


@dataclasses.dataclass(frozen=True)
class ScalingRecommendation:
    """A single actionable scaling recommendation.

    Attributes:
        recommendation_type: Category of the recommendation.
        priority: Urgency level.
        title: Short human-readable title.
        description: Detailed explanation.
        suggested_action: Concrete action the operator should take.
        estimated_impact: Qualitative description of expected impact.
        metric_evidence: Dict mapping metric names to current values
            that triggered this recommendation.
    """

    recommendation_type: RecommendationType
    priority: RecommendationPriority
    title: str
    description: str
    suggested_action: str
    estimated_impact: str
    metric_evidence: Dict[str, Any]


@dataclasses.dataclass(frozen=True)
class GPUUsageSession:
    """A recorded period of GPU usage for billing.

    Attributes:
        session_id: Unique identifier for this session.
        node_id: Worker node identifier.
        gpu_type: GPU model string.
        start_time: When usage began.
        end_time: When usage ended (or ``None`` if still running).
    """

    session_id: str
    node_id: str
    gpu_type: str
    start_time: datetime
    end_time: Optional[datetime]


@dataclasses.dataclass(frozen=True)
class CapacityPlanReport:
    """Comprehensive output of ``CapacityPlanner.plan()``.

    Attributes:
        generated_at: When the report was generated.
        gpu_trend: Trend for average GPU utilisation, or ``None``.
        memory_trend: Trend for average GPU memory utilisation, or
            ``None``.
        request_rate_trend: Trend for request rate, or ``None``.
        throughput_trend: Trend for token throughput, or ``None``.
        active_nodes: Current number of active nodes.
        scaling_triggers: List of triggered scaling alerts.
        recommendations: Full list of recommendations.
        snapshot_count: Number of historical snapshots used.
    """

    generated_at: datetime
    gpu_trend: Optional[TrendResult]
    memory_trend: Optional[TrendResult]
    request_rate_trend: Optional[TrendResult]
    throughput_trend: Optional[TrendResult]
    active_nodes: int
    scaling_triggers: List[Dict[str, Any]]
    recommendations: List[ScalingRecommendation]
    snapshot_count: int


# ===================================================================
# Internal helpers
# ===================================================================


def _now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def _to_timestamp_seconds(dt: datetime) -> float:
    """Convert a datetime to Unix epoch seconds.

    Ensures the datetime is timezone-aware and normalised to UTC.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _linear_regression(
    x: np.ndarray, y: np.ndarray
) -> Tuple[float, float, float]:
    """Fit a line ``y = slope * x + intercept`` and compute R-squared.

    Args:
        x: Predictor array (e.g. epoch seconds).
        y: Response array (e.g. utilisation values).

    Returns:
        ``(slope, intercept, r_squared)`` tuple.  Returns ``(0, mean, 0)``
        when there are fewer than 2 data points.
    """
    if len(x) < 2:
        mean_y = float(np.mean(y)) if len(y) > 0 else 0.0
        return 0.0, mean_y, 0.0

    slope, intercept = np.polyfit(x, y, 1)
    y_pred = np.polyval([slope, intercept], x)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(slope), float(intercept), r_squared


def _gpu_hours(start: datetime, end: Optional[datetime]) -> float:
    """Compute GPU-hours between *start* and *end*.

    Args:
        start: Session start time.
        end: Session end time, or ``None`` for "now".

    Returns:
        GPU-hours as a float.
    """
    end_dt = end if end is not None else _now()
    delta = end_dt - start
    return max(0.0, delta.total_seconds() / _HOUR_SECONDS)


def _next_id(counter: List[int]) -> int:
    """Increment a single-element list counter and return the new value."""
    counter[0] += 1
    return counter[0]


# ===================================================================
# CapacityPlanner
# ===================================================================


class CapacityPlanner:
    """Tracks capacity-related metrics and produces scaling recommendations.

    Maintains a rolling window of ``PerformanceSnapshot`` data (up to 90 days
    by default).  Periodically (or on demand) computes linear regression trends
    for GPU utilisation, memory pressure, request rate, and token throughput.

    Scaling triggers are raised when average GPU utilisation exceeds 80 % for
    more than one hour.  ``plan()`` synthesises all signals into a list of
    ``ScalingRecommendation`` instances.

    Thread-safe for concurrent recording and inspection.
    """

    def __init__(
        self,
        max_snapshot_age: timedelta = timedelta(days=_DEFAULT_MAX_SNAPSHOT_AGE_DAYS),
        util_threshold: float = _UTILIZATION_TRIGGER_THRESHOLD,
        util_window: timedelta = _UTILIZATION_TRIGGER_WINDOW,
    ) -> None:
        """Initialise the capacity planner.

        Args:
            max_snapshot_age: Maximum age for retained snapshots.  Older
                snapshots are pruned during ``record_snapshot``.  Defaults
                to 90 days.
            util_threshold: GPU utilisation fraction that triggers a
                scaling alert (default 0.80).
            util_window: Duration over which utilisation is averaged for
                trigger evaluation (default 1 hour).

        Raises:
            InvalidConfigurationError: If thresholds are out of range.
        """
        if not 0.0 < util_threshold <= 1.0:
            raise InvalidConfigurationError(
                f"util_threshold must be in (0, 1], got {util_threshold}"
            )
        if util_window <= timedelta(0):
            raise InvalidConfigurationError(
                f"util_window must be positive, got {util_window}"
            )
        if max_snapshot_age <= timedelta(0):
            raise InvalidConfigurationError(
                f"max_snapshot_age must be positive, got {max_snapshot_age}"
            )

        self._max_snapshot_age = max_snapshot_age
        self._util_threshold = util_threshold
        self._util_window = util_window
        self._lock = threading.Lock()
        self._snapshots: List[PerformanceSnapshot] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def util_threshold(self) -> float:
        """GPU utilisation trigger threshold as a fraction."""
        return self._util_threshold

    @property
    def util_window(self) -> timedelta:
        """Duration for trigger evaluation."""
        return self._util_window

    @property
    def snapshot_count(self) -> int:
        """Number of historical snapshots retained."""
        with self._lock:
            return len(self._snapshots)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        """Record a single performance snapshot.

        Old snapshots outside the retention window are pruned on each call.

        Args:
            snapshot: The performance snapshot to record.
        """
        with self._lock:
            cutoff = _to_timestamp_seconds(
                _now() - self._max_snapshot_age
            )
            self._snapshots = [
                s
                for s in self._snapshots
                if _to_timestamp_seconds(s.timestamp) >= cutoff
            ]
            self._snapshots.append(snapshot)

    def record_gpu_snapshot(
        self,
        node_id: str,
        gpu_type: str,
        gpu_utilization: float,
        memory_used: int,
        memory_total: int,
        request_rate: float = 0.0,
        token_throughput: float = 0.0,
        avg_latency_ms: float = 0.0,
        active_nodes: int = 1,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Convenience method to record a single-GPU snapshot.

        Builds a ``PerformanceSnapshot`` containing one ``GPUSnapshot`` and
        passes it to ``record_snapshot``.

        Args:
            node_id: Worker node identifier.
            gpu_type: GPU model string.
            gpu_utilization: GPU utilisation as a fraction in [0, 1].
            memory_used: GPU memory used in bytes.
            memory_total: Total GPU memory in bytes.
            request_rate: Current request rate (req/s).
            token_throughput: Current token throughput (tokens/s).
            avg_latency_ms: Average latency in milliseconds.
            active_nodes: Number of active nodes.
            timestamp: When the measurement was taken (defaults to now).
        """
        ts = timestamp if timestamp is not None else _now()
        gpu_snap = GPUSnapshot(
            node_id=node_id,
            gpu_type=gpu_type,
            gpu_utilization=gpu_utilization,
            memory_used=memory_used,
            memory_total=memory_total,
            timestamp=ts,
        )
        perf_snap = PerformanceSnapshot(
            timestamp=ts,
            request_rate=request_rate,
            token_throughput=token_throughput,
            avg_latency_ms=avg_latency_ms,
            active_nodes=active_nodes,
            gpu_snapshots=(gpu_snap,),
        )
        self.record_snapshot(perf_snap)

    # ------------------------------------------------------------------
    # Trend computation
    # ------------------------------------------------------------------

    def _compute_trend(
        self,
        value_extractor: Callable[[PerformanceSnapshot], float],
    ) -> Optional[TrendResult]:
        """Run linear regression on extracted values over time.

        Args:
            value_extractor: Callable that returns a float from a
                ``PerformanceSnapshot``.

        Returns:
            A ``TrendResult``, or ``None`` if fewer than 2 snapshots
            are available.
        """
        with self._lock:
            if len(self._snapshots) < 2:
                return None

            epochs = np.array(
                [
                    _to_timestamp_seconds(s.timestamp)
                    for s in self._snapshots
                ],
                dtype=np.float64,
            )
            values = np.array(
                [value_extractor(s) for s in self._snapshots],
                dtype=np.float64,
            )

        slope, intercept, r_squared = _linear_regression(epochs, values)

        now_epoch = _to_timestamp_seconds(_now())
        forecast_7d = float(
            np.polyval([slope, intercept], now_epoch + 7 * _DAY_SECONDS)
        )
        forecast_30d = float(
            np.polyval([slope, intercept], now_epoch + 30 * _DAY_SECONDS)
        )
        forecast_90d = float(
            np.polyval([slope, intercept], now_epoch + 90 * _DAY_SECONDS)
        )

        return TrendResult(
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
            forecast_7d=forecast_7d,
            forecast_30d=forecast_30d,
            forecast_90d=forecast_90d,
        )

    def gpu_trend(self) -> Optional[TrendResult]:
        """Linear regression trend for average GPU utilisation.

        Returns:
            TrendResult or ``None`` if insufficient data.
        """
        return self._compute_trend(
            lambda s: (
                float(np.mean([g.gpu_utilization for g in s.gpu_snapshots]))
                if s.gpu_snapshots
                else 0.0
            )
        )

    def memory_trend(self) -> Optional[TrendResult]:
        """Linear regression trend for average GPU memory utilisation.

        Returns:
            TrendResult or ``None`` if insufficient data.
        """
        return self._compute_trend(
            lambda s: (
                float(
                    np.mean(
                        [
                            g.memory_used / g.memory_total
                            for g in s.gpu_snapshots
                            if g.memory_total > 0
                        ]
                    )
                )
                if s.gpu_snapshots
                else 0.0
            )
        )

    def request_rate_trend(self) -> Optional[TrendResult]:
        """Linear regression trend for request rate (req/s).

        Returns:
            TrendResult or ``None`` if insufficient data.
        """
        return self._compute_trend(lambda s: s.request_rate)

    def throughput_trend(self) -> Optional[TrendResult]:
        """Linear regression trend for token throughput (tokens/s).

        Returns:
            TrendResult or ``None`` if insufficient data.
        """
        return self._compute_trend(lambda s: s.token_throughput)

    # ------------------------------------------------------------------
    # Scaling triggers
    # ------------------------------------------------------------------

    def _check_utilization_trigger(self) -> List[Dict[str, Any]]:
        """Check if GPU utilisation has exceeded the threshold.

        Examines snapshots within ``_util_window`` and returns a trigger
        dict if the average utilisation exceeds ``_util_threshold``.

        Returns:
            A list of trigger dicts (usually zero or one).
        """
        with self._lock:
            cutoff = _now() - self._util_window
            window_snapshots = [
                s
                for s in self._snapshots
                if s.timestamp >= cutoff
            ]

            if not window_snapshots:
                return []

            all_utils: List[float] = []
            for s in window_snapshots:
                for g in s.gpu_snapshots:
                    all_utils.append(g.gpu_utilization)

            if not all_utils:
                return []

            avg_util = float(np.mean(all_utils))

        if avg_util > self._util_threshold:
            return [
                {
                    "type": "gpu_utilization",
                    "severity": (
                        "critical"
                        if avg_util > 0.95
                        else "high" if avg_util > 0.90
                        else "warning"
                    ),
                    "average_utilization": round(avg_util, 4),
                    "threshold": self._util_threshold,
                    "window_seconds": int(
                        self._util_window.total_seconds()
                    ),
                    "triggered_at": _now().isoformat(),
                    "message": (
                        f"GPU utilisation averaged "
                        f"{avg_util:.1%} over the last "
                        f"{self._util_window} "
                        f"(threshold: {self._util_threshold:.0%})."
                    ),
                }
            ]

        return []

    def _check_growth_triggers(
        self, trends: Dict[str, Optional[TrendResult]]
    ) -> List[Dict[str, Any]]:
        """Generate triggers from rapid growth trends.

        Args:
            trends: Dict mapping metric names to their trends.

        Returns:
            List of trigger dicts.
        """
        triggers: List[Dict[str, Any]] = []

        for metric_name, trend in trends.items():
            if trend is None:
                continue
            if trend.r_squared < 0.7:
                continue  # low confidence

            # If the 30-day forecast exceeds 80% of some implied capacity
            # (utilisation or memory), flag it.
            if metric_name in ("gpu", "memory"):
                if trend.forecast_30d > self._util_threshold:
                    triggers.append(
                        {
                            "type": f"{metric_name}_growth",
                            "severity": "info",
                            "metric": metric_name,
                            "slope": round(trend.slope, 8),
                            "r_squared": round(trend.r_squared, 4),
                            "forecast_30d": round(trend.forecast_30d, 4),
                            "triggered_at": _now().isoformat(),
                            "message": (
                                f"{metric_name} utilisation is trending "
                                f"upward (R²={trend.r_squared:.2f}) and "
                                f"is projected to reach "
                                f"{trend.forecast_30d:.1%} in 30 days."
                            ),
                        }
                    )

        return triggers

    # ------------------------------------------------------------------
    # Plan
    # ------------------------------------------------------------------

    def plan(self) -> CapacityPlanReport:
        """Generate a comprehensive capacity plan.

        Computes trends, evaluates scaling triggers, and compiles a list
        of recommendations.

        Returns:
            A ``CapacityPlanReport`` containing all findings.
        """
        gpu_trend = self.gpu_trend()
        mem_trend = self.memory_trend()
        req_trend = self.request_rate_trend()
        tp_trend = self.throughput_trend()

        trends: Dict[str, Optional[TrendResult]] = {
            "gpu": gpu_trend,
            "memory": mem_trend,
            "request_rate": req_trend,
            "throughput": tp_trend,
        }

        scaling_triggers: List[Dict[str, Any]] = []
        scaling_triggers.extend(self._check_utilization_trigger())
        scaling_triggers.extend(self._check_growth_triggers(trends))

        recommendations = self._build_recommendations(
            trends, scaling_triggers
        )

        with self._lock:
            active_nodes = (
                self._snapshots[-1].active_nodes if self._snapshots else 0
            )
            snapshot_count = len(self._snapshots)

        return CapacityPlanReport(
            generated_at=_now(),
            gpu_trend=gpu_trend,
            memory_trend=mem_trend,
            request_rate_trend=req_trend,
            throughput_trend=tp_trend,
            active_nodes=active_nodes,
            scaling_triggers=scaling_triggers,
            recommendations=recommendations,
            snapshot_count=snapshot_count,
        )

    def _build_recommendations(
        self,
        trends: Dict[str, Optional[TrendResult]],
        triggers: List[Dict[str, Any]],
    ) -> List[ScalingRecommendation]:
        """Translate trends and triggers into recommendations.

        Args:
            trends: Dict mapping metric names to trend results.
            triggers: List of trigger dicts from trigger checks.

        Returns:
            Ordered list of recommendations (highest priority first).
        """
        recs: List[ScalingRecommendation] = []

        # --- Check utilisation triggers for immediate action ----------
        for t in triggers:
            if t["type"] == "gpu_utilization":
                sev = t["severity"]
                priority = (
                    RecommendationPriority.CRITICAL
                    if sev == "critical"
                    else (
                        RecommendationPriority.HIGH
                        if sev == "high"
                        else RecommendationPriority.MEDIUM
                    )
                )
                recs.append(
                    ScalingRecommendation(
                        recommendation_type=RecommendationType.SCALE_UP,
                        priority=priority,
                        title="GPU utilisation exceeding threshold",
                        description=t["message"],
                        suggested_action=(
                            "Add one or more GPU nodes to the cluster, or "
                            "redistribute existing workloads."
                        ),
                        estimated_impact=(
                            "Reduces GPU utilisation, improves tail "
                            "latency, and provides headroom for traffic "
                            "spikes."
                        ),
                        metric_evidence={
                            "average_utilization": t["average_utilization"],
                            "threshold": t["threshold"],
                        },
                    )
                )

        # --- Growth trends -------------------------------------------
        gpu_trend = trends.get("gpu")
        req_trend = trends.get("request_rate")
        tp_trend = trends.get("throughput")

        if gpu_trend is not None and gpu_trend.r_squared > 0.7:
            if gpu_trend.forecast_30d > self._util_threshold:
                recs.append(
                    ScalingRecommendation(
                        recommendation_type=RecommendationType.ADD_NODE,
                        priority=RecommendationPriority.HIGH,
                        title="GPU utilisation growth trend detected",
                        description=(
                            f"GPU utilisation is growing at "
                            f"{gpu_trend.slope:.2e}/s "
                            f"(R²={gpu_trend.r_squared:.2f}).  Projected "
                            f"to reach {gpu_trend.forecast_30d:.1%} "
                            f"in 30 days."
                        ),
                        suggested_action=(
                            "Schedule capacity addition within the next "
                            "2-4 weeks to stay ahead of the trend."
                        ),
                        estimated_impact=(
                            "Prevents utilisation from hitting the "
                            "scaling trigger threshold and avoids "
                            "performance degradation."
                        ),
                        metric_evidence={
                            "slope": gpu_trend.slope,
                            "r_squared": gpu_trend.r_squared,
                            "forecast_30d": gpu_trend.forecast_30d,
                        },
                    )
                )

        if req_trend is not None and req_trend.r_squared > 0.7:
            # If the request rate is growing and throughput is flat,
            # recommend a batch-size optimisation.
            tp_growing = (
                tp_trend is not None and tp_trend.slope > 0
            ) if tp_trend else False

            if not tp_growing:
                recs.append(
                    ScalingRecommendation(
                        recommendation_type=RecommendationType.OPTIMIZE_BATCH,
                        priority=RecommendationPriority.MEDIUM,
                        title="Request rate increasing without throughput growth",
                        description=(
                            f"Request rate is increasing at "
                            f"{req_trend.slope:.2e}/s "
                            f"(R²={req_trend.r_squared:.2f}) but token "
                            f"throughput is not keeping pace."
                        ),
                        suggested_action=(
                            "Increase batch size or optimise batching "
                            "strategy to improve throughput per node."
                        ),
                        estimated_impact=(
                            "Improves token throughput without adding "
                            "hardware, delaying the need for a scale-up."
                        ),
                        metric_evidence={
                            "request_rate_slope": req_trend.slope,
                            "request_rate_r_squared": req_trend.r_squared,
                        },
                    )
                )

        # --- Scale down opportunity (sustained low utilisation) ------
        if gpu_trend is not None and gpu_trend.r_squared > 0.5:
            latest_util = gpu_trend.intercept  # approximate
            if latest_util < 0.3 and gpu_trend.slope < 0:
                recs.append(
                    ScalingRecommendation(
                        recommendation_type=RecommendationType.SCALE_DOWN,
                        priority=RecommendationPriority.LOW,
                        title="Sustained low GPU utilisation",
                        description=(
                            f"GPU utilisation is trending downward "
                            f"(slope={gpu_trend.slope:.2e}/s) and is "
                            f"currently below 30 %.  Consider reducing "
                            f"cluster capacity."
                        ),
                        suggested_action=(
                            "Remove underutilised nodes or migrate "
                            "workloads to a smaller instance type."
                        ),
                        estimated_impact=(
                            "Reduces infrastructure cost without "
                            "affecting throughput."
                        ),
                        metric_evidence={
                            "slope": gpu_trend.slope,
                            "current_util_estimate": latest_util,
                        },
                    )
                )

        # Sort: critical first, then high, medium, low
        order = {
            RecommendationPriority.CRITICAL: 0,
            RecommendationPriority.HIGH: 1,
            RecommendationPriority.MEDIUM: 2,
            RecommendationPriority.LOW: 3,
        }
        recs.sort(key=lambda r: order.get(r.priority, 99))

        return recs

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all recorded snapshots."""
        with self._lock:
            self._snapshots.clear()

    def get_recent_utilization(
        self, window: timedelta = timedelta(hours=1)
    ) -> float:
        """Return the average GPU utilisation over *window*.

        Args:
            window: Look-back window (default 1 hour).

        Returns:
            Average utilisation as a fraction in [0, 1].  Returns 0.0
            when no data is available.
        """
        with self._lock:
            cutoff = _now() - window
            all_utils = [
                g.gpu_utilization
                for s in self._snapshots
                if s.timestamp >= cutoff
                for g in s.gpu_snapshots
            ]
        return float(np.mean(all_utils)) if all_utils else 0.0


# ===================================================================
# WhatIfCalculator
# ===================================================================


class WhatIfCalculator:
    """Answers capacity "what if" questions using historical data.

    Uses the linear model extracted from a ``CapacityPlanner`` to estimate
    the impact of hardware or configuration changes.  All estimates are
    based on a simplified linear throughput model:

        ``throughput ≈ per_node_throughput × num_nodes``

    GPU upgrade improvements are estimated from a known performance index
    (see ``GPU_PERFORMANCE_INDEX``).

    Thread-safe.
    """

    def __init__(self, planner: CapacityPlanner) -> None:
        """Initialise the what-if calculator.

        Args:
            planner: A ``CapacityPlanner`` instance whose historical data
                is used for estimation.
        """
        self._planner = planner
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def planner(self) -> CapacityPlanner:
        """Underlying capacity planner."""
        return self._planner

    # ------------------------------------------------------------------
    # What-if helpers
    # ------------------------------------------------------------------

    def _estimate_per_node_throughput(self) -> float:
        """Estimate average throughput per node from historical data.

        Uses the median of ``token_throughput / active_nodes`` across
        recent snapshots.

        Returns:
            Tokens per second per node.  Returns 100.0 as a conservative
            default when no data is available.
        """
        with self._planner._lock:
            ratios = [
                s.token_throughput / max(s.active_nodes, 1)
                for s in self._planner._snapshots
                if s.token_throughput > 0 and s.active_nodes > 0
            ]
        if not ratios:
            return 100.0
        return float(np.median(ratios))

    def _estimate_current_nodes(self) -> int:
        """Return the most recent active node count."""
        with self._planner._lock:
            if not self._planner._snapshots:
                return 1
            return self._planner._snapshots[-1].active_nodes

    def _estimate_current_throughput(self) -> float:
        """Return the most recent token throughput."""
        with self._planner._lock:
            if not self._planner._snapshots:
                return 0.0
            return self._planner._snapshots[-1].token_throughput

    # ------------------------------------------------------------------
    # What-if API
    # ------------------------------------------------------------------

    def what_if_add_gpu(
        self, count: int = 1, model: str = "A100"
    ) -> Dict[str, Any]:
        """Estimate the impact of adding *count* GPUs of type *model*.

        The estimate assumes the new GPUs are added to new nodes, each
        with the same per-node throughput as the current average.

        Args:
            count: Number of GPUs (nodes) to add (default 1).
            model: GPU model string (default ``"A100"``).  Used only for
                the performance ratio relative to the current fleet.

        Returns:
            Dict with keys: ``current_throughput``, ``projected_throughput``,
            ``improvement_pct``, ``added_nodes``, ``assumptions``.
        """
        if count < 1:
            return {
                "current_throughput": self._estimate_current_throughput(),
                "projected_throughput": self._estimate_current_throughput(),
                "improvement_pct": 0.0,
                "added_nodes": 0,
                "assumptions": [],
            }

        per_node = self._estimate_per_node_throughput()
        current_nodes = self._estimate_current_nodes()
        current_tp = self._estimate_current_throughput()

        # Adjust for GPU performance difference
        perf_current = GPU_PERFORMANCE_INDEX.get(
            self._infer_current_gpu_type(), 0.5
        )
        perf_new = GPU_PERFORMANCE_INDEX.get(model, 0.5)
        perf_ratio = perf_new / perf_current if perf_current > 0 else 1.0

        additional_throughput = count * per_node * perf_ratio
        projected = current_tp + additional_throughput
        improvement = (
            ((projected - current_tp) / current_tp * 100.0)
            if current_tp > 0
            else 0.0
        )

        return {
            "current_throughput": round(current_tp, 2),
            "projected_throughput": round(projected, 2),
            "improvement_pct": round(improvement, 2),
            "added_nodes": count,
            "assumptions": [
                f"Each node provides ~{per_node:.1f} tokens/s",
                f"New nodes use {model} (performance ratio "
                f"{perf_ratio:.2f}x vs current fleet)",
                "Linear throughput scaling (no interconnect bottleneck)",
            ],
        }

    def what_if_upgrade_gpu(
        self, existing: str = "A100", new: str = "H100"
    ) -> Dict[str, Any]:
        """Estimate the improvement from upgrading GPU type.

        Args:
            existing: Current GPU model string.
            new: Target GPU model string.

        Returns:
            Dict with keys: ``current_gpu``, ``target_gpu``,
            ``performance_ratio``, ``estimated_throughput_improvement_pct``,
            ``estimated_latency_reduction_pct``, ``assumptions``.
        """
        perf_existing = GPU_PERFORMANCE_INDEX.get(existing, 0.5)
        perf_new = GPU_PERFORMANCE_INDEX.get(new, 0.5)
        ratio = perf_new / perf_existing if perf_existing > 0 else 1.0

        current_tp = self._estimate_current_throughput()
        projected_tp = current_tp * ratio

        tp_improvement = (
            ((projected_tp - current_tp) / current_tp * 100.0)
            if current_tp > 0
            else 0.0
        )

        # Latency improvement is roughly inverse-square-root of perf ratio
        # (simple approximation: more compute → faster token generation)
        latency_improvement = (1.0 - 1.0 / math.sqrt(ratio)) * 100.0

        return {
            "current_gpu": existing,
            "target_gpu": new,
            "performance_ratio": round(ratio, 3),
            "estimated_throughput_improvement_pct": round(tp_improvement, 2),
            "estimated_latency_reduction_pct": round(
                max(0.0, latency_improvement), 2
            ),
            "assumptions": [
                f"{existing} → {new} provides {ratio:.2f}x raw "
                f"compute uplift",
                "Throughput scales linearly with compute capacity",
                "Memory capacity is sufficient for the target model",
            ],
        }

    def what_if_change_batch_size(
        self, new_batch_size: int
    ) -> Dict[str, Any]:
        """Estimate the throughput impact of changing batch size.

        Uses a simple saturation model:
            ``throughput(N) ≈ throughput_base × N / (N + α)``

        where ``α`` is a saturation factor estimated from the planner's
        historical data or defaulted to 8.

        Args:
            new_batch_size: Target batch size (must be >= 1).

        Returns:
            Dict with keys: ``current_batch_size``, ``new_batch_size``,
            ``estimated_throughput_change_pct``, ``estimated_throughput``,
            ``estimated_latency_change_pct``, ``assumptions``.
        """
        if new_batch_size < 1:
            raise InvalidConfigurationError(
                f"new_batch_size must be >= 1, got {new_batch_size}"
            )

        # Infer current batch size from request rate and throughput:
        #   throughput ≈ batch_size × request_rate
        with self._planner._lock:
            recent = [
                s
                for s in self._planner._snapshots
                if s.request_rate > 0 and s.token_throughput > 0
            ]

        if recent:
            ratios = [
                s.token_throughput / s.request_rate
                for s in recent[-20:]  # last 20 snapshots
            ]
            current_batch = float(np.median(ratios))
        else:
            current_batch = 1.0

        alpha = 8.0  # saturation factor

        current_tp = self._estimate_current_throughput()
        current_normalized = current_batch / (current_batch + alpha)
        new_normalized = new_batch_size / (new_batch_size + alpha)

        if current_normalized > 0:
            tp_multiplier = new_normalized / current_normalized
            projected_tp = current_tp * tp_multiplier
            tp_change_pct = (tp_multiplier - 1.0) * 100.0
        else:
            projected_tp = current_tp
            tp_change_pct = 0.0

        # Larger batches increase latency proportionally
        latency_change_pct = (
            (new_batch_size / current_batch - 1.0) * 100.0
            if current_batch > 0
            else 0.0
        )

        return {
            "current_batch_size": round(current_batch, 1),
            "new_batch_size": new_batch_size,
            "estimated_throughput_change_pct": round(tp_change_pct, 2),
            "estimated_throughput": round(projected_tp, 2),
            "estimated_latency_change_pct": round(latency_change_pct, 2),
            "assumptions": [
                f"Throughput model: tp(N) ∝ N / (N + {alpha})",
                "Current batch size estimated from historical "
                f"throughput/rate ratio (~{current_batch:.1f})",
                "Latency scales linearly with batch size",
            ],
        }

    def _infer_current_gpu_type(self) -> str:
        """Guess the most common GPU type from recent snapshots."""
        with self._planner._lock:
            gpu_types = [
                g.gpu_type
                for s in self._planner._snapshots[-50:]
                for g in s.gpu_snapshots
                if g.gpu_type
            ]
        if not gpu_types:
            return "unknown"
        return max(set(gpu_types), key=gpu_types.count)


# ===================================================================
# HPAConfig
# ===================================================================


class HPAConfig:
    """Generates Kubernetes HPA and VPA YAML manifests.

    Supports three metric types for HPA:

    * ``cpu`` — resource utilisation target.
    * ``memory`` — resource utilisation target.
    * ``requests_per_second`` — pod-level average-value target.

    VPA output recommends resource requests based on observed usage.
    """

    def __init__(self, default_cpu_target: int = 80,
                 default_memory_target: int = 80,
                 default_rps_target: int = 1000) -> None:
        """Initialise HPA/VPA configuration.

        Args:
            default_cpu_target: Default CPU utilisation percentage
                target (default 80).
            default_memory_target: Default memory utilisation
                percentage target (default 80).
            default_rps_target: Default requests-per-second target
                per pod (default 1000).
        """
        self._cpu_target = default_cpu_target
        self._memory_target = default_memory_target
        self._rps_target = default_rps_target

    # ------------------------------------------------------------------
    # HPA
    # ------------------------------------------------------------------

    def generate_hpa_yaml(
        self,
        service: str,
        min_replicas: int = 1,
        max_replicas: int = 10,
        metrics: Optional[List[str]] = None,
        cpu_target: Optional[int] = None,
        memory_target: Optional[int] = None,
        rps_target: Optional[int] = None,
    ) -> str:
        """Generate a Kubernetes HPA v2 YAML manifest.

        Args:
            service: Name of the Kubernetes Deployment to autoscale.
            min_replicas: Minimum number of replicas (default 1).
            max_replicas: Maximum number of replicas (default 10).
            metrics: List of metric types to include.  Valid values:
                ``"cpu"``, ``"memory"``, ``"requests_per_second"``.
                Defaults to ``["cpu", "memory"]``.
            cpu_target: CPU utilisation target percentage.  Falls back
                to the instance default.
            memory_target: Memory utilisation target percentage.  Falls
                back to the instance default.
            rps_target: Requests-per-second target per pod.  Falls back
                to the instance default.

        Returns:
            YAML string of the HPA manifest.  Returns a fallback comment
            if PyYAML is not installed.

        Raises:
            InvalidConfigurationError: If parameters are out of range.
        """
        if min_replicas < 1:
            raise InvalidConfigurationError(
                f"min_replicas must be >= 1, got {min_replicas}"
            )
        if max_replicas < min_replicas:
            raise InvalidConfigurationError(
                f"max_replicas ({max_replicas}) must be >= "
                f"min_replicas ({min_replicas})"
            )

        metric_list: List[Dict[str, Any]] = []
        selected = (
            list(metrics) if metrics is not None else ["cpu", "memory"]
        )
        cpu_t = cpu_target if cpu_target is not None else self._cpu_target
        mem_t = (
            memory_target
            if memory_target is not None
            else self._memory_target
        )
        rps_t = (
            rps_target if rps_target is not None else self._rps_target
        )

        if "cpu" in selected:
            metric_list.append(
                {
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": cpu_t,
                        },
                    },
                }
            )
        if "memory" in selected:
            metric_list.append(
                {
                    "type": "Resource",
                    "resource": {
                        "name": "memory",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": mem_t,
                        },
                    },
                }
            )
        if "requests_per_second" in selected:
            metric_list.append(
                {
                    "type": "Pods",
                    "pods": {
                        "metric": {
                            "name": "requests_per_second",
                        },
                        "target": {
                            "type": "AverageValue",
                            "averageValue": str(rps_t),
                        },
                    },
                }
            )

        hpa: Dict[str, Any] = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": service,
                "labels": {
                    "app": service,
                    "component": "distllm",
                },
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": service,
                },
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
                "metrics": metric_list,
            },
        }

        if YAML_AVAILABLE:
            return yaml_dump(hpa, default_flow_style=False, sort_keys=False)
        return yaml_dump(hpa)

    # ------------------------------------------------------------------
    # VPA
    # ------------------------------------------------------------------

    def generate_vpa_yaml(
        self,
        service: str,
        min_cpu: str = "100m",
        max_cpu: str = "8",
        min_memory: str = "256Mi",
        max_memory: str = "32Gi",
        update_mode: str = "Auto",
    ) -> str:
        """Generate a Kubernetes VPA YAML manifest.

        Args:
            service: Name of the Kubernetes Deployment.
            min_cpu: Minimum allowed CPU (default ``"100m"``).
            max_cpu: Maximum allowed CPU (default ``"8"``).
            min_memory: Minimum allowed memory (default ``"256Mi"``).
            max_memory: Maximum allowed memory (default ``"32Gi"``).
            update_mode: VPA update mode (``"Auto"``, ``"Initial"``, or
                ``"Off"``).  Defaults to ``"Auto"``.

        Returns:
            YAML string of the VPA manifest.  Returns a fallback comment
            if PyYAML is not installed.

        Raises:
            InvalidConfigurationError: If *update_mode* is invalid.
        """
        valid_modes = {"Auto", "Initial", "Off"}
        if update_mode not in valid_modes:
            raise InvalidConfigurationError(
                f"update_mode must be one of {valid_modes}, "
                f"got {update_mode!r}"
            )

        vpa: Dict[str, Any] = {
            "apiVersion": "autoscaling.k8s.io/v1",
            "kind": "VerticalPodAutoscaler",
            "metadata": {
                "name": f"{service}-vpa",
                "labels": {
                    "app": service,
                    "component": "distllm",
                },
            },
            "spec": {
                "targetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": service,
                },
                "updatePolicy": {
                    "updateMode": update_mode,
                },
                "resourcePolicy": {
                    "containerPolicies": [
                        {
                            "containerName": "*",
                            "minAllowed": {
                                "cpu": min_cpu,
                                "memory": min_memory,
                            },
                            "maxAllowed": {
                                "cpu": max_cpu,
                                "memory": max_memory,
                            },
                        }
                    ],
                },
            },
        }

        if YAML_AVAILABLE:
            return yaml_dump(vpa, default_flow_style=False, sort_keys=False)
        return yaml_dump(vpa)


# ===================================================================
# CapacityDashboard
# ===================================================================


class CapacityDashboard:
    """Generates a Grafana dashboard for capacity planning and cost visibility.

    Produces a complete, importable Grafana dashboard JSON model with
    these panel rows:

    * Row 1 — usage trend: area charts for GPU utilisation, memory
      utilisation, request rate, and token throughput, each with
      7d / 30d / 90d selectable range.
    * Row 2 — forecast: line charts overlaying actual data with
      linear-regression forecast lines.
    * Row 3 — scaling recommendations: stat panels showing the current
      number of recommendations by priority.
    * Row 4 — GPU-hour cost projection: time-series of daily GPU cost.
    """

    def __init__(
        self,
        planner: CapacityPlanner,
        datasource: str = "Prometheus",
        dashboard_title: str = "DistLLM Capacity Planning",
        refresh_interval: str = "30s",
    ) -> None:
        """Initialise the dashboard generator.

        Args:
            planner: A ``CapacityPlanner`` instance used to populate
                trend data embedded in the dashboard.
            datasource: Grafana datasource name (default
                ``"Prometheus"``).
            dashboard_title: Dashboard title.
            refresh_interval: Auto-refresh interval (e.g. ``"30s"``).
        """
        self._planner = planner
        self._datasource = datasource
        self._title = dashboard_title
        self._refresh = refresh_interval

    # ------------------------------------------------------------------
    # Dashboard generation
    # ------------------------------------------------------------------

    def generate_grafana_json(self) -> str:
        """Generate a complete, importable Grafana dashboard JSON model.

        The dashboard uses a 24-column grid with four rows of panels:

        * Row 1 (y=0): GPU utilisation, memory utilisation, request rate,
          token throughput — 4 time-series panels (w=6 each).
        * Row 2 (y=9): Forecast overlays — 2 panels (w=12 each).
        * Row 3 (y=18): Scaling recommendations — 2 stat panels (w=12 each).
        * Row 4 (y=27): GPU-hour cost projection (w=24).

        Returns:
            Pretty-printed JSON string.
        """
        panels: List[Dict[str, Any]] = []
        pid = [0]  # mutable counter

        # ---- Row 1: usage trends (y=0) -------------------------------
        row1_y = 0

        panels.append(
            self._build_timeseries_panel(
                pid=_next_id(pid),
                title="GPU Utilisation",
                query=(
                    "avg(distllm_node_gpu_utilization_percent) / 100"
                ),
                y=row1_y,
                x=0,
                w=6,
                h=_GRAFANA_PANEL_H,
                unit="percentunit",
            )
        )
        panels.append(
            self._build_timeseries_panel(
                pid=_next_id(pid),
                title="GPU Memory Utilisation",
                query=(
                    "avg(distllm_node_gpu_memory_bytes"
                    " / on(node_id) "
                    "distllm_node_gpu_memory_total_bytes)"
                ),
                y=row1_y,
                x=6,
                w=6,
                h=_GRAFANA_PANEL_H,
                unit="percentunit",
            )
        )
        panels.append(
            self._build_timeseries_panel(
                pid=_next_id(pid),
                title="Request Rate",
                query="sum(rate(distllm_requests_total[5m]))",
                y=row1_y,
                x=12,
                w=6,
                h=_GRAFANA_PANEL_H,
                unit="reqps",
            )
        )
        panels.append(
            self._build_timeseries_panel(
                pid=_next_id(pid),
                title="Token Throughput",
                query="sum(rate(distllm_tokens_generated_total[5m]))",
                y=row1_y,
                x=18,
                w=6,
                h=_GRAFANA_PANEL_H,
                unit="tokens/s",
            )
        )

        # ---- Row 2: forecast (y=9) -----------------------------------
        row2_y = row1_y + _GRAFANA_PANEL_H + _GRAFANA_ROW_GAP

        gpu_trend = self._planner.gpu_trend()
        tp_trend = self._planner.throughput_trend()

        panels.append(
            self._build_forecast_panel(
                pid=_next_id(pid),
                title="GPU Utilisation Forecast",
                trend=gpu_trend,
                query_actual=(
                    "avg(distllm_node_gpu_utilization_percent) / 100"
                ),
                y=row2_y,
                x=0,
                w=12,
                h=_GRAFANA_PANEL_H,
            )
        )
        panels.append(
            self._build_forecast_panel(
                pid=_next_id(pid),
                title="Token Throughput Forecast",
                trend=tp_trend,
                query_actual=(
                    "sum(rate(distllm_tokens_generated_total[5m]))"
                ),
                y=row2_y,
                x=12,
                w=12,
                h=_GRAFANA_PANEL_H,
            )
        )

        # ---- Row 3: scaling recommendations (y=18) -------------------
        row3_y = row2_y + _GRAFANA_PANEL_H + _GRAFANA_ROW_GAP

        report = self._planner.plan()

        panels.append(
            self._build_recommendation_stat_panel(
                pid=_next_id(pid),
                title="Scaling Recommendations",
                report=report,
                y=row3_y,
                x=0,
                w=12,
                h=_GRAFANA_PANEL_H,
            )
        )
        panels.append(
            self._build_cost_projection_stat_panel(
                pid=_next_id(pid),
                title="GPU-Hour Cost Projection (30d)",
                report=report,
                y=row3_y,
                x=12,
                w=12,
                h=_GRAFANA_PANEL_H,
            )
        )

        # ---- Row 4: GPU-hour cost (y=27) -----------------------------
        row4_y = row3_y + _GRAFANA_PANEL_H + _GRAFANA_ROW_GAP

        panels.append(
            self._build_cost_timeseries_panel(
                pid=_next_id(pid),
                y=row4_y,
                x=0,
                w=24,
                h=_GRAFANA_PANEL_H,
            )
        )

        dashboard: Dict[str, Any] = {
            "title": self._title,
            "description": (
                "Capacity planning dashboard generated by DistLLM "
                "CapacityDashboard.  Monitors utilisation, forecasts "
                "growth, and tracks GPU-hour costs."
            ),
            "refresh": self._refresh,
            "tags": [
                "capacity-planning",
                "distllm",
                "observability",
                "cost",
            ],
            "schemaVersion": 39,
            "version": 1,
            "time": {
                "from": "now-7d",
                "to": "now",
            },
            "timepicker": {
                "refresh_intervals": [
                    "5s",
                    "10s",
                    "30s",
                    "1m",
                    "5m",
                    "15m",
                    "30m",
                    "1h",
                    "2h",
                    "1d",
                ],
            },
            "panels": panels,
            "templating": {
                "list": [
                    {
                        "name": "datasource",
                        "type": "datasource",
                        "query": "prometheus",
                        "current": {
                            "value": self._datasource,
                            "text": self._datasource,
                        },
                        "hide": 0,
                    }
                ]
            },
        }

        return json.dumps(dashboard, indent=2, default=str)

    # ==================================================================
    # Panel builders
    # ==================================================================

    @staticmethod
    def _build_timeseries_panel(
        pid: int,
        title: str,
        query: str,
        y: int,
        x: int,
        w: int,
        h: int,
        unit: str = "none",
    ) -> Dict[str, Any]:
        """Build a standard time-series panel.

        Args:
            pid: Unique panel ID.
            title: Panel title.
            query: PromQL query string.
            y: Grid Y position.
            x: Grid X position.
            w: Panel width (columns).
            h: Panel height (rows).
            unit: Grafana unit string.

        Returns:
            Panel dict.
        """
        return {
            "id": pid,
            "title": title,
            "type": "timeseries",
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "datasource": {"type": "prometheus", "uid": "$datasource"},
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "color": {"mode": "palette-classic"},
                    "custom": {
                        "lineInterpolation": "smooth",
                        "spanNulls": False,
                        "showPoints": "never",
                        "gradientMode": "opacity",
                    },
                },
                "overrides": [],
            },
            "targets": [
                {
                    "expr": query,
                    "legendFormat": "__auto",
                    "refId": "A",
                },
            ],
            "options": {
                "legend": {
                    "displayMode": "list",
                    "placement": "bottom",
                    "showLegend": True,
                },
                "tooltip": {"mode": "multi", "sort": "desc"},
            },
        }

    @staticmethod
    def _build_forecast_panel(
        pid: int,
        title: str,
        trend: Optional[TrendResult],
        query_actual: str,
        y: int,
        x: int,
        w: int,
        h: int,
    ) -> Dict[str, Any]:
        """Build a time-series panel with actual data and forecast.

        When a trend is available, adds a constant-line override for the
        30-day forecast value.

        Args:
            pid: Unique panel ID.
            title: Panel title.
            trend: Trend result from the planner, or ``None``.
            query_actual: PromQL for the actual metric.
            y, x, w, h: Grid positioning.

        Returns:
            Panel dict.
        """
        overrides: List[Dict[str, Any]] = []
        if trend is not None:
            overrides.append(
                {
                    "matcher": {
                        "id": "byName",
                        "options": "30d Forecast",
                    },
                    "properties": [
                        {
                            "id": "custom.lineStyle",
                            "value": {
                                "fill": "dash",
                                "dash": [10, 10],
                            },
                        },
                        {"id": "color", "value": {"mode": "fixed"}},
                        {"id": "custom.lineWidth", "value": 1},
                    ],
                }
            )

        targets: List[Dict[str, Any]] = [
            {
                "expr": query_actual,
                "legendFormat": "Actual",
                "refId": "A",
            },
        ]
        if trend is not None:
            targets.append(
                {
                    "expr": f"{trend.forecast_30d}",
                    "legendFormat": "30d Forecast",
                    "refId": "B",
                }
            )

        return {
            "id": pid,
            "title": title,
            "type": "timeseries",
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "datasource": {"type": "prometheus", "uid": "$datasource"},
            "fieldConfig": {
                "defaults": {
                    "unit": "percentunit",
                    "color": {"mode": "palette-classic"},
                    "custom": {
                        "lineInterpolation": "smooth",
                        "spanNulls": False,
                        "showPoints": "never",
                        "gradientMode": "opacity",
                    },
                },
                "overrides": overrides,
            },
            "targets": targets,
            "options": {
                "legend": {
                    "displayMode": "list",
                    "placement": "bottom",
                    "showLegend": True,
                },
                "tooltip": {"mode": "multi", "sort": "desc"},
            },
        }

    @staticmethod
    def _build_recommendation_stat_panel(
        pid: int,
        title: str,
        report: CapacityPlanReport,
        y: int,
        x: int,
        w: int,
        h: int,
    ) -> Dict[str, Any]:
        """Stat panel showing the count of scaling recommendations.

        Args:
            pid: Unique panel ID.
            title: Panel title.
            report: The capacity plan report.
            y, x, w, h: Grid positioning.

        Returns:
            Panel dict.
        """
        critical = sum(
            1
            for r in report.recommendations
            if r.priority == RecommendationPriority.CRITICAL
        )
        high = sum(
            1
            for r in report.recommendations
            if r.priority == RecommendationPriority.HIGH
        )

        return {
            "id": pid,
            "title": title,
            "type": "stat",
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "datasource": {"type": "prometheus", "uid": "$datasource"},
            "fieldConfig": {
                "defaults": {
                    "unit": "none",
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {
                                "color": "semi-dark-green",
                                "value": None,
                            },
                            {
                                "color": "semi-dark-yellow",
                                "value": 0,
                            },
                            {"color": "semi-dark-red", "value": 1},
                        ],
                    },
                    "color": {"mode": "thresholds"},
                    "mappings": [],
                    "min": 0,
                    "max": 10,
                },
                "overrides": [],
            },
            "targets": [
                {
                    "expr": str(critical + high),
                    "legendFormat": "Critical + High",
                    "refId": "A",
                },
            ],
            "options": {
                "reduceOptions": {
                    "calcs": ["lastNotNull"],
                    "fields": "",
                    "values": False,
                },
                "orientation": "auto",
                "textMode": "auto",
                "colorMode": "value",
                "graphMode": "area",
                "justifyMode": "auto",
            },
        }

    @staticmethod
    def _build_cost_projection_stat_panel(
        pid: int,
        title: str,
        report: CapacityPlanReport,
        y: int,
        x: int,
        w: int,
        h: int,
    ) -> Dict[str, Any]:
        """Stat panel showing projected monthly GPU cost.

        Uses the active node count and an assumed hourly rate to estimate
        30-day cost.

        Args:
            pid: Unique panel ID.
            title: Panel title.
            report: The capacity plan report.
            y, x, w, h: Grid positioning.

        Returns:
            Panel dict.
        """
        # Rough estimate: $2.00/hour per GPU.
        estimated_daily = report.active_nodes * 2.0 * 24
        estimated_monthly = estimated_daily * 30

        return {
            "id": pid,
            "title": title,
            "type": "stat",
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "datasource": {"type": "prometheus", "uid": "$datasource"},
            "fieldConfig": {
                "defaults": {
                    "unit": "currencyUSD",
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {
                                "color": "semi-dark-green",
                                "value": None,
                            },
                            {
                                "color": "semi-dark-yellow",
                                "value": 10000,
                            },
                            {
                                "color": "semi-dark-red",
                                "value": 50000,
                            },
                        ],
                    },
                    "color": {"mode": "thresholds"},
                    "mappings": [],
                    "min": 0,
                },
                "overrides": [],
            },
            "targets": [
                {
                    "expr": str(estimated_monthly),
                    "legendFormat": "Est. Monthly Cost",
                    "refId": "A",
                },
            ],
            "options": {
                "reduceOptions": {
                    "calcs": ["lastNotNull"],
                    "fields": "",
                    "values": False,
                },
                "orientation": "auto",
                "textMode": "auto",
                "colorMode": "value",
                "graphMode": "area",
                "justifyMode": "auto",
            },
        }

    @staticmethod
    def _build_cost_timeseries_panel(
        pid: int,
        y: int,
        x: int,
        w: int,
        h: int,
    ) -> Dict[str, Any]:
        """Time-series panel showing estimated daily GPU cost.

        Args:
            pid: Unique panel ID.
            y, x, w, h: Grid positioning.

        Returns:
            Panel dict.
        """
        return {
            "id": pid,
            "title": "Estimated Daily GPU Cost",
            "type": "timeseries",
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "datasource": {"type": "prometheus", "uid": "$datasource"},
            "fieldConfig": {
                "defaults": {
                    "unit": "currencyUSD",
                    "color": {"mode": "palette-classic"},
                    "custom": {
                        "lineInterpolation": "smooth",
                        "spanNulls": False,
                        "showPoints": "never",
                        "gradientMode": "opacity",
                    },
                },
                "overrides": [],
            },
            "targets": [
                {
                    "expr": (
                        "sum(distllm_node_gpu_utilization_percent"
                        " * on(node_id) "
                        "distllm_node_gpu_memory_bytes"
                        " / 1e9 * 24 * 2.0)"
                    ),
                    "legendFormat": "Daily GPU Cost",
                    "refId": "A",
                },
            ],
            "options": {
                "legend": {
                    "displayMode": "list",
                    "placement": "bottom",
                    "showLegend": True,
                },
                "tooltip": {"mode": "multi", "sort": "desc"},
            },
        }


# ===================================================================
# GPUBilling
# ===================================================================


class GPUBilling:
    """Tracks GPU usage and computes costs.

    Maintains an in-memory store of GPU usage sessions.  Each session
    records a node, GPU type, and time range.  Usage is reported in
    GPU-hours, and costs are calculated against configurable per-type
    hourly rates.

    Thread-safe for concurrent recording and querying.
    """

    def __init__(self) -> None:
        """Initialise the billing tracker."""
        self._lock = threading.Lock()
        self._sessions: List[GPUUsageSession] = []
        self._next_id: int = 1

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def track_usage(
        self,
        node_id: str,
        gpu_type: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
    ) -> str:
        """Record a GPU usage session.

        Args:
            node_id: Worker node identifier.
            gpu_type: GPU model string (e.g. ``"A100"``).
            start_time: When usage began.
            end_time: When usage ended.  Pass ``None`` for a still-running
                session (default).

        Returns:
            The unique session ID string.
        """
        if end_time is not None and end_time < start_time:
            raise InvalidConfigurationError(
                f"end_time ({end_time}) must not be before "
                f"start_time ({start_time})"
            )

        session_id = f"gpu-{self._next_id:06d}"
        self._next_id += 1

        session = GPUUsageSession(
            session_id=session_id,
            node_id=node_id,
            gpu_type=gpu_type,
            start_time=start_time,
            end_time=end_time,
        )

        with self._lock:
            self._sessions.append(session)

        return session_id

    def stop_usage(self, session_id: str, end_time: Optional[datetime] = None) -> None:
        """Stop a running GPU usage session.

        Args:
            session_id: The session ID returned by ``track_usage``.
            end_time: When usage ended (defaults to now).

        Raises:
            ValueError: If *session_id* is not found or already stopped.
        """
        end = end_time if end_time is not None else _now()
        with self._lock:
            for i, s in enumerate(self._sessions):
                if s.session_id == session_id:
                    if s.end_time is not None:
                        raise ValueError(
                            f"Session {session_id} already stopped at "
                            f"{s.end_time.isoformat()}"
                        )
                    self._sessions[i] = dataclasses.replace(
                        s, end_time=end
                    )
                    return
            raise ValueError(
                f"Session {session_id} not found"
            )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_usage(
        self,
        period: timedelta,
        *,
        node_id: Optional[str] = None,
        gpu_type: Optional[str] = None,
    ) -> float:
        """Compute total GPU-hours over *period*.

        Args:
            period: Look-back window from now.
            node_id: Filter to a specific node (optional).
            gpu_type: Filter to a specific GPU type (optional).

        Returns:
            Total GPU-hours as a float.
        """
        cutoff = _now() - period
        total = 0.0

        with self._lock:
            for s in self._sessions:
                if s.end_time is not None and s.end_time < cutoff:
                    continue
                if node_id is not None and s.node_id != node_id:
                    continue
                if gpu_type is not None and s.gpu_type != gpu_type:
                    continue

                start = max(s.start_time, cutoff)
                total += _gpu_hours(start, s.end_time)

        return total

    def get_cost(
        self,
        period: timedelta,
        rates: Optional[Dict[str, float]] = None,
        *,
        node_id: Optional[str] = None,
        gpu_type: Optional[str] = None,
    ) -> float:
        """Compute total GPU cost over *period*.

        Args:
            period: Look-back window from now.
            rates: Dict mapping GPU type to hourly rate in USD.
                Unknown GPU types default to $2.00/hour.
            node_id: Filter to a specific node (optional).
            gpu_type: Filter to a specific GPU type (optional).

        Returns:
            Total cost in USD.
        """
        effective_rates: Dict[str, float] = {
            "A100": 3.00,
            "A100-80GB": 3.50,
            "V100": 1.50,
            "T4": 0.80,
            "L4": 1.20,
            "H100": 5.00,
            "H100-80GB": 5.50,
            "H200": 6.00,
            "RTX-3090": 1.00,
            "RTX-4090": 1.50,
            "A10G": 1.80,
            "A6000": 2.00,
            "unknown": 2.00,
        }
        if rates is not None:
            effective_rates.update(rates)

        cutoff = _now() - period
        total_cost = 0.0

        with self._lock:
            for s in self._sessions:
                if s.end_time is not None and s.end_time < cutoff:
                    continue
                if node_id is not None and s.node_id != node_id:
                    continue
                if gpu_type is not None and s.gpu_type != gpu_type:
                    continue

                start = max(s.start_time, cutoff)
                hours = _gpu_hours(start, s.end_time)
                rate = effective_rates.get(s.gpu_type, 2.00)
                total_cost += hours * rate

        return total_cost

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def generate_report(
        self,
        period: timedelta,
        rates: Optional[Dict[str, float]] = None,
        fmt: str = "json",
    ) -> str:
        """Generate a GPU billing report for *period*.

        The report includes per-node and per-GPU-type breakdowns, total
        GPU-hours, and total cost.

        Args:
            period: Look-back window from now.
            rates: Optional per-GPU-type hourly rates in USD.
            fmt: Output format — ``"json"`` (default) or ``"csv"``.

        Returns:
            Report as a string (JSON or CSV).

        Raises:
            InvalidConfigurationError: If *fmt* is not recognised.
        """
        cutoff = _now() - period

        # Collect raw data
        with self._lock:
            relevant = [
                s
                for s in self._sessions
                if (s.end_time is None or s.end_time >= cutoff)
                and s.start_time <= _now()
            ]

        # Build per-node breakdown
        node_breakdown: Dict[str, float] = {}
        type_breakdown: Dict[str, float] = {}
        all_types: set = set()

        for s in relevant:
            start = max(s.start_time, cutoff)
            hours = _gpu_hours(start, s.end_time)
            node_breakdown[s.node_id] = (
                node_breakdown.get(s.node_id, 0.0) + hours
            )
            type_breakdown[s.gpu_type] = (
                type_breakdown.get(s.gpu_type, 0.0) + hours
            )
            all_types.add(s.gpu_type)

        total_hours = sum(node_breakdown.values())

        effective_rates: Dict[str, float] = {
            "A100": 3.00,
            "A100-80GB": 3.50,
            "V100": 1.50,
            "T4": 0.80,
            "L4": 1.20,
            "H100": 5.00,
            "H100-80GB": 5.50,
            "H200": 6.00,
            "RTX-3090": 1.00,
            "RTX-4090": 1.50,
            "A10G": 1.80,
            "A6000": 2.00,
            "unknown": 2.00,
        }
        if rates is not None:
            effective_rates.update(rates)

        # Per-GPU-type cost
        type_cost: Dict[str, float] = {}
        total_cost = 0.0
        for gpu_type, hours in type_breakdown.items():
            rate = effective_rates.get(gpu_type, 2.00)
            cost = hours * rate
            type_cost[gpu_type] = cost
            total_cost += cost

        report: Dict[str, Any] = {
            "generated_at": _now().isoformat(),
            "period_days": period.total_seconds() / _DAY_SECONDS,
            "period_start": cutoff.isoformat(),
            "period_end": _now().isoformat(),
            "total_gpu_hours": round(total_hours, 4),
            "total_cost_usd": round(total_cost, 2),
            "node_breakdown": {
                node: round(hours, 4)
                for node, hours in sorted(node_breakdown.items())
            },
            "gpu_type_breakdown": {
                gpu: {
                    "hours": round(type_breakdown[gpu], 4),
                    "cost_usd": round(type_cost[gpu], 2),
                }
                for gpu in sorted(type_breakdown)
            },
            "session_count": len(relevant),
        }

        if fmt == "json":
            return json.dumps(report, indent=2, default=str)

        if fmt == "csv":
            return self._report_to_csv(report)

        raise InvalidConfigurationError(
            f"Unsupported format {fmt!r}; expected 'json' or 'csv'"
        )

    @staticmethod
    def _report_to_csv(report: Dict[str, Any]) -> str:
        """Convert a billing report dict to CSV format.

        Args:
            report: The report dict from ``generate_report``.

        Returns:
            CSV-formatted string.
        """
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow(["section", "key", "value"])

        writer.writerow(["meta", "generated_at", report.get("generated_at", "")])
        writer.writerow(
            ["meta", "period_days", report.get("period_days", 0)]
        )
        writer.writerow(
            ["meta", "total_gpu_hours", report.get("total_gpu_hours", 0)]
        )
        writer.writerow(
            ["meta", "total_cost_usd", report.get("total_cost_usd", 0)]
        )
        writer.writerow(
            ["meta", "session_count", report.get("session_count", 0)]
        )

        writer.writerow([])
        writer.writerow(["node_breakdown", "node_id", "gpu_hours"])
        for node, hours in report.get("node_breakdown", {}).items():
            writer.writerow(["node_breakdown", node, hours])

        writer.writerow([])
        writer.writerow(
            ["gpu_type_breakdown", "gpu_type", "hours", "cost_usd"]
        )
        for gpu_type, details in report.get(
            "gpu_type_breakdown", {}
        ).items():
            writer.writerow(
                [
                    "gpu_type_breakdown",
                    gpu_type,
                    details.get("hours", 0),
                    details.get("cost_usd", 0),
                ]
            )

        return output.getvalue()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all recorded sessions."""
        with self._lock:
            self._sessions.clear()

    def active_sessions(self) -> int:
        """Number of currently active (still running) sessions."""
        with self._lock:
            return sum(1 for s in self._sessions if s.end_time is None)

    def total_sessions(self) -> int:
        """Total number of recorded sessions."""
        with self._lock:
            return len(self._sessions)


# ===================================================================
# CapacityConfigurator
# ===================================================================


class CapacityConfigurator:
    """Top-level orchestrator for capacity planning.

    Combines ``CapacityPlanner``, ``WhatIfCalculator``, ``HPAConfig``,
    ``CapacityDashboard``, and ``GPUBilling`` into a single interface.

    Typical usage::

        configurator = CapacityConfigurator()
        configurator.start()  # begins periodic data collection
        ...
        report_path = configurator.generate_report("./reports/")
        configurator.stop()

    The configurator can also be used as a context manager::

        with CapacityConfigurator() as cc:
            cc.record_gpu_snapshot(...)
            print(cc.planner.plan())
    """

    def __init__(
        self,
        planner: Optional[CapacityPlanner] = None,
        billing: Optional[GPUBilling] = None,
        collection_interval_seconds: float = 60.0,
    ) -> None:
        """Initialise the capacity configurator.

        Args:
            planner: A ``CapacityPlanner`` instance, or ``None`` to
                create a default one.
            billing: A ``GPUBilling`` instance, or ``None`` to create a
                default one.
            collection_interval_seconds: How often (in seconds) to
                collect a snapshot when running (default 60.0).
        """
        self._planner = planner if planner is not None else CapacityPlanner()
        self._billing = billing if billing is not None else GPUBilling()
        self._what_if = WhatIfCalculator(self._planner)
        self._hpa = HPAConfig()
        self._dashboard = CapacityDashboard(self._planner)
        self._collection_interval = collection_interval_seconds
        self._running = False
        self._collector_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def planner(self) -> CapacityPlanner:
        """Capacity planner instance."""
        return self._planner

    @property
    def billing(self) -> GPUBilling:
        """GPU billing tracker instance."""
        return self._billing

    @property
    def what_if(self) -> WhatIfCalculator:
        """What-if calculator instance."""
        return self._what_if

    @property
    def hpa(self) -> HPAConfig:
        """HPA/VPA config generator instance."""
        return self._hpa

    @property
    def dashboard(self) -> CapacityDashboard:
        """Capacity dashboard generator instance."""
        return self._dashboard

    @property
    def is_running(self) -> bool:
        """Whether periodic collection is active."""
        return self._running

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin periodic snapshot collection in a background thread.

        Safe to call multiple times — subsequent calls are no-ops when
        already running.
        """
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._collector_thread = threading.Thread(
            target=self._collection_loop,
            name="capacity-collector",
            daemon=True,
        )
        self._collector_thread.start()

    def stop(self) -> None:
        """Stop periodic snapshot collection and wait for the thread.

        Safe to call multiple times — subsequent calls are no-ops when
        already stopped.
        """
        if not self._running:
            return

        self._stop_event.set()
        if (
            self._collector_thread is not None
            and self._collector_thread.is_alive()
        ):
            self._collector_thread.join(timeout=10.0)
        self._running = False

    def _collection_loop(self) -> None:
        """Background loop that periodically collects a snapshot.

        Calls ``_collect_snapshot`` then sleeps for
        ``_collection_interval`` seconds (checking the stop event every
        second).
        """
        while not self._stop_event.is_set():
            try:
                self._collect_snapshot()
            except Exception:
                # Log but don't crash the collector thread
                warnings.warn(
                    "Capacity configurator snapshot collection failed",
                    RuntimeWarning,
                )

            # Sleep in 1-second increments so stop() is responsive.
            for _ in range(int(self._collection_interval)):
                if self._stop_event.is_set():
                    return
                self._stop_event.wait(timeout=1.0)

    def _collect_snapshot(self) -> None:
        """Collect a single performance snapshot from the system.

        This is a no-op placeholder; real implementations should wire
        in actual metric collection (e.g. from Prometheus, node
        exporters, or the cluster manager).
        """
        # Override in subclass or wire to actual metric sources.
        pass

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def record_gpu_snapshot(
        self,
        node_id: str,
        gpu_type: str,
        gpu_utilization: float,
        memory_used: int,
        memory_total: int,
        request_rate: float = 0.0,
        token_throughput: float = 0.0,
        avg_latency_ms: float = 0.0,
        active_nodes: int = 1,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a GPU snapshot (delegates to ``CapacityPlanner``).

        Args:
            node_id: Worker node identifier.
            gpu_type: GPU model string.
            gpu_utilization: GPU utilisation as a fraction in [0, 1].
            memory_used: GPU memory used in bytes.
            memory_total: Total GPU memory in bytes.
            request_rate: Current request rate (req/s).
            token_throughput: Current token throughput (tokens/s).
            avg_latency_ms: Average latency in milliseconds.
            active_nodes: Number of active nodes.
            timestamp: When the measurement was taken (defaults to now).
        """
        self._planner.record_gpu_snapshot(
            node_id=node_id,
            gpu_type=gpu_type,
            gpu_utilization=gpu_utilization,
            memory_used=memory_used,
            memory_total=memory_total,
            request_rate=request_rate,
            token_throughput=token_throughput,
            avg_latency_ms=avg_latency_ms,
            active_nodes=active_nodes,
            timestamp=timestamp,
        )

    def track_billing_usage(
        self,
        node_id: str,
        gpu_type: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
    ) -> str:
        """Track a GPU billing session (delegates to ``GPUBilling``).

        Args:
            node_id: Worker node identifier.
            gpu_type: GPU model string.
            start_time: When usage began.
            end_time: When usage ended (defaults to ``None`` for
                still-running).

        Returns:
            The session ID string.
        """
        return self._billing.track_usage(
            node_id=node_id,
            gpu_type=gpu_type,
            start_time=start_time,
            end_time=end_time,
        )

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        path: str,
        period: timedelta = timedelta(days=30),
        rates: Optional[Dict[str, float]] = None,
        include_dashboard: bool = True,
        include_hpa: bool = False,
        service_name: str = "distllm",
    ) -> Dict[str, str]:
        """Generate a comprehensive capacity-planning report.

        Writes the following files under *path*:

        * ``capacity_plan.json`` — plan output from ``CapacityPlanner.plan()``
        * ``billing_report.json`` — billing report from ``GPUBilling``
        * ``grafana_dashboard.json`` — Grafana dashboard JSON (if
          *include_dashboard* is ``True``)
        * ``hpa.yaml`` — HPA manifest (if *include_hpa* is ``True``)
        * ``vpa.yaml`` — VPA manifest (if *include_hpa* is ``True``)

        Creates the directory if it does not exist.

        Args:
            path: Directory path for report files.
            period: Look-back window for billing (default 30 days).
            rates: Optional per-GPU-type hourly rates.
            include_dashboard: Whether to generate the Grafana dashboard
                (default ``True``).
            include_hpa: Whether to generate HPA/VPA manifests (default
                ``False``).
            service_name: Service name for HPA/VPA manifests (default
                ``"distllm"``).

        Returns:
            Dict mapping logical file name to absolute path.

        Raises:
            CapacityPlanningError: If output could not be written.
        """
        import os

        os.makedirs(path, exist_ok=True)

        out: Dict[str, str] = {}

        # -- Capacity plan --
        plan = self._planner.plan()
        plan_path = os.path.join(path, "capacity_plan.json")
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at": plan.generated_at.isoformat(),
                    "gpu_trend": dataclasses.asdict(plan.gpu_trend)
                    if plan.gpu_trend
                    else None,
                    "memory_trend": dataclasses.asdict(plan.memory_trend)
                    if plan.memory_trend
                    else None,
                    "request_rate_trend": dataclasses.asdict(
                        plan.request_rate_trend
                    )
                    if plan.request_rate_trend
                    else None,
                    "throughput_trend": dataclasses.asdict(
                        plan.throughput_trend
                    )
                    if plan.throughput_trend
                    else None,
                    "active_nodes": plan.active_nodes,
                    "scaling_triggers": plan.scaling_triggers,
                    "recommendations": [
                        {
                            "type": r.recommendation_type.value,
                            "priority": r.priority.value,
                            "title": r.title,
                            "description": r.description,
                            "suggested_action": r.suggested_action,
                            "estimated_impact": r.estimated_impact,
                            "metric_evidence": r.metric_evidence,
                        }
                        for r in plan.recommendations
                    ],
                    "snapshot_count": plan.snapshot_count,
                },
                f,
                indent=2,
                default=str,
            )
        out["capacity_plan"] = os.path.abspath(plan_path)

        # -- Billing report --
        billing_report = self._billing.generate_report(
            period=period, rates=rates, fmt="json"
        )
        billing_path = os.path.join(path, "billing_report.json")
        with open(billing_path, "w", encoding="utf-8") as f:
            f.write(billing_report)
        out["billing_report"] = os.path.abspath(billing_path)

        # -- Dashboard --
        if include_dashboard:
            dashboard_json = self._dashboard.generate_grafana_json()
            dashboard_path = os.path.join(path, "grafana_dashboard.json")
            with open(dashboard_path, "w", encoding="utf-8") as f:
                f.write(dashboard_json)
            out["grafana_dashboard"] = os.path.abspath(dashboard_path)

        # -- HPA / VPA --
        if include_hpa:
            hpa_yaml = self._hpa.generate_hpa_yaml(service=service_name)
            hpa_path = os.path.join(path, "hpa.yaml")
            with open(hpa_path, "w", encoding="utf-8") as f:
                f.write(hpa_yaml)
            out["hpa"] = os.path.abspath(hpa_path)

            vpa_yaml = self._hpa.generate_vpa_yaml(service=service_name)
            vpa_path = os.path.join(path, "vpa.yaml")
            with open(vpa_path, "w", encoding="utf-8") as f:
                f.write(vpa_yaml)
            out["vpa"] = os.path.abspath(vpa_path)

        return out

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> CapacityConfigurator:
        """Enter context: starts collection."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        """Exit context: stops collection."""
        self.stop()


# ===================================================================
# Public API
# ===================================================================

__all__ = [
    # Exceptions
    "CapacityPlanningError",
    "InsufficientDataError",
    "InvalidConfigurationError",
    # Data models
    "GPUSnapshot",
    "PerformanceSnapshot",
    "TrendResult",
    "ScalingRecommendation",
    "GPUUsageSession",
    "CapacityPlanReport",
    "RecommendationPriority",
    "RecommendationType",
    # Classes
    "CapacityPlanner",
    "WhatIfCalculator",
    "HPAConfig",
    "CapacityDashboard",
    "GPUBilling",
    "CapacityConfigurator",
]
