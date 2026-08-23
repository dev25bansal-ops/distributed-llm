"""Data models, exceptions, constants, and helpers for capacity planning.

This module contains the shared data layer used by all capacity-planning
service classes in ``capacity_planning.py``.
"""

from __future__ import annotations

import csv
import dataclasses
import io
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, IO, List, Optional, Sequence, Tuple

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

_HOUR_SECONDS: int = 3600
_DAY_SECONDS: int = 86400
_DEFAULT_MAX_SNAPSHOT_AGE_DAYS: int = 90
_UTILIZATION_TRIGGER_THRESHOLD: float = 0.80
_UTILIZATION_TRIGGER_WINDOW: timedelta = timedelta(hours=1)
_GRAFANA_PANEL_H: int = 8
_GRAFANA_PANEL_W: int = 8
_GRAFANA_ROW_GAP: int = 1

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

GPU_BILLING_RATES: Dict[str, float] = {
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
    writer.writerow(["meta", "period_days", report.get("period_days", 0)])
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
    writer.writerow(["gpu_type_breakdown", "gpu_type", "hours", "cost_usd"])
    for gpu_type, details in report.get("gpu_type_breakdown", {}).items():
        writer.writerow(
            [
                "gpu_type_breakdown",
                gpu_type,
                details.get("hours", 0),
                details.get("cost_usd", 0),
            ]
        )

    return output.getvalue()


# ===================================================================
# Public API
# ===================================================================

__all__ = [
    # Constants
    "GPU_PERFORMANCE_INDEX",
    "GPU_BILLING_RATES",
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
    # Helpers (internal, but exported for testing)
    "_now",
    "_to_timestamp_seconds",
    "_linear_regression",
    "_gpu_hours",
    "_next_id",
    "_report_to_csv",
    "YAML_AVAILABLE",
    "yaml_dump",
]
