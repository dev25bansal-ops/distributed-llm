"""Speculative Dashboard — comparison reporting for speculative methods.

Tracks performance metrics for each speculative decoding method and
provides comparison reports to help operators choose the best method
for their workload.

Usage::

    dashboard = SpeculativeDashboard()
    dashboard.update_method("ngram", "code", 0.7, 1.5, 100.0, 50)
    dashboard.update_method("eagle", "code", 0.5, 1.3, 80.0, 30)

    report = dashboard.get_comparison_report()
    comparison = dashboard.record_comparison("ngram", "eagle", ...)
    data = dashboard.export_json()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class MethodStats:
    """Performance stats for a single speculative method."""
    method: str = ""
    workload_type: str = ""
    acceptance_rate: float = 0.0
    avg_speedup: float = 1.0
    tokens_per_sec: float = 0.0
    samples: int = 0
    last_updated: float = 0.0


@dataclass
class MethodComparison:
    """Result of comparing two speculative methods."""
    method_a: str = ""
    method_b: str = ""
    winner: str = ""
    metric: str = ""
    value_a: float = 0.0
    value_b: float = 0.0
    timestamp: float = 0.0


class SpeculativeDashboard:
    """Tracks and compares speculative decoding method performance.

    Usage::

        dashboard = SpeculativeDashboard()
        dashboard.update_method("ngram", "code", 0.7, 1.5, 100.0, 50)
        report = dashboard.get_comparison_report()
    """

    def __init__(self) -> None:
        self._methods: dict[str, MethodStats] = {}
        self._comparisons: list[MethodComparison] = []

    def update_method(
        self,
        method: str,
        workload_type: str,
        acceptance_rate: float,
        avg_speedup: float,
        tokens_per_sec: float,
        samples: int,
    ) -> None:
        """Update performance metrics for a method."""
        self._methods[method] = MethodStats(
            method=method,
            workload_type=workload_type,
            acceptance_rate=acceptance_rate,
            avg_speedup=avg_speedup,
            tokens_per_sec=tokens_per_sec,
            samples=samples,
            last_updated=time.time(),
        )

    def record_comparison(
        self,
        method_a: str,
        method_b: str,
        stats_a: dict[str, Any],
        stats_b: dict[str, Any],
        metric: str = "acceptance_rate",
    ) -> MethodComparison:
        """Record a comparison between two methods."""
        val_a = stats_a.get(metric, 0.0)
        val_b = stats_b.get(metric, 0.0)
        winner = method_a if val_a >= val_b else method_b

        comparison = MethodComparison(
            method_a=method_a,
            method_b=method_b,
            winner=winner,
            metric=metric,
            value_a=val_a,
            value_b=val_b,
            timestamp=time.time(),
        )
        self._comparisons.append(comparison)
        return comparison

    def get_comparison_report(self) -> dict[str, Any]:
        """Generate a comparison report of all tracked methods."""
        method_summary: dict[str, dict[str, Any]] = {}
        for method, stats in self._methods.items():
            method_summary[method] = {
                "acceptance_rate": stats.acceptance_rate,
                "avg_speedup": stats.avg_speedup,
                "tokens_per_sec": stats.tokens_per_sec,
                "samples": stats.samples,
                "workload_type": stats.workload_type,
            }

        return {
            "method_summary": method_summary,
            "total_comparisons": len(self._comparisons),
        }

    def export_json(self) -> str:
        """Export dashboard data as JSON string."""
        report = self.get_comparison_report()
        return json.dumps(report, indent=2)

    def reset(self) -> None:
        """Clear all tracked data."""
        self._methods.clear()
        self._comparisons.clear()
