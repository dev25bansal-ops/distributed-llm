"""Speculative decoding dashboard metrics.

Collects and exports per-method comparison metrics for the
auto-speculative selection system.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class MethodComparison:
    """Comparison data between two speculative methods."""
    method_a: str
    method_b: str
    acceptance_rate_a: float
    acceptance_rate_b: float
    avg_speedup_a: float
    avg_speedup_b: float
    tokens_per_sec_a: float
    tokens_per_sec_b: float
    samples_a: int
    samples_b: int
    timestamp: float = 0.0

    @property
    def winner(self) -> str:
        """Return the method with higher throughput."""
        if self.tokens_per_sec_a > self.tokens_per_sec_b:
            return self.method_a
        return self.method_b


class SpeculativeDashboard:
    """Collects and exports speculative decoding metrics.

    Tracks per-method acceptance rates, speedups, and throughput.
    Provides comparison reports and REST API data export.
    """

    def __init__(self) -> None:
        self._method_data: dict[str, dict[str, Any]] = {}
        self._comparisons: list[MethodComparison] = []
        self._last_update: float = 0.0

    def update_method(
        self,
        method: str,
        workload_type: str,
        acceptance_rate: float,
        avg_speedup: float,
        tokens_per_sec: float,
        samples: int,
    ) -> None:
        """Update metrics for a method/workload combination."""
        key = f"{method}:{workload_type}"
        self._method_data[key] = {
            "method": method,
            "workload_type": workload_type,
            "acceptance_rate": acceptance_rate,
            "avg_speedup": avg_speedup,
            "tokens_per_sec": tokens_per_sec,
            "samples": samples,
            "last_updated": time.time(),
        }
        self._last_update = time.time()

    def record_comparison(
        self,
        method_a: str,
        method_b: str,
        metrics_a: dict[str, float],
        metrics_b: dict[str, float],
        workload_type: str = "all",
    ) -> MethodComparison:
        """Record a comparison between two methods."""
        comparison = MethodComparison(
            method_a=method_a,
            method_b=method_b,
            acceptance_rate_a=metrics_a.get("acceptance_rate", 0),
            acceptance_rate_b=metrics_b.get("acceptance_rate", 0),
            avg_speedup_a=metrics_a.get("avg_speedup", 1.0),
            avg_speedup_b=metrics_b.get("avg_speedup", 1.0),
            tokens_per_sec_a=metrics_a.get("tokens_per_sec", 0),
            tokens_per_sec_b=metrics_b.get("tokens_per_sec", 0),
            samples_a=int(metrics_a.get("samples", 0)),
            samples_b=int(metrics_b.get("samples", 0)),
            timestamp=time.time(),
        )
        self._comparisons.append(comparison)
        logger.debug(
            f"Speculative comparison: {method_a} vs {method_b} on {workload_type} -> "
            f"winner: {comparison.winner}"
        )
        return comparison

    def get_comparison_report(self) -> dict[str, Any]:
        """Return a full comparison report of all methods."""
        # Group by method
        method_summary: dict[str, dict[str, Any]] = {}
        for key, data in self._method_data.items():
            method = data["method"]
            if method not in method_summary:
                method_summary[method] = {
                    "method": method,
                    "workload_types": {},
                    "overall_acceptance_rate": 0.0,
                    "overall_tokens_per_sec": 0.0,
                    "total_samples": 0,
                }
            method_summary[method]["workload_types"][data["workload_type"]] = {
                "acceptance_rate": data["acceptance_rate"],
                "avg_speedup": data["avg_speedup"],
                "tokens_per_sec": data["tokens_per_sec"],
                "samples": data["samples"],
            }
            method_summary[method]["total_samples"] += data["samples"]

        # Compute overall averages (weighted by samples)
        for method, summary in method_summary.items():
            total_samples = summary["total_samples"]
            if total_samples > 0:
                weighted_rate = sum(
                    d["acceptance_rate"] * d["samples"]
                    for d in summary["workload_types"].values()
                ) / total_samples
                weighted_tps = sum(
                    d["tokens_per_sec"] * d["samples"]
                    for d in summary["workload_types"].values()
                ) / total_samples
                summary["overall_acceptance_rate"] = round(weighted_rate, 4)
                summary["overall_tokens_per_sec"] = round(weighted_tps, 2)

        # Recent comparisons
        recent = self._comparisons[-10:]
        comparison_list = [
            {
                "method_a": c.method_a,
                "method_b": c.method_b,
                "winner": c.winner,
                "timestamp": c.timestamp,
            }
            for c in recent
        ]

        return {
            "method_summary": method_summary,
            "recent_comparisons": comparison_list,
            "last_updated": self._last_update,
        }

    def export_json(self) -> str:
        """Export dashboard data as JSON for REST API."""
        return json.dumps(self.get_comparison_report(), indent=2)

    def reset(self) -> None:
        """Clear all dashboard data."""
        self._method_data.clear()
        self._comparisons.clear()
        self._last_update = 0.0
