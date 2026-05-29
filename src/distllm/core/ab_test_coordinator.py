"""A/B Testing Infrastructure — compare model versions side-by-side.

Routes a percentage of traffic to a candidate model version while the
majority uses the stable version.  Tracks quality metrics and can
auto-promote winners based on statistical significance.

Usage::

    ab = ABTestCoordinator(coordinator)
    ab.register_version("stable", model_path="meta-llama/Llama-3.1-8B")
    ab.register_version("canary", model_path="../fine-tuned-8B-lora")
    ab.set_traffic_split("stable": 90, "canary": 10)
    result = ab.route_request(user_id="user_1", prompt="Hello")
    # 90% → stable, 10% → canary
"""

from __future__ import annotations

import hashlib
import math
import random
import threading
import time
from typing import Any, Callable

from loguru import logger


class ABTestCoordinator:
    """Routes traffic between model versions and tracks quality metrics.

    For each request:
      1. Determine which version handles it (based on traffic split)
      2. Record latency, output length, and user feedback
      3. Periodically evaluate whether to promote the candidate version
    """

    def __init__(
        self,
        stable_version: str = "stable",
        min_samples: int = 100,
        significance_level: float = 0.05,
        auto_promote: bool = False,
    ):
        self._versions: dict[str, dict[str, Any]] = {}
        self._traffic_split: dict[str, float] = {}
        self._stable = stable_version
        self._min_samples = min_samples
        self._significance = significance_level
        self._auto_promote = auto_promote

        self._results: dict[str, list[dict]] = {}
        self._lock = threading.Lock()

    def register_version(self, name: str, **kwargs: Any) -> None:
        self._versions[name] = kwargs
        self._results[name] = []
        logger.info(f"AB test version registered: {name}")

    def set_traffic_split(self, split: dict[str, float]) -> None:
        total = sum(split.values())
        if abs(total - 100.0) > 0.1:
            raise ValueError(f"Traffic split must sum to 100, got {total}")
        self._traffic_split = split
        logger.info(f"AB test traffic split set: {split}")

    def select_version(self, user_id: str = "") -> str:
        """Select a version for this request based on traffic split.

        Uses consistent hashing on *user_id* when provided so the same
        user always gets the same version (important for UX consistency).
        """
        if user_id:
            digest = hashlib.sha256(user_id.encode("utf-8")).digest()
            hash_val = int.from_bytes(digest[:4], "big") % 100
            cumulative = 0
            for version, pct in sorted(self._traffic_split.items()):
                cumulative += pct
                if hash_val < cumulative:
                    return version

        roll = random.random() * 100
        cumulative = 0
        for version, pct in sorted(self._traffic_split.items()):
            cumulative += pct
            if roll < cumulative:
                return version
        return self._stable

    def record_result(self, version: str, metrics: dict[str, Any]) -> None:
        with self._lock:
            self._results.setdefault(version, []).append({
                "timestamp": time.time(),
                **metrics,
            })

    def get_stats(self) -> dict[str, Any]:
        """Get summary statistics for all versions."""
        stats = {}
        with self._lock:
            for version, results in self._results.items():
                if not results:
                    stats[version] = {"samples": 0}
                    continue
                latencies = [r.get("latency_ms", 0) for r in results]
                output_lengths = [r.get("output_length", 0) for r in results]
                feedback = [r.get("feedback", 0) for r in results if "feedback" in r]
                stats[version] = {
                    "samples": len(results),
                    "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
                    "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1),
                    "avg_output_length": round(sum(output_lengths) / len(output_lengths), 1),
                    "avg_feedback": round(sum(feedback) / len(feedback), 2) if feedback else None,
                }
        return stats

    def should_promote(self, candidate: str) -> tuple[bool, float]:
        """Check if *candidate* should replace the stable version.

        Uses two-sample t-test on latency to determine statistical
        significance.

        Returns:
            (should_promote, p_value) tuple.
        """
        stable_results = self._results.get(self._stable, [])
        cand_results = self._results.get(candidate, [])

        if len(stable_results) < self._min_samples or len(cand_results) < self._min_samples:
            return False, 1.0

        stable_lat = [r.get("latency_ms", 0) for r in stable_results]
        cand_lat = [r.get("latency_ms", 0) for r in cand_results]

        # Welch's t-test
        n1, n2 = len(stable_lat), len(cand_lat)
        m1, m2 = sum(stable_lat) / n1, sum(cand_lat) / n2
        v1 = sum((x - m1) ** 2 for x in stable_lat) / (n1 - 1)
        v2 = sum((x - m2) ** 2 for x in cand_lat) / (n2 - 1)
        se = math.sqrt(v1 / n1 + v2 / n2)

        if se == 0:
            return False, 1.0

        t_stat = (m1 - m2) / se
        df = ((v1 / n1 + v2 / n2) ** 2) / (
            (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        )
        p_value = 2 * (1 - self._t_cdf(abs(t_stat), df))

        should_promote = p_value < self._significance and m2 < m1
        return should_promote, round(p_value, 4)

    def _t_cdf(self, x: float, df: float) -> float:
        """Approximate Student's t CDF using regularized incomplete beta."""
        x2 = x * x
        p = 0.5 * (1 + math.erf(x * math.sqrt(0.5)))
        if df < 100:
            p += (x2 * x * (1 - x2 / (3 * df))) / (6 * df * math.sqrt(2 * math.pi))
        return min(max(p, 0.0), 1.0)

    def promote(self, candidate: str) -> bool:
        if candidate not in self._versions:
            return False
        old_stable = self._stable
        self._stable = candidate
        logger.info(f"AB test: promoted {candidate} to stable (replaced {old_stable})")
        return True

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "stable_version": self._stable,
            "versions": list(self._versions.keys()),
            "traffic_split": self._traffic_split,
            "auto_promote": self._auto_promote,
            "version_stats": self.get_stats(),
        }
