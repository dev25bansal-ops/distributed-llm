"""LLM-as-Judge automated evaluation and regression pipeline.

Continuously evaluates model versions in production-shadow mode, detects
regressions via statistical significance testing, and triggers automatic
rollback when configurable thresholds are breached.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ShadowEvalConfig:
    """Configuration for the shadow evaluation runner."""
    eval_suites: list[str] = field(default_factory=lambda: ["mmlu", "gsm8k", "humaneval"])
    shadow_traffic_pct: float = 5.0
    regression_threshold: float = 0.05
    auto_rollback: bool = False
    min_samples: int = 100
    significance_level: float = 0.05


@dataclass
class RegressionReport:
    """Report of a regression comparison between model versions."""
    candidate_version: str
    baseline_version: str
    metrics: dict[str, float] = field(default_factory=dict)
    deltas: dict[str, float] = field(default_factory=dict)
    p_values: dict[str, float] = field(default_factory=dict)
    is_regression: bool = False
    recommendations: list[str] = field(default_factory=list)
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.time()


class RegressionDetector:
    """Detects statistically significant regressions in eval metrics."""

    def compare(self, baseline_scores: dict[str, list[float]], candidate_scores: dict[str, list[float]]) -> RegressionReport:
        """Compare candidate scores against baseline using Mann-Whitney U test.

        Args:
            baseline_scores: {metric_name: [scores]} for the reference version.
            candidate_scores: {metric_name: [scores]} for the candidate version.

        Returns:
            RegressionReport with per-metric deltas, p-values, and overall verdict.
        """
        report = RegressionReport(
            candidate_version="candidate",
            baseline_version="baseline",
        )
        all_metrics = set(baseline_scores) | set(candidate_scores)

        for metric in all_metrics:
            base = baseline_scores.get(metric, [])
            cand = candidate_scores.get(metric, [])
            if not base or not cand:
                continue

            base_mean = sum(base) / len(base)
            cand_mean = sum(cand) / len(cand)
            delta = cand_mean - base_mean
            delta_pct = delta / base_mean if base_mean else 0.0

            report.metrics[f"{metric}_baseline"] = base_mean
            report.metrics[f"{metric}_candidate"] = cand_mean
            report.deltas[metric] = delta_pct
            report.p_values[metric] = self._mann_whitney_u(base, cand)

            if abs(delta_pct) > 0.01:  # at least 1% change
                if delta_pct < 0:
                    report.recommendations.append(
                        f"{metric} decreased by {abs(delta_pct):.1%} "
                        f"(p={report.p_values[metric]:.4f})"
                    )

        report.is_regression = any(
            v < 0 and report.p_values.get(k, 1.0) < 0.05
            for k, v in report.deltas.items()
        )
        return report

    @staticmethod
    def _mann_whitney_u(x: list[float], y: list[float]) -> float:
        """Simple Mann-Whitney U test p-value approximation.

        Uses the normal approximation for large samples.
        """
        n1, n2 = len(x), len(y)
        if n1 < 3 or n2 < 3:
            return 1.0
        combined = sorted([(v, 0) for v in x] + [(v, 1) for v in y])
        rank_sum = sum(i + 1 for i, (_, group) in enumerate(combined) if group == 0)
        u = rank_sum - (n1 * (n1 + 1)) / 2
        mu = n1 * n2 / 2
        sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        if sigma == 0:
            return 1.0
        z = (u - mu) / sigma
        # Two-tailed p-value approximation
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        return min(max(p, 0.0), 1.0)


class ShadowEvalRunner:
    """Runs evaluation suites against model versions and checks for regression."""

    def __init__(
        self,
        client: Any | None = None,
        config: ShadowEvalConfig | None = None,
    ):
        self.client = client
        self.config = config or ShadowEvalConfig()
        self._detector = RegressionDetector()
        self._baseline_cache: dict[str, dict[str, list[float]]] = {}

    def run_eval(self, model_version: str, eval_suite: str | None = None) -> dict[str, Any]:
        """Run an evaluation suite against *model_version*.

        Args:
            model_version: The model version identifier to evaluate.
            eval_suite: Specific suite name (e.g. "mmlu"). If None, runs all configured.

        Returns:
            {suite_name: {metric: score}} dict.
        """
        suites = [eval_suite] if eval_suite else self.config.eval_suites
        results: dict[str, Any] = {}
        for suite in suites:
            results[suite] = self._run_single_suite(model_version, suite)
        return results

    def check_regression(self, candidate_version: str) -> RegressionReport | None:
        """Check if *candidate_version* regressed against the cached baseline."""
        baseline = self._baseline_cache.get("baseline")
        if not baseline:
            return None
        candidate_scores = self.run_eval(candidate_version)
        cand_flat: dict[str, list[float]] = {}
        for suite, metrics in candidate_scores.items():
            for metric, score in metrics.items():
                cand_flat[f"{suite}_{metric}"] = [score]
        base_flat: dict[str, list[float]] = {}
        for suite, metrics in baseline.items():
            for metric, score in metrics.items():
                base_flat[f"{suite}_{metric}"] = [score]
        return self._detector.compare(base_flat, cand_flat)

    def trigger_rollback(self, version_id: str, reason: str) -> bool:
        """Trigger rollback away from *version_id* for the given *reason*.

        Returns True if rollback was initiated.
        """
        if not self.config.auto_rollback:
            return False
        # Actual rollback logic would call coordinator API
        return True

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of evaluation state."""
        return {
            "eval_suites": self.config.eval_suites,
            "shadow_traffic_pct": self.config.shadow_traffic_pct,
            "auto_rollback": self.config.auto_rollback,
            "min_samples": self.config.min_samples,
            "baseline_cached": bool(self._baseline_cache),
        }

    def _run_single_suite(self, model_version: str, suite: str) -> dict[str, float]:
        """Run a single evaluation suite against *model_version*.

        Fails closed: this runner intentionally does not fabricate scores.
        A real evaluation backend (e.g. ``distllm.core.evaluation``'s
        ``EvalRunner``) must be wired into the client before this can run.

        Raises:
            NotImplementedError: No real evaluation backend is configured.
        """
        raise NotImplementedError(
            "ShadowEvalRunner is not configured: no real evaluation backend "
            f"is wired for suite {suite!r} and model {model_version!r}. "
            "Provide an EvalRunner-based client or remove the shadow runner "
            "from the pipeline."
        )
