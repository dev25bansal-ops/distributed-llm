"""Report generator for LLM evaluation results.

Extracted from :mod:`distllm.core.evaluation_harness`.
"""

from __future__ import annotations

from typing import Any

from distllm.core.evaluation.constants import EvalStatus
from distllm.core.evaluation.models import EvalReport, EvalResult


class ReportGenerator:
    """Generates aggregated evaluation reports from raw results."""

    def generate(
        self,
        model_id: str,
        dataset: str,
        config: dict[str, Any],
        results: list[EvalResult],
        duration_s: float,
    ) -> EvalReport:
        """Aggregate results into a scored report."""
        scored = [r for r in results if r.error is None]
        errors = [r for r in results if r.error is not None]

        scores = [r.score for r in scored]
        latencies = [r.latency_ms for r in scored]
        prompt_tokens = sum(r.prompt_tokens for r in scored)
        generated_tokens = sum(r.generated_tokens for r in scored)

        metrics: dict[str, float] = {
            "accuracy": round(sum(scores) / max(len(scores), 1), 4),
            "mean_score": round(sum(scores) / max(len(scores), 1), 4),
            "median_score": round(sorted(scores)[len(scores) // 2], 4) if scores else 0.0,
            "std_score": round(self._std(scores), 4) if len(scores) > 1 else 0.0,
            "total_samples": len(results),
            "scored_samples": len(scored),
            "error_samples": len(errors),
            "error_rate": round(len(errors) / max(len(results), 1), 4),
            "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 2),
            "p50_latency_ms": round(sorted(latencies)[len(latencies) // 2], 2) if latencies else 0.0,
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if len(latencies) > 1 else 0.0,
            "p99_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 2) if len(latencies) > 2 else 0.0,
            "total_prompt_tokens": prompt_tokens,
            "total_generated_tokens": generated_tokens,
            "duration_s": round(duration_s, 2),
        }

        # Per-category breakdown
        categories: dict[str, list[float]] = {}
        for r in scored:
            cat = r.sample.category
            categories.setdefault(cat, []).append(r.score)
        for cat, cat_scores in categories.items():
            metrics[f"{cat}_accuracy"] = round(sum(cat_scores) / len(cat_scores), 4)

        return EvalReport(
            model_id=model_id,
            dataset=dataset,
            config=config,
            metrics=metrics,
            results=results,
            status=EvalStatus.COMPLETED,
            duration_s=duration_s,
        )

    @staticmethod
    def _std(values: list[float]) -> float:
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5


__all__ = [
    "ReportGenerator",
]
