"""Evaluation service -- orchestrates benchmark runs for the API layer.

Usage::

    from distllm.api.services.eval_service import EvalService

    service = EvalService(coordinator)
    results = service.run_benchmarks(
        model_id="my-model",
        benchmarks=["mmlu", "gsm8k"],
    )
"""

from __future__ import annotations

import threading
from typing import Any

from loguru import logger

from distllm.core.evaluation_harness import EvalReport, EvalRunner


class EvalService:
    """Orchestrates LLM evaluation benchmarks.

    The constructor takes a *coordinator* (not importing from ``api_state``).
    Provides a unified interface to run MMLU, GSM8K, HumanEval, MT-Bench,
    and Chatbot Arena evaluations.

    The underlying ``EvalRunner`` is lazily created on first use and reused
    across subsequent calls.
    """

    def __init__(self, coordinator: Any) -> None:
        self._coordinator = coordinator
        self._runner: EvalRunner | None = None
        self._runner_lock = threading.RLock()

    # -- lazy runner -----------------------------------------------------------

    def _get_runner(self) -> EvalRunner:
        """Return the shared ``EvalRunner``, creating it on first call.

        Thread-safe via ``threading.Lock()``.
        """
        if self._runner is not None:
            return self._runner
        with self._runner_lock:
            if self._runner is None:
                self._runner = EvalRunner(coordinator=self._coordinator)
                logger.debug("EvalRunner created lazily")
        return self._runner

    # -- run benchmarks --------------------------------------------------------

    def run_benchmarks(
        self,
        model_id: str,
        benchmarks: list[str] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        num_samples: int = 20,
        coordinator_url: str = "",
        model_b: str = "",
        coordinator_url_b: str = "",
    ) -> dict[str, dict[str, Any]]:
        """Run one or more evaluation benchmarks.

        Args:
            model_id: Identifier for the primary model being evaluated.
            benchmarks: List of benchmarks to run. Supported values are
                ``"mmlu"``, ``"gsm8k"``, ``"humaneval"``, ``"mt_bench"``,
                and ``"arena"``. Defaults to all five.
            max_tokens: Maximum generation tokens per sample.
            temperature: Sampling temperature.
            num_samples: Number of samples for HEIM and Arena benchmarks.
                For MT-Bench this caps the number of categories (1-8).
            coordinator_url: Remote API URL for the primary model. If empty,
                uses the local coordinator.
            model_b: Identifier for the second model (Arena only).
            coordinator_url_b: Remote API URL for model B (Arena only).
                Falls back to *coordinator_url* when empty.

        Returns:
            A dict mapping each benchmark name to its serialized
            ``EvalReport`` dict.
        """
        if benchmarks is None:
            benchmarks = ["mmlu", "gsm8k", "humaneval", "mt_bench", "arena"]

        runner = self._get_runner()
        results: dict[str, dict[str, Any]] = {}

        for benchmark in benchmarks:
            logger.info("Running benchmark: {} (model={})", benchmark, model_id)

            if benchmark in ("mmlu", "gsm8k", "humaneval"):
                report = runner.run_heim(
                    benchmark=benchmark,
                    model_id=model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    coordinator_url=coordinator_url,
                    num_samples=num_samples,
                )
            elif benchmark == "mt_bench":
                report = runner.run_mt_bench(
                    model_id=model_id,
                    max_tokens=max_tokens or 1024,
                    temperature=temperature or 0.7,
                    coordinator_url=coordinator_url,
                    num_categories=min(num_samples, 8),
                )
            elif benchmark == "arena":
                report = runner.run_arena(
                    model_a=model_id,
                    model_b=model_b or f"{model_id}_opponent",
                    max_tokens=max_tokens or 512,
                    temperature=temperature or 0.7,
                    coordinator_url_a=coordinator_url,
                    coordinator_url_b=coordinator_url_b,
                    num_samples=num_samples,
                )
            else:
                logger.warning("Unknown benchmark: {}, skipping", benchmark)
                continue

            results[benchmark] = self._report_to_dict(report)

        return results

    # -- read / query / delete (proxied to EvalRunner.EvalDB) -----------------

    def list_reports(
        self,
        model_id: str | None = None,
        dataset: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List persisted evaluation reports with optional filtering."""
        return self._get_runner().list_reports(
            model_id=model_id,
            dataset=dataset,
            limit=limit,
            offset=offset,
        )

    def get_report(self, report_id: str) -> dict[str, Any] | None:
        """Retrieve a single report header by ID."""
        return self._get_runner().get_report(report_id)

    def get_report_results(self, report_id: str) -> list[dict[str, Any]]:
        """Retrieve all result rows for a given report."""
        return self._get_runner().get_report_results(report_id)

    def delete_report(self, report_id: str) -> bool:
        """Delete a report and its associated results."""
        return self._get_runner().delete_report(report_id)

    # -- serialization ---------------------------------------------------------

    def _report_to_dict(self, report: EvalReport) -> dict[str, Any]:
        """Convert an ``EvalReport`` to a JSON-serializable dict.

        Args:
            report: The evaluation report to serialize.

        Returns:
            A plain dict with report metadata, metrics, and per-sample
            results.
        """
        return {
            "report_id": report.report_id,
            "model_id": report.model_id,
            "dataset": report.dataset,
            "status": report.status.value,
            "config": report.config,
            "metrics": report.metrics,
            "created_at": report.created_at,
            "duration_s": report.duration_s,
            "results": [
                {
                    "question": r.sample.question,
                    "answer": r.sample.answer,
                    "category": r.sample.category,
                    "prediction": r.prediction,
                    "score": r.score,
                    "latency_ms": r.latency_ms,
                    "prompt_tokens": r.prompt_tokens,
                    "generated_tokens": r.generated_tokens,
                    "error": r.error,
                    "metadata": r.sample.metadata,
                }
                for r in report.results
            ],
        }
