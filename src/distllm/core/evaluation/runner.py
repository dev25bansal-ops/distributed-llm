"""Main evaluation runner — EvalRunner class and run_all_heim convenience function."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from distllm.core.evaluation.constants import (
    _EVAL_TIMEOUT_S,
    _MAX_WORKERS,
    _SecretStr,
)
from distllm.core.evaluation.db import EvalDB
from distllm.core.evaluation.formatters import (
    _ArenaPromptFormatter,
    _HeimPromptFormatter,
    _MTBenchPromptFormatter,
    PromptFormatter,
)
from distllm.core.evaluation.loaders import (
    _ArenaLoader,
    _GSM8KLoader,
    _HumanEvalLoader,
    _MMLULoader,
    _MTBenchLoader,
    DatasetLoader,
)
from distllm.core.evaluation.models import EvalReport, EvalResult, EvalSample
from distllm.core.evaluation.report import ReportGenerator
from distllm.core.evaluation.scorers import (
    _ArenaScorer,
    _ExactMatchScorer,
    _MTBenchScorer,
    Scorer,
)
from distllm.core.evaluation.worker import _WorkerPool, _count_tokens


# ---------------------------------------------------------------------------
# EvalRunner — public API
# ---------------------------------------------------------------------------


class EvalRunner:
    """Main evaluation runner.

    Coordinates dataset loading, prompt formatting, model inference,
    scoring, and report generation.

    Args:
        coordinator: Optional coordinator instance for local inference.
            If not set, ``generate_fn`` must be provided to ``run()``.
        db_path: Path to SQLite database for persisting results.
        max_workers: Number of parallel evaluation workers.
        api_key: OpenAI API key for judge-based evaluations (MT-Bench, Arena).
            Falls back to ``OPENAI_API_KEY`` env var.
    """

    def __init__(
        self,
        coordinator: Any = None,
        db_path: str | Path = "",
        max_workers: int = _MAX_WORKERS,
        api_key: str = "",
        judge_model: str = "gpt-4",
        judge_api_base: str = "https://api.openai.com/v1",
    ) -> None:
        self._coordinator = coordinator
        self._pool = _WorkerPool(max_workers=max_workers)
        self._db = EvalDB(db_path=db_path)
        self._report_gen = ReportGenerator()
        raw_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._api_key = _SecretStr(raw_key) if raw_key else _SecretStr("")
        self._judge_model = judge_model
        self._judge_api_base = judge_api_base.rstrip("/")
        self._db.initialize()

    # ── Dataset loaders ───────────────────────────────────────────────────

    def _get_loader(self, benchmark: str) -> DatasetLoader:
        mapping: dict[str, type[DatasetLoader]] = {
            "mmlu": _MMLULoader,
            "gsm8k": _GSM8KLoader,
            "humaneval": _HumanEvalLoader,
            "mt_bench": _MTBenchLoader,
            "arena": _ArenaLoader,
        }
        cls = mapping.get(benchmark)
        if cls is None:
            raise ValueError(f"Unknown benchmark: {benchmark}. Choose from {list(mapping.keys())}")
        return cls()

    # ── Prompt formatters ─────────────────────────────────────────────────

    def _get_formatter(self, benchmark: str) -> PromptFormatter:
        mapping: dict[str, Callable[[], PromptFormatter]] = {
            "mmlu": lambda: _HeimPromptFormatter("mmlu"),
            "gsm8k": lambda: _HeimPromptFormatter("gsm8k"),
            "humaneval": lambda: _HeimPromptFormatter("humaneval"),
            "mt_bench": lambda: _MTBenchPromptFormatter(),
            "arena": lambda: _ArenaPromptFormatter(),
        }
        factory = mapping.get(benchmark)
        if factory is None:
            raise ValueError(f"Unknown benchmark: {benchmark}")
        return factory()

    # ── Scorers ───────────────────────────────────────────────────────────

    def _get_scorer(self, benchmark: str) -> Scorer:
        mapping: dict[str, Callable[[], Scorer]] = {
            "mmlu": lambda: _ExactMatchScorer("mmlu"),
            "gsm8k": lambda: _ExactMatchScorer("gsm8k"),
            "humaneval": lambda: _ExactMatchScorer("humaneval"),
            "mt_bench": lambda: _MTBenchScorer(
                api_key=self._api_key.get(),
                judge_model=self._judge_model,
                api_base=self._judge_api_base,
            ),
            "arena": lambda: _ArenaScorer(
                api_key=self._api_key.get(),
                judge_model=self._judge_model,
                api_base=self._judge_api_base,
            ),
        }
        factory = mapping.get(benchmark)
        if factory is None:
            raise ValueError(f"Unknown benchmark: {benchmark}")
        return factory()

    # ── Model inference ──────────────────────────────────────────────────

    def _generate(
        self,
        prompt: str,
        model_id: str = "",
        max_tokens: int = 256,
        temperature: float = 0.0,
        coordinator_url: str = "",
    ) -> tuple[str, float, int, int]:
        """Run model inference via coordinator or API URL.

        Returns:
            ``(prediction_text, latency_ms, prompt_tokens, generated_tokens)``
        """
        if self._coordinator is not None:
            return self._generate_local(prompt, max_tokens, temperature)
        if coordinator_url:
            return self._generate_remote(prompt, coordinator_url, model_id, max_tokens, temperature)
        raise RuntimeError(
            "No coordinator or API URL provided. Pass ``coordinator`` or set ``coordinator_url``."
        )

    def _generate_local(
        self, prompt: str, max_tokens: int = 256, temperature: float = 0.0
    ) -> tuple[str, float, int, int]:
        """Generate using the local coordinator."""
        if self._coordinator is None:
            raise RuntimeError("Coordinator not set")

        start = time.monotonic()
        prediction = self._coordinator.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        ptokens = _count_tokens(prompt)
        gtokens = _count_tokens(prediction)
        return prediction, elapsed_ms, ptokens, gtokens

    def _generate_remote(
        self,
        prompt: str,
        url: str,
        model_id: str = "",
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> tuple[str, float, int, int]:
        """Generate via a remote API endpoint."""
        import httpx

        base_url = url.rstrip("/")
        start = time.monotonic()
        resp = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model_id or "default",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=_EVAL_TIMEOUT_S,
        )
        resp.raise_for_status()
        elapsed_ms = (time.monotonic() - start) * 1000
        data = resp.json()
        prediction = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        ptokens = usage.get("prompt_tokens", _count_tokens(prompt))
        gtokens = usage.get("completion_tokens", _count_tokens(prediction))
        return prediction, elapsed_ms, ptokens, gtokens

    # ── Public run methods ────────────────────────────────────────────────

    def run_heim(
        self,
        benchmark: str,
        model_id: str = "",
        max_tokens: int = 256,
        temperature: float = 0.0,
        coordinator_url: str = "",
        num_samples: int = 20,
    ) -> EvalReport:
        """Run a HEIM-style benchmark (MMLU, GSM8K, HumanEval).

        Args:
            benchmark: One of ``"mmlu"``, ``"gsm8k"``, ``"humaneval"``.
            model_id: Identifier for the model being evaluated.
            max_tokens: Maximum generation tokens per sample.
            temperature: Sampling temperature (0.0 for deterministic).
            coordinator_url: Remote API URL. If empty, uses local coordinator.
            num_samples: Number of samples to evaluate.

        Returns:
            An ``EvalReport`` with aggregated metrics.
        """
        if benchmark not in ("mmlu", "gsm8k", "humaneval"):
            raise ValueError(f"HEIM benchmark must be one of: mmlu, gsm8k, humaneval, got {benchmark}")

        logger.info("Starting HEIM benchmark: {} (model={})", benchmark, model_id)

        loader = self._get_loader(benchmark)
        formatter = self._get_formatter(benchmark)
        scorer = self._get_scorer(benchmark)

        # Override num_samples for loaders that accept it
        if hasattr(loader, "_num_samples"):
            loader._num_samples = num_samples  # type: ignore[assignment]

        samples = loader.load()
        config = {
            "benchmark": benchmark,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "num_samples": len(samples),
        }

        start_time = time.monotonic()

        def _progress(done: int, total: int) -> None:
            if done % max(1, total // 10) == 0 or done == total:
                logger.info("HEIM {} [{}/{}] - {:.0%}", benchmark, done, total, done / total)

        results = self._pool.run(
            samples=samples,
            generate_fn=lambda q: self._generate(
                formatter.format(EvalSample(question=q)),
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                coordinator_url=coordinator_url,
            ),
            progress_cb=_progress,
        )

        # Score results
        for r in results:
            if r.error is None:
                r.score = scorer.score(r.sample, r.prediction)

        duration_s = time.monotonic() - start_time
        report = self._report_gen.generate(model_id, benchmark, config, results, duration_s)

        self._db.save_report(report)
        logger.info("HEIM {} complete: accuracy={}", benchmark, report.metrics.get("accuracy"))
        return report

    def run_mt_bench(
        self,
        model_id: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        coordinator_url: str = "",
        num_categories: int = 8,
    ) -> EvalReport:
        """Run MT-Bench evaluation (multi-turn chat quality).

        Args:
            model_id: Identifier for the model being evaluated.
            max_tokens: Maximum generation tokens per turn.
            temperature: Sampling temperature for generation.
            coordinator_url: Remote API URL. If empty, uses local coordinator.
            num_categories: Number of MT-Bench categories to evaluate (1-8).

        Returns:
            An ``EvalReport`` with quality scores (1-10 scale, normalized to 0-1).
        """
        logger.info("Starting MT-Bench evaluation (model={})", model_id)

        loader = _MTBenchLoader(num_samples=num_categories)
        formatter = _MTBenchPromptFormatter()
        scorer = _MTBenchScorer(api_key=self._api_key)

        samples = loader.load()
        config = {
            "benchmark": "mt_bench",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "num_categories": len(samples),
            "judge_model": "gpt-4",
        }

        start_time = time.monotonic()

        results = self._pool.run(
            samples=samples,
            generate_fn=lambda q: self._generate(
                formatter.format(EvalSample(question=q)),
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                coordinator_url=coordinator_url,
            ),
            progress_cb=lambda d, t: logger.info("MT-Bench [{}/{}]", d, t),
        )

        for r in results:
            if r.error is None:
                r.score = scorer.score(r.sample, r.prediction)

        duration_s = time.monotonic() - start_time
        report = self._report_gen.generate(model_id, "mt_bench", config, results, duration_s)

        self._db.save_report(report)
        logger.info("MT-Bench complete: mean_score={}", report.metrics.get("mean_score"))
        return report

    def run_arena(
        self,
        model_a: str = "",
        model_b: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
        coordinator_url_a: str = "",
        coordinator_url_b: str = "",
        num_samples: int = 10,
    ) -> EvalReport:
        """Run Chatbot Arena-style pairwise comparison.

        Both models respond to the same prompts, then a GPT-4 judge
        determines which response is better.

        Args:
            model_a: Identifier for model A.
            model_b: Identifier for model B.
            max_tokens: Maximum generation tokens per sample.
            temperature: Sampling temperature.
            coordinator_url_a: URL for model A's API. If empty, uses local coordinator.
            coordinator_url_b: URL for model B's API. Falls back to ``coordinator_url_a``.
            num_samples: Number of prompts to compare.

        Returns:
            An ``EvalReport`` where ``accuracy`` represents model A's win rate.
        """
        logger.info("Starting Arena comparison: {} vs {}", model_a, model_b)

        loader = _ArenaLoader(num_samples=num_samples)
        formatter = _ArenaPromptFormatter()
        scorer = _ArenaScorer(api_key=self._api_key)

        samples = loader.load()
        config = {
            "benchmark": "arena",
            "model_a": model_a,
            "model_b": model_b,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "num_samples": len(samples),
            "judge_model": "gpt-4",
        }

        url_b = coordinator_url_b or coordinator_url_a
        start_time = time.monotonic()
        results: list[EvalResult] = []

        for i, sample in enumerate(samples):
            prompt = formatter.format(sample)
            try:
                # Generate from model A
                pred_a, lat_a, pt_a, gt_a = self._generate(
                    prompt, model_id=model_a, max_tokens=max_tokens,
                    temperature=temperature, coordinator_url=coordinator_url_a,
                )
                # Generate from model B
                pred_b, lat_b, pt_b, gt_b = self._generate(
                    prompt, model_id=model_b, max_tokens=max_tokens,
                    temperature=temperature, coordinator_url=url_b,
                )
                combined = f"{pred_a}\n---\n{pred_b}"
                combined_latency = lat_a + lat_b
                results.append(EvalResult(
                    sample=sample,
                    prediction=combined,
                    latency_ms=combined_latency,
                    prompt_tokens=pt_a + pt_b,
                    generated_tokens=gt_a + gt_b,
                ))
            except Exception as exc:
                logger.error("Arena sample {} failed: {}", i, exc)
                results.append(EvalResult(
                    sample=sample,
                    prediction="",
                    error=str(exc),
                ))

            if (i + 1) % max(1, num_samples // 5) == 0 or i + 1 == num_samples:
                logger.info("Arena [{}/{}]", i + 1, num_samples)

        for r in results:
            if r.error is None:
                r.score = scorer.score(r.sample, r.prediction)

        duration_s = time.monotonic() - start_time
        report = self._report_gen.generate(
            f"{model_a}_vs_{model_b}", "arena", config, results, duration_s,
        )

        # Add arena-specific metrics
        win_rate = sum(1 for r in results if r.score == 1.0) / max(len(results), 1)
        tie_rate = sum(1 for r in results if r.score == 0.5) / max(len(results), 1)
        loss_rate = sum(1 for r in results if r.score == 0.0) / max(len(results), 1)
        report.metrics["win_rate"] = round(win_rate, 4)
        report.metrics["tie_rate"] = round(tie_rate, 4)
        report.metrics["loss_rate"] = round(loss_rate, 4)

        self._db.save_report(report)
        logger.info("Arena complete: win_rate={:.1%}, tie_rate={:.1%}", win_rate, tie_rate)
        return report

    # ── Report access ─────────────────────────────────────────────────────

    def list_reports(
        self,
        model_id: str | None = None,
        dataset: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List evaluation reports with optional filtering."""
        return self._db.list_reports(
            model_id=model_id, dataset=dataset, limit=limit, offset=offset,
        )

    def get_report(self, report_id: str) -> dict[str, Any] | None:
        """Get a single report by ID."""
        return self._db.get_report(report_id)

    def get_report_results(self, report_id: str) -> list[dict[str, Any]]:
        """Get detailed results for a report."""
        return self._db.get_report_results(report_id)

    def delete_report(self, report_id: str) -> bool:
        """Delete a report and its results."""
        return self._db.delete_report(report_id)

    def close(self) -> None:
        """Close database connection."""
        self._db.close()


# ---------------------------------------------------------------------------
# Convenience: run all HEIM benchmarks
# ---------------------------------------------------------------------------


def run_all_heim(
    model_id: str = "",
    coordinator_url: str = "",
    num_samples: int = 20,
    runner: EvalRunner | None = None,
) -> dict[str, EvalReport]:
    """Run all three HEIM benchmarks (MMLU, GSM8K, HumanEval).

    Args:
        model_id: Model identifier.
        coordinator_url: Remote API URL.
        num_samples: Samples per benchmark.
        runner: Reusable EvalRunner instance. Creates one if not provided.

    Returns:
        Dict mapping benchmark name to EvalReport.
    """
    close_runner = runner is None
    runner = runner or EvalRunner()
    try:
        reports: dict[str, EvalReport] = {}
        for benchmark in ("mmlu", "gsm8k", "humaneval"):
            reports[benchmark] = runner.run_heim(
                benchmark=benchmark,
                model_id=model_id,
                coordinator_url=coordinator_url,
                num_samples=num_samples,
            )
        return reports
    finally:
        if close_runner:
            runner.close()


__all__ = [
    "EvalRunner",
    "run_all_heim",
]
