"""Evaluation harness for LLM outputs.

Provides local heuristic-based evaluation metrics and model comparison tools.
Uses pure-Python token overlap and lexical diversity heuristics — no external
evaluation dependencies required.
"""

from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

import httpx


__all__ = [
    "EvalMetric",
    "EvalResult",
    "EvalRun",
    "EvalHarness",
]


class EvalMetric(Enum):
    """Evaluation metric types supported by the harness.

    Each member corresponds to a heuristic scorer built into
    :class:`EvalHarness`.  Use ``EvalMetric.CUSTOM`` with
    :meth:`EvalHarness.run_custom` for user-defined scorers.
    """

    RAGAS = "ragas"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXT_PRECISION = "context_precision"
    FAITHFULNESS = "faithfulness"
    HALLUCINATION = "hallucination"
    CUSTOM = "custom"


@dataclass(frozen=True)
class EvalResult:
    """A single evaluation metric result.

    Attributes:
        metric: The metric that was evaluated.
        score: Normalised score between 0.0 and 1.0 (higher is better).
        details: Arbitrary metadata describing how the score was computed
            (per-item scores, token counts, overlap lists, etc.).
        threshold: Pass/fail threshold used for this result.
        passed: ``True`` when ``score >= threshold``.
    """

    metric: EvalMetric
    score: float
    details: dict[str, Any]
    threshold: float
    passed: bool


@dataclass(frozen=True)
class EvalRun:
    """Results of a full evaluation run across one or more metrics.

    Attributes:
        id: Unique run identifier (UUID4 hex).
        model: Model name or identifier evaluated.
        metrics: Per-metric results.
        avg_score: Arithmetic mean of all metric scores in this run.
        timestamp: ISO-8601 UTC timestamp of when the run was created.
    """

    id: str
    model: str
    metrics: list[EvalResult]
    avg_score: float
    timestamp: str


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Split *text* into whitespace-separated, lowercased tokens."""
    return text.lower().split()


def _counter_intersection_score(
    tokens_a: list[str],
    tokens_b: list[str],
) -> float:
    """Fraction of *tokens_b* instances whose token appears in *tokens_a*.

    Counts how many token **occurrences** in *tokens_b* match the **unique**
    vocabulary of *tokens_a*, then divides by the total number of tokens
    in *tokens_b*.  This gives higher weight to repeated tokens in B that
    are also present in A.

    Returns 0.0 if either list is empty.
    """
    if not tokens_a or not tokens_b:
        return 0.0
    unique_a = set(tokens_a)
    counter_b = Counter(tokens_b)
    total = sum(counter_b.values())
    if total == 0:
        return 0.0
    matched = sum(count for token, count in counter_b.items() if token in unique_a)
    return matched / total


# ---------------------------------------------------------------------------
# EvalHarness
# ---------------------------------------------------------------------------


class EvalHarness:
    """Evaluation harness for scoring LLM outputs.

    Supports built-in heuristic metrics (answer relevancy, faithfulness,
    hallucination, context precision, RAGAS composite) and custom scoring
    functions.  An optional ``httpx.AsyncClient`` can be provided for API
    calls — for example via :meth:`compare_models` — but evaluation runs
    themselves are performed entirely locally.

    Typical usage::

        harness = EvalHarness()
        run = await harness.evaluate(
            model="my-model",
            questions=["What is the capital of France?"],
            answers=["Paris is the capital of France."],
            contexts=["France is a country in Europe. Its capital is Paris."],
        )
        print(run.avg_score, run.metrics[0].passed)
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        default_threshold: float = 0.5,
    ) -> None:
        """Initialise the evaluation harness.

        Args:
            http_client: Optional ``httpx.AsyncClient`` for API-based operations
                such as ``compare_models()``.
            default_threshold: Default pass/fail threshold applied to all metrics
                unless overridden per-call.
        """
        self._http_client = http_client
        self._default_threshold = default_threshold

    # -- Public API ---------------------------------------------------------------

    async def evaluate(
        self,
        model: str,
        questions: list[str],
        answers: list[str],
        contexts: list[str] | None = None,
        metrics: list[EvalMetric] | None = None,
        threshold: float | None = None,
    ) -> EvalRun:
        """Run evaluation on model outputs using local heuristic metrics.

        Each metric is computed per question/answer (and optionally per
        context) pair, then averaged across all items for the final score.

        Args:
            model: Model name or identifier.
            questions: List of input questions.
            answers: List of model-generated answers (must match question count).
            contexts: Optional list of reference contexts (must match question
                count).  When omitted, metrics that rely on context
                (faithfulness, hallucination, context precision, RAGAS) will
                receive an empty string and produce a score of 0.0.
            metrics: Metrics to compute.  Defaults to all supported metrics
                except ``EvalMetric.CUSTOM`` (which has no built-in scorer).
            threshold: Pass/fail threshold for this run.  Falls back to
                ``default_threshold`` when omitted.

        Returns:
            An ``EvalRun`` containing per-metric results and an average score.

        Raises:
            ValueError: If input lists have mismatched lengths.
        """
        n = len(questions)
        if len(answers) != n:
            raise ValueError(
                f"question/answer count mismatch: {n} questions vs "
                f"{len(answers)} answers"
            )
        if contexts is not None and len(contexts) != n:
            raise ValueError(
                f"question/context count mismatch: {n} questions vs "
                f"{len(contexts)} contexts"
            )

        threshold = threshold if threshold is not None else self._default_threshold

        # Determine the active metric set, stripping CUSTOM (no built-in scorer).
        if metrics is not None:
            active_metrics = [m for m in metrics if m != EvalMetric.CUSTOM]
        else:
            active_metrics = [
                m
                for m in EvalMetric
                if m not in (EvalMetric.CUSTOM,)
            ]

        contexts = contexts or [""] * n

        results: list[EvalResult] = []
        for metric in active_metrics:
            scores: list[float] = []
            per_item_details: list[dict[str, Any]] = []
            for i, question in enumerate(questions):
                answer = answers[i]
                context = contexts[i]
                score, details = self._score_single(metric, question, answer, context)
                scores.append(score)
                per_item_details.append(details)

            avg_score = sum(scores) / len(scores) if scores else 0.0
            passed = avg_score >= threshold

            results.append(
                EvalResult(
                    metric=metric,
                    score=avg_score,
                    details={
                        "per_item_scores": scores,
                        "per_item_details": per_item_details,
                        "count": len(scores),
                    },
                    threshold=threshold,
                    passed=passed,
                )
            )

        avg_score = (
            sum(r.score for r in results) / len(results) if results else 0.0
        )

        return EvalRun(
            id=str(uuid.uuid4()),
            model=model,
            metrics=results,
            avg_score=avg_score,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    async def compare_models(
        self,
        models: list[str],
        questions: list[str],
        metrics: list[EvalMetric] | None = None,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
    ) -> dict[str, EvalRun]:
        """Run the same questions through multiple models and compare results.

        Creates one :class:`~distllm_sdk.client.DistLLMClient` per model
        (imported locally to avoid circular dependencies), sends the questions
        as chat prompts, collects responses, then evaluates each model's
        answers with the same metric set.

        Args:
            models: Model identifiers to compare.
            questions: List of questions to ask each model.
            metrics: Metrics to evaluate.  Defaults to all supported metrics.
            base_url: DistLLM API base URL.
            api_key: Optional API key for authenticated endpoints.

        Returns:
            A dict mapping each model name to its ``EvalRun`` result,
            keyed by model identifier.
        """
        # Local import avoids a circular dependency in the SDK package.
        from distllm_sdk.client import DistLLMClient

        metrics = metrics or [
            m for m in EvalMetric if m != EvalMetric.CUSTOM
        ]
        results: dict[str, EvalRun] = {}

        for model in models:
            async with DistLLMClient(
                base_url=base_url,
                api_key=api_key,
            ) as client:
                answers: list[str] = []
                for question in questions:
                    resp = await client.chat_completions(
                        messages=[{"role": "user", "content": question}],
                        model=model,
                    )
                    answer_text = ""
                    if resp.choices and resp.choices[0].message:
                        answer_text = resp.choices[0].message.content or ""
                    answers.append(answer_text)

                eval_run = await self.evaluate(
                    model=model,
                    questions=questions,
                    answers=answers,
                    metrics=metrics,
                )
                results[model] = eval_run

        return results

    async def run_custom(
        self,
        metric_name: str,
        scorer_fn: Callable[[list[dict[str, Any]]], float],
        data: list[dict[str, Any]],
        threshold: float | None = None,
    ) -> EvalResult:
        """Run a custom evaluation using a user-provided scoring function.

        The scorer function receives the full ``data`` list and should return
        a single float in the range ``[0.0, 1.0]`` (values will be clamped).
        Results are returned as an ``EvalResult`` with
        ``metric=EvalMetric.CUSTOM``.

        Args:
            metric_name: Human-readable label for this custom metric (stored
                in ``details``).
            scorer_fn: Callable that receives the ``data`` list and returns a
                score between 0.0 and 1.0.
            data: Arbitrary list of dicts to pass to the scorer function.
            threshold: Pass/fail threshold.  Falls back to
                ``default_threshold`` when omitted.

        Returns:
            An ``EvalResult`` with ``metric=EvalMetric.CUSTOM``.
        """
        threshold = threshold if threshold is not None else self._default_threshold
        score = scorer_fn(data)
        score = max(0.0, min(1.0, float(score)))
        passed = score >= threshold

        return EvalResult(
            metric=EvalMetric.CUSTOM,
            score=score,
            details={
                "metric_name": metric_name,
                "data_count": len(data),
            },
            threshold=threshold,
            passed=passed,
        )

    # -- Metric computation helpers -----------------------------------------------

    def _score_single(
        self,
        metric: EvalMetric,
        question: str,
        answer: str,
        context: str,
    ) -> tuple[float, dict[str, Any]]:
        """Dispatch a single question/answer/context triple to the correct scorer."""
        q_tokens = _tokenize(question)
        a_tokens = _tokenize(answer)
        c_tokens = _tokenize(context) if context else []

        if metric == EvalMetric.ANSWER_RELEVANCY:
            return self._score_answer_relevancy(q_tokens, a_tokens)
        if metric == EvalMetric.FAITHFULNESS:
            return self._score_faithfulness(a_tokens, c_tokens)
        if metric == EvalMetric.HALLUCINATION:
            return self._score_hallucination(a_tokens, c_tokens)
        if metric == EvalMetric.CONTEXT_PRECISION:
            return self._score_context_precision(a_tokens, c_tokens)
        if metric == EvalMetric.RAGAS:
            return self._score_ragas(q_tokens, a_tokens, c_tokens)

        return 0.0, {"error": f"unsupported metric: {metric}"}

    # -- Individual scorers -------------------------------------------------------

    @staticmethod
    def _score_answer_relevancy(
        q_tokens: list[str],
        a_tokens: list[str],
    ) -> tuple[float, dict[str, Any]]:
        """Answer relevancy: token overlap between question and answer.

        Uses Counter intersection — higher scores when the answer shares
        vocabulary with the question.  Returns 0.0 for empty inputs.
        """
        score = _counter_intersection_score(q_tokens, a_tokens)
        return score, {
            "q_token_count": len(q_tokens),
            "a_token_count": len(a_tokens),
            "overlap_tokens": sorted(set(q_tokens) & set(a_tokens)),
        }

    @staticmethod
    def _score_faithfulness(
        a_tokens: list[str],
        c_tokens: list[str],
    ) -> tuple[float, dict[str, Any]]:
        """Faithfulness: fraction of answer tokens present in context.

        A score of 1.0 means every answer token appears in the context.
        Returns 0.0 when no context is provided (no evidence to support).
        Vacuously returns 1.0 for an empty answer.
        """
        if not a_tokens:
            return 1.0, {"reason": "empty answer — vacuously faithful"}
        if not c_tokens:
            return 0.0, {"reason": "no context provided"}
        c_set = set(c_tokens)
        supported = sum(1 for t in a_tokens if t in c_set)
        score = supported / len(a_tokens)
        return score, {
            "a_token_count": len(a_tokens),
            "supported_count": supported,
            "unsupported_count": len(a_tokens) - supported,
        }

    @staticmethod
    def _score_hallucination(
        a_tokens: list[str],
        c_tokens: list[str],
    ) -> tuple[float, dict[str, Any]]:
        """Hallucination score (inverted): 1 - novelty ratio.

        *Novelty ratio* = fraction of answer tokens *not* found in context.
        The returned score is inverted so higher = less hallucination (better).

        Returns 0.0 when no context is available (every token is novel).
        Vacuously returns 1.0 for an empty answer.
        """
        if not a_tokens:
            return 1.0, {"reason": "empty answer — no hallucination possible"}
        if not c_tokens:
            return 0.0, {"reason": "no context — every token is novel"}
        c_set = set(c_tokens)
        novel = sum(1 for t in a_tokens if t not in c_set)
        novelty_ratio = novel / len(a_tokens)
        score = max(0.0, 1.0 - novelty_ratio)
        return score, {
            "a_token_count": len(a_tokens),
            "novel_count": novel,
            "novelty_ratio": round(novelty_ratio, 4),
        }

    @staticmethod
    def _score_context_precision(
        a_tokens: list[str],
        c_tokens: list[str],
    ) -> tuple[float, dict[str, Any]]:
        """Context precision: fraction of context tokens relevant to the answer.

        Measures how much of the provided context is actually leveraged in
        the answer.  Returns 0.0 when no context is provided.
        """
        if not c_tokens:
            return 0.0, {"reason": "no context provided"}
        if not a_tokens:
            return 1.0, {"reason": "empty answer — precision is vacuous"}
        a_set = set(a_tokens)
        relevant = sum(1 for t in c_tokens if t in a_set)
        score = relevant / len(c_tokens)
        return score, {
            "c_token_count": len(c_tokens),
            "relevant_count": relevant,
        }

    @staticmethod
    def _score_ragas(
        q_tokens: list[str],
        a_tokens: list[str],
        c_tokens: list[str],
    ) -> tuple[float, dict[str, Any]]:
        """RAGAS composite: arithmetic mean of per-component scores.

        When context is available the composite includes:
          - answer_relevancy
          - faithfulness
          - context_precision
          - hallucination_free (inverted hallucination)

        Falls back to answer_relevancy only when no context is provided.
        """
        relevancy = _counter_intersection_score(q_tokens, a_tokens)

        if not c_tokens:
            return relevancy, {
                "component_scores": {"answer_relevancy": round(relevancy, 4)},
                "note": "context unavailable — based on answer relevancy only",
            }

        a_set = set(a_tokens)
        c_set = set(c_tokens)

        # Faithfulness: fraction of answer tokens found in context.
        faithfulness = (
            sum(1 for t in a_tokens if t in c_set) / len(a_tokens)
            if a_tokens
            else 1.0
        )

        # Context precision: fraction of context tokens relevant to answer.
        context_precision = (
            sum(1 for t in c_tokens if t in a_set) / len(c_tokens)
            if c_tokens
            else 0.0
        )

        # Hallucination-free (inverted novelty ratio).
        novelty_ratio = (
            sum(1 for t in a_tokens if t not in c_set) / len(a_tokens)
            if a_tokens
            else 0.0
        )
        hallucination_free = max(0.0, 1.0 - novelty_ratio)

        component_scores = {
            "answer_relevancy": round(relevancy, 4),
            "faithfulness": round(faithfulness, 4),
            "context_precision": round(context_precision, 4),
            "hallucination_free": round(hallucination_free, 4),
        }
        score = sum(component_scores.values()) / len(component_scores)
        return score, {"component_scores": component_scores}
