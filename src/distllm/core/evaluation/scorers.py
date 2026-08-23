"""Scorer classes for the LLM Evaluation Harness.

Extracted from :mod:`distllm.core.evaluation_harness`.
"""

from __future__ import annotations

import abc
import json
import os
import re

import httpx
from loguru import logger

from distllm.core.evaluation.constants import (
    _ARENA_SYSTEM_PROMPT,
    _MTBENCH_SYSTEM_PROMPT,
    _SecretStr,
)
from distllm.core.evaluation.models import EvalSample


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


class Scorer(abc.ABC):
    """Abstract base for scoring model predictions."""

    @abc.abstractmethod
    def score(self, sample: EvalSample, prediction: str) -> float:
        """Return a score between 0.0 and 1.0."""
        ...


class _ExactMatchScorer(Scorer):
    """Exact match scorer for deterministic benchmarks.

    For math (GSM8K), extracts the numeric answer from the prediction
    and compares it to the reference. For general QA, uses substring
    matching on key phrases.
    """

    def __init__(self, benchmark: str) -> None:
        self._benchmark = benchmark

    def score(self, sample: EvalSample, prediction: str) -> float:
        if not sample.answer:
            return 0.0

        reference = sample.answer.strip().lower()

        if self._benchmark == "gsm8k":
            # Extract the final numeric answer (last number in prediction)
            numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", prediction.replace(",", ""))
            if numbers:
                # Try exact match, then compare numerically
                pred_num = numbers[-1].replace(",", "")
                ref_num = reference.replace(",", "")
                try:
                    return 1.0 if float(pred_num) == float(ref_num) else 0.0
                except ValueError:
                    pass
            # Fallback: look for the reference in the prediction
            return 1.0 if reference in prediction.lower() else 0.0

        if self._benchmark == "humaneval":
            # Check if the prediction contains the reference function body
            ref_lines = [l.strip().lower() for l in sample.answer.split("\n") if l.strip()]
            pred_lower = prediction.lower()
            matches = sum(1 for line in ref_lines if line in pred_lower)
            return matches / max(len(ref_lines), 1)

        # Default: exact match or substring containment
        return 1.0 if reference in prediction.lower() else 0.0


class _MTBenchScorer(Scorer):
    """MT-Bench scorer that uses an LLM-as-judge via API.

    Supports any OpenAI-compatible API endpoint (including self-hosted
    judges, Claude, local models).

    Falls back to heuristic scoring if the judge API is unavailable.

    Args:
        api_key: API key for the judge model. Supports ``_SecretStr`` for masking.
        judge_model: Model name (default "gpt-4", supports "gpt-4o", "claude-3-opus", etc.).
        api_base: OpenAI-compatible base URL. Defaults to ``https://api.openai.com/v1``.
            Override for self-hosted judges.
    """

    def __init__(self, api_key: str = "", judge_model: str = "gpt-4",
                 api_base: str = "https://api.openai.com/v1") -> None:
        raw_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._api_key = _SecretStr(raw_key) if raw_key else _SecretStr("")
        self._judge_model = judge_model
        self._api_base = api_base.rstrip("/")

    def score(self, sample: EvalSample, prediction: str) -> float:
        if self._api_key:
            try:
                return self._judge_via_api(sample, prediction)
            except Exception as exc:
                logger.warning("GPT-4 judge API call failed, using heuristic: {}", exc)
        return self._heuristic_score(sample, prediction)

    def _judge_via_api(self, sample: EvalSample, prediction: str) -> float:
        """Score via LLM-as-judge API.

        Uses ``_api_base`` for the endpoint and ``_judge_model`` for the model,
        making it compatible with any OpenAI-API proxy or self-hosted judge.
        """
        data = json.loads(sample.question)
        category = data.get("category", "general")
        turns = data.get("turns", [])

        conversation = ""
        for i, turn in enumerate(turns):
            conversation += f"User: {turn}\n"
        conversation += f"\nAssistant: {prediction[:2000]}"

        messages = [
            {"role": "system", "content": _MTBENCH_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Category: {category}\n\n"
                    f"Conversation:\n{conversation}\n\n"
                    f"Please rate this response on a scale of 1 to 10. "
                    f"Return only a number."
                ),
            },
        ]

        resp = httpx.post(
            f"{self._api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key.get()}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._judge_model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 10,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        result = resp.json()
        resp_content = result["choices"][0]["message"]["content"].strip()

        # Parse numeric score (1-10)
        match = re.search(r"(\d+)", resp_content)
        if match:
            score = max(1, min(10, int(match.group(1))))
            return score / 10.0

        return 0.5

    def _heuristic_score(self, sample: EvalSample, prediction: str) -> float:
        """Heuristic fallback: length-based and keyword coverage."""
        if not prediction.strip():
            return 0.0
        length_score = min(1.0, len(prediction) / 500.0)
        # Category-specific keyword presence
        data = json.loads(sample.question)
        category = data.get("category", "general")
        keywords = {
            "coding": ["def ", "return", "import", "class ", "function"],
            "math": ["=", "+", "-", "*", "/", "solve", "equation"],
            "reasoning": ["because", "therefore", "if", "then", "thus"],
            "writing": ["however", "furthermore", "moreover", "consequently"],
            "extraction": ["{", "}", '"', "[", "]"],
        }
        kws = keywords.get(category, [])
        kw_score = sum(1 for kw in kws if kw.lower() in prediction.lower())
        kw_score = min(1.0, kw_score / max(len(kws), 1))
        return round(length_score * 0.4 + kw_score * 0.6, 4)


class _ArenaScorer(Scorer):
    """Scorer for Chatbot Arena pairwise comparisons.

    Uses LLM-as-judge to determine which response is better.
    Supports any OpenAI-compatible API endpoint.

    Falls back to heuristic (length-based) scoring.
    """

    def __init__(self, api_key: str = "", judge_model: str = "gpt-4",
                 api_base: str = "https://api.openai.com/v1") -> None:
        raw_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._api_key = _SecretStr(raw_key) if raw_key else _SecretStr("")
        self._judge_model = judge_model
        self._api_base = api_base.rstrip("/")

    def score(self, sample: EvalSample, prediction: str) -> float:
        """Score is 1.0 if model_a wins, 0.0 if model_b wins, 0.5 if tie.

        For Arena, ``prediction`` is expected to be the concatenated
        responses from both models in the format:
        ``"MODEL_A: ...\n---\nMODEL_B: ..."``
        """
        return self._compare_via_api(sample, prediction)

    def _compare_via_api(self, sample: EvalSample, combined: str) -> float:
        """Use LLM judge to pick the better response."""
        if not self._api_key.get():
            return self._heuristic_compare(sample, combined)

        # Split combined into model_a and model_b responses
        parts = combined.split("\n---\n")
        response_a = parts[0] if len(parts) > 0 else ""
        response_b = parts[1] if len(parts) > 1 else ""

        messages = [
            {"role": "system", "content": _ARENA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Prompt: {sample.question}\n\n"
                    f"Response A:\n{response_a[:2000]}\n\n"
                    f"Response B:\n{response_b[:2000]}\n\n"
                    f"Which response is better? Reply with 'A', 'B', or 'Tie'."
                ),
            },
        ]

        try:
            resp = httpx.post(
                f"{self._api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key.get()}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._judge_model,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 10,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            result = resp.json()
            verdict = result["choices"][0]["message"]["content"].strip().upper()

            if "A" in verdict and "B" not in verdict:
                return 1.0  # model_a wins
            if "B" in verdict and "A" not in verdict:
                return 0.0  # model_b wins
            return 0.5  # tie
        except Exception as exc:
            logger.warning("Arena judge API call failed, using heuristic: {}", exc)
            return self._heuristic_compare(sample, combined)

    def _heuristic_compare(self, sample: EvalSample, combined: str) -> float:
        """Fallback: longer response wins."""
        parts = combined.split("\n---\n")
        len_a = len(parts[0]) if len(parts) > 0 else 0
        len_b = len(parts[1]) if len(parts) > 1 else 0
        if len_a > len_b * 1.2:
            return 1.0
        if len_b > len_a * 1.2:
            return 0.0
        return 0.5


__all__ = [
    "Scorer",
    "_ExactMatchScorer",
    "_MTBenchScorer",
    "_ArenaScorer",
]
