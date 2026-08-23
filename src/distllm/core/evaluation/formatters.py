"""Prompt formatters for LLM evaluation benchmarks.

Extracted from :mod:`distllm.core.evaluation_harness`.
"""

from __future__ import annotations

import abc
import json

from distllm.core.evaluation.models import EvalSample


class PromptFormatter(abc.ABC):
    """Abstract base for formatting prompts for model evaluation."""

    @abc.abstractmethod
    def format(self, sample: EvalSample) -> str:
        """Format a sample into a model prompt string."""
        ...


class _HeimPromptFormatter(PromptFormatter):
    """Prompt formatter for HEIM-style benchmarks (MMLU, GSM8K, HumanEval)."""

    def __init__(self, benchmark: str) -> None:
        self._benchmark = benchmark

    def format(self, sample: EvalSample) -> str:
        if self._benchmark == "mmlu":
            return (
                f"Answer the following multiple-choice question:\n\n"
                f"{sample.question}\n\n"
                f"Answer:"
            )
        if self._benchmark == "gsm8k":
            return (
                f"Solve the following math problem step by step:\n\n"
                f"{sample.question}\n\n"
                f"Answer:"
            )
        if self._benchmark == "humaneval":
            return (
                f"Write a Python function for the following task. "
                f"Return only the code, no explanation.\n\n"
                f"{sample.question}\n\n"
                f"```python"
            )
        return sample.question


class _MTBenchPromptFormatter(PromptFormatter):
    """Formats MT-Bench multi-turn conversations."""

    def format(self, sample: EvalSample) -> str:
        data = json.loads(sample.question)
        turns = data.get("turns", [])
        category = data.get("category", "general")
        prompt = f"Category: {category}\n\n"
        for i, turn in enumerate(turns):
            prompt += f"Turn {i + 1}: {turn}\n"
        prompt += "\nRespond to the conversation above."
        return prompt


class _ArenaPromptFormatter(PromptFormatter):
    """Formats prompts for pairwise comparison."""

    def format(self, sample: EvalSample) -> str:
        return (
            f"Please respond to the following prompt:\n\n"
            f"{sample.question}\n\n"
            f"Response:"
        )


__all__ = [
    "PromptFormatter",
    "_HeimPromptFormatter",
    "_MTBenchPromptFormatter",
    "_ArenaPromptFormatter",
]
