"""Constants and enums for the LLM Evaluation Harness.

Extracted from :mod:`distllm.core.evaluation_harness`.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path


class _SecretStr:
    """Minimal secret string wrapper that redacts in repr/str to prevent key leaks."""
    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "_SecretStr(***)"

    def __str__(self) -> str:
        return "***"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = Path.home() / ".distllm" / "eval_results.db"
_MAX_WORKERS = min(8, (os.cpu_count() or 4))
_EVAL_TIMEOUT_S = 120.0
_MTBENCH_CATEGORIES = [
    "writing",
    "roleplay",
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
]
_ARENA_SYSTEM_PROMPT = (
    "You are an impartial judge comparing two AI assistant responses. "
    "Evaluate which response is more helpful, accurate, and safe."
)
_MTBENCH_SYSTEM_PROMPT = (
    "You are an impartial judge evaluating the quality of an AI assistant's response. "
    "Score the response on a scale of 1 to 10 based on helpfulness, accuracy, and relevance."
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EvalBenchmark(str, Enum):
    """Supported evaluation benchmarks."""
    MMLU = "mmlu"
    GSM8K = "gsm8k"
    HUMANEVAL = "humaneval"
    MT_BENCH = "mt_bench"
    ARENA = "arena"


class EvalStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


__all__ = [
    "_SecretStr",
    "_DEFAULT_DB_PATH",
    "_MAX_WORKERS",
    "_EVAL_TIMEOUT_S",
    "_MTBENCH_CATEGORIES",
    "_ARENA_SYSTEM_PROMPT",
    "_MTBENCH_SYSTEM_PROMPT",
    "EvalBenchmark",
    "EvalStatus",
]
