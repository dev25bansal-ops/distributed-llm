"""Data models for the LLM Evaluation Harness.

Extracted from :mod:`distllm.core.evaluation_harness`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from distllm.core.evaluation.constants import EvalStatus


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalSample:
    """A single evaluation sample (question + reference + metadata)."""
    question: str
    answer: str | None = None  # reference answer
    category: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of evaluating a single sample."""
    sample: EvalSample
    prediction: str
    score: float = 0.0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    error: str | None = None


@dataclass
class EvalReport:
    """Aggregated evaluation report."""
    model_id: str
    dataset: str
    config: dict[str, Any]
    metrics: dict[str, float]
    results: list[EvalResult] = field(default_factory=list)
    status: EvalStatus = EvalStatus.PENDING
    report_id: str = ""
    created_at: float = 0.0
    duration_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.report_id:
            self.report_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = time.time()


__all__ = [
    "EvalSample",
    "EvalResult",
    "EvalReport",
]
