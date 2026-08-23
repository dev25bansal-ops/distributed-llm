"""Cost tracking middleware for DistLLM integrations.

Tracks token usage, GPU time, and estimated cost across one or more
LLM calls.  Works with any integration that reports usage in
``ChatResult.llm_output["token_usage"]``.

Usage::

    from distllm_langchain import DistLLMChat, CostTracker

    tracker = CostTracker()
    llm = DistLLMChat(model="llama-70b", cost_tracker=tracker)

    # ... run chains / agents ...

    print(tracker.summary())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CallRecord:
    """Single inference call record."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


class CostTracker:
    """Accumulate token usage and cost estimates across LLM calls.

    Parameters
    ----------
    cost_per_1k_prompt_tokens : float
        Dollar cost per 1 000 prompt tokens (default: $0.0002).
    cost_per_1k_completion_tokens : float
        Dollar cost per 1 000 completion tokens (default: $0.0006).
    gpu_cost_per_hour : float
        Dollar cost per GPU-hour (default: $2.50 for A100-class).
    """

    def __init__(
        self,
        cost_per_1k_prompt_tokens: float = 0.0002,
        cost_per_1k_completion_tokens: float = 0.0006,
        gpu_cost_per_hour: float = 2.50,
    ):
        self.cost_per_1k_prompt_tokens = cost_per_1k_prompt_tokens
        self.cost_per_1k_completion_tokens = cost_per_1k_completion_tokens
        self.gpu_cost_per_hour = gpu_cost_per_hour
        self._records: list[CallRecord] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, llm_output: dict[str, Any], model: str = "") -> None:
        """Record a single call's usage from ``ChatResult.llm_output``."""
        usage = llm_output.get("token_usage", {})
        latency = llm_output.get("distllm_latency_ms", 0.0)
        self._records.append(
            CallRecord(
                model=model or llm_output.get("model_name", "unknown"),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                latency_ms=latency,
            )
        )

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    @property
    def total_calls(self) -> int:
        return len(self._records)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self._records)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self._records)

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self._records)

    @property
    def total_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self._records)

    @property
    def total_gpu_seconds(self) -> float:
        """Approximate GPU time in seconds (latency-based)."""
        return self.total_latency_ms / 1000.0

    @property
    def estimated_cost(self) -> float:
        """Estimated dollar cost (token-based + GPU-time-based)."""
        token_cost = (
            self.total_prompt_tokens / 1000 * self.cost_per_1k_prompt_tokens
            + self.total_completion_tokens / 1000 * self.cost_per_1k_completion_tokens
        )
        gpu_cost = self.total_gpu_seconds / 3600 * self.gpu_cost_per_hour
        return round(token_cost + gpu_cost, 6)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a summary dict suitable for logging or dashboards."""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "total_gpu_seconds": round(self.total_gpu_seconds, 3),
            "estimated_cost_usd": self.estimated_cost,
        }

    def reset(self) -> None:
        """Clear all recorded calls."""
        self._records.clear()

    def __repr__(self) -> str:
        return (
            f"CostTracker(calls={self.total_calls}, "
            f"tokens={self.total_tokens}, "
            f"cost=${self.estimated_cost:.6f})"
        )
