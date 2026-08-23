"""Graceful Degradation: when overloaded, return partial responses instead of 503.

Monitors system load metrics (queue depth, latency, memory usage) and
automatically degrades response quality under pressure:
  - Reduces max_new_tokens
  - Falls back to smaller model
  - Returns cached/stale response
  - Truncates context window
  - Returns partial response as fallback

Gives a 'degraded but alive' experience instead of hard 503 errors.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "GracefulDegradation",
    "DegradationLevel",
    "DegradationPlan",
    "LoadSnapshot",
]


class DegradationLevel(Enum):
    """Progressive degradation levels."""
    NONE = 0          # Full quality
    LIGHT = 1         # Reduce max_tokens
    MODERATE = 2       # Smaller model, lower quality
    SEVERE = 3         # Stale/cached responses only
    CRITICAL = 4       # Partial/truncated responses


@dataclass
class LoadSnapshot:
    """Current system load metrics."""
    queue_depth: int = 0
    avg_latency_ms: float = 0.0
    memory_util_pct: float = 0.0
    request_rate: float = 0.0  # requests/sec
    active_requests: int = 0

    def score(self) -> float:
        """Compute a composite load score (0=idle, 1=overloaded)."""
        q = min(self.queue_depth / 50, 1.0) * 0.3
        l = min(self.avg_latency_ms / 5000, 1.0) * 0.3
        m = min(self.memory_util_pct / 90, 1.0) * 0.2
        r = min(self.request_rate / 100, 1.0) * 0.2
        return q + l + m + r


@dataclass
class DegradationPlan:
    """What actions to take to degrade gracefully."""
    level: DegradationLevel = DegradationLevel.NONE
    max_tokens: int | None = None          # Reduced token limit
    model_override: str | None = None       # Smaller model
    use_stale: bool = False                 # Return cached response
    truncate_prompt: int | None = None      # Truncated context length
    partial_ok: bool = False                # Allow partial response

    def apply_to_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return a new params dict with degradation adjustments applied.

        Does not mutate the input dict (immutability compliance).
        """
        result = dict(params)  # Copy to avoid in-place mutation
        if self.max_tokens is not None:
            result["max_new_tokens"] = min(
                result.get("max_new_tokens", 2048), self.max_tokens
            )
        if self.truncate_prompt is not None and "prompt" in result:
            prompt = result["prompt"]
            if isinstance(prompt, str) and "tokenizer" in result:
                tokenizer = result["tokenizer"]
                encoded = tokenizer.encode(prompt)
                if len(encoded) > self.truncate_prompt:
                    truncated = encoded[:self.truncate_prompt]
                    result["prompt"] = tokenizer.decode(truncated)
            elif isinstance(prompt, str):
                result["prompt"] = prompt[:self.truncate_prompt]
        return result


class GracefulDegradation:
    """Monitors load and provides degradation plans.

    Usage:
        gd = GracefulDegradation()
        plan = gd.evaluate(queue_depth=40, avg_latency_ms=3000)
        if plan.level != DegradationLevel.NONE:
            adjusted_params = plan.apply_to_params(params)
    """

    def __init__(
        self,
        enabled: bool = True,
        light_threshold: float = 0.3,
        moderate_threshold: float = 0.5,
        severe_threshold: float = 0.7,
        critical_threshold: float = 0.85,
        fallback_model: str | None = None,
        partial_response: str | None = None,
    ):
        self._enabled = enabled
        self._light_threshold = light_threshold
        self._moderate_threshold = moderate_threshold
        self._severe_threshold = severe_threshold
        self._critical_threshold = critical_threshold
        self._fallback_model = fallback_model
        self._partial_response = partial_response or ""

        self._history: list[tuple[float, float]] = []  # (timestamp, score)
        self._lock = threading.Lock()

    def evaluate(self, load: LoadSnapshot | None = None, **kwargs) -> DegradationPlan:
        """Evaluate load and return degradation plan."""
        if not self._enabled:
            return DegradationPlan()

        if load is None:
            load = LoadSnapshot(**kwargs)

        score = load.score()

        with self._lock:
            self._history.append((time.time(), score))
            if len(self._history) > 1000:
                self._history.pop(0)

        plan = DegradationPlan(level=DegradationLevel.NONE)

        if score >= self._critical_threshold:
            plan.level = DegradationLevel.CRITICAL
            plan.max_tokens = 64
            plan.use_stale = True
            plan.partial_ok = True
            plan.truncate_prompt = 512
        elif score >= self._severe_threshold:
            plan.level = DegradationLevel.SEVERE
            plan.max_tokens = 256
            plan.use_stale = True
            plan.truncate_prompt = 1024
        elif score >= self._moderate_threshold:
            plan.level = DegradationLevel.MODERATE
            plan.max_tokens = 512
            plan.model_override = self._fallback_model
        elif score >= self._light_threshold:
            plan.level = DegradationLevel.LIGHT
            plan.max_tokens = 1024

        return plan

    @property
    def current_level(self) -> DegradationLevel:
        """Return current degradation level based on recent load history."""
        with self._lock:
            if not self._history:
                return DegradationLevel.NONE
            recent = self._history[-5:]
            avg_score = sum(s for _, s in recent) / len(recent)
        return self._score_to_level(avg_score)

    def _score_to_level(self, score: float) -> DegradationLevel:
        if score >= self._critical_threshold:
            return DegradationLevel.CRITICAL
        if score >= self._severe_threshold:
            return DegradationLevel.SEVERE
        if score >= self._moderate_threshold:
            return DegradationLevel.MODERATE
        if score >= self._light_threshold:
            return DegradationLevel.LIGHT
        return DegradationLevel.NONE

    def get_partial_response(self, request_id: str, reason: str = "overloaded") -> dict:
        """Return partial response when full generation is not possible."""
        return {
            "request_id": request_id,
            "choices": [{"text": self._partial_response, "finish_reason": "degraded"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "degraded": True,
            "reason": reason,
        }

    def stats(self) -> dict[str, Any]:
        with self._lock:
            recent = self._history[-10:] if self._history else []
            avg_score = sum(s for _, s in recent) / max(len(recent), 1) if recent else 0.0
            level = self._score_to_level(avg_score)
            return {
                "enabled": self._enabled,
                "current_level": level.value,
                "current_score": avg_score,
                "history_size": len(self._history),
                "thresholds": {
                    "light": self._light_threshold,
                    "moderate": self._moderate_threshold,
                    "severe": self._severe_threshold,
                    "critical": self._critical_threshold,
                },
            }


# ── Recovery-aware degradation tiers ─────────────────────────────────────

RECOVERY_DEGRADATION_TIERS = [
    {
        "name": "full",
        "batch_size": 4,
        "precision": "fp16",
        "max_tokens": 2048,
        "description": "Normal operation",
    },
    {
        "name": "reduced",
        "batch_size": 2,
        "precision": "fp16",
        "max_tokens": 1024,
        "description": "Reduced batch after node loss",
    },
    {
        "name": "minimal",
        "batch_size": 1,
        "precision": "int8",
        "max_tokens": 512,
        "description": "Minimal operation with degraded precision",
    },
    {
        "name": "emergency",
        "batch_size": 1,
        "precision": "int4",
        "max_tokens": 256,
        "description": "Emergency mode — minimal quality",
    },
]


def get_recovery_tier(nodes_remaining: int, nodes_total: int) -> dict:
    """Return the appropriate degradation tier based on surviving node ratio.

    Args:
        nodes_remaining: Number of healthy nodes remaining.
        nodes_total: Original total number of nodes.

    Returns:
        Tier dict with batch_size, precision, max_tokens.
    """
    if nodes_total <= 0:
        return RECOVERY_DEGRADATION_TIERS[0]

    ratio = nodes_remaining / nodes_total
    if ratio > 0.75:
        return RECOVERY_DEGRADATION_TIERS[0]  # full
    elif ratio > 0.5:
        return RECOVERY_DEGRADATION_TIERS[1]  # reduced
    elif ratio > 0.25:
        return RECOVERY_DEGRADATION_TIERS[2]  # minimal
    else:
        return RECOVERY_DEGRADATION_TIERS[3]  # emergency
