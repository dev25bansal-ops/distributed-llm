"""Automatic model selection based on user requirements.

Matches user requirements (task, quality, speed, cost) to available
models and recommends the best fit.

Usage::

    selector = ModelSelector(available_models)
    recommendation = selector.select(
        task="code generation",
        quality="high",
        max_latency_ms=500,
    )
    print(recommendation.model_name)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class ModelProfile:
    """Profile of a model's capabilities."""
    name: str
    parameter_count: str = ""  # e.g., "7B", "70B"
    tasks: list[str] = field(default_factory=list)  # e.g., ["code", "chat", "analysis"]
    quality_score: float = 0.0  # 0-1, relative quality
    speed_score: float = 0.0  # 0-1, relative speed
    cost_per_1m_tokens: float = 0.0
    context_length: int = 4096
    supports_tools: bool = False
    supports_vision: bool = False
    quantized: bool = False


@dataclass
class ModelRecommendation:
    """A model recommendation with reasoning."""
    model_name: str
    confidence: float  # 0-1
    reasoning: str
    alternatives: list[str] = field(default_factory=list)


# Default model profiles for common models
DEFAULT_PROFILES = [
    ModelProfile(
        name="meta-llama/Llama-3-70B-Instruct",
        parameter_count="70B",
        tasks=["chat", "code", "analysis", "creative"],
        quality_score=0.95,
        speed_score=0.3,
        cost_per_1m_tokens=0.88,
        context_length=8192,
        supports_tools=True,
    ),
    ModelProfile(
        name="meta-llama/Llama-3-8B-Instruct",
        parameter_count="8B",
        tasks=["chat", "code"],
        quality_score=0.7,
        speed_score=0.9,
        cost_per_1m_tokens=0.20,
        context_length=8192,
        supports_tools=True,
    ),
    ModelProfile(
        name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        parameter_count="47B",
        tasks=["chat", "code", "analysis"],
        quality_score=0.85,
        speed_score=0.5,
        cost_per_1m_tokens=0.60,
        context_length=32768,
        supports_tools=True,
    ),
    ModelProfile(
        name="Qwen/Qwen2-72B-Instruct",
        parameter_count="72B",
        tasks=["chat", "code", "analysis", "math"],
        quality_score=0.93,
        speed_score=0.3,
        cost_per_1m_tokens=0.90,
        context_length=32768,
        supports_tools=True,
    ),
    ModelProfile(
        name="deepseek-ai/DeepSeek-Coder-V2-Instruct",
        parameter_count="236B",
        tasks=["code", "analysis"],
        quality_score=0.97,
        speed_score=0.2,
        cost_per_1m_tokens=1.20,
        context_length=128000,
        supports_tools=True,
    ),
    ModelProfile(
        name="codellama/CodeLlama-34b-Instruct-hf",
        parameter_count="34B",
        tasks=["code"],
        quality_score=0.8,
        speed_score=0.6,
        cost_per_1m_tokens=0.50,
        context_length=16384,
    ),
]


class ModelSelector:
    """Selects the best model for a given set of requirements."""

    def __init__(self, profiles: list[ModelProfile] | None = None):
        self._profiles = profiles or DEFAULT_PROFILES
        self._custom_profiles: list[ModelProfile] = []

    def add_profile(self, profile: ModelProfile) -> None:
        """Add a custom model profile."""
        self._custom_profiles.append(profile)

    def select(
        self,
        task: str = "",
        quality: str = "balanced",
        max_latency_ms: float = 0,
        max_cost_per_1m: float = 0,
        context_length: int = 0,
        requires_tools: bool = False,
        requires_vision: bool = False,
        available_models: list[str] | None = None,
    ) -> ModelRecommendation:
        """Select the best model for the given requirements.

        Args:
            task: Task type (code, chat, analysis, creative, math).
            quality: Quality preference (high, balanced, fast).
            max_latency_ms: Maximum acceptable latency (0 = no limit).
            max_cost_per_1m: Maximum cost per 1M tokens (0 = no limit).
            context_length: Required context length.
            requires_tools: Whether tool calling is required.
            requires_vision: Whether vision support is required.
            available_models: Filter to only these models.

        Returns:
            ModelRecommendation with the best model and reasoning.
        """
        candidates = self._profiles + self._custom_profiles

        # Filter by availability
        if available_models:
            candidates = [p for p in candidates if p.name in available_models]

        # Filter by hard requirements
        if requires_tools:
            candidates = [p for p in candidates if p.supports_tools]
        if requires_vision:
            candidates = [p for p in candidates if p.supports_vision]
        if context_length > 0:
            candidates = [p for p in candidates if p.context_length >= context_length]
        if max_cost_per_1m > 0:
            candidates = [p for p in candidates if p.cost_per_1m_tokens <= max_cost_per_1m]

        if not candidates:
            return ModelRecommendation(
                model_name="",
                confidence=0.0,
                reasoning="No models match the requirements",
            )

        # Score candidates
        scored = []
        for profile in candidates:
            score = self._score_candidate(profile, task, quality, max_latency_ms)
            scored.append((score, profile))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best = scored[0]

        # Build reasoning
        reasons = []
        if task and task in best.tasks:
            reasons.append(f"optimized for {task}")
        if quality == "high":
            reasons.append(f"high quality (score={best.quality_score:.2f})")
        elif quality == "fast":
            reasons.append(f"fast inference (speed={best.speed_score:.2f})")
        if best.quantized:
            reasons.append("quantized for efficiency")

        alternatives = [p.name for _, p in scored[1:3]]

        return ModelRecommendation(
            model_name=best.name,
            confidence=min(best_score, 1.0),
            reasoning=", ".join(reasons) if reasons else "best overall match",
            alternatives=alternatives,
        )

    def _score_candidate(
        self,
        profile: ModelProfile,
        task: str,
        quality: str,
        max_latency_ms: float,
    ) -> float:
        """Score a candidate model for the given requirements."""
        score = 0.5  # Base score

        # Task match
        if task and task in profile.tasks:
            score += 0.2
        elif task:
            score -= 0.1

        # Quality preference
        if quality == "high":
            score += profile.quality_score * 0.3
        elif quality == "fast":
            score += profile.speed_score * 0.3
        else:  # balanced
            score += (profile.quality_score + profile.speed_score) * 0.15

        # Cost efficiency (lower is better)
        if profile.cost_per_1m_tokens > 0:
            cost_score = max(0, 1.0 - profile.cost_per_1m_tokens / 2.0)
            score += cost_score * 0.1

        return score

    def list_models(self) -> list[dict]:
        """List all available model profiles."""
        return [
            {
                "name": p.name,
                "parameters": p.parameter_count,
                "tasks": p.tasks,
                "quality": p.quality_score,
                "speed": p.speed_score,
                "cost_per_1m": p.cost_per_1m_tokens,
            }
            for p in self._profiles + self._custom_profiles
        ]
