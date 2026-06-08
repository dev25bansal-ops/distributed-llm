"""Smart model routing based on task complexity estimation.

Routes requests to the optimal model based on query complexity:
- Simple queries (greetings, yes/no) → small model (1B)
- Medium queries (explanations, summaries) → medium model (8B)
- Complex queries (analysis, code, math) → large model (70B)
- Code queries → code-specialized model

Integrates with ModelRouter for rule-based routing and adds
automatic complexity estimation on top.

Usage::

    router = SmartModelRouter(
        small_model="llama-3.2-1b",
        medium_model="llama-3.1-8b",
        large_model="llama-3.1-70b",
        code_model="codellama-34b",
    )
    model = router.route("What is 2+2?")  # → small_model
    model = router.route("Write a distributed sorting algorithm")  # → code_model
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from loguru import logger


class TaskComplexity(str, Enum):
    """Estimated task complexity levels."""
    TRIVIAL = "trivial"    # Greetings, yes/no, simple facts
    SIMPLE = "simple"      # Short explanations, translations
    MEDIUM = "medium"      # Summaries, moderate analysis
    COMPLEX = "complex"    # Long analysis, multi-step reasoning
    EXPERT = "expert"      # Code, math, research, creative writing


@dataclass
class ComplexityEstimate:
    """Result of complexity estimation."""
    level: TaskComplexity
    score: float          # 0.0-1.0 complexity score
    confidence: float     # 0.0-1.0 confidence in estimate
    signals: dict[str, float]  # Individual signal scores
    recommended_model: str


@dataclass
class ModelTier:
    """A model tier for a complexity level."""
    name: str
    model: str
    max_complexity: float  # Upper bound for this tier
    cost_per_token: float  # Relative cost


class TaskComplexityEstimator:
    """Estimates the complexity of a query for model routing.

    Uses multiple signals:
    - Query length (longer = more complex)
    - Vocabulary complexity (technical terms, code patterns)
    - Question type (factual vs analytical)
    - Instruction depth (multi-step, chain-of-thought)
    - Domain indicators (code, math, science)
    """

    # Simple patterns (low complexity)
    _SIMPLE_PATTERNS = [
        re.compile(r"^(hi|hello|hey|thanks|yes|no|ok|sure)\s*[.!?]?\s*$", re.I),
        re.compile(r"^(what is|who is|when did|where is)\s+\w+\??$", re.I),
        re.compile(r"^(translate|convert|summarize)\s+", re.I),
    ]

    # Complex patterns (high complexity)
    _COMPLEX_PATTERNS = [
        re.compile(r"(implement|design|architect|optimize|refactor)\s+(a|the|this)?\s*(system|algorithm|architecture|solution)", re.I),
        re.compile(r"(analyze|evaluate|compare|contrast)\s+(the|these|all)\s+", re.I),
        re.compile(r"(step[- ]by[- ]step|in detail|comprehensive|thorough)", re.I),
        re.compile(r"(edge case|trade-?off|pros and cons|advantages?.*disadvantages?)", re.I),
    ]

    # Code patterns
    _CODE_PATTERNS = [
        re.compile(r"(def |class |import |from |function |const |let |var |async )", re.I),
        re.compile(r"(write|create|implement|fix|debug)\s+(a|the|this)?\s*(function|class|method|script|api)", re.I),
        re.compile(r"```[a-z]*\n"),
        re.compile(r"(test|unit test|integration test|e2e test)", re.I),
    ]

    # Math patterns
    _MATH_PATTERNS = [
        re.compile(r"(calculate|compute|solve|prove|derive)\s+", re.I),
        re.compile(r"(integral|derivative|matrix|vector|equation|theorem)", re.I),
        re.compile(r"\d+\s*[+\-*/^]\s*\d+"),
    ]

    def estimate(self, text: str) -> ComplexityEstimate:
        """Estimate the complexity of a query.

        Args:
            text: The query text to analyze.

        Returns:
            ComplexityEstimate with level, score, and recommended model.
        """
        if not text or not text.strip():
            return ComplexityEstimate(
                level=TaskComplexity.TRIVIAL,
                score=0.0,
                confidence=0.9,
                signals={},
                recommended_model="",
            )

        signals = {}
        text_lower = text.lower().strip()

        # Signal 1: Query length
        word_count = len(text.split())
        length_score = min(word_count / 100, 1.0)
        signals["length"] = length_score

        # Signal 2: Simple pattern match
        simple_hits = sum(1 for p in self._SIMPLE_PATTERNS if p.search(text_lower))
        signals["simple_patterns"] = simple_hits / max(len(self._SIMPLE_PATTERNS), 1)

        # Signal 3: Complex pattern match
        complex_hits = sum(1 for p in self._COMPLEX_PATTERNS if p.search(text_lower))
        signals["complex_patterns"] = complex_hits / max(len(self._COMPLEX_PATTERNS), 1)

        # Signal 4: Code indicators
        code_hits = sum(1 for p in self._CODE_PATTERNS if p.search(text))
        signals["code"] = code_hits / max(len(self._CODE_PATTERNS), 1)

        # Signal 5: Math indicators
        math_hits = sum(1 for p in self._MATH_PATTERNS if p.search(text_lower))
        signals["math"] = math_hits / max(len(self._MATH_PATTERNS), 1)

        # Signal 6: Vocabulary complexity (technical terms)
        technical_terms = [
            "algorithm", "architecture", "implementation", "optimization",
            "distributed", "concurrent", "asynchronous", "parallel",
            "neural", "transformer", "attention", "gradient",
            "kubernetes", "docker", "microservice", "infrastructure",
            "authentication", "authorization", "encryption", "security",
        ]
        tech_hits = sum(1 for term in technical_terms if term in text_lower)
        signals["technical_vocab"] = min(tech_hits / 5, 1.0)

        # Signal 7: Multi-step indicators
        multi_step = len(re.findall(r"\b(first|second|third|then|next|finally|step \d)\b", text_lower))
        signals["multi_step"] = min(multi_step / 3, 1.0)

        # Compute weighted complexity score
        score = (
            length_score * 0.15 +
            signals["complex_patterns"] * 0.25 +
            signals["code"] * 0.20 +
            signals["math"] * 0.15 +
            signals["technical_vocab"] * 0.10 +
            signals["multi_step"] * 0.10 +
            (1.0 - signals["simple_patterns"]) * 0.05
        )

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))

        # Determine complexity level
        if score < 0.15:
            level = TaskComplexity.TRIVIAL
        elif score < 0.35:
            level = TaskComplexity.SIMPLE
        elif score < 0.55:
            level = TaskComplexity.MEDIUM
        elif score < 0.75:
            level = TaskComplexity.COMPLEX
        else:
            level = TaskComplexity.EXPERT

        # Confidence based on signal strength
        signal_strength = sum(abs(v - 0.5) for v in signals.values()) / max(len(signals), 1)
        confidence = min(0.5 + signal_strength, 1.0)

        return ComplexityEstimate(
            level=level,
            score=round(score, 3),
            confidence=round(confidence, 3),
            signals={k: round(v, 3) for k, v in signals.items()},
            recommended_model="",
        )


class SmartModelRouter:
    """Routes queries to the optimal model based on task complexity.

    Combines complexity estimation with model tier selection to
    automatically route simple queries to small models and complex
    queries to large models.

    Args:
        small_model: Model for trivial/simple queries (1B-3B).
        medium_model: Model for medium queries (7B-8B).
        large_model: Model for complex queries (70B+).
        code_model: Model for code queries (optional).
        math_model: Model for math queries (optional).
    """

    def __init__(
        self,
        small_model: str = "llama-3.2-1b",
        medium_model: str = "llama-3.1-8b",
        large_model: str = "llama-3.1-70b",
        code_model: str = "",
        math_model: str = "",
        base_router: Any = None,
    ):
        self._estimator = TaskComplexityEstimator()
        self._base_router = base_router  # Optional ModelRouter for rule-based fallback
        self._tiers = [
            ModelTier("small", small_model, 0.35, 0.1),
            ModelTier("medium", medium_model, 0.55, 0.5),
            ModelTier("large", large_model, 1.0, 5.0),
        ]
        self._code_model = code_model or large_model
        self._math_model = math_model or large_model
        self._default_model = medium_model
        self._stats = {
            "total_routes": 0,
            "routes_by_tier": {"small": 0, "medium": 0, "large": 0},
            "routes_by_type": {"code": 0, "math": 0, "general": 0},
            "avg_complexity": 0.0,
        }
        self._lock = threading.Lock()

    def route(self, text: str) -> tuple[str, ComplexityEstimate]:
        """Route a query to the optimal model.

        Args:
            text: The query text.

        Returns:
            Tuple of (model_name, complexity_estimate).
        """
        estimate = self._estimator.estimate(text)

        # Low-confidence → fall through to base_router for rule-based routing
        if estimate.confidence < 0.4 and self._base_router is not None:
            base_result = self._base_router.resolve(text)
            if base_result and base_result != self._default_model:
                model = base_result
                route_type = "rule_based"
                estimate.recommended_model = model
                with self._lock:
                    self._stats["total_routes"] += 1
                    self._stats["routes_by_type"]["rule_based"] = (
                        self._stats["routes_by_type"].get("rule_based", 0) + 1
                    )
                return model, estimate

        # Specialized routing
        if estimate.signals.get("code", 0) > 0.3:
            model = self._code_model
            route_type = "code"
        elif estimate.signals.get("math", 0) > 0.3:
            model = self._math_model
            route_type = "math"
        else:
            # Tier-based routing
            model = self._default_model
            route_type = "general"
            for tier in self._tiers:
                if estimate.score <= tier.max_complexity:
                    model = tier.model
                    break

        estimate.recommended_model = model

        with self._lock:
            self._stats["total_routes"] += 1
            self._stats["routes_by_type"][route_type] = (
                self._stats["routes_by_type"].get(route_type, 0) + 1
            )
            # Update tier stats
            for tier in self._tiers:
                if model == tier.model:
                    self._stats["routes_by_tier"][tier.name] = (
                        self._stats["routes_by_tier"].get(tier.name, 0) + 1
                    )
                    break

            # Running average complexity
            n = self._stats["total_routes"]
            self._stats["avg_complexity"] = (
                self._stats["avg_complexity"] * (n - 1) + estimate.score
            ) / n

        return model, estimate

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)
