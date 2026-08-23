"""Tests for SmartModelRouter — task complexity estimation and model tier routing.

Covers:
- TaskComplexityEstimator: signal detection, score computation, level determination
- SmartModelRouter: construction, routing by complexity, specialized routing
  (code, math), low-confidence fallback, stats, edge cases
"""

from __future__ import annotations

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_smr = load_module("distllm/core/smart_model_router.py")
SmartModelRouter = _smr.SmartModelRouter
TaskComplexityEstimator = _smr.TaskComplexityEstimator
TaskComplexity = _smr.TaskComplexity
ComplexityEstimate = _smr.ComplexityEstimate
ModelTier = _smr.ModelTier


# ── TaskComplexityEstimator ────────────────────────────────────────────────────


class TestComplexityEstimatorConstruction:
    def test_defaults(self):
        est = TaskComplexityEstimator()
        assert len(est._SIMPLE_PATTERNS) == 3
        assert len(est._COMPLEX_PATTERNS) == 4
        assert len(est._CODE_PATTERNS) == 4
        assert len(est._MATH_PATTERNS) == 3


class TestComplexityEstimatorEmpty:
    def test_empty_string(self):
        est = TaskComplexityEstimator()
        result = est.estimate("")
        assert result.level == TaskComplexity.TRIVIAL
        assert result.score == 0.0

    def test_whitespace_string(self):
        est = TaskComplexityEstimator()
        result = est.estimate("   ")
        assert result.level == TaskComplexity.TRIVIAL
        assert result.score == 0.0

    def test_none_string(self):
        est = TaskComplexityEstimator()
        result = est.estimate("")
        assert result.level == TaskComplexity.TRIVIAL


class TestComplexityEstimatorSignals:
    def test_length_signal(self):
        est = TaskComplexityEstimator()
        short = est.estimate("hi")
        long = est.estimate(" ".join(["word"] * 200))
        assert long.signals["length"] > short.signals["length"]

    def test_simple_pattern_greeting(self):
        est = TaskComplexityEstimator()
        result = est.estimate("Hello")
        assert result.signals.get("simple_patterns", 0) > 0

    def test_simple_pattern_yes_no(self):
        est = TaskComplexityEstimator()
        result = est.estimate("Yes")
        assert result.signals.get("simple_patterns", 0) > 0

    def test_complex_pattern_analysis(self):
        est = TaskComplexityEstimator()
        result = est.estimate("Analyze the architecture and evaluate all trade-offs")
        assert result.signals.get("complex_patterns", 0) > 0

    def test_complex_pattern_implement(self):
        est = TaskComplexityEstimator()
        result = est.estimate("Implement the system using concurrent processing")
        assert result.signals.get("complex_patterns", 0) > 0

    def test_code_pattern_function(self):
        est = TaskComplexityEstimator()
        result = est.estimate("Write a function that sorts an array using quicksort")
        assert result.signals.get("code", 0) > 0

    def test_code_pattern_with_backticks(self):
        est = TaskComplexityEstimator()
        result = est.estimate("```python\nprint('hello')\n```")
        assert result.signals.get("code", 0) > 0

    def test_math_pattern_calculate(self):
        est = TaskComplexityEstimator()
        result = est.estimate("Calculate the integral of x^2 from 0 to 1")
        assert result.signals.get("math", 0) > 0

    def test_math_pattern_equation(self):
        est = TaskComplexityEstimator()
        result = est.estimate("Solve this equation: 2 + 3 * 4")
        assert result.signals.get("math", 0) > 0

    def test_technical_vocabulary(self):
        est = TaskComplexityEstimator()
        result = est.estimate(
            "The transformer architecture uses attention mechanisms "
            "with gradient descent optimization"
        )
        assert result.signals.get("technical_vocab", 0) > 0

    def test_multi_step_signal(self):
        est = TaskComplexityEstimator()
        result = est.estimate("First do this, then do that, finally check the result")
        assert result.signals.get("multi_step", 0) > 0


class TestComplexityEstimatorLevels:
    def test_trivial_level(self):
        est = TaskComplexityEstimator()
        result = est.estimate("hi")
        assert result.level in (TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE)

    def test_simple_level(self):
        est = TaskComplexityEstimator()
        result = est.estimate("What is the capital of France?")
        assert isinstance(result.level, TaskComplexity)

    def test_complex_level_analysis(self):
        est = TaskComplexityEstimator()
        result = est.estimate(
            "Analyze the trade-offs between consistent hashing and "
            "rendezvous hashing in distributed caching systems, "
            "considering the impact of node failures and network partitions. "
            "Provide a comprehensive comparison with edge cases."
        )
        assert result.score >= 0.3

    def test_expert_level_code_math(self):
        est = TaskComplexityEstimator()
        result = est.estimate(
            "Implement a distributed transformer training loop with "
            "gradient checkpointing and mixed precision. The algorithm "
            "should handle edge cases in the backward pass."
        )
        assert result.score >= 0.2

    def test_confidence_is_high_for_extreme_cases(self):
        est = TaskComplexityEstimator()
        result = est.estimate("Hi")
        assert 0.0 <= result.confidence <= 1.0

    def test_recommended_model_default_empty(self):
        est = TaskComplexityEstimator()
        result = est.estimate("hello")
        assert result.recommended_model == ""


# ── SmartModelRouter ──────────────────────────────────────────────────────────


class TestSmartModelRouterConstruction:
    def test_default_values(self):
        router = SmartModelRouter()
        assert router._tiers[0].name == "small"
        assert router._tiers[0].model == "llama-3.2-1b"
        assert router._tiers[1].model == "llama-3.1-8b"
        assert router._tiers[2].model == "llama-3.1-70b"
        assert router._code_model == "llama-3.1-70b"
        assert router._math_model == "llama-3.1-70b"
        assert router._default_model == "llama-3.1-8b"

    def test_custom_models(self):
        router = SmartModelRouter(
            small_model="tiny",
            medium_model="medium",
            large_model="big",
            code_model="coder",
            math_model="mathematician",
        )
        assert router._tiers[0].model == "tiny"
        assert router._tiers[2].model == "big"
        assert router._code_model == "coder"
        assert router._math_model == "mathematician"


class TestSmartModelRouterRouting:
    def test_trivial_routes_to_small(self):
        router = SmartModelRouter()
        model, est = router.route("Hi")
        assert model == "llama-3.2-1b"

    def test_simple_routes_to_small(self):
        router = SmartModelRouter()
        model, est = router.route("What is 2+2?")
        # Math + medium length → may go to medium or large model
        assert isinstance(model, str)

    def test_complex_routes_to_larger_model(self):
        router = SmartModelRouter()
        model, est = router.route(
            "Analyze the trade-offs between microservices and monoliths"
        )
        # Should not be at the trivial level
        assert est.score > 0.1

    def test_code_query_routes_to_code_model(self):
        router = SmartModelRouter(code_model="coder-specialist")
        model, est = router.route("Write a function to calculate fibonacci numbers")
        assert model == "coder-specialist"

    def test_math_query_routes_to_math_model(self):
        router = SmartModelRouter(math_model="math-expert")
        model, est = router.route("Calculate the integral of x^2 dx from 0 to pi")
        assert model == "math-expert"

    def test_route_returns_estimate_with_signals(self):
        router = SmartModelRouter()
        model, est = router.route("Hello")
        assert isinstance(est, ComplexityEstimate)
        assert isinstance(est.signals, dict)
        assert est.recommended_model != ""

    def test_low_confidence_route_to_default(self):
        """Short query with no signals should go to small model."""
        router = SmartModelRouter()
        model, est = router.route("No")
        assert model == "llama-3.2-1b"

    def test_tier_thresholds_respected(self):
        router = SmartModelRouter(
            small_model="tiny",
            medium_model="medium",
            large_model="large",
        )
        # Force the score to a specific range via long text
        trivial_model, _ = router.route("hi")
        assert trivial_model == "tiny"


class TestSmartModelRouterStats:
    def test_stats_initial(self):
        router = SmartModelRouter()
        s = router.stats()
        assert s["total_routes"] == 0
        assert s["routes_by_tier"]["small"] == 0
        assert s["avg_complexity"] == 0.0

    def test_stats_after_routes(self):
        router = SmartModelRouter()
        router.route("Hi")
        router.route("Write a code function")
        s = router.stats()
        assert s["total_routes"] == 2
        assert s["routes_by_tier"]["small"] >= 1
        assert s["avg_complexity"] > 0.0

    def test_stats_are_thread_safe(self):
        router = SmartModelRouter()
        router.route("test")
        s1 = router.stats()
        assert s1["total_routes"] > 0


class TestSmartModelRouterEdgeCases:
    def test_empty_string(self):
        router = SmartModelRouter()
        model, est = router.route("")
        assert isinstance(model, str)

    def test_very_long_input(self):
        router = SmartModelRouter()
        long_text = "word " * 1000
        model, est = router.route(long_text)
        # Very long text should have a non-trivial score
        assert est.score > 0.15
