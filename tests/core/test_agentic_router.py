"""Tests for AgenticRouter: RoutingDecision, RouterJudge fallback,
RouterJudge._parse_response, and AgenticRouter routing.

All tests run without GPU or ML dependencies using fallback paths.
"""

import json
from unittest.mock import patch

import pytest

from distllm.core.agentic_router import (
    AgenticRouter,
    RouterJudge,
    RoutingDecision,
)


@pytest.fixture(autouse=True)
def _no_router_model(monkeypatch):
    """Ensure DISTLLM_ROUTER_MODEL is unset so no ML imports are attempted."""
    monkeypatch.delenv("DISTLLM_ROUTER_MODEL", raising=False)


# ── 1. RoutingDecision dataclass ────────────────────────────────────────────


class TestRoutingDecision:
    """RoutingDecision dataclass: defaults, custom values, to_dict()."""

    def test_defaults(self):
        """Default values for optional fields."""
        decision = RoutingDecision(model="test-model")

        assert decision.model == "test-model"
        assert decision.reason == ""
        assert decision.confidence == 0.5
        assert decision.suggested_quantization == ""
        assert decision.estimated_latency_ms == 0.0

    def test_custom_values(self):
        """All fields set explicitly."""
        decision = RoutingDecision(
            model="codellama-7b",
            reason="strong code model",
            confidence=0.85,
            suggested_quantization="int4",
            estimated_latency_ms=150.0,
        )

        assert decision.model == "codellama-7b"
        assert decision.reason == "strong code model"
        assert decision.confidence == 0.85
        assert decision.suggested_quantization == "int4"
        assert decision.estimated_latency_ms == 150.0

    def test_to_dict(self):
        """Serialization to dictionary."""
        decision = RoutingDecision(
            model="llama-3-8b",
            reason="general purpose",
            confidence=0.75,
            suggested_quantization="fp16",
            estimated_latency_ms=200.0,
        )

        d = decision.to_dict()

        assert d == {
            "model": "llama-3-8b",
            "reason": "general purpose",
            "confidence": 0.75,
            "suggested_quantization": "fp16",
            "estimated_latency_ms": 200.0,
        }


# ── 2. RouterJudge without model (fallback path) ────────────────────────────


class TestRouterJudgeFallback:
    """RouterJudge without model loaded — always uses fallback path."""

    def test_is_loaded_false_without_model(self):
        """RouterJudge with no model path reports not loaded."""
        judge = RouterJudge()

        assert judge.is_loaded is False
        assert judge._model is None
        assert judge._tokenizer is None

    def test_decide_returns_fallback_decision(self):
        """decide() with no model falls back to first available model."""
        judge = RouterJudge()

        decision = judge.decide(
            "Write a Python function",
            [{"name": "modelA", "quantization": "int4"}],
        )

        assert isinstance(decision, RoutingDecision)
        assert decision.model == "modelA"
        assert "fallback" in decision.reason.lower()
        assert decision.confidence == 0.3


# ── 3. RouterJudge._parse_response ──────────────────────────────────────────


class TestRouterJudgeParseResponse:
    """RouterJudge._parse_response JSON parsing and fallback."""

    def test_parse_valid_json_full(self):
        """Full valid JSON returns correctly populated RoutingDecision."""
        judge = RouterJudge()
        models = [{"name": "modelA"}, {"name": "modelB"}]
        response = json.dumps({
            "model": "modelA",
            "reason": "best for code",
            "confidence": 0.9,
            "suggested_quantization": "int8",
            "estimated_latency_ms": 100.0,
        })

        decision = judge._parse_response(response, models)

        assert decision.model == "modelA"
        assert decision.reason == "best for code"
        assert decision.confidence == 0.9
        assert decision.suggested_quantization == "int8"
        assert decision.estimated_latency_ms == 100.0

    def test_parse_minimal_json(self):
        """Minimal JSON uses defaults for missing fields."""
        judge = RouterJudge()
        models = [{"name": "modelA"}]

        decision = judge._parse_response('{"model": "modelA"}', models)

        assert decision.model == "modelA"
        assert decision.reason == ""
        assert decision.confidence == 0.5
        assert decision.suggested_quantization == ""
        assert decision.estimated_latency_ms == 0.0

    def test_parse_json_with_surrounding_text(self):
        """JSON embedded in model output text is extracted."""
        judge = RouterJudge()
        models = [{"name": "modelA"}]
        response = (
            "I think the best model is...\n\n"
            '{"model": "modelA", "reason": "good match"}\n\n'
            "Hope that helps."
        )

        decision = judge._parse_response(response, models)

        assert decision.model == "modelA"
        assert decision.reason == "good match"

    def test_parse_model_not_in_available_list(self):
        """Unknown model name falls back to first available model name."""
        judge = RouterJudge()
        models = [{"name": "modelA"}, {"name": "modelB"}]

        decision = judge._parse_response(
            '{"model": "unknown-model", "reason": "my choice"}', models,
        )

        assert decision.model == "modelA"
        assert decision.reason == "my choice"

    def test_parse_empty_available_models(self):
        """No available models: model name kept as-is from JSON."""
        judge = RouterJudge()

        decision = judge._parse_response(
            '{"model": "custom-model", "reason": "best"}', [],
        )

        assert decision.model == "custom-model"

    def test_parse_complex_nested_json(self):
        """Extra top-level fields in JSON are ignored, required ones extracted."""
        judge = RouterJudge()
        models = [{"name": "modelA"}]
        response = json.dumps({
            "model": "modelA",
            "reason": "fastest option",
            "confidence": 0.82,
            "suggested_quantization": "int4",
            "estimated_latency_ms": 75.0,
            "extra_field": "should be ignored",
            "nested": {"foo": 1},
        })

        decision = judge._parse_response(response, models)

        assert decision.model == "modelA"
        assert decision.reason == "fastest option"
        assert decision.confidence == 0.82
        assert decision.suggested_quantization == "int4"
        assert decision.estimated_latency_ms == 75.0

    def test_parse_string_confidence_converted(self):
        """Numeric fields as strings in JSON are converted to float."""
        judge = RouterJudge()
        models = [{"name": "modelA"}]
        response = json.dumps({
            "model": "modelA",
            "confidence": "0.95",
            "estimated_latency_ms": "200",
        })

        decision = judge._parse_response(response, models)

        assert decision.confidence == 0.95
        assert decision.estimated_latency_ms == 200.0

    # -- Fallback paths (invalid / no JSON) --

    @patch.object(RouterJudge, "_fallback")
    def test_parse_invalid_json_calls_fallback(self, mock_fallback):
        """Completely non-JSON response triggers _fallback."""
        mock_fallback.return_value = RoutingDecision(
            model="modelA", reason="fallback from parse", confidence=0.3,
        )
        judge = RouterJudge()

        decision = judge._parse_response("not json at all", [{"name": "modelA"}])

        mock_fallback.assert_called_once()
        assert decision.model == "modelA"

    @patch.object(RouterJudge, "_fallback")
    def test_parse_no_json_braces_calls_fallback(self, mock_fallback):
        """Response with no '{' or '}' triggers _fallback."""
        mock_fallback.return_value = RoutingDecision(
            model="modelA", reason="no json found", confidence=0.3,
        )
        judge = RouterJudge()

        decision = judge._parse_response(
            "just some text without braces", [{"name": "modelA"}],
        )

        mock_fallback.assert_called_once()
        assert decision.model == "modelA"

    @patch.object(RouterJudge, "_fallback")
    def test_parse_malformed_json_calls_fallback(self, mock_fallback):
        """Malformed JSON inside braces triggers _fallback."""
        mock_fallback.return_value = RoutingDecision(
            model="modelA", reason="malformed", confidence=0.3,
        )
        judge = RouterJudge()

        decision = judge._parse_response("{bad json here}", [{"name": "modelA"}])

        mock_fallback.assert_called_once()
        assert decision.model == "modelA"

    @patch.object(RouterJudge, "_fallback")
    def test_parse_unmatched_braces_still_parses(self, mock_fallback):
        """Extra closing brace before JSON is handled by _parse_response."""
        mock_fallback.return_value = RoutingDecision(
            model="fallback", reason="fallback triggered", confidence=0.3,
        )
        judge = RouterJudge()
        models = [{"name": "modelA"}]
        # rfind('}') finds the last '}', start finds the first '{'
        response = '} extra leading brace not valid {"model": "modelA", "reason": "works"}'

        decision = judge._parse_response(response, models)

        mock_fallback.assert_not_called()
        assert decision.model == "modelA"

    @patch.object(RouterJudge, "_fallback")
    def test_parse_empty_string_calls_fallback(self, mock_fallback):
        """Empty response string triggers _fallback."""
        mock_fallback.return_value = RoutingDecision(
            model="modelA", reason="empty response", confidence=0.3,
        )
        judge = RouterJudge()

        decision = judge._parse_response("", [{"name": "modelA"}])

        mock_fallback.assert_called_once()
        assert decision.model == "modelA"


# ── 4 & 5. AgenticRouter without model ──────────────────────────────────────


class TestAgenticRouter:
    """AgenticRouter without loaded judge — tests routing, stats, outcomes."""

    def test_route_returns_routing_decision(self):
        """route() always returns a RoutingDecision."""
        models = [{"name": "modelA", "quantization": "int4"}]
        router = AgenticRouter(available_models=models)

        decision = router.route("hello")

        assert isinstance(decision, RoutingDecision)
        assert decision.model  # non-empty

    def test_stats_contains_expected_keys(self):
        """stats property exposes expected metrics."""
        router = AgenticRouter(available_models=[{"name": "m1"}])
        router.route("test")

        stats = router.stats

        assert set(stats.keys()) >= {
            "total_routes",
            "judge_calls",
            "fallback_calls",
            "judge_loaded",
            "preferences_collected",
        }
        assert stats["total_routes"] == 1
        assert stats["judge_loaded"] is False

    def test_record_outcome_appends_preference(self):
        """record_outcome stores a preference without error."""
        router = AgenticRouter(available_models=[{"name": "m1"}])
        decision = RoutingDecision(model="m1", reason="test")

        router.record_outcome(
            decision, user_rating=0.9, latency_ms=100.0, cost_usd=0.01,
        )

        assert router.stats["preferences_collected"] == 1

    def test_record_outcome_without_optional_args(self):
        """record_outcome works when only decision is provided."""
        router = AgenticRouter(available_models=[{"name": "m1"}])
        decision = RoutingDecision(model="m1", reason="test")

        router.record_outcome(decision)

        assert router.stats["preferences_collected"] == 1

    def test_multiple_routes_update_stats_correctly(self):
        """Multiple route calls accurately increment counters."""
        router = AgenticRouter(available_models=[{"name": "m1"}])

        router.route("first")
        router.route("second")
        router.route("third")

        stats = router.stats
        assert stats["total_routes"] == 3
        assert stats["fallback_calls"] == 3  # All fall through to heuristic
        assert stats["judge_calls"] == 0

    def test_heuristic_fallback_returns_valid_decision(self):
        """Without a loaded judge, route() uses heuristic fallback."""
        models = [{"name": "modelA"}, {"name": "modelB"}]
        router = AgenticRouter(available_models=models)

        decision = router.route("translate hello to french")

        assert isinstance(decision, RoutingDecision)
        assert decision.model == "modelA"  # first available model
        assert decision.reason == "first available"
        assert decision.confidence == 0.3

    def test_heuristic_fallback_no_models(self):
        """Heuristic fallback with empty model list returns empty model."""
        router = AgenticRouter(available_models=[])

        decision = router.route("test")

        assert isinstance(decision, RoutingDecision)
        assert decision.model == ""
        assert decision.reason == "no models"
        assert decision.confidence == 0.0
