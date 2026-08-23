"""Tests for EvalStatus, EvalSample, EvalResult, and data model classes.

Minimal no-mock tests for the evaluation harness dataclasses and enums.
"""

from __future__ import annotations

import uuid

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/evaluation_harness.py")
EvalStatus = _mod.EvalStatus
EvalSample = _mod.EvalSample
EvalResult = _mod.EvalResult
EvalReport = _mod.EvalReport
EvalBenchmark = _mod.EvalBenchmark


class TestEvalStatus:
    """EvalStatus enum."""

    def test_values(self) -> None:
        assert EvalStatus.PENDING.value == "pending"
        assert EvalStatus.RUNNING.value == "running"
        assert EvalStatus.COMPLETED.value == "completed"
        assert EvalStatus.FAILED.value == "failed"

    def test_enum_from_string(self) -> None:
        assert EvalStatus("pending") == EvalStatus.PENDING
        assert EvalStatus("completed") == EvalStatus.COMPLETED


class TestEvalBenchmark:
    """EvalBenchmark enum."""

    def test_values(self) -> None:
        assert EvalBenchmark.MMLU.value == "mmlu"
        assert EvalBenchmark.GSM8K.value == "gsm8k"
        assert EvalBenchmark.HUMANEVAL.value == "humaneval"
        assert EvalBenchmark.MT_BENCH.value == "mt_bench"
        assert EvalBenchmark.ARENA.value == "arena"


class TestEvalSample:
    """EvalSample frozen dataclass construction."""

    def test_creation_with_required_fields(self) -> None:
        sample = EvalSample(question="What is 2+2?")
        assert sample.question == "What is 2+2?"
        assert sample.answer is None
        assert sample.category == "general"
        assert sample.metadata == {}

    def test_creation_with_all_fields(self) -> None:
        sample = EvalSample(
            question="What is the capital of France?",
            answer="Paris",
            category="geography",
            metadata={"source": "test", "index": 0},
        )
        assert sample.answer == "Paris"
        assert sample.category == "geography"
        assert sample.metadata["source"] == "test"

    def test_immutability(self) -> None:
        sample = EvalSample(question="test")
        try:
            sample.question = "changed"
            assert False, "frozen dataclass should not allow mutation"
        except Exception:
            pass


class TestEvalResult:
    """EvalResult dataclass construction."""

    def test_creation_with_sample_only(self) -> None:
        sample = EvalSample(question="Q1")
        result = EvalResult(sample=sample, prediction="A1")
        assert result.sample is sample
        assert result.prediction == "A1"
        assert result.score == 0.0
        assert result.latency_ms == 0.0
        assert result.prompt_tokens == 0
        assert result.generated_tokens == 0
        assert result.error is None

    def test_creation_with_all_fields(self) -> None:
        sample = EvalSample(question="Q", answer="A")
        result = EvalResult(
            sample=sample, prediction="pred", score=0.85,
            latency_ms=150.0, prompt_tokens=50, generated_tokens=100,
            error=None,
        )
        assert result.score == 0.85
        assert result.latency_ms == 150.0
        assert result.prompt_tokens == 50

    def test_with_error(self) -> None:
        sample = EvalSample(question="Q")
        result = EvalResult(sample=sample, prediction="", error="timeout")
        assert result.error == "timeout"


class TestEvalReport:
    """EvalReport dataclass with auto-generated fields."""

    def test_auto_generates_report_id_and_created_at(self) -> None:
        report = EvalReport(
            model_id="test-model",
            dataset="test",
            config={},
            metrics={},
        )
        assert report.report_id != ""
        # Validate UUID format
        uuid.UUID(report.report_id)
        assert report.created_at > 0

    def test_custom_report_id_preserved(self) -> None:
        report = EvalReport(
            model_id="m", dataset="d", config={}, metrics={},
            report_id="custom-id-123",
        )
        assert report.report_id == "custom-id-123"

    def test_status_defaults_to_pending(self) -> None:
        report = EvalReport(model_id="m", dataset="d", config={}, metrics={})
        assert report.status == EvalStatus.PENDING
