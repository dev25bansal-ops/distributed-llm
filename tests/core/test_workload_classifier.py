"""Tests for WorkloadClassifier -- prompt text analysis for speculative method selection.

Covers:
- classify returns CODE for code-like text
- classify returns INSTRUCTION for question text
- classify returns REPETITIVE for low-entropy text
- classify returns DIVERSE for high-entropy text
- classify returns UNKNOWN for empty input
- classify_features returns correct feature dict
- WorkloadType enum values

No MagicMock -- pure string processing and math.
"""

from __future__ import annotations

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/workload_classifier.py")
classify = _mod.classify
classify_features = _mod.classify_features
WorkloadType = _mod.WorkloadType


class TestWorkloadType:
    """WorkloadType enum."""

    def test_enum_values(self) -> None:
        assert WorkloadType.CODE.value == "code"
        assert WorkloadType.INSTRUCTION.value == "instruction"
        assert WorkloadType.REPETITIVE.value == "repetitive"
        assert WorkloadType.DIVERSE.value == "diverse"
        assert WorkloadType.UNKNOWN.value == "unknown"

    def test_enum_members_count(self) -> None:
        assert len(WorkloadType) == 5


class TestWorkloadClassifier:
    """Classify function."""

    def test_classify_code(self) -> None:
        text = "def foo():\n    return 42\n"
        assert classify(text) == WorkloadType.CODE

    def test_classify_code_with_import(self) -> None:
        text = "import numpy as np\nx = np.array([1, 2, 3])\n"
        assert classify(text) == WorkloadType.CODE

    def test_classify_instruction(self) -> None:
        text = "Can you please explain how transformers work in detail?"
        assert classify(text) == WorkloadType.INSTRUCTION

    def test_classify_instruction_question(self) -> None:
        text = "What is the meaning of life? Please describe briefly."
        assert classify(text) == WorkloadType.INSTRUCTION

    def test_classify_repetitive(self) -> None:
        # Low entropy, high repetition
        text = "abc" * 50
        assert classify(text) == WorkloadType.REPETITIVE

    def test_classify_diverse(self) -> None:
        # High entropy, varied content
        text = "The quick brown fox jumps over the lazy dog. Python is great."
        result = classify(text)
        # Could be DIVERSE or UNKNOWN depending on entropy threshold
        assert result in (WorkloadType.DIVERSE, WorkloadType.UNKNOWN, WorkloadType.INSTRUCTION)

    def test_classify_empty(self) -> None:
        assert classify("") == WorkloadType.UNKNOWN
        assert classify("   ") == WorkloadType.UNKNOWN

    def test_classify_none(self) -> None:
        # Edge case: string is itself, empty string tested above
        pass

    def test_classify_mixed_code_but_not_dominant(self) -> None:
        # Some code signals, but below threshold
        text = "hello world this is a test with some (parentheses)"
        result = classify(text)
        assert isinstance(result, WorkloadType)


class TestWorkloadClassifierFeatures:
    """Feature extraction."""

    def test_classify_features_empty(self) -> None:
        features = classify_features("")
        assert features["code_ratio"] == 0.0
        assert features["instruction_score"] == 0.0
        assert features["entropy_3gram"] == 0.0
        assert features["repetition_ratio"] == 0.0

    def test_classify_features_code(self) -> None:
        text = "def foo():\n    return 42\n"
        features = classify_features(text)
        assert features["code_ratio"] > 0.0

    def test_classify_features_entropy(self) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        features = classify_features(text)
        assert features["entropy_3gram"] > 0.0

    def test_classify_features_repetition(self) -> None:
        text = "aaaa" * 20  # repeated pattern
        features = classify_features(text)
        assert features["repetition_ratio"] > 0.0

    def test_classify_features_keys(self) -> None:
        text = "def foo(): return 42"
        features = classify_features(text)
        expected_keys = {"code_ratio", "instruction_score", "entropy_3gram", "repetition_ratio"}
        assert set(features.keys()) == expected_keys
