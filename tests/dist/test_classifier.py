"""Tests for distllm.dist.scheduling.classifier module.

Covers the full public API: WorkloadType, classify, classify_features.
Deterministic -- no GPU, no network, no timing-dependent assertions.
"""

from __future__ import annotations

import pytest

from distllm.dist.scheduling.classifier import (
    WorkloadType,
    classify,
    classify_features,
)


# ── WorkloadType Enum ────────────────────────────────────────────────────────


class TestWorkloadType:
    """WorkloadType enum values, string representation, and membership."""

    def test_values(self) -> None:
        assert WorkloadType.CODE.value == "code"
        assert WorkloadType.REPETITIVE.value == "repetitive"
        assert WorkloadType.DIVERSE.value == "diverse"
        assert WorkloadType.INSTRUCTION.value == "instruction"
        assert WorkloadType.UNKNOWN.value == "unknown"

    def test_is_str_enum(self) -> None:
        assert issubclass(WorkloadType, str)

    def test_all_members(self) -> None:
        names = {e.name for e in WorkloadType}
        assert names == {"CODE", "REPETITIVE", "DIVERSE", "INSTRUCTION", "UNKNOWN"}

    def test_from_value(self) -> None:
        assert WorkloadType("code") is WorkloadType.CODE
        assert WorkloadType("repetitive") is WorkloadType.REPETITIVE

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            WorkloadType("bogus")


# ── classify() ───────────────────────────────────────────────────────────────


class TestClassify:
    """classify(text) returns a WorkloadType based on content heuristics."""

    # -- Empty / boundary inputs --

    def test_empty_string(self) -> None:
        assert classify("") is WorkloadType.UNKNOWN

    def test_none_input(self) -> None:
        assert classify(None) is WorkloadType.UNKNOWN  # type: ignore[arg-type]

    def test_whitespace_only(self) -> None:
        """Whitespace with no word tokens -> entropy 0 -> REPETITIVE."""
        assert classify("   \n\n   ") is WorkloadType.REPETITIVE

    def test_punctuation_only(self) -> None:
        """No word tokens, entropy 0 -> REPETITIVE."""
        assert classify("!? !? !? !? !? !?") is WorkloadType.REPETITIVE

    def test_single_word(self) -> None:
        assert classify("hello") is WorkloadType.REPETITIVE

    def test_two_words(self) -> None:
        assert classify("hello world") is WorkloadType.REPETITIVE

    # -- CODE classification --

    def test_code_with_keywords(self) -> None:
        text = "def foo():\n    return 42\n"
        assert classify(text) is WorkloadType.CODE

    def test_code_with_markers(self) -> None:
        text = "```python\nprint('hello')\n```"
        assert classify(text) is WorkloadType.CODE

    def test_code_with_line_endings(self) -> None:
        text = "int x = 1;\nchar* s = \"hello\";"
        assert classify(text) is WorkloadType.CODE

    def test_code_with_hash_comment(self) -> None:
        text = "# this is a comment\nx = 1"
        assert classify(text) is WorkloadType.CODE

    def test_code_with_slash_comment(self) -> None:
        text = "// this is a comment\nint x = 1;"
        assert classify(text) is WorkloadType.CODE

    def test_code_high_keyword_density(self) -> None:
        """Text with many code keywords on a single line."""
        text = "def class import return if for while"
        assert classify(text) is WorkloadType.CODE

    # -- INSTRUCTION classification --

    def test_instruction_please(self) -> None:
        text = "please tell me about transformers"
        assert classify(text) is WorkloadType.INSTRUCTION

    def test_instruction_can_you(self) -> None:
        text = "can you explain gradient descent"
        assert classify(text) is WorkloadType.INSTRUCTION

    def test_instruction_how_to_high_density(self) -> None:
        """Triggers the second threshold (keyword subset > 0.5)."""
        text = "please can you how to what is"
        assert classify(text) is WorkloadType.INSTRUCTION

    def test_instruction_write_explain(self) -> None:
        text = "please write a story about dragons"
        assert classify(text) is WorkloadType.INSTRUCTION

    # -- REPETITIVE classification --

    def test_repetitive_high_ratio(self) -> None:
        """Repeated 3-grams cross the 0.25 threshold."""
        text = "the cat and the cat and the cat and the cat and"
        assert classify(text) is WorkloadType.REPETITIVE

    def test_repetitive_all_same_word(self) -> None:
        text = "a a a a a a"
        assert classify(text) is WorkloadType.REPETITIVE

    # -- DIVERSE classification --

    def test_diverse_high_entropy(self) -> None:
        """Many unique tokens produce 3-gram entropy > 4.0."""
        text = (
            "apple banana cherry date elderberry fig grape "
            "honeydew jackfruit kiwi lemon mango nectarine "
            "orange papaya quince raspberry strawberry "
            "tangerine vanilla"
        )
        result = classify(text)
        assert result is WorkloadType.DIVERSE, f"Expected DIVERSE, got {result}"

    # -- UNKNOWN classification --

    def test_unknown_moderate_entropy(self) -> None:
        """Moderate entropy (1.5-4.0) with no repetition -> UNKNOWN."""
        text = "the quick brown fox jumps over the lazy dog"
        assert classify(text) is WorkloadType.UNKNOWN

    def test_unknown_six_unique_words(self) -> None:
        """Six unique short words: moderate entropy, no repetition."""
        text = "alpha beta gamma delta epsilon zeta"
        assert classify(text) is WorkloadType.UNKNOWN

    # -- Priority ordering --

    def test_code_beats_instruction(self) -> None:
        text = "please def foo(): return 42"
        assert classify(text) is WorkloadType.CODE

    def test_code_beats_repetitive(self) -> None:
        text = "def a\ndef a\ndef a\ndef a\ndef a\ndef a"
        assert classify(text) is WorkloadType.CODE

    def test_instruction_beats_repetitive(self) -> None:
        text = "please write please write please write please write"
        assert classify(text) is WorkloadType.INSTRUCTION

    # -- Unicode --

    def test_unicode_text(self) -> None:
        """Unicode word chars work; entropy < 1.5 -> REPETITIVE."""
        text = "cafe deja cafe deja cafe deja"
        result = classify(text)
        assert result is WorkloadType.REPETITIVE

    def test_unicode_code_markers(self) -> None:
        text = "```python\nprint('cafe')\n```"
        assert classify(text) is WorkloadType.CODE


# ── classify_features() ─────────────────────────────────────────────────────


class TestClassifyFeatures:
    """classify_features(text) returns a dict of numeric features."""

    def test_empty_string(self) -> None:
        features = classify_features("")
        assert features["word_count"] == 0.0
        assert features["code_ratio"] == 0.0
        assert features["entropy_3gram"] == 0.0

    def test_none_input(self) -> None:
        features = classify_features(None)  # type: ignore[arg-type]
        assert features["word_count"] == 0.0

    def test_returns_all_expected_keys(self) -> None:
        expected = {
            "entropy_3gram",
            "entropy_2gram",
            "repetition_3gram",
            "repetition_2gram",
            "code_ratio",
            "code_keyword_density",
            "instruction_density",
            "word_count",
            "avg_word_length",
        }
        features = classify_features("some input text here")
        assert set(features.keys()) == expected

    def test_all_values_are_floats_or_int(self) -> None:
        features = classify_features("def foo(): return 42")
        for key, val in features.items():
            assert isinstance(val, (float, int)), f"{key} is {type(val).__name__}"
        # word_count is specifically an int from the source
        assert isinstance(features["word_count"], int)

    def test_word_count(self) -> None:
        features = classify_features("one two three four")
        assert features["word_count"] == 4.0

    def test_avg_word_length(self) -> None:
        features = classify_features("a bb ccc dddd")
        assert features["avg_word_length"] == pytest.approx(2.5)

    def test_code_ratio_for_code(self) -> None:
        features = classify_features("def foo():\n    pass")
        assert features["code_ratio"] > 0.3

    def test_code_keyword_density_positive(self) -> None:
        features = classify_features("def class import return if for while async")
        assert features["code_keyword_density"] > 0.2

    def test_instruction_density_positive(self) -> None:
        text = "please can you write a summary"
        features = classify_features(text)
        assert features["instruction_density"] > 0.1

    def test_entropy_2gram_geq_3gram(self) -> None:
        """2-gram entropy is always >= 3-gram entropy for the same input."""
        features = classify_features("the cat in the hat sat on the mat")
        assert features["entropy_2gram"] >= features["entropy_3gram"]

    def test_repetition_ratio_in_unit_interval(self) -> None:
        features = classify_features("a a a a a a a a a a")
        assert 0.0 <= features["repetition_3gram"] <= 1.0

    def test_diverse_text_high_entropy(self) -> None:
        text = (
            "apple banana cherry date elderberry fig grape "
            "honeydew jackfruit kiwi lemon mango nectarine "
            "orange papaya quince raspberry strawberry "
            "tangerine vanilla"
        )
        features = classify_features(text)
        assert features["entropy_3gram"] > 4.0

    def test_repetitive_text_high_repetition(self) -> None:
        text = "repeat this phrase over and over repeat this phrase over and over"
        features = classify_features(text)
        assert features["repetition_3gram"] > 0.0

    def test_whitespace_only_features(self) -> None:
        features = classify_features("   \n   ")
        assert features["word_count"] == 0.0
        assert features["avg_word_length"] == 0.0

    def test_code_text_zero_instruction_density(self) -> None:
        features = classify_features("def foo():\n    return 42")
        assert features["instruction_density"] == 0.0
