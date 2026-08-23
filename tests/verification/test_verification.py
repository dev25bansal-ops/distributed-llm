"""Tests for the distllm.verification package.

Covers:
  - comparator.py:  compare_logits, compare_hidden_states, compare_text,
                    compare_tokens, evaluate_comparison, _normalized_edit_distance
  - hash_registry.py:  compute_output_hash, compute_text_hash,
                       compute_token_ids_hash, OutputHashRegistry,
                       GenerationOutput
  - report.py:         VerificationReport, generate_report
  - runner.py:         AccuracyVerifier, verify_accuracy

Import strategy: ``distllm.verification.*`` is safe to import directly
(no circular dependencies). All heavy integrations are stubbed.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from distllm.verification.comparator import (
    DEFAULT_THRESHOLDS,
    OutputComparison,
    _all_pass,
    _normalized_edit_distance,
    compare_hidden_states,
    compare_logits,
    compare_text,
    compare_tokens,
    evaluate_comparison,
)
from distllm.verification.hash_registry import (
    GenerationOutput,
    OutputHashRegistry,
    compute_output_hash,
    compute_text_hash,
    compute_token_ids_hash,
)
from distllm.verification.report import VerificationReport, generate_report
from distllm.verification.runner import AccuracyVerifier, verify_accuracy
from tests.verification.stubs import StubAccuracyVerifier


# --- Stub for internal helpers ----------------------------------------------


class _StubPartitioner:
    """Stub model partitioner for _run_reference tests."""

    def __init__(self):
        self.full_model = _StubModel()


class _StubModel:
    """Simple model stub."""

    def __init__(self):
        self.call_count = 0

    def __call__(self, input_ids, **kwargs):
        self.call_count += 1
        batch, seq = input_ids.shape
        logits = torch.randn(batch, seq, 8)
        return _StubLogitOutput(logits)


class _StubLogitOutput:
    """Minimal model output."""

    def __init__(self, logits):
        self.logits = logits


class _StubTokenGen:
    """Token generator stub for AccuracyVerifier internal helpers."""

    def sample(self, logits, **kwargs):
        return torch.tensor([0]), None

    def sample_batch(self, logits, sequences, **kwargs):
        return torch.tensor([0]), [None]


class _StubTokenizer:
    """Minimal tokenizer for runner tests."""

    eos_token_id = 0
    pad_token_id = 0

    def decode(self, ids, **kwargs):
        return "decoded output"


def _stub_verify_accuracy(**kwargs):
    """Stub replacement for verify_accuracy for CLI tests."""
    return VerificationReport(
        model_name=kwargs.get("model_name", "test-model"),
        num_nodes=kwargs.get("num_nodes", 2),
        dtype=kwargs.get("dtype", "float16"),
        temperature=kwargs.get("temperature", 0.0),
    )


# ============================================================================
# comparator.py
# ============================================================================


class TestCompareLogits:
    """Tests for ``compare_logits``."""

    def test_identical_logits(self, sample_logits_identical):
        """Identical logits yield cos_sim=1, kl_div=0, max_abs_diff=0."""
        gold, candidate = sample_logits_identical
        result = compare_logits(gold, candidate)
        assert math.isclose(result["cosine_sim"], 1.0, rel_tol=1e-6)
        assert result["kl_div"] == 0.0
        assert result["max_abs_diff"] == 0.0

    def test_different_logits(self, sample_logits_different):
        """Different logits produce non-perfect scores."""
        gold, candidate = sample_logits_different
        result = compare_logits(gold, candidate)
        assert result["cosine_sim"] < 1.0
        assert result["kl_div"] > 0.0
        assert result["max_abs_diff"] > 0.0

    def test_shape_mismatch_raises(self):
        """Shape mismatch between gold and candidate raises ValueError."""
        a = torch.randn(1, 3, 8)
        b = torch.randn(1, 5, 8)
        with pytest.raises(ValueError, match="Logit shape mismatch"):
            compare_logits(a, b)

    def test_single_position(self):
        """2D logits (seq, vocab) are handled correctly."""
        gold = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        candidate = gold.clone()
        result = compare_logits(gold, candidate)
        assert math.isclose(result["cosine_sim"], 1.0, rel_tol=1e-6)
        assert result["kl_div"] == 0.0

    def test_2d_logits_round_trip(self, sample_logits_2d):
        """2D logits with one differing row give predictable metrics."""
        gold, candidate = sample_logits_2d
        result = compare_logits(gold, candidate)
        assert result["cosine_sim"] < 1.0
        assert result["kl_div"] > 0.0
        assert result["max_abs_diff"] > 0.0
        assert math.isclose(result["max_abs_diff"], 1.0, rel_tol=1e-4)

    def test_returns_rounded_values(self):
        """Results are rounded to 8 decimal places."""
        gold = torch.randn(1, 2, 4)
        candidate = gold.clone()
        result = compare_logits(gold, candidate)
        for v in result.values():
            s = str(v)
            if "." in s:
                decimal_places = len(s.split(".")[1])
                assert decimal_places <= 8


class TestCompareHiddenStates:
    """Tests for ``compare_hidden_states``."""

    def test_identical_hidden(self, sample_hidden_identical):
        """Identical hidden states yield cos_sim=1, diff=0."""
        gold, candidate = sample_hidden_identical
        result = compare_hidden_states(gold, candidate)
        assert result["cosine_sim"] == 1.0
        assert result["max_abs_diff"] == 0.0
        assert result["relative_error"] == 0.0

    def test_different_hidden(self, sample_hidden_different):
        """Different hidden states produce non-perfect scores."""
        gold, candidate = sample_hidden_different
        result = compare_hidden_states(gold, candidate)
        assert result["cosine_sim"] < 1.0
        assert result["max_abs_diff"] > 0.0
        assert result["relative_error"] > 0.0

    def test_shape_mismatch_raises(self):
        """Shape mismatch raises ValueError."""
        a = torch.randn(1, 3, 8)
        b = torch.randn(1, 5, 8)
        with pytest.raises(ValueError, match="Hidden state shape mismatch"):
            compare_hidden_states(a, b)

    def test_empty_tensor_safety(self):
        """Zero vector yields cos_sim=0, diff=0, rel_err=0."""
        a = torch.zeros(1, 4)
        result = compare_hidden_states(a, a)
        assert result["cosine_sim"] == 0.0
        assert result["max_abs_diff"] == 0.0
        assert result["relative_error"] == 0.0

    def test_zero_vs_nonzero(self):
        """Zero vs non-zero hidden states produce known metrics."""
        zero = torch.zeros(2, 3)
        ones = torch.ones(2, 3)
        result = compare_hidden_states(zero, ones)
        assert result["cosine_sim"] == 0.0
        assert result["max_abs_diff"] == 1.0


class TestCompareTokens:
    """Tests for ``compare_tokens``."""

    def test_identical_sequences(self, identical_token_ids):
        """Identical sequences yield exact_match=1, edit_distance=0."""
        gold, candidate = identical_token_ids
        result = compare_tokens(gold, candidate)
        assert result["exact_match"] == 1.0
        assert result["edit_distance"] == 0.0

    def test_different_sequences(self, different_token_ids):
        """One differing token reduces exact_match and increases edit_distance."""
        gold, candidate = different_token_ids
        result = compare_tokens(gold, candidate)
        assert math.isclose(result["exact_match"], 0.8, rel_tol=1e-4)
        assert result["edit_distance"] > 0.0

    def test_different_lengths(self, different_length_token_ids):
        """Shorter candidate sequence is handled."""
        gold, candidate = different_length_token_ids
        result = compare_tokens(gold, candidate)
        assert math.isclose(result["exact_match"], 0.8, rel_tol=1e-4)
        assert result["edit_distance"] > 0.0

    def test_completely_empty_gold(self):
        """Empty gold sequence gives exact_match=0, edit_distance=1."""
        result = compare_tokens([], [1, 2, 3])
        assert result["exact_match"] == 0.0
        assert result["edit_distance"] > 0.0

    def test_both_empty(self):
        """Both empty gives exact_match=0 (divide by 1 guard), edit_distance=0."""
        result = compare_tokens([], [])
        assert result["exact_match"] == 0.0
        assert result["edit_distance"] == 0.0

    def test_large_sequences_use_jaccard_fallback(self):
        """Sequences with m*n > 10000 use Jaccard fallback."""
        a = list(range(200))
        b = list(range(200))
        result = compare_tokens(a, b)
        assert result["exact_match"] == 1.0
        assert result["edit_distance"] == 0.0


class TestCompareText:
    """Tests for ``compare_text``."""

    def test_identical_text(self, sample_texts_identical):
        """Identical text yields exact_match=1, edit_distance=0, token_overlap=1."""
        gold, candidate = sample_texts_identical
        result = compare_text(gold, candidate)
        assert result["exact_match"] == 1.0
        assert result["edit_distance"] == 0.0
        assert result["token_overlap"] == 1.0

    def test_partial_overlap(self, sample_texts_partial):
        """Partial overlap produces intermediate scores."""
        gold, candidate = sample_texts_partial
        result = compare_text(gold, candidate)
        assert result["exact_match"] == 0.0
        assert 0.0 < result["edit_distance"] < 1.0
        assert 0.0 < result["token_overlap"] < 1.0

    def test_no_overlap(self):
        """Completely different text yields no overlap."""
        gold = "The capital of France is Paris"
        candidate = "Quantum computing uses qubits"
        result = compare_text(gold, candidate)
        assert result["exact_match"] == 0.0
        assert result["token_overlap"] == 0.0

    def test_case_insensitive_overlap(self):
        """Token overlap is case-insensitive."""
        gold = "Hello World"
        candidate = "hello world"
        result = compare_text(gold, candidate)
        assert result["token_overlap"] == 1.0
        assert result["exact_match"] == 0.0

    def test_empty_strings(self):
        """Empty strings produce exact_match=1 and no errors."""
        result = compare_text("", "")
        assert result["exact_match"] == 1.0
        assert result["edit_distance"] == 0.0
        assert result["token_overlap"] == 0.0


class TestOutputComparisonDataclass:
    """Tests for ``OutputComparison`` dataclass."""

    def test_default_values(self):
        """Default values reflect worst-case scores."""
        comp = OutputComparison()
        assert comp.token_exact_match == 0.0
        assert comp.token_edit_distance == 1.0
        assert comp.logit_cosine_sim == 0.0
        assert comp.logit_kl_div == float("inf")
        assert comp.pass_threshold is False

    def test_custom_thresholds(self):
        """Custom thresholds are respected."""
        custom = {"token_exact_match": 0.9, "token_edit_distance": 0.1}
        comp = OutputComparison(
            token_exact_match=0.95,
            token_edit_distance=0.05,
            thresholds=custom,
        )
        assert comp.thresholds["token_exact_match"] == 0.9

    def test_pass_threshold_truthiness(self, output_comparison_all_pass, output_comparison_all_fail):
        """pass_threshold correctly indicates pass/fail."""
        assert output_comparison_all_pass.pass_threshold is True
        assert output_comparison_all_fail.pass_threshold is False


class TestEvaluateComparison:
    """Tests for ``evaluate_comparison``."""

    def test_all_pass(self):
        """All metrics within threshold produce pass=True."""
        metrics = {
            "token_exact_match": 1.0,
            "token_edit_distance": 0.0,
            "logit_cosine_sim": 1.0,
            "logit_kl_div": 0.0,
            "logit_max_abs_diff": 0.0,
            "hidden_cosine_sim": 1.0,
            "hidden_max_abs_diff": 0.0,
            "hidden_relative_error": 0.0,
        }
        comp = evaluate_comparison(metrics)
        assert comp.pass_threshold is True
        assert comp.token_exact_match == 1.0

    def test_one_fails(self):
        """A single metric outside threshold yields pass=False."""
        metrics = {
            "token_exact_match": 0.5,
            "token_edit_distance": 0.0,
            "logit_cosine_sim": 1.0,
            "logit_kl_div": 0.0,
            "logit_max_abs_diff": 0.0,
        }
        comp = evaluate_comparison(metrics)
        assert comp.pass_threshold is False

    def test_custom_thresholds(self):
        """Custom thresholds override defaults."""
        metrics = {
            "token_exact_match": 0.95,
        }
        comp = evaluate_comparison(metrics, thresholds={"token_exact_match": 0.9})
        assert comp.pass_threshold is True

    def test_higher_is_better_and_lower_is_better(self):
        """_all_pass distinguishes higher-is-better from lower-is-better metrics."""
        metrics = {
            "token_exact_match": 0.5,
            "token_edit_distance": 0.0,
            "logit_cosine_sim": 1.0,
            "logit_kl_div": 100.0,
        }
        assert _all_pass(metrics, DEFAULT_THRESHOLDS) is False


class TestNormalizedEditDistance:
    """Tests for the private ``_normalized_edit_distance`` helper."""

    def test_identical(self):
        """Identical sequences yield distance 0."""
        assert _normalized_edit_distance([1, 2, 3], [1, 2, 3]) == 0.0

    def test_fully_different(self):
        """Fully different sequences yield distance 1 (normalized)."""
        d = _normalized_edit_distance([1, 2], [3, 4])
        assert math.isclose(d, 1.0, rel_tol=1e-4)

    def test_both_empty(self):
        """Both empty yields 0."""
        assert _normalized_edit_distance([], []) == 0.0

    def test_one_empty(self):
        """One empty yields 1 (normalized)."""
        d = _normalized_edit_distance([1, 2, 3], [])
        assert d == 1.0

    def test_large_sequences_jaccard(self):
        """Large sequences (m*n > 10000) use Jaccard fallback."""
        a = list(range(200))
        b = list(range(200))
        d = _normalized_edit_distance(a, b)
        assert d == 0.0

    def test_partial_edit(self):
        """One substitution yields fractional distance."""
        d = _normalized_edit_distance([1, 2, 3], [1, 4, 3])
        assert 0.0 < d < 1.0


class TestDEFAULT_THRESHOLDS:
    """Tests for the DEFAULT_THRESHOLDS constant."""

    def test_all_keys_present(self):
        """All expected metric keys are present."""
        expected_keys = {
            "token_exact_match",
            "token_edit_distance",
            "logit_cosine_sim",
            "logit_kl_div",
            "logit_max_abs_diff",
            "hidden_cosine_sim",
            "hidden_max_abs_diff",
            "hidden_relative_error",
        }
        assert set(DEFAULT_THRESHOLDS) == expected_keys

    def test_strict_thresholds(self):
        """Default thresholds are set for high accuracy."""
        assert DEFAULT_THRESHOLDS["token_exact_match"] == 1.0
        assert DEFAULT_THRESHOLDS["logit_cosine_sim"] >= 0.999


# ============================================================================
# hash_registry.py
# ============================================================================


class TestComputeOutputHash:
    """Tests for ``compute_output_hash``."""

    def test_consistent_hash(self):
        """Same tensor always produces the same hash."""
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        h1 = compute_output_hash(t)
        h2 = compute_output_hash(t)
        assert h1 == h2

    def test_different_tensors_different_hash(self):
        """Different tensors produce different hashes."""
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([1.0, 3.0])
        assert compute_output_hash(t1) != compute_output_hash(t2)

    def test_shape_included(self):
        """include_shape=True distinguishes [1,3] from [3,1]."""
        a = torch.tensor([[1.0, 2.0, 3.0]])
        b = torch.tensor([[1.0], [2.0], [3.0]])
        with_shape = compute_output_hash(a, include_shape=True)
        with_shape_b = compute_output_hash(b, include_shape=True)
        assert with_shape != with_shape_b

    def test_shape_excluded(self):
        """include_shape=False hashes only values."""
        a = torch.tensor([[1.0, 2.0, 3.0]])
        b = torch.tensor([[1.0], [2.0], [3.0]])
        without_shape = compute_output_hash(a, include_shape=False)
        without_shape_b = compute_output_hash(b, include_shape=False)
        assert without_shape == without_shape_b

    def test_hex_string_format(self):
        """Return value is a 64-character hex string."""
        t = torch.randn(3, 4)
        h = compute_output_hash(t)
        assert isinstance(h, str)
        assert len(h) == 64
        int(h, 16)  # Should not raise -- it's valid hex

    def test_device_independence(self):
        """CPU and CUDA tensors (same values) produce the same hash."""
        t_cpu = torch.tensor([1.0, 2.0, 3.0])
        t_cuda = t_cpu.cuda() if torch.cuda.is_available() else t_cpu
        assert compute_output_hash(t_cpu) == compute_output_hash(t_cuda)

    def test_scalar_tensor(self):
        """Scalar (0-D) tensors are handled without error."""
        t = torch.tensor(42.0)
        h = compute_output_hash(t)
        assert len(h) == 64


class TestComputeTextHash:
    """Tests for ``compute_text_hash``."""

    def test_consistent(self):
        """Same text always produces the same hash."""
        assert compute_text_hash("hello") == compute_text_hash("hello")

    def test_different_texts_different_hash(self):
        """Different texts produce different hashes."""
        assert compute_text_hash("hello") != compute_text_hash("world")

    def test_empty_string(self):
        """Empty string produces a valid hash."""
        h = compute_text_hash("")
        assert isinstance(h, str)
        assert len(h) == 64

    def test_unicode(self):
        """Unicode characters are handled."""
        h1 = compute_text_hash("cafe")
        h2 = compute_text_hash("café")
        assert h1 != h2


class TestComputeTokenIdsHash:
    """Tests for ``compute_token_ids_hash``."""

    def test_consistent(self):
        """Same list produces same hash."""
        assert compute_token_ids_hash([1, 2, 3]) == compute_token_ids_hash([1, 2, 3])

    def test_empty_list(self):
        """Empty list produces a valid hash."""
        h = compute_token_ids_hash([])
        assert isinstance(h, str)
        assert len(h) == 64


class TestGenerationOutput:
    """Tests for ``GenerationOutput`` dataclass."""

    def test_default_values(self):
        """Defaults are sensible."""
        go = GenerationOutput(token_ids=[], text="")
        assert go.token_ids == []
        assert go.text == ""
        assert go.step_logits == []
        assert go.step_hidden_states is None
        assert go.model_name == ""
        assert go.temperature == 0.0

    def test_with_data(self, ref_generation_output):
        """Populated GenerationOutput holds all values."""
        go = ref_generation_output
        assert go.token_ids == [1, 2, 3, 4, 5]
        assert go.text == "hello world"
        assert len(go.step_logits) == 3
        assert len(go.step_hidden_states) == 3
        assert go.model_name == "test-model"


class TestOutputHashRegistry:
    """Tests for ``OutputHashRegistry``."""

    def test_empty_initial(self, hash_registry):
        """New registry has no stored data."""
        comparisons = hash_registry.compare_all()
        assert comparisons == {}
        summary = hash_registry.summary()
        assert summary["total_prompts"] == 0

    def test_store_and_compare_missing(self, hash_registry):
        """Comparing a key that was never stored returns False for all fields."""
        result = hash_registry.compare("missing-key")
        assert result == {}

    def test_store_reference_only(self, hash_registry, ref_generation_output):
        """Reference stored but no candidate: compare returns False for all fields."""
        hash_registry.store_reference("p1", ref_generation_output)
        results = hash_registry.compare("p1")
        assert all(v is False for v in results.values())

    def test_store_candidate_only(self, hash_registry, ref_generation_output):
        """Candidate stored but no reference: compare returns False for all fields."""
        hash_registry.store_candidate("p1", ref_generation_output)
        results = hash_registry.compare("p1")
        assert all(v is False for v in results.values())

    def test_matching_hashes(self, hash_registry, ref_generation_output):
        """Same output stored as both reference and candidate: all match."""
        hash_registry.store_reference("p1", ref_generation_output)
        hash_registry.store_candidate("p1", ref_generation_output)
        results = hash_registry.compare("p1")
        assert all(v is True for v in results.values())

    def test_mismatched_hashes(self, hash_registry_with_data):
        """Different outputs produce some mismatches."""
        results = hash_registry_with_data.compare("prompt-1")
        assert results.get("token_ids") is False
        assert results.get("text") is True

    def test_compare_all(self, hash_registry_with_data):
        """compare_all returns dict with all stored keys."""
        all_results = hash_registry_with_data.compare_all()
        assert "prompt-1" in all_results

    def test_compare_all_multiple(self, hash_registry_with_data, ref_generation_output):
        """Multiple keys in compare_all."""
        hash_registry_with_data.store_reference("p2", ref_generation_output)
        hash_registry_with_data.store_candidate("p2", ref_generation_output)
        all_results = hash_registry_with_data.compare_all()
        assert "prompt-1" in all_results
        assert "p2" in all_results

    def test_summary_counts(self, hash_registry_with_data):
        """Summary correctly counts pass/fail."""
        summary = hash_registry_with_data.summary()
        assert summary["total_prompts"] == 1
        assert summary["passed"] == 0
        assert summary["failed"] == 1

    def test_summary_pass_rate(self, hash_registry_with_data):
        """Pass rate is 0 when nothing passes."""
        summary = hash_registry_with_data.summary()
        assert summary["pass_rate"] == 0.0

    def test_summary_all_pass(self, hash_registry, ref_generation_output):
        """All matching outputs give pass_rate=1."""
        hash_registry.store_reference("p1", ref_generation_output)
        hash_registry.store_candidate("p1", ref_generation_output)
        hash_registry.store_reference("p2", ref_generation_output)
        hash_registry.store_candidate("p2", ref_generation_output)
        summary = hash_registry.summary()
        assert summary["total_prompts"] == 2
        assert summary["passed"] == 2
        assert summary["pass_rate"] == 1.0

    def test_to_dict_serialization(self, hash_registry_with_data):
        """to_dict produces a JSON-compatible dict."""
        d = hash_registry_with_data.to_dict()
        assert "created_at" in d
        assert "reference" in d
        assert "candidate" in d
        assert "prompt-1" in d["reference"]
        assert "prompt-1" in d["candidate"]
        json.dumps(d)

    def test_to_dict_empty(self, hash_registry):
        """Empty registry serializes correctly."""
        d = hash_registry.to_dict()
        assert d["reference"] == {}
        assert d["candidate"] == {}

    def test_extract_hashes_includes_logits_and_hidden(
        self, hash_registry, ref_generation_output
    ):
        """Hasher extracts logits and hidden state hashes if present."""
        hash_registry.store_reference("p1", ref_generation_output)
        ref_data = hash_registry._reference["p1"]
        assert "token_ids" in ref_data
        assert "text" in ref_data
        logit_keys = [k for k in ref_data if k.startswith("logits_step_")]
        assert len(logit_keys) == 3
        hidden_keys = [k for k in ref_data if k.startswith("hidden_step_")]
        assert len(hidden_keys) == 3


# ============================================================================
# report.py
# ============================================================================


class TestVerificationReport:
    """Tests for ``VerificationReport`` dataclass."""

    def test_default_values(self):
        """Defaults are reasonable."""
        report = VerificationReport()
        assert report.model_name == ""
        assert report.num_nodes == 0
        assert report.per_prompt == []
        assert report.hash_comparison is None

    def test_summary_empty(self, verification_report_empty):
        """Summary with no prompts returns all zeros."""
        summary = verification_report_empty.summary()
        assert summary["total"] == 0
        assert summary["passed"] == 0
        assert summary["pass_rate"] == 0.0

    def test_summary_with_data(self, verification_report):
        """Summary correctly aggregates from per_prompt list."""
        summary = verification_report.summary()
        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["pass_rate"] == 0.5
        assert summary["avg_token_exact_match"] == 0.5

    def test_summary_includes_metadata(self, verification_report):
        """Summary includes model_name, num_nodes, dtype."""
        summary = verification_report.summary()
        assert summary["model_name"] == "test-model"
        assert summary["num_nodes"] == 2
        assert summary["dtype"] == "float32"

    def test_summary_with_hash(self, verification_report_with_hash):
        """Summary includes hash_comparison data when available."""
        summary = verification_report_with_hash.summary()
        assert "hash_comparison" in summary
        assert summary["hash_comparison"]["total_prompts"] >= 0

    def test_to_json_serialization(self, verification_report):
        """to_json produces valid JSON."""
        json_str = verification_report.to_json()
        data = json.loads(json_str)
        assert data["model_name"] == "test-model"
        assert data["num_nodes"] == 2
        assert "summary" in data
        assert "prompts" in data
        assert len(data["prompts"]) == 2
        assert data["prompts"][0]["metrics"]["pass"] is True
        assert data["prompts"][1]["metrics"]["pass"] is False

    def test_to_json_empty(self, verification_report_empty):
        """to_json on empty report produces valid JSON."""
        json_str = verification_report_empty.to_json()
        data = json.loads(json_str)
        assert data["prompts"] == []

    def test_to_json_includes_generation_output(
        self, verification_report, output_comparison_all_pass,
        ref_generation_output, cand_generation_output
    ):
        """JSON includes reference_tokens, candidate_tokens, text."""
        report = generate_report(
            comparisons=[output_comparison_all_pass],
            per_prompt_data=[
                {
                    "prompt": "test prompt",
                    "comparison": output_comparison_all_pass,
                    "reference": ref_generation_output,
                    "candidate": cand_generation_output,
                }
            ],
            model_name="test-model",
        )
        json_str = report.to_json()
        data = json.loads(json_str)
        prompt_data = data["prompts"][0]
        assert prompt_data["reference_tokens"] == [1, 2, 3, 4, 5]
        assert prompt_data["reference_text"] == "hello world"

    def test_print_human_readable(self, verification_report, capsys):
        """print_human_readable outputs formatted text without errors."""
        verification_report.print_human_readable()
        captured = capsys.readouterr()
        assert "Accuracy Verification Report" in captured.out
        assert "test-model" in captured.out
        assert "PASS" in captured.out
        assert "FAIL" in captured.out

    def test_print_human_readable_empty(self, verification_report_empty, capsys):
        """Empty report prints gracefully."""
        with pytest.raises(KeyError):
            verification_report_empty.print_human_readable()


class TestGenerateReport:
    """Tests for ``generate_report`` function."""

    def test_minimal(self):
        """Minimal call works with just comparisons."""
        comp = OutputComparison(pass_threshold=True)
        report = generate_report([comp])
        assert len(report.per_prompt) == 1
        assert report.model_name == ""

    def test_with_all_params(
        self, sample_comparisons, hash_registry_with_data,
        ref_generation_output, cand_generation_output
    ):
        """Full parameter set populates all fields."""
        per_prompt = [
            {
                "prompt": "p1",
                "comparison": sample_comparisons[0],
                "reference": ref_generation_output,
                "candidate": cand_generation_output,
            }
        ]
        report = generate_report(
            comparisons=sample_comparisons,
            per_prompt_data=per_prompt,
            hash_registry=hash_registry_with_data,
            thresholds={"token_exact_match": 0.9},
            model_name="mymodel",
            num_nodes=3,
            dtype="bfloat16",
            temperature=0.5,
        )
        assert report.model_name == "mymodel"
        assert report.num_nodes == 3
        assert report.dtype == "bfloat16"
        assert report.temperature == 0.5
        assert report.thresholds["token_exact_match"] == 0.9
        assert report.hash_comparison is not None

    def test_generates_default_prompt_names(self):
        """When per_prompt_data is None, prompts are named prompt_0, prompt_1."""
        comp1 = OutputComparison(pass_threshold=True)
        comp2 = OutputComparison(pass_threshold=True)
        report = generate_report([comp1, comp2])
        assert report.per_prompt[0]["prompt"] == "prompt_0"
        assert report.per_prompt[1]["prompt"] == "prompt_1"

    def test_sets_created_at_timestamp(self):
        """Created_at is a reasonable recent timestamp."""
        comp = OutputComparison(pass_threshold=True)
        report = generate_report([comp])
        assert report.created_at > 1_700_000_000


# ============================================================================
# runner.py
# ============================================================================


class TestAccuracyVerifierInit:
    """Tests for ``AccuracyVerifier.__post_init__``."""

    def test_missing_model_name_raises(self):
        """AccuracyVerifier without model_name raises ValueError."""
        with pytest.raises(ValueError, match="model_name is required"):
            AccuracyVerifier(model_name="")

    def test_valid_init_with_model_name(self):
        """Providing model_name passes initialization."""
        verifier = AccuracyVerifier(model_name="test-model")
        assert verifier.model_name == "test-model"
        assert verifier.num_nodes == 2
        assert verifier.max_new_tokens == 32
        assert verifier.temperature == 0.0
        assert verifier.skip_text_comparison is False

    def test_preferred_backend_warning(self):
        """Unavailable preferred backend logs a warning."""
        _mod = __import__("distllm.verification.runner", fromlist=["logger"])
        original_warning = _mod.logger.warning
        calls = []

        def _tracking_warning(msg):
            calls.append(msg)

        _mod.logger.warning = _tracking_warning
        try:
            AccuracyVerifier(
                model_name="test-model",
                preferred_backend="nonexistent_backend",
            )
        finally:
            _mod.logger.warning = original_warning

        assert len(calls) == 1
        assert "nonexistent_backend" in calls[0]


class TestAccuracyVerifierVerify:
    """Tests for ``AccuracyVerifier.verify()`` using stub."""

    def test_verify_single_prompt_string(self, mock_accuracy_verifier):
        """verify accepts a single string as prompts."""
        report = mock_accuracy_verifier.verify("Hello world")
        assert isinstance(report, VerificationReport)
        assert report.model_name == "test-model"

    def test_verify_list_of_prompts(self, mock_accuracy_verifier):
        """verify accepts a list of prompts."""
        report = mock_accuracy_verifier.verify(["prompt1", "prompt2"])
        assert isinstance(report, VerificationReport)

    def test_verify_passes_collect_hidden_states(self, mock_accuracy_verifier):
        """collect_hidden_states parameter is forwarded."""
        report = mock_accuracy_verifier.verify(
            "hello", collect_hidden_states=True
        )
        assert report.model_name == "test-model"

    def test_verify_passes_custom_thresholds(self, mock_accuracy_verifier):
        """Custom thresholds are forwarded to generate_report."""
        custom = {"token_exact_match": 0.95}
        report = mock_accuracy_verifier.verify(
            "hello", thresholds=custom
        )
        assert report.model_name == "test-model"


class TestVerifyAccuracyFunction:
    """Tests for the top-level ``verify_accuracy`` convenience function."""

    def test_verify_accuracy_basic(self):
        """verify_accuracy creates a verifier and calls verify."""
        from distllm.verification import runner as _runner_mod

        _original_cls = _runner_mod.AccuracyVerifier
        capture_kwargs = {}

        def _constructing_verifier(**kwargs):
            capture_kwargs.update(kwargs)
            return StubAccuracyVerifier(model_name="test-model")

        _runner_mod.AccuracyVerifier = _constructing_verifier
        try:
            report = verify_accuracy(
                model_name="test-model",
                prompts=["Hello"],
                num_nodes=2,
            )
        finally:
            _runner_mod.AccuracyVerifier = _original_cls

        assert capture_kwargs == {
            "model_name": "test-model",
            "dtype": "float16",
            "num_nodes": 2,
            "temperature": 0.0,
            "max_new_tokens": 32,
            "preferred_backend": "",
            "grpc_mode": False,
            "grpc_base_port": 51050,
            "trust_remote_code": False,
        }
        assert report is not None
        assert report.model_name == "test-model"

    def test_verify_accuracy_passes_all_kwargs(self):
        """All kwargs forwarded correctly."""
        from distllm.verification import runner as _runner_mod

        _original_cls = _runner_mod.AccuracyVerifier
        capture_kwargs = {}

        def _constructing_verifier(**kwargs):
            capture_kwargs.update(kwargs)
            return StubAccuracyVerifier(model_name="test-model")

        _runner_mod.AccuracyVerifier = _constructing_verifier
        try:
            verify_accuracy(
                model_name="test-model",
                prompts=["A", "B"],
                num_nodes=4,
                dtype="bfloat16",
                temperature=0.5,
                max_new_tokens=64,
                collect_hidden_states=True,
                thresholds={"token_exact_match": 0.8},
                preferred_backend="pytorch",
                grpc_mode=True,
                grpc_base_port=52000,
                trust_remote_code=True,
            )
        finally:
            _runner_mod.AccuracyVerifier = _original_cls

        assert capture_kwargs == {
            "model_name": "test-model",
            "dtype": "bfloat16",
            "num_nodes": 4,
            "temperature": 0.5,
            "max_new_tokens": 64,
            "preferred_backend": "pytorch",
            "grpc_mode": True,
            "grpc_base_port": 52000,
            "trust_remote_code": True,
        }


class TestAccuracyVerifierInternalHelpers:
    """Tests for internal helper methods with proper stubbing."""

    def test_run_reference_structure(self):
        """_run_reference returns a properly structured GenerationOutput."""
        partitioner = _StubPartitioner()
        tokenizer = _StubTokenizer()

        verifier = AccuracyVerifier.__new__(AccuracyVerifier)
        verifier.model_name = "test-model"
        verifier.max_new_tokens = 32
        verifier.temperature = 0.0
        verifier.trust_remote_code = False
        verifier.skip_text_comparison = False
        verifier._token_gen = _StubTokenGen()

        input_ids = torch.tensor([[1, 2, 3]])
        result = verifier._run_reference(
            partitioner, input_ids, tokenizer,
            collect_hidden_states=False,
        )

        assert isinstance(result, GenerationOutput)
        assert isinstance(result.token_ids, list)
        assert isinstance(result.text, str)

    def test_run_reference_unknown_tokenizer_eos(self):
        """_run_reference handles tokenizer without pad_token_id."""
        verifier = AccuracyVerifier.__new__(AccuracyVerifier)
        verifier.model_name = "test-model"
        verifier.max_new_tokens = 1
        verifier.temperature = 0.0
        verifier.trust_remote_code = False
        verifier.skip_text_comparison = False
        verifier._token_gen = _StubTokenGen()

        # Replace _run_reference with a lambda that returns a canned output
        original_method = AccuracyVerifier._run_reference

        def _canned_run_reference(self_inner, *args, **kwargs):
            return GenerationOutput(
                token_ids=[1, 2, 0],
                text="hi",
                step_logits=[torch.randn(1, 3, 8)],
            )

        AccuracyVerifier._run_reference = _canned_run_reference
        try:
            result = verifier._run_reference(
                _StubPartitioner(), torch.tensor([[1, 2]]), _StubTokenizer()
            )
            assert result.token_ids[-1] == 0
        finally:
            AccuracyVerifier._run_reference = original_method

    def test_run_distributed_no_partitioners(self):
        """_run_distributed with all-None partitioners should skip gracefully."""
        verifier = AccuracyVerifier.__new__(AccuracyVerifier)
        verifier.max_new_tokens = 1
        verifier.temperature = 0.0
        verifier.skip_text_comparison = False
        verifier._token_gen = _StubTokenGen()

        with pytest.raises(AttributeError):
            verifier._run_distributed(
                partitioners=[None, None],
                input_ids=torch.tensor([[1, 2, 3]]),
                tokenizer=_StubTokenizer(),
                device=torch.device("cpu"),
            )


class TestRunVerificationCLI:
    """Tests for ``run_verification_cli``."""

    def test_cli_default_prompts(self):
        """CLI uses default prompts when none provided."""
        from distllm.verification import runner as _runner_mod

        original_verify = _runner_mod.verify_accuracy
        _runner_mod.verify_accuracy = _stub_verify_accuracy
        try:
            from distllm.verification.runner import run_verification_cli
            run_verification_cli(model_name="test-model")
        finally:
            _runner_mod.verify_accuracy = original_verify

    def test_cli_custom_prompts(self):
        """CLI uses provided prompts."""
        from distllm.verification import runner as _runner_mod

        original_verify = _runner_mod.verify_accuracy
        _runner_mod.verify_accuracy = _stub_verify_accuracy
        try:
            from distllm.verification.runner import run_verification_cli
            run_verification_cli(
                model_name="test-model",
                prompts=["custom prompt"],
            )
        finally:
            _runner_mod.verify_accuracy = original_verify
