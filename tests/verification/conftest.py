"""Fixtures for verification module tests.

Provides reusable fixtures for Comparator, HashRegistry, and Report tests.

Backend selection is stubbed via an autouse fixture so that
AccuracyVerifier construction tests do not need real inference
backends or mocking in individual test functions.
"""

from __future__ import annotations

import pytest
import torch

from distllm.verification.comparator import (
    DEFAULT_THRESHOLDS,
    OutputComparison,
)
from distllm.verification.hash_registry import GenerationOutput, OutputHashRegistry
from distllm.verification.report import VerificationReport, generate_report
from tests.verification.stubs import StubAccuracyVerifier, select_backend as _stub_select_backend


@pytest.fixture(autouse=True)
def _patch_select_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent ``AccuracyVerifier.__post_init__`` from doing real backend selection.

    The real ``select_backend`` may trigger imports of heavy inference
    engines (pytorch_backend, vllm_backend, ...) or fail when no GPUs
    are available.  This fixture replaces it with the lightweight stub
    from ``tests.verification.stubs`` so that construction-only and
    verify-flow tests work in any environment.
    """
    monkeypatch.setattr("distllm.backends.select_backend", _stub_select_backend)


# ── Tensor Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def sample_logits_identical() -> torch.Tensor:
    """Return a pair of identical logit tensors (batch=1, seq=3, vocab=8)."""
    torch.manual_seed(42)
    t = torch.randn(1, 3, 8)
    return t, t.clone()


@pytest.fixture
def sample_logits_different() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a pair of different logit tensors."""
    torch.manual_seed(42)
    gold = torch.randn(1, 3, 8)
    torch.manual_seed(99)
    candidate = torch.randn(1, 3, 8)
    return gold, candidate


@pytest.fixture
def sample_logits_2d() -> tuple[torch.Tensor, torch.Tensor]:
    """Return 2D logit tensors (seq=5, vocab=4)."""
    gold = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0],
                          [1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    candidate = gold.clone()
    candidate[-1] = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    return gold, candidate


@pytest.fixture
def sample_hidden_identical() -> torch.Tensor:
    """Return a pair of identical hidden state tensors."""
    torch.manual_seed(1)
    t = torch.randn(2, 4, 16)
    return t, t.clone()


@pytest.fixture
def sample_hidden_different() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a pair of different hidden state tensors."""
    torch.manual_seed(1)
    gold = torch.randn(2, 4, 16)
    torch.manual_seed(2)
    candidate = torch.randn(2, 4, 16)
    return gold, candidate


# ── Token / Text Fixtures ────────────────────────────────────────────────


@pytest.fixture
def identical_token_ids() -> tuple[list[int], list[int]]:
    """Return identical token ID sequences."""
    return [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]


@pytest.fixture
def different_token_ids() -> tuple[list[int], list[int]]:
    """Return different token ID sequences."""
    return [1, 2, 3, 4, 5], [1, 2, 99, 4, 5]


@pytest.fixture
def different_length_token_ids() -> tuple[list[int], list[int]]:
    """Return token ID sequences of different lengths."""
    return [1, 2, 3, 4, 5], [1, 2, 3, 4]


@pytest.fixture
def sample_texts_identical() -> tuple[str, str]:
    """Return identical text strings."""
    return "The capital of France is Paris", "The capital of France is Paris"


@pytest.fixture
def sample_texts_partial() -> tuple[str, str]:
    """Return text strings with partial overlap."""
    return "The capital of France is Paris", "The capital of France is London"


@pytest.fixture
def sample_texts_different() -> tuple[str, str]:
    """Return completely different text strings."""
    return "The capital of France is Paris", "Machine learning is fun"


# ── GenerationOutput Fixtures ────────────────────────────────────────────


@pytest.fixture
def ref_generation_output() -> GenerationOutput:
    """Create a reference GenerationOutput with sample data."""
    torch.manual_seed(42)
    return GenerationOutput(
        token_ids=[1, 2, 3, 4, 5],
        text="hello world",
        step_logits=[torch.randn(1, 3, 8) for _ in range(3)],
        step_hidden_states=[torch.randn(1, 3, 16) for _ in range(3)],
        model_name="test-model",
        temperature=0.0,
        prompt="hello",
    )


@pytest.fixture
def cand_generation_output(ref_generation_output) -> GenerationOutput:
    """Create a candidate GenerationOutput that differs slightly from ref."""
    torch.manual_seed(99)
    return GenerationOutput(
        token_ids=[1, 2, 99, 4, 5],
        text="hello world",
        step_logits=[torch.randn(1, 3, 8) for _ in range(3)],
        step_hidden_states=[torch.randn(1, 3, 16) for _ in range(3)],
        model_name="test-model",
        temperature=0.0,
        prompt="hello",
    )


# ── Comparison Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def output_comparison_all_pass() -> OutputComparison:
    """OutputComparison that passes all default thresholds."""
    return OutputComparison(
        token_exact_match=1.0,
        token_edit_distance=0.0,
        logit_cosine_sim=1.0,
        logit_kl_div=0.0,
        logit_max_abs_diff=0.0,
        hidden_cosine_sim=1.0,
        hidden_max_abs_diff=0.0,
        hidden_relative_error=0.0,
        pass_threshold=True,
        thresholds=DEFAULT_THRESHOLDS,
    )


@pytest.fixture
def output_comparison_all_fail() -> OutputComparison:
    """OutputComparison that fails all default thresholds."""
    return OutputComparison(
        token_exact_match=0.0,
        token_edit_distance=1.0,
        logit_cosine_sim=-1.0,
        logit_kl_div=100.0,
        logit_max_abs_diff=100.0,
        hidden_cosine_sim=-1.0,
        hidden_max_abs_diff=100.0,
        hidden_relative_error=100.0,
        pass_threshold=False,
        thresholds=DEFAULT_THRESHOLDS,
    )


# ── HashRegistry Fixtures ────────────────────────────────────────────────


@pytest.fixture
def hash_registry() -> OutputHashRegistry:
    """Return an empty OutputHashRegistry."""
    return OutputHashRegistry()


@pytest.fixture
def hash_registry_with_data(
    hash_registry: OutputHashRegistry,
    ref_generation_output: GenerationOutput,
    cand_generation_output: GenerationOutput,
) -> OutputHashRegistry:
    """Return a HashRegistry with one prompt stored for ref and candidate."""
    hash_registry.store_reference("prompt-1", ref_generation_output)
    hash_registry.store_candidate("prompt-1", cand_generation_output)
    return hash_registry


# ── Report Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def sample_comparisons(
    output_comparison_all_pass: OutputComparison,
    output_comparison_all_fail: OutputComparison,
) -> list[OutputComparison]:
    """Return a list of comparisons (one pass, one fail)."""
    return [output_comparison_all_pass, output_comparison_all_fail]


@pytest.fixture
def verification_report(
    sample_comparisons: list[OutputComparison],
) -> VerificationReport:
    """Return a populated VerificationReport."""
    return generate_report(
        comparisons=sample_comparisons,
        model_name="test-model",
        num_nodes=2,
        dtype="float32",
        temperature=0.0,
    )


@pytest.fixture
def verification_report_with_hash(
    sample_comparisons: list[OutputComparison],
    hash_registry_with_data: OutputHashRegistry,
) -> VerificationReport:
    """Return a VerificationReport with hash comparison data."""
    return generate_report(
        comparisons=sample_comparisons,
        hash_registry=hash_registry_with_data,
        model_name="test-model",
        num_nodes=2,
    )


@pytest.fixture
def verification_report_empty() -> VerificationReport:
    """Return an empty VerificationReport."""
    return VerificationReport(
        model_name="test-model",
        num_nodes=2,
        dtype="float32",
        temperature=0.0,
        thresholds=DEFAULT_THRESHOLDS,
    )


# ── Runner Fixtures (mocked) ─────────────────────────────────────────────


@pytest.fixture
def mock_accuracy_verifier() -> StubAccuracyVerifier:
    """Return a StubAccuracyVerifier that returns canned verify() results.

    Unlike a plain ``MagicMock``, this is a real ``AccuracyVerifier``
    subclass that provides type-correct ``verify()`` return values
    and has the correct attribute structure.  It avoids any real
    model loading or backend selection.
    """
    return StubAccuracyVerifier(model_name="test-model")
