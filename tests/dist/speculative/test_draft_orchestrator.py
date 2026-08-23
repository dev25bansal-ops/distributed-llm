"""Tests for draft orchestrator, Thompson sampling bandit, and acceptance matrix."""

from __future__ import annotations

import pytest

from distllm.dist.speculative.draft_orchestrator import (
    DomainAcceptanceMatrix,
    DraftOrchestrator,
    ThompsonSamplingBandit,
)


class TestThompsonSamplingSelection:
    """ThompsonSamplingBandit selection behaviour."""

    def test_select_top_k(self) -> None:
        bandit = ThompsonSamplingBandit(alpha_prior=1.0, beta_prior=1.0)
        # Register two drafts with different posteriors via updates
        bandit.update("draft_a", "code", accepted=50, rejected=5)
        bandit.update("draft_b", "code", accepted=10, rejected=20)

        selected = bandit.select(["draft_a", "draft_b"], domain="code", k=2)
        assert len(selected) == 2
        assert "draft_a" in selected
        assert "draft_b" in selected

    def test_select_returns_empty_for_k_zero(self) -> None:
        bandit = ThompsonSamplingBandit()
        assert bandit.select(["a", "b"], "general", k=0) == []


class TestAcceptanceMatrixUpdate:
    """DomainAcceptanceMatrix set/get/decay behaviour."""

    def test_get_falls_back_to_prior(self) -> None:
        matrix = DomainAcceptanceMatrix(alpha_prior=2.0, beta_prior=5.0)
        alpha, beta = matrix.get("draft_x", "math")
        assert alpha == 2.0
        assert beta == 5.0

    def test_set_and_get_round_trip(self) -> None:
        matrix = DomainAcceptanceMatrix()
        matrix.set("draft_x", "math", 10.0, 3.0)
        alpha, beta = matrix.get("draft_x", "math")
        assert alpha == 10.0
        assert beta == 3.0

    def test_decay_reduces_counts(self) -> None:
        matrix = DomainAcceptanceMatrix()
        matrix.set("draft_x", "math", 100.0, 50.0)
        matrix.decay(rate=0.5)
        alpha, beta = matrix.get("draft_x", "math")
        assert alpha == 50.0
        assert beta == 25.0


class TestOrchestratorRegisterAndSelect:
    """DraftOrchestrator registration and draft selection."""

    def test_register_and_select_drafts(self) -> None:
        orch = DraftOrchestrator()
        orch.register_draft("fast-v0", "model-small", "RTX-4090", 0.01)
        orch.register_draft("big-v1", "model-large", "A100-80GB", 0.05)
        assert len(orch.draft_bank) == 2

        selected = orch.select_drafts(
            request_text="Write a quicksort",
            domain="code",
            k=2,
        )
        assert len(selected) == 2

    def test_duplicate_registration_raises(self) -> None:
        orch = DraftOrchestrator()
        orch.register_draft("d1", "m1", "A100", 0.01)
        with pytest.raises(ValueError, match="already registered"):
            orch.register_draft("d1", "m2", "H100", 0.02)
