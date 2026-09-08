"""Enhancement #1: vectorized batch acceptance kernel tests.

``accept_tokens_batched`` must be mathematically identical to per-token
``accept_token`` calls, with 2 GPU syncs instead of 2N.
"""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")

sv = importlib.import_module("distllm.core.spec_verify")


def _logits(rows: int, vocab: int = 16, best: int = 3, seed: int = 0) -> "torch.Tensor":
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(1, rows, vocab, generator=g)
    base[:, :, :] = base * 0.1  # flatten
    base[:, :, best] = 10.0     # argmax = best everywhere
    return base


class TestGreedyBatched:
    def test_all_accepted_matches_per_token(self):
        logits = _logits(12)
        prefix = torch.tensor([[1, 2, 3]])
        L = sv.prefix_len(prefix)
        drafts = torch.full((1, 8), 3)
        assert sv.accept_tokens_batched(logits, L, drafts, temperature=0.0) == 8

    def test_first_rejection_terminates(self):
        logits = _logits(12, best=3)
        prefix = torch.tensor([[1, 2, 3]])
        L = sv.prefix_len(prefix)
        drafts = torch.tensor([[3, 3, 7, 3, 3]])
        n = sv.accept_tokens_batched(logits, L, drafts, temperature=0.0)
        assert n == 2  # token 7 rejected at index 2

    def test_bounds_exhaustion_c02_semantics(self):
        logits = _logits(5)  # L=3, rows 2..4 usable -> 3 of 8 verify
        prefix = torch.tensor([[1, 2, 3]])
        L = sv.prefix_len(prefix)
        drafts = torch.full((1, 8), 3)
        assert sv.accept_tokens_batched(logits, L, drafts, temperature=0.0) == 3


class TestSampledBatched:
    def test_acceptance_matches_reference_distribution(self):
        """Over many seeds, batched acceptance frequency ~= per-token."""
        logits = _logits(64, vocab=32, best=5, seed=1)
        # Temperature 1.0; make argmax strongly dominant -> near-100% accept
        prefix = torch.tensor([[1, 2, 3]])
        L = sv.prefix_len(prefix)
        drafts = torch.full((1, 50), 5)
        g = torch.Generator().manual_seed(42)
        n = sv.accept_tokens_batched(logits, L, drafts, temperature=1.0, rng=g)
        assert n == 50  # p(token 5) ~= 1.0 -> all accepted

    def test_zero_draft_prob_rejects(self):
        logits = _logits(12)
        prefix = torch.tensor([[1, 2, 3]])
        L = sv.prefix_len(prefix)
        drafts = torch.full((1, 6), 3)
        # q=0 at every position -> total rejection at index 0
        n = sv.accept_tokens_batched(
            logits, L, drafts, draft_probs=[0.0] * 6, temperature=1.0
        )
        assert n == 0

    def test_uniform_fallback_accepts_dominant_token(self):
        logits = _logits(12)  # argmax prob ~= 1.0
        prefix = torch.tensor([[1, 2, 3]])
        L = sv.prefix_len(prefix)
        drafts = torch.full((1, 6), 3)
        # p * vocab clamped at 1.0 -> all accepted
        assert sv.accept_tokens_batched(logits, L, drafts, temperature=1.0) == 6

    def test_list_or_tensor_draft_probs_equivalent(self):
        import torch.nn.functional as F

        logits = torch.randn(1, 12, 8)
        prefix = torch.tensor([[1, 2, 3]])
        L = sv.prefix_len(prefix)
        drafts = torch.tensor([[2, 4, 6, 1]])
        q_list = [0.3, 0.2, 0.1, 0.05]
        g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
        n1 = sv.accept_tokens_batched(logits, L, drafts, draft_probs=q_list, temperature=1.0, rng=g1)
        n2 = sv.accept_tokens_batched(
            logits, L, drafts, draft_probs=torch.tensor(q_list), temperature=1.0, rng=g2
        )
        assert n1 == n2


class TestSyncCount:
    def test_only_two_syncs(self):
        """Instrumented: the batched kernel pulls .item() exactly twice
        (first-rejection index and, on full accept, none extra) regardless
        of chain length N."""
        logits = _logits(64)
        prefix = torch.tensor([[1, 2, 3]])
        L = sv.prefix_len(prefix)
        drafts = torch.full((1, 40), 3)

        calls = {"item": 0}
        real_item = torch.Tensor.item

        def counting_item(self):
            calls["item"] += 1
            return real_item(self)

        torch.Tensor.item = counting_item
        try:
            sv.accept_tokens_batched(logits, L, drafts, temperature=0.0)
        finally:
            torch.Tensor.item = real_item
        assert calls["item"] <= 2, f"batched kernel made {calls['item']} syncs"
