"""Real-object speculative-decoding test — no test doubles.

Drives a real ``SpeculativeDecoder`` with real CPU torch forward functions (a
tiny deterministic model), exercising the actual draft+verify acceptance math
that was previously only covered with mocks.
"""

from __future__ import annotations

from typing import Any

import torch

from distllm.core.speculative_decoder import SpeculativeDecoder


VOCAB = 50


def _edge_model(input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Deterministic toy model: predicts the next token id = (cur + 1) % VOCAB
    at every position.  Same function serves as target AND draft."""
    batch, seq = input_ids.shape
    logits = torch.full((batch, seq, VOCAB), -1e9, dtype=torch.float32)
    for b in range(batch):
        for i in range(seq):
            logits[b, i, (int(input_ids[b, i]) + 1) % VOCAB] = 0.0
    return logits


class TestRealSpeculative:
    def test_generation_produces_valid_finite_tokens(self) -> None:
        decoder = SpeculativeDecoder(
            target_forward=_edge_model,
            draft_forward=_edge_model,
            num_candidates=3,
            top_k=20,
            temperature=1.0,
            device="cpu",
        )
        prompt = torch.arange(0, 8, dtype=torch.long).reshape(1, 8)
        out = decoder.generate(torch.tensor(prompt), max_new_tokens=16)

        # Real output: valid token ids in the vocab range, finite.
        assert out is not None
        assert out.dim() == 2
        assert out.shape[1] > 8  # generation progressed past the prompt
        assert bool(torch.isfinite(out.float()).all())
        assert bool((out >= 0).all()) and bool((out < VOCAB).all())

    def test_acceptance_runs_without_error_and_reports_stats(self) -> None:
        decoder = SpeculativeDecoder(
            target_forward=_edge_model,
            draft_forward=_edge_model,
            num_candidates=4,
            device="cpu",
        )
        prompt = torch.arange(0, 8, dtype=torch.long).reshape(1, 8)
        decoder.generate(torch.tensor(prompt), max_new_tokens=12)

        stats = decoder.stats  # the computed-stats property
        assert stats["draft_calls"] > 0
        assert stats["target_calls"] > 0
        assert stats["total_proposed"] > 0
        rate = stats.get("acceptance_rate")
        assert isinstance(rate, float)
        # acceptance_rate must be in [0, 1]: correction tokens are no longer
        # counted as acceptances (they are fresh target samples, not drafts).
        assert 0.0 <= rate <= 1.0

    def test_verify_uses_correct_logits_position(self) -> None:
        """Pin the logits convention: logits[k] predicts token[k+1], so draft
        token i (at prefix_len+i) must be verified against logits[prefix_len+i-1].
        A verifier using prefix_len+i would reject this correctly-accepted token."""
        decoder = SpeculativeDecoder(
            target_forward=_edge_model,
            draft_forward=_edge_model,
            num_candidates=2,
            device="cpu",
        )
        decoder._temperature = 0  # greedy verify path

        prefix = torch.tensor([[0, 1, 2, 3, 4]])   # prompt_len = 5
        draft = torch.tensor([[5]])                 # one draft token
        full_input = torch.cat([prefix, draft], dim=1)  # (1, 6)

        logits = torch.full((1, 6, 100), -10.0)
        logits[0, 4, 5] = 5.0  # position 4 predicts token 5 -> CORRECT acceptance
        logits[0, 5, 6] = 5.0  # position 5 predicts token 6 (off-by-one would reject)

        accepted = decoder._verify_tokens(prefix, full_input, draft, logits)
        assert accepted == 1  # correct verifier accepts; buggy (+0) returns 0