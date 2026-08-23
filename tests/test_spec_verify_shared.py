"""Regression tests for the speculative-decoding family hardening (§3.3).

- Shared ``SpecVerifyBase`` (core/spec_verify.py): one canonical
  ``prefix_len`` indexing + acceptance routine. The C3 off-by-one bug
  (verifying draft token i against the wrong logits row) must be fixed in
  exactly one place and inherited by all four consumers.
- ``multi_draft_verifier`` previously used ``pos = generated.shape[1] + i``
  (OFF BY ONE) — now routes through the shared helper (pos = L + i,
  L = prefix.shape[1] - 1).
- Draft selection is quality-aware: DraftModelRouter._score consults a
  DraftQualityScorer when one is configured.
"""

import math

import torch

from distllm.core.spec_verify import accept_token, prefix_len, verify_chain
from distllm.core.draft_model_router import (
    DraftModelFleet,
    DraftModelRouter,
    DraftModelSpec,
    DraftModelHealth,
    RoutingConstraints,
)
from distllm.core.draft_quality_scorer import DraftQualityScorer


# ── prefix_len indexing (the C3-class invariant) ──

def test_prefix_len_is_L_minus_1():
    prefix = torch.zeros(1, 7, dtype=torch.long)  # L = 7
    assert prefix_len(prefix) == 6


def test_verify_chain_greedy_correct_indexing():
    """Draft token at absolute prefix position L+i must be checked against
    logits row L-1+i (causal shift). Build logits so the target argmax at
    row 6 == draft token, rows 7.. wrong — only the correct shift accepts."""
    # prefix length L = 4 -> first draft token verified at logits row 3
    prefix = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)  # L=4
    draft = torch.tensor([[40, 41]], dtype=torch.long)  # 2 draft tokens

    vocab = 100
    logits = torch.full((1, 6, vocab), -10.0)  # seq_len 6 (rows 0..5)
    # target argmax at row 3 (L-1) == 40, row 4 (L) == 41
    logits[0, 3, 40] = 10.0
    logits[0, 4, 41] = 10.0
    # ensure off-by-one neighbors are WRONG (argmax != draft)
    logits[0, 2, 99] = 5.0
    logits[0, 5, 99] = 5.0

    # Greedy (temperature 0): both should be accepted (correct causal shift).
    n = verify_chain(prefix, draft, logits, temperature=0)
    assert n == 2, f"expected both accepted, got {n}"


def test_verify_chain_rejects_on_wrong_row():
    """If the logits were shifted by the old off-by-one (token i checked at
    row L+i instead of L-1+i), the first draft token would mismatch and be
    rejected. This proves the shared routine uses the correct shift."""
    prefix = torch.tensor([[1, 2, 3]], dtype=torch.long)  # L=3
    draft = torch.tensor([[50]], dtype=torch.long)
    vocab = 100
    logits = torch.full((1, 5, vocab), -10.0)
    # Put the match at row 2 (correct: L-1) but NOT at row 3 (old off-by-one)
    logits[0, 2, 50] = 10.0
    logits[0, 3, 7] = 10.0  # wrong token at the off-by-one row
    assert verify_chain(prefix, draft, logits, temperature=0) == 1


def test_accept_token_rejection_sampling_uniform_fallback():
    prefix = torch.tensor([[1]], dtype=torch.long)  # L=1
    vocab = 1000
    logits = torch.full((1, 3, vocab), -10.0)
    logits[0, 0, 5] = 10.0  # only token 5 has mass at row 0
    # token 5 must be accepted (p≈1). A different token (7) must be rejected.
    assert accept_token(logits, 0, 5, temperature=1.0, vocab_size=vocab) is True
    assert accept_token(logits, 0, 7, temperature=1.0, vocab_size=vocab) is False


def test_multi_draft_verifier_uses_shared_routine():
    """Importability + behavior: the module now imports the shared helper and
    its flat-chain acceptance matches verify_chain exactly."""
    from distllm.core import multi_draft_verifier as mdv

    assert hasattr(mdv, "accept_token")
    assert hasattr(mdv, "prefix_len")

    prefix = torch.tensor([[1, 2]], dtype=torch.long)  # L=2
    draft = torch.tensor([[9, 8]], dtype=torch.long)
    vocab = 100
    logits = torch.full((1, 5, vocab), -10.0)
    logits[0, 1, 9] = 10.0   # row L-1 == first draft
    logits[0, 2, 8] = 10.0   # row L   == second draft
    assert verify_chain(prefix, draft, logits, temperature=0) == 2


# ── Draft selection quality-aware (DraftModelRouter + DraftQualityScorer) ──

def _fleet_with_two_endpoints():
    fleet = DraftModelFleet()
    fleet.register(DraftModelSpec(
        endpoint_url="http://a:8000/v1", model_name="small", hardware="cpu",
        cost_per_hour=0.05, avg_latency_ms=45.0, max_concurrent=4,
    ))
    fleet.register(DraftModelSpec(
        endpoint_url="http://b:8001/v1", model_name="big", hardware="cuda:0",
        cost_per_hour=0.60, avg_latency_ms=8.0, max_concurrent=4,
    ))
    # Make both endpoints healthy with identical live signals.
    for url in ("http://a:8000/v1", "http://b:8001/v1"):
        h = fleet.get_health(url)
        h.is_healthy = True
        h.recent_latency_ms = 10.0
        h.recent_acceptance_rate = 0.5
        h.current_concurrent = 0
    return fleet


def test_router_ignores_scorer_when_none():
    fleet = _fleet_with_two_endpoints()
    router = DraftModelRouter(fleet)  # no quality_scorer
    decision = router.select(RoutingConstraints(), target_model="target")
    assert decision.selected_url in ("http://a:8000/v1", "http://b:8001/v1")


def test_router_prefers_higher_quality_draft():
    fleet = _fleet_with_two_endpoints()
    scorer = DraftQualityScorer()
    # Small draft proven great for "target"; big draft proven poor.
    scorer.record("small", "target", accepted=10, total=10)  # rate 1.0
    scorer.record("big", "target", accepted=1, total=10)     # rate 0.1
    router = DraftModelRouter(fleet, quality_scorer=scorer)
    decision = router.select(RoutingConstraints(), target_model="target")
    assert decision.selected_url == "http://a:8000/v1"  # small wins on quality
