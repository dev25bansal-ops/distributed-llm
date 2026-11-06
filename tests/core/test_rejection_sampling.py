"""Regression tests for rejection sampling in speculative acceptance (C-03).

Bug C-03: acceptance compared `torch.rand(1).item() >= p` (the target model's
absolute probability of the draft token) instead of true rejection sampling
`min(1, p_target / q_draft)`. Because p_target is typically 0.05-0.6,
acceptance collapsed to mean(p) (~10-30%), making speculative decoding
SLOWER than plain decoding at temperature > 0 while appearing to work.

Fix: route all acceptance loops through `spec_verify.accept_token`, which
implements the correct `min(1, p/q)` comparison.
"""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")

spec_verify = importlib.import_module("distllm.core.spec_verify")


def test_accept_token_uses_rejection_sampling_high_draft_prob():
    """When the draft is confident (q high) and target agrees, accept.

    q=0.9, p=0.9 -> acceptance = min(1, 0.9/0.9) = 1.0 -> always accept.
    Under the OLD buggy code this compared rand >= 0.9, accepting only ~90%
    of the time. Under the fix, acceptance is 100%.
    """
    vocab = 4
    logits = torch.zeros(1, 1, vocab)
    logits[0, 0, 2] = 10.0  # token 2 is argmax
    # temperature such that softmax gives p ~= 0.9 for token 2
    # With logits [0,0,10,0] at T: p = e^(10/T) / (3 + e^(10/T))
    # Choose T so p is high. T=4 -> e^2.5=12.18 -> p=12.18/15.18=0.802
    temp = 4.0
    probs = torch.softmax(logits[0, 0] / temp, dim=-1)
    p = float(probs[2].item())
    assert p > 0.7, f"setup: expected p>0.7, got {p}"

    # q (draft prob) equal to p -> acceptance should be 1.0 (always accept)
    results = [
        spec_verify.accept_token(logits, 0, 2, draft_prob=p, temperature=temp)
        for _ in range(200)
    ]
    assert all(results), (
        "With q == p, acceptance = min(1, p/q) = 1.0, so every draw must "
        f"accept. Got {sum(results)}/200 accepted. If this fails, the "
        "rejection-sampling fix (C-03) regressed to the raw-p comparison."
    )


def test_accept_token_rejects_when_target_disagrees():
    """When target probability is far below draft probability, mostly reject.

    q=0.9 (draft confident), p=0.05 (target thinks unlikely)
    -> acceptance = min(1, 0.05/0.9) = 0.055 -> ~5% accepted.
    Under the OLD code this compared rand >= 0.05 -> ~95% accepted (WRONG).
    """
    vocab = 4
    logits = torch.zeros(1, 1, vocab)
    logits[0, 0, 1] = 10.0  # argmax is token 1
    temp = 1.0
    # p for token 2 (not argmax) is tiny
    probs = torch.softmax(logits[0, 0] / temp, dim=-1)
    p_token2 = float(probs[2].item())
    assert p_token2 < 0.01, f"setup: expected p<0.01, got {p_token2}"

    results = [
        spec_verify.accept_token(logits, 0, 2, draft_prob=0.9, temperature=temp)
        for _ in range(300)
    ]
    accepted_rate = sum(results) / len(results)
    # acceptance ~= p/q = 0.0001/0.9 ~ 0 (essentially never)
    assert accepted_rate < 0.05, (
        f"Expected near-zero acceptance (p<<q), got {accepted_rate:.3f}. "
        "If this is high, the code compares rand >= p instead of p/q."
    )


def test_accept_token_partial_acceptance_rate_matches_p_over_q():
    """Empirical acceptance rate should approximate min(1, p/q).

    Setup: q=0.5, choose logits/temp so p ~= 0.25 -> expected rate ~= 0.5.
    """
    # Make a distribution where token 0 has probability ~0.25 at temp=1
    logits = torch.tensor([[[0.0, 1.0, 1.0, 1.0]]])  # softmax: e^0/(e^0+3e^1)
    temp = 1.0
    probs = torch.softmax(logits[0, 0] / temp, dim=-1)
    p = float(probs[0].item())
    q = 0.5
    expected = min(1.0, p / q)

    n = 4000
    results = [
        spec_verify.accept_token(logits, 0, 0, draft_prob=q, temperature=temp)
        for _ in range(n)
    ]
    empirical = sum(results) / n
    # Allow generous tolerance for sampling noise
    assert abs(empirical - expected) < 0.08, (
        f"Expected acceptance ~= {expected:.3f} (p/q), got {empirical:.3f}. "
        "Rejection sampling is not implemented correctly."
    )


def test_accept_token_greedy_matches_argmax():
    """At temperature 0, acceptance is argmax equality (deterministic)."""
    vocab = 4
    logits = torch.zeros(1, 1, vocab)
    logits[0, 0, 3] = 10.0
    assert spec_verify.accept_token(logits, 0, 3, temperature=0.0) is True
    assert spec_verify.accept_token(logits, 0, 1, temperature=0.0) is False


def test_imports_accept_token_in_both_decoders():
    """M-02 partial: both decoders must import the shared kernel.

    This is the invariant guard. If a future refactor removes the import
    from either module, acceptance math silently diverges again.
    """
    import distllm.core.inference_engine as ie
    import distllm.core.draft_tree as dt

    assert hasattr(ie, "accept_token"), (
        "inference_engine must import spec_verify.accept_token"
    )
    assert hasattr(dt, "accept_token"), (
        "draft_tree must import spec_verify.accept_token"
    )
