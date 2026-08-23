"""Property-based tests: speculative-decoding acceptance monotonicity.

Real module under test (read before writing):
  ``distllm.core.spec_verify.accept_token`` -- the canonical speculative-
  decoding acceptance decision.  In sampled mode it uses proper rejection
  sampling with acceptance probability

      accept_prob(p, q) = min(1, p / q)

  where ``p`` is the target-model probability of the draft token and ``q`` is
  the draft-model probability of that same token.

Invariant under test: ``accept_prob`` is **monotonically non-decreasing** in
the target probability ``p`` for a fixed draft probability ``q > 0``.  That is
the comparator / score->acceptance monotonicity property used by the
speculative-decoding verifier: a token the target model considers *more*
likely is never *less* likely to be accepted.

For a fixed RNG draw ``u`` the decision is ``u < min(1, p/q)``.  Therefore,
with ``p_lo <= p_hi`` and a *single* shared ``u``,

    min(1, p_lo/q) <= min(1, p_hi/q)
    ==>  (u < min(1, p_lo/q)) <= (u < min(1, p_hi/q))

i.e. acceptance at ``p_lo`` never exceeds acceptance at ``p_hi``.  We draw one
``u`` per example and compare both target probs against it.  We also assert
the real ``accept_token`` agrees with the decision ``u < min(1, p/q)`` for a
fixed RNG, and that the acceptance threshold always lies in [0, 1].
"""

from __future__ import annotations

import math

import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from distllm.core.spec_verify import accept_token


_PBT_SETTINGS = dict(max_examples=30, deadline=None)


def _prob():
    # Valid probability in (0, 1] (q kept strictly > 0 so p/q is well-defined;
    # p spans (0, 1]).
    return st.floats(1e-6, 1.0, allow_nan=False, allow_infinity=False)


def _target_logits(p: float) -> torch.Tensor:
    """(1, 1, vocab=2) logits whose softmax[:, 0] == p exactly.

    Setting logits to [log(p), log(1-p)] reproduces probabilities [p, 1-p]
    under softmax (softmax of log-probs == the probs), which is the correct
    way to encode a target distribution in logit space.
    """
    vocab = 2
    pos = 0
    logits = torch.zeros(1, 1, vocab)
    logits[0, pos, 0] = math.log(p)
    logits[0, pos, 1] = math.log(1.0 - p) if p < 1.0 else math.log(1e-9)
    return logits


@settings(**_PBT_SETTINGS)
@given(p_lo=_prob(), p_hi=_prob(), q=_prob())
def test_acceptance_monotonic_in_target_prob(p_lo, p_hi, q):
    """Fixed draft prob q>0, p_lo<=p_hi => acceptance(p_lo) <= acceptance(p_hi)
    for one shared RNG draw u."""
    assume(p_lo <= p_hi)
    assume(q > 0)

    logits_lo = _target_logits(p_lo)
    logits_hi = _target_logits(p_hi)

    # Both decisions draw from an independent generator seeded identically and
    # each consumes exactly ONE torch.rand() call, so they see the SAME draw u.
    rng_lo = torch.Generator().manual_seed(1234)
    rng_hi = torch.Generator().manual_seed(1234)

    a_lo = accept_token(logits_lo, 0, 0, draft_prob=q, temperature=1.0, rng=rng_lo)
    a_hi = accept_token(
        logits_hi, 0, 0, draft_prob=q, temperature=1.0, rng=rng_hi
    )

    # Monotonic: cannot be True at the lower p and False at the higher p.
    assert not (a_lo and not a_hi)


@settings(**_PBT_SETTINGS)
@given(p_lo=_prob(), p_hi=_prob(), q=_prob())
def test_accept_prob_threshold_monotonic(p_lo, p_hi, q):
    """The acceptance threshold min(1, p/q) is non-decreasing in p for any
    fixed draw u (pure math check of the comparator)."""
    assume(p_lo <= p_hi)
    assume(q > 0)
    acc_lo = min(1.0, p_lo / q)
    acc_hi = min(1.0, p_hi / q)
    assert acc_lo <= acc_hi

    # For a single shared draw u the decisions must respect monotonicity.
    u = torch.rand(1).item()
    d_lo = u < acc_lo
    d_hi = u < acc_hi
    assert not (d_lo and not d_hi)


@settings(**_PBT_SETTINGS)
@given(p=_prob(), q=_prob())
def test_accept_token_matches_min_one_formula(p, q):
    """accept_token's sampled-mode decision equals u < min(1, p/q) for a fixed
    RNG draw, and the bound is always in [0, 1]."""
    logits = _target_logits(p)
    # accept_token draws exactly ONE torch.rand() from the supplied generator.
    # Re-create a generator with the same seed and draw once to recover that
    # exact internal draw u.
    rng = torch.Generator().manual_seed(7)
    decision = accept_token(logits, 0, 0, draft_prob=q, temperature=1.0, rng=rng)

    if q <= 0:
        assert decision is False
    else:
        acc = min(1.0, p / q)
        assert 0.0 <= acc <= 1.0
        rng2 = torch.Generator().manual_seed(7)
        u_internal = torch.rand(1, generator=rng2).item()
        assert decision == (u_internal < acc)
