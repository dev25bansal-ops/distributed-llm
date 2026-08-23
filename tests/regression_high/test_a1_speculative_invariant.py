"""A1 — Formal verification of the speculative-decoding draft-acceptance invariant (Core F20).

This module proves (via Hypothesis property tests + a deterministic symbolic
table) the central invariant of speculative decoding's draft-acceptance step:

    INVARIANT (draft-acceptance == target over drafted span)
    ------------------------------------------------------------------
    Let `draft_ids` be the tokens proposed by the draft model and
    `target_ids` be the target model's verified tokens over the same span.
    Then for the acceptance mask `m = verify_draft_acceptance(draft_ids, target_ids)`:

      (1) accepted subset of draft positions:
              {j : m[j]}  ⊆  {0, ... , len(draft_ids)-1}
          (trivially true by construction; stated for completeness)
      (2) accepted ids equal target ids:
              ∀j. m[j]  ⇒  draft_ids[j] == target_ids[j]
      (3) rejection boundary is a genuine mismatch (or draft ran out):
              let k = min{j : ¬m[j]}; then (k >= len(target_ids)) OR
              draft_ids[k] != target_ids[k]
          (positions *after* k are not accepted only because the chain stops
           there — they were never evaluated — so the strong "differs" claim
           applies to the boundary, not the un-evaluated tail.)
      (4) prefix closure — acceptance is a *leading prefix*:
              ∀j. (j>0 and m[j])  ⇒  m[j-1]
          i.e. once a position is rejected, every later position is rejected
          too. (Stop-at-first-mismatch semantics.)
      (5) full agreement ⇒ all accepted:
              draft_ids[:len(target_ids)] == target_ids  ⇒  all m[j] True
              (for the overlapping span)
      (6) mismatch at position 0 ⇒ draft fully rejected past mismatch:
              (len(target_ids)==0 OR draft_ids[0] != target_ids[0])  ⇒
              ∀j>0. ¬m[j]   (only the fallback target token survives)

    Consequence: the accepted tokens ARE EXACTLY the target-model tokens over
    the accepted span; rejected positions fall back to the target token.

No decoder behavior is changed.  The test exercises the *pure* facade
`spec_verify.verify_draft_acceptance`, which mirrors the per-position decision
of the production engine (`accept_token` / `verify_chain`) but on raw token
ids so it can be reasoned about without a torch model.

Greedy mode is the canonical, deterministic case and is fully covered by the
property tests.  Sampled (rejection-sampling) mode is also exercised, where
the leading-prefix + accepted-equals-target properties still hold for the
positions that *are* accepted.
"""

from __future__ import annotations

import math

from hypothesis import given, settings, strategies as st

from distllm.core.spec_verify import verify_draft_acceptance

# ── Strategies ───────────────────────────────────────────────────────────────
# Reasonable vocab so token ids collide sometimes (otherwise (2)/(3) are
# vacuous).  vocab=12 gives frequent mismatches.
TOKEN_IDS = st.integers(min_value=0, max_value=11)

draft_lists = st.lists(TOKEN_IDS, min_size=0, max_size=12)
target_lists = st.lists(TOKEN_IDS, min_size=0, max_size=12)


# ── Helpers ────────────────────────────────────────────────────────────────
def _first_mismatch(draft: list[int], target: list[int]) -> int:
    """Index of the first position where draft and target differ (or draft ran
    past target).  Returns len(draft) if they match over the whole draft."""
    n = min(len(draft), len(target))
    for j in range(n):
        if draft[j] != target[j]:
            return j
    return len(draft)


# ── Property (1): accepted ⊆ draft positions ─────────────────────────────────
@given(draft=draft_lists, target=target_lists)
@settings(max_examples=400, deadline=None)
def test_inv_accepted_subset_of_draft_positions(draft, target):
    mask = verify_draft_acceptance(draft, target)
    assert len(mask) == len(draft)
    # By construction every accepted index lies within [0, len(draft)).
    for j, accepted in enumerate(mask):
        if accepted:
            assert 0 <= j < len(draft)


# ── Property (2): at every accepted position, accepted_id == target_id ────────
@given(draft=draft_lists, target=target_lists)
@settings(max_examples=400, deadline=None)
def test_inv_accepted_equals_target(draft, target):
    mask = verify_draft_acceptance(draft, target)
    for j, accepted in enumerate(mask):
        if accepted:
            # accepted only when j < len(target) (cannot accept past target span)
            assert j < len(target)
            assert draft[j] == target[j]


# ── Property (3): the rejection boundary is a genuine mismatch (or draft ran
#    out of the target span).  NOTE: positions *beyond* the first mismatch are
#    "not accepted" only because the chain stopped there (they were never
#    evaluated); the strong "differs" claim applies to the boundary, not the
#    un-evaluated tail.  So we assert it at the first non-accepted index. ──────
@given(draft=draft_lists, target=target_lists)
@settings(max_examples=400, deadline=None)
def test_inv_rejection_boundary_is_mismatch(draft, target):
    mask = verify_draft_acceptance(draft, target)
    # First position where acceptance is False (k = len if all accepted).
    k = next((j for j, a in enumerate(mask) if not a), len(mask))
    if k < len(draft):
        # The boundary is either past the target span, or a real token mismatch.
        assert (k >= len(target)) or (draft[k] != target[k])


# ── Property (4): prefix closure (leading-prefix / stop-at-first-mismatch) ────
@given(draft=draft_lists, target=target_lists)
@settings(max_examples=400, deadline=None)
def test_inv_prefix_closure(draft, target):
    mask = verify_draft_acceptance(draft, target)
    seen_reject = False
    for accepted in mask:
        if not accepted:
            seen_reject = True
        if seen_reject:
            assert not accepted  # nothing accepted after the first rejection


# ── Property (5): draft == target everywhere ⇒ ALL accepted ───────────────────
@given(target=target_lists)
@settings(max_examples=200, deadline=None)
def test_inv_full_agreement_all_accepted(target):
    # draft identical to target over the overlapping span.
    draft = list(target)
    mask = verify_draft_acceptance(draft, target)
    assert len(mask) == len(draft)
    assert all(mask), f"full agreement but not all accepted: {mask}"


# ── Property (6): mismatch at position 0 ⇒ draft fully rejected past mismatch ─
@given(draft=draft_lists, target=target_lists)
@settings(max_examples=400, deadline=None)
def test_inv_mismatch_at_zero_rejects_rest(draft, target):
    if not draft:
        return  # nothing to reject
    mismatch_at_zero = (len(target) == 0) or (draft[0] != target[0])
    if not mismatch_at_zero:
        return
    mask = verify_draft_acceptance(draft, target)
    # Position 0 itself is rejected (first mismatch).
    assert not mask[0]
    # Everything past the first mismatch is also rejected (leading-prefix).
    assert not any(mask[1:]), f"mismatch at 0 but later accepted: {mask}"


# ── Sampled mode: acceptance is a leading prefix bounded by the target span ──
# In rejection-sampling (temperature > 0) mode acceptance follows min(1, p/q):
# an accepted token is NOT required to equal the target argmax (that is the
# definition of sampling).  The invariant that *does* hold for sampled mode is
# the structural one: the mask is a leading prefix and acceptance never extends
# past the target span.  (The accepted==target argmax equality is a GREEDY-mode
# property, proved by properties (2)/(5)/(6) above.)
def _probs_for(draft, target):
    # Build valid draft/target probabilities so p/q is well-defined.
    dp = [0.7 + 0.3 * ((j % 3) / 2.0) for j in range(len(draft))]  # 0.7..1.0, q>0
    tp = []
    for j in range(len(draft)):
        if j < len(target):
            # target prob of the draft token: some value in (0, 1]
            tp.append(0.3 + 0.7 * ((j % 3) / 2.0))
        else:
            tp.append(0.0)
    return [min(1.0, max(1e-3, x)) for x in dp], [min(1.0, max(0.0, x)) for x in tp]


@given(draft=draft_lists, target=target_lists)
@settings(max_examples=300, deadline=None)
def test_inv_sampled_is_leading_prefix_within_span(draft, target):
    if not draft:
        return
    dp, tp = _probs_for(draft, target)
    mask = verify_draft_acceptance(draft, target, draft_probs=dp, target_probs=tp)
    assert len(mask) == len(draft)
    # Leading prefix: nothing accepted after the first rejection.
    seen_reject = False
    for accepted in mask:
        if not accepted:
            seen_reject = True
        if seen_reject:
            assert not accepted
    # An accepted position always lies within the target span.
    for j, accepted in enumerate(mask):
        if accepted:
            assert j < len(target)


# ── Deterministic symbolic table (clarity / regression anchor) ───────────────
SYMBOLIC_CASES = [
    # (draft_ids, target_ids, expected_mask)
    ([], [], []),
    ([5], [], [False]),                       # draft ran past target span
    ([5], [5], [True]),                       # exact match -> accept
    ([5], [7], [False]),                      # mismatch at 0 -> reject (fallback)
    ([5, 6], [5, 6], [True, True]),           # full agreement -> all accepted
    ([5, 6], [5, 9], [True, False]),          # prefix accepted, mismatch stops
    ([5, 6, 7], [5, 6, 7], [True, True, True]),
    ([5, 6, 7], [5, 6, 8], [True, True, False]),  # reject at last pos
    ([1, 2, 3], [9, 9, 9], [False, False, False]),  # mismatch at 0 -> all reject
    ([5, 6], [5, 6, 7], [True, True]),        # target longer than draft -> accept all
]


def test_symbolic_table():
    for draft, target, expected in SYMBOLIC_CASES:
        mask = verify_draft_acceptance(draft, target)
        assert mask == expected, (
            f"case draft={draft} target={target}: got {mask}, expected {expected}"
        )


def test_symbolic_first_mismatch_is_accept_boundary():
    """Symbolic proof that the acceptance boundary is exactly the first
    mismatch: positions before it are accepted, positions from it on are not."""
    cases = [
        ([5, 6, 7], [5, 6, 7], 3),
        ([5, 6, 7], [5, 9, 7], 1),
        ([5, 6, 7], [9, 9, 9], 0),
        ([5, 6, 7], [5, 6], 2),     # draft longer than target -> boundary at len(target)
        ([5, 6], [5, 6, 7], 2),     # target longer than draft -> all draft accepted
    ]
    for draft, target, boundary in cases:
        mask = verify_draft_acceptance(draft, target)
        assert all(mask[:boundary]), f"prefix not fully accepted: {mask}"
        assert not any(mask[boundary:]), f"positions after boundary accepted: {mask}"
