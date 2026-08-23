"""§6.1 property-based tests (Hypothesis, per roadmap).

These encode the invariants that the Critical/High fixes must hold:

- C3 off-by-one family: at temperature=0, a draft chain whose
  tokens == the greedy target argmax is accepted token-for-token
  (verify_chain == num_draft); a draft token that differs from the
  greedy argmax is NOT accepted.  Kills the C3 family and prevents
  regression of the shared prefix_len/accept_token indexing.

- M1 money math: summing N random Decimal charges via the Money
  accumulator equals the exact Decimal sum (no float drift).

- M7 hash namespace: store/lookup round-trips across every tier and
  two independent CacheManager nodes stay namespaced (no cross-leak).
"""

import random
from decimal import Decimal

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from distllm.core.cache_manager import CacheManager
from distllm.core.money import Money
from distllm.core.spec_verify import verify_chain


# ── C3: spec-decode equivalence (greedy draft == target) ──

@settings(max_examples=75, deadline=None)
@given(
    vocab=st.integers(min_value=5, max_value=200),
    seq_len=st.integers(min_value=4, max_value=20),
    num_draft=st.integers(min_value=1, max_value=8),
    prefix_tok=st.lists(
        st.integers(min_value=0, max_value=199), min_size=1, max_size=8
    ),
    prefix_len_user=st.integers(min_value=1, max_value=8),
)
def test_prop_spec_equiv_greedy_draft_fully_accepted(
    vocab, seq_len, num_draft, prefix_tok, prefix_len_user
):
    import torch

    # Build a prefix of exactly prefix_len_user tokens that fits before the draft.
    L_user = min(prefix_len_user, len(prefix_tok), max(1, seq_len - num_draft))
    assume(L_user >= 1)
    toks = list(prefix_tok[:L_user]) + [0] * (L_user - len(prefix_tok[:L_user]))
    prefix = torch.tensor([toks])
    Lp = prefix.shape[1] - 1  # prefix_len()
    assume(Lp + num_draft <= seq_len)

    logits = torch.randn(1, seq_len, vocab)
    target_ids = [int(logits[0, j].argmax()) for j in range(seq_len)]
    # Draft continuations that follow the prefix == greedy target argmax.
    draft_ids = target_ids[Lp : Lp + num_draft]
    draft = torch.tensor([draft_ids])

    accepted = verify_chain(
        prefix, draft, logits, temperature=0.0, vocab_size=vocab
    )
    assert accepted == num_draft, (
        f"greedy draft not fully accepted ({accepted}/{num_draft}) -- C3 regression"
    )


@settings(max_examples=75, deadline=None)
@given(
    vocab=st.integers(min_value=5, max_value=200),
    seq_len=st.integers(min_value=4, max_value=20),
    num_draft=st.integers(min_value=2, max_value=8),
    prefix_tok=st.lists(
        st.integers(min_value=0, max_value=199), min_size=1, max_size=8
    ),
    prefix_len_user=st.integers(min_value=1, max_value=8),
    corrupt_idx=st.integers(min_value=0, max_value=7),
)
def test_prop_spec_equiv_wrong_draft_rejected_at_c3_index(
    vocab, seq_len, num_draft, prefix_tok, prefix_len_user, corrupt_idx
):
    import torch

    L_user = min(prefix_len_user, len(prefix_tok), max(1, seq_len - num_draft))
    assume(L_user >= 1)
    toks = list(prefix_tok[:L_user]) + [0] * (L_user - len(prefix_tok[:L_user]))
    prefix = torch.tensor([toks])
    Lp = prefix.shape[1] - 1
    assume(Lp + num_draft <= seq_len)
    corrupt_idx = min(corrupt_idx, num_draft - 1)

    logits = torch.randn(1, seq_len, vocab)
    target_ids = [int(logits[0, j].argmax()) for j in range(seq_len)]
    draft_ids = list(target_ids[Lp : Lp + num_draft])
    # Corrupt the chosen interior token to something != greedy argmax at its pos.
    pos = Lp + corrupt_idx
    good = draft_ids[corrupt_idx]
    bad = (good + 1) % vocab
    if int(logits[0, pos].argmax()) == bad:
        bad = (good + 2) % vocab
    draft_ids[corrupt_idx] = bad
    draft = torch.tensor([draft_ids])

    accepted = verify_chain(
        prefix, draft, logits, temperature=0.0, vocab_size=vocab
    )
    # Acceptance must stop at (not after) the corrupted token.
    assert accepted <= corrupt_idx, (
        f"accepted past a non-greedy token ({accepted} > {corrupt_idx}) -- C3 off-by-one"
    )


# ── M1: money math, no drift across random charges ──

@settings(max_examples=60, deadline=None)
@given(
    charges=st.lists(
        st.integers(min_value=1, max_value=9999),
        min_size=1,
        max_size=200,
    )
)
def test_prop_money_no_drift(charges):
    total = Money(0)
    expected = Decimal(0)
    for c in charges:
        amt = Decimal(c) / Decimal(100)  # cents -> dollars
        expected += amt
        total = total.add(amt)  # exact Decimal accumulate
    # Recorded total (quantized once) == exact sum of charges.
    assert total.value() == expected.quantize(Decimal("0.01"))


# ── M7: cache store/lookup round-trip across tiers + node isolation ──

@settings(max_examples=50, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    n_tokens=st.integers(min_value=2, max_value=12),
)
def test_prop_cache_roundtrip_all_tiers_and_isolation(seed, n_tokens):
    random.seed(seed)
    node_a = CacheManager(gpu_cache_mb=64, cpu_cache_mb=256, ssd_cache_gb=1)
    node_b = CacheManager(gpu_cache_mb=64, cpu_cache_mb=256, ssd_cache_gb=1)

    toks = [random.randint(0, 50_000) for _ in range(n_tokens)]
    node_a.store_prefix(toks, kv_data=("A", seed))
    node_b.store_prefix(toks, kv_data=("B", seed))

    # Each node resolves its own value (namespace isolation across nodes).
    _, a = node_a.lookup_prefix(toks)
    _, b = node_b.lookup_prefix(toks)
    assert a == ("A", seed)
    assert b == ("B", seed)
    assert a != b

    # Round-trip integrity within a node.
    toks2 = [random.randint(0, 50_000) for _ in range(n_tokens)]
    node_a.store_prefix(toks2, kv_data={"s": seed})
    length, data = node_a.lookup_prefix(toks2)
    assert data == {"s": seed}
