"""Regression tests for spec_verify.verify_chain bounds bug (C-02).

When the target_logits tensor runs out of rows before all draft tokens
are verified, verify_chain previously executed `break` and fell through
to `return num`, asserting ALL draft tokens accepted — including the
never-verified positions. Fix: return the number actually verified (i).
"""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")

spec_verify = importlib.import_module("distllm.core.spec_verify")


def _make_logits(rows: int, vocab: int = 16, best_token: int = 3) -> "torch.Tensor":
    """Build a logits tensor where best_token is always the argmax."""
    logits = torch.full((1, rows, vocab), -10.0)
    logits[:, :, best_token] = 10.0
    return logits


def test_verify_chain_all_rows_available_accepts_all():
    """All 8 draft tokens have logits rows -> all accepted (greedy path)."""
    prefix = torch.tensor([[1, 2, 3]])
    drafts = torch.full((1, 8), 3)  # all best_token
    logits = _make_logits(rows=11)  # prefix(3) + 8 draft positions
    accepted = spec_verify.verify_chain(prefix, drafts, logits, temperature=0.0)
    assert accepted == 8


def test_verify_chain_bounds_break_returns_verified_count():
    """C-02: fewer logits rows than draft positions -> return verified count.

    Causal indexing: draft token i verifies against logits row (L-1)+i
    where L = prefix length. With L=3, rows=4: valid rows are 0..3, so
    (L-1)+i = 2+i must be < 4 -> i in {0, 1}. Exactly 2 verified.
    """
    prefix = torch.tensor([[1, 2, 3]])
    drafts = torch.full((1, 8), 3)
    logits = _make_logits(rows=4)
    accepted = spec_verify.verify_chain(prefix, drafts, logits, temperature=0.0)
    assert accepted == 2, (
        f"Expected 2 verified tokens, got {accepted}. "
        "If this returns 8, the C-02 bounds-break regression is back."
    )


def test_verify_chain_bounds_break_zero_rows():
    """No rows available beyond (L-1) -> 0 tokens verified, not num.

    With L=3 and rows=3, (L-1)+i = 2+i < 3 -> only i=0 fits... no:
    2 < 3 is true for i=0, so 1 token verifies. Use rows=2 so even
    row (L-1)=2 is out of bounds -> 0 verified.
    """
    prefix = torch.tensor([[1, 2, 3]])
    drafts = torch.full((1, 4), 3)
    logits = _make_logits(rows=2)  # last usable row index 1 < (L-1)=2
    accepted = spec_verify.verify_chain(prefix, drafts, logits, temperature=0.0)
    assert accepted == 0


def test_verify_chain_partial_bounds_exact_boundary():
    """Exactly enough rows for half the drafts -> return exactly half.

    L=3, rows=5: (L-1)+i = 2+i < 5 -> i in {0,1,2} -> 3 verified.
    """
    prefix = torch.tensor([[1, 2, 3]])
    drafts = torch.full((1, 6), 3)
    logits = _make_logits(rows=5)
    accepted = spec_verify.verify_chain(prefix, drafts, logits, temperature=0.0)
    assert accepted == 3


def test_verify_chain_rejection_still_returns_rejection_index():
    """A rejected token still terminates at its index (not affected by fix)."""
    prefix = torch.tensor([[1, 2, 3]])
    drafts = torch.tensor([[3, 3, 7, 3]])  # token 7 will be rejected
    logits = _make_logits(rows=7, best_token=3)
    accepted = spec_verify.verify_chain(prefix, drafts, logits, temperature=0.0)
    # i=0: pos=3 token 3 accepted; i=1: pos=4 token 3 accepted;
    # i=2: pos=5 token 7 rejected (argmax is 3) -> return 2
    assert accepted == 2
