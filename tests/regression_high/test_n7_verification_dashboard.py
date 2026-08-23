"""Regression test N7 — verification report dashboard tab.

Proves the additive per-(model, partition) verification history store and its
wiring into the dashboard MetricsCollector:

(a) VerificationHistoryStore.record + history returns entries;
(b) logit_cosine for identical logits == 1.0 and orthogonal == ~0;
(c) token_match is computed correctly;
(d) MetricsCollector integrates the store without breaking the existing
    spec_acceptance_rate().

Model-free: all logit tensors are constructed by hand.
"""

from __future__ import annotations

import math

import torch

from distllm.dashboard.verification_history import (
    VerificationHistoryStore,
    compute_logit_cosine,
)
from distllm.dashboard.ws_handler import MetricsCollector


# ---------------------------------------------------------------------------
# (a) record + history
# ---------------------------------------------------------------------------

def test_record_and_history_returns_entries():
    store = VerificationHistoryStore()
    d = torch.tensor([1.0, 2.0, 3.0])
    t = torch.tensor([1.0, 2.0, 3.0])

    e1 = store.record("m1", "0-15", d, t, draft_token_id=7, target_token_id=7)
    e2 = store.record("m1", "0-15", d, t, draft_token_id=3, target_token_id=9)

    assert e1["model"] == "m1"
    assert e1["partition"] == "0-15"
    assert "timestamp" in e1

    hist = store.history("m1", "0-15")
    assert len(hist) == 2
    assert hist[0] is not None and hist[1]["token_match"] == 0

    # Windowing returns only the most recent N.
    assert len(store.history("m1", "0-15", window=1)) == 1

    # Unknown key -> empty list, no crash.
    assert store.history("nope", "nope") == []


def test_history_keys_and_partition_isolation():
    store = VerificationHistoryStore()
    d = torch.tensor([1.0, 0.0])
    store.record("mA", "0-3", d, d, 1, 1)
    store.record("mA", "4-7", d, d, 1, 1)
    store.record("mB", "0-3", d, d, 1, 1)

    keys = store.keys()
    assert ("mA", "0-3") in keys
    assert ("mA", "4-7") in keys
    assert ("mB", "0-3") in keys
    assert len(store.history("mA", "0-3")) == 1
    assert len(store.history("mA", "4-7")) == 1


# ---------------------------------------------------------------------------
# (b) logit cosine: identical == 1.0, orthogonal == ~0
# ---------------------------------------------------------------------------

def test_logit_cosine_identical_is_one():
    v = torch.tensor([0.5, -1.2, 3.3, 0.1])
    assert math.isclose(compute_logit_cosine(v, v.clone()), 1.0, abs_tol=1e-5)


def test_logit_cosine_orthogonal_is_zero():
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0])
    cos = compute_logit_cosine(a, b)
    assert abs(cos) < 1e-5


def test_logit_cosine_opposite_is_negative_one():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([-1.0, -2.0, -3.0])
    assert math.isclose(compute_logit_cosine(a, b), -1.0, abs_tol=1e-5)


def test_logit_cosine_guards_none_and_shape_mismatch():
    a = torch.tensor([1.0, 2.0])
    assert compute_logit_cosine(None, a) is None
    assert compute_logit_cosine(a, None) is None
    assert compute_logit_cosine(a, torch.tensor([1.0, 2.0, 3.0])) is None
    # Multi-dim tensors are flattened; identical flattened -> 1.0
    m = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert math.isclose(compute_logit_cosine(m, m.clone()), 1.0, abs_tol=1e-5)


def test_record_stores_cosine_and_guards_none_logits():
    store = VerificationHistoryStore()
    v = torch.tensor([1.0, 1.0, 1.0])
    e = store.record("m", "p", v, v.clone(), 5, 5)
    assert math.isclose(e["logit_cosine"], 1.0, abs_tol=1e-5)

    e2 = store.record("m", "p", None, v, 5, 5)
    assert e2["logit_cosine"] is None


# ---------------------------------------------------------------------------
# (c) token_match correctness
# ---------------------------------------------------------------------------

def test_token_match_correct():
    store = VerificationHistoryStore()
    v = torch.tensor([1.0, 2.0])
    match = store.record("m", "p", v, v.clone(), draft_token_id=42, target_token_id=42)
    nomatch = store.record("m", "p", v, v.clone(), draft_token_id=42, target_token_id=7)
    assert match["token_match"] == 1
    assert nomatch["token_match"] == 0

    # None token ids -> no match.
    none_case = store.record("m", "p", v, v.clone(), draft_token_id=None, target_token_id=3)
    assert none_case["token_match"] == 0


def test_acceptance_defaults_to_token_match_and_explicit_override():
    store = VerificationHistoryStore()
    v = torch.tensor([1.0, 2.0])
    default_case = store.record("m", "p", v, v, 1, 1)
    assert default_case["acceptance"] == 1  # defaults to token_match

    override = store.record("m", "p", v, v, 1, 1, acceptance=False)
    assert override["acceptance"] == 0  # explicit override wins


def test_aggregate_rolling_stats():
    store = VerificationHistoryStore()
    v = torch.tensor([1.0, 0.0, 0.0])
    ortho = torch.tensor([0.0, 1.0, 0.0])
    store.record("m", "p", v, v.clone(), 1, 1)       # cos 1.0, match
    store.record("m", "p", v, ortho, 1, 2)           # cos 0.0, no match
    agg = store.aggregate("m", "p")
    assert agg["count"] == 2
    assert math.isclose(agg["mean_logit_cosine"], 0.5, abs_tol=1e-5)
    assert math.isclose(agg["token_match_rate"], 0.5, abs_tol=1e-6)

    empty = store.aggregate("x", "y")
    assert empty["count"] == 0
    assert empty["mean_logit_cosine"] is None


# ---------------------------------------------------------------------------
# (d) MetricsCollector integration without breaking spec_acceptance_rate
# ---------------------------------------------------------------------------

def test_metrics_collector_integrates_store_without_breaking_spec():
    mc = MetricsCollector()

    # Existing spec path still works.
    mc.record_speculative(draft_count=10, accepted_count=6)
    assert math.isclose(mc.spec_acceptance_rate(), 0.6, abs_tol=1e-6)

    # New verification path records into the store.
    v = torch.tensor([1.0, 2.0, 3.0])
    entry = mc.record_verification("m1", "0-15", v, v.clone(), 7, 7, acceptance=True)
    assert entry["token_match"] == 1
    assert math.isclose(entry["logit_cosine"], 1.0, abs_tol=1e-5)

    snap = mc.verification_history()
    assert any(s["model"] == "m1" and s["partition"] == "0-15" for s in snap)

    # Recording an accepted verification folded into spec counters (11 drafts, 7 accepted).
    assert math.isclose(mc.spec_acceptance_rate(), 7 / 11, abs_tol=1e-6)


def test_summary_includes_verification_history():
    mc = MetricsCollector()
    v = torch.tensor([1.0, 2.0])
    mc.record_verification("m", "p", v, v.clone(), 1, 1)
    summary = mc.summary()
    assert "verification_history" in summary
    assert "speculative" in summary  # existing key still present
    assert isinstance(summary["verification_history"], list)


def test_verification_recording_without_acceptance_does_not_touch_spec():
    mc = MetricsCollector()
    mc.record_speculative(4, 2)
    before = mc.spec_acceptance_rate()
    v = torch.tensor([1.0, 2.0])
    # No acceptance kwarg -> spec counters untouched.
    mc.record_verification("m", "p", v, v.clone(), 1, 1)
    assert math.isclose(mc.spec_acceptance_rate(), before, abs_tol=1e-9)
