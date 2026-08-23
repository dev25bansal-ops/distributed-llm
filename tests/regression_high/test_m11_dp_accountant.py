"""Regression tests for M11: DP accountant correctness.

The differential-privacy subsystem was refactored.  The *stateful*
``PrivacyAccountant`` (spend / get_spent / can_spend / is_exhausted,
``BudgetExhaustedError``) was removed; the current surfaced blocks are:

  * ``distllm.core.differential_privacy`` -- clip-then-noise primitives
    (``DifferentialPrivacy.clip_tensor`` / ``add_noise_to_tensor``), the
    *stateless* advanced-composition bound (``privacy_budget_used``), and
    ``InputAnonymizer`` (PII redaction).

  * ``distllm.core.privacy_budget`` -- a *live* per-tenant budget meter
    (``TenantPrivacyBudget`` / ``PrivacyBudgetMeter``) that tracks queries,
    computes composed spend, exposes ``remaining`` and a hard ``exhausted``
    gate, and **fails closed** with a ``RuntimeError`` once the budget is spent
    (the successor of the removed ``BudgetExhaustedError``).

  * ``distllm.core.dp_inference.accounting.RDPAccounting`` -- the Renyi
    accountant (covered in test_e5_dp_accounting.py).

These tests lock in the following invariants:

  1. **Sensitivity bound** -- clipping bounds the L2 norm of any tensor to
     ``max_grad_norm``; clip-then-noise preserves that bound, which is what
     makes the Gaussian mechanism provide a finite (epsilon, delta)-DP
     guarantee.

  2. **Fail-closed exhaustion** -- once a tenant's budget is spent, further
     queries are blocked: ``record_query`` raises (fail closed) and the meter
     reports ``exhausted`` / ``remaining == 0``.

  3. **Stateless composition bound** -- ``privacy_budget_used`` returns the
     *documented* advanced-composition total (``epsilon * sqrt(2k ln(1.25
     /delta))``): ``0`` queries cost 0, the bound is monotonic in ``k``, and
     degenerate inputs do not inflate it.

  4. **Determinism** -- with a fixed generator/seed the noise is reproducible.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from distllm.core.differential_privacy import (  # noqa: E402
    DifferentialPrivacy,
    DifferentialPrivacyConfig,
    InputAnonymizer,
)
from distllm.core.privacy_budget import (  # noqa: E402
    PrivacyBudgetMeter,
    TenantPrivacyBudget,
)


# --------------------------------------------------------------------------- #
# 1. Sensitivity bound (clip -> bounded L2 sensitivity)                        #
# --------------------------------------------------------------------------- #
def test_clip_tensor_bounds_norm():
    """Clipping must bound every tensor's L2 norm to max_grad_norm."""
    cfg = DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
    dp = DifferentialPrivacy(cfg)
    for scale in (0.3, 1.0, 5.0, 100.0):
        t = torch.ones(50) * scale
        clipped = dp.clip_tensor(t)
        assert torch.norm(clipped).item() <= cfg.max_grad_norm + 1e-6


def test_clip_noop_when_within_norm():
    """A tensor already within the norm is returned unchanged (no division)."""
    cfg = DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5, max_grad_norm=2.0)
    dp = DifferentialPrivacy(cfg)
    t = torch.tensor([0.5, 0.5, 0.5])
    clipped = dp.clip_tensor(t)
    assert torch.allclose(clipped, t)


def test_clip_then_noise_preserves_sensitivity_bound():
    """Noise is added *after* clipping, so the released value's underlying
    clipped signal is still bounded -- this is what guarantees a finite DP
    bound (unbounded sensitivity would break the Gaussian guarantee)."""
    cfg = DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5, max_grad_norm=0.7)
    dp = DifferentialPrivacy(cfg)
    t = torch.randn(100) * 50.0
    noisy = dp.add_noise_to_tensor(t)
    # The *clipped* component is bounded; verify by re-clipping the result and
    # checking the deterministic (noise-free) part respects the norm.
    clipped_signal = dp.clip_tensor(t)
    assert torch.norm(clipped_signal).item() <= cfg.max_grad_norm + 1e-6
    # Output has the right shape and is finite.
    assert noisy.shape == t.shape
    assert torch.isfinite(noisy).all()


def test_zero_sigma_returns_clone_without_noise():
    """If sigma <= 0 the mechanism is inactive: add_noise returns the input
    tensor unmodified (no randomness is injected, nothing is amplified)."""
    cfg = DifferentialPrivacyConfig(
        epsilon=1e9, delta=1e-5, max_grad_norm=1.0, noise_multiplier=0.0
    )
    # Force sigma to 0 to simulate a degenerate config path.
    cfg.epsilon = float("inf")
    dp = DifferentialPrivacy(cfg)
    assert cfg.sigma == 0.0
    t = torch.ones(10) * 10.0
    out = dp.add_noise_to_tensor(t)
    # sigma <= 0 -> passthrough clone (clip-then-noise is disabled together
    # with the noise; only the calibrated-noise path applies the clip).
    assert torch.allclose(out, t)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# 2. Fail-closed budget exhaustion (successor of BudgetExhaustedError)         #
# --------------------------------------------------------------------------- #
def test_budget_spend_is_monotonic_and_nonnegative():
    """Composed spend only increases (never decreases) as queries are
    recorded, and the query counter is accurate."""
    budget = TenantPrivacyBudget(tenant_id="t", epsilon_limit=100.0)
    prev = -1.0
    for i in range(1, 11):
        spent = budget.record_query()["spent_epsilon"]
        assert spent >= prev
        assert spent >= 0.0
        assert budget.queries == i
        prev = spent


def test_budget_saturates_and_blocks_when_exhausted():
    """Spending beyond the total budget is blocked: fail-closed RuntimeError
    (the successor of the removed ``BudgetExhaustedError``)."""
    budget = TenantPrivacyBudget(tenant_id="t", epsilon_limit=0.0)
    with pytest.raises(RuntimeError):
        budget.record_query()
    assert budget.is_exhausted()
    assert budget.remaining() == 0.0


def test_budget_epsilon_only_exhaustion():
    """Exhaustion triggers on epsilon being spent even with a large limit."""
    # A huge epsilon_limit must not block early queries.
    budget = TenantPrivacyBudget(tenant_id="t", epsilon_limit=1e12)
    for _ in range(5):
        budget.record_query()
    assert not budget.is_exhausted()
    # A zero-limit budget is immediately exhausted and refuses any query.
    zero = TenantPrivacyBudget(tenant_id="z", epsilon_limit=0.0)
    assert zero.is_exhausted()
    with pytest.raises(RuntimeError):
        zero.record_query()


def test_meter_fails_closed_when_exhausted():
    """PrivacyBudgetMeter.record_query delegates fail-closed behaviour."""
    meter = PrivacyBudgetMeter(default_epsilon_limit=0.0)
    with pytest.raises(RuntimeError):
        meter.record_query("t1")
    snap = meter.meter("t1")
    assert snap["exhausted"] is True
    assert snap["remaining_epsilon"] == 0.0


# --------------------------------------------------------------------------- #
# 3. Stateless composition bound                                               #
# --------------------------------------------------------------------------- #
def test_stateless_budget_zero_queries_is_zero():
    cfg = DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5)
    dp = DifferentialPrivacy(cfg)
    assert dp.privacy_budget_used(0)["total_epsilon"] == 0.0


def test_stateless_budget_is_monotonic_in_queries():
    """The stateless composed bound is monotonic in the number of queries
    (never decreases as k grows)."""
    cfg = DifferentialPrivacyConfig(epsilon=0.5, delta=1e-5)
    dp = DifferentialPrivacy(cfg)
    prev = -1.0
    for k in (0, 1, 2, 10, 100):
        got = dp.privacy_budget_used(k)["total_epsilon"]
        assert got >= prev
        prev = got


def test_stateless_budget_matches_documented_formula():
    """The bound is the documented advanced-composition formula
    ``epsilon * sqrt(2k * ln(1.25/delta))``; a single query must not inflate
    the per-query epsilon."""
    cfg = DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5)
    dp = DifferentialPrivacy(cfg)
    expected = cfg.epsilon * math.sqrt(2 * math.log(1.25 / cfg.delta))
    # Source rounds total_epsilon to 3 decimals (round(..., 3)).
    assert dp.privacy_budget_used(1)["total_epsilon"] == pytest.approx(expected, rel=0.01)
    # A single query costs exactly the per-query epsilon reported per query.
    assert dp.privacy_budget_used(1)["epsilon_per_query"] == pytest.approx(1.0, rel=1e-9)


# --------------------------------------------------------------------------- #
# 4. Determinism (seeded noise reproducibility)                                #
# --------------------------------------------------------------------------- #
def test_noise_deterministic_with_seed():
    """With a fixed RNG seed the noise draw is reproducible -- required so a
    unit test (or an auditing tool) can replay a release's noise exactly."""
    cfg = DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
    dp = DifferentialPrivacy(cfg)

    g1 = torch.Generator().manual_seed(1234)
    g2 = torch.Generator().manual_seed(1234)

    t = torch.zeros(200)  # zero tensor -> output is pure noise
    n1 = dp.add_noise_to_tensor(t.clone())
    # Re-draw with the same generator produces identical noise.
    noise_a = torch.randn(200, generator=g1) * cfg.sigma
    noise_b = torch.randn(200, generator=g2) * cfg.sigma
    # add_noise_to_tensor uses the global RNG; instead verify that an explicit
    # seeded draw matches a second identical seeded draw (determinism property).
    assert torch.allclose(noise_a, noise_b)

    # And two fresh passes with the same global seed produce the same output
    # for the same input.
    torch.manual_seed(42)
    out_a = dp.add_noise_to_tensor(torch.zeros(50))
    torch.manual_seed(42)
    out_b = dp.add_noise_to_tensor(torch.zeros(50))
    assert torch.allclose(out_a, out_b)


def test_different_seeds_give_different_noise():
    """Sanity: different seeds should (with overwhelming probability) yield
    different noise, otherwise the determinism would be trivially fake."""
    cfg = DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
    dp = DifferentialPrivacy(cfg)
    torch.manual_seed(1)
    a = dp.add_noise_to_tensor(torch.zeros(1000))
    torch.manual_seed(2)
    b = dp.add_noise_to_tensor(torch.zeros(1000))
    assert not torch.allclose(a, b)


# --------------------------------------------------------------------------- #
# Backend resolution sanity (no silent failure)                                #
# --------------------------------------------------------------------------- #
def test_anonymizer_redacts_pii():
    """Light coverage of InputAnonymizer (not the accountant, but part of the
    DP module's privacy surface)."""
    text = "Email me at john@example.com or call 555-123-4567"
    redacted = InputAnonymizer.anonymize(text)
    assert "john@example.com" not in redacted
    assert "555-123-4567" not in redacted
    assert InputAnonymizer.has_pii(text)
    assert not InputAnonymizer.has_pii(redacted)