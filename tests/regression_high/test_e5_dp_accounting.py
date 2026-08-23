"""Regression tests for E5: DP epsilon-accounting.

The DP subsystem was refactored: the pluggable ``PrivacyAccountant`` / opacus /
Google ``dp_accounting`` backend was replaced by an in-house pure-Python Renyi
accountant (``distllm.core.dp_inference.accounting.RDPAccounting``).  These
tests must run WITHOUT any heavy DP accounting dependency installed -- they
exercise the pure-Python RDP path.

Locked invariants:

  1. **Sensitivity bound** -- ``clip_tensor`` bounds the L2 norm to
     ``max_grad_norm`` (the DP clipping that already exists).  This is what
     makes the Gaussian mechanism's finite (epsilon, delta) guarantee valid.

  2. **get_epsilon is well-defined on Gaussian params** -- returns a *finite*
     non-negative epsilon for valid (sigma, delta); epsilon grows monotonically
     with the number of queries (more queries -> more privacy spend).

  3. **Accountant reports composed epsilon** -- ``get_privacy_spent`` returns
     a composed epsilon that increases with each recorded query.

Regression history: ``test_epsilon_for_zero_steps_or_bad_params_is_zero``
asserted the removed ``PrivacyAccountant.epsilon_for`` returned ``0.0`` for
degenerate inputs.  ``RDPAccounting`` intentionally does *not* silently coerce
an invalid delta to ``0.0`` -- ``get_epsilon`` raises ``ValueError`` when
``delta <= 0`` (a math-domain constraint from ``log(delta)``), which is the
fail-safe behaviour we now lock in.
"""

from __future__ import annotations

import importlib.util
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
)
# Load RDPAccounting directly (accounting.py is pure stdlib) to bypass
# dp_inference/__init__.py whose engine chain pulls distllm.config.settings
# (partial under the fake-import test shim).
_ACC_FILE = _REPO_SRC / "distllm" / "core" / "dp_inference" / "accounting.py"
_acc_spec = importlib.util.spec_from_file_location(
    "distllm.core.dp_inference.accounting", str(_ACC_FILE)
)
assert _acc_spec and _acc_spec.loader
_acc_mod = importlib.util.module_from_spec(_acc_spec)
sys.modules["distllm.core.dp_inference.accounting"] = _acc_mod
_acc_spec.loader.exec_module(_acc_mod)
RDPAccounting = _acc_mod.RDPAccounting  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. Sensitivity bound (clipping) -- prerequisite for a finite DP guarantee    #
# --------------------------------------------------------------------------- #
def test_clip_tensor_bounds_norm():
    """Clipping must bound every tensor's L2 norm to max_grad_norm; this is
    the sensitivity bound the Gaussian mechanism relies on."""
    cfg = DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
    dp = DifferentialPrivacy(cfg)
    for scale in (0.3, 1.0, 5.0, 100.0):
        t = torch.ones(50) * scale
        clipped = dp.clip_tensor(t)
        assert torch.norm(clipped).item() <= cfg.max_grad_norm + 1e-6


def test_clip_supports_finite_dp_guarantee():
    """Without clipping the released signal's influence is unbounded, so no
    finite (eps, delta) holds.  After clipping, the norm is bounded -- the
    condition under which get_epsilon() can return a finite epsilon."""
    cfg = DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5, max_grad_norm=0.5)
    dp = DifferentialPrivacy(cfg)
    huge = torch.randn(200) * 1000.0
    clipped = dp.clip_tensor(huge)
    # Bounded sensitivity -> the Gaussian mechanism has a finite (eps,delta).
    assert torch.norm(clipped).item() <= cfg.max_grad_norm + 1e-6
    # The accountant resolves to a callable, pure-Python epsilon bound.
    acc = RDPAccounting()
    assert callable(acc.get_epsilon)


# --------------------------------------------------------------------------- #
# 2. get_epsilon: well-defined and monotonic on Gaussian params                #
# --------------------------------------------------------------------------- #
def test_get_epsilon_finite_on_gaussian_params():
    """get_epsilon returns a finite, non-negative epsilon for valid Gaussian
    params, with no external accounting library installed."""
    for sigma in (0.5, 1.0, 2.0, 5.0):
        for delta in (1e-6, 1e-5, 1e-4):
            acc = RDPAccounting()
            acc.add_query(sigma)
            eps = acc.get_epsilon(delta)
            assert math.isfinite(eps), (sigma, delta, eps)
            assert eps >= 0.0


def test_get_epsilon_monotonic_in_steps():
    """More queries must spend more privacy: epsilon is monotonically
    non-decreasing in the number of queries."""
    delta = 1e-5
    prev = -1.0
    for k in (1, 2, 10, 50, 200):
        acc = RDPAccounting()
        for _ in range(k):
            acc.add_query(sigma=1.0)
        eps = acc.get_epsilon(delta)
        assert eps >= prev
        prev = eps
    # Strictly increasing once we have at least one query.
    one = RDPAccounting()
    one.add_query(sigma=1.0)
    many = RDPAccounting()
    for _ in range(200):
        many.add_query(sigma=1.0)
    assert many.get_epsilon(delta) > one.get_epsilon(delta)


def test_get_epsilon_zero_queries_is_base_value():
    """A fresh accountant reports the base (delta-only) epsilon and it is
    finite.  Recording a zero-sigma query adds no RDP and must not increase
    the spend."""
    delta = 1e-5
    base = RDPAccounting().get_epsilon(delta)
    assert math.isfinite(base)

    acc = RDPAccounting()
    acc.add_query(sigma=0.0)  # RDP(0) == 0 -> no additional spend
    assert acc.get_epsilon(delta) == pytest.approx(base, rel=1e-9)


def test_get_epsilon_rejects_nonpositive_delta():
    """delta must be a probability in (0, 1); a degenerate delta fails closed
    (raises) rather than silently returning 0.0 (the old epsilon_for returned
    0.0 -- see module docstring)."""
    acc = RDPAccounting()
    acc.add_query(sigma=1.0)
    for bad in (0.0, -1e-6):
        with pytest.raises(ValueError):
            acc.get_epsilon(bad)


# --------------------------------------------------------------------------- #
# 3. Accountant reports composed epsilon after spends (per-query add_query)    #
# --------------------------------------------------------------------------- #
def test_accountant_reports_composed_epsilon_per_query():
    """Each add_query() accumulation is reflected in get_privacy_spent's
    composed epsilon, which grows monotonically; fields are well-formed."""
    acc = RDPAccounting()
    prev = -1.0
    for i in range(1, 21):
        acc.add_query(sigma=2.0)
        spent = acc.get_privacy_spent(1e-5)
        assert spent["orders_used"] == len(acc.orders)
        comp = spent["epsilon"]
        assert comp is not None
        assert math.isfinite(comp)
        # More queries => more privacy spent (monotonic by construction).
        assert comp >= prev
        prev = comp
        assert spent["delta"] == 1e-5


def test_accountant_spend_is_monotonic_with_real_queries():
    """Spending real per-query costs (mimicking N DP-SGD steps) still yields a
    composed epsilon that is finite and increases with N."""
    acc = RDPAccounting()
    epsilons = []
    for _ in range(10):
        acc.add_query(sigma=1.0)
        epsilons.append(acc.get_privacy_spent(1e-5)["epsilon"])
    assert all(math.isfinite(e) for e in epsilons)
    assert epsilons == sorted(epsilons)  # monotonic non-decreasing
    assert epsilons[-1] > epsilons[0]    # strictly increasing