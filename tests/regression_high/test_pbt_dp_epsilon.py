"""Property-based tests: differential-privacy epsilon composition (RDP).

Real module under test:
  * ``distllm.core.dp_inference.accounting.RDPAccounting`` -- Renyi
    Differential Privacy accountant.  Adding more queries (``add_query``)
    accumulates RDP and must never reduce the composed (epsilon, delta) —
    "spending more never reduces the consumed budget."

Invariants checked:
  1. BASIC COMPOSITION (monotonicity): composed spend after two queries
     dominates each individual query (total RDP >= each component).
  2. SPEND MONOTONICITY: cumulative epsilon only increases as more queries are
     recorded (never decreases).
  3. BUDGET BOUND: composed epsilon scales monotonically with the number of
     queries (more queries -> strictly more spend) for a fixed sigma.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from hypothesis import given, settings
from hypothesis import strategies as st

# Load RDPAccounting directly from its file to bypass dp_inference/__init__.py,
# whose heavy engine chain pulls distllm.config.settings (partial under the
# fake-import test shim).  accounting.py itself is pure stdlib.  We load under
# the REAL dotted name so dataclass __module__ introspection resolves.
_SRC = Path(__file__).resolve().parents[2] / "src"
_spec = importlib.util.spec_from_file_location(
    "distllm.core.dp_inference.accounting",
    str(_SRC / "distllm" / "core" / "dp_inference" / "accounting.py"),
)
assert _spec and _spec.loader
_acc_mod = importlib.util.module_from_spec(_spec)
sys.modules["distllm.core.dp_inference.accounting"] = _acc_mod
_spec.loader.exec_module(_acc_mod)
RDPAccounting = _acc_mod.RDPAccounting


_PBT_SETTINGS = dict(max_examples=30, deadline=None)


def _sigma():
    return st.floats(0.5, 8.0, allow_nan=False, allow_infinity=False)


def _delta():
    return st.floats(1e-9, 1e-2, allow_nan=False, allow_infinity=False)


@settings(**_PBT_SETTINGS)
@given(sigma=_sigma(), delta=_delta())
def test_basic_composition_is_monotonic(sigma, delta):
    """Composed spend after two queries dominates each single query."""
    one = RDPAccounting()
    one.add_query(sigma)
    e1 = one.get_epsilon(delta)

    two = RDPAccounting()
    two.add_query(sigma)
    two.add_query(sigma)
    e2 = two.get_epsilon(delta)

    # Two queries never cost less privacy than one.
    assert e2 >= e1


@settings(**_PBT_SETTINGS)
@given(sigma=_sigma(), delta=_delta())
def test_spend_is_monotonically_non_decreasing(sigma, delta):
    """Cumulative epsilon never decreases as queries are recorded."""
    acc = RDPAccounting()
    prev_e = 0.0
    for _ in range(3):
        acc.add_query(sigma)
        cur_e = acc.get_epsilon(delta)
        assert cur_e >= prev_e, f"epsilon decreased: {prev_e} -> {cur_e}"
        prev_e = cur_e


@settings(**_PBT_SETTINGS)
@given(sigma=_sigma(), delta=_delta())
def test_privacy_budget_used_scales_with_queries(sigma, delta):
    """Composed epsilon is strictly monotonic with the number of queries."""
    n0, n1 = 1, 5
    a0 = RDPAccounting()
    for _ in range(n0):
        a0.add_query(sigma)
    a1 = RDPAccounting()
    for _ in range(n1):
        a1.add_query(sigma)
    assert a1.get_epsilon(delta) > a0.get_epsilon(delta)


@settings(**_PBT_SETTINGS)
@given(sigma=_sigma(), delta=_delta(), steps0=st.integers(1, 5), steps1=st.integers(10, 20))
def test_epsilon_for_monotonic_in_steps(sigma, delta, steps0, steps1):
    """Composed epsilon is monotonic in the number of steps."""
    e0 = RDPAccounting()
    for _ in range(steps0):
        e0.add_query(sigma)
    e1 = RDPAccounting()
    for _ in range(steps1):
        e1.add_query(sigma)
    assert e1.get_epsilon(delta) >= e0.get_epsilon(delta)