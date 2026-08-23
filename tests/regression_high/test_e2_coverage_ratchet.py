"""
E2 regression test — the coverage RATCHET is the single coverage gate.

Asserts that:

  (1) ``check_baseline`` (the pure ratchet comparison) FAILS when current
      coverage drops below ``baseline - tolerance`` and PASSES when it is
      at or above it.

  (2) The ratchet IGNORES any hard 80% floor.  A baseline of 70% with
      current 71% PASSES even though 71% < 80% — coverage is allowed to sit
      below 80% as long as it does not regress.  This is the exact
      contradiction that was removed (the old ``--cov-fail-under=80`` /
      ``fail_under = 80`` over a truncated graph).

  (3) ``fmt_summary`` produces the per-PR delta markdown line
      ``Coverage: 73.2% (Δ +1.4% vs baseline 71.8%)``.

  (4) The CI workflows no longer pass a hard ``--cov-fail-under`` floor and
      instead invoke ``scripts/ci/coverage_ratchet.py`` as the gate
      (the contradictory setting was removed).

The ratchet comparison logic is intentionally pure (no file IO) so this
test never needs to run the full suite or generate coverage.xml.
"""

import importlib.util
import os

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "ci", "coverage_ratchet.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("coverage_ratchet", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cr = _load_module()


# ---------------------------------------------------------------------------
# Pure ratchet comparison
# ---------------------------------------------------------------------------
def test_baseline_regression_fails():
    """current < baseline - tolerance  -> FAIL."""
    passed, msg = cr.check_baseline(current=69.0, baseline=71.8, tolerance=0.0)
    assert passed is False
    assert "FAILED" in msg


def test_baseline_regression_within_tolerance_passes():
    """current >= baseline - tolerance  -> PASS."""
    passed, msg = cr.check_baseline(current=71.0, baseline=71.8, tolerance=1.0)
    assert passed is True
    assert "OK" in msg


def test_baseline_at_threshold_passes():
    """Boundary: current exactly == baseline - tolerance passes."""
    passed, _ = cr.check_baseline(current=70.5, baseline=71.8, tolerance=1.3)
    assert passed is True


# ---------------------------------------------------------------------------
# The whole point: ratchet ignores the hard 80% floor
# ---------------------------------------------------------------------------
def test_ratchet_ignores_hard_80_floor():
    """Baseline 70%, current 71% PASSES even though < 80%."""
    passed, msg = cr.check_baseline(current=71.0, baseline=70.0, tolerance=0.0)
    assert passed is True
    # Explicitly assert current is below the old hard floor of 80%.
    assert 71.0 < 80.0


def test_ratchet_below_floor_but_above_baseline_passes():
    """Coverage far below 80% still passes if it didn't regress vs baseline."""
    passed, _ = cr.check_baseline(current=55.0, baseline=54.0, tolerance=0.0)
    assert passed is True
    assert 55.0 < 80.0


def test_ratchet_only_blocks_regressions():
    """Any regression vs baseline fails, regardless of absolute level."""
    passed, _ = cr.check_baseline(current=79.0, baseline=82.0, tolerance=0.0)
    assert passed is False  # 79% < 82% baseline even though it looks "high"


# ---------------------------------------------------------------------------
# Per-PR delta reporting
# ---------------------------------------------------------------------------
def test_fmt_summary_positive_delta():
    assert cr.fmt_summary(73.2, 71.8) == "Coverage: 73.2% (Δ +1.4% vs baseline 71.8%)"


def test_fmt_summary_negative_delta():
    assert cr.fmt_summary(70.4, 71.8) == "Coverage: 70.4% (Δ -1.4% vs baseline 71.8%)"


def test_fmt_delta_sign():
    assert cr.fmt_delta(72.0, 71.0) == "+1.0%"
    assert cr.fmt_delta(70.0, 71.0) == "-1.0%"
    assert cr.fmt_delta(71.0, 71.0) == "+0.0%"


# ---------------------------------------------------------------------------
# Contradictory setting removed from CI config
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("wf", ["ci.yml", "quality-gates.yml"])
def test_no_hard_cov_fail_under_in_ci(wf):
    """The hard pytest-cov floor must be gone; ratchet is the gate.

    We assert no *pytest invocation* still passes ``--cov-fail-under``.
    (Explanatory comments mentioning the old flag are allowed.)
    """
    text = open(os.path.join(REPO_ROOT, ".github", "workflows", wf)).read()
    assert "pytest" in text.lower()
    # A real usage would be a `run:` line invoking pytest with --cov-fail-under.
    import re
    real_usage = re.search(r"pytest\b[^\n]*--cov-fail-under", text)
    assert real_usage is None, f"{wf} still passes --cov-fail-under to pytest"
    # And the ratchet script must be wired in as the gate.
    assert "coverage_ratchet.py" in text, f"{wf} does not invoke the ratchet"


def test_pyproject_has_no_fail_under():
    """pyproject.toml must not impose a hard coverage floor setting."""
    text = open(os.path.join(REPO_ROOT, "pyproject.toml")).read()
    # A real setting is `fail_under = 80` on its own logical line, not a
    # comment. Strip comment lines, then assert no `fail_under` remains.
    code = "\n".join(
        line.split("#", 1)[0] for line in text.splitlines()
    )
    assert "fail_under" not in code, "pyproject.toml still sets fail_under"
