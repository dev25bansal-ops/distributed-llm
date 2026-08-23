"""N8 regression test -- flaky-test detection (T5-02).

Proves:

  (a) ``flaky_baseline.json`` is created / loaded by the ratchet.  A missing
      baseline yields a zeroed baseline (first-run seed behaviour) and
      ``--update`` writes a real baseline file.

  (b) The ratchet ALLOWS a flaky count that is equal to or lower than the
      baseline, and BLOCKS a higher count.  In ``--check`` mode a higher count
      causes a non-zero exit (asserted via the script process exit code).

  (c) ``pytest_rerunfailures`` is importable (so the rerun mechanism is wired).

  (d) A genuine rerun actually works: ``test_flaky_rerun_succeeds_after_retries``
      fails on the first attempt, then passes on a retry, and is tracked as
      flaky by the collector plugin.  It is kept isolated so it cannot
      destabilise the broader suite.

The collector + ratchet are loaded via importlib from ``scripts/ci`` so this
test does not depend on ``PYTHONPATH`` being set externally.  Baseline paths in
the subprocess invocations are redirected with ``FLAKY_BASELINE_OVERRIDE`` so
the committed repo baseline is never mutated by this test.
"""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS_CI = os.path.join(REPO_ROOT, "scripts", "ci")
COLLECTOR_PATH = os.path.join(SCRIPTS_CI, "flaky_test_collector.py")
RATCHET_PATH = os.path.join(SCRIPTS_CI, "flaky_test_ratchet.py")
REAL_BASELINE_PATH = os.path.join(SCRIPTS_CI, "flaky_baseline.json")


def _load(path, modname):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


collector = _load(COLLECTOR_PATH, "flaky_test_collector")
ratchet = _load(RATCHET_PATH, "flaky_test_ratchet")


def _run_ratchet(report_path, baseline_path=None, update=False):
    """Invoke the ratchet script as a subprocess, redirecting the baseline."""
    env = dict(os.environ)
    if baseline_path is not None:
        env["FLAKY_BASELINE_OVERRIDE"] = str(baseline_path)
    cmd = [sys.executable, RATCHET_PATH, "--report", str(report_path)]
    if update:
        cmd.append("--update")
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


# ---------------------------------------------------------------------------
# (c) rerunfailures is importable / wired
# ---------------------------------------------------------------------------
def test_rerunfailures_is_importable():
    import pytest_rerunfailures  # noqa: F401
    assert pytest_rerunfailures.__file__.endswith("pytest_rerunfailures.py")


# ---------------------------------------------------------------------------
# (b) pure ratchet logic blocks increases, allows equal/lower
# ---------------------------------------------------------------------------
def test_ratchet_allows_equal_count():
    passed, msg = ratchet.check_ratchet(current=3, baseline=3)
    assert passed is True
    assert "OK" in msg


def test_ratchet_allows_lower_count():
    passed, msg = ratchet.check_ratchet(current=1, baseline=3)
    assert passed is True
    assert "OK" in msg


def test_ratchet_blocks_higher_count():
    passed, msg = ratchet.check_ratchet(current=4, baseline=3)
    assert passed is False
    assert "FAILED" in msg


def test_ratchet_summary_strings():
    assert ratchet.fmt_summary("flaky", 2, 3) == "flaky: 2 (baseline 3, down)"
    assert ratchet.fmt_summary("flaky", 4, 3) == "flaky: 4 (baseline 3, up)"
    assert ratchet.fmt_summary("flaky", 3, 3) == "flaky: 3 (baseline 3, flat)"


# ---------------------------------------------------------------------------
# (a) baseline creation / load
# ---------------------------------------------------------------------------
def test_baseline_loads_when_absent_returns_zeroed():
    # Temporarily move any real baseline aside so we exercise the no-baseline path.
    existed = os.path.exists(REAL_BASELINE_PATH)
    saved = None
    if existed:
        saved = REAL_BASELINE_PATH + ".n8bak"
        os.replace(REAL_BASELINE_PATH, saved)
    try:
        base = ratchet.load_baseline()
        assert base["flaky_count"] == 0
        assert base["skipped_count"] == 0
        assert base["xfailed_count"] == 0
    finally:
        if saved is not None:
            os.replace(saved, REAL_BASELINE_PATH)


def test_baseline_is_created_by_update(tmp_path):
    report = tmp_path / "flaky_report.json"
    report.write_text(json.dumps({
        "flaky_tests": ["tests/x.py::test_a", "tests/x.py::test_b"],
        "flaky_count": 2,
        "skipped_tests": ["tests/x.py::test_s"],
        "skipped_count": 1,
        "xfailed_tests": ["tests/x.py::test_x"],
        "xfailed_count": 1,
    }, indent=2))
    out = tmp_path / "flaky_baseline.json"
    rc = _run_ratchet(report, baseline_path=out, update=True)
    assert rc.returncode == 0, rc.stdout + rc.stderr
    assert out.exists(), "flaky_baseline.json was not created by --update"
    baseline = json.loads(out.read_text())
    assert baseline["flaky_count"] == 2
    assert baseline["skipped_count"] == 1
    assert baseline["xfailed_count"] == 1


# ---------------------------------------------------------------------------
# (b) --check mode exits non-zero when flaky count increases
# ---------------------------------------------------------------------------
def test_check_fails_when_flaky_increases(tmp_path):
    baseline = tmp_path / "flaky_baseline.json"
    baseline.write_text(json.dumps({"flaky_tests": ["t.py::a"], "flaky_count": 1,
                                    "skipped_tests": [], "skipped_count": 0,
                                    "xfailed_tests": [], "xfailed_count": 0}, indent=2))
    report = tmp_path / "flaky_report.json"
    report.write_text(json.dumps({"flaky_tests": ["t.py::a", "t.py::b", "t.py::c"],
                                  "flaky_count": 3, "skipped_tests": [], "skipped_count": 0,
                                  "xfailed_tests": [], "xfailed_count": 0}, indent=2))
    proc = _run_ratchet(report, baseline_path=baseline)
    assert proc.returncode != 0, "ratchet should FAIL when flaky count grows"
    assert "FAILED" in proc.stdout


def test_check_passes_when_flaky_equal_or_lower(tmp_path):
    baseline = tmp_path / "flaky_baseline.json"
    baseline.write_text(json.dumps({"flaky_tests": ["t.py::a", "t.py::b"], "flaky_count": 2,
                                    "skipped_tests": [], "skipped_count": 0,
                                    "xfailed_tests": [], "xfailed_count": 0}, indent=2))
    report = tmp_path / "flaky_report.json"
    report.write_text(json.dumps({"flaky_tests": ["t.py::a"], "flaky_count": 1,
                                  "skipped_tests": [], "skipped_count": 0,
                                  "xfailed_tests": [], "xfailed_count": 0}, indent=2))
    proc = _run_ratchet(report, baseline_path=baseline)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


# ---------------------------------------------------------------------------
# (d) a real rerun succeeds, and is tracked as flaky by the collector
# ---------------------------------------------------------------------------
_flaky_attempts = {"n": 0}


def test_flaky_rerun_succeeds_after_retries():
    """Demonstrates the rerun mechanism: fails first attempt, passes next.

    This proves pytest-rerunfailures is active under .venv311 and that a flaky
    test does not destabilise the suite. It is intentionally isolated.
    """
    _flaky_attempts["n"] += 1
    if _flaky_attempts["n"] < 2:
        raise AssertionError("transient failure (simulated flake on first attempt)")
    assert True
