"""
E1 regression test — chaos / perf / resilience CI gates must be BLOCKING on push:main.

Asserts that the resilience, performance, and load gates:
  (1) are triggered on `push: branches: [main]` (and/or pull_request),
  (2) do NOT use `continue-on-error: true` on the gate step that runs the
      actual chaos/perf/load tests, and
  (3) are not silently padded with `|| true` so a failure passes anyway.

We parse the workflow YAML (PyYAML) and additionally keep robust
grep-based assertions so the test does not depend on subtle YAML structure.
"""

import os

import pytest
import yaml

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
WORKFLOWS = os.path.join(REPO_ROOT, ".github", "workflows")


def _load(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    # PyYAML 1.1 parses the YAML `on:` key as the boolean True.
    # Normalise so we can look it up as either "on" or True.
    on = data.get("on", data.get(True))
    return data, on


def _has_push_main(on):
    if not isinstance(on, dict):
        return False
    push = on.get("push")
    if not push:
        return False
    branches = push.get("branches") if isinstance(push, dict) else None
    if branches is None and isinstance(push, dict):
        branches = push.get("branches-ignore")
    return branches is not None and "main" in branches


def _has_pr_main(on):
    if not isinstance(on, dict):
        return False
    pr = on.get("pull_request")
    if not pr:
        return False
    branches = pr.get("branches") if isinstance(pr, dict) else None
    return branches is not None and "main" in branches


def _step_has_continue_on_error(job):
    for step in job.get("steps", []):
        if not isinstance(step, dict):
            continue
        if step.get("continue-on-error") is True:
            return True, step.get("name", "")
    return False, ""


def _step_script_contains_or_true(job):
    """Find gate steps whose run script ends in `|| true` (silent pass)."""
    offenders = []
    for step in job.get("steps", []):
        if not isinstance(step, dict):
            continue
        run = step.get("run")
        if not run:
            continue
        # normalise whitespace; flag a trailing `|| true` (silent pass padding)
        if "|| true" in run:
            offenders.append(step.get("name", ""))
    return offenders


def _job_for_workflow(data, job_name):
    jobs = data.get("jobs", {})
    return jobs.get(job_name)


# --------------------------------------------------------------------------
# Chaos gate (chaos.yml) — previously schedule-only + continue-on-error:true
# --------------------------------------------------------------------------
def test_chaos_gate_blocks_push_main():
    path = os.path.join(WORKFLOWS, "chaos.yml")
    assert os.path.exists(path), "chaos.yml missing"
    data, on = _load(path)

    # (1) triggered on push:main (and PRs); schedule retained for nightly report
    assert _has_push_main(on), "chaos.yml must trigger on push: branches: [main]"
    assert _has_pr_main(on), "chaos.yml must also run on pull_request"
    assert "schedule" in on, "chaos.yml should keep its nightly schedule"

    # (2) no continue-on-error on the chaos-test job steps
    job = _job_for_workflow(data, "chaos-test")
    assert job is not None, "chaos-test job missing"
    has_coe, name = _step_has_continue_on_error(job)
    assert not has_coe, "chaos-test must NOT use continue-on-error (name=%r)" % name

    # (3) no `|| true` silent-padding on the gate step
    offenders = _step_script_contains_or_true(job)
    assert not offenders, "chaos-test step(s) padded with `|| true`: %r" % offenders


def test_chaos_gate_no_continue_on_error_raw():
    """Grep-based fallback: file must have push:main and no continue-on-error.

    Comment lines are stripped first so explanatory comments (e.g.
    "No continue-on-error ...") are not mistaken for actual steps.
    """
    raw = open(os.path.join(WORKFLOWS, "chaos.yml")).read()
    # drop comment lines so the assertion only inspects real YAML keys
    text = "\n".join(
        ln for ln in raw.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "push:" in text and "branches: [main]" in text
    # chaos.yml should no longer contain any continue-on-error in real steps
    assert "continue-on-error" not in text, "chaos.yml still has continue-on-error"


# --------------------------------------------------------------------------
# Perf gate (benchmark-regression.yml) — had `|| true` silent pass
# --------------------------------------------------------------------------
def test_benchmark_regression_gate_blocks_push_main():
    path = os.path.join(WORKFLOWS, "benchmark-regression.yml")
    assert os.path.exists(path), "benchmark-regression.yml missing"
    data, on = _load(path)

    # (1) triggered on push:main (paths-scoped is fine)
    assert _has_push_main(on), "benchmark-regression.yml must trigger on push:main"

    # (3) the regression-check step must NOT be padded with `|| true`
    job = _job_for_workflow(data, "regression")
    assert job is not None, "regression job missing"
    offenders = _step_script_contains_or_true(job)
    assert not offenders, (
        "benchmark regression step padded with `|| true` (silent pass): %r" % offenders
    )
    # the run must still invoke regression_check (the actual gate)
    run_scripts = [s.get("run", "") for s in job.get("steps", []) if isinstance(s, dict)]
    combined = "\n".join(run_scripts)
    assert "regression_check.py" in combined, "regression check invocation missing"


# --------------------------------------------------------------------------
# Load / resilience gate (load-testing.yml) — was pull_request only
# --------------------------------------------------------------------------
def test_load_testing_gate_wired_into_push_main():
    path = os.path.join(WORKFLOWS, "load-testing.yml")
    assert os.path.exists(path), "load-testing.yml missing"
    data, on = _load(path)

    # (1) now wired into push:main (in addition to pull_request)
    assert _has_push_main(on), "load-testing.yml must trigger on push: main"
    assert _has_pr_main(on), "load-testing.yml must also run on pull_request"

    # (2) the load-test job must fail closed (no continue-on-error)
    job = _job_for_workflow(data, "load-test")
    assert job is not None, "load-test job missing"
    has_coe, name = _step_has_continue_on_error(job)
    assert not has_coe, "load-test must NOT use continue-on-error (name=%r)" % name

    # (3) SLO-check must fail closed — no `|| true` on the gate
    offenders = _step_script_contains_or_true(job)
    assert not offenders, "load-test step(s) padded with `|| true`: %r" % offenders


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
