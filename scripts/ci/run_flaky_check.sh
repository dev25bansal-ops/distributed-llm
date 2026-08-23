#!/usr/bin/env bash
# Run the test suite and produce BOTH a junit XML report and a flaky/skip/xfail
# JSON report, then gate on the flaky ratchet.
#
# N8 (T5-02): flaky-test detection.
#   * pytest-rerunfailures (wired in pytest.ini as addopts: --reruns 2) transparently
#     retries transient failures so a single flake doesn't fail CI.
#   * flaky_test_collector.py records which tests NEEDED a rerun (this is NOT
#     available from junit XML alone, so we need the collector plugin).
#   * flaky_test_ratchet.py RATCHETS the flaky/skip/xfail counts vs baseline so the
#     flaky set can only shrink over time.
#
# Usage:
#   scripts/ci/run_flaky_check.sh [pytest-args...]
#
# Examples:
#   scripts/ci/run_flaky_check.sh tests/regression_high
#   scripts/ci/run_flaky_check.sh -m "not integration"
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-python}"
# Prefer the project venv if present.
if [ -x .venv311/Scripts/python.exe ]; then
  PY=.venv311/Scripts/python.exe
elif [ -x .venv311/bin/python ]; then
  PY=.venv311/bin/python
fi

JUNIT="${JUNITXML:-junit.xml}"
REPORT="${FLAKY_REPORT:-flaky_report.json}"

# Ensure the collector plugin is importable.
export PYTHONPATH="${REPO_ROOT}/scripts/ci:${PYTHONPATH:-}"

echo "[run_flaky_check] pytest -> junit=$JUNIT flaky_report=$REPORT"
"$PY" -m pytest "$@" \
  -p no:cacheprovider -p flaky_test_collector \
  --flaky-report "$REPORT" \
  --junitxml "$JUNIT"

echo "[run_flaky_check] flaky ratchet gate"
"$PY" scripts/ci/flaky_test_ratchet.py --report "$REPORT" --emit-md
