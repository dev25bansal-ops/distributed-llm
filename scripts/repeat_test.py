"""Repeat a single test N times to detect timing-dependent flakiness.

pytest-repeat is not installed, so this drives pytest directly and reports the
pass/fail count per run. Used to establish whether the end-to-end verifier test
in test_pipelined_speculative_f-040.py fails on its own merits or only when the
machine is loaded.
"""

from __future__ import annotations

import subprocess
import sys

TARGET = (
    "tests/core/test_pipelined_speculative_f-040.py::"
    "TestDraftWorkerPopulatesVerifierInputs::"
    "test_generate_end_to_end_verifier_sees_real_inputs"
)

RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 15

args = [
    sys.executable, "-m", "pytest", TARGET,
    "-q", "--no-header", "--cov-fail-under=0",
    "-p", "no:randomly", "-p", "no:cacheprovider",
]

passed = failed = 0
for run in range(1, RUNS + 1):
    proc = subprocess.run(args, capture_output=True, text=True, timeout=120)
    tail = [ln for ln in (proc.stdout or "").splitlines() if " passed" in ln or " failed" in ln]
    verdict = tail[-1] if tail else f"no verdict (rc={proc.returncode})"
    if " failed" in verdict:
        failed += 1
    elif " passed" in verdict:
        passed += 1
    print(f"run {run:2d}: {verdict}", flush=True)

print(f"\n{TARGET}\n{RUNS} runs: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)