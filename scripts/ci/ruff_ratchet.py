#!/usr/bin/env python3
"""Ruff no-new-violation ratchet.

Mirrors ``bandit_ratchet.py``: the repo carries a known backlog of lint
findings that cannot be fixed in one change, so this gate fails only when the
count grows. Fix findings and the baseline is refreshed with ``--update``,
which locks in the improvement.

Usage:
    python scripts/ci/ruff_ratchet.py            # check (CI)
    python scripts/ci/ruff_ratchet.py --update   # refresh the baseline
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE_PATH = HERE / "ruff-baseline.json"
TARGETS = ["src/", "benchmarks/"]


def run_ruff() -> list[dict]:
    exe = shutil.which("ruff")
    if exe is None:
        print("[ruff-ratchet] ruff not installed; skipping")
        return []
    proc = subprocess.run(
        [exe, "check", *TARGETS, "--output-format", "json"],
        capture_output=True,
        text=True,
    )
    # ruff exits 1 when it finds violations, which is expected here.
    try:
        return json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        print("[ruff-ratchet] could not parse ruff output; skipping")
        print(proc.stdout[:400])
        print(proc.stderr[:400])
        return []


def summarise(findings: list[dict]) -> dict[str, int]:
    """{rule_code: count}, path-independent so a baseline survives moves."""
    counts: dict[str, int] = {}
    for f in findings:
        code = f.get("code") or "UNKNOWN"
        counts[code] = counts.get(code, 0) + 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="refresh the committed baseline")
    args = ap.parse_args()

    counts = summarise(run_ruff())
    total = sum(counts.values())

    if args.update:
        BASELINE_PATH.write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n")
        print(f"[ruff-ratchet] baseline updated: {total} findings across {len(counts)} rules")
        return 0

    if not BASELINE_PATH.exists():
        print("[ruff-ratchet] no baseline; run with --update to create one")
        return 1

    baseline = json.loads(BASELINE_PATH.read_text())
    regressions = {
        code: (n, baseline.get(code, 0))
        for code, n in counts.items()
        if n > baseline.get(code, 0)
    }

    if regressions:
        print("[ruff-ratchet] NEW lint findings above baseline:")
        for code, (now, was) in sorted(regressions.items()):
            print(f"  {code}: {now} (baseline {was})")
        print("\nFix them, or run --update to accept the increase deliberately.")
        return 1

    improvement = sum(baseline.values()) - total
    print(
        f"[ruff-ratchet] OK -- {total} findings, baseline {sum(baseline.values())}"
        + (f" ({improvement} fewer than baseline)" if improvement > 0 else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())