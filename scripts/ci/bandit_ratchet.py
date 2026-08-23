#!/usr/bin/env python3
"""Bandit no-new-violation ratchet (§6.6.1).

Runs Bandit over ``src/`` and compares the set of findings against a
committed baseline (``bandit-baseline.json``).  The gate FAILS when a
*new* HIGH or MEDIUM finding appears (regardless of the baseline), and
when a previously-known finding was *fixed* it is pruned from the
baseline on the next ``--update``.

This makes Bandit a *blocking* gate that only allows the known set to
persist -- new HIGH/MEDIUM issues are rejected at PR time.

Usage::

    # CI gate: fail on any new HIGH/MEDIUM (or any HIGH ever).
    python scripts/ci/bandit_ratchet.py

    # After intentionally accepting/fixing findings, refresh the baseline:
    python scripts/ci/bandit_ratchet.py --update
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
BASELINE_PATH = HERE / "bandit-baseline.json"
SRC = HERE.parent.parent / "src"


def run_bandit() -> list[dict]:
    """Run bandit and return the list of finding dicts.

    We write JSON to a temp file (``-o``) rather than parsing stdout,
    because Bandit's quiet/JSON output routing differs between stdout
    and file and between platforms (path separators).  Using a file makes
    the scan identical to how the baseline is generated.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        out_path = tf.name
    try:
        subprocess.run(
            [sys.executable, "-m", "bandit", "-r", str(SRC), "-f", "json", "-q", "-o", out_path],
            capture_output=True,
            text=True,
        )
        with open(out_path) as fh:
            data = json.load(fh)
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass
    return data.get("results", [])


def key_of(f: dict) -> str:
    # Key on the finding *class* (file + test id).  Normalize to a
    # repo-relative path so a baseline built from a relative ``src/`` run
    # matches a scan driven by an absolute SRC path, and so Windows "\"
    # vs POSIX "/" don't matter.  What we block is a *new* test firing in a
    # file that didn't have it before.
    fname = str(f.get("filename", "")).replace("\\", "/")
    # Strip an absolute repo-root prefix so keys look like "src/distllm/...".
    try:
        rel = Path(fname).resolve().relative_to(REPO_ROOT.resolve())
        fname = str(rel).replace("\\", "/")
    except Exception:
        pass
    return f"{fname}:{f.get('test_id')}"


def severity_rank(sev: str) -> int:
    # Bandit sometimes reports compound severities like "MEDIUM/HIGH".
    s = str(sev).upper()
    if "HIGH" in s:
        return 3
    if "MEDIUM" in s:
        return 2
    if "LOW" in s:
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Bandit no-new-violation ratchet (§6.6.1)")
    ap.add_argument("--update", action="store_true", help="refresh the committed baseline")
    args = ap.parse_args()

    findings = run_bandit()
    current = {key_of(f): f for f in findings}

    if args.update:
        baseline = {k: {"severity": f["issue_severity"], "confidence": f["issue_confidence"]}
                    for k, f in current.items()}
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
        print(f"[bandit-ratchet] baseline updated: {len(baseline)} findings.")
        return 0

    baseline = json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else {}

    new_high = [f for k, f in current.items() if k not in baseline and severity_rank(f["issue_severity"]) >= 3]
    new_medium = [f for k, f in current.items() if k not in baseline and severity_rank(f["issue_severity"]) == 2]

    if new_high:
        print("[bandit-ratchet] FAILED -- NEW HIGH severity findings:")
        for f in new_high:
            print(f"  {f['filename']}:{f['line_number']} {f['test_id']} ({f['issue_severity']}/{f['issue_confidence']})")
    if new_medium:
        print("[bandit-ratchet] FAILED -- NEW MEDIUM severity findings:")
        for f in new_medium:
            print(f"  {f['filename']}:{f['line_number']} {f['test_id']} ({f['issue_severity']}/{f['issue_confidence']})")

    if new_high or new_medium:
        print("\nRun `python scripts/ci/bandit_ratchet.py --update` ONLY after intentional triage.")
        return 1

    print(f"[bandit-ratchet] OK -- {len(current)} findings, none new (HIGH/MEDIUM).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
