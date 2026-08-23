#!/usr/bin/env python3
"""Flaky / skip / xfail ratchet gate (N8, T5-02).

After a pytest run that used ``-p flaky_test_collector --flaky-report=...``
(see :mod:`flaky_test_collector`), this script compares the *current* set of
flaky tests against a committed baseline (``flaky_baseline.json``).

Ratchet semantics
-----------------
The flaky count is a **ratchet that only goes down or stays equal**.  A flaky
test that needs a retry in CI is a real signal of instability; we refuse to let
the set grow.  If the current flaky count is *greater* than the baseline, the
gate FAILS and must be acknowledged with ``--update`` (after the team has
actually *fixed* flakiness or accepted a deliberate, documented regression).

skip / xfail are emitted for visibility (and also ratcheted so they cannot
silently grow) but are not by themselves a hard failure the way an *increase*
in flakiness is -- growing skip/xfail is reported and, by default, also fails
the gate (a growing skip/xfail set usually means tests being disabled rather
than fixed).  Use ``--no-skip-gate`` to make skip/xfail advisory only.

Usage::

    # CI gate: fail if flaky (or skip/xfail) count grew beyond baseline.
    PYTHONPATH=scripts/ci python scripts/ci/flaky_test_ratchet.py \\
        --report flaky_report.json

    # After intentionally acknowledging a change: refresh the baseline.
    PYTHONPATH=scripts/ci python scripts/ci/flaky_test_ratchet.py \\
        --report flaky_report.json --update

The comparator logic ``check_ratchet`` is pure (no file IO) so it can be
unit-tested without producing a flaky report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
# Baseline lookup order: repo-root, then beside this script.
BASELINE_PATHS = [
    REPO_ROOT / "flaky_baseline.json",
    HERE / "flaky_baseline.json",
]


# ---------------------------------------------------------------------------
# Pure comparison logic (no file IO -- unit-testable, mirrors coverage_ratchet)
# ---------------------------------------------------------------------------
def check_ratchet(current: int, baseline: int) -> tuple[bool, str]:
    """Pure ratchet check: count must not increase.

    Returns ``(passed, message)``.  The gate PASSES when
    ``current <= baseline`` -- the flaky set may shrink or stay flat, but must
    never grow without an explicit ``--update``.  A baseline of 3 with current
    2 PASSES; a baseline of 3 with current 4 FAILS.
    """
    passed = current <= baseline
    if passed:
        delta = baseline - current
        msg = (
            f"OK -- {current} flaky (<= baseline {baseline}; "
            f"{delta} fewer than baseline)."
        )
    else:
        msg = (
            f"FAILED -- {current} flaky exceeds baseline {baseline} by "
            f"{current - baseline}. Flaky count must not increase. "
            f"Fix the flakiness or run with --update after acknowledgement."
        )
    return passed, msg


def fmt_summary(metric: str, current: int, baseline: int) -> str:
    arrow = "down" if current < baseline else ("up" if current > baseline else "flat")
    return f"{metric}: {current} (baseline {baseline}, {arrow})"


# ---------------------------------------------------------------------------
# Report / baseline IO
# ---------------------------------------------------------------------------
def load_report(report_path: Path) -> dict:
    data = json.loads(report_path.read_text())
    return {
        "flaky_tests": data.get("flaky_tests", []),
        "flaky_count": int(data.get("flaky_count", len(data.get("flaky_tests", [])))),
        "skipped_tests": data.get("skipped_tests", []),
        "skipped_count": int(data.get("skipped_count", len(data.get("skipped_tests", [])))),
        "xfailed_tests": data.get("xfailed_tests", []),
        "xfailed_count": int(data.get("xfailed_count", len(data.get("xfailed_tests", [])))),
    }


def load_baseline() -> dict:
    override = os.environ.get("FLAKY_BASELINE_OVERRIDE")
    paths = [Path(override)] if override else BASELINE_PATHS
    for path in paths:
        if path.exists():
            data = json.loads(path.read_text())
            return {
                "flaky_tests": data.get("flaky_tests", []),
                "flaky_count": int(data.get("flaky_count", len(data.get("flaky_tests", [])))),
                "skipped_tests": data.get("skipped_tests", []),
                "skipped_count": int(data.get("skipped_count", len(data.get("skipped_tests", [])))),
                "xfailed_tests": data.get("xfailed_tests", []),
                "xfailed_count": int(data.get("xfailed_count", len(data.get("xfailed_tests", [])))),
            }
    # No baseline yet -> first run seeds from current (count 0 by default).
    return {
        "flaky_tests": [],
        "flaky_count": 0,
        "skipped_tests": [],
        "skipped_count": 0,
        "xfailed_tests": [],
        "xfailed_count": 0,
    }


def write_baseline(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _resolve_baseline_path() -> Path:
    # Allow tests / CI to point at an alternate baseline location.
    override = os.environ.get("FLAKY_BASELINE_OVERRIDE")
    if override:
        return Path(override)
    for path in BASELINE_PATHS:
        if path.exists():
            return path
    return BASELINE_PATHS[0]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_check(report_path: Path, skip_gate: bool, emit_md: bool) -> int:
    if not report_path.exists():
        print(f"[flaky-ratchet] {report_path} not found -- skipping (no report produced).")
        return 0

    current = load_report(report_path)
    baseline = load_baseline()

    flaky_ok, flaky_msg = check_ratchet(current["flaky_count"], baseline["flaky_count"])
    print(f"[flaky-ratchet] {flaky_msg}")
    print(f"[flaky-ratchet] {fmt_summary('flaky', current['flaky_count'], baseline['flaky_count'])}")

    failures: list[str] = []
    if not flaky_ok:
        failures.append("flaky")

    skip_ok, skip_msg = check_ratchet(current["skipped_count"], baseline["skipped_count"])
    xfail_ok, xfail_msg = check_ratchet(current["xfailed_count"], baseline["xfailed_count"])
    print(f"[flaky-ratchet] {fmt_summary('skipped', current['skipped_count'], baseline['skipped_count'])}")
    print(f"[flaky-ratchet] {fmt_summary('xfailed', current['xfailed_count'], baseline['xfailed_count'])}")

    if skip_gate:
        if not skip_ok:
            failures.append("skipped")
            print(f"[flaky-ratchet] skipped FAILED -- {skip_msg}")
        if not xfail_ok:
            failures.append("xfailed")
            print(f"[flaky-ratchet] xfailed FAILED -- {xfail_msg}")
    else:
        print("[flaky-ratchet] skip/xfail gate disabled (--no-skip-gate); reporting only.")

    if emit_md:
        md = (
            "## Flaky-test ratchet\n\n"
            f"- flaky: {current['flaky_count']} (baseline {baseline['flaky_count']})\n"
            f"- skipped: {current['skipped_count']} (baseline {baseline['skipped_count']})\n"
            f"- xfailed: {current['xfailed_count']} (baseline {baseline['xfailed_count']})\n"
        )
        Path("flaky-ratchet.md").write_text(md)
        print("[flaky-ratchet] wrote flaky-ratchet.md")

    if failures:
        joined = ", ".join(failures)
        print(f"\n[flaky-ratchet] RATCHET FAILED for: {joined}.")
        print("Run `PYTHONPATH=scripts/ci python scripts/ci/flaky_test_ratchet.py "
              "--report <report> --update` ONLY after acknowledging the change.")
        return 1

    print("[flaky-ratchet] OK -- flaky/skip/xfail counts at or below baseline.")
    return 0


def cmd_update(report_path: Path) -> int:
    if not report_path.exists():
        print(f"[flaky-ratchet] {report_path} not found -- cannot update baseline.")
        return 1
    current = load_report(report_path)
    baseline_path = _resolve_baseline_path()
    write_baseline(current, baseline_path)
    print(
        f"[flaky-ratchet] baseline updated -> {baseline_path}: "
        f"flaky={current['flaky_count']}, skipped={current['skipped_count']}, "
        f"xfailed={current['xfailed_count']}."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Flaky/skip/xfail ratchet gate (N8, T5-02)")
    ap.add_argument("--report", default="flaky_report.json", type=Path,
                    help="flaky report JSON produced by flaky_test_collector.py")
    ap.add_argument("--update", action="store_true",
                    help="refresh the committed baseline (flaky_baseline.json)")
    ap.add_argument("--no-skip-gate", action="store_true",
                    help="make skip/xfail counts advisory only (still reported)")
    ap.add_argument("--emit-md", action="store_true",
                    help="write flaky-ratchet.md per-PR summary (for PR comments)")
    args = ap.parse_args()

    if args.update:
        return cmd_update(args.report)
    return cmd_check(args.report, skip_gate=not args.no_skip_gate, emit_md=args.emit_md)


if __name__ == "__main__":
    raise SystemExit(main())
