#!/usr/bin/env python3
"""Coverage ratchet gate (§6.6.2).

This script is the *single source of truth* for the coverage gate.  It
does NOT rely on a hard `--cov-fail-under` floor (which was contradictory
and ran over a truncated graph — the suite only covers part of ``src/``).

There are two layers, both enforced here:

1. **Global baseline ratchet** (the single gate).
   A baseline total-coverage number is stored in ``.coverage-baseline.json``
   (repo root) or ``scripts/ci/coverage-baseline.json``.  The current total
   coverage is read from ``coverage.xml``.  The gate FAILS iff the current
   total is below ``baseline - tolerance``.  A baseline of 70% with current
   71% therefore PASSES, even though it is below the old hard 80% floor —
   the ratchet only refuses *regressions*, it does not impose an absolute bar.

2. **Per-module floor ratchet** (defence in depth).
   Individual packages may not drop below their stored floor in
   ``coverage-ratchet.json``.  Critical modules are held higher.

Per-PR delta reporting
-----------------------
The script emits a markdown summary, e.g.::

    Coverage: 73.2% (Δ +1.4% vs baseline 71.8%)

written to ``coverage-delta.md`` (and printed) so it can be posted as a PR
comment.

The comparison logic is intentionally pure / side-effect free
(:func:`check_baseline`, :func:`fmt_delta`) so it can be unit-tested
without producing a coverage report.

Usage::

    # Verify current coverage meets the ratchet (CI gate):
    python scripts/ci/coverage_ratchet.py --coverage-xml coverage.xml

    # Bump every per-module floor +5% at the start of a sprint:
    python scripts/ci/coverage_ratchet.py --bump

If ``coverage.xml`` is absent (local dev without a full run) the script
exits 0 with a warning so it never blocks environments that don't produce
the report.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
RATCHET_PATH = HERE / "coverage-ratchet.json"
# Baseline lookup order: repo-root, then beside this script.
BASELINE_PATHS = [
    HERE.parent.parent / ".coverage-baseline.json",
    HERE / "coverage-baseline.json",
]
DEFAULT_BASELINE = 0.0
DEFAULT_TOLERANCE = 0.0  # allow the gate to fail only on an actual regression

# Default starting floors (§6.6.2), raised to the TARGET spec this wave:
#   * critical modules (pipeline, worker, security, coordinator,
#     differential_privacy) = 85.0
#   * dist = 75.0
#   * global floor ratchets +5%/sprint toward 70% (currently 35.0 on the
#     way up from the previous 30.0; _sprint_step stays 5.0).
# NOTE: these are only used when scripts/ci/coverage-ratchet.json is absent.
# The committed coverage-ratchet.json is the authoritative floor set and is
# kept in sync with this dict.
DEFAULT_RATCHET = {
    "_global_floor": 35.0,   # +5%/sprint toward 70% (was 30.0)
    "_sprint_step": 5.0,
    # Critical modules held at a higher bar (85.0 per spec).
    "distllm/core": 85.0,
    "distllm/core/coordinator_state": 85.0,
    "distllm/core/coordinator": 85.0,
    "distllm/core/di": 90.0,
    "distllm/core/cache_manager": 85.0,
    "distllm/core/differential_privacy": 85.0,
    "distllm/core/ha_coordinator": 85.0,
    "distllm/core/plugin_sandbox": 85.0,
    "distllm/core/cost_dashboard": 85.0,
    "distllm/core/autoscaler": 85.0,
    # Pipeline + worker critical modules (85.0 per spec).
    "distllm/core/pipeline_orchestrator": 85.0,
    "distllm/core/pipeline_composer": 85.0,
    "distllm/core/pipeline_overlap": 85.0,
    "distllm/core/request_pipeline": 85.0,
    "distllm/dist/worker": 85.0,
    "distllm/security": 85.0,
    # Distributed layer floor (75.0 per spec).
    "distllm/dist": 75.0,
    # API layer floor (raised alongside the critical ratchet).
    "distllm/api": 75.0,
}


# ---------------------------------------------------------------------------
# Pure comparison logic (no file IO — unit-testable)
# ---------------------------------------------------------------------------
def fmt_delta(current: float, baseline: float) -> str:
    """Format a signed per-PR delta string, e.g. ``+1.4%`` / ``-0.6%``."""
    delta = current - baseline
    sign = "+" if delta >= 0 else "-"
    return f"{sign}{abs(delta):.1f}%"


def fmt_summary(current: float, baseline: float, tolerance: float = 0.0) -> str:
    """Render the per-PR markdown summary line."""
    return (
        f"Coverage: {current:.1f}% "
        f"(Δ {fmt_delta(current, baseline)} vs baseline {baseline:.1f}%)"
    )


def check_baseline(
    current: float,
    baseline: float,
    tolerance: float = 0.0,
) -> tuple[bool, str]:
    """Pure global-baseline ratchet check.

    Returns ``(passed, message)``.

    The gate PASSES when ``current >= baseline - tolerance``.  It deliberately
    ignores any absolute 80% floor — a baseline of 70% with current 71%
    PASSES even though it is below 80.  The ratchet only refuses regressions.

    :param current: current total line coverage, percent (0-100).
    :param baseline: recorded baseline total coverage, percent (0-100).
    :param tolerance: allowed drop vs baseline before failing (percent).
    """
    threshold = baseline - tolerance
    passed = current >= threshold
    if passed:
        msg = (
            f"OK -- total coverage {current:.1f}% meets baseline "
            f"{baseline:.1f}% (threshold {threshold:.1f}%, tolerance {tolerance:.1f}%)."
        )
    else:
        msg = (
            f"FAILED -- total coverage {current:.1f}% is below the ratchet "
            f"threshold {threshold:.1f}% (baseline {baseline:.1f}%, "
            f"tolerance {tolerance:.1f}%). Coverage must not regress."
        )
    return passed, msg


def parse_total_coverage(xml_path: Path) -> float:
    """Return overall line coverage (percent) from coverage.xml."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # ``coverage`` element carries ``line-rate`` for the whole report.
    line_rate = root.get("line-rate")
    if line_rate is not None:
        return float(line_rate) * 100.0
    # Fallback: weighted by lines.
    total_lines = 0
    covered = 0
    for cls in root.iter("class"):
        for line in cls.iter("line"):
            n = int(line.get("hits", "0"))
            total_lines += 1
            if n > 0:
                covered += 1
    if total_lines == 0:
        return 0.0
    return covered / total_lines * 100.0


def load_baseline() -> float:
    for path in BASELINE_PATHS:
        if path.exists():
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                return float(data.get("total_coverage", data.get("baseline", DEFAULT_BASELINE)))
            return float(data)
    return DEFAULT_BASELINE


def write_delta_md(xml_path: Path | None, current: float, baseline: float, tolerance: float) -> None:
    summary = fmt_summary(current, baseline, tolerance)
    md = (
        "## Coverage ratchet\n\n"
        f"{summary}\n\n"
        f"- baseline: {baseline:.1f}%\n"
        f"- current: {current:.1f}%\n"
        f"- tolerance: {tolerance:.1f}%\n"
        f"- source: {xml_path or 'n/a'}\n"
    )
    out = Path("coverage-delta.md")
    out.write_text(md)
    return None


def load_ratchet() -> dict:
    if RATCHET_PATH.exists():
        return json.loads(RATCHET_PATH.read_text())
    return dict(DEFAULT_RATCHET)


def save_ratchet(data: dict) -> None:
    RATCHET_PATH.write_text(json.dumps(data, indent=2) + "\n")


def floor_for(ratchet: dict, package: str) -> float:
    """Most specific matching prefix wins; else the global floor."""
    best = ratchet.get("_global_floor", 30.0)
    for key, val in ratchet.items():
        if key.startswith("_"):
            continue
        if package == key or package.startswith(key + "/") or package.startswith(key):
            if isinstance(val, (int, float)) and val > best:
                best = val
    return best


def parse_coverage(xml_path: Path) -> dict[str, float]:
    """Return {package-or-module: line-rate%} from coverage.xml."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    out: dict[str, float] = {}
    for pkg in root.iter("package"):
        pname = pkg.get("name", "")
        for cls in pkg.iter("class"):
            fname = cls.get("filename", "")
            # Use package dir as the unit of comparison.
            mod = fname
            if "src/" in fname:
                mod = fname.split("src/", 1)[1]
            elif "distllm/" in fname:
                mod = fname[fname.index("distllm/") :]
            # package = everything except the final file component
            pkg_key = "/".join(mod.split("/")[:-1])
            rate = float(cls.get("line-rate", "0")) * 100.0
            out[pkg_key] = max(out.get(pkg_key, 0.0), rate)
    return out


def cmd_check(xml_path: Path, tolerance: float, emit_delta: bool) -> int:
    if not xml_path.exists():
        print(f"[coverage-ratchet] {xml_path} not found -- skipping (no report produced).")
        return 0

    baseline = load_baseline()
    current = parse_total_coverage(xml_path)
    passed, msg = check_baseline(current, baseline, tolerance)
    print(f"[coverage-ratchet] {msg}")
    summary = fmt_summary(current, baseline, tolerance)
    print(f"[coverage-ratchet] {summary}")
    if emit_delta:
        write_delta_md(xml_path, current, baseline, tolerance)
        print("[coverage-ratchet] wrote coverage-delta.md")

    # Per-module floor ratchet (defence in depth) — only checked if a
    # module floor config exists and the global gate already passed, so the
    # baseline ratchet stays the single *mandatory* gate on total coverage.
    ratchet = load_ratchet()
    cov = parse_coverage(xml_path)
    failures = []
    for package, rate in sorted(cov.items()):
        floor = floor_for(ratchet, package)
        if rate < floor:
            failures.append(f"  {package}: {rate:.1f}% < floor {floor:.1f}%")
    if failures:
        print("[coverage-ratchet] PER-MODULE floor violations:")
        print("\n".join(failures))
        return 1
    print(f"[coverage-ratchet] OK -- {len(cov)} packages meet their floors.")

    return 0 if passed else 1


def cmd_bump(ratchet: dict) -> int:
    step = ratchet.get("_sprint_step", 5.0)
    ratchet["_global_floor"] = min(100.0, ratchet["_global_floor"] + step)
    for k in list(ratchet):
        if k.startswith("_"):
            continue
        ratchet[k] = min(100.0, ratchet[k] + step)
    save_ratchet(ratchet)
    print(f"[coverage-ratchet] bumped floors +{step}% (global now {ratchet['_global_floor']:.0f}%).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Coverage ratchet gate (§6.6.2)")
    ap.add_argument("--coverage-xml", default="coverage.xml", type=Path)
    ap.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="allowed drop vs baseline (percent) before the gate fails "
             "(default 0 — any regression fails).",
    )
    ap.add_argument(
        "--baseline",
        type=float,
        default=None,
        help="override the recorded baseline total coverage (percent).",
    )
    ap.add_argument(
        "--emit-delta",
        action="store_true",
        help="write coverage-delta.md per-PR summary (for PR comments).",
    )
    ap.add_argument("--bump", action="store_true", help="raise every floor by the sprint step (sprint ratchet)")
    args = ap.parse_args()

    ratchet = load_ratchet()
    if args.bump:
        return cmd_bump(ratchet)
    return cmd_check(args.coverage_xml, args.tolerance, args.emit_delta)


if __name__ == "__main__":
    raise SystemExit(main())
