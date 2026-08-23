#!/usr/bin/env python3
"""Fast perf-regression gate for the push:main smoke (§CI perf-smoke job).

Reads a Locust ``--csv`` ``*_stats.csv`` file (produced by the
``perf-smoke`` CI job) and extracts the p95 latency for the aggregate
("Aggregated") row.  Two modes:

* **Comparison mode** -- given ``--baseline perf-baseline.json``, fail the
  build if the current p95 exceeds the baseline p95 by more than
  ``--p95-threshold-pct`` (default 5.0%).  No ``|| true`` in CI, so a real
  regression blocks ``push:main``.

* **Baseline-emit mode** -- given ``--baseline-output perf-baseline.json``
  (no existing baseline), record the current p95 as the baseline and exit 0.

The script is intentionally dependency-free (stdlib only) so it can run in
the lightweight CI image.  If the stats CSV is missing or malformed it
exits 0 with a warning so it never blocks environments without a report.

Usage (comparison, blocking)::

    python tests/load/locust/perf_smoke_gate.py \
        --baseline tests/load/locust/perf-baseline.json \
        --csv tests/load/results/perf_smoke_stats.csv \
        --p95-threshold-pct 5.0

Usage (emit baseline)::

    python tests/load/locust/perf_smoke_gate.py \
        --baseline-output tests/load/locust/perf-baseline.json \
        --csv tests/load/results/perf_smoke_stats.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _read_p95(stats_csv: Path) -> float | None:
    """Return the p95 (Request Column '95%') for the Aggregated row, in ms.

    Locust's ``--csv`` stats file has a header row and a trailing
    ``"Aggregated",<...>,<50%>,<66%>,<75%>,<80%>,<90%>,<95%>,<98%>,<99%>,<99.9%>,...``
    row.  The p95 column is the 8th numeric column after the name (index 7 in
    the percentile block, i.e. CSV column index 8).
    """
    if not stats_csv.exists():
        return None
    with stats_csv.open(newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        return None

    header = rows[0]
    # Locust columns: Type,Name,Requests,Failures,Median,Average,Min,Max,
    #                 95%ile,99%ile,..  (older versions: 50%ile..99.9%ile block)
    # Find the p95 column by header name when possible.
    p95_idx = None
    for i, col in enumerate(header):
        if col.strip().lower().startswith("95"):
            p95_idx = i
            break
    # Fallback: the 8th column (index 7) is the p95 in modern Locust.
    if p95_idx is None and len(header) > 7:
        p95_idx = 7

    if p95_idx is None:
        return None

    for row in rows[1:]:
        if len(row) <= p95_idx:
            continue
        if row[0].strip().lower() == "aggregated":
            try:
                return float(row[p95_idx])
            except (ValueError, TypeError):
                return None
    # No aggregated row — fall back to the max p95 across request rows.
    values = []
    for row in rows[1:]:
        if len(row) <= p95_idx or not row[0].strip():
            continue
        try:
            values.append(float(row[p95_idx]))
        except (ValueError, TypeError):
            continue
    return max(values) if values else None


def _load_baseline(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return float(data.get("p95_ms", data.get("p95", None)))
    return None


def _emit_baseline(path: Path, p95: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"p95_ms": p95}, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fast perf-regression p95 gate")
    ap.add_argument("--csv", required=True, type=Path,
                    help="Locust *_stats.csv produced by the smoke run")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="Baseline JSON (contains p95_ms) to compare against")
    ap.add_argument("--baseline-output", type=Path, default=None,
                    help="Write the measured p95 as the new baseline JSON")
    ap.add_argument("--p95-threshold-pct", type=float, default=5.0,
                    help="Max allowed p95 regression vs baseline (percent)")
    args = ap.parse_args()

    current = _read_p95(args.csv)
    if current is None:
        print(f"[perf-smoke-gate] No p95 found in {args.csv} -- skipping (no report).")
        return 0

    print(f"[perf-smoke-gate] current p95 = {current:.2f} ms")

    # Emit-only mode.
    if args.baseline_output is not None and args.baseline is None:
        _emit_baseline(args.baseline_output, current)
        print(f"[perf-smoke-gate] wrote baseline -> {args.baseline_output}")
        return 0

    baseline = _load_baseline(args.baseline) if args.baseline else None
    if baseline is None:
        print(f"[perf-smoke-gate] No baseline available at {args.baseline} -- "
              f"treating current as baseline (no regression to block).")
        return 0

    print(f"[perf-smoke-gate] baseline p95 = {baseline:.2f} ms "
          f"(threshold {args.p95_threshold_pct:.1f}%)")
    allowed = baseline * (1.0 + args.p95_threshold_pct / 100.0)
    if current > allowed:
        delta = current - baseline
        pct = (delta / baseline) * 100.0 if baseline else 0.0
        print(f"[perf-smoke-gate] FAILED -- p95 {current:.2f} ms exceeds allowed "
              f"{allowed:.2f} ms (baseline {baseline:.2f} ms, +{pct:.1f}% regression).")
        return 1

    print(f"[perf-smoke-gate] OK -- p95 within {args.p95_threshold_pct:.1f}% of baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
