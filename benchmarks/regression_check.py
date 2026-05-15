"""Regression check: compares current benchmark results against baseline.

Usage:
    python benchmarks/regression_check.py --baseline benchmarks/baseline.json --current results.json

Exits with code 1 if any metric regresses beyond its tolerance threshold.
"""

import argparse
import json
import sys
from pathlib import Path


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def check_regression(baseline: dict, current: dict) -> list[str]:
    """Compare current metrics against baseline with tolerance thresholds.

    Returns a list of regression messages (empty if no regressions).
    """
    regressions = []
    baseline_metrics = baseline.get("metrics", {})

    for name, spec in baseline_metrics.items():
        baseline_val = spec["value"]
        tolerance_pct = spec.get("tolerance_pct", 10.0)
        direction = spec.get("direction", "higher_is_better")

        # Find the current value — support nested dicts (pytest-benchmark format)
        current_val = current.get(name)
        if current_val is None:
            regressions.append(f"  MISSING: {name} not found in current results")
            continue

        if isinstance(current_val, dict):
            # pytest-benchmark format: {"mean": ..., "min": ..., "max": ...}
            current_val = current_val.get("mean", current_val.get("value"))

        if direction == "higher_is_better":
            # Value must not drop more than tolerance_pct
            threshold = baseline_val * (1 - tolerance_pct / 100.0)
            if current_val < threshold:
                drop_pct = (baseline_val - current_val) / baseline_val * 100
                regressions.append(
                    f"  REGRESSION: {name} = {current_val:.2f} "
                    f"(baseline: {baseline_val:.2f}, dropped {drop_pct:.1f}%, "
                    f"threshold: {threshold:.2f})"
                )
        else:
            # lower_is_better: value must not increase more than tolerance_pct
            threshold = baseline_val * (1 + tolerance_pct / 100.0)
            if current_val > threshold:
                increase_pct = (current_val - baseline_val) / baseline_val * 100
                regressions.append(
                    f"  REGRESSION: {name} = {current_val:.2f} "
                    f"(baseline: {baseline_val:.2f}, increased {increase_pct:.1f}%, "
                    f"threshold: {threshold:.2f})"
                )

    return regressions


def main():
    parser = argparse.ArgumentParser(description="Check for benchmark regressions")
    parser.add_argument(
        "--baseline",
        default="benchmarks/baseline.json",
        help="Path to baseline JSON file",
    )
    parser.add_argument(
        "--current",
        required=True,
        help="Path to current results JSON file",
    )
    args = parser.parse_args()

    baseline = load_json(args.baseline)
    current = load_json(args.current)

    regressions = check_regression(baseline, current)

    if regressions:
        print("Benchmark regression detected!")
        for msg in regressions:
            print(msg)
        print(
            f"\n{len(regressions)} regression(s) found. "
            "Update baseline with 'make bench-update' if improvements are intentional."
        )
        sys.exit(1)
    else:
        print("No regressions detected. All metrics within tolerance thresholds.")
        sys.exit(0)


if __name__ == "__main__":
    main()
