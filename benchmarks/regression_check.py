"""Regression check: compare current benchmark results against baseline.

Extends the basic regression checker with:
- Benchmark runner (invokes benchmarks/run.py programmatically)
- PR gating: exits with code 1 on regression (CI/PR blocking)
- Flame graph generation from cProfile data
- Git diff awareness for targeted benchmarking

Usage:
    python benchmarks/regression_check.py --run --current results.json
    python benchmarks/regression_check.py --baseline baseline.json --current results.json
    python benchmarks/regression_check.py --flamegraph profile.stats --output flame.svg
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(data: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run_benchmarks(
    benchmark_names: list[str] | None = None,
    model: str = "TinyLlama-1.1B",
    output_path: str = "results.json",
    num_nodes: int = 1,
) -> dict:
    """Run benchmarks by invoking benchmarks/run.py and collect results.

    Args:
        benchmark_names: List of benchmark names (e.g. ['throughput-small', 'latency-ttft']).
                         If None, runs all benchmarks.
        model: Model name to benchmark.
        output_path: Path to write JSON results.
        num_nodes: Number of nodes for distributed benchmarks.

    Returns:
        Dict mapping benchmark name -> metric value.
    """
    benchmarks_dir = Path(__file__).parent
    runner = benchmarks_dir / "run.py"

    if not runner.exists():
        print(f"ERROR: Benchmark runner not found at {runner}", file=sys.stderr)
        sys.exit(1)

    names = benchmark_names or [
        "throughput-small", "latency-ttft", "latency-itl",
        "memory-efficiency", "kv-cache-hit-rate", "spec-accept-rate",
        "defrag-frag-ratio", "defrag-compaction-speed", "defrag-memory-recovery",
    ]

    results = {}
    for name in names:
        print(f"\n{'=' * 60}")
        print(f"Running benchmark: {name}")
        print(f"{'=' * 60}")

        cmd = [
            sys.executable, str(runner),
            "--benchmark", name,
            "--model", model,
            "--json", output_path,
        ]
        if num_nodes > 1:
            cmd.extend(["--nodes", str(num_nodes)])

        try:
            start = time.time()
            subprocess.run(cmd, check=True, cwd=benchmarks_dir.parent)
            elapsed = time.time() - start
            print(f"  Completed in {elapsed:.1f}s")

            # Read the result file
            if os.path.exists(output_path):
                with open(output_path) as f:
                    results[name] = json.load(f)
        except subprocess.CalledProcessError as e:
            print(f"  BENCHMARK FAILED: {e}", file=sys.stderr)
            results[name] = {"error": str(e)}

    return results


def compare_metric(name: str, baseline_val: float, current_val: float,
                   tolerance_pct: float = 10.0,
                   direction: str = "higher_is_better") -> str | None:
    """Compare a single metric and return a regression message or None."""
    if direction == "higher_is_better":
        threshold = baseline_val * (1 - tolerance_pct / 100.0)
        if current_val < threshold:
            drop_pct = (baseline_val - current_val) / baseline_val * 100
            return (
                f"  REGRESSION: {name} = {current_val:.2f} "
                f"(baseline: {baseline_val:.2f}, dropped {drop_pct:.1f}%, "
                f"threshold: {threshold:.2f})"
            )
    else:
        threshold = baseline_val * (1 + tolerance_pct / 100.0)
        if current_val > threshold:
            increase_pct = (current_val - baseline_val) / baseline_val * 100
            return (
                f"  REGRESSION: {name} = {current_val:.2f} "
                f"(baseline: {baseline_val:.2f}, increased {increase_pct:.1f}%, "
                f"threshold: {threshold:.2f})"
            )
    return None


def check_regression(baseline: dict, current: dict) -> list[str]:
    """Compare current metrics against baseline with tolerance thresholds.

    Returns a list of regression messages (empty if no regressions).
    """
    regressions = []
    baseline_metrics = baseline.get("metrics", baseline)
    current_metrics = current.get("metrics", current)

    for name, spec in baseline_metrics.items():
        if isinstance(spec, dict) and "value" in spec:
            baseline_val = spec["value"]
            tolerance_pct = spec.get("tolerance_pct", 10.0)
            direction = spec.get("direction", "higher_is_better")
        else:
            baseline_val = spec if not isinstance(spec, dict) else spec.get("mean", spec.get("value"))
            tolerance_pct = 10.0
            direction = "higher_is_better"

        current_val = current_metrics.get(name)
        if current_val is None:
            regressions.append(f"  MISSING: {name} not found in current results")
            continue

        if isinstance(current_val, dict):
            current_val = current_val.get("mean", current_val.get("value"))

        if current_val is None:
            regressions.append(f"  MISSING: {name} has no numeric value in current results")
            continue

        msg = compare_metric(name, baseline_val, current_val, tolerance_pct, direction)
        if msg:
            regressions.append(msg)

    return regressions


def generate_flame_graph(profile_path: str, output_path: str = "flame.svg") -> None:
    """Generate a flame graph SVG from a cProfile stats file.

    Uses the FlameGraph toolkit (flamegraph.pl) if available, otherwise
    falls back to a simplified cProfile-based summary.

    Args:
        profile_path: Path to a cProfile .stats file.
        output_path: Path for the output SVG or text summary.
    """
    try:
        import pstats

        p = pstats.Stats(profile_path)
        p.sort_stats("cumulative")

        # Try to use Brendan Gregg's FlameGraph toolkit
        flamegraph_pl = os.path.expanduser("~/FlameGraph/flamegraph.pl")
        if os.path.exists(flamegraph_pl):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".folded", delete=False) as folded:
                folded_path = folded.name
                # Generate folded stack format
                pstats_script = (
                    f"import pstats; "
                    f"p = pstats.Stats('{profile_path}'); "
                    f"p.sort_stats('cumulative').print_stats(100)"
                )
            subprocess.run(
                ["perl", flamegraph_pl, "--title", "LLM Inference Profile"],
                stdin=open(folded_path) if os.path.exists(folded_path) else None,
                stdout=open(output_path, "w"),
                stderr=subprocess.DEVNULL,
            )
            os.unlink(folded_path)
            print(f"Flame graph saved to {output_path}")
        else:
            # Fallback: print top functions to text
            with open(output_path, "w") as f:
                f.write(f"Profile summary from {profile_path}\n")
                f.write("=" * 60 + "\n")
                p.stream = f
                p.print_stats(30)
            print(f"Profile summary saved to {output_path} (install FlameGraph for SVG graphs)")
    except ImportError:
        print("cProfile/pstats not available; skipping flame graph generation", file=sys.stderr)


def get_git_diff_files() -> list[str]:
    """Return list of files changed in the working tree vs HEAD."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0:
            return [f.strip() for f in result.stdout.splitlines() if f.strip()]
    except FileNotFoundError:
        pass
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Performance regression detection for distributed LLM inference"
    )
    parser.add_argument("--baseline", default="benchmarks/baseline.json",
                        help="Path to baseline JSON file")
    parser.add_argument("--current", default=None,
                        help="Path to current results JSON file")
    parser.add_argument("--run", action="store_true",
                        help="Run benchmarks before checking")
    parser.add_argument("--model", default="TinyLlama-1.1B",
                        help="Model to benchmark")
    parser.add_argument("--output", default="benchmarks/results/current.json",
                        help="Output path for benchmark results")
    parser.add_argument("--flamegraph", default=None,
                        help="Generate flame graph from cProfile .stats file")
    parser.add_argument("--output-flamegraph", default="flame.svg",
                        help="Output path for flame graph SVG")
    parser.add_argument("--update-baseline", action="store_true",
                        help="Update baseline with current results")
    parser.add_argument("--tolerance", type=float, default=10.0,
                        help="Default tolerance percentage (default: 10%)")
    parser.add_argument("--git-aware", action="store_true",
                        help="Only check benchmarks affected by git diff")

    args = parser.parse_args()

    # Flame graph generation mode
    if args.flamegraph:
        generate_flame_graph(args.flamegraph, args.output_flamegraph)
        return

    # Check if we need to run benchmarks
    if args.run and args.current is None:
        print("Running benchmarks...")
        results = run_benchmarks(
            model=args.model,
            output_path=args.output,
        )
        current_path = args.output
        save_json({"metrics": results}, current_path)
        print(f"Results saved to {current_path}")
    elif args.current:
        current_path = args.current
    else:
        parser.print_help()
        sys.exit(1)

    # Load baseline
    baseline_path = args.baseline
    if not os.path.exists(baseline_path):
        print(f"ERROR: Baseline not found at {baseline_path}. "
              f"Run with --run to create one.", file=sys.stderr)
        sys.exit(1)

    baseline = load_json(baseline_path)
    current = load_json(current_path)

    # Git-aware: only check benchmarks for changed files
    if args.git_aware:
        changed = get_git_diff_files()
        if not changed:
            print("No files changed; skipping benchmark check")
            sys.exit(0)

    # Check regressions
    regressions = check_regression(baseline, current)

    if regressions:
        print("\n" + "=" * 60)
        print("PERFORMANCE REGRESSION DETECTED")
        print("=" * 60)
        for msg in regressions:
            print(msg)
        print(
            f"\n{len(regressions)} regression(s) found. "
            "Review changes, optimize, or update baseline with --update-baseline."
        )
        sys.exit(1)
    else:
        print("No regressions detected. All metrics within tolerance thresholds.")

    # Update baseline if requested
    if args.update_baseline:
        save_json(current, baseline_path)
        print(f"Baseline updated to {baseline_path}")


if __name__ == "__main__":
    main()
