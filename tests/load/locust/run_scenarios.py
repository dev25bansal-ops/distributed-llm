"""Run all load test scenarios and generate a report.

Usage:
    python tests/load/locust/run_scenarios.py
    python tests/load/locust/run_scenarios.py --host http://localhost:8000 --users 10 --time 5m
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
SCENARIOS_DIR = HERE / "scenarios"
RESULTS_DIR = HERE.parent / "results"

SCENARIOS = [
    ("chat", "chat_scenario.py", 10, 2, "2m"),
    ("streaming", "streaming_scenario.py", 5, 1, "2m"),
    ("embeddings", "embeddings_scenario.py", 5, 2, "1m"),
    ("batch", "batch_scenario.py", 3, 1, "1m"),
    ("mixed", "mixed_scenario.py", 20, 4, "5m"),
]


def run_scenario(name, file, users, spawn_rate, run_time, host):
    """Run a single scenario with headless Locust."""
    print(f"\n{'='*60}")
    print(f"  Scenario: {name}  ({users} users, {spawn_rate}/s, {run_time})")
    print(f"{'='*60}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_prefix = str(RESULTS_DIR / name)

    cmd = [
        sys.executable, "-m", "locust",
        "-f", str(SCENARIOS_DIR / file),
        "--headless",
        "-u", str(users),
        "-r", str(spawn_rate),
        "--run-time", run_time,
        "--host", host,
        "--csv", csv_prefix,
        "--csv-full-body",
        "--stop-timeout", "10",
    ]

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"  ✓ Completed in {elapsed:.1f}s")
    else:
        print(f"  ⚠  Exited with code {result.returncode} in {elapsed:.1f}s")
        if result.stderr:
            for line in result.stderr.split("\n")[-5:]:
                if line.strip():
                    print(f"     {line.strip()}")

    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run all load test scenarios")
    parser.add_argument("--host", default="http://localhost:8000", help="Target host")
    parser.add_argument("--scenario", choices=[s[0] for s in SCENARIOS] + ["all"], default="all")
    args = parser.parse_args()

    failures = 0
    for name, file, users, spawn_rate, run_time in SCENARIOS:
        if args.scenario != "all" and args.scenario != name:
            continue
        rc = run_scenario(name, file, users, spawn_rate, run_time, args.host)
        if rc != 0:
            failures += 1

    print(f"\n{'='*60}")
    if failures:
        print(f"  ✗ {failures} scenario(s) failed")
        sys.exit(1)
    else:
        print("  ✓ All scenarios completed")
        print(f"\n  Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
