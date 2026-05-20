"""Performance SLO gate for CI pipeline.

Reads Locust CSV results and fails CI if SLO thresholds are exceeded.
"""

import argparse
import csv
import sys


def check_slo(csv_path: str, p95_ms: float, p99_ms: float, error_rate: float) -> bool:
    """Check if Locust results meet SLO thresholds."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Name") == "Aggregated":
                avg_ms = float(row.get("Average Median Response Time", 0))
                p95_actual = float(row.get("Average 95% Response Time", 0))
                p99_actual = float(row.get("Average 99% Response Time", 0))
                total = int(row.get("Num Requests", 0))
                failures = int(row.get("Num Failures", 0))
                actual_error_rate = failures / max(total, 1)

                print(f"SLO Check Results:")
                print(f"  p95 latency: {p95_actual:.0f}ms (threshold: {p95_ms:.0f}ms) {'PASS' if p95_actual <= p95_ms else 'FAIL'}")
                print(f"  p99 latency: {p99_actual:.0f}ms (threshold: {p99_ms:.0f}ms) {'PASS' if p99_actual <= p99_ms else 'FAIL'}")
                print(f"  error rate: {actual_error_rate:.4f} (threshold: {error_rate:.4f}) {'PASS' if actual_error_rate <= error_rate else 'FAIL'}")

                failures_list = []
                if p95_actual > p95_ms:
                    failures_list.append(f"p95 latency {p95_actual:.0f}ms > {p95_ms:.0f}ms")
                if p99_actual > p99_ms:
                    failures_list.append(f"p99 latency {p99_actual:.0f}ms > {p99_ms:.0f}ms")
                if actual_error_rate > error_rate:
                    failures_list.append(f"error rate {actual_error_rate:.4f} > {error_rate:.4f}")

                if failures_list:
                    print(f"\nSLO violations:")
                    for f_msg in failures_list:
                        print(f"  - {f_msg}")
                    return False
                return True

    print("No aggregated row found in CSV")
    return False


def main():
    parser = argparse.ArgumentParser(description="Check Locust results against SLO thresholds")
    parser.add_argument("--csv", required=True, help="Path to Locust CSV stats file")
    parser.add_argument("--p95-latency-ms", type=float, default=5000, help="p95 latency threshold (ms)")
    parser.add_argument("--p99-latency-ms", type=float, default=10000, help="p99 latency threshold (ms)")
    parser.add_argument("--error-rate", type=float, default=0.01, help="Maximum error rate (0.01 = 1%)")
    args = parser.parse_args()

    passed = check_slo(args.csv, args.p95_latency_ms, args.p99_latency_ms, args.error_rate)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
