"""Generate a performance report from k6 and Locust test results."""

import argparse
import json
import os
from datetime import datetime


def load_k6_results(k6_dir: str) -> dict:
    """Load k6 JSON results if available."""
    results = {}
    for f in os.listdir(k6_dir):
        if f.endswith(".json"):
            with open(os.path.join(k6_dir, f)) as fp:
                results[f] = json.load(fp)
    return results


def load_locust_results(locust_dir: str) -> dict:
    """Load Locust CSV results."""
    results = {}
    for f in os.listdir(locust_dir):
        if f.endswith(".csv"):
            path = os.path.join(locust_dir, f)
            with open(path) as fp:
                results[f] = fp.read()
    return results


def generate_report(k6_data: dict, locust_data: dict) -> str:
    """Generate markdown performance report."""
    lines = [
        "# Performance Test Report",
        f"Generated: {datetime.utcnow().isoformat()}",
        "",
        "## k6 Load Test Results",
    ]

    for filename, data in k6_data.items():
        lines.append(f"\n### {filename}")
        if isinstance(data, list) and len(data) > 0:
            metrics = data[0].get("metrics", {}) if isinstance(data[0], dict) else {}
            for metric_name, metric_data in metrics.items():
                avg = metric_data.get("avg", "N/A")
                p95 = metric_data.get("p(95)", "N/A")
                lines.append(f"- **{metric_name}**: avg={avg}, p95={p95}")

    lines.extend(["", "## Locust Results", ""])
    for filename, content in locust_data.items():
        lines.append(f"\n### {filename}")
        lines.append("```")
        lines.append(content[:2000])  # Limit output
        lines.append("```")

    lines.extend([
        "",
        "## Summary",
        "- k6 tests executed: " + str(len(k6_data)),
        "- Locust result files: " + str(len(locust_data)),
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k6", required=True)
    parser.add_argument("--locust", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    k6_data = load_k6_results(args.k6)
    locust_data = load_locust_results(args.locust)
    report = generate_report(k6_data, locust_data)

    with open(args.output, "w") as f:
        f.write(report)

    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
