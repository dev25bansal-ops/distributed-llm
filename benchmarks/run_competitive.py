#!/usr/bin/env python3
"""Automated competitive benchmark — DistLLM vs vLLM vs TGI vs llama.cpp.

Runs the same workloads against multiple inference engines and produces
a comparison report with latency, throughput, cost, and quality metrics.

Usage:
    python benchmarks/run_competitive.py --engines distllm,vllm,llama.cpp
    python benchmarks/run_competitive.py --model meta-llama/Llama-3.1-8B --gpus 1
    python benchmarks/run_competitive.py --output results/report.json
"""

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkResult:
    """Result from a single engine benchmark."""
    engine: str
    model: str
    hardware: str
    # Latency metrics (ms)
    ttft_p50: float = 0.0
    ttft_p95: float = 0.0
    ttft_p99: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    # Throughput
    throughput_tok_s: float = 0.0
    max_throughput_tok_s: float = 0.0
    # Cost
    cost_per_1m_tokens: float = 0.0
    gpu_cost_per_hour: float = 0.0
    # Quality
    output_match_pct: float = 100.0  # % match vs reference
    # Context
    max_context_length: int = 0
    max_batch_size: int = 0
    # Metadata
    timestamp: float = field(default_factory=time.time)
    commit_hash: str = ""
    notes: str = ""


@dataclass
class Workload:
    """A benchmark workload definition."""
    name: str
    description: str
    prompt: str
    max_tokens: int = 256
    num_requests: int = 100
    concurrency: int = 1
    temperature: float = 0.0


# Standard benchmark workloads
WORKLOADS = [
    Workload(
        name="short_qa",
        description="Short question answering (typical chatbot use)",
        prompt="What is the capital of France?",
        max_tokens=50,
        num_requests=200,
        concurrency=10,
    ),
    Workload(
        name="long_generation",
        description="Long-form text generation",
        prompt="Write a detailed essay about the history of artificial intelligence, covering key milestones from the 1950s to present day.",
        max_tokens=512,
        num_requests=50,
        concurrency=5,
    ),
    Workload(
        name="code_generation",
        description="Code generation task",
        prompt="Write a Python function that implements a binary search tree with insert, delete, and search operations. Include type hints and docstrings.",
        max_tokens=256,
        num_requests=100,
        concurrency=10,
    ),
    Workload(
        name="summarization",
        description="Document summarization",
        prompt="Summarize the following text in 3 bullet points: Distributed systems are complex. They require careful coordination between nodes. Network partitions can cause split-brain scenarios. Consensus algorithms like Raft help maintain consistency. However, they add latency to every operation.",
        max_tokens=100,
        num_requests=150,
        concurrency=15,
    ),
    Workload(
        name="batch_throughput",
        description="Maximum batch throughput test",
        prompt="Hello, how are you?",
        max_tokens=20,
        num_requests=500,
        concurrency=50,
    ),
]


class CompetitiveBenchmark:
    """Runs benchmarks against multiple inference engines."""

    def __init__(
        self,
        engines: list[str] | None = None,
        model: str = "meta-llama/Llama-3.1-8B",
        hardware: str = "1x A100-80GB",
        gpu_cost_per_hour: float = 1.80,
        output_dir: str = "benchmarks/results",
    ):
        self.engines = engines or ["distllm", "vllm", "llama.cpp"]
        self.model = model
        self.hardware = hardware
        self.gpu_cost_per_hour = gpu_cost_per_hour
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[BenchmarkResult] = []

    def run_all(self, workloads: list[Workload] | None = None) -> list[BenchmarkResult]:
        """Run all workloads against all engines."""
        workloads = workloads or WORKLOADS
        results = []

        for engine in self.engines:
            print(f"\n{'='*60}")
            print(f"  Benchmarking: {engine}")
            print(f"{'='*60}")

            for workload in workloads:
                print(f"\n  Workload: {workload.name} ({workload.description})")
                result = self._run_workload(engine, workload)
                results.append(result)
                self._print_result(result)

        self.results = results
        return results

    def _run_workload(self, engine: str, workload: Workload) -> BenchmarkResult:
        """Run a single workload against a single engine."""
        # In production, this would make real HTTP requests to the engine
        # For now, return a placeholder that demonstrates the structure
        return BenchmarkResult(
            engine=engine,
            model=self.model,
            hardware=self.hardware,
            gpu_cost_per_hour=self.gpu_cost_per_hour,
            notes=f"Workload: {workload.name}",
        )

    def _print_result(self, result: BenchmarkResult) -> None:
        """Print a benchmark result."""
        print(f"    TTFT P50: {result.ttft_p50:.0f}ms")
        print(f"    Throughput: {result.throughput_tok_s:.1f} tok/s")
        print(f"    Cost/1M tokens: ${result.cost_per_1m_tokens:.2f}")

    def generate_report(self) -> dict:
        """Generate a comparison report."""
        report = {
            "model": self.model,
            "hardware": self.hardware,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "engines": {},
            "workloads": [w.name for w in WORKLOADS],
        }

        for engine in self.engines:
            engine_results = [r for r in self.results if r.engine == engine]
            if engine_results:
                avg_throughput = sum(r.throughput_tok_s for r in engine_results) / len(engine_results)
                avg_ttft = sum(r.ttft_p50 for r in engine_results) / len(engine_results)
                report["engines"][engine] = {
                    "avg_throughput_tok_s": round(avg_throughput, 1),
                    "avg_ttft_ms": round(avg_ttft, 0),
                    "results": [asdict(r) for r in engine_results],
                }

        return report

    def save_report(self, filename: str = "competitive_benchmark.json") -> Path:
        """Save the benchmark report to disk."""
        report = self.generate_report()
        path = self.output_dir / filename
        path.write_text(json.dumps(report, indent=2))
        print(f"\nReport saved to: {path}")
        return path

    def compare_with_baseline(self, baseline_path: str) -> list[dict]:
        """Compare current results with a baseline."""
        baseline = json.loads(Path(baseline_path).read_text())
        regressions = []

        for engine in self.engines:
            current = [r for r in self.results if r.engine == engine]
            baseline_engine = baseline.get("engines", {}).get(engine, {})
            baseline_throughput = baseline_engine.get("avg_throughput_tok_s", 0)

            if current and baseline_throughput > 0:
                avg_throughput = sum(r.throughput_tok_s for r in current) / len(current)
                change_pct = ((avg_throughput - baseline_throughput) / baseline_throughput) * 100

                if change_pct < -10:  # More than 10% regression
                    regressions.append({
                        "engine": engine,
                        "metric": "throughput",
                        "baseline": baseline_throughput,
                        "current": round(avg_throughput, 1),
                        "change_pct": round(change_pct, 1),
                    })

        return regressions


def main():
    parser = argparse.ArgumentParser(description="Competitive benchmark suite")
    parser.add_argument("--engines", default="distllm,vllm,llama.cpp", help="Comma-separated engine list")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B", help="Model to benchmark")
    parser.add_argument("--hardware", default="1x A100-80GB", help="Hardware description")
    parser.add_argument("--gpu-cost", type=float, default=1.80, help="GPU cost per hour")
    parser.add_argument("--output", default="benchmarks/results/competitive_benchmark.json", help="Output path")
    parser.add_argument("--compare", default=None, help="Baseline to compare against")
    parser.add_argument("--workloads", default=None, help="Comma-separated workload names")
    args = parser.parse_args()

    engines = [e.strip() for e in args.engines.split(",")]
    workloads = None
    if args.workloads:
        names = [w.strip() for w in args.workloads.split(",")]
        workloads = [w for w in WORKLOADS if w.name in names]

    benchmark = CompetitiveBenchmark(
        engines=engines,
        model=args.model,
        hardware=args.hardware,
        gpu_cost_per_hour=args.gpu_cost,
    )

    benchmark.run_all(workloads)
    report_path = benchmark.save_report(Path(args.output).name)

    if args.compare:
        regressions = benchmark.compare_with_baseline(args.compare)
        if regressions:
            print(f"\n⚠️  Regressions detected:")
            for r in regressions:
                print(f"  {r['engine']}: {r['metric']} {r['change_pct']}% (baseline: {r['baseline']}, current: {r['current']})")
        else:
            print("\n✅ No regressions detected")


if __name__ == "__main__":
    main()
