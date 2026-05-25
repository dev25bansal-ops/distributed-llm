#!/usr/bin/env python3
"""Benchmark: Ray pipeline performance.

Since gRPC has been removed in favor of Ray-native transport, this script
benchmarks Ray object store latency, remote call overhead, and end-to-end
generation throughput.

Usage:
    python benchmarks/ray_vs_grpc.py
    python benchmarks/ray_vs_grpc.py --num-trials 200 --num-runs 10
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

RAY_AVAILABLE = False
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    ray = None


# ── Benchmark 1: Ray Object Store Tensor Transfer Latency ────────────────────

def benchmark_ray_put_latency(
    size_mb: list[int],
    num_trials: int = 100,
) -> dict[str, float]:
    """Measure ray.put + ray.get round-trip latency for tensors of given sizes.

    Args:
        size_mb: List of tensor sizes in megabytes to test.
        num_trials: Number of round-trip iterations per size.

    Returns:
        Dict mapping size label -> {"mean_ms", "p50_ms", "p99_ms", "min_ms", "max_ms"}.
    """
    if not RAY_AVAILABLE:
        return {"error": 0.0, "error_msg": "ray not installed"}

    results = {}
    for mb in size_mb:
        tensor = torch.randn(int(mb * 256 * 1024), dtype=torch.float16)
        latencies = []
        for _ in range(num_trials):
            start = time.perf_counter()
            ref = ray.put(tensor)
            _ = ray.get(ref)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        results[f"{mb}MB"] = {
            "mean_ms": statistics.mean(latencies),
            "p50_ms": statistics.median(latencies),
            "p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
            "min_ms": min(latencies),
            "max_ms": max(latencies),
        }
    return results


# ── Benchmark 2: Ray Remote Call Latency ─────────────────────────────────────

if RAY_AVAILABLE:
    @ray.remote(num_cpus=1)
    class _BenchmarkActor:
        def echo(self, x: torch.Tensor) -> torch.Tensor:
            return x

        def forward_small(self) -> dict:
            return {"status": "ok", "value": 42}
else:
    class _BenchmarkActor:
        def echo(self, x: torch.Tensor) -> torch.Tensor:
            return x
        def forward_small(self) -> dict:
            return {"status": "ok", "value": 42}


def benchmark_ray_remote_latency(num_trials: int = 100) -> dict[str, float]:
    """Measure Ray actor remote call round-trip latency.

    Creates a single actor and calls .remote() + ray.get() repeatedly.

    Args:
        num_trials: Number of remote call iterations.

    Returns:
        Dict with mean/p50/p99/min/max latency in milliseconds.
    """
    if not RAY_AVAILABLE:
        return {"error": 0.0, "error_msg": "ray not installed"}

    actor = _BenchmarkActor.remote()
    ray.get(actor.echo.remote(torch.tensor([1])))
    latencies = []
    for _ in range(num_trials):
        start = time.perf_counter()
        ref = actor.forward_small.remote()
        _ = ray.get(ref)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    return {
        "mean_ms": statistics.mean(latencies),
        "p50_ms": statistics.median(latencies),
        "p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }


# ── Benchmark 3: End-to-End Generation Throughput ────────────────────────────

if RAY_AVAILABLE:
    @ray.remote(num_cpus=1)
    class _MockPipelineStage:
        def __init__(self, stage_id: int, delay_s: float = 0.01):
            self.stage_id = stage_id
            self.delay_s = delay_s

        def forward(self, x: Optional[torch.Tensor]) -> torch.Tensor:
            if self.delay_s > 0:
                time.sleep(self.delay_s)
            if x is None:
                x = torch.randn(1, 4096)
            return x * 1.0
else:
    class _MockPipelineStage:
        def __init__(self, stage_id: int, delay_s: float = 0.01):
            self.stage_id = stage_id
            self.delay_s = delay_s

        def forward(self, x: Optional[torch.Tensor]) -> torch.Tensor:
            if self.delay_s > 0:
                time.sleep(self.delay_s)
            if x is None:
                x = torch.randn(1, 4096)
            return x * 1.0


def benchmark_end_to_end(
    prompt: str,
    num_tokens: int = 128,
    num_runs: int = 5,
    num_stages: int = 4,
) -> dict[str, float]:
    """Benchmark end-to-end token generation throughput with a mock Ray pipeline.

    Creates a chain of Ray actors simulating pipeline stages and measures
    tokens-per-second throughput.

    Args:
        prompt: Input text (used for token count estimation).
        num_tokens: Number of tokens to generate per run.
        num_runs: Number of full generation runs.
        num_stages: Number of pipeline stages (actors).

    Returns:
        Dict with throughput metrics.
    """
    if not RAY_AVAILABLE:
        return {"error": 0.0, "error_msg": "ray not installed"}

    stages = [_MockPipelineStage.remote(i) for i in range(num_stages)]
    ray.get([s.forward.remote(None) for s in stages])
    prompt_tokens = max(1, len(prompt) // 4)

    run_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        x = None
        for _ in range(num_tokens):
            for stage in stages:
                x = stage.forward.remote(x)
            x = ray.get(x)
        elapsed = time.perf_counter() - start
        run_times.append(elapsed)

    total_tokens = num_tokens * num_runs
    total_time = sum(run_times)
    throughputs = [num_tokens / t for t in run_times]

    return {
        "prompt_tokens": prompt_tokens,
        "num_tokens_per_run": num_tokens,
        "num_runs": num_runs,
        "num_stages": num_stages,
        "mean_throughput_tok_s": statistics.mean(throughputs),
        "median_throughput_tok_s": statistics.median(throughputs),
        "mean_time_per_run_s": statistics.mean(run_times),
        "total_time_s": total_time,
        "total_tokens_generated": total_tokens,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_ms(d: dict) -> str:
    if "error" in d:
        return f"[SKIP] {d.get('error_msg', 'N/A')}"
    return (
        f"mean={d['mean_ms']:.3f}ms  "
        f"p50={d['p50_ms']:.3f}ms  "
        f"p99={d['p99_ms']:.3f}ms  "
        f"min={d['min_ms']:.3f}ms  "
        f"max={d['max_ms']:.3f}ms"
    )


def _format_throughput(d: dict) -> str:
    if "error" in d:
        return f"[SKIP] {d.get('error_msg', 'N/A')}"
    return (
        f"{d['mean_throughput_tok_s']:.1f} tok/s "
        f"(median: {d['median_throughput_tok_s']:.1f} tok/s, "
        f"{d['num_runs']} runs x {d['num_tokens_per_run']} tok, "
        f"{d['num_stages']} stages)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Ray pipeline performance"
    )
    parser.add_argument(
        "--num-trials", type=int, default=100,
        help="Number of iterations for latency benchmarks (default: 100)",
    )
    parser.add_argument(
        "--num-runs", type=int, default=5,
        help="Number of end-to-end generation runs (default: 5)",
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[1, 10, 100],
        help="Tensor sizes in MB for put/get benchmark (default: 1 10 100)",
    )
    parser.add_argument(
        "--tokens", type=int, default=128,
        help="Tokens per generation run (default: 128)",
    )
    parser.add_argument(
        "--stages", type=int, default=4,
        help="Number of pipeline stages (default: 4)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results JSON to this path",
    )
    args = parser.parse_args()

    if not RAY_AVAILABLE:
        print("Ray is not installed. Install with: pip install ray[default]")
        sys.exit(0)

    ray.init(ignore_reinit_error=True, log_to_driver=False)

    print("=" * 72)
    print("  Ray Pipeline Performance Benchmarks")
    print("=" * 72)
    print(f"\n  Ray version: {ray.__version__}")
    print(f"  PyTorch version: {torch.__version__}")

    # Benchmark 1
    print(f"\n{'─' * 72}")
    print("  1. Object Store Tensor Transfer Latency (ray.put + ray.get)")
    print(f"{'─' * 72}")
    put_results = benchmark_ray_put_latency(args.sizes, args.num_trials)
    for label, stats in put_results.items():
        print(f"    {label:<8}  {_format_ms(stats)}")

    # Benchmark 2
    print(f"\n{'─' * 72}")
    print("  2. Ray Remote Call Latency (actor.forward.remote + ray.get)")
    print(f"{'─' * 72}")
    remote_results = benchmark_ray_remote_latency(args.num_trials)
    print(f"    {'':<8}  {_format_ms(remote_results)}")

    # Benchmark 3
    print(f"\n{'─' * 72}")
    print("  3. End-to-End Generation Throughput")
    print(f"{'─' * 72}")
    prompt = "Explain the benefits of distributed computing for LLM inference."
    throughput_results = benchmark_end_to_end(
        prompt=prompt,
        num_tokens=args.tokens,
        num_runs=args.num_runs,
        num_stages=args.stages,
    )
    print(f"    {_format_throughput(throughput_results)}")

    print(f"\n{'─' * 72}")
    print("  Done")
    print(f"{'─' * 72}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "ray_version": ray.__version__,
                    "torch_version": torch.__version__,
                    "args": vars(args),
                    "put_latency": put_results,
                    "remote_latency": remote_results,
                    "throughput": throughput_results,
                },
                f,
                indent=2,
            )
        print(f"\n  Results saved to {output_path}")

    ray.shutdown()


if __name__ == "__main__":
    main()
