#!/usr/bin/env python3
"""Benchmark suite for distributed LLM inference.

Measures:
- Single-node throughput (tokens/sec)
- Distributed throughput (tokens/sec)
- Per-node latency breakdown
- Network bandwidth utilization
- Memory usage

Usage:
    python benchmarks/run.py --model roneneldan/TinyStories-1M
    python benchmarks/run.py --model gpt2 --mode distributed --nodes localhost:50051:0:5 localhost:50052:6:11
"""

import sys
import os
import time
import json
import argparse
import statistics
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from coordinator import Coordinator


class BenchmarkResult:
    """Stores benchmark metrics."""

    def __init__(self, name: str, model: str, mode: str):
        self.name = name
        self.model = model
        self.mode = mode
        self.latencies: List[float] = []
        self.tokens_per_second: List[float] = []
        self.total_tokens = 0
        self.total_time = 0.0
        self.memory_peak = 0
        self.memory_end = 0

    def add_sample(self, latency: float, tokens: int, memory_mb: float = 0):
        self.latencies.append(latency)
        self.tokens_per_second.append(tokens / latency)
        self.total_tokens += tokens
        self.total_time += latency
        self.memory_peak = max(self.memory_peak, memory_mb)
        self.memory_end = memory_mb

    def summary(self) -> dict:
        if not self.latencies:
            return {}
        return {
            "name": self.name,
            "model": self.model,
            "mode": self.mode,
            "total_tokens": self.total_tokens,
            "total_time_sec": round(self.total_time, 3),
            "avg_latency_sec": round(statistics.mean(self.latencies), 4),
            "p50_latency_sec": round(statistics.median(self.latencies), 4),
            "p95_latency_sec": round(sorted(self.latencies)[int(len(self.latencies) * 0.95)], 4),
            "avg_tokens_per_sec": round(statistics.mean(self.tokens_per_second), 2),
            "memory_peak_mb": round(self.memory_peak, 1),
            "memory_end_mb": round(self.memory_end, 1),
            "samples": len(self.latencies),
        }


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    import torch
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def benchmark_local(coordinator: Coordinator, prompts: List[str], max_tokens: int = 50) -> BenchmarkResult:
    """Benchmark local (single-node) inference."""
    result = BenchmarkResult("local", coordinator.model_name, "local")

    for i, prompt in enumerate(prompts):
        start = time.perf_counter()
        mem_before = get_gpu_memory_mb()

        generated = coordinator.generate(prompt, max_new_tokens=max_tokens)

        mem_after = get_gpu_memory_mb()
        elapsed = time.perf_counter() - start

        tokens = len(coordinator.tokenizer.encode(generated))
        result.add_sample(elapsed, tokens, mem_after)

        print(f"  [{i+1}/{len(prompts)}] {len(generated)} chars, {tokens} tokens, {elapsed:.2f}s, {tokens/elapsed:.1f} tok/s")

    return result


def benchmark_distributed(coordinator: Coordinator, prompts: List[str], max_tokens: int = 50) -> BenchmarkResult:
    """Benchmark distributed inference."""
    result = BenchmarkResult("distributed", coordinator.model_name, "distributed")

    for i, prompt in enumerate(prompts):
        start = time.perf_counter()

        generated = coordinator.generate(prompt, max_new_tokens=max_tokens)

        elapsed = time.perf_counter() - start

        tokens = len(coordinator.tokenizer.encode(generated))
        result.add_sample(elapsed, tokens)

        print(f"  [{i+1}/{len(prompts)}] {len(generated)} chars, {tokens} tokens, {elapsed:.2f}s, {tokens/elapsed:.1f} tok/s")

    return result


def run_benchmark(args):
    """Run the benchmark suite."""
    print(f"\n{'='*60}")
    print(f"Distributed LLM Benchmark")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Prompts: {args.num_prompts}")
    print(f"Max tokens: {args.max_tokens}")
    print()

    # Test prompts
    test_prompts = [
        "Once upon a time, there was a",
        "The quick brown fox jumps over",
        "In a world where artificial intelligence",
        "Machine learning has transformed",
        "The future of computing is",
        "Natural language processing enables",
        "Deep learning models have achieved",
        "Neural networks can learn to",
        "The development of large language",
        "Transformer architectures have revolutionized",
    ]
    prompts = test_prompts[:args.num_prompts]

    results = []

    # Local benchmark
    print("[1/2] Running local benchmark...")
    coordinator = Coordinator(model_name=args.model, dtype=args.dtype)
    coordinator.load_local_model()

    result = benchmark_local(coordinator, prompts, args.max_tokens)
    results.append(result.summary())
    print(f"\n  Average: {result.summary()['avg_tokens_per_sec']} tokens/sec\n")

    # Save results
    if args.output:
        output_dir = os.path.dirname(args.output) or "benchmarks/results"
        os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark distributed LLM inference")
    parser.add_argument("--model", type=str, default="roneneldan/TinyStories-1M", help="Model to benchmark")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--num-prompts", type=int, default=5, help="Number of test prompts")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate per prompt")
    parser.add_argument("--output", type=str, default="benchmarks/results/benchmark.json", help="Output file")
    parser.add_argument("--mode", type=str, default="local", choices=["local", "distributed"])

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
