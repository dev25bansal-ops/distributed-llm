#!/usr/bin/env python3
"""Benchmark comparison: Distributed LLM vs vLLM vs SGLang.

Runs identical prompts through each backend and compares:
- Time to first token (TTFT)
- Throughput (tokens/sec)
- Latency percentiles (p50, p95, p99)
- End-to-end generation time

Usage:
    # Run all benchmarks
    python benchmarks/compare.py

    # Run with specific model
    python benchmarks/compare.py --model meta-llama/Llama-3.1-8B-Instruct

    # Run with custom prompts file
    python benchmarks/compare.py --prompts benchmarks/prompts.txt

    # Run specific backend only
    python benchmarks/compare.py --backends distllm vllm

Requirements:
    pip install httpx tabulate
    # vLLM: pip install vllm
    # SGLang: pip install sglang
"""

import os
import sys
import time
import json
import argparse
import asyncio
import subprocess
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path

import httpx


# --- Configuration ---

DEFAULT_PROMPTS = [
    "Explain the concept of pipeline parallelism in one paragraph.",
    "What is the time complexity of self-attention in a Transformer?",
    "Write a Python function to compute the Fibonacci sequence iteratively.",
    "Describe the differences between batch processing and stream processing.",
    "What are the advantages of using KV caching in autoregressive generation?",
    "Explain how speculative decoding works and when it's effective.",
    "Compare data parallelism, tensor parallelism, and pipeline parallelism.",
    "What is the purpose of rotary positional embeddings (RoPE)?",
]

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_MAX_TOKENS = 128
DEFAULT_NUM_WARMUP = 2
DEFAULT_NUM_RUNS = 5


@dataclass
class BenchmarkResult:
    """Results for a single benchmark run."""
    backend: str
    prompt: str
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float  # Time to first token
    total_ms: float  # Total generation time
    throughput: float  # tokens/sec
    output: str = ""


@dataclass
class BackendConfig:
    """Configuration for a backend system."""
    name: str
    api_base: str
    model: str
    start_command: Optional[List[str]] = None
    stop_command: Optional[List[str]] = None
    ready_check: str = "/health"


# --- Benchmark Runner ---

class BenchmarkRunner:
    """Runs benchmarks against a specified OpenAI-compatible API."""

    def __init__(
        self,
        api_base: str,
        model: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,  # Deterministic
        timeout: float = 120.0,
    ):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    async def run_prompt(self, prompt: str) -> BenchmarkResult:
        """Run a single prompt and measure metrics."""
        # Measure TTFT via streaming
        ttft_ms = None
        tokens_generated = 0
        output_chunks = []
        start_time = time.perf_counter()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.api_base}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "stream": True,
                },
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line or line == "data: [DONE]":
                        continue
                    if not line.startswith("data: "):
                        continue

                    try:
                        data = json.loads(line[6:])
                        choice = data.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")

                        if content:
                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - start_time) * 1000
                            tokens_generated += 1
                            output_chunks.append(content)
                    except json.JSONDecodeError:
                        continue

        total_ms = (time.perf_counter() - start_time) * 1000
        throughput = (tokens_generated / total_ms * 1000) if total_ms > 0 else 0

        # Count prompt tokens via non-streaming call
        prompt_tokens = await self._count_tokens(prompt)

        return BenchmarkResult(
            backend=self.model,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            generated_tokens=tokens_generated,
            ttft_ms=ttft_ms or 0,
            total_ms=total_ms,
            throughput=throughput,
            output="".join(output_chunks),
        )

    async def _count_tokens(self, prompt: str) -> int:
        """Estimate prompt token count (rough approximation)."""
        # Rough: ~4 chars per token for English text
        return max(1, len(prompt) // 4)

    async def run_benchmark(
        self,
        prompts: List[str],
        num_warmup: int = DEFAULT_NUM_WARMUP,
        num_runs: int = DEFAULT_NUM_RUNS,
    ) -> List[BenchmarkResult]:
        """Run full benchmark suite."""
        results = []

        # Warmup
        print(f"  Warming up ({num_warmup} runs)...")
        for i in range(min(num_warmup, len(prompts))):
            try:
                await self.run_prompt(prompts[i])
            except Exception:
                pass

        # Actual runs
        print(f"  Running {num_runs} iterations over {len(prompts)} prompts...")
        for run_idx in range(num_runs):
            for prompt in prompts:
                try:
                    result = await self.run_prompt(prompt)
                    results.append(result)
                except Exception as e:
                    print(f"  Error on prompt: {e}")
                    results.append(BenchmarkResult(
                        backend=self.model,
                        prompt=prompt,
                        prompt_tokens=0,
                        generated_tokens=0,
                        ttft_ms=0,
                        total_ms=0,
                        throughput=0,
                        output="",
                    ))

        return results


# --- Backend Managers ---

class DistLLMBackend:
    """Manage Distributed LLM server for benchmarking."""

    def __init__(self, model: str, port: int = 8000):
        self.model = model
        self.port = port
        self.api_base = f"http://localhost:{port}"
        self.process = None

    def start(self):
        print(f"Starting DistLLM server with model {self.model}...")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "distllm.api.server", "--model", self.model, "--port", str(self.port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._wait_ready()

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=10)

    def _wait_ready(self, timeout: float = 60):
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = httpx.get(f"{self.api_base}/health", timeout=2)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1)
        raise TimeoutError("DistLLM server did not start in time")


class vLLMBackend:
    """Manage vLLM server for benchmarking."""

    def __init__(self, model: str, port: int = 8001):
        self.model = model
        self.port = port
        self.api_base = f"http://localhost:{port}"
        self.process = None

    def start(self):
        print(f"Starting vLLM server with model {self.model}...")
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                    "--model", self.model,
                    "--port", str(self.port),
                    "--disable-log-requests",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            print("vLLM not installed. Install with: pip install vllm")
            raise
        self._wait_ready()

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=10)

    def _wait_ready(self, timeout: float = 120):
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = httpx.get(f"{self.api_base}/health", timeout=2)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1)
        raise TimeoutError("vLLM server did not start in time")


class SGLangBackend:
    """Manage SGLang server for benchmarking."""

    def __init__(self, model: str, port: int = 8002):
        self.model = model
        self.port = port
        self.api_base = f"http://localhost:{port}"
        self.process = None

    def start(self):
        print(f"Starting SGLang server with model {self.model}...")
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable, "-m", "sglang.launch_server",
                    "--model-path", self.model,
                    "--port", str(self.port),
                    "--log-level", "warning",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            print("SGLang not installed. Install with: pip install sglang")
            raise
        self._wait_ready()

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=10)

    def _wait_ready(self, timeout: float = 120):
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = httpx.get(f"{self.api_base}/health", timeout=2)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1)
        raise TimeoutError("SGLang server did not start in time")


# --- Output Formatting ---

def compute_stats(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Compute aggregate statistics from benchmark results."""
    valid = [r for r in results if r.total_ms > 0]
    if not valid:
        return {"error": "No successful runs"}

    ttfts = [r.ttft_ms for r in valid if r.ttft_ms > 0]
    throughputs = [r.throughput for r in valid if r.throughput > 0]
    totals = [r.total_ms for r in valid]

    def percentile(data: List[float], p: float) -> float:
        if not data:
            return 0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]

    return {
        "successful_runs": len(valid),
        "failed_runs": len(results) - len(valid),
        "ttft_ms": {
            "mean": sum(ttfts) / len(ttfts) if ttfts else 0,
            "p50": percentile(ttfts, 50),
            "p95": percentile(ttfts, 95),
            "p99": percentile(ttfts, 99),
        },
        "throughput_tps": {
            "mean": sum(throughputs) / len(throughputs) if throughputs else 0,
            "p50": percentile(throughputs, 50),
            "p95": percentile(throughputs, 95),
        },
        "total_ms": {
            "mean": sum(totals) / len(totals),
            "p50": percentile(totals, 50),
            "p95": percentile(totals, 95),
        },
    }


def print_comparison_table(all_stats: Dict[str, Dict[str, Any]]):
    """Print a markdown comparison table."""
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON")
    print("=" * 80)

    # Header
    print(f"\n| Metric | {' | '.join(all_stats.keys())} |")
    print(f"|--------|{'|'.join(['--------'] * len(all_stats))}|")

    # TTFT
    ttft_row = ["TTFT (ms) mean"]
    for backend, stats in all_stats.items():
        ttft = stats.get("ttft_ms", {}).get("mean", 0)
        ttft_row.append(f"{ttft:.0f}")
    print(f"| {' | '.join(ttft_row)} |")

    ttft_p95_row = ["TTFT (ms) p95"]
    for backend, stats in all_stats.items():
        ttft = stats.get("ttft_ms", {}).get("p95", 0)
        ttft_p95_row.append(f"{ttft:.0f}")
    print(f"| {' | '.join(ttft_p95_row)} |")

    # Throughput
    tp_row = ["Throughput (tok/s) mean"]
    for backend, stats in all_stats.items():
        tp = stats.get("throughput_tps", {}).get("mean", 0)
        tp_row.append(f"{tp:.1f}")
    print(f"| {' | '.join(tp_row)} |")

    # Total time
    total_row = ["Total (ms) mean"]
    for backend, stats in all_stats.items():
        total = stats.get("total_ms", {}).get("mean", 0)
        total_row.append(f"{total:.0f}")
    print(f"| {' | '.join(total_row)} |")

    total_p95_row = ["Total (ms) p95"]
    for backend, stats in all_stats.items():
        total = stats.get("total_ms", {}).get("p95", 0)
        total_p95_row.append(f"{total:.0f}")
    print(f"| {' | '.join(total_p95_row)} |")

    # Success rate
    success_row = ["Success rate"]
    for backend, stats in all_stats.items():
        ok = stats.get("successful_runs", 0)
        fail = stats.get("failed_runs", 0)
        total = ok + fail
        rate = ok / total * 100 if total > 0 else 0
        success_row.append(f"{rate:.0f}%")
    print(f"| {' | '.join(success_row)} |")

    print("\n" + "=" * 80)


# --- Main ---

async def run_benchmarks(
    backends: List[str],
    model: str,
    prompts: List[str],
    num_warmup: int,
    num_runs: int,
    max_tokens: int,
):
    """Run benchmarks for specified backends."""
    backend_map = {
        "distllm": DistLLMBackend,
        "vllm": vLLMBackend,
        "sglang": SGLangBackend,
    }

    all_stats = {}

    for backend_name in backends:
        if backend_name not in backend_map:
            print(f"Unknown backend: {backend_name}")
            continue

        backend_cls = backend_map[backend_name]
        backend = backend_cls(model=model)

        try:
            backend.start()
            runner = BenchmarkRunner(
                api_base=backend.api_base,
                model=model,
                max_tokens=max_tokens,
            )

            print(f"\nBenchmarking {backend_name}...")
            results = await runner.run_benchmark(
                prompts=prompts,
                num_warmup=num_warmup,
                num_runs=num_runs,
            )

            stats = compute_stats(results)
            all_stats[backend_name] = stats

            print(f"  {backend_name}: {stats.get('successful_runs', 0)} successful runs")
            ttft = stats.get("ttft_ms", {}).get("mean", 0)
            tp = stats.get("throughput_tps", {}).get("mean", 0)
            print(f"  TTFT: {ttft:.0f}ms, Throughput: {tp:.1f} tok/s")

        except Exception as e:
            print(f"  {backend_name} failed: {e}")
            all_stats[backend_name] = {"error": str(e)}
        finally:
            try:
                backend.stop()
            except Exception:
                pass

    print_comparison_table(all_stats)

    # Save results
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = output_dir / f"benchmark-{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump({
            "model": model,
            "prompts_count": len(prompts),
            "num_runs": num_runs,
            "max_tokens": max_tokens,
            "results": all_stats,
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark: DistLLM vs vLLM vs SGLang")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to benchmark")
    parser.add_argument("--prompts", type=str, help="Path to prompts file (one per line)")
    parser.add_argument("--backends", nargs="+", default=["distllm", "vllm", "sglang"],
                        choices=["distllm", "vllm", "sglang"], help="Backends to compare")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_NUM_WARMUP)
    parser.add_argument("--runs", type=int, default=DEFAULT_NUM_RUNS)
    parser.add_argument("--external", action="store_true",
                        help="Use externally running servers (don't start/stop)")

    args = parser.parse_args()

    # Load prompts
    if args.prompts and Path(args.prompts).exists():
        prompts = Path(args.prompts).read_text().strip().split("\n")
    else:
        prompts = DEFAULT_PROMPTS

    if args.external:
        # Use externally running servers
        backend_urls = {
            "distllm": "http://localhost:8000",
            "vllm": "http://localhost:8001",
            "sglang": "http://localhost:8002",
        }
        all_stats = {}
        for backend_name in args.backends:
            url = backend_urls.get(backend_name)
            if not url:
                continue
            print(f"\nBenchmarking {backend_name} at {url}...")
            runner = BenchmarkRunner(api_base=url, model=args.model, max_tokens=args.max_tokens)
            results = asyncio.run(runner.run_benchmark(prompts, args.warmup, args.runs))
            stats = compute_stats(results)
            all_stats[backend_name] = stats
            print(f"  TTFT: {stats.get('ttft_ms', {}).get('mean', 0):.0f}ms, "
                  f"Throughput: {stats.get('throughput_tps', {}).get('mean', 0):.1f} tok/s")
        print_comparison_table(all_stats)
    else:
        asyncio.run(run_benchmarks(
            backends=args.backends,
            model=args.model,
            prompts=prompts,
            num_warmup=args.warmup,
            num_runs=args.runs,
            max_tokens=args.max_tokens,
        ))


if __name__ == "__main__":
    main()
