"""Benchmark suite: throughput vs model size, nodes, and batch size.

Measures and reports:
- Throughput (tokens/second) as a function of model parameters
- Scaling efficiency with number of nodes/GPUs
- Optimal batch size for given hardware
- Latency breakdown by pipeline stage
- Memory usage vs model size

Produces CSV reports and JSON summaries for analysis.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from loguru import logger


@dataclass
class BenchmarkResult:
    model_size_b: float = 0.0       # Billions of parameters
    num_nodes: int = 1
    num_gpus_per_node: int = 1
    batch_size: int = 1
    seq_len: int = 512
    max_tokens: int = 128
    precision: str = "fp16"

    throughput_tok_s: float = 0.0
    prefill_latency_ms: float = 0.0
    decode_latency_ms: float = 0.0
    memory_used_gb: float = 0.0
    memory_peak_gb: float = 0.0
    ttft_ms: float = 0.0           # Time to first token
    itl_ms: float = 0.0            # Inter-token latency

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def csv_header(self) -> List[str]:
        return list(asdict(self).keys())

    def csv_row(self) -> List[str]:
        return [str(v) for v in asdict(self).values()]


@dataclass
class ScalingConfig:
    model_sizes_b: List[float] = field(default_factory=lambda: [1.0, 3.0, 7.0, 13.0, 34.0, 70.0])
    node_counts: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    seq_lengths: List[int] = field(default_factory=lambda: [512, 1024, 2048, 4096])
    warmup_iters: int = 3
    benchmark_iters: int = 10
    output_dir: str = "./benchmark_results"


class ScalingBenchmark:
    """Runs scaling benchmarks to measure throughput across configurations.

    Usage:
        benchmark = ScalingBenchmark(
            forward_fn=model.forward,
            config=ScalingConfig(),
        )
        results = benchmark.run_all()
        benchmark.report(results)
    """

    def __init__(
        self,
        forward_fn: Optional[Callable] = None,
        config: Optional[ScalingConfig] = None,
        device: str = "cuda",
    ):
        self._forward_fn = forward_fn
        self._config = config or ScalingConfig()
        self._device = device
        self._output_dir = Path(self._config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def run_all(self) -> List[BenchmarkResult]:
        """Run benchmarks across all configured model sizes, nodes, and batch sizes."""
        results: List[BenchmarkResult] = []

        for model_size in self._config.model_sizes_b:
            for batch_size in self._config.batch_sizes:
                for seq_len in self._config.seq_lengths:
                    for num_nodes in self._config.node_counts:
                        logger.info(
                            f"Benchmark: {model_size}B, {num_nodes} nodes, "
                            f"batch={batch_size}, seq={seq_len}"
                        )
                        result = self.run_single(
                            model_size_b=model_size,
                            batch_size=batch_size,
                            seq_len=seq_len,
                            num_nodes=num_nodes,
                        )
                        results.append(result)
                        self._save_result(result)

        self._save_all(results)
        return results

    def run_single(
        self,
        model_size_b: float,
        batch_size: int,
        seq_len: int,
        num_nodes: int = 1,
        num_gpus_per_node: int = 1,
        max_tokens: int = 128,
    ) -> BenchmarkResult:
        """Run a single benchmark configuration.

        If forward_fn is provided, measures actual performance.
        Otherwise estimates based on a performance model.
        """
        result = BenchmarkResult(
            model_size_b=model_size_b,
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            batch_size=batch_size,
            seq_len=seq_len,
            max_tokens=max_tokens,
        )

        if self._forward_fn is not None:
            return self._measure_actual(result)
        return self._estimate(result)

    def _measure_actual(self, result: BenchmarkResult) -> BenchmarkResult:
        """Measure actual performance using the forward function."""
        dtype = torch.float16 if result.precision == "fp16" else torch.float32
        hidden_dim = self._estimate_hidden_dim(result.model_size_b)

        for _ in range(self._config.warmup_iters):
            x = torch.randn(result.batch_size, result.seq_len, hidden_dim, device=self._device, dtype=dtype)
            self._forward_fn(x)

        # Measure prefill
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        for _ in range(self._config.benchmark_iters):
            x = torch.randn(result.batch_size, result.seq_len, hidden_dim, device=self._device, dtype=dtype)
            self._forward_fn(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        prefill_time = (time.time() - start) / self._config.benchmark_iters

        result.prefill_latency_ms = prefill_time * 1000

        # Measure decode (single token)
        x = torch.randn(result.batch_size, 1, hidden_dim, device=self._device, dtype=dtype)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        for _ in range(self._config.benchmark_iters):
            self._forward_fn(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        decode_time = (time.time() - start) / self._config.benchmark_iters

        result.decode_latency_ms = decode_time * 1000
        result.itl_ms = decode_time * 1000
        result.ttft_ms = result.prefill_latency_ms

        # Throughput
        tokens_per_step = result.batch_size * result.seq_len
        result.throughput_tok_s = tokens_per_step / max(prefill_time, 1e-12)

        # Memory
        if torch.cuda.is_available():
            result.memory_used_gb = torch.cuda.memory_allocated() / (1024**3)
            result.memory_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

        return result

    def _estimate(self, result: BenchmarkResult) -> BenchmarkResult:
        """Estimate performance using a roofline model."""
        hidden_dim = self._estimate_hidden_dim(result.model_size_b)
        num_layers = self._estimate_num_layers(result.model_size_b)
        num_heads = self._estimate_num_heads(result.model_size_b)
        head_dim = hidden_dim // max(num_heads, 1)

        # Compute FLOPS
        batch = result.batch_size
        seq = result.seq_len

        # Attention: 2 * batch * seq^2 * num_heads * head_dim
        attn_flops = 2 * batch * seq * seq * num_heads * head_dim

        # MLP: 6 * batch * seq * hidden_dim^2 (approx)
        mlp_flops = 6 * batch * seq * hidden_dim * hidden_dim * 4  # 4x for gated MLP

        total_flops = num_layers * (attn_flops + mlp_flops)

        # Available FLOPS (estimate based on GPU)
        gpu_flops = self._estimate_gpu_flops(result.num_nodes * result.num_gpus_per_node)

        # Communication overhead
        comm_overhead = 1.0 - 0.05 * (result.num_nodes - 1)

        compute_time = total_flops / max(gpu_flops * comm_overhead, 1)

        result.prefill_latency_ms = compute_time * 1000

        # Decode: similar but with seq_len=1
        attn_flops_decode = 2 * batch * 1 * 1 * num_heads * head_dim * seq  # KV cache attention
        mlp_flops_decode = 6 * batch * 1 * hidden_dim * hidden_dim * 4
        decode_flops = num_layers * (attn_flops_decode + mlp_flops_decode)
        decode_time = decode_flops / max(gpu_flops, 1)

        result.decode_latency_ms = decode_time * 1000
        result.itl_ms = decode_time * 1000
        result.ttft_ms = result.prefill_latency_ms

        tokens_per_step = batch * seq
        result.throughput_tok_s = tokens_per_step / max(compute_time, 1e-12)

        # Memory estimate
        mem_per_param = 2 if result.precision == "fp16" else 4
        result.memory_used_gb = result.model_size_b * 1e9 * mem_per_param / (1024**3)
        result.memory_peak_gb = result.memory_used_gb * 1.2

        return result

    def _estimate_hidden_dim(self, model_size_b: float) -> int:
        sizes = {1.0: 2048, 3.0: 3200, 7.0: 4096, 13.0: 5120, 34.0: 7168, 70.0: 8192}
        closest = min(sizes.keys(), key=lambda k: abs(k - model_size_b))
        return sizes[closest]

    def _estimate_num_layers(self, model_size_b: float) -> int:
        sizes = {1.0: 24, 3.0: 32, 7.0: 32, 13.0: 40, 34.0: 48, 70.0: 80}
        closest = min(sizes.keys(), key=lambda k: abs(k - model_size_b))
        return sizes[closest]

    def _estimate_num_heads(self, model_size_b: float) -> int:
        sizes = {1.0: 16, 3.0: 32, 7.0: 32, 13.0: 40, 34.0: 64, 70.0: 64}
        closest = min(sizes.keys(), key=lambda k: abs(k - model_size_b))
        return sizes[closest]

    def _estimate_gpu_flops(self, num_gpus: int) -> float:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_flops = props.multi_processor_count * 256 * 2 * props.clock_rate * 1e3
            return gpu_flops * num_gpus
        return 100e12 * num_gpus  # ~100 TFLOPS per GPU

    # -------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------

    def _save_result(self, result: BenchmarkResult) -> None:
        path = self._output_dir / f"benchmark_{result.model_size_b}b_b{result.batch_size}_s{result.seq_len}.json"
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def _save_all(self, results: List[BenchmarkResult]) -> None:
        path = self._output_dir / "all_results.csv"
        if not results:
            return
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(results[0].csv_header())
            for r in results:
                writer.writerow(r.csv_row())
        logger.info(f"Saved {len(results)} benchmark results to {path}")

    def report(self, results: List[BenchmarkResult]) -> str:
        """Generate a human-readable report from benchmark results."""
        lines = ["=" * 80, "DistLLM Scaling Benchmark Report", "=" * 80]

        if not results:
            return "\n".join(lines) + "\nNo results."

        # Best throughput
        best = max(results, key=lambda r: r.throughput_tok_s)
        lines.append(f"\nBest throughput: {best.throughput_tok_s:.0f} tok/s")
        lines.append(f"  Config: {best.model_size_b}B, {best.num_nodes} nodes, batch={best.batch_size}, seq={best.seq_len}")

        # Summary table by model size
        lines.append(f"\n{'Model':>8} {'Nodes':>6} {'Batch':>6} {'Seq':>6} {'Throughput':>12} {'TTFT':>8} {'ITL':>8} {'Memory':>8}")
        lines.append("-" * 70)
        for r in sorted(results, key=lambda r: (-r.model_size_b, r.throughput_tok_s))[:20]:
            lines.append(
                f"{r.model_size_b:>6.1f}B {r.num_nodes:>6} {r.batch_size:>6} {r.seq_len:>6} "
                f"{r.throughput_tok_s:>10.0f} {r.ttft_ms:>7.1f}ms {r.itl_ms:>6.2f}ms {r.memory_used_gb:>6.1f}GB"
            )

        return "\n".join(lines)
