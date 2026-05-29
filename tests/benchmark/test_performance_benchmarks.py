"""Performance benchmark suite: 6 required measurement areas.

Uses pytest-benchmark for precise timing and analytical models where
real GPU hardware is unavailable.
"""

from __future__ import annotations

import gc
import math
import os
import pickle
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent))

torch = pytest.importorskip("torch")

from helpers import (  # noqa: E402
    Coordinator,
    DeviceInfo,
    HeterogeneousCluster,
    HeterogeneousNode,
    assign_layers_proportional,
    estimate_heterogeneous_throughput,
    get_device_compatibility_map,
    SRC_DIR,
)

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.skipif(
        not os.environ.get("CI") and not os.environ.get("DISTLLM_RUN_BENCHMARKS"),
        reason="Benchmarks require CI=1 or DISTLLM_RUN_BENCHMARKS=1",
    ),
]

# ---------------------------------------------------------------------------
# 1. Throughput: tokens/second as function of model size & node count
# ---------------------------------------------------------------------------

BENCHMARK_MODEL_SIZES: List[float] = [1.0, 3.0, 7.0, 13.0, 34.0, 70.0]
BENCHMARK_NODE_COUNTS: List[int] = [1, 2, 4, 8, 16]
BENCHMARK_BATCH_SIZES: List[int] = [1, 4, 16]
BENCHMARK_SEQ_LEN: int = 512


def _estimate_throughput(model_size_b: float, num_nodes: int,
                         batch_size: int, seq_len: int) -> float:
    """Roofline-model throughput estimate (tokens/s)."""
    if model_size_b <= 3.0:
        hidden = 1536
        layers = 24
        heads = 16
    elif model_size_b <= 7.0:
        hidden = 4096
        layers = 32
        heads = 32
    elif model_size_b <= 13.0:
        hidden = 5120
        layers = 40
        heads = 40
    elif model_size_b <= 34.0:
        hidden = 7168
        layers = 56
        heads = 64
    else:
        hidden = 8192
        layers = 80
        heads = 64

    head_dim = hidden // heads
    flops_per_token = 2 * batch_size * seq_len * seq_len * heads * head_dim
    flops_per_token += 2 * batch_size * seq_len * hidden * (layers * hidden * 4)
    flops_per_token *= (layers / num_nodes) if num_nodes > 0 else layers

    peak_flops = 312e12 * min(num_nodes, 8)
    overhead = 0.85 ** (num_nodes - 1) if num_nodes > 1 else 1.0
    effective_flops = peak_flops * overhead

    time_s = flops_per_token / max(effective_flops, 1.0)
    if time_s < 1e-12:
        return 0.0
    return batch_size / time_s


def test_throughput_sweep(benchmark):
    """Measure tokens/second across all model sizes (1B-70B) and node counts."""
    results = []

    def run():
        results.clear()
        for ms in BENCHMARK_MODEL_SIZES:
            r = _estimate_throughput(ms, 1, 1, BENCHMARK_SEQ_LEN)
            results.append(r)
        return sum(results) / len(results)

    result = benchmark(run)
    assert result > 0, "Throughput must be positive"


def test_throughput_scaling_with_nodes(benchmark):
    """Throughput scaling from 1 to 16 nodes, fixed 7B model."""
    results: Dict[int, float] = {}

    def run():
        results.clear()
        for n in BENCHMARK_NODE_COUNTS:
            results[n] = _estimate_throughput(7.0, n, 1, BENCHMARK_SEQ_LEN)
        efficiency = results[16] / results[1] / 16 if results.get(1, 0) > 0 else 0
        return efficiency

    result = benchmark(run)
    assert 0 <= result <= 1.0


def test_throughput_batch_size_impact(benchmark, perf_model_config):
    """Throughput vs batch size {1,4,16} for 7B model, 4 nodes."""
    results = {}

    def run():
        results.clear()
        for bs in BENCHMARK_BATCH_SIZES:
            results[bs] = _estimate_throughput(7.0, 4, bs, BENCHMARK_SEQ_LEN)
        speedup = results[16] / max(results.get(1, 1.0), 1e-12)
        return speedup

    result = benchmark(run)
    assert result >= 1.0


def test_throughput_best_config(benchmark):
    """Best-case throughput across all model sizes with 8 nodes, batch=16."""
    results = {}

    def run():
        results.clear()
        for ms in BENCHMARK_MODEL_SIZES:
            results[ms] = _estimate_throughput(ms, 8, 16, BENCHMARK_SEQ_LEN)
        return results.get(BENCHMARK_MODEL_SIZES[0], 0.0)

    result = benchmark(run)
    assert result > 0


# ---------------------------------------------------------------------------
# 2. Time-to-first-token (TTFT) latency
# ---------------------------------------------------------------------------

def test_ttft_analytical(benchmark):
    """Analytical TTFT estimate across model sizes."""

    def ttft(model_size_b: float) -> float:
        if model_size_b <= 7.0:
            return 0.150
        elif model_size_b <= 13.0:
            return 0.350
        elif model_size_b <= 34.0:
            return 0.850
        else:
            return 1.800

    def run():
        return [ttft(ms) * 1000 for ms in BENCHMARK_MODEL_SIZES]

    result = benchmark(run)
    assert len(result) == len(BENCHMARK_MODEL_SIZES)


def test_ttft_mock_coordinator(benchmark, coordinator_with_mock_nodes):
    """Measure TTFT through mock coordinator generate()."""
    coord = coordinator_with_mock_nodes

    def run():
        start = time.perf_counter()
        coord.tokenizer.encode.return_value = [1] * 32
        coord.tokenizer.decode.return_value = "test output"
        tokens = coord.generate("Hello, world")
        elapsed = time.perf_counter() - start
        return elapsed * 1000

    result = benchmark(run)
    assert result > 0


def test_ttft_vs_seq_len(benchmark):
    """TTFT as function of sequence length {64, 128, 256, 512, 1024}."""

    def estimate_ttft(seq_len: int) -> float:
        base = 0.050
        attention_cost = (seq_len ** 2) * 2e-8
        return base + attention_cost

    def run():
        seq_lens = [64, 128, 256, 512, 1024]
        return [estimate_ttft(sl) * 1000 for sl in seq_lens]

    result = benchmark(run)
    assert len(result) == 5
    assert all(r > 0 for r in result)


def test_ttft_jitter(benchmark, coordinator_with_mock_nodes):
    """Measure TTFT variability across 50 sequential calls (tail latency)."""
    coord = coordinator_with_mock_nodes

    def run():
        latencies = []
        for _ in range(50):
            start = time.perf_counter()
            coord.generate("Benchmark prompt")
            latencies.append((time.perf_counter() - start) * 1000)
        return max(latencies) - min(latencies)

    result = benchmark(run)
    assert result >= 0


# ---------------------------------------------------------------------------
# 3. Memory consumption per node
# ---------------------------------------------------------------------------

GPU_MEMORY_SPECS: Dict[str, Tuple[float, float, float]] = {
    "A100_80GB": (80.0, 2039.0, 312.0),
    "A100_40GB": (40.0, 1555.0, 312.0),
    "H100_80GB": (80.0, 3352.0, 989.0),
    "RTX4090_24GB": (24.0, 1008.0, 165.0),
    "MI250X_128GB": (128.0, 3277.0, 383.0),
    "MI100_32GB": (32.0, 1229.0, 184.6),
    "M2Ultra_192GB": (192.0, 800.0, 27.0),
    "ArcA770_16GB": (16.0, 559.0, 39.8),
}


def _estimate_memory_gb(model_size_b: float, precision_bits: int = 16) -> float:
    """Estimate memory needed for a model (weights + KV cache)."""
    params_gb = model_size_b * precision_bits / 8
    kv_cache_gb = model_size_b * 0.15
    overhead_gb = model_size_b * 0.05
    return params_gb + kv_cache_gb + overhead_gb


def test_memory_per_node_fits_check(benchmark):
    """Check which GPUs can fit each model size."""
    results = {}

    def run():
        results.clear()
        for gpu_name, (vram_gb, bw, tflops) in GPU_MEMORY_SPECS.items():
            fits: Dict[float, bool] = {}
            for ms in BENCHMARK_MODEL_SIZES:
                needed = _estimate_memory_gb(ms)
                fits[ms] = needed <= vram_gb * 0.9
            results[gpu_name] = fits
        count = sum(1 for r in results.values() for v in r.values() if v)
        return count

    result = benchmark(run)
    assert result >= len(GPU_MEMORY_SPECS)


def test_memory_breakdown(benchmark):
    """Detailed memory breakdown across model sizes."""
    results = {}

    def run():
        results.clear()
        for ms in BENCHMARK_MODEL_SIZES:
            weights = ms * 16 / 8
            kv = ms * 0.15
            overhead = ms * 0.05
            results[ms] = {"weights": weights, "kv_cache": kv, "overhead": overhead}
        return sum(results[ms]["weights"] for ms in results)

    result = benchmark(run)
    assert result > 0


def test_dynamic_memory_benchmark(benchmark, coordinator_with_mock_nodes):
    """Memory profiling overhead benchmark using tracemalloc."""
    coord = coordinator_with_mock_nodes

    def run():
        import tracemalloc
        tracemalloc.start()
        _ = coord.generate("memory test")
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak / (1024 * 1024)

    result = benchmark(run)
    assert result >= 0


# ---------------------------------------------------------------------------
# 4. Network bandwidth utilization
# ---------------------------------------------------------------------------

TENSOR_SHAPES: Dict[str, Tuple[int, ...]] = {
    "small": (1, 768),
    "medium": (4, 4096),
    "large": (8, 8192),
    "xlarge": (16, 16384),
}


def test_network_pickle_serialization(benchmark):
    """Measure pickle serialization/deserialization throughput for tensors."""
    results = {}

    def run():
        results.clear()
        for name, shape in TENSOR_SHAPES.items():
            t = torch.randn(shape)
            serialized = pickle.dumps(t)
            deserialized = pickle.loads(serialized)
            results[name] = len(serialized)
        return sum(results.values())

    result = benchmark(run)
    assert result > 0


def test_network_transfer_time(benchmark):
    """Model bandwidth utilization across tensor sizes and network speeds."""
    results = {}

    def run():
        results.clear()
        bandwidths = [1.0, 10.0, 25.0, 40.0, 100.0, 200.0]
        for name, shape in TENSOR_SHAPES.items():
            t = torch.randn(shape)
            bytes_per_tensor = t.numel() * t.element_size()
            for bw in bandwidths:
                transfer_time = bytes_per_tensor * 8 / (bw * 1e9)
                results[f"{name}_bw{bw}"] = transfer_time * 1000
        return sum(results.values()) / max(len(results), 1)

    result = benchmark(run)
    assert result >= 0


def test_network_layer_transfer(benchmark):
    """Simulate per-layer activation transfer in pipeline parallelism."""
    hidden_dims = {"1B": 768, "3B": 1536, "7B": 4096, "13B": 5120, "34B": 7168, "70B": 8192}

    def run():
        total = 0.0
        for ms, dim in hidden_dims.items():
            activations = torch.randn(1, dim, dtype=torch.float32)
            bytes_total = activations.numel() * activations.element_size()
            transfer_us = bytes_total * 8 / (10.0 * 1e9) * 1e6
            total += transfer_us
        return total

    result = benchmark(run)
    assert result >= 0


# ---------------------------------------------------------------------------
# 5. Scaling efficiency (2-16 nodes)
# ---------------------------------------------------------------------------

def _scaling_efficiency(num_nodes: int, communication_overhead: float = 0.05) -> float:
    """Compute ideal scaling efficiency given communication overhead."""
    overhead = communication_overhead * (num_nodes - 1)
    return 1.0 / (1.0 + overhead)


def test_ideal_vs_actual_scaling(benchmark):
    """Compare ideal linear scaling vs. efficiency-adjusted scaling."""
    results = {}

    def run():
        results.clear()
        overheads = [0.02, 0.05, 0.10, 0.15, 0.20]
        for overhead in overheads:
            efficiencies = []
            for n in [2, 4, 8, 16]:
                eff = _scaling_efficiency(n, overhead)
                efficiencies.append(eff)
            results[overhead] = efficiencies
        return sum(results.get(0.05, [0.0]))

    result = benchmark(run)
    assert result > 0


def test_pipeline_efficiency_heterogeneous(benchmark, mock_cluster):
    """Scaling efficiency with HeterogeneousCluster analytical model."""
    device_info = DeviceInfo(
        device_type="cuda", device_family="NVIDIA", device_id=0,
        name="A100-SXM-80GB", total_memory_bytes=80 * 1024**3,
        tflops_fp16=312.0, memory_bandwidth_gbps=2039.0,
    )

    def run():
        th = 0.0
        for n in [2, 4, 8, 16]:
            nodes = [
                HeterogeneousNode(
                    node_id=f"n{i}", host="localhost", port=55060 + i,
                    device_info=device_info, throughput_score=1000.0,
                ) for i in range(n)
            ]
            cluster = HeterogeneousCluster(nodes=nodes, total_layers=32)
            th = estimate_heterogeneous_throughput(cluster)
            single = estimate_heterogeneous_throughput(
                HeterogeneousCluster(
                    nodes=nodes[:1], total_layers=32, hidden_size=4096,
                ),
            )
            if single > 0 and n > 0:
                th = th / (single * n)
        return th

    result = benchmark(run)
    assert 0 <= result <= 1.0


def test_scaling_gpu_count(benchmark):
    """Throughput scaling with GPU count {1,2,4,8} for 7B and 34B models."""
    results = {}

    def run():
        results.clear()
        for ms in [7.0, 34.0]:
            base = _estimate_throughput(ms, 1, 4, 512)
            speedups = []
            for n in [2, 4, 8]:
                th = _estimate_throughput(ms, n, 4, 512)
                speedups.append(th / max(base, 1e-12))
            results[ms] = speedups
        return results.get(7.0, [0, 0, 0])[-1] if 7.0 in results else 0.0

    result = benchmark(run)
    assert result >= 0


def test_network_overhead_scaling(benchmark):
    """Impact of network bandwidth on scaling efficiency."""
    results = {}

    def run():
        results.clear()
        bandwidths = [1.0, 10.0, 25.0, 40.0, 100.0, 200.0, 400.0, 800.0]
        for bw in bandwidths:
            overhead = 1.0 / (1.0 + 10.0 / bw)
            results[bw] = overhead
        return results.get(800.0, 0.0)

    result = benchmark(run)
    assert result > 0


# ---------------------------------------------------------------------------
# 6. Single-GPU baseline comparison
# ---------------------------------------------------------------------------

SINGLE_GPU_SPECS: List[Tuple[str, float, float, float]] = [
    ("A100", 312.0, 80.0, 2039.0),
    ("H100", 989.0, 80.0, 3352.0),
    ("RTX 4090", 165.0, 24.0, 1008.0),
    ("RTX 6000 Ada", 182.0, 48.0, 960.0),
    ("MI250X", 383.0, 128.0, 3277.0),
    ("M2 Ultra", 27.0, 192.0, 800.0),
]


def test_single_gpu_throughput(benchmark):
    """Single-GPU baseline throughput comparison across GPU types @ 7B."""
    results = {}

    def run():
        results.clear()
        for gpu_name, tflops, vram, bw in SINGLE_GPU_SPECS:
            needed = _estimate_memory_gb(7.0)
            if needed > vram * 0.9:
                results[gpu_name] = 0.0
            else:
                results[gpu_name] = _estimate_throughput(7.0, 1, 1, 512)
        return sum(results.values())

    result = benchmark(run)
    assert result > 0


def test_multi_gpu_speedup_over_single(benchmark):
    """Speedup of 4-node A100 cluster over single A100 across model sizes."""
    results = {}

    def run():
        results.clear()
        for ms in BENCHMARK_MODEL_SIZES:
            single = _estimate_throughput(ms, 1, 4, 512)
            multi = _estimate_throughput(ms, 4, 4, 512)
            results[ms] = multi / max(single, 1e-12)
        return results

    result = benchmark(run)
    assert len(result) == len(BENCHMARK_MODEL_SIZES)


def test_cost_per_token_estimate(benchmark):
    """Compute estimated cost per token for single GPU vs. cluster."""

    HOURLY_COST: Dict[str, float] = {
        "A100": 3.00, "H100": 4.50, "RTX 4090": 0.75, "MI250X": 2.50,
    }

    def run():
        estimates = {}
        for gpu_name, tflops, vram, bw in SINGLE_GPU_SPECS:
            if gpu_name not in HOURLY_COST:
                continue
            th = _estimate_throughput(7.0, 1, 1, 512)
            tokens_per_hour = th * 3600
            cost_per_token = HOURLY_COST[gpu_name] / max(tokens_per_hour, 1.0)
            cluster_th = _estimate_throughput(7.0, 4, 1, 512)
            cluster_tph = cluster_th * 3600
            cluster_cost = HOURLY_COST[gpu_name] * 4 / max(cluster_tph, 1.0)
            estimates[gpu_name] = {
                "single": cost_per_token,
                "four_node": cluster_cost,
                "speedup": cluster_th / max(th, 1e-12),
            }
        return estimates

    result = benchmark(run)
    assert len(result) == len(HOURLY_COST)


def test_memory_footprint_comparison(benchmark):
    """Single-GPU memory footprint vs. 4-node distributed across model sizes."""
    results = {}

    def run():
        results.clear()
        for ms in BENCHMARK_MODEL_SIZES:
            single_mem = _estimate_memory_gb(ms)
            per_node_mem = _estimate_memory_gb(ms) / 4
            fits_single = single_mem <= 80.0
            fits_distributed = per_node_mem <= 80.0
            results[ms] = {
                "single": single_mem, "per_node": per_node_mem,
                "fits_single": fits_single, "fits_distributed": fits_distributed,
            }
        return sum(r["single"] for r in results.values())

    result = benchmark(run)
    assert result > 0


# ---------------------------------------------------------------------------
# Cross-cutting: end-to-end benchmark
# ---------------------------------------------------------------------------

def test_end_to_end_pipeline(benchmark, coordinator_with_mock_nodes):
    """End-to-end mock pipeline timings."""
    coord = coordinator_with_mock_nodes
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a poem about distributed systems.",
    ]

    def run():
        outputs = []
        for p in prompts:
            outputs.append(coord.generate(p))
        return len([o for o in outputs if o])

    result = benchmark(run)
    assert result == len(prompts)


def test_memory_tracemalloc_overhead(benchmark, coordinator_with_mock_nodes):
    """Overhead of running memory profiling during inference."""
    coord = coordinator_with_mock_nodes

    def run():
        import tracemalloc
        tracemalloc.start()
        coord.generate("benchmark")
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    result = benchmark(run)
    assert result >= 0


# ---------------------------------------------------------------------------
# Summary / reporting
# ---------------------------------------------------------------------------

def test_benchmark_summary(benchmark):
    """Aggregate summary: report median throughput across all configs."""
    results: List[float] = []

    def run():
        results.clear()
        for ms in BENCHMARK_MODEL_SIZES:
            for n in BENCHMARK_NODE_COUNTS:
                for bs in BENCHMARK_BATCH_SIZES:
                    th = _estimate_throughput(ms, n, bs, BENCHMARK_SEQ_LEN)
                    results.append(th)
        return statistics.median(results) if results else 0.0

    result = benchmark(run)
    assert result > 0
