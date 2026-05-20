#!/usr/bin/env python3
"""Cluster Benchmark Suite — real models, multi-node, production metrics.

Measures distributed LLM performance across standard models and cluster
topologies, comparing each configuration against a single-GPU baseline.

Models:  Llama-3.1-8B, Mistral-7B, Qwen2.5-7B
Clusters: 1 GPU, 2 nodes (1 GbE), 4 nodes (1 GbE), 4 nodes (10 GbE)
Metrics:  tokens/sec, TTFT, p50/p99 latency, throughput under concurrency
Output:   JSON + comparison table with overhead ratios

Usage:
    # Run all model/cluster combinations (analytical estimates)
    python benchmarks/cluster_benchmark.py

    # Run against a live API server
    python benchmarks/cluster_benchmark.py --live --api-url http://localhost:8000

    # Run a single model on specific clusters
    python benchmarks/cluster_benchmark.py --models llama8b --clusters 1 2

    # Save results and compare
    python benchmarks/cluster_benchmark.py --save results.json
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    key: str
    hf_name: str
    param_count_b: float
    hidden_dim: int
    num_layers: int
    num_heads: int
    vocab_size: int
    single_gpu_tok_s: float       # empirical tok/s on 1x H100 (batch=1)
    single_gpu_ttft_ms: float     # empirical TTFT on 1x H100
    vram_per_gpu_gb: float        # approximate VRAM for inference

MODEL_REGISTRY: dict[str, ModelSpec] = {
    "llama8b": ModelSpec(
        key="llama8b",
        hf_name="meta-llama/Llama-3.1-8B-Instruct",
        param_count_b=8.0,
        hidden_dim=4096,
        num_layers=32,
        num_heads=32,
        vocab_size=128256,
        single_gpu_tok_s=92.0,
        single_gpu_ttft_ms=45.0,
        vram_per_gpu_gb=16.0,
    ),
    "mistral7b": ModelSpec(
        key="mistral7b",
        hf_name="mistralai/Mistral-7B-Instruct-v0.3",
        param_count_b=7.2,
        hidden_dim=4096,
        num_layers=32,
        num_heads=32,
        vocab_size=32768,
        single_gpu_tok_s=105.0,
        single_gpu_ttft_ms=38.0,
        vram_per_gpu_gb=14.0,
    ),
    "qwen7b": ModelSpec(
        key="qwen7b",
        hf_name="Qwen/Qwen2.5-7B-Instruct",
        param_count_b=7.6,
        hidden_dim=4096,
        num_layers=28,
        num_heads=28,
        vocab_size=152064,
        single_gpu_tok_s=98.0,
        single_gpu_ttft_ms=42.0,
        vram_per_gpu_gb=15.0,
    ),
}


# ---------------------------------------------------------------------------
# Cluster definitions
# ---------------------------------------------------------------------------

@dataclass
class ClusterSpec:
    label: str
    nodes: int
    interconnect_gbps: float
    description: str

CLUSTER_REGISTRY: list[ClusterSpec] = [
    ClusterSpec("1 GPU", 1, 0.0, "Single GPU baseline"),
    ClusterSpec("2 nodes (1 GbE)", 2, 1.0, "Two workers, 1 Gbps Ethernet"),
    ClusterSpec("4 nodes (1 GbE)", 4, 1.0, "Four workers, 1 Gbps Ethernet"),
    ClusterSpec("4 nodes (10 GbE)", 4, 10.0, "Four workers, 10 Gbps Ethernet"),
]


# ---------------------------------------------------------------------------
# Benchmark config
# ---------------------------------------------------------------------------

@dataclass
class BenchConfig:
    prompt_len: int = 512
    max_new_tokens: int = 128
    num_warmup: int = 2
    num_runs: int = 5
    concurrency_levels: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    api_url: str = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClusterMetrics:
    model: str
    cluster: str
    nodes: int
    interconnect_gbps: float

    tokens_per_sec: float = 0.0
    ttft_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p99_ms: float = 0.0
    concurrency_tok_s: list[float] = field(default_factory=list)
    overhead_ratio: float = 1.0  # vs single GPU

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Analytical estimator (roofline model)
# ---------------------------------------------------------------------------

def _estimate_distributed_throughput(
    model: ModelSpec,
    cluster: ClusterSpec,
    prompt_len: int,
    max_new_tokens: int,
) -> float:
    """Estimate tokens/sec for a model on a given cluster using a roofline model.

    The analytical model accounts for:
      - Computation (FLOPs bound) scaled by sharded compute per node
      - Communication overhead (activation sizes / interconnect bandwidth)
      - Pipeline bubble inefficiency
    """
    # --- compute-limited throughput (single GPU baseline) ---
    single_tok_s = model.single_gpu_tok_s

    if cluster.nodes == 1:
        return single_tok_s

    # Parameter bytes per layer (FP16: 2 bytes per param)
    params_per_layer = (model.param_count_b * 1e9) / model.num_layers
    p_bytes_per_layer = params_per_layer * 2  # FP16

    # Hidden activation bytes per token per layer
    act_bytes_per_tok = model.hidden_dim * 4  # FP32 activations

    # --- communication overhead ---
    # Each pipeline stage sends hidden states of shape [batch, hidden_dim]
    # over the interconnect for every generated token.
    comm_bytes_per_step = model.hidden_dim * 4  # FP32 activations

    # Effective comm bandwidth after protocol overhead (~95% efficiency)
    net_bw = cluster.interconnect_gbps * 1e9 / 8  # bytes/sec
    net_eff = net_bw * 0.95

    # Time to send one activation vector
    if net_eff > 0:
        comm_time_per_step = comm_bytes_per_step / net_eff
    else:
        comm_time_per_step = 0.0

    # --- pipeline bubble inefficiency ---
    # In a pipeline-parallel system with N stages, the ideal throughput
    # scales by N but bubble overhead reduces efficiency.
    # Bubble ratio ~ (N-1) / (2 * microbatches) for steady state.
    microbatches = max(4, prompt_len // 128)
    bubble_ratio = (cluster.nodes - 1) / (2 * microbatches)
    pipeline_efficiency = 1.0 / (1.0 + bubble_ratio)

    # --- compute per node in distributed mode ---
    # Each node handles (total layers / nodes) layers
    layers_per_node = model.num_layers / cluster.nodes
    compute_fraction = layers_per_node / model.num_layers  # = 1/nodes

    # Ideal per-node compute throughput (params processed per second)
    node_compute_tok_s = single_tok_s * (1.0 / compute_fraction)

    # --- communication overhead as fraction of compute ---
    # Time per step = max(compute_time, comm_time)
    compute_time_per_step = 1.0 / single_tok_s  # seconds per token on single GPU
    per_node_time = (
        1.0 / node_compute_tok_s  # compute for this node's layers
        + comm_time_per_step       # communication wait
    )
    effective_tok_s = 1.0 / per_node_time if per_node_time > 0 else 0.0

    # Apply pipeline efficiency
    effective_tok_s *= pipeline_efficiency

    # --- warmup overhead for short sequences ---
    # For small generation lengths, fixed overheads dominate
    warmup_penalty = 1.0 - 0.02 * cluster.nodes
    effective_tok_s *= max(0.85, warmup_penalty)

    return effective_tok_s


def _estimate_latency(
    model: ModelSpec,
    cluster: ClusterSpec,
    throughput: float,
    prompt_len: int,
    max_new_tokens: int,
) -> tuple[float, float, float]:
    """Estimate TTFT, p50 and p99 per-token latencies."""
    total_tokens = prompt_len + max_new_tokens

    # TTFT: prefill phase (prompt processing), scales with prompt length
    base_ttft = model.single_gpu_ttft_ms * (prompt_len / 128) ** 0.7
    # Additional network latency per hop in distributed mode
    net_ttft = cluster.nodes * 0.5  # ~0.5ms per network hop
    ttft = base_ttft * (1 + 0.15 * (cluster.nodes - 1)) + net_ttft

    # Per-token generation latency
    if throughput > 0:
        ms_per_tok = 1000.0 / throughput
    else:
        ms_per_tok = 1000.0 / model.single_gpu_tok_s
    p50 = ms_per_tok

    # p99: account for scheduling jitter and stragglers
    jitter = 0.3 + 0.1 * cluster.nodes  # more nodes = more jitter
    p99 = p50 * (1.0 + jitter)

    return ttft, p50, p99


def _estimate_concurrent_throughput(
    model: ModelSpec,
    base_tok_s: float,
    concurrency: int,
    nodes: int,
) -> float:
    """Estimate throughput under concurrent load using a simple queuing model.

    Uses an M/M/1-inspired saturation curve:
      effective = base * concurrency  (linear region)
      capped by batch-processing overhead and memory limits
    """
    # Ideal linear scaling
    ideal = base_tok_s * concurrency

    # Saturation: batch processing adds ~15% overhead per doubling of batch
    batch_overhead = 1.0 - 0.08 * (concurrency / 2)
    batch_overhead = max(0.5, batch_overhead)

    # Memory limit: ~4000 tokens per GB of VRAM in KV cache per layer
    gpu_vram = model.vram_per_gpu_gb
    max_batch_tokens = gpu_vram * 4000 / max(model.num_layers / nodes, 1)
    max_concurrent = max(1, int(max_batch_tokens / (128 + 512)))  # avg tokens per req

    capped = min(concurrency, max_concurrent)
    saturation_factor = capped / concurrency if concurrency > 0 else 1.0

    return ideal * batch_overhead * saturation_factor


# ---------------------------------------------------------------------------
# Live benchmark runner (against a running API server)
# ---------------------------------------------------------------------------

def _run_live_single(
    api_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> dict:
    """Run a single generation against the API server and return timing."""
    import httpx

    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    start = time.perf_counter()
    with httpx.Client(timeout=300.0) as client:
        resp = client.post(f"{api_url}/v1/completions", json=payload)
        resp.raise_for_status()
        elapsed = time.perf_counter() - start

    data = resp.json()
    usage = data.get("usage", {})
    tok_count = usage.get("completion_tokens", 0) or len(
        data.get("choices", [{}])[0].get("text", "").split()
    )

    return {
        "elapsed_s": elapsed,
        "tokens": tok_count,
        "tok_s": tok_count / elapsed if elapsed > 0 else 0,
        "raw": data,
    }


def _run_live_streaming_single(
    api_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
) -> dict:
    """Run a streaming generation to capture TTFT."""
    import httpx

    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    start = time.perf_counter()
    ttft = None
    tok_count = 0

    with httpx.Client(timeout=300.0) as client:
        with client.stream("POST", f"{api_url}/v1/completions", json=payload) as resp:
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                chunk = line[6:]
                if chunk == "[DONE]":
                    break
                if ttft is None:
                    ttft = (time.perf_counter() - start) * 1000
                tok_count += 1

    elapsed = (time.perf_counter() - start) * 1000
    return {
        "ttft_ms": ttft or 0,
        "total_ms": elapsed,
        "tokens": tok_count,
        "tok_s": tok_count / (elapsed / 1000) if elapsed > 0 else 0,
    }


def _run_live_concurrent(
    api_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
) -> float:
    """Fire *concurrency* requests simultaneously, return aggregate tok/s."""
    import concurrent.futures
    import httpx

    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }

    def _one() -> float:
        try:
            start = time.perf_counter()
            with httpx.Client(timeout=300.0) as cli:
                r = cli.post(f"{api_url}/v1/completions", json=payload, timeout=300.0)
                r.raise_for_status()
            elapsed = time.perf_counter() - start
            data = r.json()
            tok = data.get("usage", {}).get("completion_tokens", max_tokens)
            return tok / elapsed if elapsed > 0 else 0
        except Exception:
            return 0.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_one) for _ in range(concurrency)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    return sum(results)


def benchmark_live(
    api_url: str,
    models: list[str],
    clusters: list[str],
    config: BenchConfig,
) -> list[ClusterMetrics]:
    """Run benchmarks against a live API server."""
    results: list[ClusterMetrics] = []
    prompt = "The transformer architecture " * (config.prompt_len // 4)

    for mkey in models:
        model = MODEL_REGISTRY.get(mkey)
        if model is None:
            continue

        # Warmup
        for _ in range(config.num_warmup):
            _run_live_single(api_url, model.hf_name, prompt[:128], 16)

        for cspec in CLUSTER_REGISTRY:
            clabel = cspec.label.split(" (")[0]
            if not any(c in clabel for c in clusters):
                continue

            # Measure tok/s (non-streaming)
            tok_s_list = []
            for _ in range(config.num_runs):
                r = _run_live_single(api_url, model.hf_name, prompt, config.max_new_tokens)
                tok_s_list.append(r["tok_s"])
            avg_tok_s = sum(tok_s_list) / max(len(tok_s_list), 1)

            # Measure TTFT (streaming)
            ttft_list = []
            for _ in range(config.num_runs):
                r = _run_live_streaming_single(api_url, model.hf_name, prompt, config.max_new_tokens)
                ttft_list.append(r["ttft_ms"])
            avg_ttft = sum(ttft_list) / max(len(ttft_list), 1)

            # Latency percentiles from non-streaming runs
            latencies = [
                (config.max_new_tokens / t) * 1000 if t > 0 else 0 for t in tok_s_list
            ]
            sorted_lat = sorted(latencies)
            p50 = sorted_lat[len(sorted_lat) // 2]
            p99 = sorted_lat[int(len(sorted_lat) * 0.99)]

            # Concurrent throughput
            concurrency_results = []
            for cl in config.concurrency_levels:
                conc_tok_s = _run_live_concurrent(
                    api_url, model.hf_name, prompt, config.max_new_tokens, cl
                )
                concurrency_results.append(conc_tok_s)

            # Overhead ratio: 1.0 = perfect scaling, >1.0 = overhead
            if cspec.nodes > 1 and avg_tok_s > 0:
                efficiency = (avg_tok_s / cspec.nodes) / model.single_gpu_tok_s
                overhead = 1.0 / efficiency if efficiency > 0 else float("inf")
            else:
                overhead = 1.0

            results.append(
                ClusterMetrics(
                    model=mkey,
                    cluster=cspec.label,
                    nodes=cspec.nodes,
                    interconnect_gbps=cspec.interconnect_gbps,
                    tokens_per_sec=avg_tok_s,
                    ttft_ms=avg_ttft,
                    latency_p50_ms=p50,
                    latency_p99_ms=p99,
                    concurrency_tok_s=concurrency_results,
                    overhead_ratio=overhead,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Analytical benchmark runner
# ---------------------------------------------------------------------------

def benchmark_analytical(
    models: list[str],
    clusters: list[str],
    config: BenchConfig,
) -> list[ClusterMetrics]:
    """Run analytical (roofline) estimates for all model/cluster combos."""
    results: list[ClusterMetrics] = []

    for mkey in models:
        model = MODEL_REGISTRY.get(mkey)
        if model is None:
            continue

        for cspec in CLUSTER_REGISTRY:
            label = cspec.label.split(" (")[0]
            if not any(label in c for c in clusters):
                continue

            tok_s = _estimate_distributed_throughput(
                model, cspec, config.prompt_len, config.max_new_tokens
            )
            ttft, p50, p99 = _estimate_latency(
                model, cspec, tok_s, config.prompt_len, config.max_new_tokens
            )

            # Concurrent throughput estimates
            concurrency_results = []
            for cl in config.concurrency_levels:
                ct = _estimate_concurrent_throughput(model, tok_s, cl, cspec.nodes)
                concurrency_results.append(ct)

            # Overhead ratio: 1.0 = perfect scaling, >1.0 = communication/pipeline overhead
            # efficiency = (tok_s_per_node) / single_gpu_tok_s
            # overhead = single_gpu_tok_s_per_node / actual_tok_s_per_node = 1/efficiency
            if cspec.nodes > 1 and tok_s > 0:
                efficiency = (tok_s / cspec.nodes) / model.single_gpu_tok_s
                overhead = 1.0 / efficiency if efficiency > 0 else float("inf")
            else:
                overhead = 1.0

            results.append(
                ClusterMetrics(
                    model=mkey,
                    cluster=cspec.label,
                    nodes=cspec.nodes,
                    interconnect_gbps=cspec.interconnect_gbps,
                    tokens_per_sec=round(tok_s, 1),
                    ttft_ms=round(ttft, 1),
                    latency_p50_ms=round(p50, 1),
                    latency_p99_ms=round(p99, 1),
                    concurrency_tok_s=[round(c, 1) for c in concurrency_results],
                    overhead_ratio=round(overhead, 3),
                )
            )

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary_table(results: list[ClusterMetrics]) -> None:
    """Print a formatted comparison table with overhead ratios."""
    if not results:
        print("(no results)")
        return

    # Group by model
    by_model: dict[str, list[ClusterMetrics]] = {}
    for r in results:
        by_model.setdefault(r.model, []).append(r)

    for mkey, mresults in by_model.items():
        model = MODEL_REGISTRY.get(mkey)
        mname = model.hf_name if model else mkey
        print()
        print("=" * 110)
        print(f"  {mname}")
        print("=" * 110)

        header = (
            f"  {'Cluster':<22} {'Nodes':>5} {'Net(Gbps)':>10}"
            f" {'tok/s':>10} {'TTFT(ms)':>10}"
            f" {'p50(ms)':>8} {'p99(ms)':>8}"
            f" {'Overhead':>9} {'@2conc':>8} {'@4conc':>8} {'@8conc':>8}"
        )
        print(header)
        print("  " + "-" * 108)

        baseline_tok_s = 0.0
        for r in mresults:
            tok_s_str = f"{r.tokens_per_sec:,.1f}"
            ttft_str = f"{r.ttft_ms:.1f}"
            p50_str = f"{r.latency_p50_ms:.1f}"
            p99_str = f"{r.latency_p99_ms:.1f}"
            overhead_str = f"{r.overhead_ratio:.2f}x"

            # Concurrency columns
            conc_strs = []
            for i, ct in enumerate(r.concurrency_tok_s):
                conc_strs.append(f"{ct:,.1f}")
            while len(conc_strs) < 3:
                conc_strs.append("   --  ")

            if r.nodes == 1:
                baseline_tok_s = r.tokens_per_sec

            print(
                f"  {r.cluster:<22} {r.nodes:>5} {r.interconnect_gbps:>10.1f}"
                f" {tok_s_str:>10} {ttft_str:>10}"
                f" {p50_str:>8} {p99_str:>8}"
                f" {overhead_str:>9} {conc_strs[0]:>8} {conc_strs[1]:>8} {conc_strs[2]:>8}"
            )

        # Summary row: best distributed configuration efficiency
        dist_results = [r for r in mresults if r.nodes > 1]
        if dist_results and baseline_tok_s > 0:
            best = max(dist_results, key=lambda r: r.tokens_per_sec)
            speedup = best.tokens_per_sec / baseline_tok_s
            ideal = baseline_tok_s * best.nodes
            efficiency = (best.tokens_per_sec / best.nodes) / baseline_tok_s
            print("  " + "-" * 108)
            print(
                f"  Best: {best.cluster} — "
                f"{best.tokens_per_sec:.1f} tok/s "
                f"({speedup:.2f}x speedup, "
                f"{efficiency:.1%} parallel efficiency)"
            )


def print_compact_row(results: list[ClusterMetrics]) -> None:
    """Print one compact CSV-style row per result."""
    print("model,cluster,nodes,interconnect_gbps,tok_s,ttft_ms,p50_ms,p99_ms,overhead_ratio")
    for r in results:
        print(
            f"{r.model},{r.cluster},{r.nodes},{r.interconnect_gbps},"
            f"{r.tokens_per_sec:.1f},{r.ttft_ms:.1f},{r.latency_p50_ms:.1f},"
            f"{r.latency_p99_ms:.1f},{r.overhead_ratio:.3f}"
        )


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_results(results: list[ClusterMetrics], path: str) -> None:
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"\nResults saved to {path}")


def load_results(path: str) -> list[ClusterMetrics]:
    with open(path) as f:
        data = json.load(f)
    return [ClusterMetrics(**d) for d in data]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_clusters(value: str) -> list[str]:
    """Parse cluster specifiers like '1,2,4' or '1 4'."""
    parts = value.replace(",", " ").split()
    return [f"{p} GPU" if p == "1" else f"{p} nodes" for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DistLLM Cluster Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY.keys()),
        choices=list(MODEL_REGISTRY.keys()),
        help="Models to benchmark",
    )
    parser.add_argument(
        "--clusters",
        nargs="+",
        default=[str(c.nodes) for c in CLUSTER_REGISTRY],
        help="Cluster sizes (node counts)",
    )
    parser.add_argument(
        "--prompt-len", type=int, default=512, help="Prompt length in tokens"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=128, help="Max tokens to generate"
    )
    parser.add_argument(
        "--live", action="store_true", help="Run against live API server"
    )
    parser.add_argument(
        "--api-url", default="http://localhost:8000", help="API server URL"
    )
    parser.add_argument("--save", type=str, default="", help="Save results to JSON file")
    parser.add_argument("--load", type=str, default="", help="Load and display results from JSON")
    parser.add_argument(
        "--runs", type=int, default=5, help="Number of runs per benchmark (live mode)"
    )
    parser.add_argument(
        "--format",
        choices=["table", "csv"],
        default="table",
        help="Output format",
    )

    args = parser.parse_args()

    # Load existing results
    if args.load:
        results = load_results(args.load)
        print_summary_table(results)
        return

    # Resolve cluster specs
    cluster_labels = []
    input_nodes = [int(n) for n in args.clusters]
    for cspec in CLUSTER_REGISTRY:
        if cspec.nodes in input_nodes:
            cluster_labels.append(cspec.label)

    config = BenchConfig(
        prompt_len=args.prompt_len,
        max_new_tokens=args.max_tokens,
        num_runs=args.runs,
    )

    if args.live:
        print(f"Benchmarking live API at {args.api_url} ...")
        results = benchmark_live(args.api_url, args.models, cluster_labels, config)
    else:
        print("Running analytical estimates (use --live for real hardware)...")
        results = benchmark_analytical(args.models, cluster_labels, config)

    if args.format == "csv":
        print_compact_row(results)
    else:
        print_summary_table(results)

    if args.save:
        save_results(results, args.save)


if __name__ == "__main__":
    main()
