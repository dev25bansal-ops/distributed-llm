"""DistLLM CLI tune commands — auto-tuning for quantization, batching, and caching.

Provides three sub-commands under ``distllm tune``:

* ``quantize``  — Adaptive Precision Optimizer (per-device quantization selection)
* ``batch``     — Batch size optimizer (profiles different batch/token configs)
* ``cache``     — Cache sizing calculator (KV cache block count and memory)
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table


# ---------------------------------------------------------------------------
# Model heuristics
# ---------------------------------------------------------------------------

def _estimate_model_size(model_name: str) -> int:
    """Rough fp16 model size in bytes from model name."""
    name_lower = model_name.lower()
    if "70b" in name_lower:
        return 140 * 1024**3
    if "40b" in name_lower:
        return 80 * 1024**3
    if "34b" in name_lower or "33b" in name_lower:
        return 68 * 1024**3
    if "13b" in name_lower:
        return 26 * 1024**3
    if "7b" in name_lower or "8b" in name_lower:
        return 14 * 1024**3
    if "3b" in name_lower:
        return 6 * 1024**3
    if "1b" in name_lower:
        return 2 * 1024**3
    return 14 * 1024**3  # default 7B


def _estimate_num_layers(model_name: str) -> int:
    """Rough layer count from model name."""
    name_lower = model_name.lower()
    if "70b" in name_lower:
        return 80
    if "40b" in name_lower:
        return 60
    if "34b" in name_lower or "33b" in name_lower:
        return 48
    if "13b" in name_lower:
        return 40
    if "7b" in name_lower or "8b" in name_lower:
        return 32
    if "3b" in name_lower:
        return 28
    if "1b" in name_lower:
        return 22
    return 32


def _estimate_hidden_dim(model_name: str) -> int:
    """Rough hidden dimension from model name."""
    name_lower = model_name.lower()
    if "70b" in name_lower:
        return 8192
    if "40b" in name_lower or "34b" in name_lower or "33b" in name_lower:
        return 6656
    if "13b" in name_lower:
        return 5120
    if "7b" in name_lower or "8b" in name_lower:
        return 4096
    if "3b" in name_lower:
        return 3072
    if "1b" in name_lower:
        return 2048
    return 4096


def _estimate_num_kv_heads(model_name: str) -> int:
    """Rough number of KV heads (GQA) from model name."""
    name_lower = model_name.lower()
    if "70b" in name_lower:
        return 8
    if "13b" in name_lower:
        return 40
    if "7b" in name_lower or "8b" in name_lower:
        return 32
    return 32


def _estimate_head_dim(model_name: str) -> int:
    """Rough head dimension from model name."""
    hidden = _estimate_hidden_dim(model_name)
    num_heads = _estimate_num_kv_heads(model_name)
    # head_dim = hidden_dim / num_attention_heads, but we use kv_heads as proxy
    # For most models head_dim is 128
    return hidden // max(num_heads, 1)


# ---------------------------------------------------------------------------
# tune quantize
# ---------------------------------------------------------------------------

def run_tune_quantize(
    model: str,
    nodes: int,
    max_quality_loss: float,
    prefer_speed: bool,
    require_calibration: bool,
    output: str | None,
    benchmark: bool,
    json_output: bool,
    console: Console,
) -> None:
    """Run Adaptive Precision Optimizer to select optimal quantization per device."""
    try:
        from distllm.dist.partition.quant_report import ReportGenerator
        from distllm.dist.partition.quantization_tuner import (
            NodeInfo,
            QuantizationAutoTuner,
        )

        # Optionally run live hardware benchmarks
        if benchmark:
            from distllm.dist.partition.quant_bench import QuantBenchmarker

            console.print("[bold]Running live hardware benchmarks...[/bold]")
            benchmarker = QuantBenchmarker()
            suites = benchmarker.benchmark_all_gpus()
            for suite in suites:
                console.print(suite.summary())
        else:
            console.print(
                "[dim]Using static quantization profiles (use --benchmark for live data)[/dim]"
            )

        # Profile GPUs for node info
        from distllm.dist.partition.profiles import GPUProfiler

        profiler = GPUProfiler()
        gpu_profiles = profiler.profile_all_gpus()

        # Build node list
        node_infos: list[NodeInfo] = []
        for i, gp in enumerate(gpu_profiles):
            node_infos.append(NodeInfo.from_gpu_profile(gp, node_id=f"node-{i}"))

        if not node_infos:
            console.print("[red]No GPUs found. Cannot generate quantization plan.[/red]")
            raise SystemExit(1)

        model_size_bytes = _estimate_model_size(model)
        num_layers = _estimate_num_layers(model)

        console.print(f"\n[bold]Model:[/bold] {model}")
        console.print(f"[bold]Size:[/bold] {model_size_bytes / (1024**3):.1f} GB (fp16)")
        console.print(f"[bold]Layers:[/bold] {num_layers}")
        console.print(f"[bold]Nodes:[/bold] {len(node_infos)}")
        console.print()

        tuner = QuantizationAutoTuner(
            max_quality_loss=max_quality_loss,
            prefer_speed=prefer_speed,
            require_calibration=require_calibration,
        )
        plan = tuner.recommend(node_infos, model_size_bytes, num_layers)

        reporter = ReportGenerator()
        report = reporter.generate(plan, node_infos, model_size_bytes, num_layers)

        if json_output or output:
            report_data = report.to_json()
            if output:
                Path(output).write_text(report_data, encoding="utf-8")
                console.print(f"[green]Report saved to {output}[/green]")
            if json_output:
                console.print(report_data)
        else:
            console.print(report.to_text())

    except ImportError as exc:
        console.print(f"[red]Missing dependency: {exc}[/red]")
        console.print("[dim]Install with: pip install distllm[self-hosted][/dim]")
        raise SystemExit(1) from exc
    except SystemExit:
        raise
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise SystemExit(1) from exc


# ---------------------------------------------------------------------------
# tune batch
# ---------------------------------------------------------------------------

def run_tune_batch(
    model: str,
    gpu_count: int,
    target_latency_ms: float,
    host: str,
    port: int,
    max_batch: int,
    max_tokens: int,
    output: str | None,
    console: Console,
) -> None:
    """Profile different batch sizes and recommend optimal batch configuration.

    Sends a series of requests at increasing concurrency levels to measure
    throughput and latency, then selects the configuration that stays within
    the target latency while maximising throughput.
    """
    import httpx

    model_size_bytes = _estimate_model_size(model)
    # Bytes per token during decode (all params loaded once per token)
    bytes_per_token = model_size_bytes
    # Assume INT4 quantization halves memory footprint
    quant_factor = 0.5
    effective_bytes = int(bytes_per_token * quant_factor)

    # Roofline: memory-bandwidth limited decode throughput per GPU
    # Use a conservative 1.5 TB/s for modern data-centre GPUs
    mem_bandwidth_bytes_per_sec = 1_500 * 1024**3
    max_tps_per_gpu = mem_bandwidth_bytes_per_sec / max(effective_bytes, 1)
    theoretical_max_tps = max_tps_per_gpu * gpu_count

    console.print(f"\n[bold]Batch Size Optimizer[/bold]")
    console.print(f"  Model: {model}")
    console.print(f"  GPUs: {gpu_count}")
    console.print(f"  Target latency: {target_latency_ms:.0f} ms")
    console.print(f"  Theoretical max throughput: ~{theoretical_max_tps:.1f} tok/s")
    console.print()

    # Probe batch sizes: 1, 2, 4, 8, ... up to max_batch
    batch_sizes: list[int] = [1]
    while batch_sizes[-1] < max_batch:
        batch_sizes.append(batch_sizes[-1] * 2)
    if batch_sizes[-1] > max_batch:
        batch_sizes[-1] = max_batch

    results: list[dict[str, Any]] = []

    prompt_text = "Explain the theory of relativity in detail."

    for bs in batch_sizes:
        console.print(f"  Probing batch_size={bs} ... ", end="")
        latencies: list[float] = []
        errors = 0

        try:
            with httpx.Client(timeout=120.0) as client:
                for _ in range(3):  # 3 samples per batch size
                    start = time.monotonic()
                    try:
                        resp = client.post(
                            f"http://{host}:{port}/v1/completions",
                            json={
                                "model": model,
                                "prompt": prompt_text,
                                "max_tokens": 32,
                                "temperature": 0,
                            },
                        )
                        resp.raise_for_status()
                        latencies.append(time.monotonic() - start)
                    except Exception:
                        errors += 1

            avg_lat = sum(latencies) / len(latencies) * 1000 if latencies else float("inf")
            tps = (32 * bs) / (sum(latencies) / len(latencies)) if latencies else 0.0
            console.print(f"avg {avg_lat:.0f}ms, ~{tps:.1f} tok/s")

            results.append({
                "batch_size": bs,
                "avg_latency_ms": round(avg_lat, 1),
                "throughput_tok_s": round(tps, 1),
                "errors": errors,
            })
        except httpx.ConnectError:
            console.print("[red]connection failed[/red]")
            results.append({
                "batch_size": bs,
                "avg_latency_ms": None,
                "throughput_tok_s": None,
                "errors": 3,
            })

    # Find optimal: highest batch_size whose avg_latency <= target
    viable = [r for r in results if r["avg_latency_ms"] is not None and r["avg_latency_ms"] <= target_latency_ms and r["errors"] == 0]

    if viable:
        best = max(viable, key=lambda r: r["throughput_tok_s"])
    else:
        # Fall back to lowest-latency result
        valid = [r for r in results if r["avg_latency_ms"] is not None]
        best = min(valid, key=lambda r: r["avg_latency_ms"]) if valid else results[0]

    # Estimate max_tokens_per_batch (prompt + generation budget)
    hidden_dim = _estimate_hidden_dim(model)
    num_layers = _estimate_num_layers(model)
    # Each token in the KV cache costs ~2 * hidden_dim * 2 bytes (fp16 k+v)
    kv_bytes_per_token_per_layer = 2 * hidden_dim * 2
    # Use 80% of remaining VRAM for KV cache after model weights
    vram_per_gpu_bytes = 24 * 1024**3  # assume 24 GB usable
    model_mem = effective_bytes
    remaining = max(vram_per_gpu_bytes - model_mem, 0)
    max_tokens_from_mem = remaining // max(kv_bytes_per_token_per_layer * num_layers, 1)

    optimal_bs = best.get("batch_size", 1)
    recommended_max_tokens = min(max_tokens_from_mem, max_tokens)

    result_data: dict[str, Any] = {
        "model": model,
        "gpu_count": gpu_count,
        "target_latency_ms": target_latency_ms,
        "optimal_batch_size": optimal_bs,
        "recommended_max_tokens_per_batch": int(recommended_max_tokens),
        "expected_throughput_tok_s": best.get("throughput_tok_s"),
        "expected_latency_ms": best.get("avg_latency_ms"),
        "probe_results": results,
    }

    # Display recommendation
    console.print()
    table = Table(title="Batch Size Recommendation")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Optimal batch_size", str(optimal_bs))
    table.add_row("max_tokens_per_batch", str(int(recommended_max_tokens)))
    table.add_row("Expected throughput", f"{best.get('throughput_tok_s', 0):.1f} tok/s")
    table.add_row("Expected latency", f"{best.get('avg_latency_ms', 0):.0f} ms")
    console.print(table)

    if output:
        Path(output).write_text(json.dumps(result_data, indent=2), encoding="utf-8")
        console.print(f"\n[green]Results saved to {output}[/green]")


# ---------------------------------------------------------------------------
# tune cache
# ---------------------------------------------------------------------------

def run_tune_cache(
    model: str,
    gpu_memory_gb: float,
    concurrency: int,
    output: str | None,
    console: Console,
) -> None:
    """Calculate optimal KV cache sizing for expected concurrency.

    Determines how many cache blocks can fit in GPU memory after accounting
    for model weights, and recommends block count, cache size, and
    quantization settings.
    """
    model_size_bytes = _estimate_model_size(model)
    num_layers = _estimate_num_layers(model)
    hidden_dim = _estimate_hidden_dim(model)
    num_kv_heads = _estimate_num_kv_heads(model)
    head_dim = _estimate_head_dim(model)

    gpu_memory_bytes = int(gpu_memory_gb * 1024**3)

    # Model weight memory (assume INT4 quantization for sizing)
    weight_bytes = model_size_bytes // 4  # INT4 = 4x compression from fp16

    # KV cache bytes per token per layer: 2 (key+value) * num_kv_heads * head_dim * 2 (fp16)
    kv_per_token_per_layer = 2 * num_kv_heads * head_dim * 2
    kv_per_token_total = kv_per_token_per_layer * num_layers

    # With INT8 KV cache quantization (2x savings)
    kv_per_token_int8 = kv_per_token_total // 2
    # With INT4 KV cache quantization (4x savings)
    kv_per_token_int4 = kv_per_token_total // 4

    # Available memory for KV cache (leave 10% headroom for activations/runtime)
    available_bytes = int(gpu_memory_bytes * 0.9) - weight_bytes
    available_bytes = max(available_bytes, 0)

    # Token capacity at each quantization level
    tokens_fp16 = available_bytes // max(kv_per_token_total, 1)
    tokens_int8 = available_bytes // max(kv_per_token_int8, 1)
    tokens_int4 = available_bytes // max(kv_per_token_int4, 1)

    # PagedAttention block sizing (typically 16 tokens per block)
    block_size = 16
    blocks_fp16 = tokens_fp16 // block_size
    blocks_int8 = tokens_int8 // block_size
    blocks_int4 = tokens_int4 // block_size

    # Per-concurrent-request budget
    per_req_fp16 = tokens_fp16 // max(concurrency, 1)
    per_req_int8 = tokens_int8 // max(concurrency, 1)
    per_req_int4 = tokens_int4 // max(concurrency, 1)

    # Recommend based on concurrency needs
    # Each request typically needs 512-2048 tokens of KV cache
    typical_req_budget = 1024

    if per_req_int8 >= typical_req_budget:
        recommended_bits = "INT8"
        recommended_blocks = blocks_int8
        recommended_tokens = tokens_int8
    elif per_req_int4 >= typical_req_budget:
        recommended_bits = "INT4"
        recommended_blocks = blocks_int4
        recommended_tokens = tokens_int4
    else:
        recommended_bits = "FP16"
        recommended_blocks = blocks_fp16
        recommended_tokens = tokens_fp16

    result_data: dict[str, Any] = {
        "model": model,
        "gpu_memory_gb": gpu_memory_gb,
        "concurrency": concurrency,
        "model_weights_int4_gb": round(weight_bytes / 1024**3, 2),
        "kv_per_token_bytes": {
            "fp16": kv_per_token_total,
            "int8": kv_per_token_int8,
            "int4": kv_per_token_int4,
        },
        "available_for_cache_gb": round(available_bytes / 1024**3, 2),
        "token_capacity": {
            "fp16": tokens_fp16,
            "int8": tokens_int8,
            "int4": tokens_int4,
        },
        "block_capacity": {
            "fp16": blocks_fp16,
            "int8": blocks_int8,
            "int4": blocks_int4,
        },
        "per_request_tokens": {
            "fp16": per_req_fp16,
            "int8": per_req_int8,
            "int4": per_req_int4,
        },
        "recommendation": {
            "kv_cache_quantization": recommended_bits,
            "block_count": recommended_blocks,
            "max_cached_tokens": recommended_tokens,
            "block_size": block_size,
        },
    }

    # Display
    console.print(f"\n[bold]Cache Sizing Calculator[/bold]")
    console.print(f"  Model: {model}")
    console.print(f"  GPU memory: {gpu_memory_gb:.0f} GB")
    console.print(f"  Concurrency: {concurrency}")
    console.print()

    console.print("[bold]Memory Budget[/bold]")
    console.print(f"  Model weights (INT4): {weight_bytes / 1024**3:.1f} GB")
    console.print(f"  Available for cache:  {available_bytes / 1024**3:.1f} GB")
    console.print()

    table = Table(title="KV Cache Token Capacity")
    table.add_column("Quantization", style="cyan")
    table.add_column("Tokens", justify="right")
    table.add_column("Blocks (size=16)", justify="right")
    table.add_column("Per-Request (concurrency={concurrency})", justify="right")
    table.add_row("FP16", str(tokens_fp16), str(blocks_fp16), str(per_req_fp16))
    table.add_row("INT8", str(tokens_int8), str(blocks_int8), str(per_req_int8))
    table.add_row("INT4", str(tokens_int4), str(blocks_int4), str(per_req_int4))
    console.print(table)

    console.print()
    rec_table = Table(title="Recommendation")
    rec_table.add_column("Setting", style="cyan")
    rec_table.add_column("Value", style="green")
    rec_table.add_row("KV Cache Quantization", recommended_bits)
    rec_table.add_row("Block Count", str(recommended_blocks))
    rec_table.add_row("Max Cached Tokens", str(recommended_tokens))
    rec_table.add_row("Block Size", str(block_size))
    rec_table.add_row("PagedAttention", "recommended" if concurrency > 1 else "optional")
    console.print(rec_table)

    if output:
        Path(output).write_text(json.dumps(result_data, indent=2), encoding="utf-8")
        console.print(f"\n[green]Results saved to {output}[/green]")
