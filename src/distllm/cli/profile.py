"""Profile command for DistLLM CLI.

Profiles model inference performance: latency, throughput, memory.
"""

import time
import json
from typing import Any

import httpx
from rich.console import Console
from rich.table import Table


console = Console()


def run_profile(
    model: str,
    host: str,
    port: int,
    prompt_len: int,
    gen_len: int,
    num_iterations: int,
    output: str | None,
    console: Console,
) -> None:
    """Profile model inference performance."""
    console.print(f"[bold]Profiling:[/bold] {model}")
    console.print(f"  Host: {host}:{port}")
    console.print(f"  Prompt length: {prompt_len} tokens")
    console.print(f"  Generation length: {gen_len} tokens")
    console.print(f"  Iterations: {num_iterations}")
    console.print()

    # Generate a dummy prompt of the specified length
    prompt_tokens = [1000 + i % 100 for i in range(prompt_len)]

    results: list[dict[str, Any]] = []

    console.print("[bold]Running profiling iterations...[/bold]")
    for i in range(num_iterations):
        start = time.time()
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"http://{host}:{port}/v1/completions",
                    json={
                        "model": model,
                        "prompt": f"Token sequence: {prompt_tokens[:10]}...",
                        "max_tokens": gen_len,
                        "temperature": 0,
                    },
                )
                response.raise_for_status()
                data = response.json()

            latency = time.time() - start
            usage = data.get("usage", {})
            completion_tokens = usage.get("completion_tokens", gen_len)
            tps = usage.get("tokens_per_second", 0)

            # Estimate TTFT from response headers or generation_time
            gen_time = data.get("generation_time", latency)
            ttft_estimate = latency - gen_time if gen_time < latency else latency * 0.3

            results.append({
                "iteration": i + 1,
                "total_latency_ms": round(latency * 1000, 2),
                "ttft_ms": round(ttft_estimate * 1000, 2),
                "generation_time_ms": round(gen_time * 1000, 2),
                "completion_tokens": completion_tokens,
                "tokens_per_second": round(tps, 2) if tps else round(completion_tokens / max(gen_time, 0.001), 2),
            })
            console.print(f"  [{i+1}/{num_iterations}] {latency*1000:.1f}ms, {completion_tokens} tokens")
        except httpx.ConnectError:
            console.print(f"  [{i+1}/{num_iterations}] [red]Connection failed[/red]")
            results.append({"iteration": i + 1, "error": "connection_failed"})
        except Exception as e:
            console.print(f"  [{i+1}/{num_iterations}] [red]Error: {e}[/red]")
            results.append({"iteration": i + 1, "error": str(e)})

    # Compute statistics
    successful = [r for r in results if "error" not in r]
    if not successful:
        console.print("[red]No successful iterations. Check that the API server is running.[/red]")
        return

    latencies = [r["total_latency_ms"] for r in successful]
    ttfts = [r["ttft_ms"] for r in successful]
    tpss = [r["tokens_per_second"] for r in successful]

    def percentile(data: list[float], p: float) -> float:
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]

    # Print results table
    console.print()
    table = Table(title="Profiling Results")
    table.add_column("Metric", style="cyan")
    table.add_column("P50", style="green")
    table.add_column("P95", style="green")
    table.add_column("P99", style="green")
    table.add_column("Avg", style="yellow")

    table.add_row(
        "Total Latency (ms)",
        f"{percentile(latencies, 50):.1f}",
        f"{percentile(latencies, 95):.1f}",
        f"{percentile(latencies, 99):.1f}",
        f"{sum(latencies) / len(latencies):.1f}",
    )
    table.add_row(
        "TTFT (ms)",
        f"{percentile(ttfts, 50):.1f}",
        f"{percentile(ttfts, 95):.1f}",
        f"{percentile(ttfts, 99):.1f}",
        f"{sum(ttfts) / len(ttfts):.1f}",
    )
    table.add_row(
        "Throughput (tok/s)",
        f"{percentile(tpss, 50):.1f}",
        f"{percentile(tpss, 95):.1f}",
        f"{percentile(tpss, 99):.1f}",
        f"{sum(tpss) / len(tpss):.1f}",
    )

    console.print(table)

    if output:
        summary = {
            "model": model,
            "prompt_len": prompt_len,
            "gen_len": gen_len,
            "iterations": num_iterations,
            "successful": len(successful),
            "latency_p50_ms": percentile(latencies, 50),
            "latency_p95_ms": percentile(latencies, 95),
            "latency_p99_ms": percentile(latencies, 99),
            "latency_avg_ms": sum(latencies) / len(latencies),
            "ttft_p50_ms": percentile(ttfts, 50),
            "ttft_p95_ms": percentile(ttfts, 95),
            "ttft_p99_ms": percentile(ttfts, 99),
            "throughput_p50": percentile(tpss, 50),
            "throughput_p95": percentile(tpss, 95),
            "throughput_avg": sum(tpss) / len(tpss),
            "results": results,
        }
        with open(output, "w") as f:
            json.dump(summary, f, indent=2)
        console.print(f"\n[yellow]Results saved to {output}[/yellow]")
