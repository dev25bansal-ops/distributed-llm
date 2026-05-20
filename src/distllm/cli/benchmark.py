"""Benchmark command for DistLLM CLI."""

import time
import json
import os
from pathlib import Path
import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table


TEST_PROMPTS = [
    "Once upon a time in a distant galaxy,",
    "The quick brown fox jumps over",
    "In the beginning, there was",
    "Artificial intelligence is transforming",
    "Distributed computing allows us to",
    "Machine learning models can be",
    "The future of technology depends on",
    "Cloud computing has revolutionized how we",
    "Neural networks are designed to",
    "Natural language processing enables computers to",
]

DEFAULT_BASELINE_PATH = Path.home() / ".distllm" / "benchmarks" / "baseline.json"


def _run_benchmarks(
    model: str,
    host: str,
    port: int,
    num_prompts: int,
    max_tokens: int,
    console: Console,
) -> list:
    """Run benchmark and return results list."""
    base_url = f"http://{host}:{port}"
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Running benchmarks...", total=num_prompts)

        try:
            with httpx.Client(timeout=120.0) as client:
                for i in range(num_prompts):
                    prompt = TEST_PROMPTS[i % len(TEST_PROMPTS)]

                    start_time = time.time()
                    response = client.post(
                        f"{base_url}/v1/completions",
                        json={
                            "model": model,
                            "prompt": prompt,
                            "max_tokens": max_tokens,
                            "temperature": 0.7,
                        },
                    )
                    elapsed = time.time() - start_time

                    if response.status_code == 200:
                        data = response.json()
                        generated_text = data["choices"][0]["text"]
                        token_count = data.get("usage", {}).get("completion_tokens")
                        if token_count is None:
                            token_count = len(generated_text.split())

                        results.append({
                            "prompt": prompt,
                            "elapsed": elapsed,
                            "tokens": token_count,
                            "tokens_per_sec": token_count / elapsed if elapsed > 0 else 0,
                        })
                    else:
                        console.print(f"[red]Request {i+1} failed: {response.status_code}[/red]")

                    progress.advance(task)

        except httpx.ConnectError:
            console.print(f"[red]Error:[/red] Could not connect to {base_url}")
            return []
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            return []

    return results


def _print_results(results: list, console: Console, label: str = "Results"):
    """Print benchmark results as a table."""
    if not results:
        console.print("[yellow]No successful benchmarks[/yellow]")
        return

    avg_time = sum(r["elapsed"] for r in results) / len(results)
    avg_tokens = sum(r["tokens"] for r in results) / len(results)
    avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)

    console.print(f"\n[bold green]{label}[/bold green]\n")

    table = Table(title=f"Performance Summary ({len(results)} runs)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Avg Latency", f"{avg_time:.2f}s")
    table.add_row("Avg Tokens", f"{avg_tokens:.1f}")
    table.add_row("Avg Throughput", f"{avg_tps:.2f} tokens/sec")

    console.print(table)
    return avg_time, avg_tokens, avg_tps


def run_benchmark(
    model: str,
    host: str,
    port: int,
    num_prompts: int,
    max_tokens: int,
    local: bool,
    console: Console,
):
    """Run benchmarks against the API server."""
    console.print(f"\n[bold blue]DistLLM Benchmark[/bold blue]\n")
    console.print(f"Model: {model}")
    console.print(f"Prompts: {num_prompts}")
    console.print(f"Max tokens: {max_tokens}\n")

    results = _run_benchmarks(model, host, port, num_prompts, max_tokens, console)
    _print_results(results, console, "Benchmark Results")
    return results


def run_benchmark_json(
    model: str,
    host: str,
    port: int,
    num_prompts: int,
    max_tokens: int,
    local: bool,
) -> str:
    """Run benchmarks and return JSON output (for CI)."""
    import json
    # Use a null console to suppress rich output
    null_console = Console(quiet=True)
    results = _run_benchmarks(model, host, port, num_prompts, max_tokens, null_console)
    if not results:
        return json.dumps({"error": "No successful benchmarks", "results": []})
    avg_time = sum(r["elapsed"] for r in results) / len(results)
    avg_tokens = sum(r["tokens"] for r in results) / len(results)
    avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)
    return json.dumps({
        "model": model,
        "prompts": num_prompts,
        "max_tokens": max_tokens,
        "num_runs": len(results),
        "avg_latency_seconds": round(avg_time, 3),
        "avg_tokens": round(avg_tokens, 1),
        "avg_throughput_tps": round(avg_tps, 2),
        "results": results,
    }, indent=2)


def run_benchmark_compare(
    model: str,
    host: str,
    port: int,
    num_prompts: int,
    max_tokens: int,
    baseline_path: str | None,
    save_baseline: bool,
    console: Console,
):
    """Run benchmark and compare against saved baseline."""
    console.print(f"\n[bold blue]DistLLM Benchmark Compare[/bold blue]\n")
    console.print(f"Model: {model}")
    console.print(f"Prompts: {num_prompts}")
    console.print(f"Max tokens: {max_tokens}\n")

    # Run current benchmark
    results = _run_benchmarks(model, host, port, num_prompts, max_tokens, console)
    if not results:
        return

    cur_avg_time, cur_avg_tokens, cur_avg_tps = _print_results(results, console, "Current Results")

    # Load or save baseline
    baseline_file = Path(baseline_path) if baseline_path else DEFAULT_BASELINE_PATH

    if save_baseline:
        baseline_file.parent.mkdir(parents=True, exist_ok=True)
        baseline_data = {
            "model": model,
            "num_prompts": num_prompts,
            "max_tokens": max_tokens,
            "avg_latency": cur_avg_time,
            "avg_tokens": cur_avg_tokens,
            "avg_throughput": cur_avg_tps,
            "timestamp": time.time(),
        }
        baseline_file.write_text(json.dumps(baseline_data, indent=2))
        console.print(f"\n[green]Baseline saved to:[/green] {baseline_file}")
        return

    if not baseline_file.exists():
        console.print(f"[yellow]No baseline found at {baseline_file}[/yellow]")
        console.print("Run with --save-baseline to create one first.")
        return

    baseline = json.loads(baseline_file.read_text())
    bl_avg_time = baseline.get("avg_latency", 0)
    bl_avg_tokens = baseline.get("avg_tokens", 0)
    bl_avg_tps = baseline.get("avg_throughput", 0)

    # Comparison table
    console.print(f"\n[bold]Comparison[/bold]  (baseline from {time.strftime('%Y-%m-%d', time.localtime(baseline.get('timestamp', 0)))})\n")

    table = Table(title="Baseline vs Current")
    table.add_column("Metric", style="cyan")
    table.add_column("Baseline", style="dim")
    table.add_column("Current", style="green")
    table.add_column("Delta", style="yellow")

    latency_delta = cur_avg_time - bl_avg_time
    latency_sign = "+" if latency_delta > 0 else ""
    table.add_row("Avg Latency", f"{bl_avg_time:.2f}s", f"{cur_avg_time:.2f}s", f"{latency_sign}{latency_delta:.2f}s")

    tokens_delta = cur_avg_tokens - bl_avg_tokens
    tokens_sign = "+" if tokens_delta > 0 else ""
    table.add_row("Avg Tokens", f"{bl_avg_tokens:.1f}", f"{cur_avg_tokens:.1f}", f"{tokens_sign}{tokens_delta:.1f}")

    tps_delta = cur_avg_tps - bl_avg_tps
    tps_sign = "+" if tps_delta > 0 else ""
    tps_pct = (tps_delta / bl_avg_tps * 100) if bl_avg_tps > 0 else 0
    table.add_row("Avg Throughput", f"{bl_avg_tps:.2f} tok/s", f"{cur_avg_tps:.2f} tok/s", f"{tps_sign}{tps_delta:.2f} tok/s ({tps_pct:+.1f}%)")

    console.print(table)
