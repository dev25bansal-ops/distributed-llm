"""Benchmark command for DistLLM CLI."""

import time
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
    base_url = f"http://{host}:{port}"
    console.print(f"\n[bold blue]DistLLM Benchmark[/bold blue]\n")
    console.print(f"Model: {model}")
    console.print(f"Prompts: {num_prompts}")
    console.print(f"Max tokens: {max_tokens}\n")

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
                        # Use tokenizer for accurate token count
                        token_count = data.get("usage", {}).get("completion_tokens")
                        if token_count is None:
                            # Fallback: estimate from response generation_time
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
            return
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            return

    # Display results
    if not results:
        console.print("[yellow]No successful benchmarks[/yellow]")
        return

    avg_time = sum(r["elapsed"] for r in results) / len(results)
    avg_tokens = sum(r["tokens"] for r in results) / len(results)
    avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)

    console.print(f"\n[bold green]Benchmark Results[/bold green]\n")

    table = Table(title=f"Performance Summary ({len(results)} runs)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Avg Latency", f"{avg_time:.2f}s")
    table.add_row("Avg Tokens", f"{avg_tokens:.1f}")
    table.add_row("Avg Throughput", f"{avg_tps:.2f} tokens/sec")

    console.print(table)
