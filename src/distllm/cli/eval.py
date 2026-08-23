"""CLI command for running model evaluation benchmarks.

Imported by :mod:`distllm.cli.main` which registers the ``eval`` sub-command
on the ``benchmark`` group.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from distllm.cli.client import DistLLMClient


console = Console()


def eval_run(
    model: str = typer.Argument(..., help="Model identifier to evaluate"),
    benchmarks: str = typer.Option(
        "mmlu,gsm8k",
        "--benchmarks",
        "-b",
        help="Comma-separated list of benchmarks: mmlu, gsm8k, humaneval, mt_bench, arena",
    ),
    coordinator_url: str = typer.Option(
        "",
        "--url",
        "-u",
        help="Coordinator API URL (default: http://localhost:8000)",
    ),
    max_tokens: int = typer.Option(256, "--max-tokens", "-t", help="Max generation tokens"),
    temperature: float = typer.Option(0.0, "--temperature", help="Sampling temperature"),
    num_samples: int = typer.Option(20, "--samples", "-n", help="Samples per benchmark"),
    model_b: str = typer.Option("", "--model-b", help="Second model for arena comparison"),
    coordinator_url_b: str = typer.Option(
        "",
        "--url-b",
        help="URL for model B (defaults to --url)",
    ),
    output_json: str = typer.Option("", "--json", "-j", help="Output JSON file path"),
    host: str = typer.Option("localhost", "--host", help="Coordinator host (legacy)"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port (legacy)"),
    api_key: str = typer.Option("", "--api-key", help="API key for authentication"),
) -> None:
    """Run evaluation benchmarks against a model.

    Supports HEIM tasks (MMLU, GSM8K, HumanEval), MT-Bench quality scoring,
    and Chatbot Arena pairwise comparisons.

    Examples::

        # Run MMLU and GSM8K (default)
        distllm benchmark eval my-model

        # Run all HEIM benchmarks
        distllm benchmark eval my-model --benchmarks mmlu,gsm8k,humaneval

        # Run MT-Bench with GPT-4 judge (requires OPENAI_API_KEY)
        distllm benchmark eval my-model --benchmarks mt_bench --samples 4

        # Arena: compare two models
        distllm benchmark eval my-model --benchmarks arena --model-b other-model --samples 10

        # Output as JSON
        distllm benchmark eval my-model --benchmarks mmlu --json results.json
    """
    # Resolve coordinator URL
    if not coordinator_url:
        coordinator_url = f"http://{host}:{port}"

    api_key_resolved = api_key or ""
    url_b = coordinator_url_b or coordinator_url

    # Parse benchmark list
    benchmark_list = [b.strip() for b in benchmarks.split(",") if b.strip()]
    valid = {"mmlu", "gsm8k", "humaneval", "mt_bench", "arena"}
    for b in benchmark_list:
        if b not in valid:
            console.print(f"[red]Unknown benchmark: {b}[/red]")
            console.print(f"[yellow]Valid: {', '.join(sorted(valid))}[/yellow]")
            raise typer.Exit(1)

    console.print(f"\n[bold blue]DistLLM Evaluation Harness[/bold blue]\n")
    console.print(f"Model:         {model}")
    console.print(f"Benchmarks:    {', '.join(benchmark_list)}")
    console.print(f"Coordinator:   {coordinator_url}")
    console.print(f"Samples:       {num_samples}")
    console.print(f"Max tokens:    {max_tokens}")
    console.print(f"Temperature:   {temperature}")
    if "arena" in benchmark_list and model_b:
        console.print(f"Model B:       {model_b}")
    console.print()

    # Build the request payload
    payload = {
        "model_id": model,
        "benchmarks": benchmark_list,
        "coordinator_url": coordinator_url,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "num_samples": num_samples,
        "model_b": model_b,
        "coordinator_url_b": url_b,
    }

    try:
        client = DistLLMClient(base_url=coordinator_url, api_key=api_key_resolved)
        response = client.post("/api/v1/eval/run", json=payload, timeout=600.0)
    except Exception as exc:
        console.print(f"[red]Error connecting to coordinator: {exc}[/red]")
        raise typer.Exit(1)

    if not response.get("success"):
        console.print(f"[red]Evaluation failed: {response.get('error', 'Unknown error')}[/red]")
        raise typer.Exit(1)

    reports = response.get("reports", {})

    for benchmark_name, report in reports.items():
        metrics = report.get("metrics", {})
        config = report.get("config", {})

        table = Table(
            title=f"{benchmark_name.upper()} Results — {report.get('report_id', '')[:8]}",
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        if benchmark_name == "arena":
            table.add_row("Win Rate", f"{metrics.get('win_rate', 0):.1%}")
            table.add_row("Tie Rate", f"{metrics.get('tie_rate', 0):.1%}")
            table.add_row("Loss Rate", f"{metrics.get('loss_rate', 0):.1%}")
        elif benchmark_name == "mt_bench":
            table.add_row("Mean Score", f"{metrics.get('mean_score', 0):.4f}")
        else:
            accuracy = metrics.get("accuracy", 0)
            table.add_row("Accuracy", f"{accuracy:.2%}")

        table.add_row("Samples", str(metrics.get("total_samples", 0)))
        table.add_row("Errors", str(metrics.get("error_samples", 0)))
        table.add_row("Avg Latency", f"{metrics.get('avg_latency_ms', 0):.1f} ms")
        table.add_row("Duration", f"{metrics.get('duration_s', 0):.1f}s")

        # Per-category breakdown
        for key, value in sorted(metrics.items()):
            if key.endswith("_accuracy") and key != "accuracy":
                cat = key.replace("_accuracy", "")
                table.add_row(f"  {cat} accuracy", f"{value:.2%}")

        console.print()
        console.print(table)

    # Write JSON if requested
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(response, indent=2, default=str))
        console.print(f"\n[green]Results written to: {output_path}[/green]")

    console.print()
    console.print("[green]Evaluation complete.[/green]")
