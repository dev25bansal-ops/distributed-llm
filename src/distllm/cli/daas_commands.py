"""Draft-as-a-Service CLI commands for DistLLM CLI."""

import asyncio
import time

import httpx
import typer
from rich.console import Console

from distllm.dist.daas_server import DaaSConfig, DaaSServer

console = Console()


def daas_serve(
    model: str = typer.Option("SmolLM-135M", "--model", "-m", help="Draft model name"),
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(9000, "--port", "-p", help="Bind port"),
    api_key: str = typer.Option("", "--api-key", help="API key for authentication"),
    max_concurrent: int = typer.Option(10, "--max-concurrent", help="Max concurrent requests"),
    rate_limit: int = typer.Option(60, "--rate-limit", help="Requests per minute per key"),
    cost_per_hour: float = typer.Option(0.05, "--cost", help="Cost per hour"),
    hardware: str = typer.Option("cpu", "--hardware", help="Hardware type (cpu, cuda:0, mps)"),
    dtype: str = typer.Option("float16", "--dtype", help="Data type"),
) -> None:
    """Start a Draft-as-a-Service inference server.

    Exposes an OpenAI-compatible /v1/completions endpoint that serves
    draft tokens for speculative decoding from any CPU/edge device.

    Example::

        distllm daas serve --model SmolLM-135M --port 9000
    """
    config = DaaSConfig(
        model_name=model,
        host=host,
        port=port,
        api_key=api_key,
        max_concurrent=max_concurrent,
        rate_limit_per_minute=rate_limit,
        cost_per_hour=cost_per_hour,
        hardware=hardware,
        dtype=dtype,
    )
    server = DaaSServer(config)
    server.run()


def daas_status(
    host: str = typer.Option("localhost", "--host", help="DaaS server host"),
    port: int = typer.Option(9000, "--port", "-p", help="DaaS server port"),
) -> None:
    """Show DaaS server status and metrics."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"http://{host}:{port}/metrics")
            resp.raise_for_status()
            metrics = resp.json()

            console.print("\n[bold]Draft-as-a-Service Status[/bold]")
            console.print(f"  Model: {metrics.get('model', 'N/A')}")
            console.print(f"  Hardware: {metrics.get('hardware', 'N/A')}")
            console.print(f"  Total requests: {metrics.get('total_requests', 0)}")
            console.print(f"  Total tokens: {metrics.get('total_tokens_generated', 0)}")
            console.print(f"  Avg latency: {metrics.get('avg_latency_ms', 0):.1f}ms")
            console.print(f"  Tokens/sec: {metrics.get('tokens_per_second', 0):.1f}")
            console.print(f"  Active: {metrics.get('active_requests', 0)}")
            console.print(f"  Errors: {metrics.get('errors', 0)}")
            console.print(f"  Uptime: {metrics.get('uptime_s', 0):.0f}s")
            console.print(f"  Cost: ${metrics.get('cost_per_hour', 0):.2f}/hr")
    except httpx.ConnectError:
        console.print(f"[red]Could not connect to DaaS server at {host}:{port}[/red]")


def daas_benchmark(
    host: str = typer.Option("localhost", "--host", help="DaaS server host"),
    port: int = typer.Option(9000, "--port", "-p", help="DaaS server port"),
    requests: int = typer.Option(100, "--requests", "-n", help="Number of requests"),
    tokens: int = typer.Option(16, "--tokens", "-t", help="Tokens per request"),
    concurrent: int = typer.Option(4, "--concurrent", "-c", help="Concurrent requests"),
) -> None:
    """Benchmark a DaaS server's throughput and latency."""

    async def _bench():
        url = f"http://{host}:{port}/v1/completions"
        payload = {"prompt": [1, 2, 3], "max_tokens": tokens}
        latencies: list[float] = []
        errors = 0

        semaphore = asyncio.Semaphore(concurrent)

        async def _single_request(client: httpx.AsyncClient) -> None:
            nonlocal errors
            async with semaphore:
                start = time.monotonic()
                try:
                    resp = await client.post(url, json=payload, timeout=30.0)
                    resp.raise_for_status()
                    latencies.append(time.monotonic() - start)
                except Exception:
                    errors += 1

        async with httpx.AsyncClient() as client:
            tasks = [_single_request(client) for _ in range(requests)]
            total_start = time.monotonic()
            await asyncio.gather(*tasks)
            total_time = time.monotonic() - total_start

        if latencies:
            latencies.sort()
            console.print("\n[bold]DaaS Benchmark Results[/bold]")
            console.print(f"  Requests: {requests} ({errors} errors)")
            console.print(f"  Total time: {total_time:.2f}s")
            console.print(f"  Throughput: {len(latencies) / total_time:.1f} req/s")
            console.print(f"  Tokens/sec: {len(latencies) * tokens / total_time:.0f}")
            console.print(f"  Latency p50: {latencies[len(latencies) // 2] * 1000:.1f}ms")
            console.print(f"  Latency p95: {latencies[int(len(latencies) * 0.95)] * 1000:.1f}ms")
            console.print(f"  Latency p99: {latencies[int(len(latencies) * 0.99)] * 1000:.1f}ms")
        else:
            console.print("[red]All requests failed[/red]")

    asyncio.run(_bench())
