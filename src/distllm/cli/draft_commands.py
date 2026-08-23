"""CLI commands for draft model fleet management.

Functions are imported by :mod:`distllm.cli.main` which registers them
as Typer commands on the ``draft`` sub-group.
"""

from rich.console import Console
from rich.table import Table


def draft_fleet_status(
    host: str = "localhost",
    port: int = 8000,
) -> None:
    """Show status of the heterogeneous draft model fleet."""
    import httpx

    console = Console()

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"http://{host}:{port}/v1/speculative/fleet")
            resp.raise_for_status()
            data = resp.json()

            console.print("\n[bold]Draft Model Fleet[/bold]")
            console.print(f"  Total endpoints: {data.get('total_endpoints', 0)}")
            console.print(f"  Healthy: {data.get('healthy_endpoints', 0)}")
            console.print(f"  Total calls: {data.get('total_calls', 0)}")
            console.print(f"  Avg latency: {data.get('avg_latency_ms', 0):.1f}ms")
            console.print(f"  Error rate: {data.get('error_rate', 0):.1%}")

            endpoints = data.get("endpoints", [])
            if endpoints:
                table = Table(title="Draft Endpoints")
                table.add_column("Model")
                table.add_column("Hardware")
                table.add_column("Cost/hr")
                table.add_column("Latency")
                table.add_column("Calls")
                table.add_column("Healthy")
                for ep in endpoints:
                    table.add_row(
                        ep.get("model", "N/A"),
                        ep.get("hardware", "N/A"),
                        f"${ep.get('cost_per_hour', 0):.2f}",
                        f"{ep.get('avg_latency_ms', 0):.1f}ms",
                        str(ep.get("calls", 0)),
                        "✓" if ep.get("healthy") else "✗",
                    )
                console.print(table)
    except httpx.ConnectError:
        console.print(f"[red]Could not connect to coordinator at {host}:{port}[/red]")


def draft_migration_status(
    host: str = "localhost",
    port: int = 8000,
) -> None:
    """Show auto-migration status (CPU↔GPU draft model placement)."""
    import httpx

    console = Console()

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"http://{host}:{port}/v1/speculative/migration")
            resp.raise_for_status()
            data = resp.json()

            console.print("\n[bold]Draft Migration Status[/bold]")
            console.print(f"  Enabled: {data.get('enabled', False)}")
            console.print(f"  Active: {data.get('active_endpoint', 'N/A')} ({data.get('active_hardware', 'N/A')})")
            console.print(f"  CPU endpoints: {data.get('cpu_endpoints', 0)}")
            console.print(f"  GPU endpoints: {data.get('gpu_endpoints', 0)}")
            console.print(f"  Total migrations: {data.get('total_migrations', 0)}")
            config = data.get("config", {})
            if config:
                console.print(f"  GPU high threshold: {config.get('gpu_high_threshold', 80)}%")
                console.print(f"  GPU low threshold: {config.get('gpu_low_threshold', 40)}%")
    except httpx.ConnectError:
        console.print(f"[red]Could not connect to coordinator at {host}:{port}[/red]")
