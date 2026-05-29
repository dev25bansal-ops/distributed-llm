"""CLI commands for the model router: rules, test, stats, patterns."""

from __future__ import annotations

import json
import time

import typer
from rich.console import Console
from rich.table import Table

router_app = typer.Typer(help="Model router: rules, test, stats, patterns")
console = Console()


def _get_router(coordinator_url: str):
    """Fetch router state from the coordinator API."""
    import httpx
    try:
        resp = httpx.get(f"{coordinator_url}/v1/router/capabilities", timeout=5.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        console.print(f"[red]Error connecting to coordinator: {e}[/red]")
        raise typer.Exit(1)


@router_app.command("rules")
def list_rules(
    coordinator_url: str = typer.Option(
        "http://localhost:8000", "--coordinator", "-c", help="Coordinator API URL"
    ),
):
    """List all configured routing rules."""
    data = _get_router(coordinator_url)
    rules = data.get("rules", [])

    if not rules:
        console.print("[yellow]No routing rules configured.[/yellow]")
        return

    table = Table(title="Routing Rules")
    table.add_column("Name", style="cyan")
    table.add_column("Match Type", style="green")
    table.add_column("Pattern", style="white")
    table.add_column("Target Model", style="magenta")
    table.add_column("Priority", justify="right")

    for r in rules:
        table.add_row(
            r.get("name", ""),
            r.get("match_type", ""),
            r.get("pattern", "")[:50],
            r.get("target_model", ""),
            str(r.get("priority", 0)),
        )

    console.print(table)


@router_app.command("test")
def test_routing(
    query: str = typer.Argument(..., help="Query text to route"),
    coordinator_url: str = typer.Option(
        "http://localhost:8000", "--coordinator", "-c", help="Coordinator API URL"
    ),
):
    """Dry-run routing against a query."""
    import httpx
    try:
        resp = httpx.post(
            f"{coordinator_url}/v1/router/test",
            json={"query": query},
            timeout=5.0,
        )
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Query:[/cyan] {query}")
    console.print(f"[green]Model:[/green] {result.get('model', 'N/A')}")
    console.print(f"[green]Rule:[/green]  {result.get('rule_name', 'N/A')}")
    console.print(f"[green]Confidence:[/green] {result.get('confidence', 0):.3f}")
    console.print(f"[green]Latency:[/green] {result.get('latency_ms', 0):.2f}ms")


@router_app.command("stats")
def show_stats(
    coordinator_url: str = typer.Option(
        "http://localhost:8000", "--coordinator", "-c", help="Coordinator API URL"
    ),
):
    """Show routing statistics."""
    data = _get_router(coordinator_url)
    stats = data.get("stats", {})

    table = Table(title="Router Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    for key, value in stats.items():
        if isinstance(value, dict):
            table.add_row(key, json.dumps(value, indent=2))
        else:
            table.add_row(key, str(value))

    console.print(table)


@router_app.command("patterns")
def show_patterns(
    coordinator_url: str = typer.Option(
        "http://localhost:8000", "--coordinator", "-c", help="Coordinator API URL"
    ),
):
    """Show workload classification patterns."""
    data = _get_router(coordinator_url)
    patterns = data.get("workload_patterns", {})

    if not patterns:
        console.print("[yellow]No workload patterns configured.[/yellow]")
        return

    for workload, pats in patterns.items():
        console.print(f"\n[bold cyan]{workload}[/bold cyan]")
        keywords = pats.get("keywords", [])
        regex = pats.get("regex", [])
        console.print(f"  Keywords ({len(keywords)}): {', '.join(keywords[:10])}{'...' if len(keywords) > 10 else ''}")
        console.print(f"  Regex ({len(regex)}): {', '.join(regex[:5])}{'...' if len(regex) > 5 else ''}")
