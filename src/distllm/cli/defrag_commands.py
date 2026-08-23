"""GPU memory defragmentation commands for DistLLM CLI."""

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def defrag_status(
    host: str = typer.Option("localhost", help="Coordinator host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """Show defragmentation status and fragmentation ratio."""
    import httpx

    base_url = f"http://{host}:{port}"
    try:
        resp = httpx.get(f"{base_url}/v1/defrag/status", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if not data.get("enabled"):
        console.print("[yellow]Defragmentation is disabled[/yellow]")
        return

    t = Table(title="Defragmentation Status")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", style="green")
    t.add_row("Policy", data.get("policy", "N/A"))
    t.add_row("Fragmentation Ratio", f"{data.get('fragmentation_ratio', 0):.2%}")
    t.add_row("Predictive (5 steps)", f"{data.get('predictive_fragmentation', 0):.2%}")
    stats = data.get("stats", {})
    t.add_row("Total Passes", str(stats.get("defrag_count", 0)))
    t.add_row("Blocks Moved", str(stats.get("blocks_moved", 0)))
    t.add_row("Bytes Compacted", f"{stats.get('bytes_compacted', 0) / 1024 / 1024:.1f} MB")
    t.add_row("Total Time", f"{stats.get('total_time_ms', 0):.1f} ms")
    t.add_row("Peak Fragmentation", f"{stats.get('peak_fragmentation_ratio', 0):.2%}")
    console.print(t)

    config = data.get("config", {})
    if config:
        ct = Table(title="Configuration")
        ct.add_column("Setting", style="cyan")
        ct.add_column("Value", style="green")
        ct.add_row("Interval", f"{config.get('interval_seconds', 0)}s")
        ct.add_row("Max Blocks/Pass", str(config.get("max_blocks_per_pass", 0) or "unlimited"))
        ct.add_row("Tiered Compaction", str(config.get("tiered_compaction", False)))
        ct.add_row("Predictive", str(config.get("enable_predictive", False)))
        console.print(ct)


def defrag_run(
    host: str = typer.Option("localhost", help="Coordinator host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """Trigger an immediate defragmentation pass."""
    import httpx

    base_url = f"http://{host}:{port}"
    try:
        resp = httpx.post(f"{base_url}/v1/defrag/run", timeout=120.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if "error" in data:
        console.print(f"[red]{data['error']}[/red]")
        return

    for backend_key, result in data.items():
        t = Table(title=f"Defrag Result — {backend_key}")
        t.add_column("Metric", style="cyan")
        t.add_column("Value", style="green")
        t.add_row("Blocks Moved", str(result.get("blocks_moved", 0)))
        t.add_row("Bytes Compacted", f"{result.get('bytes_compacted', 0) / 1024 / 1024:.1f} MB")
        t.add_row("Duration", f"{result.get('time_ms', 0):.1f} ms")
        t.add_row("Frag Before", f"{result.get('fragmentation_before', 0):.2%}")
        t.add_row("Frag After", f"{result.get('fragmentation_after', 0):.2%}")
        t.add_row("Tier", result.get("tier_used", "N/A"))
        console.print(t)


def defrag_stats(
    host: str = typer.Option("localhost", help="Coordinator host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """Show historical defragmentation statistics."""
    import httpx

    base_url = f"http://{host}:{port}"
    try:
        resp = httpx.get(f"{base_url}/v1/defrag/stats", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if not data.get("enabled"):
        console.print("[yellow]Defragmentation is disabled[/yellow]")
        return

    stats = data.get("stats", {})
    t = Table(title="Defragmentation Statistics")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", style="green")
    t.add_row("Total Passes", str(stats.get("defrag_count", 0)))
    t.add_row("Blocks Moved", str(stats.get("blocks_moved", 0)))
    t.add_row("Bytes Compacted", f"{stats.get('bytes_compacted', 0) / 1024 / 1024:.1f} MB")
    t.add_row("Total Time", f"{stats.get('total_time_ms', 0):.1f} ms")
    t.add_row("Avg Time/Pass", f"{stats.get('total_time_ms', 0) / max(stats.get('defrag_count', 1), 1):.1f} ms")
    t.add_row("L1 (Hot) Passes", str(stats.get("l1_count", 0)))
    t.add_row("L2 (Warm) Passes", str(stats.get("l2_count", 0)))
    t.add_row("L3 (Cold) Passes", str(stats.get("l3_count", 0)))
    t.add_row("Peak Fragmentation", f"{stats.get('peak_fragmentation_ratio', 0):.2%}")
    t.add_row("Current Fragmentation", f"{stats.get('last_fragmentation_ratio', 0):.2%}")
    console.print(t)

    history = data.get("fragmentation_history", [])
    if history:
        console.print(f"\n[bold]Fragmentation History[/bold] (last {len(history)} samples)")
        console.print(f"  Min: {min(history):.2%}, Max: {max(history):.2%}, Avg: {sum(history)/len(history):.2%}")
