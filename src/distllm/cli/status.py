"""Cluster status command for DistLLM CLI."""

import httpx
from rich.console import Console
from rich.table import Table


def show_status(host: str, port: int, console: Console):
    """Show cluster status and node health."""
    base_url = f"http://{host}:{port}"

    try:
        with httpx.Client() as client:
            # Health check
            response = client.get(f"{base_url}/health", timeout=10.0)
            response.raise_for_status()
            health = response.json()

            # Models
            models_response = client.get(f"{base_url}/v1/models", timeout=10.0)
            models_response.raise_for_status()
            models = models_response.json()

            # Metrics
            metrics_response = client.get(f"{base_url}/metrics", timeout=10.0)
            metrics_response.raise_for_status()
            metrics_text = metrics_response.text

        # Display health
        console.print(f"\n[bold blue]Cluster Status[/bold blue]\n")

        status_table = Table(title="System Health")
        status_table.add_column("Metric", style="cyan")
        status_table.add_column("Value", style="green")

        status_table.add_row("Status", health.get("status", "unknown"))
        status_table.add_row("Model", health.get("model", "N/A"))
        status_table.add_row("Nodes", str(health.get("nodes", 0)))

        if "node_health" in health:
            for node_id, node_info in health["node_health"].items():
                healthy = "✓ Healthy" if node_info.get("healthy") else "✗ Unhealthy"
                status_table.add_row(f"Node: {node_id}", healthy)
                if "memory_used" in node_info and node_info.get("memory_total", 0) > 0:
                    mem_pct = node_info["memory_used"] / node_info["memory_total"] * 100
                    status_table.add_row(f"  Memory", f"{mem_pct:.1f}%")

        console.print(status_table)

        # Display models
        console.print(f"\n[bold]Available Models:[/bold]")
        for model_data in models.get("data", []):
            console.print(f"  • {model_data['id']}")

        # Display key metrics
        console.print(f"\n[bold]Metrics:[/bold]")
        for line in metrics_text.split("\n")[:10]:  # Show first 10 metrics
            if line and not line.startswith("#"):
                console.print(f"  {line}")

    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Could not connect to {base_url}")
        console.print("Is the API server running?")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
