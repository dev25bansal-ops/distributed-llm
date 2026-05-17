"""Cluster command group: distllm cluster."""

import httpx
from rich.console import Console
from rich.table import Table
from typing import Optional

console = Console()


def _get_client(host: str, port: int) -> httpx.Client:
    return httpx.Client(base_url=f"http://{host}:{port}", timeout=30.0)


def _cluster_status(host: str, port: int):
    """Show cluster status and node health."""
    try:
        with _get_client(host, port) as client:
            resp = client.get("/v1/cluster/status")
            resp.raise_for_status()
            data = resp.json()

        nodes = data.get("nodes", [])
        if not nodes:
            console.print("[yellow]No nodes in cluster[/yellow]")
            return

        table = Table(title="Cluster Nodes")
        table.add_column("Node ID", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("GPU", style="dim")
        table.add_column("Memory", style="dim")
        table.add_column("Requests", style="dim")
        table.add_column("Layers", style="dim")

        for node in nodes:
            table.add_row(
                node.get("id", ""),
                node.get("status", "unknown"),
                node.get("gpu_name", ""),
                node.get("memory_used", ""),
                str(node.get("active_requests", 0)),
                f"{node.get('start_layer', '?')}-{node.get('end_layer', '?')}",
            )

        console.print(table)

        # Summary
        summary = data.get("summary", {})
        if summary:
            console.print(f"\n[bold]Total nodes:[/bold] {summary.get('total_nodes', 0)}")
            console.print(f"[bold]Healthy:[/bold] {summary.get('healthy_nodes', 0)}")
            console.print(f"[bold]Total GPU memory:[/bold] {summary.get('total_gpu_memory', '')}")
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Could not connect to {host}:{port}")
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error:[/red] {e.response.status_code} {e.response.text}")


def _cluster_scale(host: str, port: int, nodes: int, gpu_type: Optional[str] = None):
    """Scale cluster to target node count."""
    try:
        with _get_client(host, port) as client:
            resp = client.post("/v1/cluster/scale", json={
                "target_nodes": nodes,
                "gpu_type": gpu_type,
            })
            resp.raise_for_status()
            data = resp.json()

        console.print(f"[green]Scaling initiated:[/green] {data.get('message', '')}")
        if data.get("job_id"):
            console.print(f"Job ID: {data['job_id']}")
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Could not connect to {host}:{port}")
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error:[/red] {e.response.status_code} {e.response.text}")


def _cluster_drain(host: str, port: int, node_id: str):
    """Drain a node (gracefully remove from service)."""
    try:
        with _get_client(host, port) as client:
            resp = client.post("/v1/cluster/drain", json={"node_id": node_id})
            resp.raise_for_status()
            data = resp.json()

        console.print(f"[green]Node drain initiated:[/green] {node_id}")
        console.print(f"Status: {data.get('status', 'pending')}")
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Could not connect to {host}:{port}")
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error:[/red] {e.response.status_code} {e.response.text}")


def _cluster_rebalance(host: str, port: int, strategy: str = "balanced"):
    """Rebalance load across cluster nodes."""
    try:
        with _get_client(host, port) as client:
            resp = client.post("/v1/cluster/rebalance", json={"strategy": strategy})
            resp.raise_for_status()
            data = resp.json()

        console.print(f"[green]Rebalance initiated:[/green] strategy={strategy}")
        if data.get("message"):
            console.print(data["message"])
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Could not connect to {host}:{port}")
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error:[/red] {e.response.status_code} {e.response.text}")
