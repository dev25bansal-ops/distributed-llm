"""Adapters command group: distllm adapters."""

import httpx
from rich.console import Console
from rich.table import Table

console = Console()


def _get_client(host: str, port: int) -> httpx.Client:
    return httpx.Client(base_url=f"http://{host}:{port}", timeout=30.0)


def _list_adapters(host: str, port: int):
    """List loaded adapters."""
    try:
        with _get_client(host, port) as client:
            resp = client.get("/v1/adapters")
            resp.raise_for_status()
            data = resp.json()

        adapters = data.get("adapters", [])
        if not adapters:
            console.print("[yellow]No adapters loaded[/yellow]")
            return

        table = Table(title="Loaded Adapters")
        table.add_column("Adapter ID", style="cyan")
        table.add_column("Source", style="dim")
        table.add_column("Status", style="green")
        table.add_column("Active", style="dim")

        for adapter in adapters:
            table.add_row(
                adapter.get("id", ""),
                adapter.get("source", ""),
                adapter.get("status", "unknown"),
                "Yes" if adapter.get("active") else "No",
            )

        console.print(table)
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Could not connect to {host}:{port}")
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error:[/red] {e.response.status_code} {e.response.text}")


def _load_adapter(host: str, port: int, adapter_id: str, source: str, adapter_type: str = "lora"):
    """Load an adapter (LoRA, etc.)."""
    try:
        with _get_client(host, port) as client:
            resp = client.post("/v1/adapters/load", json={
                "adapter_id": adapter_id,
                "source": source,
                "type": adapter_type,
            })
            resp.raise_for_status()
            data = resp.json()

        console.print(f"[green]Adapter loaded:[/green] {adapter_id}")
        console.print(f"Source: {source}")
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Could not connect to {host}:{port}")
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error:[/red] {e.response.status_code} {e.response.text}")


def _set_adapter(host: str, port: int, adapter_id: str):
    """Set an adapter as the active adapter."""
    try:
        with _get_client(host, port) as client:
            resp = client.post("/v1/adapters/set", json={"adapter_id": adapter_id})
            resp.raise_for_status()

        console.print(f"[green]Adapter activated:[/green] {adapter_id}")
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Could not connect to {host}:{port}")
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error:[/red] {e.response.status_code} {e.response.text}")


def _unload_adapter(host: str, port: int, adapter_id: str):
    """Unload an adapter."""
    try:
        with _get_client(host, port) as client:
            resp = client.post("/v1/adapters/unload", json={"adapter_id": adapter_id})
            resp.raise_for_status()

        console.print(f"[green]Adapter unloaded:[/green] {adapter_id}")
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Could not connect to {host}:{port}")
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error:[/red] {e.response.status_code} {e.response.text}")
