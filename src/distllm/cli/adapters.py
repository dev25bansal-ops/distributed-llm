"""Adapters command group: distllm adapters."""

import httpx
import typer
from rich.console import Console
from rich.table import Table

from distllm.cli.http_client import client_for, format_http_error

console = Console()


def _get_client(host: str, port: int) -> httpx.Client:
    return client_for(host, port)


_ADAPTERS_UNSUPPORTED = (
    "Adapter management is not supported by this server build: there is no "
    "/v1/adapters route.\n"
    "Per-request adapters still work -- pass `adapter` in a chat request "
    "(e.g. via the API / SDK) to apply a loaded LoRA.\n"
    "This command is a placeholder until an adapter registry exists."
)


def _unsupported() -> None:
    """Report adapter management as unsupported instead of a bare 404."""
    console.print(f"[yellow]{_ADAPTERS_UNSUPPORTED}[/yellow]")
    raise typer.Exit(2)


def _list_adapters(host: str, port: int):
    """List loaded adapters."""
    _unsupported()
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
        console.print(f"[red]Error:[/red] {format_http_error(e)}")


def _load_adapter(host: str, port: int, adapter_id: str, source: str, adapter_type: str = "lora"):
    """Load an adapter (LoRA, etc.)."""
    _unsupported()
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
        console.print(f"[red]Error:[/red] {format_http_error(e)}")


def _set_adapter(host: str, port: int, adapter_id: str):
    """Set an adapter as the active adapter."""
    _unsupported()
    try:
        with _get_client(host, port) as client:
            resp = client.post("/v1/adapters/set", json={"adapter_id": adapter_id})
            resp.raise_for_status()

        console.print(f"[green]Adapter activated:[/green] {adapter_id}")
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Could not connect to {host}:{port}")
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error:[/red] {format_http_error(e)}")


def _unload_adapter(host: str, port: int, adapter_id: str):
    """Unload an adapter."""
    _unsupported()
    try:
        with _get_client(host, port) as client:
            resp = client.post("/v1/adapters/unload", json={"adapter_id": adapter_id})
            resp.raise_for_status()

        console.print(f"[green]Adapter unloaded:[/green] {adapter_id}")
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Could not connect to {host}:{port}")
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error:[/red] {format_http_error(e)}")
