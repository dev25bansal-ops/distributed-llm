"""Models command group: distllm models."""

import httpx
from rich.console import Console
from rich.table import Table
from typing import Optional

console = Console()


def _get_client(host: str, port: int) -> httpx.Client:
    return httpx.Client(base_url=f"http://{host}:{port}", timeout=30.0)


def _list_models(host: str, port: int):
    """List available and loaded models."""
    try:
        with _get_client(host, port) as client:
            resp = client.get("/v1/models")
            resp.raise_for_status()
            data = resp.json()

        models = data.get("data", [])
        if not models:
            console.print("[yellow]No models available[/yellow]")
            return

        table = Table(title="Models")
        table.add_column("ID", style="cyan")
        table.add_column("Object", style="dim")
        table.add_column("Owned By", style="dim")

        for m in models:
            table.add_row(m.get("id", ""), m.get("object", ""), m.get("owned_by", ""))

        console.print(table)
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Could not connect to {host}:{port}")
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error:[/red] {e.response.status_code} {e.response.text}")


def _model_info(host: str, port: int, model_id: str):
    """Show detailed model information."""
    try:
        with _get_client(host, port) as client:
            resp = client.get("/v1/models")
            resp.raise_for_status()
            data = resp.json()

        models = data.get("data", [])
        target = next((m for m in models if m.get("id") == model_id), None)

        if not target:
            console.print(f"[red]Model '{model_id}' not found[/red]")
            return

        table = Table(title=f"Model: {model_id}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        for key, value in target.items():
            table.add_row(key, str(value))

        console.print(table)
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Could not connect to {host}:{port}")
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error:[/red] {e.response.status_code} {e.response.text}")


def _load_model(host: str, port: int, model: str, dtype: str = "float16"):
    """Load a model into the server."""
    try:
        with _get_client(host, port) as client:
            resp = client.post("/v1/models/load", json={
                "model": model,
                "dtype": dtype,
            })
            resp.raise_for_status()
            data = resp.json()

        console.print(f"[green]Model loaded:[/green] {data.get('model', model)}")
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Could not connect to {host}:{port}")
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error:[/red] {e.response.status_code} {e.response.text}")


def _unload_model(host: str, port: int, model_id: str):
    """Unload a model from the server."""
    try:
        with _get_client(host, port) as client:
            resp = client.post("/v1/models/unload", json={"model": model_id})
            resp.raise_for_status()

        console.print(f"[green]Model unloaded:[/green] {model_id}")
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Could not connect to {host}:{port}")
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error:[/red] {e.response.status_code} {e.response.text}")
