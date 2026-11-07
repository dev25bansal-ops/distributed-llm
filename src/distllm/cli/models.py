"""Pydantic models for CLI argument validation + models command helpers.

Provides structured validation for the argparse-based CLI commands.
Each model mirrors a CLI command's arguments with type coercion,
range validation, and default values.

Also implements the HTTP helpers behind ``distllm model list/info/load/unload``
(registered in ``distllm.cli.main`` as the ``model`` group subcommands).
"""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator, model_validator
from rich.console import Console
from rich.table import Table

console = Console()


class CoordinatorArgs(BaseModel):
    """Validated arguments for distllm-coordinator."""
    model_name: str = Field(..., description="Model name or path")
    port: int = Field(default=50050, ge=1024, le=65535)
    dtype: str = Field(default="float16", pattern=r"^(float16|float32|bfloat16)$")
    local: bool = Field(default=False)
    chat_mode: bool = Field(default=False)
    trust_remote_code: bool = Field(default=False)
    debug: bool = Field(default=False)

    @field_validator("model_name")
    @classmethod
    def model_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model_name must not be empty")
        return v.strip()


class ApiServerArgs(BaseModel):
    """Validated arguments for distllm-api."""
    model_name: str = Field(..., description="Model name or path")
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000, ge=1024, le=65535)
    dtype: str = Field(default="float16", pattern=r"^(float16|float32|bfloat16)$")
    local: bool = Field(default=False)
    debug: bool = Field(default=False)
    no_auth: bool = Field(default=False)


class WorkerArgs(BaseModel):
    """Validated arguments for distllm-node."""
    node_id: str = Field(..., description="Unique worker node ID")
    model_name: str = Field(..., description="Model name")
    start_layer: int = Field(..., ge=0, description="First layer index")
    end_layer: int = Field(..., ge=0, description="Last layer index")
    total_layers: int = Field(..., ge=1, description="Total model layers")
    coordinator_host: str = Field(default="localhost")
    coordinator_port: int = Field(default=50050, ge=1024, le=65535)
    port: int = Field(default=50051, ge=1024, le=65535)
    dtype: str = Field(default="float16", pattern=r"^(float16|float32|bfloat16)$")

    @field_validator("end_layer")
    @classmethod
    def end_gte_start(cls, v: int, info: Any) -> int:
        if "start_layer" in info.data and v < info.data["start_layer"]:
            raise ValueError(f"end_layer ({v}) must be >= start_layer ({info.data['start_layer']})")
        return v

    @model_validator(mode="after")
    def _validate_layer_bounds(self) -> WorkerArgs:
        """Ensure the worker covers at least one layer (no zero-layer workers)."""
        if self.start_layer >= self.total_layers:
            raise ValueError(
                f"start_layer ({self.start_layer}) must be < total_layers ({self.total_layers})"
            )
        if self.end_layer >= self.total_layers:
            raise ValueError(
                f"end_layer ({self.end_layer}) must be < total_layers ({self.total_layers})"
            )
        return self


# ── ``distllm model`` group helpers ─────────────────────────────────────────


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
