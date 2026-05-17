"""Logs command: distllm logs."""

import httpx
import json
import time
from rich.console import Console
from rich.live import Live
from typing import Optional

console = Console()


def _stream_logs(
    host: str,
    port: int,
    follow: bool = False,
    lines: int = 50,
    level: Optional[str] = None,
    component: Optional[str] = None,
    search: Optional[str] = None,
):
    """Stream or fetch logs from the server."""
    params = {"lines": lines}
    if level:
        params["level"] = level
    if component:
        params["component"] = component
    if search:
        params["search"] = search

    try:
        with httpx.Client(base_url=f"http://{host}:{port}", timeout=30.0) as client:
            if follow:
                params["follow"] = True
                with client.stream("GET", "/v1/logs", params=params) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if line:
                            try:
                                entry = json.loads(line)
                                _print_log_entry(entry)
                            except json.JSONDecodeError:
                                console.print(line)
            else:
                resp = client.get("/v1/logs", params=params)
                resp.raise_for_status()
                data = resp.json()

                logs = data.get("logs", [])
                if not logs:
                    console.print("[yellow]No log entries found[/yellow]")
                    return

                for entry in logs:
                    _print_log_entry(entry)
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Could not connect to {host}:{port}")
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error:[/red] {e.response.status_code} {e.response.text}")


def _print_log_entry(entry: dict):
    """Print a single log entry with color formatting."""
    level = entry.get("level", "INFO").upper()
    timestamp = entry.get("timestamp", "")
    component = entry.get("component", "")
    message = entry.get("message", "")

    level_colors = {
        "DEBUG": "blue",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold red",
    }
    color = level_colors.get(level, "white")

    parts = []
    if timestamp:
        parts.append(f"[dim]{timestamp}[/dim]")
    parts.append(f"[{color}]{level}[/{color}]")
    if component:
        parts.append(f"[cyan]{component}[/cyan]")
    parts.append(message)

    console.print(" ".join(parts))
