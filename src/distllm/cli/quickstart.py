"""`distllm quickstart` — one command from nothing to a running cluster.

A new user had to work out that `cluster start --model … --local` both loads
the model and serves the API, that the API key lives in .distllm/api_key, and
that `model search` is how you pick a model. This command does that discovery
for them: it sizes the machine, picks the best model that fits, starts the
cluster, waits for health, and prints what to do next.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _detect() -> dict:
    from distllm.cli.onboard import _detect_hardware

    return _detect_hardware()


def _pick_model(query: str, hardware: dict, explicit: str | None) -> str | None:
    """Best model that fits, or *explicit* if the user named one."""
    if explicit:
        return explicit

    from distllm.models.catalog import search_huggingface
    from distllm.models.recommend import HardwareProfile, rank

    hw = HardwareProfile(
        gpu_memory_gb=float(hardware.get("gpu_memory_gb") or 0.0),
        gpu_count=int(hardware.get("gpu_count") or 0),
        gpu_name=str(hardware.get("gpu_name") or ""),
        ram_gb=float(hardware.get("ram_gb") or 0.0),
    )
    found = search_huggingface(query or "instruct", limit=40)
    pairs = [(c.model_id, c.params) for c in found if c.params]
    if not pairs:
        return None
    fits, _ = rank(pairs, hw, max_results=1)
    return fits[0].model_id if fits else None


def _wait_for_health(port: int, timeout_s: int = 180) -> bool:
    """Poll /health until the server answers or the deadline passes."""
    import httpx

    from distllm.cli.http_client import client_for

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with client_for("127.0.0.1", port, timeout=3.0) as client:
                response = client.get("/health")
            if response.status_code == 200:
                return True
        except (httpx.HTTPError, OSError):
            pass
        time.sleep(2)
    return False


def run_quickstart(
    *,
    model: str | None,
    query: str,
    port: int,
    api_port: int,
    no_start: bool,
) -> int:
    """Entry point wired from ``distllm.cli.main``."""
    console.print(
        Panel.fit(
            "[bold]DistLLM quickstart[/bold]\n"
            "Detect hardware → pick a model → start a cluster → verify",
            border_style="cyan",
        )
    )

    # 1. Hardware
    hw = _detect()
    table = Table(title="1. This machine", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    if hw.get("cuda_available"):
        table.add_row("GPU", f"{hw.get('gpu_name', '?')} ({hw.get('gpu_count', 0)}x)")
        table.add_row("VRAM", f"{hw.get('gpu_memory_gb', 0):.1f} GB per device")
    else:
        table.add_row("GPU", "[yellow]none detected — CPU inference is slow[/yellow]")
    table.add_row("RAM", f"{hw.get('ram_gb', 0):.1f} GB")
    table.add_row("CPU cores", str(hw.get("cpu_cores", "?")))
    console.print(table)

    # 2. Model
    chosen = _pick_model(query, hw, model)
    if not chosen:
        console.print(
            "\n[red]Could not reach Hugging Face to pick a model.[/red]\n"
            "Pass one explicitly: [cyan]distllm quickstart --model "
            "Qwen/Qwen2.5-7B-Instruct[/cyan]"
        )
        return 1
    console.print(f"\n[bold]2. Model:[/bold] [green]{chosen}[/green]")
    console.print("[dim]   (largest one measured to fit this hardware)[/dim]")

    if no_start:
        console.print(
            f"\n[yellow]--no-start:[/yellow] would run\n"
            f"  [cyan]distllm cluster start --model {chosen} --local "
            f"--api-port {api_port}[/cyan]"
        )
        return 0

    # 3. Start
    console.print(f"\n[bold]3. Starting cluster on ports {port}/{api_port}...[/bold]")
    args = [
        sys.executable,
        "-m",
        "distllm.api.server",
        "--model",
        chosen,
        "--local",
        "--port",
        str(api_port),
        "--host",
        "127.0.0.1",
    ]
    try:
        proc = subprocess.Popen(args)
    except OSError as exc:
        console.print(f"[red]Could not start the server:[/red] {exc}")
        return 1

    console.print(f"   [dim]PID {proc.pid} — loading the model, this takes a minute[/dim]")

    # 4. Verify
    console.print("\n[bold]4. Waiting for /health...[/bold]")
    healthy = _wait_for_health(api_port)
    if not healthy:
        console.print(
            f"[red]The server did not become healthy within 3 minutes.[/red]\n"
            f"It is still running as PID {proc.pid}; its output is above.\n"
            "Try a smaller model, or check `distllm cli-doctor`."
        )
        return 1

    console.print("[green]   Healthy.[/green]\n")
    console.print(
        Panel.fit(
            f"[bold green]Cluster is up[/bold green]\n\n"
            f"Chat with it:\n  [cyan]distllm chat --port {api_port}[/cyan]\n\n"
            f"Ask the API directly:\n"
            f"  curl -s localhost:{api_port}/v1/models "
            f"-H \"Authorization: Bearer $(cat .distllm/api_key)\"\n\n"
            f"Stop it:\n  [cyan]kill {proc.pid}[/cyan]",
            border_style="green",
        )
    )
    return 0
