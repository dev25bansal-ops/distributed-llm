"""Cluster status command for DistLLM CLI.

Two presentations of the same information:

* ``show_status()`` — full Rich table; used by ``distllm cluster status``.
* ``fetch_one_line_summary()`` — single JSON payload with a one-line
  summary; used by the top-level ``distllm status`` for shell prompts
  and cron jobs.

The top-level command emits a single colored line::

    OK    coordinator=localhost:8000  nodes=3  healthy=3  models=1  load=42%

Exit codes:
  0  cluster reachable and healthy
  1  cluster unreachable
  2  cluster reachable but degraded
"""

import json
from typing import Any

import httpx
import typer
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
        # CQ-08: failure is surfaced via console output; not propagated.
        console.print(f"[red]Error:[/red] {e}")


# ── One-line summary ────────────────────────────────────────────────────────


def fetch_one_line_summary(host: str, port: int, timeout: float = 3.0) -> str:
    """Hit ``/health`` and ``/v1/cluster/nodes``, return a JSON one-liner.

    Pure function — no console output. Returned JSON has fields
    ``status`` (``ok``/``degraded``/``unreachable``), ``coordinator``,
    ``nodes``, ``models``, ``load`` and the formatted ``line`` itself.
    """
    base_url = f"http://{host}:{port}"
    summary: dict[str, Any] = {
        "status": "unreachable",
        "coordinator": f"{host}:{port}",
        "nodes": 0,
        "models": 0,
        "load": 0.0,
        "line": "",
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            # /health — coordinator liveness
            try:
                health_resp = client.get(f"{base_url}/health")
                health_resp.raise_for_status()
                health = health_resp.json()
            except Exception:
                # CQ-08: failure is recorded in the output value (visible to the caller); not propagated.
                summary["line"] = (
                    f"DOWN  coordinator={host}:{port}  (health endpoint unreachable)"
                )
                return json.dumps(summary)

            # /v1/cluster/nodes — node roster (path varies; try common forms)
            nodes: list[dict[str, Any]] = []
            for path in ("/v1/cluster/nodes", "/admin/v1/nodes", "/v1/nodes"):
                try:
                    nodes_resp = client.get(f"{base_url}{path}")
                    if nodes_resp.status_code == 200:
                        data = nodes_resp.json()
                        if isinstance(data, dict):
                            nodes = list(data.get("nodes", []))
                        elif isinstance(data, list):
                            nodes = data
                        break
                except Exception:
                    # CQ-08: skip this item and continue with the rest (partial results are acceptable).
                    continue

            # /v1/models — model count
            models: list[dict[str, Any]] = []
            try:
                models_resp = client.get(f"{base_url}/v1/models")
                if models_resp.status_code == 200:
                    models_data = models_resp.json()
                    if isinstance(models_data, dict):
                        models = list(models_data.get("data", []))
            except Exception:
                # CQ-08: best-effort: failure is non-fatal (optional capability / telemetry / cleanup).
                pass

            healthy_nodes = sum(
                1
                for n in nodes
                if n.get("healthy") is True or n.get("status") == "ready"
            )
            total_nodes = len(nodes)

            # Aggregate memory pressure across reported nodes
            mem_used = 0.0
            mem_total = 0.0
            for n in nodes:
                mu = n.get("memory_used") or n.get("mem_used") or 0
                mt = n.get("memory_total") or n.get("mem_total") or 0
                if mt > 0:
                    mem_used += float(mu)
                    mem_total += float(mt)
            load = (mem_used / mem_total) if mem_total > 0 else 0.0

            overall = health.get("status", "unknown")
            if overall not in ("ok", "healthy"):
                status = "degraded"
            elif total_nodes > 0 and healthy_nodes < total_nodes:
                status = "degraded"
            else:
                status = "ok"

            summary.update(
                {
                    "status": status,
                    "nodes": total_nodes,
                    "models": len(models),
                    "load": round(load, 3),
                }
            )

            line = (
                f"{status.upper():4}  coordinator={host}:{port}  "
                f"nodes={total_nodes}  healthy={healthy_nodes}  "
                f"models={len(models)}  load={load:.0%}"
            )
            summary["line"] = line
            return json.dumps(summary)

    except httpx.ConnectError:
        summary["line"] = f"DOWN  coordinator={host}:{port}  (connection refused)"
        return json.dumps(summary)
    except Exception as e:
        # CQ-08: failure is recorded in the output value (visible to the caller); not propagated.
        summary["line"] = f"DOWN  coordinator={host}:{port}  ({e})"
        return json.dumps(summary)


def parse_status_payload(payload: str) -> dict[str, Any]:
    """Parse the JSON payload returned by ``fetch_one_line_summary``."""
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return {"status": "unreachable", "line": payload}


def render_status_line(line: str, status: str, console: Console) -> None:
    """Print a one-line status with color. Pure side effect."""
    style_map = {
        "ok": "green",
        "degraded": "yellow",
        "unreachable": "red",
    }
    style = style_map.get(status, "white")
    console.print(line, style=style)


def register(app: typer.Typer) -> None:
    """Attach the top-level ``status`` command."""

    @app.command("status")
    def status_command(
        host: str = typer.Option("localhost", "--host", help="API host"),
        port: int = typer.Option(8000, "--port", "-p", help="API port"),
        json_output: bool = typer.Option(False, "--json", help="Output JSON instead of colored line"),
    ) -> None:
        """One-line cluster health summary."""
        console = Console()
        payload = fetch_one_line_summary(host, port)
        info = parse_status_payload(payload)

        if json_output:
            console.print_json(data=info)
        else:
            render_status_line(info["line"], info["status"], console)

        # Exit code for shell scripts
        code = {"ok": 0, "degraded": 2, "unreachable": 1}.get(info["status"], 1)
        if code != 0:
            raise typer.Exit(code)
