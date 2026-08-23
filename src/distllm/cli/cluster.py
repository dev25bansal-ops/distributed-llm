"""Cluster command group: distllm cluster."""

import os
import signal
import subprocess
import sys
import time
import uuid

import httpx
from rich.console import Console
from rich.table import Table
from loguru import logger

console = Console()


def _get_client(host: str, port: int) -> httpx.Client:
    return httpx.Client(base_url=f"http://{host}:{port}", timeout=30.0)


def _admin_headers() -> dict[str, str]:
    """Authorization headers for admin-only endpoints.

    The ``/admin/v1/*`` routes require an admin-role API key; operators set
    it via ``DISTLLM_API_KEY`` (or the legacy ``API_KEY`` env var).
    """
    api_key = os.environ.get("DISTLLM_API_KEY") or os.environ.get("API_KEY")
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


def _connect_error(host: str, port: int) -> None:
    """Print a friendly message when the coordinator cannot be reached."""
    console.print(
        f"[red]Error:[/red] Could not connect to coordinator at {host}:{port}. "
        f"Is it running? Start it with [bold]distllm cluster start --model <model>[/bold] "
        f"(REST API on port {port}) or check --host/--port."
    )


def _fmt_gib(value) -> str:
    """Format a byte count as GiB, tolerating missing/non-numeric values."""
    try:
        return f"{int(value) / (1024 ** 3):.1f}GB"
    except (TypeError, ValueError):
        return "?"


def _cluster_status(host: str, port: int):
    """Show cluster status and node health (GET /v1/health)."""
    try:
        with _get_client(host, port) as client:
            resp = client.get("/v1/health")
            resp.raise_for_status()
            data = resp.json()

        table = Table(title="Cluster Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Status", str(data.get("status", "unknown")))
        table.add_row("Model", str(data.get("model", "")))
        table.add_row("Nodes", str(data.get("nodes", 0)))
        console.print(table)

        node_health = data.get("node_health") or {}
        if node_health:
            nodes_table = Table(title="Node Health")
            nodes_table.add_column("Node ID", style="cyan")
            nodes_table.add_column("Healthy", style="green")
            for node_id, info in node_health.items():
                healthy = info.get("healthy", False) if isinstance(info, dict) else bool(info)
                nodes_table.add_row(str(node_id), "yes" if healthy else "no")
            console.print(nodes_table)
    except httpx.ConnectError:
        _connect_error(host, port)
    except httpx.TransportError as e:
        console.print(f"[red]Error:[/red] Could not reach coordinator at {host}:{port} ({type(e).__name__})")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            console.print("[yellow]Coordinator reachable but no model is loaded.[/yellow]")
            console.print("Start one with: distllm system api --model <model>")
        else:
            console.print(f"[red]Error:[/red] {e.response.status_code} {e.response.text}")


def _cluster_scale(host: str, port: int, nodes: int, gpu_type: str | None = None):
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


def _cluster_start(model: str, port: int, api_port: int, local: bool, dtype: str, debug: bool):
    """Start a coordinator and optionally the REST API."""
    console.print(f"[bold]Starting coordinator...[/bold]")
    console.print(f"  Model: {model}")
    console.print(f"  gRPC port: {port}")
    console.print(f"  API port: {api_port}")
    console.print(f"  Mode: {'local' if local else 'distributed'}")

    coord_args = [
        sys.executable, "-m", "distllm.core.coordinator",
        "--model", model,
        "--port", str(port),
        "--dtype", dtype,
    ]
    if local:
        coord_args.append("--local")
    if debug:
        coord_args.append("--debug")

    api_args = [
        sys.executable, "-m", "distllm.api.server",
        "--model", model,
        "--port", str(api_port),
        "--dtype", dtype,
    ]
    if local:
        api_args.append("--local")
    if debug:
        api_args.append("--debug")

    try:
        coord_proc = subprocess.Popen(coord_args)
        console.print(f"[green]Coordinator started[/green] (PID: {coord_proc.pid})")

        # Wait for coordinator to be ready by polling health endpoint
        import httpx
        coord_ready = False
        for attempt in range(15):
            time.sleep(1)
            try:
                r = httpx.get(f"http://localhost:{port}/health", timeout=2.0)
                if r.status_code < 500:
                    coord_ready = True
                    break
            except (httpx.ConnectError, httpx.TimeoutException):
                continue
        if coord_ready:
            console.print(f"[green]Coordinator ready[/green] on port {port}")
        else:
            console.print(f"[yellow]Coordinator may not be ready[/yellow] (timeout)")

        api_proc = subprocess.Popen(api_args)
        console.print(f"[green]API server started[/green] (PID: {api_proc.pid})")

        # Wait for API server
        api_ready = False
        for attempt in range(15):
            time.sleep(1)
            try:
                r = httpx.get(f"http://localhost:{api_port}/health", timeout=2.0)
                if r.status_code < 500:
                    api_ready = True
                    break
            except (httpx.ConnectError, httpx.TimeoutException):
                continue
        if api_ready:
            console.print(f"[green]API server ready[/green] on port {api_port}")
        else:
            console.print(f"[yellow]API server may not be ready[/yellow] (timeout)")

        console.print()
        console.print(f"Workers join: distllm cluster join --coordinator localhost:{port}")
        console.print(f"API: http://localhost:{api_port}/v1/chat/completions")
        console.print("Press Ctrl+C to stop")

        try:
            coord_proc.wait()
        except KeyboardInterrupt:
            console.print("[yellow]Stopping cluster...[/yellow]")
            coord_proc.terminate()
            api_proc.terminate()
            try:
                coord_proc.wait(timeout=5)
            except Exception:
                coord_proc.kill()
            try:
                api_proc.wait(timeout=5)
            except Exception:
                api_proc.kill()
            console.print("[yellow]Cluster stopped[/yellow]")
    except FileNotFoundError:
        console.print("[red]Error:[/red] distllm modules not found")


def _cluster_join(
    coordinator_host: str, coordinator_port: int,
    node_id: str | None, start_layer: int | None,
    end_layer: int | None, total_layers: int | None,
    listen_port: int, device: str,
    cluster_key: str | None = None,
    discover: bool = False,
    model: str | None = None,
):
    """Start a worker node and connect to an existing coordinator."""
    if node_id is None:
        node_id = f"node_{uuid.uuid4().hex[:8]}"

    # Auto-discovery via mDNS
    if discover or coordinator_host == "localhost":
        console.print("[bold]Scanning LAN for coordinators...[/bold]")
        from distllm.dist.discovery import DiscoveryClient
        client = DiscoveryClient(timeout=3.0)
        found = client.discover()
        if found:
            svc = found[0]
            coordinator_host = svc["host"]
            coordinator_port = svc["port"]
            console.print(f"  [green]Discovered:[/green] {svc['name']} at {coordinator_host}:{coordinator_port}")
            if svc.get("properties", {}).get("model"):
                console.print(f"  Model: {svc['properties']['model']}")
        elif not discover:
            pass  # use default localhost
        else:
            console.print("[yellow]No coordinators found on LAN[/yellow]")
            return

    console.print(f"[bold]Joining cluster at {coordinator_host}:{coordinator_port}...[/bold]")
    console.print(f"  Node ID: {node_id}  Listen port: {listen_port}")
    if start_layer is not None and end_layer is not None:
        console.print(f"  Layers: {start_layer}-{end_layer} of {total_layers or '?'}")

    # Get model name from coordinator or use provided model
    model_name = model or "unknown"
    if model_name == "unknown":
        try:
            import httpx
            # Try to get model from coordinator's health endpoint
            resp = httpx.get(
                f"http://{coordinator_host}:{coordinator_port + 1}/health",
                timeout=5.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                model_name = data.get("model", "unknown")
            if model_name == "unknown":
                # Try to get from /v1/models
                resp2 = httpx.get(f"http://{coordinator_host}:{coordinator_port + 1}/v1/models", timeout=5.0)
                if resp2.status_code == 200:
                    models = resp2.json().get("data", [])
                    if models:
                        model_name = models[0].get("id", "unknown")
        except Exception:
            pass

    if model_name == "unknown":
        console.print("[red]Error:[/red] Could not determine model name. Use --model option.")
        return

    # Get actual IP address of this machine before spawning worker
    import socket
    worker_host = socket.gethostbyname(socket.gethostname())

    args = [
        sys.executable, "-m", "distllm.dist.worker",
        "--node-id", node_id,
        "--model", model_name,
        "--start-layer", str(start_layer or 0),
        "--end-layer", str(end_layer or 0),
        "--total-layers", str(total_layers or 1),
        "--port", str(listen_port),
        "--coordinator-host", coordinator_host,
        "--coordinator-port", str(coordinator_port),
        "--device", device,
    ]
    env = os.environ.copy()
    if cluster_key:
        env["DISTLLM_CLUSTER_KEY"] = cluster_key
    try:
        proc = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        console.print(f"[green]Worker started[/green] (PID: {proc.pid})")
        console.print(f"[dim]Connected to {coordinator_host}:{coordinator_port}[/dim]")

        # Register with coordinator — wait for worker gRPC port to be ready
        api_port = 8000
        try:
            import socket as _sock
            import time
            # Poll until the worker's gRPC port is open (model loading takes ~30s)
            console.print("[dim]Waiting for worker to finish loading model...[/dim]")
            for attempt in range(60):
                s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
                s.settimeout(2)
                result = s.connect_ex((worker_host, listen_port))
                s.close()
                if result == 0:
                    break
                time.sleep(1)
            else:
                console.print("[yellow]Worker port not open after 60s, registering anyway...[/yellow]")

            import httpx
            reg_headers = {}
            api_key = os.environ.get("API_KEY") or os.environ.get("DISTLLM_API_KEY")
            if api_key:
                reg_headers["Authorization"] = f"Bearer {api_key}"
            resp = httpx.post(
                f"http://{coordinator_host}:{api_port}/admin/v1/nodes/register",
                json={
                    "node_id": node_id,
                    "host": worker_host,
                    "port": listen_port,
                    "start_layer": start_layer or 0,
                    "end_layer": end_layer or 0,
                    "total_layers": total_layers or 1,
                    "device": device,
                },
                headers=reg_headers or None,
                timeout=60.0,
            )
            if resp.status_code in (200, 201):
                console.print(f"[green]Node registered with coordinator at {coordinator_host}:{api_port}[/green]")
            elif resp.status_code == 401:
                console.print("[yellow]Registration requires API key. Set API_KEY env var.[/yellow]")
            elif resp.status_code == 409:
                console.print("[green]Node already registered[/green]")
            else:
                console.print(f"[yellow]Registration: {resp.status_code} {resp.text[:100]}[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Registration: {e}[/yellow]")
        try:
            for line in iter(proc.stdout.readline, ''):
                print(line, end='')
        except KeyboardInterrupt:
            proc.terminate()
            console.print(f"[yellow]Worker {node_id} stopped[/yellow]")
        finally:
            proc.stdout.close()
    except FileNotFoundError:
        console.print("[red]Error:[/red] distllm modules not found")


def _cluster_leave(node_id: str, coordinator_host: str, coordinator_port: int):
    """Gracefully remove a worker node from the cluster."""
    console.print(f"Removing node [cyan]{node_id}[/cyan] from cluster...")
    try:
        with httpx.Client(base_url=f"http://{coordinator_host}:{coordinator_port}") as client:
            resp = client.post(f"/v1/cluster/nodes/{node_id}/drain")
            if resp.status_code == 200:
                console.print(f"[green]Node {node_id} drained[/green]")
            else:
                console.print(f"[yellow]Drain returned: {resp.status_code}[/yellow]")
    except httpx.ConnectError:
        console.print(f"[red]Could not connect to {coordinator_host}:{coordinator_port}[/red]")


def _cluster_list_nodes(coordinator_host: str, coordinator_port: int):
    """List registered worker nodes (GET /admin/v1/nodes)."""
    try:
        with _get_client(coordinator_host, coordinator_port) as client:
            resp = client.get("/admin/v1/nodes", headers=_admin_headers())
            if resp.status_code in (401, 403):
                console.print(
                    "[red]Unauthorized:[/red] /admin/v1/nodes requires an admin API key. "
                    "Set the [bold]DISTLLM_API_KEY[/bold] environment variable and retry."
                )
                return
            resp.raise_for_status()
            data = resp.json()

        nodes = data.get("nodes", [])
        if not nodes:
            console.print("[yellow]No nodes registered[/yellow]")
            console.print("Workers join with: distllm cluster join --coordinator <host>:<port>")
            return

        table = Table(title=f"Cluster Nodes ({data.get('total_nodes', len(nodes))} total)")
        table.add_column("Node ID", style="cyan")
        table.add_column("Endpoint", style="white")
        table.add_column("State", style="green")
        table.add_column("Layers", style="blue")
        table.add_column("GPU", style="dim")
        table.add_column("VRAM Free", style="magenta")

        for node in nodes:
            state = "draining" if node.get("draining") else node.get("state") or (
                "healthy" if node.get("healthy") else "unhealthy"
            )
            endpoint = f"{node.get('host', '?')}:{node.get('port', '?')}"
            gpu = node.get("gpu_name") or "unknown"
            table.add_row(
                node.get("node_id", "?"),
                endpoint,
                state,
                f"{node.get('start_layer', '?')}-{node.get('end_layer', '?')}",
                gpu,
                _fmt_gib(node.get("gpu_memory_free", 0)),
            )
        console.print(table)

        summary = (
            f"[bold]{data.get('total_nodes', len(nodes))}[/bold] nodes · "
            f"[green]{data.get('healthy_count', '?')} healthy[/green] · "
            f"[yellow]{data.get('draining_count', 0)} draining[/yellow]"
        )
        console.print(summary)
    except httpx.ConnectError:
        _connect_error(coordinator_host, coordinator_port)
    except httpx.TransportError as e:
        console.print(
            f"[red]Error:[/red] Could not reach coordinator at "
            f"{coordinator_host}:{coordinator_port} ({type(e).__name__}). "
            "Note: this command talks to the REST API port (default 8000), not the gRPC port."
        )
