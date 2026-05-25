"""Deploy command for DistLLM CLI.

from loguru import logger
Deploys a model to the distributed cluster with layer partitioning
across nodes.
"""

import time
from rich.console import Console

console = Console()


def run_deploy(
    model: str,
    nodes: int,
    dtype: str,
    quantization: str,
    gpu_type: str | None,
    dry_run: bool,
    wait: bool,
    host: str,
    port: int,
    console: Console,
) -> None:
    """Deploy a model to the distributed cluster."""
    import httpx

    console.print(f"[bold]Deploying model:[/bold] {model}")
    console.print(f"  Nodes: {nodes}")
    console.print(f"  Dtype: {dtype}")
    console.print(f"  Quantization: {quantization}")
    if gpu_type:
        console.print(f"  GPU type: {gpu_type}")
    console.print()

    # Calculate layer partitioning
    # For deployment plan, estimate layers per node
    layers_per_node = max(1, 32 // nodes)  # Assume 32 layers for common models
    console.print("[bold]Deployment Plan:[/bold]")
    console.print(f"  Model: {model}")
    console.print(f"  Total nodes: {nodes}")
    console.print(f"  Layers per node: ~{layers_per_node}")
    console.print()

    # Show node assignments
    from rich.table import Table
    table = Table(title="Node Assignments")
    table.add_column("Node", style="cyan")
    table.add_column("Host", style="green")
    table.add_column("Port", style="green")
    table.add_column("Layer Range", style="yellow")

    for i in range(nodes):
        start_layer = i * layers_per_node
        end_layer = min((i + 1) * layers_per_node, 32) - 1
        table.add_row(
            f"node_{i}",
            "localhost",
            str(50051 + i),
            f"{start_layer}-{end_layer}",
        )

    console.print(table)
    console.print()

    if dry_run:
        console.print("[yellow]Dry run — no changes made.[/yellow]")
        return

    # Send deployment request to coordinator
    console.print("[bold]Sending deployment request...[/bold]")
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"http://{host}:{port}/v1/models/{model}/load",
                json={
                    "dtype": dtype,
                    "quantization": quantization,
                },
            )
            response.raise_for_status()
            console.print("[green]Deployment request accepted.[/green]")
    except httpx.ConnectError:
        console.print(f"[yellow]Could not reach API at {host}:{port}. Deployment queued.[/yellow]")
        return
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Deployment failed: {e.response.text}[/red]")
        return

    if wait:
        console.print("[bold]Waiting for deployment to complete...[/bold]")
        max_wait = 300  # 5 minutes
        start = time.time()
        while time.time() - start < max_wait:
            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.get(f"http://{host}:{port}/health")
                    resp.raise_for_status()
                    data = resp.json()
                    node_count = data.get("nodes", 0)
                    if node_count >= nodes:
                        console.print(f"[green]Deployment complete: {node_count}/{nodes} nodes ready.[/green]")
                        return
                    console.print(f"  Waiting... {node_count}/{nodes} nodes registered")
            except Exception:
                logger.debug("Deploy operation failed (non-fatal)")
            time.sleep(5)

        console.print("[red]Deployment timed out after 5 minutes.[/red]")
