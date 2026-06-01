"""Deploy command for DistLLM CLI.

from loguru import logger
Deploys a model to the distributed cluster with layer partitioning
across nodes. Supports one-line HuggingFace deployment.
"""

import time
from rich.console import Console
from rich.panel import Panel

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
    hf_token: str | None = None,
    auto_partition: bool = True,
) -> None:
    """Deploy a model to the distributed cluster.

    Supports one-line HuggingFace deployment:
        distllm deploy --hf meta-llama/Llama-2-7b --nodes 2

    Args:
        model: Model name or HuggingFace model ID.
        nodes: Number of worker nodes.
        dtype: Data type (float16, bfloat16, float32).
        quantization: Quantization method.
        gpu_type: Target GPU type for VRAM estimation.
        dry_run: If True, show plan without deploying.
        wait: If True, wait for deployment to complete.
        host: Coordinator host.
        port: API server port.
        console: Rich console for output.
        hf_token: HuggingFace API token for gated models.
        auto_partition: If True, auto-calculate optimal layer partitioning.
    """
    import httpx

    # One-line HuggingFace deploy: estimate model parameters
    if "/" in model:
        console.print(Panel(
            f"[bold green]HuggingFace Model Detected[/bold green]\n"
            f"Model: {model}\n"
            f"Nodes: {nodes} | Dtype: {dtype} | Quantization: {quantization}",
            title="DistLLM Deploy",
        ))

    console.print(f"[bold]Deploying model:[/bold] {model}")
    console.print(f"  Nodes: {nodes}")
    console.print(f"  Dtype: {dtype}")
    console.print(f"  Quantization: {quantization}")
    if gpu_type:
        console.print(f"  GPU type: {gpu_type}")
    console.print()

    # Auto-estimate layers based on model name
    estimated_layers = _estimate_layers(model)
    estimated_params = _estimate_params(model)

    if auto_partition:
        layers_per_node = max(1, estimated_layers // nodes)
    else:
        layers_per_node = max(1, 32 // nodes)

    # Show deployment plan
    console.print(f"[bold]Model estimate:[/bold] ~{estimated_params}B parameters, ~{estimated_layers} layers")
    console.print()

    from rich.table import Table
    table = Table(title="Deployment Plan")
    table.add_column("Node", style="cyan")
    table.add_column("Host", style="green")
    table.add_column("Port", style="green")
    table.add_column("Layer Range", style="yellow")
    table.add_column("Est. VRAM", style="magenta")

    vram_per_layer = _estimate_vram_per_layer(estimated_params, dtype, quantization)
    for i in range(nodes):
        start_layer = i * layers_per_node
        end_layer = min((i + 1) * layers_per_node, estimated_layers) - 1
        vram_gb = layers_per_node * vram_per_layer
        table.add_row(
            f"node_{i}",
            "localhost",
            str(50051 + i),
            f"{start_layer}-{end_layer}",
            f"~{vram_gb:.1f} GB",
        )

    console.print(table)
    console.print()

    if dry_run:
        console.print("[yellow]Dry run — no changes made.[/yellow]")
        return

    # Build request headers
    headers = {"Content-Type": "application/json"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    # Send deployment request to coordinator
    console.print("[bold]Sending deployment request...[/bold]")
    try:
        with httpx.Client(timeout=30.0, headers=headers) as client:
            response = client.post(
                f"http://{host}:{port}/v1/models/{model}/load",
                json={
                    "dtype": dtype,
                    "quantization": quantization,
                    "num_nodes": nodes,
                    "auto_partition": auto_partition,
                },
            )
            response.raise_for_status()
            console.print("[green]Deployment request accepted.[/green]")
    except httpx.ConnectError:
        console.print(f"[yellow]Could not reach API at {host}:{port}. Starting local mode...[/yellow]")
        _deploy_local(model, dtype, quantization, nodes, console)
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
                        _print_usage_example(host, port, model)
                        return
                    console.print(f"  Waiting... {node_count}/{nodes} nodes registered")
            except Exception:
                logger.debug("Deploy operation failed (non-fatal)")
            time.sleep(5)

        console.print("[red]Deployment timed out after 5 minutes.[/red]")
    else:
        _print_usage_example(host, port, model)


def _estimate_layers(model_name: str) -> int:
    """Estimate number of transformer layers from model name."""
    name = model_name.lower()
    if "70b" in name or "65b" in name:
        return 80
    if "34b" in name:
        return 64
    if "13b" in name or "14b" in name:
        return 40
    if "7b" in name or "8b" in name:
        return 32
    if "3b" in name:
        return 28
    if "1b" in name or "1.5b" in name:
        return 24
    if "0.5b" in name or "350m" in name:
        return 16
    return 32  # Default


def _estimate_params(model_name: str) -> float:
    """Estimate parameter count in billions from model name."""
    name = model_name.lower()
    if "70b" in name:
        return 70.0
    if "65b" in name:
        return 65.0
    if "34b" in name:
        return 34.0
    if "13b" in name:
        return 13.0
    if "7b" in name or "8b" in name:
        return 7.0
    if "3b" in name:
        return 3.0
    if "1.5b" in name:
        return 1.5
    if "1b" in name:
        return 1.0
    if "0.5b" in name or "350m" in name:
        return 0.5
    return 7.0  # Default


def _estimate_vram_per_layer(params_b: float, dtype: str, quantization: str) -> float:
    """Estimate VRAM per layer in GB."""
    # Rough estimate: params_b * 2 bytes (fp16) / num_layers
    bytes_per_param = 2 if dtype == "float16" else 4
    if "4bit" in quantization:
        bytes_per_param = 0.5
    elif "8bit" in quantization:
        bytes_per_param = 1
    total_bytes = params_b * 1e9 * bytes_per_param
    layers = _estimate_layers(f"{params_b}b")
    return (total_bytes / layers) / 1e9


def _deploy_local(model: str, dtype: str, quantization: str, nodes: int, console: Console) -> None:
    """Deploy in local mode without a running coordinator."""
    console.print("[bold]Starting local deployment...[/bold]")
    try:
        from distllm.cli.run import run_inference
        run_inference(
            model=model,
            local=True,
            config="",
            port=8000,
            dtype=dtype,
            max_tokens=256,
            temperature=0.7,
            prompt="",
            console=console,
        )
    except Exception as e:
        console.print(f"[red]Local deployment failed: {e}[/red]")


def _print_usage_example(host: str, port: int, model: str) -> None:
    """Print usage examples after successful deployment."""
    console.print()
    console.print(Panel(
        f"[bold]Model deployed![/bold]\n\n"
        f"[dim]Try it:[/dim]\n"
        f"  curl http://{host}:{port}/v1/chat/completions \\\n"
        f"    -H 'Content-Type: application/json' \\\n"
        f"    -d '{{\"model\": \"{model}\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]}}'\n\n"
        f"[dim]Or use the SDK:[/dim]\n"
        f"  from distllm_sdk import DistLLMClient\n"
        f"  client = DistLLMClient(base_url='http://{host}:{port}')\n"
        f"  response = client.chat_completions('{model}', [{{'role': 'user', 'content': 'Hello!'}}])'",
        title="Quick Start",
    ))
