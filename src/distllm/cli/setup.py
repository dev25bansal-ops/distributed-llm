"""Interactive setup wizard for DistLLM."""

import yaml
from pathlib import Path
from rich.console import Console


def run_setup(config_path: str, console: Console):
    """Run interactive setup wizard."""
    console.print("\n[bold blue]DistLLM Setup Wizard[/bold blue]\n")

    # Step 1: Model selection
    console.print("[bold]Step 1: Model Selection[/bold]")
    model = console.input("Enter model name or path (e.g., roneneldan/TinyStories-1M): ").strip()
    if not model:
        model = "roneneldan/TinyStories-1M"
        console.print(f"Using default: {model}")

    # Step 2: Mode selection
    console.print("\n[bold]Step 2: Deployment Mode[/bold]")
    mode = console.input("Run in local mode or distributed mode? [local/distributed]: ").strip().lower()
    is_local = mode in ("local", "l")

    # Step 3: Data type
    console.print("\n[bold]Step 3: Data Type[/bold]")
    dtype = console.input("Choose dtype [float16/float32/bfloat16]: ").strip().lower()
    if dtype not in ("float16", "float32", "bfloat16"):
        dtype = "float16"

    # Build config
    config = {
        "model": {
            "name": model,
            "dtype": dtype,
        },
        "coordinator": {
            "host": "localhost",
            "port": 50050,
            "api_port": 8000,
        },
    }

    if not is_local:
        # Step 4: Node configuration
        console.print("\n[bold]Step 4: Node Configuration[/bold]")
        num_nodes = int(console.input("Number of worker nodes: ").strip() or "2")

        nodes = []
        for i in range(num_nodes):
            console.print(f"\n[bold]Node {i}[/bold]")
            host = console.input(f"  Host (default: node_{i}): ").strip() or f"node_{i}"
            port = int(console.input(f"  Port (default: {50051 + i}): ").strip() or str(50051 + i))
            start_layer = int(console.input("  Start layer: ").strip())
            end_layer = int(console.input("  End layer: ").strip())

            nodes.append({
                "node_id": f"node_{i}",
                "host": host,
                "port": port,
                "start_layer": start_layer,
                "end_layer": end_layer,
                "device": "cuda",
            })

        config["nodes"] = nodes

    # Step 5: Generation settings
    console.print("\n[bold]Step 5: Generation Settings[/bold]")
    max_tokens = int(console.input("Max new tokens (default: 256): ").strip() or "256")
    temperature = float(console.input("Temperature (default: 0.7): ").strip() or "0.7")

    config["generation"] = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
    }

    # Step 6: Network settings
    config["network"] = {
        "grpc_timeout": 30,
        "max_retries": 3,
        "retry_delay": 1.0,
    }

    # Step 7: TLS (optional)
    console.print("\n[bold]Step 6: TLS Configuration[/bold]")
    enable_tls = console.input("Enable TLS? [y/N]: ").strip().lower() in ("y", "yes")
    config["tls"] = {"enabled": enable_tls}
    if enable_tls:
        config["tls"]["cert_dir"] = "certs"

    # Write config
    config_path = Path(config_path)
    if config_path.exists():
        confirm = console.input(f"[yellow]Config file {config_path} already exists. Overwrite? [y/N]: [/yellow]").strip().lower()
        if confirm not in ("y", "yes"):
            console.print("[red]Aborted — config not written.[/red]")
            return

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    console.print(f"\n[green]✓[/green] Config written to {config_path}")
    console.print(f"\nTo start: [bold]distllm run --model {model} {'--local' if is_local else ''}[/bold]")
