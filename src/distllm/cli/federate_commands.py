"""Federated fine-tuning commands for DistLLM CLI."""

import typer
from loguru import logger
from rich.console import Console

console = Console()


def federate_train(
    model: str = typer.Option(..., "--model", "-m", help="Base model name"),
    adapter: str = typer.Option(..., "--adapter", "-a", help="Adapter ID to fine-tune"),
    data: str = typer.Option(..., "--data", "-d", help="Path to local training data (JSONL)"),
    epochs: int = typer.Option(3, "--epochs", help="Training epochs per round"),
    lr: float = typer.Option(2e-4, "--lr", help="Learning rate"),
    rounds: int = typer.Option(1, "--rounds", help="Number of federated rounds"),
    host: str = typer.Option("localhost", "--host", help="Coordinator host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """Run federated fine-tuning: train LoRA locally, submit to coordinator for merging."""
    import httpx

    console.print("\n[bold blue]Federated Fine-Tuning[/bold blue]")
    console.print(f"Model: {model}")
    console.print(f"Adapter: {adapter}")
    console.print(f"Data: {data}")
    console.print(f"Epochs: {epochs}, LR: {lr}, Rounds: {rounds}")
    console.print()

    base_url = f"http://{host}:{port}"

    for round_num in range(1, rounds + 1):
        console.print(f"[bold]Round {round_num}/{rounds}[/bold]")

        # 1. Start local training
        console.print("  Training locally...")
        try:
            from distllm.models.adapter import AdapterManager
            from distllm.models.partitioner import ModelPartitioner

            # Load model and adapter
            partitioner = ModelPartitioner(model_name=model)
            partitioner.load_full_model()

            mgr = AdapterManager()
            mgr.set_base_model(partitioner.full_model, partitioner.tokenizer)
            mgr.load_adapter(adapter, adapter)

            # Train
            result = mgr.start_federated_training(
                adapter_id=adapter,
                local_data_path=data,
                epochs=epochs,
                learning_rate=lr,
            )

            if "error" in result:
                console.print(f"  [red]Training failed:[/red] {result['error']}")
                raise typer.Exit(1)

            console.print(f"  Loss: {result['avg_loss']:.4f}, Steps: {result['steps']}")

        except Exception as e:
            console.print(f"  [red]Error:[/red] {e}")
            raise typer.Exit(1)

        # 2. Submit to coordinator for merging
        console.print("  Submitting to coordinator...")
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"{base_url}/v1/federated/rounds/submit",
                    json={
                        "node_id": f"cli-{adapter}",
                        "adapter_path": f"/tmp/distllm-federated/{adapter}.pt",
                        "loss": result["avg_loss"],
                        "dataset_size": result["steps"],
                    },
                )
                if resp.status_code == 200:
                    console.print("  [green]Submitted successfully[/green]")
                else:
                    console.print(f"  [yellow]Submission returned {resp.status_code}[/yellow]")
        except httpx.ConnectError:
            console.print("  [yellow]Coordinator not reachable (local-only mode)[/yellow]")

        # 3. Trigger merge if coordinator available
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(f"{base_url}/v1/federated/rounds/merge")
                if resp.status_code == 200:
                    data = resp.json()
                    console.print(f"  [green]Merged:[/green] {data.get('path', 'OK')}")
        except Exception:
            logger.debug("Failed to trigger federated merge")
            pass

    console.print("\n[green]Federated training complete[/green]")


def federate_status(
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """Show federated training status."""
    import httpx

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"http://{host}:{port}/v1/federated/stats")
            resp.raise_for_status()
            stats = resp.json()

            console.print("\n[bold]Federated Training Status[/bold]")
            console.print(f"  Total rounds: {stats.get('total_rounds', 0)}")
            console.print(f"  Registered nodes: {stats.get('registered_nodes', 0)}")
            console.print(f"  Active nodes: {stats.get('active_nodes', 0)}")
            console.print(f"  Total versions: {stats.get('total_versions', 0)}")
            console.print(f"  Strategy: {stats.get('merge_strategy', 'N/A')}")
            if stats.get('current_round'):
                console.print(f"  Current round: {stats['current_round']} ({stats.get('current_round_status', '')})")
            if stats.get('avg_loss_last_round') is not None:
                console.print(f"  Avg loss (last round): {stats['avg_loss_last_round']:.4f}")
    except httpx.ConnectError:
        console.print("[red]Could not connect to coordinator[/red]")
