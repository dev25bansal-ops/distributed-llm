"""CLI command for model accuracy verification."""

from typing import Annotated
import typer

verify_app = typer.Typer(help="Verify distributed inference accuracy against single-node reference")


@verify_app.command("run")
def verify_run(
    model: str = typer.Option(..., "--model", "-m", help="Model name or path"),
    prompts: list[str] = typer.Option(
        ["The capital of France is", "In the beginning", "The meaning of life is"],
        "--prompt", "-p",
        help="Prompt(s) to verify (repeat for multiple)",
    ),
    num_nodes: int = typer.Option(2, "--nodes", "-n", help="Number of distributed nodes"),
    dtype: str = typer.Option("float16", "--dtype", help="Data type (float16, float32, bfloat16)"),
    temperature: float = typer.Option(0.0, "--temperature", "-t", help="Sampling temperature (0 = greedy)"),
    max_new_tokens: int = typer.Option(32, "--max-tokens", help="Max tokens to generate per prompt"),
    collect_hidden: bool = typer.Option(False, "--collect-hidden", help="Collect intermediate hidden states"),
    backend: str = typer.Option("", "--backend", "-b", help="Preferred inference backend (auto-detect if empty)"),
    grpc: bool = typer.Option(False, "--grpc", help="Use real gRPC workers instead of in-process simulation"),
    grpc_base_port: int = typer.Option(51050, "--grpc-port", help="Base port for gRPC workers"),
    output_json: str = typer.Option("", "--output-json", "-o", help="Path to write JSON report"),
    device: str = typer.Option("auto", "--device", help="Device (auto, cuda, cpu)"),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code", help="Trust remote code in HuggingFace models"),
):
    """Run accuracy verification comparing single-node vs distributed inference."""
    from distllm.verification.runner import AccuracyVerifier
    from rich.console import Console
    console = Console()

    verifier = AccuracyVerifier(
        model_name=model,
        device=device,
        dtype=dtype,
        num_nodes=num_nodes,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        preferred_backend=backend,
        grpc_mode=grpc,
        grpc_base_port=grpc_base_port,
        trust_remote_code=trust_remote_code,
    )

    report = verifier.verify(
        prompts=prompts,
        collect_hidden_states=collect_hidden,
    )
    report.duration_ms = report.created_at  # approximate

    # Human-readable output
    report.print_human_readable()

    # JSON output
    if output_json:
        import json as _json
        with open(output_json, "w") as f:
            f.write(report.to_json(indent=2))
        console.print(f"\n[green]JSON report written to:[/] {output_json}")

    # Exit with status
    summary = report.summary()
    if summary.get("failed", 0) > 0:
        raise typer.Exit(1)


@verify_app.command("list-backends")
def verify_list_backends():
    """List available inference backends for verification."""
    from distllm.backends.registry import list_available_backends
    from rich.console import Console
    from rich.table import Table
    console = Console()

    backends = list_available_backends()
    if not backends:
        console.print("[yellow]No inference backends available[/]")
        return

    table = Table(title="Available Backends")
    table.add_column("Name", style="cyan")
    table.add_column("Display Name", style="green")
    table.add_column("Version")
    table.add_column("Description")

    for b in backends:
        cls = b.adapter_class
        table.add_row(
            b.name,
            cls.display_name(),
            cls.version(),
            cls.description()[:60],
        )
    console.print(table)
