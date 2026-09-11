"""CLI command for model accuracy verification.

Owns the ``verify_app`` Typer group registered by :mod:`distllm.cli.main`
on the ``benchmark verify`` sub-group (``distllm benchmark verify ...``).
"""

import json
import logging

import typer
from rich.console import Console

from distllm.backends.registry import list_available_backends
from distllm.verification.runner import AccuracyVerifier

logger = logging.getLogger(__name__)

verify_app = typer.Typer(help="Verify distributed inference correctness")
console = Console()


@verify_app.command("run")
def verify_run(
    model: str = typer.Option(..., "--model", "-m", help="Model name or path"),
    prompts: list[str] = typer.Option(
        ["The capital of France is", "In the beginning", "The meaning of life is"],
        "--prompt", "-p",
        help="Prompt(s) to verify (repeat for multiple)",
    ),
    num_nodes: int = typer.Option(2, "--nodes", "-n", help="Number of distributed nodes"),
    dtype: str = typer.Option("float16", "--dtype", help="Data type"),
    temperature: float = typer.Option(0.0, "--temperature", "-t", help="Sampling temperature (0 = greedy)"),
    max_new_tokens: int = typer.Option(32, "--max-tokens", help="Max tokens to generate per prompt"),
    collect_hidden: bool = typer.Option(False, "--collect-hidden", help="Collect intermediate hidden states"),
    backend: str = typer.Option("", "--backend", "-b", help="Preferred inference backend"),
    grpc: bool = typer.Option(False, "--grpc", help="Use real gRPC workers instead of in-process simulation"),
    grpc_base_port: int = typer.Option(51050, "--grpc-port", help="Base port for gRPC workers"),
    output_json: str = typer.Option("", "--output-json", "-o", help="Path to write JSON report"),
    device: str = typer.Option("auto", "--device", help="Device (auto, cuda, cpu)"),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code", help="[SECURITY RISK] Allow loading remote code from HuggingFace models (RCE vector)"),
) -> None:
    """Verify distributed inference accuracy against single-node reference."""
    if trust_remote_code:
        logger.warning(
            "--trust-remote-code enabled: loading arbitrary code from HuggingFace repos (RCE risk)"
        )

    verifier = AccuracyVerifier(
        model_name=model,
        num_nodes=num_nodes,
        dtype=dtype,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        preferred_backend=backend,
        grpc_mode=grpc,
        grpc_base_port=grpc_base_port,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    report = verifier.verify(
        prompts=list(prompts),
        collect_hidden_states=collect_hidden,
    )
    summary = report.summary()

    console.print("[bold]Verification summary:[/]")
    for key in sorted(summary):
        console.print(f"  {key}: {summary[key]}")

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        console.print(f"[green]Report written to:[/] {output_json}")

    if summary.get("failed", 0) > 0:
        raise typer.Exit(1)


@verify_app.command("list-backends")
def verify_list_backends() -> None:
    """List available inference backends."""
    backends = list_available_backends()
    if not backends:
        console.print("[yellow]No backends available[/]")
        return

    console.print("[green]Available backends:[/]")
    for b in backends:
        adapter = b.adapter_class
        console.print(
            f"  - {b.name}: {adapter.display_name()} "
            f"v{adapter.version()} — {adapter.description()}"
        )
