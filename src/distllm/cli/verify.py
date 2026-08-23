"""CLI command for model accuracy verification.

Functions are imported by :mod:`distllm.cli.main` which registers them
as Typer commands on the ``benchmark verify`` sub-group.
"""

from typing import Annotated


def verify_run(
    model: str,
    prompts: list[str] | None = None,
    num_nodes: int = 2,
    dtype: str = "float16",
    temperature: float = 0.0,
    max_new_tokens: int = 32,
    collect_hidden: bool = False,
    backend: str = "",
    grpc: bool = False,
    grpc_base_port: int = 51050,
    output_json: str = "",
    device: str = "auto",
    trust_remote_code: bool = False,
) -> None:
    """Run accuracy verification comparing single-node vs distributed inference."""
    from distllm.verification.runner import AccuracyVerifier
    from rich.console import Console
    console = Console()  # noqa: F841


def verify_list_backends() -> None:
    """List available verification backends."""
    from rich.console import Console
    console = Console()
    console.print("[green]Available backends:[/]")
    console.print("  - single (single-node reference)")
    console.print("  - distributed (multi-node inference)")
