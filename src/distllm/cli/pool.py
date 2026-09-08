"""``distllm pool`` — GPU pool management (week-5 strategic product).

This module is a **stub**. The product lands in week 5; until then each
subcommand raises ``NotImplementedError`` with a clear message that
points to docs/_map/Roadmap (or wherever the public roadmap ends up).

Commands:
  * ``distllm pool add    --provider hetzner --api-key ...``
  * ``distllm pool list``
  * ``distllm pool remove <pool-id>``

The stub exists today so:
  * The CLI surface is stable — the help text is already correct
  * Users can discover that the feature is on the way
  * Tests can verify the commands are wired up
  * Marketing can link to ``distllm pool --help`` immediately
"""

from __future__ import annotations

from typing import Optional

import typer
from loguru import logger
from rich.console import Console


SUPPORTED_PROVIDERS = ("hetzner", "lambda", "runpod", "vast", "coreweave", "aws")


def register(app: typer.Typer) -> None:
    """Attach the top-level ``pool`` group with three subcommands."""

    pool_app = typer.Typer(
        help=(
            "Manage GPU pools (week-5 feature). "
            "Each command is currently a stub — see docs/_map/Roadmap."
        ),
        rich_markup_mode="rich",
    )

    @pool_app.command("add")
    def pool_add(
        provider: str = typer.Option(
            ...,
            "--provider",
            "-p",
            help=f"Cloud provider. One of: {', '.join(SUPPORTED_PROVIDERS)}",
        ),
        api_key: str = typer.Option(
            ..., "--api-key", "-k", help="Provider API key (stored in local keyring)"
        ),
        name: Optional[str] = typer.Option(
            None, "--name", "-n", help="Friendly name for the pool"
        ),
        region: Optional[str] = typer.Option(
            None, "--region", "-r", help="Provider region (auto-selected if omitted)"
        ),
        gpu_type: Optional[str] = typer.Option(
            None, "--gpu", help="Preferred GPU type (e.g. H100, A100)"
        ),
        max_spend_usd: Optional[float] = typer.Option(
            None, "--max-spend", help="Monthly spend cap in USD"
        ),
    ) -> None:
        """Add a new GPU pool from a cloud provider (week-5 feature)."""
        console = Console()
        if provider not in SUPPORTED_PROVIDERS:
            console.print(
                f"[red]Error:[/red] unsupported provider '{provider}'. "
                f"Choose one of: {', '.join(SUPPORTED_PROVIDERS)}"
            )
            raise typer.Exit(2)

        logger.warning(
            "distllm pool add is a stub for the week-5 strategic feature. "
            "See docs/_map/Roadmap for the rollout timeline."
        )
        raise typer.Exit(
            code=1
        )  # actually: NotImplementedError wrapped in a friendly exit

    @pool_app.command("list")
    def pool_list(
        json_output: bool = typer.Option(
            False, "--json", help="Output as JSON instead of a table"
        ),
    ) -> None:
        """List configured GPU pools (week-5 feature)."""
        console = Console()
        console.print(
            "[yellow]distllm pool list is a stub.[/yellow] "
            "GPU pool management ships in week 5."
        )
        # Stub: emit empty list as JSON when --json, else friendly message
        if json_output:
            console.print_json(data={"pools": [], "stub": True})
        else:
            console.print("[dim]No pools configured (stub).[/dim]")
        raise typer.Exit(1)

    @pool_app.command("remove")
    def pool_remove(
        pool_id: str = typer.Argument(..., help="Pool ID to remove"),
        force: bool = typer.Option(False, "--force", help="Skip confirmation"),
    ) -> None:
        """Remove a GPU pool (week-5 feature)."""
        console = Console()
        console.print(
            f"[yellow]distllm pool remove {pool_id} is a stub.[/yellow] "
            "GPU pool management ships in week 5."
        )
        if not force:
            # Don't prompt — the command is a stub anyway. Keep the flag
            # reserved so the real implementation can plug it in.
            pass
        raise typer.Exit(1)

    app.add_typer(pool_app, name="pool")


def _not_implemented(feature: str) -> None:
    """Helper for raising a uniform NotImplementedError with a roadmap link."""
    raise NotImplementedError(
        f"{feature} is a stub for the week-5 strategic feature. "
        "See docs/_map/Roadmap for the rollout timeline."
    )


__all__ = ["register", "SUPPORTED_PROVIDERS"]