"""``distllm pool`` — rent GPUs from spot marketplaces.

Wraps the spot orchestrator (``distllm.cloud.spot_orchestrator``), which talks
to RunPod, Vast and Salad over their real HTTP APIs. This module was a stub
that printed "week-5 feature"; the backend it needs already existed, so the
commands are wired to it now.

Credentials come from the environment (``RUNPOD_API_KEY``, ``VAST_API_KEY``,
``SALAD_API_KEY``). A provider with no key is skipped rather than faked --
``pool list`` reports which providers it could actually reach.
"""

from __future__ import annotations

import os

import typer
from rich.console import Console
from rich.table import Table

SUPPORTED_PROVIDERS = ("runpod", "vast", "salad")

_ENV_KEYS = {
    "runpod": "RUNPOD_API_KEY",
    "vast": "VAST_API_KEY",
    "salad": "SALAD_API_KEY",
}


def _missing_keys(providers: list[str]) -> list[str]:
    """Providers that would be searched but have no API key set."""
    return [p for p in providers if not os.environ.get(_ENV_KEYS.get(p, ""))]


def register(app: typer.Typer) -> None:
    """Attach the top-level ``pool`` group with its subcommands."""

    pool_app = typer.Typer(
        help="Rent GPUs from spot marketplaces (RunPod, Vast, Salad)",
        rich_markup_mode="rich",
    )
    console = Console()

    @pool_app.command("list")
    def pool_list(
        gpu: str = typer.Option("", "--gpu", "-g", help="GPU name filter, e.g. '4090'"),
        max_price: float = typer.Option(
            10.0, "--max-price", help="USD per GPU-hour ceiling"
        ),
        region: str = typer.Option("", "--region", help="Region filter"),
        provider: str = typer.Option(
            "", "--provider", help="Limit to one provider (runpod, vast, salad)"
        ),
        limit: int = typer.Option(15, "--limit", "-n", help="Max rows to show"),
    ):
        """List rentable GPUs across marketplaces, cheapest first."""
        from distllm.cloud.spot_orchestrator import SpotOrchestrator

        providers = (
            [provider.lower()]
            if provider and provider.lower() in SUPPORTED_PROVIDERS
            else list(SUPPORTED_PROVIDERS)
        )
        missing = _missing_keys(providers)
        if len(missing) == len(providers):
            console.print(
                "[yellow]No marketplace credentials found.[/yellow] Set at least one of:\n  "
                + "\n  ".join(f"{_ENV_KEYS[p]}" for p in providers)
            )
            raise typer.Exit(1)
        if missing:
            console.print(
                f"[dim]Skipping {', '.join(missing)} (no API key set)[/dim]"
            )

        orchestrator = SpotOrchestrator()
        try:
            # An empty gpu filter means "anything"; the market matches by
            # substring, so this passes through as no constraint.
            offers = orchestrator.find_cheapest(
                gpu_type=gpu or "",
                max_price=max_price,
                region=region or None,
                providers=[p for p in providers if p not in missing],
            )
        except Exception as exc:
            console.print(f"[red]Marketplace query failed:[/red] {exc}")
            raise typer.Exit(1) from exc

        if not offers:
            console.print(
                f"[yellow]No offers under ${max_price:.2f}/hr"
                + (f" matching '{gpu}'" if gpu else "")
                + ".[/yellow]"
            )
            return

        table = Table(title=f"GPU offers (cheapest first, ≤ ${max_price:.2f}/hr)")
        table.add_column("GPU", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("$/hr", justify="right", style="green")
        table.add_column("Region")
        table.add_column("Provider")
        table.add_column("Instance")
        for inst in offers[:limit]:
            table.add_row(
                getattr(inst, "gpu_type", "?"),
                str(getattr(inst, "gpu_count", 1)),
                f"{inst.price_per_hour:.3f}",
                getattr(inst, "region", "?"),
                getattr(inst, "provider", "?"),
                getattr(inst, "instance_id", "?")[:18],
            )
        console.print(table)
        console.print(
            f"\n[dim]{len(offers)} offers found. Rent with: "
            f"distllm pool add --gpu {gpu or '<name>'} --max-price {max_price}[/dim]"
        )

    @pool_app.command("add")
    def pool_add(
        gpu: str = typer.Option(..., "--gpu", "-g", help="GPU name to rent"),
        max_price: float = typer.Option(1.0, "--max-price", help="USD per hour ceiling"),
        count: int = typer.Option(1, "--count", "-c", help="How many instances"),
        region: str = typer.Option("", "--region", help="Preferred region"),
        dry_run: bool = typer.Option(
            False, "--dry-run", help="Show what would be rented without renting"
        ),
    ):
        """Rent the cheapest matching GPU instances."""
        from distllm.cloud.spot_orchestrator import SpotOrchestrator

        providers = [p for p in SUPPORTED_PROVIDERS if not any(
            p == m for m in _missing_keys([p])
        )]
        if not providers:
            console.print(
                "[red]No marketplace credentials found.[/red] Set RUNPOD_API_KEY, "
                "VAST_API_KEY or SALAD_API_KEY."
            )
            raise typer.Exit(1)

        orchestrator = SpotOrchestrator()
        offers = orchestrator.find_cheapest(
            gpu_type=gpu, max_price=max_price, region=region or None, providers=providers
        )
        if not offers:
            console.print(f"[red]No '{gpu}' offers under ${max_price:.2f}/hr.[/red]")
            raise typer.Exit(1)

        chosen = offers[:count]
        total = sum(i.price_per_hour for i in chosen)
        console.print(
            f"[green]Cheapest match:[/green] {chosen[0].gpu_type} at "
            f"${chosen[0].price_per_hour:.3f}/hr ({getattr(chosen[0], 'provider', '?')})"
        )
        console.print(f"Renting {len(chosen)} instance(s), ${total:.3f}/hr total")

        if dry_run:
            console.print("[yellow]--dry-run: nothing rented.[/yellow]")
            return

        results = orchestrator.launch_cluster(chosen)
        table = Table(title="Rental results")
        table.add_column("Instance")
        table.add_column("Status")
        table.add_column("Detail")
        for res in results:
            ok = getattr(res, "success", getattr(res, "accepted", False))
            table.add_row(
                str(getattr(res, "instance_id", "?"))[:20],
                "[green]started[/green]" if ok else "[red]failed[/red]",
                str(getattr(res, "message", getattr(res, "reason", "")))[:50],
            )
        console.print(table)

    @pool_app.command("remove")
    def pool_remove(
        instance_id: str = typer.Argument(..., help="Instance id to terminate"),
        provider: str = typer.Option("", "--provider", help="Provider hosting it"),
        yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    ):
        """Terminate a rented instance."""
        if not yes:
            confirmed = typer.confirm(f"Terminate instance {instance_id}?")
            if not confirmed:
                console.print("Cancelled.")
                raise typer.Exit(0)

        if not provider:
            console.print(
                "[red]--provider is required[/red]: instance ids are only unique "
                "within a marketplace."
            )
            raise typer.Exit(1)

        from distllm.cloud.spot_orchestrator import GPUSpotMarket

        try:
            ok = GPUSpotMarket().cancel_bid(provider.lower(), instance_id)
        except Exception as exc:
            console.print(f"[red]Termination failed:[/red] {exc}")
            raise typer.Exit(1) from exc

        if ok:
            console.print(f"[green]Terminated {instance_id}[/green]")
        else:
            console.print(f"[red]Could not terminate {instance_id}[/red]")
            raise typer.Exit(1)

    app.add_typer(pool_app, name="pool")