"""CLI commands for usage metering and quota management.

Owns the ``quota_app`` Typer group registered by :mod:`distllm.cli.main`
on the ``config quota`` sub-group (``distllm config quota ...``).
"""

import datetime
import json
import os
import time

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from distllm.core.usage_meter import UsageMeter

quota_app = typer.Typer(help="Usage quotas and billing")
console = Console()


def _get_meter() -> UsageMeter:
    db = os.environ.get("DISTLLM_USAGE_DB", "")
    return UsageMeter(storage_path=db)


def _today() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d")


@quota_app.command("set")
def quota_set(
    tenant_id: str = typer.Argument(..., help="Tenant or team ID"),
    max_tokens_per_day: int = typer.Option(0, "--tokens-per-day", help="Max tokens per day"),
    max_requests_per_minute: int = typer.Option(0, "--rpm", help="Max requests per minute"),
    max_tokens_per_request: int = typer.Option(0, "--tokens-per-request", help="Max tokens per request"),
    max_concurrent: int = typer.Option(0, "--concurrent", help="Max concurrent requests"),
    cost_budget: float = typer.Option(0.0, "--budget", "-b", help="Monthly cost budget"),
    overage: bool = typer.Option(False, "--overage", help="Allow overage"),
) -> None:
    """Set usage quota for a tenant."""
    from distllm.core.usage_meter import QuotaLimit

    meter = _get_meter()
    quota = QuotaLimit(
        tenant_id=tenant_id,
        max_tokens_per_day=max_tokens_per_day,
        max_requests_per_minute=max_requests_per_minute,
        max_tokens_per_request=max_tokens_per_request,
        max_concurrent_requests=max_concurrent,
        cost_budget_per_month=cost_budget,
        overage_allowed=overage,
    )
    meter.set_quota(tenant_id, quota)
    console.print(f"[green]Quota set for tenant:[/] {tenant_id}")


@quota_app.command("show")
def quota_show(
    tenant_id: str = typer.Argument(..., help="Tenant or team ID"),
) -> None:
    """Show current quota and usage for a tenant."""
    meter = _get_meter()
    quota = meter.get_quota(tenant_id)
    usage = meter.tenant_usage(tenant_id)

    if not quota and not usage:
        console.print(f"[yellow]No data for tenant:[/] {tenant_id}")
        return

    table = Table(title=f"Tenant: {tenant_id}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")

    if quota:
        table.add_row("Max Tokens/Day", str(quota.max_tokens_per_day) if quota.max_tokens_per_day else "Unlimited")
        table.add_row("Max RPM", str(quota.max_requests_per_minute) if quota.max_requests_per_minute else "Unlimited")
        table.add_row("Max Tokens/Request", str(quota.max_tokens_per_request) if quota.max_tokens_per_request else "Unlimited")
        table.add_row("Max Concurrent", str(quota.max_concurrent_requests) if quota.max_concurrent_requests else "Unlimited")
        table.add_row("Monthly Budget", f"${quota.cost_budget_per_month:.2f}" if quota.cost_budget_per_month else "Unlimited")
        table.add_row("Overage Allowed", str(quota.overage_allowed))

    if usage:
        table.add_row("Total Requests", str(usage.total_requests))
        table.add_row("Total Input Tokens", str(usage.total_input_tokens))
        table.add_row("Total Output Tokens", str(usage.total_output_tokens))
        table.add_row("Total Cost", f"${usage.total_cost:.4f}")
        table.add_row("Today's Tokens", str(usage.daily_tokens.get(_today(), 0)))

    console.print(table)


@quota_app.command("list")
def quota_list() -> None:
    """List all tenants with usage data."""
    meter = _get_meter()
    tenants = meter.all_tenants()

    if not tenants:
        console.print("[yellow]No tenant usage data[/]")
        return

    table = Table(title="Tenant Usage Summary")
    table.add_column("Tenant ID", style="cyan")
    table.add_column("Requests")
    table.add_column("Input Tokens")
    table.add_column("Output Tokens")
    table.add_column("Cost")
    table.add_column("Today Tokens")

    today = _today()
    for t in tenants:
        table.add_row(
            t.tenant_id,
            str(t.total_requests),
            str(t.total_input_tokens),
            str(t.total_output_tokens),
            f"${t.total_cost:.4f}",
            str(t.daily_tokens.get(today, 0)),
        )
    console.print(table)


@quota_app.command("invoice")
def quota_invoice(
    tenant_id: str = typer.Argument(..., help="Tenant or team ID"),
) -> None:
    """Generate an invoice for a tenant's current billing period."""
    meter = _get_meter()
    invoice = meter.generate_invoice(tenant_id)
    console.print(Panel(
        f"[bold]Invoice for:[/] {tenant_id}\n"
        f"[bold]Period:[/] {invoice['period_start']:.0f} - {invoice['period_end']:.0f}\n"
        f"[bold]Requests:[/] {invoice['total_requests']}\n"
        f"[bold]Input Tokens:[/] {invoice['total_input_tokens']:,}\n"
        f"[bold]Output Tokens:[/] {invoice['total_output_tokens']:,}\n"
        f"[bold]Base Cost:[/] ${invoice['total_cost']:.4f}\n"
        f"[bold]Overage:[/] ${invoice['overage_cost']:.4f}\n"
        f"[bold]Total:[/] ${invoice['grand_total']:.4f}",
        title="[bold cyan]Usage Invoice[/]",
    ))


@quota_app.command("report")
def quota_report(
    tenant_id: str = typer.Argument("", help="Tenant ID (omit for all)"),
    days: int = typer.Option(30, "--days", "-d", help="Report period in days"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Generate a detailed usage report with per-model breakdown."""
    meter = _get_meter()
    period_start = time.time() - days * 86400

    report = meter.generate_report(
        tenant_id=tenant_id or None,
        period_start=period_start,
    )

    if json_output:
        console.print(json.dumps(report, indent=2, default=str))
        return

    console.print(Panel(
        f"[bold]Report for:[/] {report['tenant_id']}\n"
        f"[bold]Period:[/] {days} day(s)\n"
        f"[bold]Total Requests:[/] {report['total_requests']}\n"
        f"[bold]Total Tokens:[/] {report['total_tokens']:,} "
        f"(in: {report['total_input_tokens']:,}, out: {report['total_output_tokens']:,})\n"
        f"[bold]Total Cost:[/] ${report['total_cost']:.4f}",
        title="[bold cyan]Usage Report[/]",
    ))

    models = report.get("models", {})
    if models:
        mt = Table(title="Per-Model Breakdown")
        mt.add_column("Model", style="cyan")
        mt.add_column("Requests")
        mt.add_column("Input Tokens")
        mt.add_column("Output Tokens")
        mt.add_column("Cost")
        for mname, mdata in sorted(models.items()):
            mt.add_row(
                mname or "unknown",
                str(mdata["requests"]),
                str(mdata["input_tokens"]),
                str(mdata["output_tokens"]),
                f"${mdata['cost']:.4f}",
            )
        console.print(mt)


@quota_app.command("export")
def quota_export(
    filepath: str = typer.Argument(..., help="Output CSV file path"),
    tenant_id: str = typer.Option("", "--tenant", "-t", help="Filter by tenant"),
    days: int = typer.Option(30, "--days", "-d", help="Export period in days"),
) -> None:
    """Export usage records as CSV."""
    meter = _get_meter()
    period_start = time.time() - days * 86400

    result = meter.export_csv(
        filepath=filepath,
        tenant_id=tenant_id or None,
        period_start=period_start,
    )
    console.print(f"[green]Exported usage to:[/] {result}")


@quota_app.command("import")
def quota_import(
    filepath: str = typer.Argument(..., help="JSON file with quota definitions"),
) -> None:
    """Bulk import quotas from a JSON file.

    File format: ``{"quotas": [{"tenant_id": "...", "max_tokens_per_day": 100000, ...}]}``
    """
    meter = _get_meter()

    try:
        with open(filepath) as f:
            data = json.load(f)
    except Exception as e:
        console.print(f"[red]Failed to load file:[/] {e}")
        return

    quotas = data if isinstance(data, list) else data.get("quotas", data.get("keys", []))
    if isinstance(quotas, dict):
        quotas = [quotas]

    count = meter.import_quotas(quotas)
    console.print(f"[green]Imported {count} quota(s)[/]")
