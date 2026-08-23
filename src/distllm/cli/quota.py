"""CLI commands for usage metering and quota management.

Functions are imported by :mod:`distllm.cli.main` which registers them
as Typer commands on the ``config quota`` sub-group.
"""

import json
import os
import time

from rich.console import Console
from rich.table import Table


def _get_meter():
    from distllm.core.usage_meter import UsageMeter
    db = os.environ.get("DISTLLM_USAGE_DB", "")
    return UsageMeter(storage_path=db)


def quota_set(
    tenant_id: str,
    max_tokens_per_day: int = 0,
    max_requests_per_minute: int = 0,
    max_tokens_per_request: int = 0,
    max_concurrent: int = 0,
    cost_budget: float = 0.0,
    overage: bool = False,
) -> None:
    """Set usage quota for a tenant."""
    from distllm.core.usage_meter import QuotaLimit

    console = Console()
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


def quota_show(
    tenant_id: str,
) -> None:
    """Show current quota and usage for a tenant."""
    console = Console()
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
        table.add_row("Today's Tokens", str(usage.daily_tokens.get(
            __import__("datetime").datetime.now().strftime("%Y-%m-%d"), 0
        )))

    console.print(table)


def quota_list() -> None:
    """List all tenants with usage data."""
    console = Console()
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

    today = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
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


def quota_invoice(
    tenant_id: str,
) -> None:
    """Generate an invoice for a tenant's current billing period."""
    console = Console()
    meter = _get_meter()

    from rich.panel import Panel
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


def quota_report(
    tenant_id: str = "",
    days: int = 30,
    json_output: bool = False,
) -> None:
    """Generate a detailed usage report with per-model breakdown."""
    console = Console()
    meter = _get_meter()
    period_start = time.time() - days * 86400

    report = meter.generate_report(
        tenant_id=tenant_id or None,
        period_start=period_start,
    )

    if json_output:
        console.print(json.dumps(report, indent=2, default=str))
        return

    from rich.panel import Panel
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


def quota_export(
    filepath: str,
    tenant_id: str = "",
    days: int = 30,
) -> None:
    """Export usage records as CSV."""
    console = Console()
    meter = _get_meter()
    period_start = time.time() - days * 86400

    result = meter.export_csv(
        filepath=filepath,
        tenant_id=tenant_id or None,
        period_start=period_start,
    )
    console.print(f"[green]Exported usage to:[/] {result}")


def quota_import(
    filepath: str,
) -> None:
    """Bulk import quotas from a JSON file.

    File format: ``{"quotas": [{"tenant_id": "...", "max_tokens_per_day": 100000, ...}]}``
    """
    console = Console()
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
