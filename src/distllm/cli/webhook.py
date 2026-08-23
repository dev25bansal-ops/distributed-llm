"""CLI commands for webhook management.

Functions are imported by :mod:`distllm.cli.main` which registers them
as Typer commands on the ``config webhook`` sub-group.
"""

from rich.console import Console
from rich.table import Table


def webhook_register(
    url: str,
    events: list[str] | None = None,
    secret: str = "",
    label: str = "",
) -> None:
    """Register a new webhook endpoint."""
    from distllm.core.webhook_manager import WebhookManager

    console = Console()
    mgr = WebhookManager()
    ok = mgr.register(url=url, events=events or None, secret=secret, label=label)
    if ok:
        console.print(f"[green]Webhook registered:[/] {url}")
    else:
        console.print(f"[red]Failed to register webhook:[/] {url}")


def webhook_list() -> None:
    """List registered webhook endpoints."""
    from distllm.core.webhook_manager import WebhookManager

    console = Console()
    mgr = WebhookManager()
    targets = mgr.list_targets()

    if not targets:
        console.print("[yellow]No webhook targets registered[/]")
        return

    table = Table(title="Webhook Targets")
    table.add_column("URL", style="cyan")
    table.add_column("Events", style="green")
    table.add_column("Status")
    table.add_column("Success Rate")
    table.add_column("Label")

    for t in targets:
        status = "[green]Active[/]" if t.active else "[red]Inactive[/]"
        rate = f"{mgr.success_rate():.0%}"
        events_str = f"{len(t.events)} event(s)"
        table.add_row(t.url, events_str, status, rate, t.label or "")
    console.print(table)


def webhook_unregister(
    url: str,
) -> None:
    """Unregister a webhook endpoint."""
    from distllm.core.webhook_manager import WebhookManager

    console = Console()
    mgr = WebhookManager()
    if mgr.unregister(url):
        console.print(f"[green]Unregistered:[/] {url}")
    else:
        console.print(f"[red]Not found:[/] {url}")


def webhook_test(
    url: str,
    event: str = "test.ping",
) -> None:
    """Send a test webhook event."""
    from distllm.core.webhook_manager import WebhookManager, WebhookEvent

    console = Console()
    mgr = WebhookManager()
    mgr.register(url=url, events=["*"])
    try:
        event_enum = WebhookEvent(event)
    except ValueError:
        event_enum = WebhookEvent.NODE_JOINED
    mgr.dispatch(event_enum, {"test": True, "message": "Test webhook from CLI"})
    console.print(f"[green]Test {event} sent to:[/] {url}")
