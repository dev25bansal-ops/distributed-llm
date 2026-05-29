"""CLI commands for webhook management."""

import typer
from rich.console import Console
from rich.table import Table

webhook_app = typer.Typer(help="Manage webhook endpoints")


@webhook_app.command("register")
def webhook_register(
    url: str = typer.Argument(..., help="Webhook endpoint URL"),
    events: list[str] = typer.Option([], "--event", "-e", help="Events to subscribe to (all if empty)"),
    secret: str = typer.Option("", "--secret", "-s", help="HMAC signing secret"),
    label: str = typer.Option("", "--label", "-l", help="Human-readable label"),
):
    """Register a new webhook endpoint."""
    from distllm.core.webhook_manager import WebhookManager

    console = Console()
    mgr = WebhookManager()
    ok = mgr.register(url=url, events=events or None, secret=secret, label=label)
    if ok:
        console.print(f"[green]Webhook registered:[/] {url}")
    else:
        console.print(f"[red]Failed to register webhook:[/] {url}")
        raise typer.Exit(1)


@webhook_app.command("list")
def webhook_list():
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


@webhook_app.command("unregister")
def webhook_unregister(
    url: str = typer.Argument(..., help="Webhook URL to remove"),
):
    """Unregister a webhook endpoint."""
    from distllm.core.webhook_manager import WebhookManager

    console = Console()
    mgr = WebhookManager()
    if mgr.unregister(url):
        console.print(f"[green]Unregistered:[/] {url}")
    else:
        console.print(f"[red]Not found:[/] {url}")


@webhook_app.command("test")
def webhook_test(
    url: str = typer.Argument(..., help="Webhook URL to test"),
    event: str = typer.Option("test.ping", "--event", "-e", help="Event type"),
):
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
