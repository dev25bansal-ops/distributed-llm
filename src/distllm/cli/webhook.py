"""CLI commands for webhook management.

Owns the ``webhook_app`` Typer group registered by :mod:`distllm.cli.main`
on the ``config webhook`` sub-group (``distllm config webhook ...``).
"""

import typer
from rich.console import Console
from rich.table import Table

from distllm.core.webhook_manager import WebhookEvent, WebhookManager

webhook_app = typer.Typer(help="Manage webhook endpoints")
console = Console()


@webhook_app.command("register")
def webhook_register(
    url: str = typer.Argument(..., help="Webhook URL"),
    events: list[str] = typer.Option([], "--event", "-e", help="Events to subscribe to"),
    secret: str = typer.Option("", "--secret", "-s", help="HMAC signing secret"),
    label: str = typer.Option("", "--label", "-l", help="Human-readable label"),
) -> None:
    """Register a new webhook endpoint."""
    mgr = WebhookManager()
    ok = mgr.register(url=url, events=events or None, secret=secret, label=label)
    if ok:
        console.print(f"[green]Webhook registered:[/] {url}")
    else:
        console.print(f"[red]Failed to register webhook:[/] {url}")
        raise typer.Exit(1)


@webhook_app.command("list")
def webhook_list() -> None:
    """List registered webhook endpoints."""
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
        rate = f"{t.success_rate:.0%}" if hasattr(t, "success_rate") else f"{mgr.success_rate():.0%}"
        events_str = f"{len(t.events)} event(s)"
        table.add_row(t.url, events_str, status, rate, t.label or "")
    console.print(table)


@webhook_app.command("unregister")
def webhook_unregister(
    url: str = typer.Argument(..., help="Webhook URL to remove"),
) -> None:
    """Unregister a webhook endpoint."""
    mgr = WebhookManager()
    if mgr.unregister(url):
        console.print(f"[green]Unregistered:[/] {url}")
    else:
        console.print(f"[red]Not found:[/] {url}")


@webhook_app.command("test")
def webhook_test(
    url: str = typer.Argument(..., help="Webhook URL to test"),
    event: str = typer.Option(
        "node.joined", "--event", "-e",
        help="Event type to send (defaults to a real event so the delivery path is exercised)",
    ),
) -> None:
    """Send a test webhook event and report the real HTTP result.

    The previous version constructed a manager, called dispatch() (which only
    enqueues) and never started the worker, then printed "sent" -- nothing was
    ever delivered. It also used "test.ping", which is not in WebhookEvent and
    silently fell back to node.joined. This delivers synchronously and reports
    the actual status, so a failing endpoint is visible.
    """
    mgr = WebhookManager()
    if not mgr.register(url=url, events=["*"]):
        console.print(
            f"[red]Refused:[/red] {url} is not an allowed webhook target "
            "(private/loopback addresses are blocked unless DISTLLM_WEBHOOK_ALLOWLIST permits them)."
        )
        raise typer.Exit(1)
    target = next((t for t in mgr._targets if t.url == url), None)
    if target is None:
        console.print(f"[red]Could not resolve registered target for[/red] {url}")
        raise typer.Exit(1)

    try:
        event_enum = WebhookEvent(event)
    except ValueError:
        valid = ", ".join(e.value for e in WebhookEvent)
        console.print(
            f"[red]Unknown event:[/red] {event!r}. Valid events: {valid}"
        )
        raise typer.Exit(2)

    payload = {"test": True, "message": "Test webhook from CLI", "event": event}
    delivery = mgr._deliver(target, event_enum, payload)

    if delivery.success:
        console.print(
            f"[green]Delivered[/green] {event} to {url} "
            f"(HTTP {delivery.status_code}, {delivery.duration_ms:.0f} ms)"
        )
    else:
        console.print(
            f"[red]Delivery failed[/red] to {url} "
            f"(HTTP {delivery.status_code}): {delivery.error or 'no response'}"
        )
        raise typer.Exit(1)
