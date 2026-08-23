"""CLI commands for notification system.

Functions are imported by :mod:`distllm.cli.main` which registers them
as Typer commands on the ``system notify`` sub-group.
"""

from rich.console import Console
from rich.table import Table


def notify_send(
    title: str,
    message: str,
    severity: str = "info",
    channel: str = "console",
    webhook_url: str = "",
) -> None:
    """Send a test notification."""
    from distllm.core.notification_manager import (
        NotificationManager, Notification, NotificationSeverity, NotificationChannel,
    )

    console = Console()
    nm = NotificationManager()

    try:
        sev = NotificationSeverity(severity)
    except ValueError:
        sev = NotificationSeverity.INFO

    try:
        ch = NotificationChannel(channel)
    except ValueError:
        ch = NotificationChannel.CONSOLE

    if ch == NotificationChannel.SLACK and webhook_url:
        nm.configure_slack(webhook_url)
    elif ch == NotificationChannel.DISCORD and webhook_url:
        nm.configure_discord(webhook_url)
    elif ch == NotificationChannel.HTTP and webhook_url:
        nm.configure_http(webhook_url)

    ok = nm.send(Notification(
        title=title, message=message, severity=sev, channel=ch,
    ))
    if ok:
        console.print(f"[green]Notification sent via {ch.value}[/]")
    else:
        console.print(f"[red]Failed to send notification via {ch.value}[/]")


def notify_history(
    limit: int = 20,
    severity: str = "",
) -> None:
    """Show recent notification history."""
    from distllm.core.notification_manager import (
        NotificationManager, NotificationSeverity,
    )

    console = Console()
    nm = NotificationManager()
    sev = NotificationSeverity(severity) if severity else None
    recent = nm.recent(limit=limit, severity=sev)

    if not recent:
        console.print("[yellow]No notifications found[/]")
        return

    table = Table(title=f"Recent Notifications ({len(recent)})")
    table.add_column("Time", style="cyan")
    table.add_column("Severity", style="green")
    table.add_column("Channel")
    table.add_column("Title")
    table.add_column("Message")

    import datetime
    for n in recent:
        time_str = datetime.datetime.fromtimestamp(n.timestamp).strftime("%H:%M:%S")
        sev_color = {
            "debug": "dim", "info": "blue", "warning": "yellow",
            "error": "red", "critical": "magenta",
        }.get(n.severity.value, "")
        table.add_row(
            time_str,
            f"[{sev_color}]{n.severity.value}[/{sev_color}]" if sev_color else n.severity.value,
            n.channel.value,
            n.title[:40],
            n.message[:60],
        )
    console.print(table)
