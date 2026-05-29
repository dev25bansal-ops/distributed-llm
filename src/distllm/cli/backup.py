"""CLI commands for backup & restore."""

import typer
from rich.console import Console
from rich.table import Table

backup_app = typer.Typer(help="Backup and restore cluster state")


@backup_app.command("create")
def backup_create(
    backup_dir: str = typer.Option("./backups", "--dir", "-d", help="Backup directory"),
    cluster_name: str = typer.Option("default", "--cluster", "-c", help="Cluster name"),
    config_path: str = typer.Option("config.yaml", "--config", help="Config file path"),
):
    """Create a full backup of cluster state."""
    import yaml
    from distllm.core.backup_manager import BackupManager

    console = Console()
    mgr = BackupManager(backup_dir=backup_dir)

    config_data = {}
    try:
        with open(config_path) as f:
            config_data = yaml.safe_load(f) or {}
    except Exception:
        config_data = {"_note": "config file not found"}

    manifest = mgr.create_full(
        cluster_name=cluster_name,
        coordinator_config=config_data,
        node_registrations=[],
        model_assignments=[],
        custom_data={"cli_backup": True},
    )
    console.print(f"[green]Backup created:[/] {manifest.backup_id}")
    console.print(f"  Size: {manifest.size_bytes} bytes")
    console.print(f"  Entries: {manifest.entries}")
    console.print(f"  Type: {manifest.backup_type}")


@backup_app.command("list")
def backup_list(
    backup_dir: str = typer.Option("./backups", "--dir", "-d", help="Backup directory"),
):
    """List available backups."""
    from distllm.core.backup_manager import BackupManager

    console = Console()
    mgr = BackupManager(backup_dir=backup_dir)
    backups = mgr.list_backups()

    if not backups:
        console.print("[yellow]No backups found[/]")
        return

    table = Table(title="Available Backups")
    table.add_column("ID", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Created")
    table.add_column("Size")
    table.add_column("Entries")
    table.add_column("Cluster")

    for b in backups:
        created = b.created_at
        import datetime
        time_str = datetime.datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M")
        table.add_row(
            b.backup_id[:20],
            b.backup_type,
            time_str,
            f"{b.size_bytes} B",
            str(b.entries),
            b.cluster_name,
        )
    console.print(table)


@backup_app.command("restore")
def backup_restore(
    backup_id: str = typer.Argument(..., help="Backup ID to restore"),
    backup_dir: str = typer.Option("./backups", "--dir", "-d", help="Backup directory"),
    output: str = typer.Option("", "--output", "-o", help="Output file (default: stdout)"),
):
    """Restore a backup to its original state."""
    from distllm.core.backup_manager import BackupManager

    console = Console()
    mgr = BackupManager(backup_dir=backup_dir)
    data = mgr.restore(backup_id)

    if data is None:
        console.print(f"[red]Backup not found:[/] {backup_id}")
        raise typer.Exit(1)

    import json
    output_str = json.dumps(data, indent=2, default=str)
    if output:
        with open(output, "w") as f:
            f.write(output_str)
        console.print(f"[green]Restored to:[/] {output}")
    else:
        console.print(output_str)


@backup_app.command("delete")
def backup_delete(
    backup_id: str = typer.Argument(..., help="Backup ID to delete"),
    backup_dir: str = typer.Option("./backups", "--dir", "-d", help="Backup directory"),
):
    """Delete a backup."""
    from distllm.core.backup_manager import BackupManager

    console = Console()
    mgr = BackupManager(backup_dir=backup_dir)
    if mgr.delete_backup(backup_id):
        console.print(f"[green]Deleted:[/] {backup_id}")
    else:
        console.print(f"[red]Not found:[/] {backup_id}")
