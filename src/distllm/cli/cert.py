"""CLI commands for certificate management."""

import typer
from rich.console import Console
from rich.table import Table

cert_app = typer.Typer(help="Manage TLS certificates")


@cert_app.command("create")
def cert_create(
    common_name: str = typer.Argument(..., help="Certificate common name (domain)"),
    alt_names: list[str] = typer.Option([], "--alt-name", "-a", help="Subject alternative names"),
    cert_dir: str = typer.Option("./certs", "--dir", "-d", help="Certificate directory"),
    self_signed: bool = typer.Option(True, "--self-signed", help="Create self-signed certificate"),
):
    """Create a TLS certificate."""
    from distllm.core.certificate_manager import CertificateManager

    console = Console()
    mgr = CertificateManager(cert_dir=cert_dir)
    info = mgr.ensure_certificate(common_name, alt_names=alt_names)
    console.print(f"[green]Certificate created:[/] {common_name}")
    console.print(f"  Path: {info.cert_path}")
    console.print(f"  Key:  {info.key_path}")
    console.print(f"  Expires: {info.not_after}")
    console.print(f"  SANs: {', '.join(info.subject_alt_names)}")


@cert_app.command("info")
def cert_info(
    common_name: str = typer.Argument(..., help="Certificate common name"),
    cert_dir: str = typer.Option("./certs", "--dir", "-d", help="Certificate directory"),
):
    """Show certificate details."""
    from distllm.core.certificate_manager import CertificateManager

    console = Console()
    mgr = CertificateManager(cert_dir=cert_dir)
    info = mgr.get_certificate_info(common_name)
    if info is None:
        console.print(f"[red]Certificate not found:[/] {common_name}")
        raise typer.Exit(1)

    import datetime
    table = Table(title=f"Certificate: {common_name}")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Common Name", info.common_name)
    table.add_row("SANs", ", ".join(info.subject_alt_names))
    table.add_row("Issuer", info.issuer)
    table.add_row("Not Before", datetime.datetime.fromtimestamp(info.not_before).isoformat())
    table.add_row("Not After", datetime.datetime.fromtimestamp(info.not_after).isoformat())
    table.add_row("Fingerprint (SHA256)", info.fingerprint_sha256)
    table.add_row("Self-Signed", str(info.is_self_signed))
    table.add_row("Cert Path", info.cert_path)
    table.add_row("Key Path", info.key_path)
    console.print(table)


@cert_app.command("renew")
def cert_renew(
    cert_dir: str = typer.Option("./certs", "--dir", "-d", help="Certificate directory"),
):
    """Renew all certificates nearing expiry."""
    from distllm.core.certificate_manager import CertificateManager

    console = Console()
    mgr = CertificateManager(cert_dir=cert_dir)
    renewed = mgr.renew_all()
    if renewed:
        for info in renewed:
            console.print(f"[green]Renewed:[/] {info.common_name}")
    else:
        console.print("[yellow]No certificates needed renewal[/]")


@cert_app.command("revoke")
def cert_revoke(
    common_name: str = typer.Argument(..., help="Certificate common name"),
    cert_dir: str = typer.Option("./certs", "--dir", "-d", help="Certificate directory"),
):
    """Revoke a certificate."""
    from distllm.core.certificate_manager import CertificateManager

    console = Console()
    mgr = CertificateManager(cert_dir=cert_dir)
    mgr.revoke(common_name)
    console.print(f"[green]Revoked:[/] {common_name}")
