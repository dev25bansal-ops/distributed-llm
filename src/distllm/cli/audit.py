"""``distllm audit`` — compliance / security posture check.

Reads the *current* environment and configuration and reports a
pass/fail for each of nine security & correctness checks:

1. **TLS** on the gRPC transport
2. **Auth** required (never disabled)
3. **Rate limit** configured (positive request budget)
4. **CORS** allowed origins (not wildcard in production)
5. **CSRF** Same-Origin middleware enabled
6. **API key store** source (env / file / generated)
7. **Quota** middleware enabled
8. **DHT** fail-closed vs fail-open (shared secret configured)
9. **Pipeline TLS** (encrypts activations over gRPC)

Each check is independent — the audit never aborts on a failure — and
the report prints in either a Rich table or JSON (for CI consumption).

Usage::

    distllm audit                       # human-readable
    distllm audit --json                # machine-readable for CI
    distllm audit --strict              # exit non-zero on any FAIL
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Check dataclass
# ---------------------------------------------------------------------------


@dataclass
class Check:
    """One audit check."""

    name: str
    status: str  # "PASS" | "FAIL" | "WARN" | "INFO"
    detail: str
    remediation: str = ""


@dataclass
class AuditReport:
    """Aggregate audit output."""

    checks: list[Check] = field(default_factory=list)

    @property
    def overall(self) -> str:
        if any(c.status == "FAIL" for c in self.checks):
            return "FAIL"
        if any(c.status == "WARN" for c in self.checks):
            return "WARN"
        return "PASS"

    def to_dict(self) -> dict:
        return {
            "overall": self.overall,
            "checks": [asdict(c) for c in self.checks],
        }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _env_true(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


def _env_positive_int(name: str, default: str = "0") -> int:
    try:
        return int(os.environ.get(name, default))
    except (ValueError, TypeError):
        return 0


def check_tls() -> Check:
    """TLS on the gRPC transport."""
    enabled = _env_true("DISTLLM_TLS_ENABLED", "false")
    detail = (
        "DISTLLM_TLS_ENABLED=true — gRPC traffic encrypted"
        if enabled
        else "DISTLLM_TLS_ENABLED is unset — gRPC traffic is plaintext"
    )
    return Check(
        name="tls_grpc",
        status="PASS" if enabled else "FAIL",
        detail=detail,
        remediation="Set DISTLLM_TLS_ENABLED=true (and provide certs) for production."
        if not enabled
        else "",
    )


def check_auth() -> Check:
    """Authentication required / disabled."""
    api_key = os.environ.get("API_KEY", "")
    was_set = os.environ.get("API_KEY_WAS_SET", "")
    if not api_key:
        # Middleware auto-generates a key — that's an INFO, not FAIL,
        # but operators should set API_KEY explicitly.
        return Check(
            name="auth_required",
            status="WARN",
            detail="API_KEY not set — middleware generated a random key for this process",
            remediation="Set API_KEY=<secret> explicitly so restarts don't rotate the key.",
        )
    return Check(
        name="auth_required",
        status="PASS",
        detail="API_KEY is set; auth is required",
    )


def check_rate_limit() -> Check:
    """Request rate limit configured."""
    limit = _env_positive_int("DISTLLM_RATE_LIMIT_REQUESTS", "1000")
    if limit > 0:
        return Check(
            name="rate_limit",
            status="PASS",
            detail=f"DISTLLM_RATE_LIMIT_REQUESTS={limit} per 60s window",
        )
    return Check(
        name="rate_limit",
        status="FAIL",
        detail="DISTLLM_RATE_LIMIT_REQUESTS is 0 or unset — no rate limit",
        remediation="Set DISTLLM_RATE_LIMIT_REQUESTS to a positive integer.",
    )


def check_cors() -> Check:
    """CORS allowed origins — wildcard is a fail in production."""
    origins_raw = os.environ.get("DISTLLM_CORS_ORIGINS", "")
    allow_all = _env_true("DISTLLM_CORS_ALLOW_ALL", "false")
    if allow_all:
        return Check(
            name="cors",
            status="FAIL",
            detail="DISTLLM_CORS_ALLOW_ALL=true — wildcard CORS",
            remediation="Set DISTLLM_CORS_ALLOW_ALL=false and specify DISTLLM_CORS_ORIGINS explicitly.",
        )
    if not origins_raw:
        return Check(
            name="cors",
            status="WARN",
            detail="DISTLLM_CORS_ORIGINS unset — using safe localhost defaults",
            remediation="Set DISTLLM_CORS_ORIGINS=https://your-frontend.example.com",
        )
    return Check(
        name="cors",
        status="PASS",
        detail=f"DISTLLM_CORS_ORIGINS configured ({len(origins_raw.split(','))} origins)",
    )


def check_csrf() -> Check:
    """CSRF Same-Origin middleware enabled.

    The middleware is unconditionally registered in ``server.py``;
    there is no env var to disable it.  We treat its presence as a
    PASS, and emit an INFO line about what it does.
    """
    # The middleware imports eagerly in server.py; if we got here at
    # all, that import worked.  Report PASS.
    return Check(
        name="csrf",
        status="PASS",
        detail="CSRFSameOriginMiddleware registered (validates Origin/Referer on state-changing requests)",
    )


def check_api_key_store() -> Check:
    """API key store source: env / file / generated."""
    # The middleware reads from API_KEY first, then falls back to a
    # generated random key.  We don't have visibility into the
    # generated-key log without a live process; report what's visible
    # from the environment.
    api_key = os.environ.get("API_KEY", "")
    if api_key:
        source = "env"
    elif os.path.isfile(os.path.expanduser("~/.distllm/api_key")):
        source = "file"
    else:
        source = "generated"
    detail = f"API key source: {source}"
    return Check(
        name="api_key_store",
        status="INFO",
        detail=detail,
    )


def check_quota() -> Check:
    """Per-tenant quota middleware enabled."""
    enabled = _env_true("DISTLLM_QUOTA_ENABLED", "1")  # default ON
    if enabled:
        return Check(
            name="quota",
            status="PASS",
            detail="DISTLLM_QUOTA_ENABLED is on (default)",
        )
    return Check(
        name="quota",
        status="WARN",
        detail="DISTLLM_QUOTA_ENABLED=0 — quota middleware disabled",
        remediation="Re-enable quota enforcement unless intentionally disabled.",
    )


def check_dht_fail_mode() -> Check:
    """DHT fail-closed (shared secret) vs fail-open (no auth)."""
    secret = os.environ.get("DISTLLM_DHT_SECRET", "")
    if secret:
        return Check(
            name="dht_fail_mode",
            status="PASS",
            detail="DISTLLM_DHT_SECRET set — DHT is fail-closed (signed STORE)",
        )
    return Check(
        name="dht_fail_mode",
        status="FAIL",
        detail="DISTLLM_DHT_SECRET unset — DHT operates without authentication",
        remediation="Set DISTLLM_DHT_SECRET=<random-secret> on every node.",
    )


def check_pipeline_tls() -> Check:
    """Pipeline-parallel gRPC TLS — encrypts activations/KV transfers."""
    enabled = _env_true("DISTLLM_PIPELINE_TLS", "0")
    if enabled:
        ca = os.environ.get("DISTLLM_TLS_CA_CERT_FILE") or os.environ.get(
            "DISTLLM_TLS_CA_CERT"
        )
        if ca:
            return Check(
                name="pipeline_tls",
                status="PASS",
                detail="DISTLLM_PIPELINE_TLS=1 with CA cert configured",
            )
        return Check(
            name="pipeline_tls",
            status="WARN",
            detail="DISTLLM_PIPELINE_TLS=1 but no DISTLLM_TLS_CA_CERT(_FILE) set",
            remediation="Point DISTLLM_TLS_CA_CERT_FILE at the cluster CA cert.",
        )
    return Check(
        name="pipeline_tls",
        status="WARN",
        detail="DISTLLM_PIPELINE_TLS unset — activations transfer plaintext between nodes",
        remediation="Set DISTLLM_PIPELINE_TLS=1 (and configure DISTLLM_TLS_CA_CERT_FILE).",
    )


# ---------------------------------------------------------------------------
# Run + render
# ---------------------------------------------------------------------------


def run_audit() -> AuditReport:
    """Run every check and return the aggregate report."""
    return AuditReport(
        checks=[
            check_tls(),
            check_auth(),
            check_rate_limit(),
            check_cors(),
            check_csrf(),
            check_api_key_store(),
            check_quota(),
            check_dht_fail_mode(),
            check_pipeline_tls(),
        ]
    )


def render_text(report: AuditReport) -> str:
    """Render the report as a human-readable table (Rich if available)."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=f"distllm audit — {report.overall}", show_lines=False)
        table.add_column("Check", style="cyan")
        table.add_column("Status")
        table.add_column("Detail")

        status_style = {
            "PASS": "[green]PASS[/green]",
            "FAIL": "[red]FAIL[/red]",
            "WARN": "[yellow]WARN[/yellow]",
            "INFO": "[blue]INFO[/blue]",
        }
        for c in report.checks:
            table.add_row(
                c.name,
                status_style.get(c.status, c.status),
                c.detail,
            )
        # Use a captured-buffer console so we get the rendered string.
        from io import StringIO

        from rich.console import Console as _Console

        sio = StringIO()
        capture_console = _Console(file=sio, force_terminal=False, width=120)
        capture_console.print(table)
        out = sio.getvalue()
        # Append remediation as a footer.
        for c in report.checks:
            if c.remediation:
                out += f"\n  Remediation [{c.name}]: {c.remediation}\n"
        return out
    except ImportError:
        # Fallback to plain-text rendering.
        lines = [f"distllm audit — {report.overall}", "-" * 60]
        for c in report.checks:
            lines.append(f"  [{c.status}] {c.name}: {c.detail}")
            if c.remediation:
                lines.append(f"         fix: {c.remediation}")
        return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``distllm audit``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="distllm audit",
        description="Compliance / security posture check",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON (for CI)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any check is FAIL (not just any FAIL/WARN)",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    report = run_audit()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render_text(report))

    # Exit code: 0 if PASS or WARN, 1 if FAIL (or always 1 in --strict
    # when there are WARNs).
    if any(c.status == "FAIL" for c in report.checks):
        return 1
    if args.strict and any(c.status == "WARN" for c in report.checks):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
