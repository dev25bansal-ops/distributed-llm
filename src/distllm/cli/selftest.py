"""CLI self-verification in the user's own environment.

`distllm cli-doctor` checks the *machine* (CUDA, ports, disk). This checks the
*installation and the CLI's own promises*: that every console script resolves,
that the model catalog is reachable, that the version the package reports
matches what was installed, and that a real server can answer a real request.
It exists so a support question starts with "run `distllm selftest`" rather
than a guessing game.

Exit codes: 0 all pass, 1 any FAIL, 2 warnings only (unless --strict).
"""

from __future__ import annotations

import importlib
import io
import json
import socket
import sys
import time
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

console = Console()

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"


@dataclass
class Check:
    name: str
    status: str
    detail: str


def _check_version() -> Check:
    """The reported version must match the installed distribution."""
    try:
        from importlib.metadata import version as pkg_version

        installed = pkg_version("distributed-llm")
    except Exception as exc:  # not installed as a distribution
        return Check("version", WARN, f"not installed as a package ({exc})")

    try:
        import distllm

        reported = distllm.__version__
    except Exception as exc:
        return Check("version", FAIL, f"cannot import distllm: {exc}")

    if reported != installed:
        return Check(
            "version", FAIL, f"reports {reported} but {installed} is installed"
        )
    return Check("version", PASS, installed)


def _check_console_scripts() -> Check:
    """Every declared entry point must resolve to a callable.

    A development checkout exposes the same name twice: once from the
    installed distribution and once from the source tree on sys.path. Only one
    of those needs to be callable, so the check passes if any resolves.
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - Python <3.10
        return Check("console scripts", SKIP, "importlib.metadata unavailable")

    scripts = [e for e in entry_points(group="console_scripts") if "distllm" in e.name]
    if not scripts:
        return Check("console scripts", FAIL, "no distllm entry points installed")

    by_name: dict[str, list[str]] = {}
    for entry in scripts:
        try:
            obj = entry.load()
            ok = callable(obj)
        except Exception as exc:
            by_name.setdefault(entry.name, []).append(exc.__class__.__name__)
            continue
        if ok:
            by_name.setdefault(entry.name, []).append("ok")
        else:
            by_name.setdefault(entry.name, []).append("not callable")

    broken = [name for name, states in by_name.items() if "ok" not in states]
    if broken:
        return Check("console scripts", FAIL, f"unresolvable: {', '.join(broken)}")
    return Check("console scripts", PASS, f"{len(by_name)} resolve")


def _check_imports() -> Check:
    """The command modules must import without a network or a GPU."""
    modules = [
        "distllm.cli.main",
        "distllm.cli.quickstart",
        "distllm.cli.chat",
        "distllm.cli.cluster",
        "distllm.cli.models",
        "distllm.models.catalog",
        "distllm.models.recommend",
    ]
    broken = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            broken.append(f"{name} ({exc.__class__.__name__})")
    if broken:
        return Check("module imports", FAIL, "; ".join(broken))
    return Check("module imports", PASS, f"{len(modules)} modules")


def _check_catalog(timeout: float = 15.0) -> Check:
    """Hugging Face must be reachable for model search to work."""
    try:
        from distllm.models.catalog import search_huggingface

        started = time.time()
        results = search_huggingface("qwen", limit=2, timeout=timeout)
        elapsed = time.time() - started
    except Exception as exc:
        return Check("model catalog", FAIL, f"{exc.__class__.__name__}: {exc}")

    if not results:
        return Check(
            "model catalog",
            WARN,
            "Hugging Face returned nothing (offline, rate-limited, or blocked)",
        )
    return Check("model catalog", PASS, f"{len(results)} results in {elapsed:.1f}s")


def _check_port_free(port: int) -> Check:
    """A default port that is already taken will break cluster start."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return Check(f"port {port}", WARN, "already in use")
    return Check(f"port {port}", PASS, "free")


def _check_api_roundtrip() -> Check:
    """Start a tiny ASGI app and make a real request through it.

    This is the check that catches a broken dependency graph: everything else
    can pass while an import inside a route module fails on first request.

    /health answers 503 when no model is loaded, which is its documented
    contract, so any HTTP response proves the app served the request; only a
    failure to answer at all is a problem.
    """
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        return Check("api roundtrip", SKIP, "httpx/starlette test client not installed")

    try:
        from distllm.api.server import app

        # Without `with`, TestClient skips the lifespan startup -- which is
        # exactly what we want: the lifespan installs SIGTERM/SIGINT handlers
        # for graceful shutdown, and signal.signal() raises "signal only works
        # in main thread" anywhere else. The check is about the app answering a
        # request, not about its shutdown wiring, so it must work off the main
        # thread too (selftest is called from a worker in the test suite, and a
        # tool must not die because it was called from a thread).
        client = TestClient(app)
        response = client.get("/health")
    except Exception as exc:
        return Check("api roundtrip", FAIL, f"{exc.__class__.__name__}: {exc}")

    if response.status_code in (200, 503):
        note = "model loaded" if response.status_code == 200 else "no model loaded (expected)"
        return Check("api roundtrip", PASS, f"GET /health -> {response.status_code}, {note}")
    return Check("api roundtrip", FAIL, f"/health returned unexpected {response.status_code}")


def _remediation(checks: list[Check]) -> str:
    """Name the extra that actually fixes the failures that were seen.

    The old advice was unconditionally ``distributed-llm[cli]``; for the
    torch-import failure ([cli] carries no torch — it lives in [self-hosted])
    that command reinstalls the same extras closure and reproduces the
    identical failure.  Failures whose detail names a missing ML/optional
    module therefore get the [self-hosted] extra instead.
    """
    ml_markers = (
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "huggingface",
    )
    needs_ml = any(c.status == FAIL and any(m in c.detail for m in ml_markers) for c in checks)
    extra = "self-hosted" if needs_ml else "cli"
    return f"Reinstall: pip install --force-reinstall " f"[cyan]distributed-llm\\[{extra}][/cyan]"


def run_selftest(*, as_json: bool, strict: bool) -> int:
    # --json promises machine-readable output, so nothing else may reach
    # stdout: the loguru sink and the server's structured logging both write
    # there, which made the JSON unparseable.
    if as_json:
        import contextlib
        import sys

        with contextlib.redirect_stdout(io.StringIO()):
            checks = _run_all_checks()
        payload = [
            {"check": c.name, "status": c.status, "detail": c.detail} for c in checks
        ]
        sys.stdout.write(json.dumps({"checks": payload}, indent=2) + "\n")
        sys.stdout.flush()
        return _exit_code(checks, strict)

    checks = _run_all_checks()
    table = Table(title="distllm selftest")
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Detail", style="dim")
    colours = {PASS: "green", WARN: "yellow", FAIL: "red", SKIP: "dim"}
    for c in checks:
        table.add_row(c.name, f"[{colours[c.status]}]{c.status}[/]", c.detail)
    console.print(table)

    failed = [c for c in checks if c.status == FAIL]
    warned = [c for c in checks if c.status == WARN]
    if failed:
        console.print(
            f"\n[red]{len(failed)} check(s) failed.[/red] " + _remediation(checks)
        )
    elif warned:
        console.print(f"\n[yellow]{len(warned)} warning(s).[/yellow]")
    else:
        console.print("\n[green]All checks passed.[/green]")

    return _exit_code(checks, strict)


def _run_all_checks() -> list[Check]:
    return [
        _check_version(),
        _check_console_scripts(),
        _check_imports(),
        _check_catalog(),
        _check_port_free(8000),
        _check_port_free(50050),
        _check_api_roundtrip(),
    ]


def _exit_code(checks: list[Check], strict: bool) -> int:
    if any(c.status == FAIL for c in checks):
        return 1
    if strict and any(c.status == WARN for c in checks):
        return 2
    return 0
