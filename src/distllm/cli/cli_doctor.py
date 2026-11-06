"""``distllm-cli doctor`` — environment health check.

Companion to the richer ``distllm doctor`` (which focuses on a running
cluster); ``distllm-cli doctor`` focuses on **the local environment**
that the CLI itself runs in. Run it before any first install, before
opening an issue, or after upgrading torch / transformers.

Checks (each returns ``(status, message)`` where status is one of
``ok``/``warn``/``fail``):

  1. Python ≥ 3.10
  2. PyTorch installed
  3. CUDA available (only when PyTorch is present)
  4. bitsandbytes installed
  5. transformers installed (≥ 4.48.0)
  6. HuggingFace reachable (HEAD ``https://huggingface.co``)
  7. Default ports 8000 / 50050 not already bound
  8. ≥ 10 GB free disk space in the model cache
  9. ≥ 4 GB free GPU memory when CUDA is available

Output: a colored Rich table with a pass/warn/fail glyph per row.

Auto-remediation: ``--fix`` will pip-install missing optional packages
and clean ``~/.cache/huggingface`` if disk is tight. It will **never**
kill a process that owns port 8000/50050 (too dangerous in mixed
environments).
"""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

from rich.console import Console
from rich.table import Table


# Minimum supported versions (kept in sync with pyproject.toml)
MIN_PYTHON = (3, 10)
MIN_TRANSFORMERS = "4.48.0"
MIN_DISK_GB = 10.0
MIN_GPU_MEM_GB = 4.0

# Ports the CLI may want to bind — warning, not failure, when bound
DEFAULT_PORTS = (8000, 50050)

# HF model cache dir
HF_CACHE = "~/.cache/huggingface"


@dataclass
class CheckResult:
    name: str
    status: str  # "ok" | "warn" | "fail"
    message: str


def _check_python() -> CheckResult:
    v = sys.version_info
    if (v.major, v.minor) >= MIN_PYTHON:
        return CheckResult(
            "Python version", "ok", f"{v.major}.{v.minor}.{v.micro}"
        )
    return CheckResult(
        "Python version",
        "fail",
        f"{v.major}.{v.minor}.{v.micro} (need ≥ {'.'.join(map(str, MIN_PYTHON))})",
    )


def _check_pytorch() -> CheckResult:
    try:
        import torch  # noqa: F401
    except ImportError:
        return CheckResult("PyTorch", "warn", "not installed (pip install 'distllm[cuda]')")
    import torch

    return CheckResult("PyTorch", "ok", torch.__version__)


def _check_cuda() -> CheckResult:
    try:
        import torch
    except ImportError:
        return CheckResult("CUDA", "warn", "skipped — PyTorch not installed")
    if not torch.cuda.is_available():
        return CheckResult(
            "CUDA",
            "warn",
            "not available (CPU-only torch — install 'distllm[cuda]' for GPU)",
        )
    n = torch.cuda.device_count()
    return CheckResult("CUDA", "ok", f"available — {n} device(s)")


def _check_bitsandbytes() -> CheckResult:
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return CheckResult(
            "bitsandbytes",
            "warn",
            "not installed (pip install 'distllm[gpu]' for 4-bit / 8-bit quant)",
        )
    import bitsandbytes

    return CheckResult("bitsandbytes", "ok", bitsandbytes.__version__)


def _check_transformers() -> CheckResult:
    try:
        import transformers  # noqa: F401
    except ImportError:
        return CheckResult(
            "transformers",
            "fail",
            "not installed (pip install 'distllm[cuda]' or 'distllm[gpu]')",
        )
    import transformers
    from packaging.version import InvalidVersion, Version

    try:
        if Version(transformers.__version__) >= Version(MIN_TRANSFORMERS):
            return CheckResult("transformers", "ok", transformers.__version__)
        return CheckResult(
            "transformers",
            "warn",
            f"{transformers.__version__} (recommend ≥ {MIN_TRANSFORMERS})",
        )
    except InvalidVersion:
        return CheckResult("transformers", "warn", transformers.__version__)


def _check_hf_reachable() -> CheckResult:
    try:
        req = urllib.request.Request(
            "https://huggingface.co/", method="HEAD"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            if 200 <= resp.status < 400:
                return CheckResult(
                    "HuggingFace reachable", "ok", "HTTP HEAD ok"
                )
        return CheckResult(
            "HuggingFace reachable", "warn", f"HTTP {resp.status}"
        )
    except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
        return CheckResult(
            "HuggingFace reachable", "warn", f"unreachable ({e})"
        )


def _check_ports(ports: tuple[int, ...] = DEFAULT_PORTS) -> CheckResult:
    busy: list[int] = []
    for port in ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        try:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                busy.append(port)
        finally:
            s.close()
    if not busy:
        return CheckResult("Ports free", "ok", ", ".join(map(str, ports)) or "(none)")
    return CheckResult(
        "Ports free",
        "warn",
        f"in use: {', '.join(map(str, busy))} (another DistLLM may be running)",
    )


def _check_disk() -> CheckResult:
    usage = shutil.disk_usage(".")
    free_gb = usage.free / 1e9
    if free_gb >= MIN_DISK_GB:
        return CheckResult("Disk space", "ok", f"{free_gb:.1f} GB free")
    return CheckResult(
        "Disk space",
        "warn",
        f"{free_gb:.1f} GB free (need ≥ {MIN_DISK_GB} GB for model cache)",
    )


def _check_gpu_mem() -> CheckResult:
    try:
        import torch
    except ImportError:
        return CheckResult("GPU memory", "warn", "skipped — PyTorch not installed")
    if not torch.cuda.is_available():
        return CheckResult("GPU memory", "warn", "skipped — no CUDA")
    free_gb = 0.0
    for i in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(i)
        free_gb = max(free_gb, free / 1e9)
    if free_gb >= MIN_GPU_MEM_GB:
        return CheckResult("GPU memory", "ok", f"{free_gb:.1f} GB free")
    return CheckResult(
        "GPU memory",
        "warn",
        f"{free_gb:.1f} GB free (recommend ≥ {MIN_GPU_MEM_GB} GB)",
    )


# Ordered list of checks
CHECKS: list[tuple[str, Callable[[], CheckResult]]] = [
    ("python", _check_python),
    ("pytorch", _check_pytorch),
    ("cuda", _check_cuda),
    ("bitsandbytes", _check_bitsandbytes),
    ("transformers", _check_transformers),
    ("hf", _check_hf_reachable),
    ("ports", _check_ports),
    ("disk", _check_disk),
    ("gpu_mem", _check_gpu_mem),
]


# ── Rendering ──────────────────────────────────────────────────────────────


GLYPH = {"ok": "[green]OK[/green]", "warn": "[yellow]WARN[/yellow]", "fail": "[red]FAIL[/red]"}


def render_table(results: list[CheckResult], console: Console) -> None:
    """Render the doctor report as a colored Rich table."""
    table = Table(
        title="distllm-cli doctor — environment health",
        show_lines=False,
    )
    table.add_column("Check", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Detail", style="white")

    for r in results:
        table.add_row(r.name, GLYPH.get(r.status, r.status), r.message)
    console.print(table)

    fails = sum(1 for r in results if r.status == "fail")
    warns = sum(1 for r in results if r.status == "warn")
    if fails == 0 and warns == 0:
        console.print("\n[bold green]All checks passed.[/bold green]")
    elif fails == 0:
        console.print(
            f"\n[bold yellow]{warns} warning(s)[/bold yellow] — install missing "
            "extras or check connectivity. See [cyan]docs/INSTALL.md[/cyan]."
        )
    else:
        console.print(
            f"\n[bold red]{fails} failure(s), {warns} warning(s)[/bold red] — "
            "DistLLM cannot run until these are resolved."
        )


# ── Auto-remediation ───────────────────────────────────────────────────────


def _pip_install(pkg: str) -> tuple[bool, str]:
    """Try to ``pip install <pkg>``. Returns (success, message)."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            return True, f"installed {pkg}"
        return False, f"pip install failed: {result.stderr.strip()[:200]}"
    except subprocess.TimeoutExpired:
        return False, "pip install timed out (300s)"
    except Exception as e:
        # CQ-08: intentional fail-open: return False on failure; callers treat False as 'not available/enabled'.
        return False, f"pip install error: {e}"


def _clean_hf_cache() -> tuple[bool, str]:
    """Clean ``~/.cache/huggingface`` if it's huge."""
    import os

    path = os.path.expanduser(HF_CACHE)
    if not os.path.isdir(path):
        return False, "no HF cache to clean"
    try:
        total = 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        if total < 5 * 1024 * 1024 * 1024:  # < 5 GB — nothing to do
            return False, "HF cache already small"
        shutil.rmtree(path, ignore_errors=True)
        return True, f"freed {total / 1e9:.1f} GB by cleaning {path}"
    except Exception as e:
        # CQ-08: intentional fail-open: return False on failure; callers treat False as 'not available/enabled'.
        return False, f"clean error: {e}"


def attempt_fixes(results: list[CheckResult], console: Console) -> list[CheckResult]:
    """Try to remediate failed checks in-place. Returns updated results."""
    console.print("\n[bold cyan]Attempting auto-fixes...[/bold cyan]\n")

    for r in results:
        if r.status == "fail":
            if r.name == "Python version":
                console.print(f"[yellow]SKIP[/yellow] {r.name}: cannot auto-install Python")
            elif r.name == "transformers":
                ok, msg = _pip_install("transformers>=4.48.0")
                if ok:
                    r.status = "ok"
                    r.message = "installed via --fix"
                    console.print(f"[green]FIXED[/green] {r.name}: {msg}")
                else:
                    console.print(f"[red]FAIL[/red] {r.name}: {msg}")
        elif r.status == "warn":
            if r.name == "bitsandbytes":
                ok, msg = _pip_install("bitsandbytes")
                if ok:
                    r.status = "ok"
                    r.message = "installed via --fix"
                    console.print(f"[green]FIXED[/green] {r.name}: {msg}")
                else:
                    console.print(f"[red]FAIL[/red] {r.name}: {msg}")
            elif r.name == "Disk space":
                ok, msg = _clean_hf_cache()
                if ok:
                    console.print(f"[green]FIXED[/green] {r.name}: {msg}")
                else:
                    console.print(f"[yellow]SKIP[/yellow] {r.name}: {msg}")
            elif r.name in ("PyTorch", "CUDA"):
                console.print(
                    f"[yellow]SKIP[/yellow] {r.name}: cannot auto-install CUDA torch "
                    "(see docs/INSTALL.md)"
                )
            elif r.name == "Ports free":
                # INTENTIONALLY no auto-kill — too dangerous.
                console.print(
                    f"[yellow]SKIP[/yellow] {r.name}: refusing to auto-kill "
                    "processes — see `lsof -i :<port>`"
                )
            else:
                console.print(f"[yellow]SKIP[/yellow] {r.name}: no auto-fix available")

    return results


# ── CLI entry point ────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="distllm-cli doctor",
        description="Check the local environment for DistLLM.",
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="Attempt to auto-install missing packages and clean caches",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output machine-parseable JSON",
    )
    args = parser.parse_args(argv)

    console = Console()
    results = [fn() for _name, fn in CHECKS]

    if args.fix:
        results = attempt_fixes(results, console)
        console.print()  # blank line before table

    if args.json:
        import json

        console.print_json(
            data=[
                {"check": r.name, "status": r.status, "detail": r.message}
                for r in results
            ]
        )
    else:
        render_table(results, console)

    # Exit code: 0 if no failures, 1 if any
    return 1 if any(r.status == "fail" for r in results) else 0


__all__ = [
    "main",
    "render_table",
    "attempt_fixes",
    "CheckResult",
    "CHECKS",
]


if __name__ == "__main__":
    sys.exit(main())