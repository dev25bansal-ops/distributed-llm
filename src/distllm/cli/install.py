"""Pre-flight checks, installation verification, and setup guide."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.table import Table
from rich.text import Text
from loguru import logger


def run_preflight() -> dict[str, Any]:
    """Run pre-flight checks before first use."""
    console = Console()
    results: dict[str, Any] = {}

    console.print(Panel.fit("[bold cyan]Pre-Flight Checks[/]", border_style="cyan"))

    # Python version
    py_ok = sys.version_info >= (3, 10)
    results["python"] = {
        "ok": py_ok,
        "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    console.print(f"  Python {'[green]OK[/]' if py_ok else '[red]FAIL[/]'}: {results['python']['version']}")

    # CUDA
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        cuda_count = torch.cuda.device_count() if cuda_avail else 0
        cuda_version = torch.version.cuda if hasattr(torch, 'version') and torch.version.cuda else "N/A"
    except ImportError:
        cuda_avail = False
        cuda_count = 0
        cuda_version = "N/A"

    results["cuda"] = {"ok": cuda_avail, "count": cuda_count, "version": cuda_version}
    status = "[green]OK[/]" if cuda_avail else "[yellow]WARN[/]"
    console.print(f"  CUDA {status}: {cuda_count} GPU(s), v{cuda_version}")

    # Disk space
    try:
        import psutil
        disk = psutil.disk_usage(os.path.expanduser("~"))
        free_gb = disk.free / (1024**3)
        disk_ok = free_gb > 10
    except ImportError:
        free_gb = 0
        disk_ok = True

    results["disk"] = {"ok": disk_ok, "free_gb": round(free_gb, 1)}
    status = "[green]OK[/]" if disk_ok else "[red]FAIL[/]"
    console.print(f"  Disk {status}: {free_gb:.1f} GB free")

    # Docker
    docker_path = shutil.which("docker")
    docker_ok = docker_path is not None
    results["docker"] = {"ok": docker_ok, "path": docker_path or ""}
    status = "[green]OK[/]" if docker_ok else "[yellow]WARN[/]"
    console.print(f"  Docker {status}: {'found' if docker_path else 'not installed'}")

    # Network
    try:
        import httpx
        resp = httpx.get("https://huggingface.co", timeout=5.0)
        hf_ok = resp.status_code == 200
    except Exception:
        hf_ok = False
    results["network"] = {"ok": hf_ok}
    status = "[green]OK[/]" if hf_ok else "[yellow]WARN[/]"
    console.print(f"  HuggingFace {status}: {'reachable' if hf_ok else 'unreachable'}")

    return results


def download_test_model() -> bool:
    """Download a small test model (SmolLM-135M) to verify the setup works."""
    console = Console()

    console.print(Panel("[bold]Downloading Test Model[/]", border_style="blue"))

    try:
        from huggingface_hub import snapshot_download

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading SmolLM-135M...", total=100)

            def _progress_cb(current, total):
                progress.update(task, completed=int(current / max(total, 1) * 100))

            snapshot_download(
                "HuggingFaceTB/SmolLM-135M",
                local_dir=os.path.expanduser("~/.distllm/models/SmolLM-135M"),
                callback=_progress_cb,
            )

        console.print("[green]Test model downloaded successfully[/]")
        return True
    except Exception as e:
        console.print(f"[yellow]Test model download skipped: {e}[/]")
        return False


def verify_installation() -> bool:
    """Run installation verification checks."""
    console = Console()
    console.print(Panel("[bold]Installation Verification[/]", border_style="green"))

    checks = [
        ("distllm package", lambda: _check_import("distllm")),
        ("FastAPI available", lambda: _check_import("fastapi")),
        ("PyTorch available", lambda: _check_import("torch")),
        ("CUDA accessible", lambda: _check_import("torch") and __import__("torch").cuda.is_available()),
        ("Configuration dir", lambda: os.path.exists(os.path.expanduser("~/.distllm"))),
    ]

    all_ok = True
    for name, check_fn in checks:
        try:
            ok = check_fn()
        except Exception:
            ok = False
        status = "[green]PASS[/]" if ok else "[red]FAIL[/]"
        if not ok:
            all_ok = False
        console.print(f"  {status} {name}")

    if all_ok:
        console.print("\n[bold green]All checks passed![/]")
    else:
        console.print("\n[yellow]Some checks failed — review above[/]")

    return all_ok


def show_setup_guide() -> None:
    """Display a step-by-step multi-machine setup guide."""
    console = Console()
    guide = """
# DistLLM Multi-Machine Setup Guide

## Single Machine (Quick Start)
```bash
pip install distllm
distllm cluster start --model meta-llama/Llama-3.2-1B
```

## Multi-Machine (LAN)
### On the main machine (coordinator):
```bash
distllm cluster start --model meta-llama/Llama-3.2-1B --port 50050
```

### On each worker machine:
```bash
distllm cluster join --coordinator-host <coordinator-ip> --coordinator-port 50050
```

## Multi-Machine (WAN / Internet)
### On the coordinator:
```bash
distllm cluster start --model meta-llama/Llama-3.2-1B --port 50050 --wan
```

### On each remote worker:
```bash
distllm cluster join --coordinator-host <public-ip> --coordinator-port 50050 --cluster-key <key>
```

## Verifying the Cluster
```bash
distllm cluster status
distllm doctor
```
"""
    from rich.markdown import Markdown
    console.print(Markdown(guide))


def _check_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False
