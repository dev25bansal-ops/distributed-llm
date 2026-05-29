"""Cluster Health Autopsy — collect logs, metrics, and configuration into a single report.

Run after a failure to gather all diagnostic information::

    distllm cluster autopsy --since 1h --output autopsy-report.zip

The report includes:
  - Coordinator logs (last N lines)
  - Worker node health check history
  - gRPC connection states
  - System metrics (CPU, memory, GPU)
  - Configuration files
  - Recent error traces
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import zipfile
from datetime import datetime, timedelta
from typing import Any


def collect_autopsy(
    coordinator_host: str = "localhost",
    coordinator_port: int = 8000,
    since_minutes: int = 60,
    output_path: str = "",
) -> str:
    """Collect all diagnostic data into a zip report.

    Args:
        coordinator_host: Coordinator API host.
        coordinator_port: Coordinator API port.
        since_minutes: How far back to look for logs.
        output_path: Output zip path (auto-generated if empty).

    Returns:
        Path to the generated report zip.
    """
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"distllm-autopsy-{timestamp}.zip"

    import httpx

    data: dict[str, Any] = {
        "timestamp": time.time(),
        "collector": "distllm cluster autopsy",
        "since_minutes": since_minutes,
    }

    # 1. Coordinator health check
    try:
        resp = httpx.get(f"http://{coordinator_host}:{coordinator_port}/health", timeout=5)
        data["health"] = resp.json()
    except Exception as e:
        data["health"] = {"error": str(e)}

    # 2. Metrics
    try:
        resp = httpx.get(f"http://{coordinator_host}:{coordinator_port}/metrics", timeout=5)
        data["metrics"] = resp.text[:50000]
    except Exception as e:
        data["metrics"] = {"error": str(e)}

    # 3. Node list
    try:
        resp = httpx.get(f"http://{coordinator_host}:{coordinator_port}/api/cluster/nodes", timeout=5)
        data["nodes"] = resp.json()
    except Exception as e:
        data["nodes"] = {"error": str(e)}

    # 4. Reputation scores
    try:
        resp = httpx.get(f"http://{coordinator_host}:{coordinator_port}/api/cluster/reputation", timeout=5)
        data["reputation"] = resp.json()
    except Exception as e:
        data["reputation"] = {"error": str(e)}

    # 5. Configuration files
    config_files = [
        "config.yaml",
        os.path.expanduser("~/.config/distllm/config.yaml"),
        ".env",
    ]
    configs = {}
    for cf in config_files:
        if os.path.exists(cf):
            with open(cf) as f:
                configs[cf] = f.read()
    data["config_files"] = configs

    # 6. System info
    try:
        import psutil
        data["system"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage("/").percent,
        }
    except Exception:
        pass

    # 7. GPU info
    try:
        import torch
        gpu_info = []
        for i in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
            gpu_info.append({
                "name": torch.cuda.get_device_name(i),
                "memory_allocated_gb": round(torch.cuda.memory_allocated(i) / 1e9, 2),
                "memory_reserved_gb": round(torch.cuda.memory_reserved(i) / 1e9, 2),
            })
        data["gpu"] = gpu_info
    except Exception:
        pass

    # 8. Log tail from journal/systemd
    try:
        log_lines = subprocess.run(
            ["journalctl", "-u", "distllm*", "--since", f"{since_minutes} min ago", "--no-pager", "-n", "500"],
            capture_output=True, text=True, timeout=10,
        )
        if log_lines.stdout:
            data["logs"] = log_lines.stdout[-50000:]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Write report
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report.json", json.dumps(data, indent=2, default=str))
        # Add individual log files
        if "logs" in data:
            zf.writestr("logs/system.log", data["logs"][:50000])
        if "metrics" in data:
            zf.writestr("metrics/prometheus.txt", str(data["metrics"])[:50000])

    print(f"Autopsy report saved to {output_path}")
    return output_path
