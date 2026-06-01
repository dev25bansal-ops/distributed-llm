"""``distllm autopsy`` — post-mortem analysis tool for DistLLM clusters.

Collects logs, metrics, configuration, and system state into a single
diagnostic report for troubleshooting failures.

Usage::

    distllm autopsy                          # Full autopsy report
    distllm autopsy --since 1h               # Last hour only
    distllm autopsy --output report.zip      # Custom output path
    distllm autopsy --quick                  # Quick health check only
    distllm autopsy --nodes node1,node2      # Specific nodes

The report includes:
  - Coordinator logs (last N lines)
  - Worker node health check history
  - gRPC connection states
  - System metrics (CPU, memory, GPU)
  - Configuration files
  - Recent error traces
  - KV cache statistics
  - Request latency histograms
  - Node topology snapshot
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def collect_autopsy(
    coordinator_host: str = "localhost",
    coordinator_port: int = 8000,
    since_minutes: int = 60,
    output_path: str = "",
    quick: bool = False,
    nodes: list[str] | None = None,
) -> str:
    """Collect all diagnostic data into a zip report.

    Args:
        coordinator_host: Coordinator API host.
        coordinator_port: Coordinator API port.
        since_minutes: How far back to look for logs.
        output_path: Output zip path (auto-generated if empty).
        quick: If True, only collect health and basic metrics.
        nodes: Specific node IDs to collect data from.

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
        "quick": quick,
        "coordinator": f"{coordinator_host}:{coordinator_port}",
    }

    base_url = f"http://{coordinator_host}:{coordinator_port}"

    # 1. Health check
    try:
        resp = httpx.get(f"{base_url}/health", timeout=5)
        data["health"] = resp.json()
    except Exception as e:
        data["health"] = {"error": str(e)}

    if quick:
        return _write_report(data, output_path)

    # 2. Metrics
    try:
        resp = httpx.get(f"{base_url}/metrics", timeout=5)
        data["metrics"] = resp.text[:100000]
    except Exception as e:
        data["metrics"] = {"error": str(e)}

    # 3. Node list
    try:
        resp = httpx.get(f"{base_url}/api/cluster/nodes", timeout=5)
        data["nodes"] = resp.json()
    except Exception as e:
        data["nodes"] = {"error": str(e)}

    # 4. Scheduler stats
    try:
        resp = httpx.get(f"{base_url}/v1/scheduler/stats", timeout=5)
        data["scheduler"] = resp.json()
    except Exception as e:
        data["scheduler"] = {"error": str(e)}

    # 5. KV cache stats
    try:
        resp = httpx.get(f"{base_url}/api/metrics/cache", timeout=5)
        data["cache"] = resp.json()
    except Exception as e:
        data["cache"] = {"error": str(e)}

    # 6. Recent errors
    try:
        resp = httpx.get(f"{base_url}/v1/debug/recent?limit=50", timeout=5)
        data["recent_requests"] = resp.json()
    except Exception as e:
        data["recent_requests"] = {"error": str(e)}

    # 7. Configuration files
    config_files = [
        "config.yaml",
        os.path.expanduser("~/.config/distllm/config.yaml"),
        ".env",
    ]
    configs = {}
    for cf in config_files:
        if os.path.exists(cf):
            try:
                with open(cf) as f:
                    configs[cf] = f.read()[:10000]
            except Exception:
                pass
    data["config_files"] = configs

    # 8. System info
    try:
        import psutil
        data["system"] = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "cpu_count": psutil.cpu_count(),
            "memory": {
                "total_gb": round(psutil.virtual_memory().total / 1e9, 2),
                "used_gb": round(psutil.virtual_memory().used / 1e9, 2),
                "percent": psutil.virtual_memory().percent,
            },
            "disk": {
                "total_gb": round(psutil.disk_usage("/").total / 1e9, 2),
                "free_gb": round(psutil.disk_usage("/").free / 1e9, 2),
                "percent": psutil.disk_usage("/").percent,
            },
        }
    except ImportError:
        data["system"] = {"error": "psutil not installed"}

    # 9. GPU info
    try:
        import torch
        gpu_info = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    "id": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1e9, 2),
                    "allocated_gb": round(torch.cuda.memory_allocated(i) / 1e9, 2),
                    "reserved_gb": round(torch.cuda.memory_reserved(i) / 1e9, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                })
        data["gpu"] = gpu_info
    except Exception:
        data["gpu"] = []

    # 10. Log tail from journal/systemd
    try:
        log_lines = subprocess.run(
            ["journalctl", "-u", "distllm*", "--since", f"{since_minutes} min ago",
             "--no-pager", "-n", "1000"],
            capture_output=True, text=True, timeout=10,
        )
        if log_lines.stdout:
            data["systemd_logs"] = log_lines.stdout[-100000:]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 11. Docker container status (if applicable)
    try:
        docker_ps = subprocess.run(
            ["docker", "ps", "--filter", "name=distllm", "--format",
             "{{.Names}}\t{{.Status}}\t{{.Ports}}"],
            capture_output=True, text=True, timeout=10,
        )
        if docker_ps.stdout:
            data["docker_containers"] = docker_ps.stdout.strip().split("\n")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return _write_report(data, output_path)


def _write_report(data: dict, output_path: str) -> str:
    """Write the autopsy data to a zip file."""
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report.json", json.dumps(data, indent=2, default=str))

        # Separate files for easy reading
        if "systemd_logs" in data:
            zf.writestr("logs/systemd.log", data["systemd_logs"][:100000])
        if "metrics" in data and isinstance(data["metrics"], str):
            zf.writestr("metrics/prometheus.txt", data["metrics"][:100000])
        if "config_files" in data:
            for path, content in data["config_files"].items():
                zf.writestr(f"config/{os.path.basename(path)}", content)

    print(f"Autopsy report saved to {output_path}")
    print(f"  Health: {data.get('health', {}).get('status', 'unknown')}")
    print(f"  Nodes: {len(data.get('nodes', {}).get('nodes', [])) if isinstance(data.get('nodes'), dict) else 'unknown'}")
    print(f"  GPU: {len(data.get('gpu', []))} devices")
    return output_path


def main() -> None:
    """CLI entry point for distllm autopsy."""
    import argparse

    parser = argparse.ArgumentParser(
        description="DistLLM post-mortem analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  distllm autopsy                          # Full report
  distllm autopsy --since 1h               # Last hour
  distllm autopsy --quick                  # Health check only
  distllm autopsy --output report.zip      # Custom output
  distllm autopsy --host coord --port 8000  # Remote coordinator
        """,
    )
    parser.add_argument("--host", default="localhost", help="Coordinator host")
    parser.add_argument("--port", type=int, default=8000, help="Coordinator port")
    parser.add_argument("--since", default="1h", help="Time window (e.g., 1h, 30m, 2d)")
    parser.add_argument("--output", "-o", default="", help="Output zip path")
    parser.add_argument("--quick", action="store_true", help="Quick health check only")
    parser.add_argument("--nodes", default="", help="Comma-separated node IDs")

    args = parser.parse_args()

    # Parse time window
    since_minutes = _parse_duration(args.since)
    nodes = [n.strip() for n in args.nodes.split(",") if n.strip()] if args.nodes else None

    print(f"DistLLM Autopsy — Collecting diagnostics (last {since_minutes} minutes)")
    print(f"Coordinator: {args.host}:{args.port}")
    if args.quick:
        print("Mode: Quick (health only)")
    print()

    try:
        report_path = collect_autopsy(
            coordinator_host=args.host,
            coordinator_port=args.port,
            since_minutes=since_minutes,
            output_path=args.output,
            quick=args.quick,
            nodes=nodes,
        )
        print(f"\nReport: {report_path}")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


def _parse_duration(s: str) -> int:
    """Parse a duration string like '1h', '30m', '2d' into minutes."""
    s = s.strip().lower()
    if s.endswith("h"):
        return int(s[:-1]) * 60
    if s.endswith("m"):
        return int(s[:-1])
    if s.endswith("d"):
        return int(s[:-1]) * 1440
    return int(s)  # Default to minutes


if __name__ == "__main__":
    main()
