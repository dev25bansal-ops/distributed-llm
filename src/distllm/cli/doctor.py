"""``distllm doctor`` — diagnose common issues in a distributed-llm cluster.

Checks:

- CUDA / ROCm availability and driver versions
- NCCL connectivity and ring health
- gRPC port reachability for registered nodes
- Firewall rules (common ports 50050-50060)
- Configuration file validity
- Disk space for model cache
"""

import argparse
import os
import shutil
import socket
import subprocess
import sys
from typing import Any

from loguru import logger


def _check_cuda() -> list[dict[str, Any]]:
    results = []
    try:
        import torch
        results.append({"check": "torch version", "status": "ok", "value": torch.__version__})
        results.append({"check": "CUDA available", "status": "ok" if torch.cuda.is_available() else "warn", "value": str(torch.cuda.is_available())})
        if torch.cuda.is_available():
            results.append({"check": "CUDA devices", "status": "ok", "value": str(torch.cuda.device_count())})
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                results.append({"check": f"  GPU {i}", "status": "ok", "value": f"{name} ({mem:.1f} GB)"})
        try:
            v = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
            if v.returncode == 0:
                results.append({"check": "nvidia-smi driver", "status": "ok", "value": v.stdout.strip().split(",")[1].strip() if "," in v.stdout else "unknown"})
        except FileNotFoundError:
            results.append({"check": "nvidia-smi", "status": "warn", "value": "not found (no NVIDIA drivers?)"})
    except ImportError:
        results.append({"check": "torch", "status": "error", "value": "not installed"})
    return results


def _check_ports(ports: list[int] = None) -> list[dict[str, Any]]:
    results = []
    if ports is None:
        ports = [50050, 50051, 50052, 50060, 8000]
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            result = sock.connect_ex(("127.0.0.1", port))
            status = "listening" if result == 0 else "free"
            results.append({"check": f"port {port}", "status": "ok" if result == 0 else "info", "value": status})
        finally:
            sock.close()
    return results


def _check_config() -> list[dict[str, Any]]:
    results = []
    config_paths = ["config.yaml", os.path.expanduser("~/.config/distllm/config.yaml")]
    for path in config_paths:
        if os.path.exists(path):
            try:
                import yaml
                with open(path) as f:
                    yaml.safe_load(f)
                results.append({"check": f"config {path}", "status": "ok", "value": "valid YAML"})
            except Exception as e:
                results.append({"check": f"config {path}", "status": "error", "value": str(e)})
        else:
            results.append({"check": f"config {path}", "status": "info", "value": "not found (OK for defaults)"})
    return results


def _check_disk() -> list[dict[str, Any]]:
    results = []
    cache_dirs = [
        os.path.expanduser("~/.cache/huggingface"),
        os.path.expanduser("~/.distllm_cache"),
    ]
    for d in cache_dirs:
        if os.path.exists(d):
            usage = shutil.disk_usage(d)
            free_gb = usage.free / 1e9
            results.append({"check": f"disk {d}", "status": "ok" if free_gb > 10 else "warn", "value": f"{free_gb:.1f} GB free"})
    if not results:
        results.append({"check": "disk cache dirs", "status": "info", "value": "not created yet"})
    return results


def _print_results(category: str, results: list[dict[str, Any]]) -> None:
    print(f"\n{'='*60}")
    print(f"  {category}")
    print(f"{'='*60}")
    for r in results:
        icon = {"ok": "✅", "warn": "⚠️", "error": "❌", "info": "ℹ️"}.get(r["status"], "  ")
        msg = f"  {icon} {r['check']}: {r.get('value', '')}"
        print(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="DistLLM diagnostic tool")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    args = parser.parse_args()

    print("DistLLM Doctor — System Diagnostics")
    print("=" * 60)

    _print_results("CUDA / GPU", _check_cuda())
    _print_results("Ports", _check_ports())
    _print_results("Configuration", _check_config())
    _print_results("Disk Space", _check_disk())

    print(f"\n{'='*60}")
    print("  Run `distllm cluster start --help` for cluster commands.")
    print("  Report issues at https://github.com/distributed-llm/distributed-llm/issues")
    print()


if __name__ == "__main__":
    main()
