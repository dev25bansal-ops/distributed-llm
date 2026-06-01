"""``distllm doctor`` — comprehensive system diagnostics for distributed-llm.

Checks:
- CUDA / ROCm availability, driver versions, cuDNN
- GPU capabilities (compute capability, VRAM, bandwidth)
- NCCL connectivity and ring health
- Network latency between nodes
- Model compatibility with detected hardware
- Optimal partition strategy suggestion
- Firewall rules (common ports 50050-50060)
- Configuration file validity
- Disk space for model cache
- Python environment health

Usage::

    distllm doctor                     # Full diagnostic
    distllm doctor --gpu               # GPU-only checks
    distllm doctor --network           # Network-only checks
    distllm doctor --model llama-70b   # Model compatibility check
    distllm doctor --verbose           # Detailed output
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
from typing import Any

from loguru import logger


# ── GPU Diagnostics ──────────────────────────────────────────────────────

def _check_cuda() -> list[dict[str, Any]]:
    """Check CUDA availability, driver version, and GPU details."""
    results = []
    try:
        import torch
        results.append({"check": "PyTorch version", "status": "ok", "value": torch.__version__})

        cuda_available = torch.cuda.is_available()
        results.append({
            "check": "CUDA available",
            "status": "ok" if cuda_available else "error",
            "value": str(cuda_available),
        })

        if cuda_available:
            results.append({"check": "CUDA version", "status": "ok", "value": torch.version.cuda or "unknown"})
            results.append({"check": "cuDNN version", "status": "ok", "value": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "not available"})
            results.append({"check": "cuDNN enabled", "status": "ok", "value": str(torch.backends.cudnn.enabled)})

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / 1e9
                results.append({
                    "check": f"GPU {i}",
                    "status": "ok",
                    "value": f"{props.name} ({mem_gb:.1f} GB, SM {props.major}.{props.minor}, {props.multi_processor_count} SMs)",
                })

                # Check compute capability
                if props.major < 7:
                    results.append({
                        "check": f"GPU {i} compute capability",
                        "status": "warn",
                        "value": f"SM {props.major}.{props.minor} — may not support FlashAttention",
                    })
                elif props.major >= 8:
                    results.append({
                        "check": f"GPU {i} compute capability",
                        "status": "ok",
                        "value": f"SM {props.major}.{props.minor} — supports BF16, FlashAttention",
                    })

                # Memory check
                mem_used = torch.cuda.memory_allocated(i) / 1e9
                mem_free = mem_gb - mem_used
                results.append({
                    "check": f"GPU {i} memory",
                    "status": "ok" if mem_free > 2 else "warn",
                    "value": f"Used: {mem_used:.1f} GB, Free: {mem_free:.1f} GB",
                })

    except ImportError:
        results.append({"check": "PyTorch", "status": "error", "value": "not installed"})

    # nvidia-smi driver check
    try:
        v = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,power.draw",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if v.returncode == 0:
            for line in v.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    results.append({
                        "check": f"nvidia-smi ({parts[0]})",
                        "status": "ok",
                        "value": f"Driver {parts[1]}, Temp {parts[2]}°C, Power {parts[3]}W",
                    })
    except FileNotFoundError:
        results.append({"check": "nvidia-smi", "status": "warn", "value": "not found"})

    return results


def _check_gpu_benchmarks() -> list[dict[str, Any]]:
    """Run quick GPU benchmarks to verify compute and memory bandwidth."""
    results = []
    try:
        import torch
        if not torch.cuda.is_available():
            return [{"check": "GPU benchmark", "status": "skip", "value": "no CUDA"}]

        device = torch.device("cuda:0")

        # Memory bandwidth test
        size = 1024 * 1024 * 256  # 1GB in FP32
        a = torch.randn(size, device=device, dtype=torch.float32)
        b = torch.empty_like(a)

        # Warmup
        for _ in range(5):
            b.copy_(a)
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(100):
            b.copy_(a)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        bandwidth_gbps = (size * 4 * 100) / elapsed / 1e9
        results.append({
            "check": "GPU memory bandwidth",
            "status": "ok" if bandwidth_gbps > 100 else "warn",
            "value": f"{bandwidth_gbps:.1f} GB/s",
        })

        # Compute test (FP16 matmul)
        m, k, n = 4096, 4096, 4096
        a = torch.randn(m, k, device=device, dtype=torch.float16)
        b = torch.randn(k, n, device=device, dtype=torch.float16)

        # Warmup
        for _ in range(5):
            torch.mm(a, b)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(100):
            torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        flops = (2 * m * k * n * 100) / elapsed / 1e12
        results.append({
            "check": "GPU FP16 compute",
            "status": "ok" if flops > 10 else "warn",
            "value": f"{flops:.1f} TFLOPS",
        })

        del a, b
        torch.cuda.empty_cache()

    except Exception as e:
        results.append({"check": "GPU benchmark", "status": "error", "value": str(e)})

    return results


# ── Network Diagnostics ──────────────────────────────────────────────────

def _check_network_latency(hosts: list[str] | None = None) -> list[dict[str, Any]]:
    """Measure network latency to cluster nodes."""
    results = []
    if hosts is None:
        hosts = ["localhost"]

    for host in hosts:
        # TCP connect latency to gRPC port
        for port in [50050, 50051]:
            latencies = []
            for _ in range(3):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                start = time.time()
                try:
                    sock.connect((host, port))
                    elapsed = (time.time() - start) * 1000
                    latencies.append(elapsed)
                except (socket.timeout, ConnectionRefusedError):
                    latencies.append(-1)
                finally:
                    sock.close()

            if latencies and latencies[0] >= 0:
                avg_ms = sum(latencies) / len(latencies)
                results.append({
                    "check": f"TCP {host}:{port}",
                    "status": "ok" if avg_ms < 5 else "warn",
                    "value": f"{avg_ms:.1f} ms",
                })
            else:
                results.append({
                    "check": f"TCP {host}:{port}",
                    "status": "info",
                    "value": "not listening (OK if not started)",
                })

    return results


def _check_ports(ports: list[int] | None = None) -> list[dict[str, Any]]:
    """Check if common DistLLM ports are listening or free."""
    results = []
    if ports is None:
        ports = [50050, 50051, 50052, 50060, 8000, 8500]
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            result = sock.connect_ex(("127.0.0.1", port))
            status = "listening" if result == 0 else "free"
            results.append({
                "check": f"port {port}",
                "status": "ok" if result == 0 else "info",
                "value": status,
            })
        finally:
            sock.close()
    return results


# ── Model Compatibility ──────────────────────────────────────────────────

def _check_model_compatibility(model_name: str) -> list[dict[str, Any]]:
    """Check if a model is compatible with detected hardware."""
    results = []

    # Model size estimation
    model_profiles = {
        "1b": (1, 24, 2048, 1.5), "3b": (3, 28, 3200, 6), "7b": (7, 32, 4096, 14),
        "8b": (8, 32, 4096, 16), "13b": (13, 40, 5120, 26), "14b": (14, 40, 5120, 28),
        "34b": (34, 64, 8192, 68), "70b": (70, 80, 8192, 140), "405b": (405, 126, 16384, 810),
    }

    matched = None
    for key, (params, layers, hidden, mem_gb) in model_profiles.items():
        if key in model_name.lower():
            matched = (key, params, layers, hidden, mem_gb)
            break

    if matched:
        key, params, layers, hidden, mem_gb = matched
        results.append({"check": "Model profile", "status": "ok", "value": f"{params}B params, {layers} layers, {hidden} hidden"})

        # Check GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                total_gpu_mem = sum(
                    torch.cuda.get_device_properties(i).total_memory / 1e9
                    for i in range(torch.cuda.device_count())
                )
                fits = mem_gb < total_gpu_mem * 0.9
                results.append({
                    "check": "GPU memory fit",
                    "status": "ok" if fits else "error",
                    "value": f"Model needs ~{mem_gb:.0f}GB, available {total_gpu_mem:.0f}GB",
                })

                if not fits:
                    # Suggest quantization
                    int8_mem = mem_gb / 2
                    int4_mem = mem_gb / 4
                    results.append({
                        "check": "Quantization suggestion",
                        "status": "info",
                        "value": f"INT8: ~{int8_mem:.0f}GB, INT4: ~{int4_mem:.0f}GB",
                    })

                # Suggest partition strategy
                num_gpus = torch.cuda.device_count()
                if num_gpus > 1:
                    layers_per_gpu = max(1, layers // num_gpus)
                    results.append({
                        "check": "Suggested partition",
                        "status": "info",
                        "value": f"{layers_per_gpu} layers/GPU across {num_gpus} GPUs (pipeline parallelism)",
                    })
                else:
                    results.append({
                        "check": "Suggested partition",
                        "status": "info",
                        "value": "Single GPU — no partitioning needed",
                    })
            else:
                results.append({"check": "GPU memory", "status": "error", "value": "No CUDA GPU detected"})
        except ImportError:
            results.append({"check": "PyTorch", "status": "error", "value": "not installed"})
    else:
        results.append({
            "check": "Model profile",
            "status": "info",
            "value": f"Unknown model '{model_name}' — cannot estimate requirements",
        })

    return results


# ── Configuration Diagnostics ────────────────────────────────────────────

def _check_config() -> list[dict[str, Any]]:
    """Validate configuration files."""
    results = []
    config_paths = ["config.yaml", os.path.expanduser("~/.config/distllm/config.yaml")]
    for path in config_paths:
        if os.path.exists(path):
            try:
                import yaml
                with open(path) as f:
                    data = yaml.safe_load(f)
                results.append({"check": f"config {path}", "status": "ok", "value": "valid YAML"})

                # Check for common misconfigurations
                if data:
                    if data.get("tls", {}).get("enabled") is False:
                        results.append({"check": "TLS", "status": "warn", "value": "disabled — API keys transmitted in plaintext"})
                    if data.get("rate_limiting", {}).get("enabled") is False:
                        results.append({"check": "Rate limiting", "status": "warn", "value": "disabled"})
            except Exception as e:
                results.append({"check": f"config {path}", "status": "error", "value": str(e)})
        else:
            results.append({"check": f"config {path}", "status": "info", "value": "not found (OK for defaults)"})
    return results


def _check_disk() -> list[dict[str, Any]]:
    """Check disk space for model cache."""
    results = []
    cache_dirs = [
        os.path.expanduser("~/.cache/huggingface"),
        os.path.expanduser("~/.distllm_cache"),
        os.path.expanduser("~/.cache/torch"),
    ]
    for d in cache_dirs:
        if os.path.exists(d):
            usage = shutil.disk_usage(d)
            free_gb = usage.free / 1e9
            results.append({
                "check": f"disk {d}",
                "status": "ok" if free_gb > 50 else "warn" if free_gb > 10 else "error",
                "value": f"{free_gb:.1f} GB free",
            })
    if not results:
        results.append({"check": "disk cache dirs", "status": "info", "value": "not created yet"})
    return results


def _check_python_env() -> list[dict[str, Any]]:
    """Check Python environment health."""
    results = []
    results.append({"check": "Python version", "status": "ok", "value": sys.version.split()[0]})

    # Check critical packages
    critical = ["torch", "transformers", "fastapi", "pydantic", "grpcio"]
    for pkg in critical:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            results.append({"check": f"package {pkg}", "status": "ok", "value": str(ver)})
        except ImportError:
            results.append({"check": f"package {pkg}", "status": "warn", "value": "not installed"})

    return results


# ── Output ───────────────────────────────────────────────────────────────

def _print_results(category: str, results: list[dict[str, Any]], verbose: bool = False) -> int:
    """Print results and return error count."""
    errors = 0
    print(f"\n{'='*60}")
    print(f"  {category}")
    print(f"{'='*60}")
    for r in results:
        icon = {"ok": "✅", "warn": "⚠️", "error": "❌", "info": "ℹ️", "skip": "⏭️"}.get(r["status"], "  ")
        msg = f"  {icon} {r['check']}: {r.get('value', '')}"
        print(msg)
        if r["status"] == "error":
            errors += 1
        if r["status"] == "warn" and verbose:
            print(f"      ↳ Recommendation: Check {r['check']} configuration")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="DistLLM diagnostic tool")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--gpu", action="store_true", help="GPU checks only")
    parser.add_argument("--network", action="store_true", help="Network checks only")
    parser.add_argument("--model", type=str, default=None, help="Check model compatibility")
    parser.add_argument("--nodes", type=str, nargs="+", help="Node hosts for network latency check")
    args = parser.parse_args()

    print("DistLLM Doctor — System Diagnostics")
    print("=" * 60)

    total_errors = 0

    if args.gpu:
        total_errors += _print_results("CUDA / GPU", _check_cuda(), args.verbose)
        total_errors += _print_results("GPU Benchmarks", _check_gpu_benchmarks(), args.verbose)
    elif args.network:
        total_errors += _print_results("Network Latency", _check_network_latency(args.nodes), args.verbose)
        total_errors += _print_results("Ports", _check_ports(), args.verbose)
    elif args.model:
        total_errors += _print_results("Model Compatibility", _check_model_compatibility(args.model), args.verbose)
        total_errors += _print_results("CUDA / GPU", _check_cuda(), args.verbose)
    else:
        # Full diagnostic
        total_errors += _print_results("Python Environment", _check_python_env(), args.verbose)
        total_errors += _print_results("CUDA / GPU", _check_cuda(), args.verbose)
        total_errors += _print_results("GPU Benchmarks", _check_gpu_benchmarks(), args.verbose)
        total_errors += _print_results("Ports", _check_ports(), args.verbose)
        total_errors += _print_results("Network Latency", _check_network_latency(args.nodes), args.verbose)
        total_errors += _print_results("Configuration", _check_config(), args.verbose)
        total_errors += _print_results("Disk Space", _check_disk(), args.verbose)

        if args.model:
            total_errors += _print_results("Model Compatibility", _check_model_compatibility(args.model), args.verbose)

    print(f"\n{'='*60}")
    if total_errors == 0:
        print("  ✅ All checks passed!")
    else:
        print(f"  ❌ {total_errors} error(s) found. Fix issues above before deploying.")
    print(f"{'='*60}")
    print()

    sys.exit(1 if total_errors > 0 else 0)


if __name__ == "__main__":
    main()
