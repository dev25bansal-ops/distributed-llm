"""Universal device detection for heterogeneous GPU clusters.

Detects and reports capabilities of all available compute devices
across platforms: NVIDIA CUDA, AMD ROCm, Apple Metal (MPS), Intel XPU,
and CPU fallback. Used by the coordinator to build a heterogeneous
device map for pipeline scheduling.
"""

from __future__ import annotations

import os
import platform as _platform
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

from distllm.constants import (
    Device,
    DeviceFamily,
    DEVICE_FAMILY,
    INTEL_XPU_BANDWIDTH_GBPS,
    MPS_DEFAULT_MEMORY_BYTES,
    MPS_DEFAULT_GPU_CORES,
    MPS_DEFAULT_TFLOPS_FP16,
)


class PlatformType(str, Enum):
    CUDA = "cuda"
    ROCM = "rocm"
    MPS = "mps"
    XPU = "xpu"
    CPU = "cpu"


@dataclass
class DeviceInfo:
    """Capability information for a single compute device."""
    device_type: str
    device_family: DeviceFamily
    device_id: int
    name: str = ""
    total_memory_bytes: int = 0
    free_memory_bytes: int = 0
    compute_capability: str = ""
    sm_count: int = 0
    tflops_fp16: float = 0.0
    tflops_fp32: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    is_available: bool = True


def detect_platform() -> str:
    """Detect the primary compute platform available on this machine."""
    # Check ROCm first (torch.cuda.is_available is True on ROCm too)
    if _is_rocm_available():
        return "rocm"
    if _is_cuda_available():
        return "cuda"
    if _is_mps_available():
        return "mps"
    if _is_xpu_available():
        return "xpu"
    return "cpu"


def detect_all_devices() -> list[DeviceInfo]:
    """Probe all available compute devices across all platforms."""
    devices: list[DeviceInfo] = []

    if _is_cuda_available() or _is_rocm_available():
        import torch
        platform_type = "rocm" if _is_rocm_available() else "cuda"
        count = torch.cuda.device_count()
        for i in range(count):
            try:
                props = torch.cuda.get_device_properties(i)
                alloc = torch.cuda.memory_allocated(i)
                total = props.total_memory
                free = total - alloc
                cc = f"{props.major}.{props.minor}"
                dev_info = DeviceInfo(
                    device_type=platform_type,
                    device_family=DEVICE_FAMILY.get(platform_type, DeviceFamily.UNKNOWN),
                    device_id=i,
                    name=props.name,
                    total_memory_bytes=total,
                    free_memory_bytes=free,
                    compute_capability=cc,
                    sm_count=getattr(props, "multi_processor_count", 0),
                    tflops_fp16=_estimate_fp16_tflops(props),
                tflops_fp32=_estimate_fp32_tflops(props),
                )
                devices.append(dev_info)
            except Exception as e:
                logger.debug(f"Failed to probe device {i}: {e}")

    if _is_mps_available() and not devices:
        import torch
        total = _get_mps_memory()
        dev_info = DeviceInfo(
            device_type="mps",
            device_family=DeviceFamily.APPLE,
            device_id=0,
            name=f"Apple {_platform.machine()}",
            total_memory_bytes=total,
            free_memory_bytes=int(total * 0.8),
            compute_capability="",
            sm_count=_apple_gpu_cores(),
            tflops_fp16=_mps_fp16_tflops(),
            tflops_fp32=_mps_fp32_tflops(),
        )
        devices.append(dev_info)

    if _is_xpu_available() and not devices:
        import torch
        count = torch.xpu.device_count()
        for i in range(count):
            try:
                props = torch.xpu.get_device_properties(i)
                dev_info = DeviceInfo(
                    device_type="xpu",
                    device_family=DeviceFamily.INTEL,
                    device_id=i,
                    name=props.name,
                    total_memory_bytes=props.total_memory,
                    memory_bandwidth_gbps=INTEL_XPU_BANDWIDTH_GBPS,
                )
                devices.append(dev_info)
            except Exception as e:
                logger.debug(f"Failed to probe XPU device {i}: {e}")

    if not devices:
        import psutil
        dev_info = DeviceInfo(
            device_type="cpu",
            device_family=DeviceFamily.CPU,
            device_id=0,
            name=f"{_platform.processor() or 'CPU'}",
            total_memory_bytes=psutil.virtual_memory().total,
            free_memory_bytes=psutil.virtual_memory().available,
            sm_count=psutil.cpu_count(logical=True),
        )
        devices.append(dev_info)

    return devices


def _is_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and not _is_rocm_available()
    except Exception as e:
        logger.debug(f"CUDA detection failed: {e}")
        return False


def _is_rocm_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and torch.version.hip is not None
    except Exception as e:
        logger.debug(f"ROCm detection failed: {e}")
        return False


def _is_mps_available() -> bool:
    try:
        import torch
        return hasattr(torch, "mps") and torch.backends.mps.is_available()
    except Exception as e:
        logger.debug(f"MPS detection failed: {e}")
        return False


def _is_xpu_available() -> bool:
    try:
        import torch
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    except Exception as e:
        logger.debug(f"XPU detection failed: {e}")
        return False


def _get_mps_memory() -> int:
    try:
        import torch
        if hasattr(torch.mps, "current_allocated_memory"):
            return int(torch.mps.recommended_max_memory() or 8 * 1024**3)
    except Exception as e:
        logger.debug(f"MPS memory query failed: {e}")
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "hw.memsize"], capture_output=True, text=True, timeout=2
        )
        return int(result.stdout.split(":")[1].strip())
    except Exception as e:
        logger.debug(f"sysctl hw.memsize failed: {e}")
        return MPS_DEFAULT_MEMORY_BYTES


def _apple_gpu_cores() -> int:
    """Return estimated Apple GPU core count based on chip name."""
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "hw.machine"], capture_output=True, text=True, timeout=2
        )
        chip = result.stdout.strip().split(":")[-1].strip().lower()
        cores = {
            "m1": 7, "m1p": 16, "m1max": 32, "m1ultra": 64,
            "m2": 10, "m2p": 19, "m2max": 38, "m2ultra": 76,
            "m3": 10, "m3p": 18, "m3max": 40, "m3ultra": 80,
            "m4": 10, "m4p": 20, "m4max": 40,
        }
        for key, core_count in cores.items():
            if key in chip.replace(" ", "").replace("-", ""):
                return core_count
    except Exception as e:
        logger.debug(f"Apple GPU core detection failed: {e}")
    return MPS_DEFAULT_GPU_CORES


def _mps_fp16_tflops() -> float:
    """Estimated Apple M-series FP16 TFLOPS."""
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "hw.machine"], capture_output=True, text=True, timeout=2
        )
        chip = result.stdout.strip().split(":")[-1].strip().lower().replace(" ", "").replace("-", "")
        tflops_map = {
            "m1": 2.6, "m1p": 5.3, "m1max": 10.6, "m1ultra": 21.2,
            "m2": 3.6, "m2p": 6.8, "m2max": 13.6, "m2ultra": 27.2,
            "m3": 4.1, "m3p": 7.4, "m3max": 14.8, "m3ultra": 29.6,
            "m4": 4.6, "m4p": 9.2, "m4max": 18.4,
        }
        for key, val in tflops_map.items():
            if key in chip:
                return val
    except Exception as e:
        logger.debug(f"Apple FP16 TFLOPS detection failed: {e}")
    return MPS_DEFAULT_TFLOPS_FP16


def _mps_fp32_tflops() -> float:
    t = _mps_fp16_tflops()
    return t / 2.0


def _estimate_fp16_tflops(props: Any) -> float:
    sm = getattr(props, "multi_processor_count", 1)
    clock = getattr(props, "clock_rate", 1000)
    return round(sm * clock * 2 * 128 * 2 / 1e6, 2)


def _estimate_fp32_tflops(props: Any) -> float:
    sm = getattr(props, "multi_processor_count", 1)
    clock = getattr(props, "clock_rate", 1000)
    return round(sm * clock * 128 * 2 / 1e6, 2)


def get_device_family(device_type: str) -> DeviceFamily:
    """Map a device type string to its DeviceFamily."""
    return DEVICE_FAMILY.get(device_type, DeviceFamily.UNKNOWN)


def get_backend_priority(device_family: DeviceFamily, backend_name: str) -> int:
    """Get the priority of a backend for a given device family."""
    from distllm.constants import PLATFORM_BACKEND_PRIORITY
    family_priorities = PLATFORM_BACKEND_PRIORITY.get(device_family.value, {})
    return family_priorities.get(backend_name, 1)


def format_device_summary(devices: list[DeviceInfo]) -> str:
    """Return a human-readable summary of all detected devices."""
    lines = []
    for d in devices:
        mem_gb = d.total_memory_bytes / (1024**3)
        lines.append(
            f"  [{d.device_id}] {d.device_type.upper()} {d.name} "
            f"({d.device_family.value}) — {mem_gb:.1f} GB"
        )
    if not lines:
        lines.append("  No devices detected")
    return "Devices:\n" + "\n".join(lines)
