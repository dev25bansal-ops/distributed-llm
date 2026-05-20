"""Device type enumeration and specification dataclasses."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any


class DeviceType(str, enum.Enum):
    """Supported device types across all architectures."""
    CUDA = "cuda"       # NVIDIA GPU (CUDA)
    ROCM = "rocm"       # AMD GPU (ROCm)
    MPS = "mps"         # Apple Silicon (Metal Performance Shaders)
    XPU = "xpu"         # Intel GPU (oneAPI/XPU)
    CPU = "cpu"         # CPU-only (no accelerator)
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, s: str) -> DeviceType:
        mapping = {
            "cuda": cls.CUDA, "nvidia": cls.CUDA, "gpu": cls.CUDA,
            "rocm": cls.ROCM, "amd": cls.ROCM,
            "mps": cls.MPS, "metal": cls.MPS, "apple": cls.MPS,
            "xpu": cls.XPU, "intel": cls.XPU, "oneapi": cls.XPU,
            "cpu": cls.CPU,
        }
        return mapping.get(s.lower(), cls.UNKNOWN)


@dataclass
class DeviceSpec:
    """Normalized specification for a single compute device.

    Provides a unified representation across CUDA, ROCm, MPS,
    XPU, and CPU devices.
    """
    device_type: DeviceType
    device_id: int
    name: str = ""
    total_memory_bytes: int = 0
    free_memory_bytes: int = 0
    compute_capability: tuple[int, int] | None = None
    backend: str = ""  # Preferred backend: "pytorch", "vllm", "llamacpp"
    memory_bus_width_bits: int = 0
    sm_count: int = 0
    max_threads_per_sm: int = 0
    clock_rate_mhz: int = 0
    mem_clock_rate_mhz: int = 0
    pci_bandwidth_gbs: float = 0.0
    supports_graph: bool = False

    @property
    def total_memory_gb(self) -> float:
        return round(self.total_memory_bytes / (1024 ** 3), 2)

    @property
    def free_memory_gb(self) -> float:
        return round(self.free_memory_bytes / (1024 ** 3), 2)

    @property
    def is_cuda(self) -> bool:
        return self.device_type == DeviceType.CUDA

    @property
    def is_rocm(self) -> bool:
        return self.device_type == DeviceType.ROCM

    @property
    def is_mps(self) -> bool:
        return self.device_type == DeviceType.MPS

    @property
    def is_xpu(self) -> bool:
        return self.device_type == DeviceType.XPU

    @property
    def is_cpu(self) -> bool:
        return self.device_type == DeviceType.CPU

    @property
    def is_accelerator(self) -> bool:
        return self.device_type in (DeviceType.CUDA, DeviceType.ROCM, DeviceType.MPS, DeviceType.XPU)

    def summary(self) -> str:
        return (
            f"{self.device_type.value}:{self.device_id} "
            f"{self.name} "
            f"mem={self.total_memory_gb}GB "
            f"cc={self._cc_str()} "
            f"backend={self.backend}"
        )

    def _cc_str(self) -> str:
        if self.compute_capability:
            return f"{self.compute_capability[0]}.{self.compute_capability[1]}"
        return "N/A"

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_type": self.device_type.value,
            "device_id": self.device_id,
            "name": self.name,
            "total_memory_gb": self.total_memory_gb,
            "free_memory_gb": self.free_memory_gb,
            "compute_capability": (
                f"{self.compute_capability[0]}.{self.compute_capability[1]}"
                if self.compute_capability else None
            ),
            "backend": self.backend,
            "sm_count": self.sm_count,
            "clock_rate_mhz": self.clock_rate_mhz,
        }


@dataclass
class DeviceCapabilities:
    """Feature capabilities for a device type/architecture.

    Determines which optimizations and parallel strategies are
    available for a given device.
    """
    device_type: DeviceType = DeviceType.UNKNOWN

    # Precision support
    supports_fp16: bool = False
    supports_bf16: bool = False
    supports_fp8: bool = False
    supports_int8: bool = False
    supports_int4: bool = False

    # Attention backends
    supports_flash_attention: bool = False
    supports_paged_attention: bool = False
    supports_sliding_window: bool = False

    # Parallelism
    supports_tensor_parallel: bool = False
    supports_pipeline_parallel: bool = False
    supports_data_parallel: bool = True
    supports_expert_parallel: bool = False
    supports_sequence_parallel: bool = False

    # Graph capture
    supports_cuda_graph: bool = False
    supports_torch_compile: bool = False

    # KV cache
    supports_kv_cache_quantization: bool = False
    supports_kv_cache_prefix_sharing: bool = False

    # Limits
    max_tensor_parallel_size: int = 1
    max_batch_size: int = 0  # 0 = unlimited
    recommended_batch_size: int = 8

    @property
    def preferred_dtype(self) -> str:
        if self.supports_fp8:
            return "fp8"
        if self.supports_bf16:
            return "bf16"
        if self.supports_fp16:
            return "fp16"
        return "fp32"

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_type": self.device_type.value,
            "precisions": self._supported_precisions(),
            "flash_attention": self.supports_flash_attention,
            "paged_attention": self.supports_paged_attention,
            "tensor_parallel": self.supports_tensor_parallel,
            "pipeline_parallel": self.supports_pipeline_parallel,
            "cuda_graph": self.supports_cuda_graph,
            "torch_compile": self.supports_torch_compile,
            "max_tp_size": self.max_tensor_parallel_size,
            "recommended_batch_size": self.recommended_batch_size,
        }

    def _supported_precisions(self) -> list[str]:
        precisions = []
        if self.supports_fp8:
            precisions.append("fp8")
        if self.supports_bf16:
            precisions.append("bf16")
        if self.supports_fp16:
            precisions.append("fp16")
        if self.supports_int8:
            precisions.append("int8")
        if self.supports_int4:
            precisions.append("int4")
        precisions.append("fp32")
        return precisions
