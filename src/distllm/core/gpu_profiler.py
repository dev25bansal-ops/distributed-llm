"""GPU profiler for hardware capability detection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPUInfo:
    """Information about a single GPU device."""

    gpu_id: int
    name: str
    total_memory: int  # bytes
    used_memory: int = 0
    free_memory: int = 0
    utilization: float = 0.0
    compute_tflops: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    sm_count: int = 0


class GPUProfiler:
    """Profiles available GPU hardware.

    Usage::

        profiler = GPUProfiler()
        gpus = profiler.enumerate_gpus()
        for gpu in gpus:
            print(f"{gpu.name}: {gpu.total_memory / 1e9:.1f} GB")
    """

    def enumerate_gpus(self) -> list[GPUInfo]:
        """Return info for all available GPUs."""
        import torch

        gpus = []
        if not torch.cuda.is_available():
            return gpus

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_total = props.total_memory
            mem_used = torch.cuda.memory_allocated(i)
            mem_free = mem_total - mem_used
            util = torch.cuda.utilization(i) if hasattr(torch.cuda, "utilization") else 0.0

            gpus.append(GPUInfo(
                gpu_id=i,
                name=props.name,
                total_memory=mem_total,
                used_memory=mem_used,
                free_memory=mem_free,
                utilization=util,
                sm_count=props.multi_processor_count,
            ))

        return gpus
