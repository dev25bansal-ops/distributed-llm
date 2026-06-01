"""GPU Resource Manager — tracks VRAM, manages allocations, prevents OOM.

Proactively monitors GPU memory across devices (CUDA, ROCm, MPS, XPU),
reserves space for model loading, triggers eviction when memory pressure
is high, and provides a unified allocation API that all model loaders
use to avoid OOM.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


class MemoryPriority(Enum):
    """Priority level for GPU memory allocations."""
    CRITICAL = 0   # Must succeed (e.g. active model weights)
    HIGH = 1       # Should succeed (e.g. KV cache)
    NORMAL = 2     # Best effort (e.g. prefetch, secondary cache)
    LOW = 3        # Evict first (e.g. unused adapter weights)


@dataclass
class Allocation:
    """A tracked GPU memory allocation."""
    key: str
    size_mb: float
    device: int
    priority: MemoryPriority
    allocated_at: float = 0.0
    last_used: float = 0.0
    owner: str = ""


@dataclass
class GPUMemorySnapshot:
    """Point-in-time snapshot of GPU memory state."""
    device: int
    total_mb: float
    used_mb: float
    free_mb: float
    reserved_mb: float
    allocated_mb: float
    utilization_pct: float
    temperature_c: float
    allocations: list[Allocation] = field(default_factory=list)


class GPUResourceManager:
    """Central GPU memory manager — tracks, allocates, evicts.

    Singleton-like usage (one instance per process)::

        mgr = GPUResourceManager()
        mgr.register_device(0)
        ok, alloc = mgr.try_allocate("model-falcon-7b", 14000, device=0)
        mgr.release("model-falcon-7b")
        print(mgr.snapshot(0))
    """

    SAFETY_MARGIN_MB: float = 512.0

    def __init__(self, safety_margin_mb: float = SAFETY_MARGIN_MB) -> None:
        self._safety_margin = safety_margin_mb
        self._allocations: dict[str, Allocation] = {}
        self._devices: set[int] = set()
        self._total_per_device: dict[int, float] = {}
        self._eviction_callbacks: list[Any] = []

    # ── Device management ───────────────────────────────────────────────

    def register_device(self, device: int) -> None:
        """Register a GPU device for tracking (CUDA, ROCm, MPS, XPU)."""
        import torch
        from distllm.core.device_registry import detect_platform
        plat = detect_platform()

        try:
            if plat == "cuda" or plat == "rocm":
                total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
            elif plat == "xpu":
                total = torch.xpu.get_device_properties(device).total_memory / (1024 ** 2)
            elif plat == "mps":
                from distllm.core.device_registry import _get_mps_memory
                total = _get_mps_memory() / (1024 ** 2)
            else:
                logger.warning(f"No GPU detected, cannot register device {device}")
                return
            self._devices.add(device)
            self._total_per_device[device] = total
            logger.info(f"Registered {plat.upper()} device {device}: {total:.0f} MB total")
        except Exception as e:
            logger.error(f"Failed to register device {device} ({plat}): {e}")

    def register_device_nvml(self, device: int) -> None:
        """Register GPU via NVML (NVIDIA only, falls back to generic)."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total = info.total / (1024 ** 2)
            self._devices.add(device)
            self._total_per_device[device] = total
        except ImportError:
            self.register_device(device)
        except Exception:
            self.register_device(device)

    def unregister_device(self, device: int) -> None:
        """Remove a device from tracking."""
        self._devices.discard(device)
        self._total_per_device.pop(device, None)
        to_remove = [k for k, a in self._allocations.items() if a.device == device]
        for k in to_remove:
            self._allocations.pop(k, None)

    # ── Allocation ──────────────────────────────────────────────────────

    def try_allocate(
        self,
        key: str,
        size_mb: float,
        device: int = 0,
        priority: MemoryPriority = MemoryPriority.NORMAL,
        owner: str = "",
    ) -> tuple[bool, Allocation | None]:
        """Try to allocate *size_mb* on *device*.

        Returns ``(success, allocation)``.  If insufficient memory is
        available, triggers eviction of lower-priority allocations and
        retries once.
        """
        free = self._free_mb(device)
        headroom = free - size_mb - self._safety_margin

        if headroom >= 0:
            alloc = Allocation(
                key=key, size_mb=size_mb, device=device,
                priority=priority, allocated_at=time.time(),
                last_used=time.time(), owner=owner,
            )
            self._allocations[key] = alloc
            logger.debug(f"Allocated {size_mb:.0f} MB on GPU {device} for {key}")
            return True, alloc

        # Try evicting lower-priority allocations
        freed = self._evict(needed=size_mb + self._safety_margin, device=device,
                            min_priority=priority)
        if freed >= size_mb + self._safety_margin:
            alloc = Allocation(
                key=key, size_mb=size_mb, device=device,
                priority=priority, allocated_at=time.time(),
                last_used=time.time(), owner=owner,
            )
            self._allocations[key] = alloc
            logger.info(f"Allocated {size_mb:.0f} MB on GPU {device} for {key} "
                        f"(after evicting {freed:.0f} MB)")
            return True, alloc

        logger.warning(
            f"Failed to allocate {size_mb:.0f} MB on GPU {device} for {key} — "
            f"only {free:.0f} MB free (need {size_mb + self._safety_margin:.0f})"
        )
        return False, None

    def release(self, key: str) -> bool:
        """Release a previously allocated resource."""
        alloc = self._allocations.pop(key, None)
        if alloc is None:
            return False
        logger.debug(f"Released {alloc.size_mb:.0f} MB on GPU {alloc.device} for {key}")
        return True

    def touch(self, key: str) -> None:
        """Mark a tracked allocation as recently used (for LRU eviction)."""
        alloc = self._allocations.get(key)
        if alloc:
            alloc.last_used = time.time()

    # ── Queries ─────────────────────────────────────────────────────────

    def snapshot(self, device: int = 0) -> GPUMemorySnapshot | None:
        """Return a point-in-time snapshot for *device* (any platform)."""
        import torch
        from distllm.core.device_registry import detect_platform
        plat = detect_platform()

        if device not in self._devices:
            return None
        try:
            if plat == "cuda" or plat == "rocm":
                device_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
                device_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
                if self._total_per_device.get(device, 0.0) == 0:
                    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
                else:
                    total = self._total_per_device[device]
            elif plat == "xpu":
                device_alloc = torch.xpu.memory_allocated(device) / (1024 ** 2)
                device_reserved = 0.0
                total = self._total_per_device.get(device, torch.xpu.get_device_properties(device).total_memory / (1024 ** 2))
            elif plat == "mps":
                alloc_bytes = getattr(torch.mps, "current_allocated_memory", lambda: 0)()
                device_alloc = alloc_bytes / (1024 ** 2)
                device_reserved = 0.0
                total = self._total_per_device.get(device, 8192)
            else:
                return None

            util_pct = 0.0
            temp_c = 0.0
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                util_pct = util.gpu
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                temp_c = float(temp)
            except Exception:
                pass
            device_allocs = [a for a in self._allocations.values() if a.device == device]
            tracked_mb = sum(a.size_mb for a in device_allocs)
            return GPUMemorySnapshot(
                device=device, total_mb=total, used_mb=total - device_alloc,
                free_mb=device_alloc, reserved_mb=device_reserved,
                allocated_mb=tracked_mb, utilization_pct=util_pct,
                temperature_c=temp_c, allocations=device_allocs,
            )
        except Exception:
            return None

    def available_mb(self, device: int = 0) -> float:
        """Return free memory in MB, accounting for safety margin."""
        free = self._free_mb(device)
        return max(0.0, free - self._safety_margin)

    def total_mb(self, device: int = 0) -> float:
        return self._total_per_device.get(device, 0.0)

    def is_oom_risk(self, device: int = 0, threshold_pct: float = 90.0) -> bool:
        """Check if device memory usage exceeds threshold."""
        total = self.total_mb(device)
        if total == 0:
            return False
        used = total - self._free_mb(device)
        return (used / total * 100) >= threshold_pct

    def on_eviction(self, callback: Any) -> None:
        """Register a callback for eviction notifications.

        ``callback(key: str, allocation: Allocation)`` is called when
        an allocation is evicted.
        """
        self._eviction_callbacks.append(callback)

    # ── Internal helpers ────────────────────────────────────────────────

    def _free_mb(self, device: int) -> float:
        import torch
        from distllm.core.device_registry import detect_platform
        plat = detect_platform()
        try:
            if plat == "cuda" or plat == "rocm":
                alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
                total = self._total_per_device.get(device, 0.0)
                if total == 0:
                    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
            elif plat == "xpu":
                alloc = torch.xpu.memory_allocated(device) / (1024 ** 2)
                total = self._total_per_device.get(device, 0.0)
                if total == 0:
                    total = torch.xpu.get_device_properties(device).total_memory / (1024 ** 2)
            elif plat == "mps":
                alloc = getattr(torch.mps, "current_allocated_memory", lambda: 0)() / (1024 ** 2)
                total = self._total_per_device.get(device, 8192)
            else:
                return 0.0
            return max(0.0, total - alloc)
        except Exception:
            return 0.0

    def _evict(
        self, needed: float, device: int,
        min_priority: MemoryPriority = MemoryPriority.NORMAL,
    ) -> float:
        """Evict lower-priority allocations on *device* to free *needed* MB.

        Returns total MB freed.
        """
        candidates = sorted(
            [a for a in self._allocations.values()
             if a.device == device and a.priority.value > min_priority.value],
            key=lambda a: (a.priority.value, a.last_used),
        )
        freed = 0.0
        for alloc in candidates:
            if freed >= needed:
                break
            self._allocations.pop(alloc.key, None)
            freed += alloc.size_mb
            for cb in self._eviction_callbacks:
                try:
                    cb(alloc.key, alloc)
                except Exception:
                    pass
            logger.info(f"Evicted {alloc.key} ({alloc.size_mb:.0f} MB) from GPU {device}")
        return freed


# Module-level singleton (lazy init) for global access
_global_mgr: GPUResourceManager | None = None
_global_mgr_lock = threading.Lock()


def get_gpu_resource_manager() -> GPUResourceManager:
    """Return the global GPU resource manager instance (platform-aware)."""
    global _global_mgr
    if _global_mgr is None:
        with _global_mgr_lock:
            if _global_mgr is None:
                _global_mgr = GPUResourceManager()
                from distllm.core.device_registry import detect_platform, detect_all_devices
                plat = detect_platform()
                if plat != "cpu":
                    devices = detect_all_devices()
                    for dev in devices:
                        try:
                            _global_mgr.register_device(dev.device_id)
                        except Exception:
                            continue
    return _global_mgr
