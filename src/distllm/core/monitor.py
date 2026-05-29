"""System monitoring: GPU, CPU, memory, and request metrics."""

import os
import time
import atexit
from typing import Any

from loguru import logger

from distllm.core.batch_scheduler import BatchScheduler


class SystemMonitor:
    """Collects system-level metrics for operational visibility.

    Uses psutil for CPU/memory metrics and pynvml for GPU metrics.
    Gracefully degrades if GPU is not available.
    """

    def __init__(self):
        self._has_gpu = False
        self._gpu_handle: Any = None
        self._last_metrics: dict[str, Any] = {}
        self._pynvml = None
        self._last_cpu_times = None

        self._init_gpu()

    def _init_gpu(self) -> None:
        """Initialize pynvml for GPU metrics."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._has_gpu = True
            atexit.register(self._shutdown_gpu)
            logger.info("GPU monitoring enabled via pynvml")
        except (ImportError, OSError, Exception) as e:
            self._has_gpu = False
            logger.debug(f"GPU monitoring not available: {e}")

    def _shutdown_gpu(self) -> None:
        """Shut down pynvml to release resources."""
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception as e:
                logger.debug(f"GPU shutdown error: {e}")
            self._pynvml = None

    def __del__(self):
        self._shutdown_gpu()

    def collect(self) -> dict[str, Any]:
        """Collect a snapshot of current system metrics."""
        import psutil

        # Non-blocking CPU percent: compute from delta of CPU times
        # instead of blocking with interval=N.  Returns 0.0 on first call.
        cpu_times = psutil.cpu_times_percent(interval=None, percpu=False)
        cpu_pct = 100.0 - cpu_times.idle
        self._last_cpu_times = cpu_times

        metrics: dict[str, Any] = {
            "timestamp": time.time(),
            "cpu": {
                "percent": round(cpu_pct, 1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_available_mb": psutil.virtual_memory().available / 1024 / 1024,
                "memory_total_mb": psutil.virtual_memory().total / 1024 / 1024,
            },
            "process": {
                "pid": os.getpid(),
                "memory_rss_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                "cpu_percent": psutil.Process().cpu_percent(),
                "threads": psutil.Process().num_threads(),
            },
        }

        if self._has_gpu and self._gpu_handle is not None:
            try:
                import pynvml
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    self._gpu_handle, pynvml.NVML_TEMPERATURE_GPU
                )
                metrics["gpu"] = {
                    "memory_used_mb": mem.used / 1024 / 1024,
                    "memory_total_mb": mem.total / 1024 / 1024,
                    "memory_percent": round(mem.used / mem.total * 100, 1) if mem.total > 0 else 0,
                    "utilization_gpu": util.gpu,
                    "utilization_memory": util.memory,
                    "temperature_c": temp,
                }
            except Exception as e:
                logger.warning(f"Failed to collect GPU metrics: {e}")
                metrics["gpu"] = {"error": str(e)}

        self._last_metrics = metrics
        return metrics

    def get_request_metrics(self, scheduler: Any) -> dict[str, Any]:
        """Request-level metrics from the batch scheduler.

        Args:
            scheduler: BatchScheduler instance.

        Returns:
            Dict with active/pending request counts and utilization.
        """
        stats = scheduler.stats()
        return {
            "active_requests": stats["active_requests"],
            "pending_requests": stats["pending_requests"],
            "batch_utilization": round(
                stats["active_requests"] / stats["max_batch_size"], 3
            ) if stats["max_batch_size"] > 0 else 0,
        }

    def get_scheduler_stats(self, scheduler: BatchScheduler) -> dict[str, Any]:
        """Combined system + scheduler stats."""
        return {
            **self.collect(),
            "scheduler": self.get_request_metrics(scheduler),
        }

    def health_check(self, scheduler: BatchScheduler) -> dict[str, Any]:
        """Return health status for the /health endpoint."""
        metrics = self.collect()
        gpu_ok = True
        if "gpu" in metrics and "memory_percent" in metrics.get("gpu", {}):
            gpu_ok = metrics["gpu"]["memory_percent"] < 95

        return {
            "status": "healthy" if gpu_ok else "degraded",
            "gpu_memory_ok": gpu_ok,
            "active_requests": scheduler.active_count if hasattr(scheduler, 'active_count') else 0,
            "pending_requests": scheduler.pending_count if hasattr(scheduler, 'pending_count') else 0,
        }
