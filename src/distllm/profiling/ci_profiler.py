"""Memory profiler for CI — detects regressions and leaks.

Uses tracemalloc (CPU) and pynvml (GPU) to snapshot memory before/after
operations. Detects leaks via linear regression over multiple iterations.
"""

import json
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from loguru import logger

try:
    import pynvml
    HAS_GPU = True
except ImportError:
    HAS_GPU = False


@dataclass
class MemorySnapshot:
    """Point-in-time memory usage."""
    timestamp: float = 0.0
    cpu_current_mb: float = 0.0
    cpu_peak_mb: float = 0.0
    gpu_used_mb: float = 0.0
    gpu_total_mb: float = 0.0
    label: str = ""


@dataclass
class MemoryReport:
    """Full memory profiling report."""
    operation: str = ""
    snapshots: List[MemorySnapshot] = field(default_factory=list)
    iterations: int = 0
    leak_detected: bool = False
    leak_slope_mb_per_iter: float = 0.0
    leak_r_squared: float = 0.0
    budget_exceeded: bool = False
    budget_mb: float = 0.0
    actual_peak_mb: float = 0.0
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "iterations": self.iterations,
            "leak_detected": self.leak_detected,
            "leak_slope_mb_per_iter": round(self.leak_slope_mb_per_iter, 4),
            "leak_r_squared": round(self.leak_r_squared, 4),
            "budget_exceeded": self.budget_exceeded,
            "budget_mb": self.budget_mb,
            "actual_peak_mb": round(self.actual_peak_mb, 2),
            "duration_s": round(self.duration_s, 3),
            "snapshots": [
                {
                    "label": s.label,
                    "cpu_current_mb": round(s.cpu_current_mb, 2),
                    "cpu_peak_mb": round(s.cpu_peak_mb, 2),
                    "gpu_used_mb": round(s.gpu_used_mb, 2),
                    "gpu_total_mb": round(s.gpu_total_mb, 2),
                }
                for s in self.snapshots
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class LeakDetector:
    """Detects memory leaks via linear regression.

    Fits a line to per-iteration peak memory values.
    If slope > threshold and R² > threshold, a leak is flagged.
    """

    @staticmethod
    def detect(
        values: List[float],
        slope_threshold: float = 1.0,  # MB per iteration
        r_squared_threshold: float = 0.8,
    ) -> tuple:
        """Run leak detection on a list of per-iteration memory values.

        Returns:
            (leak_detected, slope_mb_per_iter, r_squared)
        """
        if len(values) < 3:
            return False, 0.0, 0.0

        n = len(values)
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n

        ss_xy = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        ss_xx = sum((i - x_mean) ** 2 for i in range(n))
        ss_yy = sum((v - y_mean) ** 2 for v in values)

        if ss_xx == 0:
            return False, 0.0, 0.0

        slope = ss_xy / ss_xx  # MB per iteration

        if ss_yy == 0:
            return False, slope, 0.0

        r_squared = (ss_xy ** 2) / (ss_xx * ss_yy)

        leak_detected = slope > slope_threshold and r_squared > r_squared_threshold
        return leak_detected, slope, r_squared


class MemoryProfiler:
    """Profiles memory usage of operations.

    Usage:
        profiler = MemoryProfiler()
        report = profiler.profile("my_operation", my_fn, iterations=5)
        report.save("report.json")
    """

    def __init__(self, track_gpu: bool = True):
        self.track_gpu = track_gpu and HAS_GPU
        if self.track_gpu:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self.track_gpu = False

    def snapshot(self, label: str = "") -> MemorySnapshot:
        """Take a memory snapshot."""
        current, peak = tracemalloc.get_traced_memory()
        snap = MemorySnapshot(
            timestamp=time.time(),
            cpu_current_mb=current / (1024 * 1024),
            cpu_peak_mb=peak / (1024 * 1024),
            label=label,
        )
        if self.track_gpu:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                snap.gpu_used_mb = info.used / (1024 * 1024)
                snap.gpu_total_mb = info.total / (1024 * 1024)
            except Exception:
                pass
        return snap

    def profile(
        self,
        operation_name: str,
        fn: Callable,
        iterations: int = 5,
        budget_mb: Optional[float] = None,
    ) -> MemoryReport:
        """Profile an operation over multiple iterations.

        Args:
            operation_name: Human-readable name for the report.
            fn: Callable to profile (called once per iteration).
            iterations: Number of times to run the operation.
            budget_mb: Maximum allowed peak memory (None = no budget check).

        Returns:
            MemoryReport with snapshots and leak analysis.
        """
        tracemalloc.start()
        report = MemoryReport(operation=operation_name, iterations=iterations)
        peak_values = []
        start_time = time.time()

        for i in range(iterations):
            # Pre-operation snapshot
            pre = self.snapshot(f"iter_{i}_pre")
            report.snapshots.append(pre)

            # Run operation
            fn()

            # Post-operation snapshot
            post = self.snapshot(f"iter_{i}_post")
            report.snapshots.append(post)
            peak_values.append(post.cpu_peak_mb)

            # Reset tracemalloc between iterations
            tracemalloc.reset_peak()

        report.duration_s = time.time() - start_time
        report.actual_peak_mb = max(peak_values) if peak_values else 0.0

        # Leak detection
        leak, slope, r_sq = LeakDetector.detect(peak_values)
        report.leak_detected = leak
        report.leak_slope_mb_per_iter = slope
        report.leak_r_squared = r_sq

        # Budget check
        if budget_mb is not None:
            report.budget_mb = budget_mb
            report.budget_exceeded = report.actual_peak_mb > budget_mb

        tracemalloc.stop()
        return report

    def profile_batch(self, operations: Dict[str, Callable]) -> List[MemoryReport]:
        """Profile multiple operations sequentially.

        Args:
            operations: Dict mapping name -> callable.

        Returns:
            List of MemoryReports.
        """
        return [self.profile(name, fn) for name, fn in operations.items()]

    @staticmethod
    def save_reports(reports: List[MemoryReport], output_path: str) -> None:
        """Save multiple reports to a JSON file."""
        data = {
            "reports": [r.to_dict() for r in reports],
            "summary": {
                "total_operations": len(reports),
                "leaks_detected": sum(1 for r in reports if r.leak_detected),
                "budgets_exceeded": sum(1 for r in reports if r.budget_exceeded),
            },
        }
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved memory profile to {output_path}")
