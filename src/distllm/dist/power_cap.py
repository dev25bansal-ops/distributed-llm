"""GPU power capping — dynamically adjust GPU power limits to reduce cloud costs.

Uses ``nvidia-smi`` to set per-GPU power limits and monitors performance
impact to find the optimal power-performance trade-off.  Can reduce power
consumption by 30-50% with minimal throughput regression (typically <5%)
by exploiting the non-linear relationship between power and clock speed.

The coordinator's health check endpoint can report current power cap
and recommend adjustments based on measured throughput impact.

Usage::

    capper = GPUPowerCapper()
    capper.auto_tune()         # Find optimal power limit via binary search
    capper.set_power_limit(200)  # Set to 200W
    print(capper.get_status())   # Current power, cap, savings
"""

from __future__ import annotations

import math
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class PowerStatus:
    """Current GPU power state."""
    gpu_index: int
    current_power_w: float = 0.0
    power_limit_w: float = 0.0
    min_power_limit_w: float = 0.0
    max_power_limit_w: float = 0.0
    default_power_limit_w: float = 0.0
    gpu_util_pct: float = 0.0
    memory_util_pct: float = 0.0
    temperature_c: float = 0.0

    @property
    def power_savings_pct(self) -> float:
        if self.default_power_limit_w <= 0:
            return 0.0
        return round(
            (1.0 - self.power_limit_w / self.default_power_limit_w) * 100, 1
        )


class GPUPowerCapper:
    """Manages GPU power limits for cost savings.

    Uses ``nvidia-smi`` for NVIDIA GPUs.  No-op on non-NVIDIA hardware
    (AMD, Apple, CPU).

    Thread-safe for concurrent access from health check and auto-tune.
    """

    def __init__(self, gpu_indices: list[int] | None = None):
        self._gpu_indices = gpu_indices or self._detect_gpus()
        self._lock = threading.Lock()
        self._auto_tune_active = False
        self._tuned_limits: dict[int, float] = {}

    # ── Detection ─────────────────────────────────────────────────────

    def _detect_gpus(self) -> list[int]:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return [int(line.strip()) for line in result.stdout.strip().split("\n") if line.strip()]
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
        return []

    @property
    def available(self) -> bool:
        return len(self._gpu_indices) > 0

    # ── Power limit control ───────────────────────────────────────────

    def set_power_limit(self, watts: int, gpu_index: int | None = None) -> bool:
        """Set the power limit for one or all GPUs.

        Args:
            watts: Target power limit in watts.
            gpu_index: Specific GPU index, or ``None`` for all GPUs.

        Returns:
            ``True`` if the limit was applied successfully.
        """
        if not self.available:
            logger.warning("No NVIDIA GPUs detected — power capping unavailable")
            return False

        indices = [gpu_index] if gpu_index is not None else self._gpu_indices
        all_ok = True

        for idx in indices:
            try:
                subprocess.run(
                    ["nvidia-smi", "-i", str(idx), "-pl", str(watts)],
                    capture_output=True, text=True, timeout=5, check=True,
                )
                with self._lock:
                    self._tuned_limits[idx] = float(watts)
                logger.info(f"GPU {idx}: power limit set to {watts}W")
            except subprocess.CalledProcessError as e:
                logger.warning(f"GPU {idx}: failed to set power limit to {watts}W: {e}")
                all_ok = False
            except FileNotFoundError:
                logger.warning("nvidia-smi not found — power capping unavailable")
                return False

        return all_ok

    def reset_to_default(self, gpu_index: int | None = None) -> bool:
        """Reset power limit to the GPU's default."""
        status = self.get_status(gpu_index)
        if gpu_index is not None:
            return self.set_power_limit(int(status.default_power_limit_w), gpu_index)
        all_ok = True
        for idx in self._gpu_indices:
            s = self.get_status(idx)
            if not self.set_power_limit(int(s.default_power_limit_w), idx):
                all_ok = False
        return all_ok

    # ── Auto-tune ────────────────────────────────────────────────────

    def auto_tune(
        self,
        min_power_pct: float = 0.6,
        max_power_pct: float = 0.9,
        step_watts: int = 25,
        benchmark_fn: Any | None = None,
    ) -> dict[int, int]:
        """Find the optimal power limit via binary search.

        Tests power limits from *max_power_pct* down to *min_power_pct*
        of the default limit, using *benchmark_fn* to measure throughput
        at each step.  Selects the lowest power that achieves at least
        95% of peak throughput.

        Args:
            min_power_pct: Minimum power limit as fraction of default.
            max_power_pct: Maximum power limit as fraction of default.
            step_watts: Step size for power adjustment.
            benchmark_fn: Callable returning ``{"tokens_per_sec": float}``.
                When ``None``, uses a simple inference benchmark.

        Returns:
            Dict of ``{gpu_index: optimal_watts}``.
        """
        if not self.available:
            return {}

        self._auto_tune_active = True
        results: dict[int, int] = {}

        for idx in self._gpu_indices:
            status = self.get_status(idx)
            min_w = int(status.default_power_limit_w * min_power_pct)
            max_w = int(status.default_power_limit_w * max_power_pct)
            logger.info(f"Auto-tuning GPU {idx}: {min_w}W — {max_w}W")

            best_watts = int(status.default_power_limit_w)
            best_throughput = 0.0

            for watts in range(max_w, min_w - 1, -step_watts):
                if not self._auto_tune_active:
                    return results
                self.set_power_limit(watts, idx)

                # Allow GPU to stabilise.
                time.sleep(2.0)

                # Measure throughput.
                if benchmark_fn:
                    try:
                        perf = benchmark_fn()
                        throughput = perf.get("tokens_per_sec", 0.0)
                    except Exception as e:
                        logger.warning(f"Benchmark failed at {watts}W: {e}")
                        throughput = 0.0
                else:
                    throughput = self._quick_bench(idx)

                logger.info(
                    f"GPU {idx} @ {watts}W: {throughput:.0f} tok/s"
                )

                if throughput >= best_throughput * 0.95:
                    best_watts = watts
                    best_throughput = max(best_throughput, throughput)

            self.set_power_limit(best_watts, idx)
            results[idx] = best_watts
            logger.info(f"GPU {idx}: optimal power limit = {best_watts}W")

        self._auto_tune_active = False
        return results

    def stop_auto_tune(self) -> None:
        """Safely interrupt an in-progress auto-tune pass."""
        self._auto_tune_active = False

    # ── Quick benchmark ──────────────────────────────────────────────

    @staticmethod
    def _quick_bench(gpu_index: int) -> float:
        """Run a minimal throughput benchmark using PyTorch.

        Measures how many FP16 matmul operations per second the GPU
        can sustain at the current power limit.
        """
        import torch
        if not torch.cuda.is_available():
            return 0.0
        try:
            device = f"cuda:{gpu_index}"
            a = torch.randn(1024, 1024, dtype=torch.float16, device=device)
            b = torch.randn(1024, 1024, dtype=torch.float16, device=device)

            # Warmup.
            for _ in range(10):
                torch.matmul(a, b)

            # Timed.
            torch.cuda.synchronize(device)
            t0 = time.time()
            for _ in range(50):
                torch.matmul(a, b)
            torch.cuda.synchronize(device)
            t1 = time.time()

            elapsed = t1 - t0
            ops = 50 / max(elapsed, 1e-6)
            return round(ops, 1)
        except Exception:
            return 0.0

    # ── Status ───────────────────────────────────────────────────────

    def get_status(self, gpu_index: int | None = None) -> PowerStatus | dict[int, PowerStatus]:
        """Query current power state from ``nvidia-smi``.

        Args:
            gpu_index: Specific GPU index, or ``None`` for all GPUs.

        Returns:
            A single :class:`PowerStatus` or a dict of them.
        """
        if not self.available:
            empty = PowerStatus(gpu_index=0)
            return empty if gpu_index is not None else {0: empty}

        try:
            query = (
                "index,power.draw,power.limit,power.min_limit,"
                "power.max_limit,utilization.gpu,utilization.memory,"
                "temperature.gpu"
            )
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip())

            lines = result.stdout.strip().split("\n")
            all_status: dict[int, PowerStatus] = {}

            for line in lines:
                parts = [p.strip() for p in line.split(", ")]
                if len(parts) < 8:
                    continue
                try:
                    idx = int(parts[0])
                    default_w = self._get_default_power(idx)
                    status = PowerStatus(
                        gpu_index=idx,
                        current_power_w=float(parts[1]),
                        power_limit_w=float(parts[2]),
                        min_power_limit_w=float(parts[3]),
                        max_power_limit_w=float(parts[4]),
                        default_power_limit_w=default_w,
                        gpu_util_pct=float(parts[5]),
                        memory_util_pct=float(parts[6]),
                        temperature_c=float(parts[7]),
                    )
                    all_status[idx] = status
                except (ValueError, IndexError):
                    continue

            if gpu_index is not None:
                return all_status.get(gpu_index, PowerStatus(gpu_index=gpu_index))
            return all_status

        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Failed to query power status: {e}")
            empty = PowerStatus(gpu_index=gpu_index or 0)
            return empty if gpu_index is not None else {0: empty}

    @staticmethod
    def _get_default_power(gpu_index: int) -> float:
        """Get the default (out-of-box) power limit."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", str(gpu_index), "-q", "-d", "POWER"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.split("\n"):
                if "Default Power Limit" in line:
                    match = re.search(r"[\d.]+", line.split(":")[-1])
                    if match:
                        return float(match.group())
        except Exception:
            pass
        return 0.0

    def get_energy_report(self) -> dict[str, Any]:
        """Return aggregate energy metrics across all GPUs."""
        statuses = self.get_status()
        if isinstance(statuses, PowerStatus):
            statuses = {statuses.gpu_index: statuses}

        total_power = sum(s.current_power_w for s in statuses.values())
        total_default = sum(s.default_power_limit_w for s in statuses.values())
        savings = sum(s.power_savings_pct for s in statuses.values()) / max(len(statuses), 1)

        return {
            "gpus": len(statuses),
            "total_current_power_w": round(total_power, 1),
            "total_default_power_w": round(total_default, 1),
            "avg_power_savings_pct": round(savings, 1),
            "gpus_tuned": len(self._tuned_limits),
            "per_gpu": {
                str(idx): {
                    "power_limit_w": s.power_limit_w,
                    "current_w": s.current_power_w,
                    "savings_pct": s.power_savings_pct,
                    "temp_c": s.temperature_c,
                }
                for idx, s in statuses.items()
            },
        }
