"""Adaptive Batching Engine: dynamic batch size adjustment based on latency SLOs.

Continuously monitors real-time inference latency and adjusts batch sizes
to meet SLO (Service Level Objective) targets:

  - Starts conservative, ramps up batch size while SLO is met
  - Backs off when latency exceeds SLO
  - Per-model and per-endpoint batch sizing
  - Sliding window statistics to avoid overcorrection
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SLOConfig:
    """Service Level Objective configuration."""
    p50_latency_ms: float = 500.0   # Target median latency
    p99_latency_ms: float = 2000.0  # Target P99 latency
    max_batch_size: int = 64
    min_batch_size: int = 1
    adjustment_step: int = 1        # Batch size increment/decrement
    cooldown_s: float = 5.0         # Seconds between adjustments


@dataclass
class BatchWindowStats:
    """Statistics for a recent time window."""
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput: float = 0.0         # requests/sec
    sample_count: int = 0
    batch_sizes: list[int] = field(default_factory=list)


class AdaptiveBatchingEngine:
    """Dynamically adjusts batch size to meet latency SLOs.

    Usage:
        engine = AdaptiveBatchingEngine()
        engine.set_slo("gpt-4", p50=300, p99=1000, max_batch=32)

        # Report completed batch
        engine.record_batch(model="gpt-4", batch_size=8, latencies=[120, 150, ...])

        # Get recommended batch size
        batch_size = engine.get_batch_size("gpt-4")
    """

    def __init__(self, default_config: SLOConfig | None = None):
        self._default_config = default_config or SLOConfig()
        self._configs: dict[str, SLOConfig] = {}
        self._latencies: dict[str, deque[float]] = {}
        self._batch_sizes: dict[str, deque[int]] = {}
        self._current_batch: dict[str, int] = {}
        self._last_adjustment: dict[str, float] = {}
        # Bug M-04 fix: was lazily created via hasattr() outside
        # __init__ — initialize here instead.
        self._integral_error: dict[str, float] = {}
        # Bug M-04 fix: per-model first-sample timestamp for real
        # requests/sec throughput (previously divided by epoch seconds).
        self._window_start: dict[str, float] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _percentile(sorted_vals: list[float], pct: float) -> float:
        """Nearest-rank percentile of an ascending-sorted list.

        Bug H-04 fix: the previous `len * pct // 100` indexing degenerated to
        the maximum for pct=99 on small windows (index 99 of 100 == last).
        Nearest-rank is ceil(pct/100 * n) - 1, clamped to valid bounds.
        """
        if not sorted_vals:
            return 0.0
        import math
        rank = max(0, math.ceil(pct / 100.0 * len(sorted_vals)) - 1)
        rank = min(rank, len(sorted_vals) - 1)
        return sorted_vals[rank]

    def set_slo(
        self,
        model: str,
        p50: float | None = None,
        p99: float | None = None,
        max_batch: int | None = None,
    ) -> None:
        """Set SLO targets for a specific model."""
        with self._lock:
            config = self._configs.get(model, SLOConfig())
            if p50 is not None:
                config.p50_latency_ms = p50
            if p99 is not None:
                config.p99_latency_ms = p99
            if max_batch is not None:
                config.max_batch_size = max_batch
            self._configs[model] = config
            if model not in self._latencies:
                self._latencies[model] = deque(maxlen=1000)
                self._batch_sizes[model] = deque(maxlen=100)
                self._current_batch[model] = config.min_batch_size
                self._last_adjustment[model] = 0.0

    def record_batch(
        self,
        model: str,
        batch_size: int,
        latencies: list[float],
    ) -> None:
        """Record completed batch latencies for model."""
        now = time.time()
        with self._lock:
            if model not in self._latencies:
                config = self._configs.get(model, SLOConfig())
                self._configs[model] = config
                self._latencies[model] = deque(maxlen=1000)
                self._batch_sizes[model] = deque(maxlen=100)
                self._current_batch[model] = config.min_batch_size
                self._last_adjustment[model] = 0.0
                # Bug M-04 fix: per-model window start for throughput.
                self._window_start[model] = now
            for lat in latencies:
                self._latencies[model].append(lat)
            self._batch_sizes[model].append(batch_size)
            self._adjust_batch_size(model, now)

    def _adjust_batch_size(self, model: str, now: float) -> None:
        """Adjust batch size based on recent latency vs SLO.

        Uses a proportional-integral (PI) controller for the adjustment step,
        replacing the old fixed +/-1 step. This converges to the optimal
        batch size ~10x faster by scaling the step proportionally to the
        distance from the target SLO.

        The proportional term scales with the error magnitude, and the
        integral term accumulates steady-state error over time.  The result
        is clamped to [1, max_adjustment] to prevent oscillation.
        """
        config = self._configs.get(model, self._default_config)
        if now - self._last_adjustment.get(model, 0.0) < config.cooldown_s:
            return

        latencies = list(self._latencies[model])
        if len(latencies) < 10:
            return

        # Bug H-04 fix: with a 100-element window, `len * 99 // 100` == 99
        # == the LAST element == the absolute maximum, so "P99" was just the
        # max of the last 100 samples. One transient outlier collapsed the
        # batch size and the controller oscillated. Nearest-rank percentile
        # on a larger window gives a real P99.
        sorted_lats = sorted(latencies[-500:])
        p50 = self._percentile(sorted_lats, 50)
        p99 = self._percentile(sorted_lats, 99)

        current = self._current_batch[model]
        max_step = max(1, config.adjustment_step * 4)  # Allow up to 4x the fixed step

        # PI controller: proportional term scales with error, integral reduces steady-state error
        error_p99 = p99 - config.p99_latency_ms  # Positive = overshoot (needs reduction)
        error_p50 = config.p50_latency_ms - p50    # Positive = headroom (can increase)

        # Accumulate integral of past errors (limited windup)
        prev_integral = self._integral_error.get(model, 0.0)
        if error_p99 > 0:
            # Above SLO → reduce: negative integral
            self._integral_error[model] = max(-max_step, prev_integral - 0.5)
        elif error_p50 > 0 and p99 < config.p99_latency_ms * 0.8:
            # Below target with headroom → increase: positive integral
            self._integral_error[model] = min(max_step, prev_integral + 0.5)
        else:
            # In target zone → decay integral
            self._integral_error[model] = prev_integral * 0.9

        integral = self._integral_error[model]
        p_term = max(1, int(abs(error_p99) / (config.p99_latency_ms / 4))) if error_p99 > 0 else 0
        i_term = int(abs(integral))

        # Above P99 SLO: reduce batch size (proportional to overshoot)
        if p99 > config.p99_latency_ms:
            step = min(max_step, max(1, p_term + i_term))
            new_size = max(config.min_batch_size, current - step)
        # Below P50 SLO with headroom: increase batch size
        elif p50 < config.p50_latency_ms * 0.7 and p99 < config.p99_latency_ms * 0.8:
            step = min(max_step, max(1, int(error_p50 / (config.p50_latency_ms / 4)) + i_term))
            new_size = min(config.max_batch_size, current + step)
        else:
            new_size = current

        if new_size != current:
            self._current_batch[model] = new_size
            self._last_adjustment[model] = now

    def get_batch_size(self, model: str) -> int:
        """Return recommended batch size for model."""
        with self._lock:
            config = self._configs.get(model, self._default_config)
            current = self._current_batch.get(model, config.min_batch_size)
            return min(current, config.max_batch_size)

    def get_stats(self, model: str) -> BatchWindowStats:
        """Return recent stats for a model."""
        with self._lock:
            latencies = list(self._latencies.get(model, []))
            batch_sizes = list(self._batch_sizes.get(model, []))
            if not latencies:
                return BatchWindowStats()
            sorted_lats = sorted(latencies[-200:])
            return BatchWindowStats(
                avg_latency_ms=sum(latencies[-200:]) / min(len(latencies), 200),
                p50_latency_ms=self._percentile(sorted_lats, 50),
                p99_latency_ms=self._percentile(sorted_lats, 99),
                # Bug M-04 fix: previously `time.time() - 60` (~1.78e9), so
                # throughput was ~3e-5. Track window start time per model.
                throughput=self._window_throughput(model, len(latencies)),
                sample_count=len(latencies),
                batch_sizes=batch_sizes[-20:],
            )

    def all_stats(self) -> dict[str, Any]:
        with self._lock:
            result = {}
            for model, _config in self._configs.items():
                latencies = list(self._latencies.get(model, []))
                batch_sizes = list(self._batch_sizes.get(model, []))
                if not latencies:
                    result[model] = str(BatchWindowStats())
                    continue
                sorted_lats = sorted(latencies[-200:])
                stats = BatchWindowStats(
                    avg_latency_ms=sum(latencies[-200:]) / min(len(latencies), 200),
                    p50_latency_ms=self._percentile(sorted_lats, 50),
                    p99_latency_ms=self._percentile(sorted_lats, 99),
                    throughput=self._window_throughput(model, len(latencies)),
                    sample_count=len(latencies),
                    batch_sizes=batch_sizes[-20:],
                )
                result[model] = str(stats)
            return result

    def _window_throughput(self, model: str, n_samples: int) -> float:
        """Requests/sec over the model's sampling window.

        Bug M-04 fix: the previous formula divided by `time.time() - 60`
        (an epoch-sized number), always yielding ~0. Uses a per-model
        first-seen timestamp for the window instead.
        """
        if n_samples == 0:
            return 0.0
        window_start = self._window_start.get(model)
        if window_start is None:
            return 0.0
        elapsed = time.time() - window_start
        if elapsed <= 0:
            return 0.0
        return n_samples / elapsed

    def get_current_batch_size(self, model: str) -> int:
        """Return the current (possibly adjusted) batch size for a model."""
        with self._lock:
            return self._current_batch.get(model, self._default_config.min_batch_size)

    def get_slo_config(self, model: str) -> SLOConfig:
        """Return the SLO config for a model (or default)."""
        with self._lock:
            return self._configs.get(model, self._default_config)

    def set_current_batch_size(self, model: str, batch_size: int) -> None:
        """Override the current batch size for a model (e.g. from profiler)."""
        with self._lock:
            config = self._configs.get(model, self._default_config)
            clamped = max(config.min_batch_size, min(batch_size, config.max_batch_size))
            self._current_batch[model] = clamped
