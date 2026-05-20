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
        self._lock = threading.Lock()

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
                self.set_slo(model)
            for lat in latencies:
                self._latencies[model].append(lat)
            self._batch_sizes[model].append(batch_size)
            self._adjust_batch_size(model, now)

    def _adjust_batch_size(self, model: str, now: float) -> None:
        """Adjust batch size based on recent latency vs SLO."""
        config = self._configs.get(model, self._default_config)
        if now - self._last_adjustment.get(model, 0.0) < config.cooldown_s:
            return

        latencies = list(self._latencies[model])
        if len(latencies) < 10:
            return

        sorted_lats = sorted(latencies[-100:])
        p50 = sorted_lats[len(sorted_lats) * 50 // 100]
        p99 = sorted_lats[len(sorted_lats) * 99 // 100]

        current = self._current_batch[model]

        # Above P99 SLO: reduce batch size
        if p99 > config.p99_latency_ms:
            new_size = max(config.min_batch_size, current - config.adjustment_step)
        # Below P50 SLO: increase batch size
        elif p50 < config.p50_latency_ms * 0.7:
            new_size = min(config.max_batch_size, current + config.adjustment_step)
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
                p50_latency_ms=sorted_lats[len(sorted_lats) * 50 // 100],
                p99_latency_ms=sorted_lats[len(sorted_lats) * 99 // 100],
                throughput=len(latencies) / max(time.time() - 60, 1) * 60 if latencies else 0,
                sample_count=len(latencies),
                batch_sizes=batch_sizes[-20:],
            )

    def all_stats(self) -> dict[str, Any]:
        with self._lock:
            return {m: str(self.get_stats(m)) for m in self._configs}
