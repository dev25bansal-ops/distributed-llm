"""Prometheus/OpenTelemetry metrics for PagedAttention KV cache.

Exports gauges and counters for block utilization, swap rate,
fragmentation ratio, and allocation latency percentiles.

Usage::

    from distllm.core.kv_cache_metrics import KVCacheMetricsCollector

    collector = KVCacheMetricsCollector(paged_attention_mgr)
    collector.start_background_update(interval_s=5.0)

    # In Prometheus scrape handler:
    metrics = collector.collect()
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from loguru import logger


@dataclass
class MetricSnapshot:
    """Point-in-time snapshot of all KV cache metrics."""
    timestamp: float

    # Block pool
    total_blocks: int = 0
    free_blocks: int = 0
    used_blocks: int = 0
    utilization: float = 0.0
    pool_memory_gb: float = 0.0

    # Sequences
    active_sequences: int = 0

    # Swap
    swapped_blocks: int = 0
    total_swaps: int = 0
    total_restores: int = 0
    swap_memory_gb: float = 0.0

    # Fragmentation
    fragmentation_ratio: float = 0.0

    # Allocation latency (microseconds)
    alloc_latency_p50_us: float = 0.0
    alloc_latency_p95_us: float = 0.0
    alloc_latency_p99_us: float = 0.0

    # Throughput
    allocations_per_sec: float = 0.0
    frees_per_sec: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items() if k != "timestamp"}

    def to_prometheus(self) -> str:
        """Format as Prometheus exposition format."""
        lines = []
        prefix = "distllm_kv_cache"
        for k, v in self.to_dict().items():
            metric_name = f"{prefix}_{k}"
            lines.append(f"# TYPE {metric_name} gauge")
            lines.append(f"{metric_name} {v}")
        return "\n".join(lines)


class KVCacheMetricsCollector:
    """Collects and exposes KV cache metrics for monitoring.

    Args:
        paged_attention_mgr: PagedAttentionManager instance (backends or dist).
        defragmenter: Optional MemoryDefragmenter for fragmentation metrics.
    """

    def __init__(
        self,
        paged_attention_mgr: Any,
        defragmenter: Any | None = None,
    ):
        self._mgr = paged_attention_mgr
        self._defrag = defragmenter
        self._lock = threading.Lock()
        self._history: deque[MetricSnapshot] = deque(maxlen=1000)
        self._alloc_latencies: deque[float] = deque(maxlen=10000)
        self._prev_allocations: int = 0
        self._prev_frees: int = 0
        self._prev_timestamp: float = time.time()
        self._bg_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def record_allocation_latency(self, latency_us: float) -> None:
        """Record a single allocation latency measurement."""
        self._alloc_latencies.append(latency_us)

    def collect(self) -> MetricSnapshot:
        """Collect current metrics from the PagedAttentionManager."""
        now = time.time()
        snap = MetricSnapshot(timestamp=now)

        # Pool stats
        pool = getattr(self._mgr, "pool", None)
        if pool is not None:
            stats = pool.stats() if hasattr(pool, "stats") else {}
            snap.total_blocks = stats.get("total_blocks", getattr(pool, "num_blocks", 0))
            snap.free_blocks = stats.get("free_blocks", getattr(pool, "free_count", 0))
            snap.used_blocks = stats.get("used_blocks", getattr(pool, "used_count", 0))
            snap.utilization = stats.get("utilization", getattr(pool, "utilization", 0.0))
            snap.pool_memory_gb = stats.get("pool_memory_gb", 0.0)
            snap.swapped_blocks = stats.get("swapped_blocks", 0)
            snap.total_swaps = stats.get("total_swaps", 0)
            snap.total_restores = stats.get("total_restores", 0)
            snap.swap_memory_gb = stats.get("swap_memory_gb", 0.0)
        else:
            # backends version
            mgr_stats = self._mgr.get_stats() if hasattr(self._mgr, "get_stats") else {}
            snap.total_blocks = mgr_stats.get("num_blocks", 0)
            snap.free_blocks = mgr_stats.get("free_blocks", 0)
            snap.used_blocks = mgr_stats.get("used_blocks", 0)
            snap.utilization = mgr_stats.get("memory_usage_pct", 0.0) / 100.0
            snap.active_sequences = mgr_stats.get("active_sequences", 0)

        # Sequences (dist version)
        if hasattr(self._mgr, "active_sequences"):
            snap.active_sequences = self._mgr.active_sequences

        # Fragmentation
        if self._defrag is not None and hasattr(self._mgr, "_blocks"):
            ratio = self._defrag._compute_fragmentation_ratio(self._mgr._blocks)
            snap.fragmentation_ratio = ratio

        # Allocation latency percentiles
        if self._alloc_latencies:
            sorted_lat = sorted(self._alloc_latencies)
            n = len(sorted_lat)
            snap.alloc_latency_p50_us = sorted_lat[int(n * 0.50)]
            snap.alloc_latency_p95_us = sorted_lat[int(n * 0.95)]
            snap.alloc_latency_p99_us = sorted_lat[min(int(n * 0.99), n - 1)]

        # Throughput
        dt = max(now - self._prev_timestamp, 0.001)
        if pool is not None:
            pool_stats = pool.stats() if hasattr(pool, "stats") else {}
            cur_allocs = pool_stats.get("total_allocations", 0)
            cur_frees = pool_stats.get("total_swaps", 0)  # proxy
        else:
            mgr_stats = self._mgr.get_stats() if hasattr(self._mgr, "get_stats") else {}
            cur_allocs = mgr_stats.get("allocations", 0)
            cur_frees = mgr_stats.get("frees", 0)

        snap.allocations_per_sec = (cur_allocs - self._prev_allocations) / dt
        snap.frees_per_sec = (cur_frees - self._prev_frees) / dt
        self._prev_allocations = cur_allocs
        self._prev_frees = cur_frees
        self._prev_timestamp = now

        with self._lock:
            self._history.append(snap)

        return snap

    def get_history(self, last_n: int = 100) -> list[MetricSnapshot]:
        """Return the last N metric snapshots."""
        with self._lock:
            return list(self._history)[-last_n:]

    def start_background_update(self, interval_s: float = 5.0) -> None:
        """Start a background thread that collects metrics periodically."""
        if self._bg_thread is not None and self._bg_thread.is_alive():
            return
        self._stop_event.clear()

        def _loop():
            while not self._stop_event.wait(interval_s):
                try:
                    self.collect()
                except Exception as e:
                    logger.debug(f"Metrics collection error: {e}")

        self._bg_thread = threading.Thread(target=_loop, daemon=True, name="kv-metrics")
        self._bg_thread.start()
        logger.info(f"KV cache metrics: background collection every {interval_s}s")

    def stop_background_update(self) -> None:
        """Stop the background collection thread."""
        self._stop_event.set()
        if self._bg_thread is not None:
            self._bg_thread.join(timeout=2.0)
            self._bg_thread = None

    def __repr__(self) -> str:
        latest = self._history[-1] if self._history else None
        if latest:
            return (
                f"KVCacheMetricsCollector(blocks={latest.used_blocks}/{latest.total_blocks}, "
                f"util={latest.utilization:.1%}, swapped={latest.swapped_blocks})"
            )
        return "KVCacheMetricsCollector(no data)"
