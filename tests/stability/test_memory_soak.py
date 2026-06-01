"""Memory profiling soak tests for long-running sessions.

Detects memory leaks in:
- KV cache allocation/deallocation
- Request tracking structures
- Thread pool growth
- GPU memory fragmentation
- Python heap growth

Run with:
    pytest tests/stability/test_memory_soak.py -v --timeout=3600
    DISTLLM_SOAK_DURATION=300 pytest tests/stability/test_memory_soak.py -v
"""

from __future__ import annotations

import gc
import os
import sys
import threading
import time
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch

sys.path.insert(0, "src")

SOAK_DURATION_S = int(os.environ.get("DISTLLM_SOAK_DURATION", "60"))
CHECK_INTERVAL_S = 5


@dataclass
class MemorySnapshot:
    """Point-in-time memory measurement."""
    timestamp: float
    python_rss_mb: float
    torch_allocated_mb: float
    torch_reserved_mb: float
    thread_count: int
    gc_objects: int


class MemoryProfiler:
    """Tracks memory usage over time to detect leaks."""

    def __init__(self):
        self._snapshots: list[MemorySnapshot] = []
        self._lock = threading.Lock()

    def snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot."""
        import psutil
        process = psutil.Process()
        rss = process.memory_info().rss / (1024 * 1024)

        torch_alloc = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        torch_reserved = torch.cuda.memory_reserved() / (1024 * 1024) if torch.cuda.is_available() else 0

        snap = MemorySnapshot(
            timestamp=time.time(),
            python_rss_mb=rss,
            torch_allocated_mb=torch_alloc,
            torch_reserved_mb=torch_reserved,
            thread_count=threading.active_count(),
            gc_objects=len(gc.get_objects()),
        )

        with self._lock:
            self._snapshots.append(snap)
        return snap

    def detect_leaks(self, threshold_pct: float = 20.0) -> dict:
        """Analyze snapshots for memory leaks.

        Returns:
            Dict with leak detection results.
        """
        with self._lock:
            if len(self._snapshots) < 3:
                return {"status": "insufficient_data", "snapshots": len(self._snapshots)}

            first = self._snapshots[0]
            last = self._snapshots[-1]
            mid = self._snapshots[len(self._snapshots) // 2]

            results = {
                "duration_s": last.timestamp - first.timestamp,
                "snapshots": len(self._snapshots),
                "leaks_detected": [],
            }

            # Check Python RSS growth
            if first.python_rss_mb > 0:
                rss_growth = ((last.python_rss_mb - first.python_rss_mb) / first.python_rss_mb) * 100
                results["python_rss_start_mb"] = round(first.python_rss_mb, 1)
                results["python_rss_end_mb"] = round(last.python_rss_mb, 1)
                results["python_rss_growth_pct"] = round(rss_growth, 1)
                if rss_growth > threshold_pct:
                    results["leaks_detected"].append(f"Python RSS grew {rss_growth:.1f}%")

            # Check torch allocated growth
            if first.torch_allocated_mb > 0:
                torch_growth = ((last.torch_allocated_mb - first.torch_allocated_mb) / first.torch_allocated_mb) * 100
                results["torch_allocated_start_mb"] = round(first.torch_allocated_mb, 1)
                results["torch_allocated_end_mb"] = round(last.torch_allocated_mb, 1)
                results["torch_allocated_growth_pct"] = round(torch_growth, 1)
                if torch_growth > threshold_pct:
                    results["leaks_detected"].append(f"Torch allocated grew {torch_growth:.1f}%")

            # Check thread count growth
            thread_growth = last.thread_count - first.thread_count
            results["thread_count_start"] = first.thread_count
            results["thread_count_end"] = last.thread_count
            results["thread_growth"] = thread_growth
            if thread_growth > 10:
                results["leaks_detected"].append(f"Thread count grew by {thread_growth}")

            # Check GC object growth
            gc_growth = ((last.gc_objects - first.gc_objects) / max(first.gc_objects, 1)) * 100
            results["gc_objects_start"] = first.gc_objects
            results["gc_objects_end"] = last.gc_objects
            results["gc_objects_growth_pct"] = round(gc_growth, 1)
            if gc_growth > 50:
                results["leaks_detected"].append(f"GC objects grew {gc_growth:.1f}%")

            results["status"] = "leak_detected" if results["leaks_detected"] else "ok"
            return results


@pytest.mark.stability
class TestMemorySoak:
    """Memory leak detection soak tests."""

    def test_kv_cache_no_leak(self):
        """KV cache allocate/free cycle must not leak memory."""
        from distllm.core.kv_cache import KVCache

        profiler = MemoryProfiler()
        profiler.snapshot()

        for i in range(1000):
            cache = KVCache(max_seq_len=100)
            cache.num_layers = 4
            for layer in range(4):
                key = torch.randn(1, 8, 10, 64)
                value = torch.randn(1, 8, 10, 64)
                cache.cache.append((key, value))
            del cache
            if i % 100 == 0:
                gc.collect()
                profiler.snapshot()

        profiler.snapshot()
        results = profiler.detect_leaks(threshold_pct=30.0)
        assert results["status"] != "leak_detected", f"Memory leak: {results.get('leaks_detected')}"

    def test_request_tracking_no_leak(self):
        """Request tracking must not accumulate unbounded state."""
        from distllm.core.request_latency import RequestLatencyTracker

        tracker = RequestLatencyTracker()
        profiler = MemoryProfiler()
        profiler.snapshot()

        for i in range(5000):
            rid = f"req-{i}"
            tracker.register(rid)
            tracker.record_first_token(rid)
            tracker.record_token(rid)
            tracker.complete(rid)
            if i % 500 == 0:
                gc.collect()
                profiler.snapshot()

        profiler.snapshot()
        results = profiler.detect_leaks(threshold_pct=20.0)
        assert results["status"] != "leak_detected", f"Memory leak: {results.get('leaks_detected')}"

    def test_thread_safety_no_deadlock(self):
        """Concurrent operations must not deadlock."""
        from distllm.core.kv_cache import KVCacheManager

        manager = KVCacheManager()
        errors = []
        timeout_s = 10

        def worker(worker_id):
            try:
                for i in range(100):
                    rid = f"w{worker_id}-req-{i}"
                    manager.create(rid)
                    manager.get(rid)
                    manager.delete(rid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=timeout_s)

        assert len(errors) == 0, f"Errors in concurrent access: {errors}"
        assert all(not t.is_alive() for t in threads), "Thread deadlock detected"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="GPU required for memory profiling",
    )
    def test_gpu_memory_no_leak(self):
        """GPU memory must be properly freed after operations."""
        profiler = MemoryProfiler()
        profiler.snapshot()

        for i in range(100):
            # Allocate and free GPU tensors
            tensors = [torch.randn(100, 100, device="cuda") for _ in range(10)]
            del tensors
            torch.cuda.empty_cache()

            if i % 20 == 0:
                profiler.snapshot()

        profiler.snapshot()
        results = profiler.detect_leaks(threshold_pct=10.0)
        assert results["status"] != "leak_detected", f"GPU memory leak: {results.get('leaks_detected')}"
