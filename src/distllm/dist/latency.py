"""Sliding-window per-node latency tracker for straggler detection and rebalancing."""

from __future__ import annotations

import threading
import statistics
from collections import deque


class LatencyTracker:
    """Tracks recent latency measurements per node using a sliding window."""

    def __init__(self, window_size: int = 100):
        self._window_size = window_size
        self._lock = threading.Lock()
        self._data: dict[str, deque] = {}

    def _ensure_node(self, node_id: str) -> deque:
        if node_id not in self._data:
            self._data[node_id] = deque(maxlen=self._window_size)
        return self._data[node_id]

    def record(self, node_id: str, value: float) -> None:
        with self._lock:
            dq = self._ensure_node(node_id)
            dq.append(value)

    def get_avg(self, node_id: str) -> float | None:
        with self._lock:
            dq = self._data.get(node_id)
            if not dq:
                return None
            return statistics.mean(dq)

    def get_p95(self, node_id: str) -> float | None:
        with self._lock:
            dq = self._data.get(node_id)
            if not dq:
                return None
            sorted_vals = sorted(dq)
            idx = max(0, int(len(sorted_vals) * 0.95) - 1)
            return sorted_vals[idx]

    def get_all_avg(self) -> dict[str, float]:
        with self._lock:
            return {nid: statistics.mean(dq) for nid, dq in self._data.items() if dq}

    def get_measurements(self, node_id: str) -> list[float]:
        with self._lock:
            dq = self._data.get(node_id)
            if not dq:
                return []
            return list(dq)

    def reset(self, node_id: str | None = None) -> None:
        with self._lock:
            if node_id is not None:
                self._data.pop(node_id, None)
            else:
                self._data.clear()
