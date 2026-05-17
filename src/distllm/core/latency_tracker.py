"""Thread-safe sliding window per-node latency tracker."""

import statistics
import threading
from collections import defaultdict, deque


class LatencyTracker:
    """Thread-safe sliding window per-node latency tracker.

    Records per-node latency measurements and provides
    average, p95, and aggregate statistics.
    """

    def __init__(self, window_size: int = 100):
        self._window_size = window_size
        self._measurements: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self._lock = threading.Lock()

    def record(self, node_id: str, latency_ms: float) -> None:
        """Record a latency measurement for a node."""
        with self._lock:
            self._measurements[node_id].append(latency_ms)

    def get_avg(self, node_id: str) -> float | None:
        """Get average latency for a node."""
        with self._lock:
            measurements = self._measurements.get(node_id)
            if not measurements:
                return None
            return statistics.mean(measurements)

    def get_p95(self, node_id: str) -> float | None:
        """Get p95 latency for a node."""
        with self._lock:
            measurements = list(self._measurements.get(node_id, []))
            if not measurements:
                return None
            measurements.sort()
            idx = int(len(measurements) * 0.95)
            return measurements[min(idx, len(measurements) - 1)]

    def get_all_avg(self) -> dict[str, float]:
        """Get average latency for all nodes with data."""
        with self._lock:
            result = {}
            for node_id, measurements in self._measurements.items():
                if measurements:
                    result[node_id] = statistics.mean(measurements)
            return result

    def get_measurements(self, node_id: str) -> list[float]:
        """Get all measurements for a node."""
        with self._lock:
            return list(self._measurements.get(node_id, []))

    def reset(self, node_id: str | None = None) -> None:
        """Reset measurements for a node or all nodes."""
        with self._lock:
            if node_id:
                self._measurements.pop(node_id, None)
            else:
                self._measurements.clear()
