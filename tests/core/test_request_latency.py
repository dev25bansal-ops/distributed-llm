"""Tests for LatencyTracker from distllm.dist.latency.

Covers:
- Construction and initial state
- record and get_avg
- get_p95 with various sample sizes
- get_all_avg across multiple nodes
- Sliding window eviction
- Empty / unknown node queries
- Thread safety
- reset single node and all nodes
- get_measurements
"""

from __future__ import annotations

import statistics
import threading

from distllm.dist.latency import LatencyTracker


class TestLatencyTrackerConstruction:
    """Construction and initial state."""

    def test_default_construction(self) -> None:
        tracker = LatencyTracker()
        assert tracker._window_size == 100
        assert tracker._data == {}

    def test_custom_window_size(self) -> None:
        tracker = LatencyTracker(window_size=50)
        assert tracker._window_size == 50

    def test_get_measurements_empty(self) -> None:
        tracker = LatencyTracker()
        assert tracker.get_measurements("unknown") == []


class TestLatencyTrackerRecordAndQuery:
    """Record values and query them."""

    def test_record_and_get_avg(self) -> None:
        tracker = LatencyTracker(window_size=100)
        tracker.record("node-1", 10.0)
        tracker.record("node-1", 20.0)
        tracker.record("node-1", 30.0)
        assert tracker.get_avg("node-1") == 20.0

    def test_get_avg_unknown_node(self) -> None:
        tracker = LatencyTracker()
        assert tracker.get_avg("unknown") is None

    def test_get_avg_single_value(self) -> None:
        tracker = LatencyTracker()
        tracker.record("node-1", 42.0)
        assert tracker.get_avg("node-1") == 42.0

    def test_get_p95_hundred_values(self) -> None:
        tracker = LatencyTracker(window_size=100)
        for i in range(100):
            tracker.record("node-1", float(i))
        # idx = max(0, int(100 * 0.95) - 1) = 94, sorted_vals[94] = 94.0
        assert tracker.get_p95("node-1") == 94.0

    def test_get_p95_small_sample(self) -> None:
        tracker = LatencyTracker(window_size=100)
        for i in range(10):
            tracker.record("node-1", float(i))
        p95 = tracker.get_p95("node-1")
        assert p95 is not None
        # idx = max(0, int(10 * 0.95) - 1) = 8, sorted_vals[8] = 8.0
        assert p95 == 8.0

    def test_get_p95_single_value(self) -> None:
        tracker = LatencyTracker()
        tracker.record("node-1", 42.0)
        assert tracker.get_p95("node-1") == 42.0

    def test_get_p95_unknown_node(self) -> None:
        tracker = LatencyTracker()
        assert tracker.get_p95("unknown") is None

    def test_get_all_avg(self) -> None:
        tracker = LatencyTracker()
        tracker.record("node-1", 10.0)
        tracker.record("node-1", 20.0)
        tracker.record("node-2", 30.0)
        tracker.record("node-2", 50.0)
        all_avg = tracker.get_all_avg()
        assert len(all_avg) == 2
        assert all_avg["node-1"] == 15.0
        assert all_avg["node-2"] == 40.0

    def test_get_all_avg_empty(self) -> None:
        tracker = LatencyTracker()
        assert tracker.get_all_avg() == {}

    def test_measurements_snapshot(self) -> None:
        tracker = LatencyTracker()
        tracker.record("node-1", 1.0)
        tracker.record("node-1", 2.0)
        measurements = tracker.get_measurements("node-1")
        assert measurements == [1.0, 2.0]


class TestLatencyTrackerSlidingWindow:
    """Sliding-window behavior."""

    def test_window_eviction(self) -> None:
        tracker = LatencyTracker(window_size=5)
        for i in range(10):
            tracker.record("node-1", float(i))
        measurements = tracker.get_measurements("node-1")
        assert len(measurements) == 5
        assert measurements == [5.0, 6.0, 7.0, 8.0, 9.0]
        assert tracker.get_avg("node-1") == 7.0

    def test_window_size_one(self) -> None:
        tracker = LatencyTracker(window_size=1)
        tracker.record("node-1", 10.0)
        tracker.record("node-1", 20.0)
        assert tracker.get_avg("node-1") == 20.0
        assert len(tracker.get_measurements("node-1")) == 1


class TestLatencyTrackerReset:
    """Reset behavior."""

    def test_reset_single_node(self) -> None:
        tracker = LatencyTracker()
        tracker.record("node-1", 10.0)
        tracker.record("node-2", 20.0)
        tracker.reset("node-1")
        assert tracker.get_avg("node-1") is None
        assert tracker.get_avg("node-2") == 20.0

    def test_reset_unknown_node(self) -> None:
        tracker = LatencyTracker()
        tracker.reset("nonexistent")  # Should not raise
        assert tracker.get_all_avg() == {}

    def test_reset_all(self) -> None:
        tracker = LatencyTracker()
        tracker.record("node-1", 10.0)
        tracker.record("node-2", 20.0)
        tracker.reset()
        assert tracker.get_all_avg() == {}

    def test_reset_then_record(self) -> None:
        tracker = LatencyTracker()
        tracker.record("node-1", 10.0)
        tracker.reset("node-1")
        assert tracker.get_avg("node-1") is None
        tracker.record("node-1", 99.0)
        assert tracker.get_avg("node-1") == 99.0


class TestLatencyTrackerThreadSafety:
    """Thread safety under concurrent access."""

    def test_concurrent_records(self) -> None:
        tracker = LatencyTracker(window_size=1000)
        errors: list[Exception] = []

        def record_range(start: int, count: int) -> None:
            try:
                for i in range(count):
                    tracker.record("node-1", float(start + i))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_range, args=(0, 100)),
            threading.Thread(target=record_range, args=(100, 100)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(tracker.get_measurements("node-1")) == 200

    def test_concurrent_record_and_query(self) -> None:
        tracker = LatencyTracker(window_size=500)
        for i in range(100):
            tracker.record("node-1", float(i))

        results: list[float | None] = []

        def query() -> None:
            results.append(tracker.get_avg("node-1"))
            results.append(tracker.get_p95("node-1"))

        t = threading.Thread(target=query)
        t.start()
        tracker.record("node-1", 200.0)
        t.join()

        assert len(results) == 2
        # avg should be between old range and including 200
        avg = tracker.get_avg("node-1")
        assert avg is not None and avg > 0
