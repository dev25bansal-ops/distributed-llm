"""Tests for LatencyTracker."""

import threading
import pytest

from distllm.core.latency_tracker import LatencyTracker


class TestLatencyTracker:
    """Tests for LatencyTracker class."""

    def test_record_and_get_avg(self):
        """Record values, verify average."""
        tracker = LatencyTracker(window_size=100)
        tracker.record("node-1", 10.0)
        tracker.record("node-1", 20.0)
        tracker.record("node-1", 30.0)

        assert tracker.get_avg("node-1") == 20.0

    def test_get_p95(self):
        """Record values, verify p95."""
        tracker = LatencyTracker(window_size=100)
        for i in range(100):
            tracker.record("node-1", float(i))

        p95 = tracker.get_p95("node-1")
        # idx = max(0, int(100 * 0.95) - 1) = max(0, 95-1) = 94, value = 94.0
        assert p95 == 94.0

    def test_get_p95_small_sample(self):
        """P95 with fewer than 95 values."""
        tracker = LatencyTracker(window_size=100)
        for i in range(10):
            tracker.record("node-1", float(i))

        p95 = tracker.get_p95("node-1")
        assert p95 is not None
        # idx = max(0, int(10 * 0.95) - 1) = max(0, 9-1) = 8, sorted_vals[8] = 8.0
        assert p95 == 8.0

    def test_get_p95_single_value(self):
        """P95 with a single value."""
        tracker = LatencyTracker()
        tracker.record("node-1", 42.0)
        assert tracker.get_p95("node-1") == 42.0

    def test_get_all_avg(self):
        """Multiple nodes, verify all averages."""
        tracker = LatencyTracker()
        tracker.record("node-1", 10.0)
        tracker.record("node-1", 20.0)
        tracker.record("node-2", 30.0)
        tracker.record("node-2", 50.0)

        all_avg = tracker.get_all_avg()
        assert len(all_avg) == 2
        assert all_avg["node-1"] == 15.0
        assert all_avg["node-2"] == 40.0

    def test_sliding_window(self):
        """Record >window_size values, verify only last N used."""
        tracker = LatencyTracker(window_size=5)
        for i in range(10):
            tracker.record("node-1", float(i))

        # Only last 5 values (5,6,7,8,9) should be used
        measurements = tracker.get_measurements("node-1")
        assert len(measurements) == 5
        assert measurements == [5.0, 6.0, 7.0, 8.0, 9.0]
        assert tracker.get_avg("node-1") == 7.0

    def test_empty_node(self):
        """get_avg for unknown node returns None."""
        tracker = LatencyTracker()
        assert tracker.get_avg("unknown") is None
        assert tracker.get_p95("unknown") is None
        assert tracker.get_measurements("unknown") == []

    def test_thread_safety(self):
        """Concurrent records from multiple threads."""
        tracker = LatencyTracker(window_size=1000)
        errors = []

        def record_range(start, count):
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

    def test_reset_single_node(self):
        """Reset one node, others preserved."""
        tracker = LatencyTracker()
        tracker.record("node-1", 10.0)
        tracker.record("node-2", 20.0)

        tracker.reset("node-1")

        assert tracker.get_avg("node-1") is None
        assert tracker.get_avg("node-2") == 20.0

    def test_reset_all(self):
        """Reset all nodes."""
        tracker = LatencyTracker()
        tracker.record("node-1", 10.0)
        tracker.record("node-2", 20.0)

        tracker.reset()

        assert tracker.get_all_avg() == {}
