"""Tests for distllm.dist.latency.LatencyTracker."""

from __future__ import annotations

import math
import pytest
from distllm.dist.latency import LatencyTracker


class TestLatencyTracker:
    """Tests for LatencyTracker -- sliding-window per-node latency tracker."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def test_default_window_size(self) -> None:
        tracker = LatencyTracker()
        assert tracker._window_size == 100

    def test_custom_window_size(self) -> None:
        tracker = LatencyTracker(window_size=5)
        assert tracker._window_size == 5

    def test_zero_window_size(self) -> None:
        """deque(maxlen=0) discards everything immediately."""
        tracker = LatencyTracker(window_size=0)
        tracker.record("n1", 1.0)
        assert tracker.get_measurements("n1") == []

    def test_negative_window_size(self) -> None:
        """deque raises ValueError for negative maxlen (raised lazily on record)."""
        tracker = LatencyTracker(window_size=-1)
        with pytest.raises(ValueError):
            tracker.record("n1", 1.0)

    # ------------------------------------------------------------------
    # record / get_measurements
    # ------------------------------------------------------------------

    def test_record_single_node(self) -> None:
        tracker = LatencyTracker()
        tracker.record("node-a", 10.0)
        tracker.record("node-a", 20.0)
        tracker.record("node-a", 30.0)
        assert tracker.get_measurements("node-a") == [10.0, 20.0, 30.0]

    def test_record_multiple_nodes(self) -> None:
        tracker = LatencyTracker()
        tracker.record("a", 1.0)
        tracker.record("b", 2.0)
        tracker.record("a", 3.0)
        assert tracker.get_measurements("a") == [1.0, 3.0]
        assert tracker.get_measurements("b") == [2.0]

    def test_record_negative_values(self) -> None:
        tracker = LatencyTracker()
        tracker.record("n1", -5.0)
        tracker.record("n1", 0.0)
        assert tracker.get_measurements("n1") == [-5.0, 0.0]

    def test_record_floats_and_ints(self) -> None:
        """Should accept int values as well (duck-typed)."""
        tracker = LatencyTracker()
        tracker.record("n1", 42)
        assert tracker.get_measurements("n1") == [42.0]

    def test_window_sliding(self) -> None:
        """Old entries are evicted when window size is exceeded."""
        tracker = LatencyTracker(window_size=3)
        for i in range(5):
            tracker.record("n1", float(i))
        # Only the last 3 values are kept
        assert tracker.get_measurements("n1") == [2.0, 3.0, 4.0]

    # ------------------------------------------------------------------
    # get_avg
    # ------------------------------------------------------------------

    def test_get_avg(self) -> None:
        tracker = LatencyTracker()
        for v in (10.0, 20.0, 30.0):
            tracker.record("n1", v)
        assert tracker.get_avg("n1") == 20.0

    def test_get_avg_unknown_node(self) -> None:
        tracker = LatencyTracker()
        assert tracker.get_avg("nonexistent") is None

    def test_get_avg_empty_deque(self) -> None:
        """Window_size=0 means deque is always empty."""
        tracker = LatencyTracker(window_size=0)
        tracker.record("n1", 1.0)
        assert tracker.get_avg("n1") is None

    def test_get_avg_single_value(self) -> None:
        tracker = LatencyTracker()
        tracker.record("n1", 7.5)
        assert tracker.get_avg("n1") == 7.5

    # ------------------------------------------------------------------
    # get_p95
    # ------------------------------------------------------------------

    def test_get_p95_exact(self) -> None:
        """With 20 values the 95th index is the 19th element (0-based)."""
        tracker = LatencyTracker()
        for i in range(1, 21):
            tracker.record("n1", float(i))
        assert tracker.get_p95("n1") == 19.0

    def test_get_p95_small_sample(self) -> None:
        """With 1 element, p95 is that element."""
        tracker = LatencyTracker()
        tracker.record("n1", 42.0)
        assert tracker.get_p95("n1") == 42.0

    def test_get_p95_two_elements(self) -> None:
        tracker = LatencyTracker()
        tracker.record("n1", 1.0)
        tracker.record("n1", 2.0)
        # sorted = [1, 2]; idx = max(0, 1*0.95 - 1) = max(0, -1) = 0
        assert tracker.get_p95("n1") == 1.0

    def test_get_p95_unknown_node(self) -> None:
        tracker = LatencyTracker()
        assert tracker.get_p95("nonexistent") is None

    def test_get_p95_empty_deque(self) -> None:
        tracker = LatencyTracker(window_size=0)
        tracker.record("n1", 1.0)
        assert tracker.get_p95("n1") is None

    def test_get_p95_unsorted_input(self) -> None:
        """Should sort internally before picking percentile."""
        tracker = LatencyTracker()
        for v in (100.0, 1.0, 50.0, 10.0, 90.0):
            tracker.record("n1", v)
        # sorted = [1, 10, 50, 90, 100]; len=5
        # idx = max(0, int(5*0.95) - 1) = max(0, 4 - 1) = 3
        assert tracker.get_p95("n1") == 90.0

    # ------------------------------------------------------------------
    # get_all_avg
    # ------------------------------------------------------------------

    def test_get_all_avg(self) -> None:
        tracker = LatencyTracker()
        for v in (1.0, 2.0, 3.0):
            tracker.record("n1", v)
        for v in (10.0, 20.0):
            tracker.record("n2", v)
        result = tracker.get_all_avg()
        assert result == {"n1": 2.0, "n2": 15.0}

    def test_get_all_avg_empty(self) -> None:
        tracker = LatencyTracker()
        assert tracker.get_all_avg() == {}

    def test_get_all_avg_no_duplicate_nodes(self) -> None:
        tracker = LatencyTracker()
        tracker.record("x", 1.0)
        tracker.record("x", 2.0)
        result = tracker.get_all_avg()
        assert list(result.keys()) == ["x"]
        assert result["x"] == 1.5

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def test_reset_single_node(self) -> None:
        tracker = LatencyTracker()
        tracker.record("n1", 1.0)
        tracker.record("n2", 2.0)
        tracker.reset("n1")
        assert tracker.get_measurements("n1") == []
        # Other node unaffected
        assert tracker.get_measurements("n2") == [2.0]

    def test_reset_all(self) -> None:
        tracker = LatencyTracker()
        tracker.record("n1", 1.0)
        tracker.record("n2", 2.0)
        tracker.reset()
        assert tracker.get_measurements("n1") == []
        assert tracker.get_measurements("n2") == []
        assert tracker.get_all_avg() == {}

    def test_reset_unknown_node(self) -> None:
        """Resetting a node that was never recorded should not raise."""
        tracker = LatencyTracker()
        tracker.reset("ghost")
        # No exception

    def test_reset_then_record(self) -> None:
        """After reset, new records should work normally."""
        tracker = LatencyTracker()
        tracker.record("n1", 1.0)
        tracker.reset("n1")
        tracker.record("n1", 99.0)
        assert tracker.get_measurements("n1") == [99.0]

    # ------------------------------------------------------------------
    # Thread safety (basic structural — no timing races required)
    # ------------------------------------------------------------------

    def test_concurrent_record(self) -> None:
        """Multiple threads recording on the same tracker should not
        cause structural errors (list corruption, etc.)."""
        tracker = LatencyTracker(window_size=500)
        n_threads = 8
        records_per_thread = 200

        def _record(thread_id: int) -> None:
            node = f"node-{thread_id}"
            for i in range(records_per_thread):
                tracker.record(node, float(i))

        threads = [
            threading.Thread(target=_record, args=(tid,))
            for tid in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for tid in range(n_threads):
            node = f"node-{tid}"
            ms = tracker.get_measurements(node)
            assert len(ms) == records_per_thread
            # Values should be in insertion order (no interleaving corruption)
            assert ms == [float(i) for i in range(records_per_thread)]

    def test_concurrent_read_and_write(self) -> None:
        """Concurrent get_avg / record should not raise."""
        tracker = LatencyTracker(window_size=50)

        def _writer() -> None:
            for i in range(100):
                tracker.record("shared", float(i))

        def _reader() -> None:
            for _ in range(100):
                tracker.get_avg("shared")
                tracker.get_p95("shared")
                tracker.get_all_avg()
                tracker.get_measurements("shared")

        threads = [
            threading.Thread(target=_writer),
            threading.Thread(target=_reader),
            threading.Thread(target=_reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # No exception means success

    # ------------------------------------------------------------------
    # Idempotency / isolation
    # ------------------------------------------------------------------

    def test_isolated_nodes(self) -> None:
        """Operations on one node do not affect another."""
        tracker = LatencyTracker()
        tracker.record("a", 10.0)
        tracker.record("a", 20.0)
        tracker.record("b", 99.0)
        assert tracker.get_avg("a") == 15.0
        assert tracker.get_avg("b") == 99.0
        tracker.reset("a")
        assert tracker.get_avg("a") is None
        assert tracker.get_avg("b") == 99.0

    def test_get_all_avg_empty_after_reset(self) -> None:
        tracker = LatencyTracker()
        tracker.record("n1", 1.0)
        tracker.reset("n1")
        assert tracker.get_all_avg() == {}

    def test_get_all_avg_some_empty_reset(self) -> None:
        tracker = LatencyTracker()
        tracker.record("n1", 1.0)
        tracker.record("n2", 2.0)
        tracker.reset("n1")
        result = tracker.get_all_avg()
        assert "n1" not in result
        assert result["n2"] == 2.0

    # ------------------------------------------------------------------
    # Edge cases for p95 index calculation
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "count,expected_p95",
        [
            (1, 1.0),
            (2, 1.0),
            (3, 2.0),
            (4, 3.0),
            (5, 4.0),
            (10, 9.0),
            (20, 19.0),
            (100, 95.0),
        ],
    )
    def test_get_p95_various_sizes(
        self, count: int, expected_p95: float
    ) -> None:
        """P95 follows the formula: idx = max(0, floor(0.95*count) - 1)."""
        tracker = LatencyTracker(window_size=count)
        for i in range(1, count + 1):
            tracker.record("n1", float(i))
        assert tracker.get_p95("n1") == expected_p95


# Import threading only for the concurrency tests
import threading
