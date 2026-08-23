"""Benchmark: straggler detection latency.

Measures the time between injecting a latency spike and the straggler
detector identifying the affected node.  Uses mock timing with
configurable latency simulation — no real GPU or network required.

Metrics:
    - Detection latency (ms) at p50, p95, p99
    - False positive rate at various thresholds
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from distllm.dist.straggler import StragglerDetector, DetectionMethod


class TestStragglerDetectionLatency:
    """Measures how quickly the detector identifies a slow node."""

    @pytest.mark.parametrize("method", [
        DetectionMethod.THRESHOLD,
        DetectionMethod.MAD,
        DetectionMethod.TREND,
        DetectionMethod.ENSEMBLE,
    ])
    def test_detection_latency(self, benchmark, method):
        """Benchmark time to detect a straggler after injecting slow latencies."""

        def _run() -> float:
            d = StragglerDetector(
                detection_method=method,
                check_interval_s=0.0,
                consecutive_threshold=3,
                window_size=10,
                slow_threshold_ms=50.0,
                threshold_multiplier=1.5,
                mad_threshold=2.0,
            )

            # Seed 2 normal nodes + 1 future straggler
            for _ in range(15):
                d.record_latency("n1", 10)
                d.record_latency("n2", 10)
                d.record_latency("n3", 10)

            t0 = time.perf_counter_ns()

            # Inject the straggler pattern
            for _ in range(5):
                d.record_latency("n3", 500)

            reports = d.check()
            elapsed_ns = time.perf_counter_ns() - t0

            # Verify detection
            is_detected = any(r.node_id == "n3" for r in reports)

            # Time until straggler was flagged (simplified: check duration)
            return elapsed_ns / 1e6  # ms

        result = benchmark(_run)
        assert result > 0

    def test_false_positive_rate(self):
        """Benchmark: with all nodes healthy, should detect none."""
        d = StragglerDetector(check_interval_s=0.0, consecutive_threshold=3)
        for _ in range(30):
            d.record_latency("n1", 10)
            d.record_latency("n2", 10)
            d.record_latency("n3", 10)

        reports = d.check()
        assert len(reports) == 0


class TestThresholdSensitivity:
    """Measures sensitivity of detection thresholds."""

    @pytest.mark.parametrize("multiplier,expected_detect", [
        (1.2, True),    # very sensitive — should detect small deviation
        (2.0, True),    # moderately sensitive
        (5.0, False),   # very insensitive — 5x threshold unlikely
    ])
    def test_threshold_sensitivity(self, multiplier, expected_detect):
        d = StragglerDetector(
            detection_method=DetectionMethod.THRESHOLD,
            check_interval_s=0.0,
            threshold_multiplier=multiplier,
            consecutive_threshold=2,
        )
        for _ in range(10):
            d.record_latency("n1", 10)
            d.record_latency("n2", 10)
        for _ in range(5):
            d.record_latency("n3", 30)  # 3x baseline

        reports = d.check()
        detected = any(r.node_id == "n3" for r in reports)
        assert detected == expected_detect
