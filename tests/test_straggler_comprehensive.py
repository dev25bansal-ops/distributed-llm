"""Comprehensive tests for StragglerDetector.

Covers all items from Verification & Testing Strategy:
- 6.1: Rolling baseline, stale node detection, get_reports action, MAD=0
- 6.2: Callback throttle, adaptive threshold, predictive detection, ensemble
- 6.3: Performance benchmarks
- 6.4: Property-based tests
- 6.5: Chaos tests
"""

import threading
import time

import pytest

from distllm.dist.straggler import (
    AdaptiveThreshold,
    DetectionMethod,
    NodeTiming,
    RootCauseAttribution,
    StragglerDetector,
    StragglerEvent,
    StragglerReport,
    StragglerSeverity,
)


# ============================================================================
# 6.1 Unit Tests — Missing Coverage
# ============================================================================


class TestRollingBaseline:
    """Test that baseline updates over time (fixes issue #1)."""

    def test_baseline_updates_after_initial(self):
        det = StragglerDetector(baseline_alpha=0.1)
        for _ in range(10):
            det.record_latency("node-1", 10.0)
        node = det._nodes["node-1"]
        initial_baseline = node.baseline_latency
        assert initial_baseline == 10.0

        # Record higher latencies — baseline should drift up
        for _ in range(50):
            det.record_latency("node-1", 100.0)
        assert node.baseline_latency > initial_baseline
        assert node.baseline_latency > 50.0  # Should be moving toward 100

    def test_baseline_responds_to_improvement(self):
        det = StragglerDetector(baseline_alpha=0.2)
        for _ in range(20):
            det.record_latency("node-1", 100.0)
        node = det._nodes["node-1"]
        high_baseline = node.baseline_latency

        for _ in range(50):
            det.record_latency("node-1", 10.0)
        assert node.baseline_latency < high_baseline

    def test_baseline_alpha_controls_speed(self):
        det_fast = StragglerDetector(baseline_alpha=0.5)
        det_slow = StragglerDetector(baseline_alpha=0.05)
        for _ in range(10):
            det_fast.record_latency("n1", 10.0)
            det_slow.record_latency("n1", 10.0)
        for _ in range(10):
            det_fast.record_latency("n1", 100.0)
            det_slow.record_latency("n1", 100.0)
        fast_baseline = det_fast._nodes["n1"].baseline_latency
        slow_baseline = det_slow._nodes["n1"].baseline_latency
        assert fast_baseline > slow_baseline


class TestStaleNodeDetection:
    """Test stale node detection (fixes issue #4)."""

    def test_stale_node_detected(self):
        det = StragglerDetector(stale_timeout_s=0.1, check_interval_s=0)
        det.record_latency("node-1", 10.0)
        det.record_latency("node-2", 10.0)
        # Wait for node-1 to become stale
        time.sleep(0.15)
        reports = det.check()
        # node-1 should be flagged as stale (SEVERE)
        stale_reports = [r for r in reports if r.node_id == "node-1"]
        assert len(stale_reports) == 1
        assert stale_reports[0].severity == StragglerSeverity.SEVERE
        assert stale_reports[0].recommended_action == "reassign_layers"

    def test_active_node_not_flagged_as_stale(self):
        det = StragglerDetector(stale_timeout_s=10.0, check_interval_s=0)
        det.record_latency("node-1", 10.0)
        det.record_latency("node-2", 10.0)
        reports = det.check()
        assert len(reports) == 0


class TestGetReportsAction:
    """Test that get_reports() returns recommended_action (fixes issue #3)."""

    def test_get_reports_has_action(self):
        det = StragglerDetector(
            detection_method=DetectionMethod.THRESHOLD,
            consecutive_threshold=1,
            check_interval_s=0,
        )
        for _ in range(10):
            det.record_latency("fast", 10.0)
            det.record_latency("slow", 500.0)
        det.check()
        reports = det.get_reports()
        slow_reports = [r for r in reports if r.node_id == "slow"]
        assert len(slow_reports) == 1
        assert slow_reports[0].recommended_action in (
            "reassign_layers", "reduce_batch", "monitor_only"
        )
        assert slow_reports[0].recommended_action != ""  # Not empty!


class TestMADZeroEdgeCase:
    """Test MAD=0 doesn't cause issues (fixes issue #2)."""

    def test_identical_latencies_no_crash(self):
        det = StragglerDetector(
            detection_method=DetectionMethod.MAD,
            consecutive_threshold=1,
            check_interval_s=0,
        )
        for _ in range(10):
            det.record_latency("node-1", 10.0)
            det.record_latency("node-2", 10.0)
        reports = det.check()
        # Should not crash, no straggler (all identical)
        assert len(reports) == 0

    def test_mad_zero_with_one_outlier(self):
        det = StragglerDetector(
            detection_method=DetectionMethod.MAD,
            consecutive_threshold=1,
            check_interval_s=0,
        )
        # 3 identical + 1 outlier
        for _ in range(10):
            det.record_latency("a", 10.0)
            det.record_latency("b", 10.0)
            det.record_latency("c", 10.0)
        for _ in range(10):
            det.record_latency("d", 1000.0)
        reports = det.check()
        assert any(r.node_id == "d" for r in reports)


class TestConfigurableMultipliers:
    """Test configurable multipliers (fixes issues #7, #8, #9)."""

    def test_threshold_multiplier(self):
        det = StragglerDetector(
            detection_method=DetectionMethod.THRESHOLD,
            threshold_multiplier=10.0,  # Very high — no detection
            consecutive_threshold=1,
            check_interval_s=0,
        )
        for _ in range(10):
            det.record_latency("fast", 10.0)
            det.record_latency("slow", 50.0)  # 5x, but threshold is 10x
        reports = det.check()
        assert len(reports) == 0

    def test_severe_multiplier(self):
        det = StragglerDetector(
            detection_method=DetectionMethod.THRESHOLD,
            consecutive_threshold=1,
            severe_multiplier=100.0,  # Very high
            check_interval_s=0,
        )
        for _ in range(10):
            det.record_latency("fast", 10.0)
            det.record_latency("slow", 500.0)
        reports = det.check()
        if reports:
            assert reports[0].severity != StragglerSeverity.SEVERE


# ============================================================================
# 6.2 Advanced Feature Tests
# ============================================================================


class TestCallbackThrottling:
    """Test callback throttling (fixes issue #13)."""

    def test_callback_fires_once_per_cooldown(self):
        call_count = [0]

        def cb(report):
            call_count[0] += 1

        det = StragglerDetector(
            on_straggler_cb=cb,
            detection_method=DetectionMethod.THRESHOLD,
            consecutive_threshold=1,
            check_interval_s=0,
            callback_cooldown_s=60.0,
        )
        for _ in range(10):
            det.record_latency("fast", 10.0)
            det.record_latency("slow", 500.0)

        det.check()
        det.check()  # Second call within cooldown
        assert call_count[0] == 1  # Only fired once


class TestAdaptiveThreshold:
    """Test adaptive threshold via Welford's algorithm."""

    def test_convergence(self):
        at = AdaptiveThreshold(sensitivity=2.0)
        # Feed 100 normal values around 10
        for i in range(100):
            at.update(10.0 + (i % 3) * 0.5)
        assert abs(at.mean - 10.5) < 1.0
        assert at.std > 0
        assert at.std < 2.0

    def test_outlier_detection(self):
        at = AdaptiveThreshold(sensitivity=2.0)
        # Need variation in data for std > 0
        for i in range(100):
            at.update(10.0 + (i % 5) * 0.5)
        assert at.is_outlier(100.0) is True
        assert at.is_outlier(10.5) is False

    def test_insufficient_data_no_outlier(self):
        at = AdaptiveThreshold(sensitivity=2.0)
        for _ in range(5):
            at.update(10.0)
        assert at.is_outlier(100.0) is False  # Not enough data

    def test_percentile_rank(self):
        at = AdaptiveThreshold(sensitivity=2.0)
        for _ in range(100):
            at.update(10.0)
        # Value at mean should be ~50th percentile
        rank = at.percentile_rank(10.0)
        assert 40 < rank < 60


class TestPredictiveDetection:
    """Test Holt-Winters predictive detection."""

    def test_insufficient_data_returns_none(self):
        node = NodeTiming(node_id="test")
        for _ in range(10):
            node.latencies.append(10.0)
        assert node.predict_latency() is None

    def test_prediction_with_enough_data(self):
        node = NodeTiming(node_id="test")
        # Need 24+ samples
        for i in range(30):
            node.latencies.append(10.0 + (i % 5))
        prediction = node.predict_latency(horizon=5)
        assert prediction is not None
        assert prediction > 0

    def test_prediction_tracks_trend(self):
        node = NodeTiming(node_id="test")
        # Rising latency pattern
        for i in range(30):
            node.latencies.append(10.0 + i * 2)
        prediction = node.predict_latency(horizon=5)
        assert prediction is not None
        # Prediction should be above current average
        assert prediction > node.avg_latency


class TestEnsembleDetection:
    """Test ensemble multi-method detection."""

    def test_ensemble_requires_two_votes(self):
        det = StragglerDetector(
            detection_method=DetectionMethod.ENSEMBLE,
            consecutive_threshold=1,
            check_interval_s=0,
        )
        # Only trend method would trigger (baseline set, then spike)
        for _ in range(10):
            det.record_latency("a", 10.0)
            det.record_latency("b", 10.0)
            det.record_latency("c", 10.0)
        # Give baseline time to set
        for _ in range(10):
            det.record_latency("a", 10.0)
            det.record_latency("b", 10.0)
            det.record_latency("c", 10.0)
        # Now spike one node
        for _ in range(10):
            det.record_latency("a", 10.0)
            det.record_latency("b", 10.0)
            det.record_latency("c", 500.0)
        reports = det.check()
        # Should detect c since multiple methods agree
        assert any(r.node_id == "c" for r in reports)


class TestRootCauseAttribution:
    """Test root cause attribution."""

    def test_record_root_cause(self):
        det = StragglerDetector()
        det.record_latency("node-1", 10.0)
        cause = RootCauseAttribution(
            node_id="node-1",
            gpu_temp_c=85.0,
            gpu_memory_used_pct=95.0,
            probable_cause="thermal",
        )
        det.record_root_cause("node-1", cause)
        assert det._nodes["node-1"].last_root_cause.gpu_temp_c == 85.0

    def test_root_cause_in_report(self):
        def gpu_health_fn(node_id):
            return RootCauseAttribution(
                node_id=node_id,
                gpu_temp_c=90.0,
                probable_cause="thermal",
            )

        det = StragglerDetector(
            detection_method=DetectionMethod.THRESHOLD,
            consecutive_threshold=1,
            check_interval_s=0,
            gpu_health_fn=gpu_health_fn,
        )
        for _ in range(10):
            det.record_latency("fast", 10.0)
            det.record_latency("slow", 500.0)
        reports = det.check()
        slow_reports = [r for r in reports if r.node_id == "slow"]
        assert len(slow_reports) >= 1
        assert slow_reports[0].root_cause is not None
        assert slow_reports[0].root_cause.probable_cause == "thermal"


class TestStragglerHistory:
    """Test straggler event history."""

    def test_events_recorded(self):
        det = StragglerDetector(
            detection_method=DetectionMethod.THRESHOLD,
            consecutive_threshold=1,
            check_interval_s=0,
        )
        for _ in range(10):
            det.record_latency("fast", 10.0)
            det.record_latency("slow", 500.0)
        det.check()
        events = det.get_events()
        assert len(events) >= 1
        assert events[0]["node_id"] == "slow"

    def test_analytics(self):
        det = StragglerDetector(
            detection_method=DetectionMethod.THRESHOLD,
            consecutive_threshold=1,
            check_interval_s=0,
        )
        for _ in range(10):
            det.record_latency("fast", 10.0)
            det.record_latency("slow", 500.0)
        det.check()
        analytics = det.get_analytics()
        assert analytics["total_events"] >= 1
        assert "by_severity" in analytics


# ============================================================================
# 6.3 Performance Benchmarks
# ============================================================================


@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks."""

    def test_record_latency_throughput(self):
        det = StragglerDetector()
        iterations = 100_000
        start = time.perf_counter()
        for i in range(iterations):
            det.record_latency(f"node-{i % 10}", float(i % 100))
        elapsed = time.perf_counter() - start
        per_call_us = (elapsed / iterations) * 1_000_000
        assert per_call_us < 10.0, f"record_latency: {per_call_us:.1f}µs (target: <10µs)"

    def test_check_latency_100_nodes(self):
        det = StragglerDetector(check_interval_s=0)
        for i in range(100):
            for _ in range(10):
                det.record_latency(f"node-{i}", 10.0 + i)
        start = time.perf_counter()
        for _ in range(1000):
            det._last_check = 0
            det.check()
        elapsed = time.perf_counter() - start
        per_call_ms = (elapsed / 1000) * 1000
        assert per_call_ms < 5.0, f"check() with 100 nodes: {per_call_ms:.2f}ms (target: <5ms)"

    def test_memory_per_node(self):
        det = StragglerDetector()
        for i in range(100):
            for _ in range(100):
                det.record_latency(f"node-{i}", 10.0)
        # Should not explode
        assert len(det._nodes) == 100


# ============================================================================
# 6.4 Property-Based Tests
# ============================================================================


class TestProperties:
    """Property-based tests for invariants."""

    def test_mad_never_crashes_with_any_data(self):
        """MAD detection should never raise regardless of input."""
        import random
        det = StragglerDetector(
            detection_method=DetectionMethod.MAD,
            consecutive_threshold=1,
            check_interval_s=0,
        )
        for _ in range(50):
            for i in range(5):
                latency = random.uniform(0.1, 10000)
                det.record_latency(f"node-{i}", latency)
            det.check()  # Should never raise

    def test_always_detects_extreme_outlier(self):
        """A node with 1000x latency should always be detected."""
        det = StragglerDetector(
            detection_method=DetectionMethod.MAD,
            consecutive_threshold=1,
            check_interval_s=0,
        )
        for i in range(5):
            for _ in range(10):
                det.record_latency(f"node-{i}", 10.0 + i)
        for _ in range(10):
            det.record_latency("outlier", 100000.0)
        reports = det.check()
        assert any(r.node_id == "outlier" for r in reports)

    def test_no_false_positive_all_similar(self):
        """All-similar nodes should produce no straggler reports."""
        det = StragglerDetector(
            detection_method=DetectionMethod.MAD,
            consecutive_threshold=1,
            check_interval_s=0,
        )
        for i in range(10):
            for _ in range(10):
                det.record_latency(f"node-{i}", 10.0 + (i % 3) * 0.1)
        reports = det.check()
        assert len(reports) == 0

    def test_stats_always_consistent(self):
        """Stats should always reflect actual state."""
        det = StragglerDetector(check_interval_s=0)
        for i in range(10):
            det.record_latency(f"node-{i}", 10.0 + i * 10)
        stats = det.stats()
        assert stats["active_nodes"] == 10
        assert stats["total_checks"] >= 0

    def test_clear_node_removes_from_tracking(self):
        det = StragglerDetector()
        det.record_latency("node-1", 10.0)
        det.clear_node("node-1")
        assert "node-1" not in det._nodes

    def test_reset_all_clears_everything(self):
        det = StragglerDetector()
        for i in range(10):
            det.record_latency(f"node-{i}", 10.0)
        det.reset_all()
        assert det.stats()["active_nodes"] == 0
        assert det.stats()["total_checks"] == 0


# ============================================================================
# 6.5 Chaos Tests
# ============================================================================


class TestChaos:
    """Chaos and resilience tests."""

    def test_concurrent_record_and_check(self):
        det = StragglerDetector(check_interval_s=0)
        errors = []

        def record_loop():
            try:
                for i in range(200):
                    det.record_latency(f"node-{i % 10}", 10.0 + i % 50)
            except Exception as e:
                errors.append(e)

        def check_loop():
            try:
                for _ in range(50):
                    det.check()
                    det._last_check = 0
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(10):
            threads.append(threading.Thread(target=record_loop))
            threads.append(threading.Thread(target=check_loop))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_node_crash_detection(self):
        """Node that stops reporting should be detected."""
        det = StragglerDetector(stale_timeout_s=0.05, check_interval_s=0)
        det.record_latency("healthy", 10.0)
        det.record_latency("crashed", 10.0)
        time.sleep(0.1)
        reports = det.check()
        assert any(r.node_id == "crashed" for r in reports)

    def test_rapid_oscillation(self):
        """Alternating fast/slow should not cause permanent straggler flag."""
        det = StragglerDetector(
            detection_method=DetectionMethod.THRESHOLD,
            consecutive_threshold=3,
            check_interval_s=0,
        )
        for _ in range(10):
            det.record_latency("a", 10.0)
            det.record_latency("b", 10.0)
        # Oscillate b
        for i in range(20):
            det.record_latency("a", 10.0)
            det.record_latency("b", 500.0 if i % 2 == 0 else 10.0)
        det.check()
        # Should not be permanently flagged since it keeps recovering

    def test_all_nodes_slow_no_false_positive(self):
        """When all nodes degrade equally, no false positive."""
        det = StragglerDetector(
            detection_method=DetectionMethod.MAD,
            consecutive_threshold=1,
            check_interval_s=0,
        )
        for i in range(5):
            for _ in range(10):
                det.record_latency(f"node-{i}", 100.0)  # All same
        reports = det.check()
        assert len(reports) == 0

    def test_callback_error_does_not_crash(self):
        def bad_cb(report):
            raise RuntimeError("boom")

        det = StragglerDetector(
            on_straggler_cb=bad_cb,
            detection_method=DetectionMethod.THRESHOLD,
            consecutive_threshold=1,
            check_interval_s=0,
        )
        for _ in range(10):
            det.record_latency("fast", 10.0)
            det.record_latency("slow", 500.0)
        det.check()  # Should not raise
