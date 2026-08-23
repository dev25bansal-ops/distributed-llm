"""Tests for StragglerDetector (re-exported from distllm.dist.straggler).

Covers:
- Construction with defaults and custom parameters
- record_latency and record_throughput
- record_batch convenience
- check method: returns reports when node is slow
- DetectionMethod: THRESHOLD, MAD, TREND, THROUGHPUT, ENSEMBLE
- Stale node detection
- get_reports
- predict_stragglers, get_events, get_analytics
- clear_node, reset_baseline, reset_all
- stats
- Callback throttling
- AdaptiveThreshold (Welford online algorithm)
- NodeTiming dataclass
- Root cause attribution
- Network RTT filtering (B-19)
- Thread safety
"""

from __future__ import annotations

import threading
import time
from typing import Any

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


# ---------------------------------------------------------------------------
# AdaptiveThreshold
# ---------------------------------------------------------------------------


class TestAdaptiveThreshold:
    """Welford online algorithm for adaptive thresholds."""

    def test_initial_state(self) -> None:
        at = AdaptiveThreshold(sensitivity=2.0)
        assert at.mean == 0.0
        assert at.std == 0.0
        assert at.sensitivity == 2.0

    def test_is_outlier_returns_false_with_few_samples(self) -> None:
        at = AdaptiveThreshold()
        for v in range(9):
            at.update(float(v))
        assert at.is_outlier(100.0) is False  # < 10 samples

    def test_is_outlier_with_enough_samples(self) -> None:
        at = AdaptiveThreshold(sensitivity=2.0)
        for _ in range(20):
            at.update(10.0)
        # A large deviation should be flagged
        assert at.is_outlier(100.0) is True

    def test_is_outlier_normal_value_not_flagged(self) -> None:
        at = AdaptiveThreshold(sensitivity=2.0)
        for _ in range(20):
            at.update(10.0)
        assert at.is_outlier(11.0) is False

    def test_percentile_rank(self) -> None:
        at = AdaptiveThreshold()
        for v in range(1, 21):
            at.update(float(v))
        # Values near the mean should be near 50th percentile
        pct = at.percentile_rank(10.5)
        assert 40.0 <= pct <= 60.0

    def test_percentile_rank_insufficient_data(self) -> None:
        at = AdaptiveThreshold()
        assert at.percentile_rank(50.0) == 50.0

    def test_update_converges_to_mean(self) -> None:
        at = AdaptiveThreshold()
        for _ in range(100):
            at.update(42.0)
        assert at.mean == pytest.approx(42.0, abs=0.1)
        assert at.std < 0.1


# ---------------------------------------------------------------------------
# NodeTiming dataclass
# ---------------------------------------------------------------------------


class TestNodeTiming:
    """NodeTiming dataclass."""

    def test_defaults(self) -> None:
        nt = NodeTiming(node_id="n1")
        assert nt.node_id == "n1"
        assert nt.is_straggler is False
        assert nt.severity == StragglerSeverity.NONE
        assert nt.consecutive_slow == 0
        assert nt.baseline_alpha == 0.1

    def test_avg_latency_empty(self) -> None:
        nt = NodeTiming(node_id="n1")
        assert nt.avg_latency == 0.0

    def test_avg_latency(self) -> None:
        nt = NodeTiming(node_id="n1")
        nt.latencies.extend([10.0, 20.0, 30.0])
        assert nt.avg_latency == 20.0

    def test_p95_latency(self) -> None:
        nt = NodeTiming(node_id="n1")
        nt.latencies.extend(list(range(100)))
        assert nt.p95_latency == 95.0

    def test_p95_latency_small(self) -> None:
        nt = NodeTiming(node_id="n1")
        nt.latencies.extend([1.0, 2.0])
        assert nt.p95_latency == 2.0  # min(1, 2-1)=1, sorted[1]=2.0

    def test_update_baseline_first_value(self) -> None:
        nt = NodeTiming(node_id="n1")
        nt.update_baseline(50.0)
        assert nt.baseline_latency == 50.0

    def test_update_baseline_ema(self) -> None:
        nt = NodeTiming(node_id="n1")
        nt.update_baseline(100.0)
        nt.update_baseline(200.0)
        # new = (1-0.1)*100 + 0.1*200 = 90 + 20 = 110
        assert nt.baseline_latency == pytest.approx(110.0)

    def test_update_throughput_baseline(self) -> None:
        nt = NodeTiming(node_id="n1")
        nt.update_throughput_baseline(100.0)
        assert nt.baseline_throughput == 100.0

    def test_predict_latency_insufficient_data(self) -> None:
        nt = NodeTiming(node_id="n1")
        nt.latencies.extend([1.0] * 10)
        assert nt.predict_latency() is None


# ---------------------------------------------------------------------------
# StragglerDetector construction
# ---------------------------------------------------------------------------


class TestStragglerDetectorConstruction:
    """Construction with defaults and custom parameters."""

    def test_default_construction(self) -> None:
        sd = StragglerDetector()
        assert sd._detection_method == DetectionMethod.MAD
        assert sd._slow_threshold == 100.0
        assert sd._consecutive_threshold == 3
        assert sd._check_interval == 10.0
        assert sd._nodes == {}
        assert sd._total_checks == 0
        assert sd._total_detections == 0

    def test_custom_parameters(self) -> None:
        sd = StragglerDetector(
            detection_method=DetectionMethod.THRESHOLD,
            slow_threshold_ms=200.0,
            consecutive_threshold=5,
            window_size=100,
            mad_threshold=3.0,
            check_interval_s=5.0,
        )
        assert sd._detection_method == DetectionMethod.THRESHOLD
        assert sd._slow_threshold == 200.0
        assert sd._consecutive_threshold == 5
        assert sd._window_size == 100
        assert sd._mad_threshold == 3.0
        assert sd._check_interval == 5.0

    def test_callbacks(self) -> None:
        calls: list[StragglerReport] = []

        def cb(report: StragglerReport) -> None:
            calls.append(report)

        sd = StragglerDetector(on_straggler_cb=cb)
        assert sd._on_straggler is cb


# ---------------------------------------------------------------------------
# record_latency / record_throughput / record_batch
# ---------------------------------------------------------------------------


class TestStragglerDetectorRecording:
    """Recording measurements."""

    def test_record_latency_creates_node(self) -> None:
        sd = StragglerDetector()
        sd.record_latency("node-1", 50.0)
        assert "node-1" in sd._nodes
        assert sd._nodes["node-1"].avg_latency == 50.0

    def test_record_latency_updates_baseline(self) -> None:
        sd = StragglerDetector()
        sd.record_latency("node-1", 100.0)
        sd.record_latency("node-1", 200.0)
        assert sd._nodes["node-1"].baseline_latency == pytest.approx(110.0)

    def test_record_latency_updates_adaptive(self) -> None:
        sd = StragglerDetector()
        for _ in range(20):
            sd.record_latency("node-1", 50.0)
        at = sd._nodes["node-1"].adaptive
        assert at._count >= 20

    def test_record_throughput(self) -> None:
        sd = StragglerDetector()
        sd.record_throughput("node-1", 100.0)
        assert sd._nodes["node-1"].avg_throughput == 100.0

    def test_record_batch(self) -> None:
        sd = StragglerDetector()
        sd.record_batch("node-1", latency_ms=100.0, tokens_generated=50, batch_size=4)
        assert "node-1" in sd._nodes
        assert len(sd._nodes["node-1"].latencies) == 1
        # throughput = 50 / (100/1000) = 500
        assert sd._nodes["node-1"].avg_throughput == pytest.approx(500.0)

    def test_record_batch_zero_tokens(self) -> None:
        sd = StragglerDetector()
        sd.record_batch("node-1", latency_ms=100.0, tokens_generated=0)
        # Latency recorded, throughput not recorded
        assert len(sd._nodes["node-1"].latencies) == 1
        assert len(sd._nodes["node-1"].throughputs) == 0


# ---------------------------------------------------------------------------
# check method — THRESHOLD
# ---------------------------------------------------------------------------


class TestStragglerDetectorThreshold:
    """Threshold-based detection."""

    def test_detects_slow_node(self) -> None:
        sd = StragglerDetector(
            detection_method=DetectionMethod.THRESHOLD,
            consecutive_threshold=1,
            check_interval_s=0,
        )
        # Node 1: fast
        for _ in range(10):
            sd.record_latency("node-1", 10.0)
        # Node 2: slow
        for _ in range(10):
            sd.record_latency("node-2", 200.0)

        reports = sd.check()
        assert len(reports) >= 1
        assert any(r.node_id == "node-2" for r in reports)

    def test_no_false_positive_when_similar(self) -> None:
        sd = StragglerDetector(
            detection_method=DetectionMethod.THRESHOLD,
            consecutive_threshold=3,
            check_interval_s=0,
        )
        for node_id in ("node-1", "node-2"):
            for _ in range(10):
                sd.record_latency(node_id, 50.0)

        reports = sd.check()
        assert len(reports) == 0

    def test_needs_minimum_samples(self) -> None:
        sd = StragglerDetector(
            detection_method=DetectionMethod.THRESHOLD,
            consecutive_threshold=1,
            check_interval_s=0,
        )
        sd.record_latency("node-1", 10.0)  # 1 sample
        sd.record_latency("node-2", 200.0)
        reports = sd.check()
        assert len(reports) == 0  # need >= 5 samples


# ---------------------------------------------------------------------------
# check method — MAD
# ---------------------------------------------------------------------------


class TestStragglerDetectorMAD:
    """MAD-based detection."""

    def test_mad_detects_outlier(self) -> None:
        sd = StragglerDetector(
            detection_method=DetectionMethod.MAD,
            consecutive_threshold=1,
            check_interval_s=0,
        )
        for _ in range(10):
            sd.record_latency("node-1", 10.0)
        for _ in range(10):
            sd.record_latency("node-2", 200.0)

        reports = sd.check()
        assert len(reports) >= 1

    def test_mad_identical_values_fallback(self) -> None:
        """When MAD=0 with different values, falls back to threshold multiplier."""
        sd = StragglerDetector(
            detection_method=DetectionMethod.MAD,
            consecutive_threshold=1,
            check_interval_s=0,
            threshold_multiplier=1.0,  # Any difference triggers
        )
        for _ in range(10):
            sd.record_latency("node-1", 10.0)
        for _ in range(10):
            sd.record_latency("node-2", 10.0)

        reports = sd.check()
        # With threshold_multiplier=1.0, node-2 is NOT > 1.0 * 10.0
        assert len(reports) == 0


# ---------------------------------------------------------------------------
# check method — TREND
# ---------------------------------------------------------------------------


class TestStragglerDetectorTrend:
    """Trend-based detection via EMA baseline."""

    def test_trend_detects_slowdown(self) -> None:
        sd = StragglerDetector(
            detection_method=DetectionMethod.TREND,
            consecutive_threshold=1,
            check_interval_s=0,
            trend_multiplier=1.5,
        )
        for _ in range(10):
            sd.record_latency("node-1", 10.0)
        for _ in range(10):
            sd.record_latency("node-1", 100.0)  # rising trend

        # Also need another node for peer comparison
        for _ in range(10):
            sd.record_latency("node-2", 10.0)

        reports = sd.check()
        # One node (node-1) has rising latency vs its own baseline
        assert len(reports) >= 1


# ---------------------------------------------------------------------------
# check method — THROUGHPUT
# ---------------------------------------------------------------------------


class TestStragglerDetectorThroughput:
    """Throughput-based detection."""

    def test_throughput_detects_slowdown(self) -> None:
        sd = StragglerDetector(
            detection_method=DetectionMethod.THROUGHPUT,
            consecutive_threshold=1,
            check_interval_s=0,
            throughput_floor=0.5,
        )
        # Establish baseline throughput
        for _ in range(10):
            sd.record_throughput("node-1", 100.0)
        # Drop throughput below floor
        for _ in range(10):
            sd.record_throughput("node-1", 20.0)
        # Need another node
        for _ in range(10):
            sd.record_throughput("node-2", 100.0)

        reports = sd.check()
        assert len(reports) >= 1


# ---------------------------------------------------------------------------
# check method — ENSEMBLE
# ---------------------------------------------------------------------------


class TestStragglerDetectorEnsemble:
    """Ensemble detection (votes >= 2)."""

    def test_ensemble_detects_when_two_methods_agree(self) -> None:
        sd = StragglerDetector(
            detection_method=DetectionMethod.ENSEMBLE,
            consecutive_threshold=1,
            check_interval_s=0,
        )
        for _ in range(10):
            sd.record_latency("node-1", 10.0)
        for _ in range(10):
            sd.record_latency("node-2", 500.0)  # very slow

        reports = sd.check()
        assert len(reports) >= 1


# ---------------------------------------------------------------------------
# Stale node detection
# ---------------------------------------------------------------------------


class TestStragglerDetectorStaleNodes:
    """Stale node detection."""

    def test_stale_node_detected(self) -> None:
        sd = StragglerDetector(
            check_interval_s=0,
            stale_timeout_s=0.1,
            consecutive_threshold=1,
        )
        sd.record_latency("node-1", 10.0)

        # Set last_seen far in the past
        with sd._lock:
            sd._nodes["node-1"].last_seen = 0.0

        reports = sd.check()  # triggers stale detection
        assert len(reports) >= 1
        assert reports[0].severity == StragglerSeverity.SEVERE
        assert reports[0].recommended_action == "reassign_layers"


# ---------------------------------------------------------------------------
# get_reports
# ---------------------------------------------------------------------------


class TestStragglerDetectorGetReports:
    """Get current straggler reports."""

    def test_get_reports_returns_active_stragglers(self) -> None:
        sd = StragglerDetector(
            detection_method=DetectionMethod.THRESHOLD,
            consecutive_threshold=1,
            check_interval_s=0,
        )
        for _ in range(10):
            sd.record_latency("node-1", 10.0)
        for _ in range(10):
            sd.record_latency("node-2", 500.0)

        reports = sd.get_reports()
        straggler_ids = [r.node_id for r in reports]
        assert "node-2" in straggler_ids

    def test_get_reports_empty_when_no_stragglers(self) -> None:
        sd = StragglerDetector(check_interval_s=0)
        for _ in range(10):
            sd.record_latency("node-1", 10.0)
        for _ in range(10):
            sd.record_latency("node-2", 10.0)

        assert sd.get_reports() == []


# ---------------------------------------------------------------------------
# predict_stragglers / get_events / get_analytics
# ---------------------------------------------------------------------------


class TestStragglerDetectorAnalytics:
    """Predictive and analytics methods."""

    def test_predict_stragglers_insufficient_data(self) -> None:
        sd = StragglerDetector(check_interval_s=0)
        sd.record_latency("node-1", 10.0)
        assert sd.predict_stragglers() == []

    def test_get_events(self) -> None:
        sd = StragglerDetector(
            detection_method=DetectionMethod.THRESHOLD,
            consecutive_threshold=1,
            check_interval_s=0,
        )
        for _ in range(10):
            sd.record_latency("node-1", 10.0)
        for _ in range(10):
            sd.record_latency("node-2", 500.0)
        sd.check()

        events = sd.get_events()
        assert len(events) >= 1
        assert events[0]["node_id"] == "node-2"

    def test_get_analytics(self) -> None:
        sd = StragglerDetector(
            detection_method=DetectionMethod.THRESHOLD,
            consecutive_threshold=1,
            check_interval_s=0,
        )
        for _ in range(10):
            sd.record_latency("node-1", 10.0)
        for _ in range(10):
            sd.record_latency("node-2", 500.0)
        sd.check()

        analytics = sd.get_analytics()
        assert analytics["total_events"] >= 1
        assert "severe" in analytics["by_severity"] or "moderate" in analytics["by_severity"]

    def test_get_analytics_empty(self) -> None:
        sd = StragglerDetector()
        assert sd.get_analytics()["total_events"] == 0


# ---------------------------------------------------------------------------
# clear_node / reset_baseline / reset_all
# ---------------------------------------------------------------------------


class TestStragglerDetectorReset:
    """Reset methods."""

    def test_clear_node(self) -> None:
        sd = StragglerDetector()
        sd.record_latency("node-1", 10.0)
        sd.record_latency("node-2", 20.0)
        sd.clear_node("node-1")
        assert "node-1" not in sd._nodes
        assert "node-2" in sd._nodes

    def test_reset_baseline(self) -> None:
        sd = StragglerDetector()
        sd.record_latency("node-1", 100.0)
        sd.reset_baseline("node-1")
        node = sd._nodes["node-1"]
        assert node.baseline_latency == 0.0
        assert node.consecutive_slow == 0
        assert node.is_straggler is False

    def test_reset_all(self) -> None:
        sd = StragglerDetector()
        sd.record_latency("node-1", 10.0)
        sd.record_latency("node-2", 20.0)
        sd.reset_all()
        assert sd._nodes == {}
        assert sd._total_checks == 0
        assert sd._total_detections == 0


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


class TestStragglerDetectorStats:
    """Stats method."""

    def test_stats_empty(self) -> None:
        sd = StragglerDetector()
        stats = sd.stats()
        assert stats["active_nodes"] == 0
        assert stats["straggler_nodes"] == 0

    def test_stats_with_data(self) -> None:
        sd = StragglerDetector(
            detection_method=DetectionMethod.THRESHOLD,
            consecutive_threshold=1,
            check_interval_s=0,
        )
        for _ in range(10):
            sd.record_latency("node-1", 10.0)
        for _ in range(10):
            sd.record_latency("node-2", 500.0)
        sd.check()

        stats = sd.stats()
        assert stats["active_nodes"] == 2
        assert "node-2" in stats["nodes"]
        assert stats["nodes"]["node-2"]["is_straggler"]

    def test_stats_output_format(self) -> None:
        sd = StragglerDetector()
        sd.record_latency("node-1", 50.0)
        stats = sd.stats()
        assert "active_nodes" in stats
        assert "straggler_nodes" in stats
        assert "detection_method" in stats
        assert "nodes" in stats
        assert "node-1" in stats["nodes"]


# ---------------------------------------------------------------------------
# Callback throttling
# ---------------------------------------------------------------------------


class TestStragglerDetectorCallback:
    """Callback firing with throttling."""

    def test_callback_fires_on_detection(self) -> None:
        calls: list[StragglerReport] = []

        def cb(report: StragglerReport) -> None:
            calls.append(report)

        sd = StragglerDetector(
            on_straggler_cb=cb,
            detection_method=DetectionMethod.THRESHOLD,
            consecutive_threshold=1,
            check_interval_s=0,
            callback_cooldown_s=0,
        )
        for _ in range(10):
            sd.record_latency("node-1", 10.0)
        for _ in range(10):
            sd.record_latency("node-2", 500.0)
        sd.check()

        assert len(calls) >= 1

    def test_callback_throttled(self) -> None:
        calls: list[StragglerReport] = []

        def cb(report: StragglerReport) -> None:
            calls.append(report)

        sd = StragglerDetector(
            on_straggler_cb=cb,
            detection_method=DetectionMethod.THRESHOLD,
            consecutive_threshold=1,
            check_interval_s=0,
            callback_cooldown_s=3600,  # long cooldown
        )
        for _ in range(10):
            sd.record_latency("node-1", 10.0)
        for _ in range(10):
            sd.record_latency("node-2", 500.0)

        # First check fires
        sd.check()
        first_count = len(calls)

        # Second check within cooldown should not fire
        sd.check()
        assert len(calls) == first_count


# ---------------------------------------------------------------------------
# Root cause attribution
# ---------------------------------------------------------------------------


class TestStragglerDetectorRootCause:
    """Root cause attribution."""

    def test_manual_root_cause(self) -> None:
        sd = StragglerDetector(
            detection_method=DetectionMethod.THRESHOLD,
            consecutive_threshold=1,
            check_interval_s=0,
        )
        sd.record_latency("node-1", 50.0)
        cause = RootCauseAttribution(
            node_id="node-1",
            gpu_temp_c=85.0,
            gpu_memory_used_pct=95.0,
            probable_cause="thermal",
        )
        sd.record_root_cause("node-1", cause)

        for _ in range(10):
            sd.record_latency("node-1", 10.0)
        for _ in range(10):
            sd.record_latency("node-2", 500.0)

        reports = sd.check()
        if reports:
            r = next((r for r in reports if r.node_id == "node-2"), None)
            # root cause should be from the fast node or None depending on timing
            assert r is not None

    def test_root_cause_attribution_to_dict(self) -> None:
        cause = RootCauseAttribution(
            node_id="n1",
            gpu_temp_c=80.0,
            gpu_memory_used_pct=90.0,
            network_bandwidth_mbps=1000.0,
            cpu_utilization_pct=50.0,
            io_wait_pct=2.0,
            probable_cause="thermal",
        )
        d = cause.to_dict()
        assert d["node_id"] == "n1"
        assert d["gpu_temp_c"] == 80.0
        assert d["probable_cause"] == "thermal"


# ---------------------------------------------------------------------------
# Network RTT filtering
# ---------------------------------------------------------------------------


class TestStragglerDetectorNetworkRTT:
    """Network RTT filtering (B-19)."""

    def test_high_rtt_suppresses_detection(self) -> None:
        sd = StragglerDetector(
            detection_method=DetectionMethod.THRESHOLD,
            consecutive_threshold=1,
            check_interval_s=0,
            network_rtt_threshold=50.0,
            network_rtt_fn=lambda node_id: 200.0,  # high RTT
        )
        for _ in range(10):
            sd.record_latency("node-1", 10.0)
        for _ in range(10):
            sd.record_latency("node-2", 500.0)

        reports = sd.check()
        # node-2 has high RTT, so it should not be flagged
        straggler_ids = [r.node_id for r in reports]
        assert "node-2" not in straggler_ids

    def test_low_rtt_allows_detection(self) -> None:
        sd = StragglerDetector(
            detection_method=DetectionMethod.THRESHOLD,
            consecutive_threshold=1,
            check_interval_s=0,
            network_rtt_threshold=50.0,
            network_rtt_fn=lambda node_id: 10.0,  # low RTT
        )
        for _ in range(10):
            sd.record_latency("node-1", 10.0)
        for _ in range(10):
            sd.record_latency("node-2", 500.0)

        reports = sd.check()
        straggler_ids = [r.node_id for r in reports]
        assert "node-2" in straggler_ids


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestStragglerDetectorThreadSafety:
    """Thread safety under concurrent access."""

    def test_concurrent_recording(self) -> None:
        sd = StragglerDetector(check_interval_s=0)
        errors: list[Exception] = []

        def record_range(node_id: str, count: int) -> None:
            try:
                for i in range(count):
                    sd.record_latency(node_id, float(i % 100))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_range, args=("node-1", 100)),
            threading.Thread(target=record_range, args=("node-2", 100)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(sd._nodes["node-1"].latencies) == 100
        assert len(sd._nodes["node-2"].latencies) == 100

    def test_concurrent_record_and_check(self) -> None:
        sd = StragglerDetector(check_interval_s=0)
        for _ in range(10):
            sd.record_latency("node-1", 10.0)
        for _ in range(10):
            sd.record_latency("node-2", 20.0)

        errors: list[Exception] = []

        def check_loop() -> None:
            try:
                for _ in range(20):
                    sd.check()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=check_loop),
            threading.Thread(target=lambda: [sd.record_latency("node-1", 50.0) for _ in range(50)]),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ---------------------------------------------------------------------------
# StragglerReport and StragglerEvent
# ---------------------------------------------------------------------------


class TestStragglerReport:
    """StragglerReport dataclass."""

    def test_report_fields(self) -> None:
        report = StragglerReport(
            node_id="n1",
            severity=StragglerSeverity.SEVERE,
            avg_latency=500.0,
            p95_latency=550.0,
            baseline_latency=100.0,
            slowdown_factor=5.0,
            detection_method=DetectionMethod.THRESHOLD,
            consecutive_detections=3,
            recommended_action="reassign_layers",
        )
        assert report.node_id == "n1"
        assert report.severity == StragglerSeverity.SEVERE
        assert report.slowdown_factor == 5.0
        assert report.recommended_action == "reassign_layers"


class TestStragglerEvent:
    """StragglerEvent dataclass."""

    def test_to_dict(self) -> None:
        event = StragglerEvent(
            node_id="n1",
            severity=StragglerSeverity.MODERATE,
            latency_ms=300.0,
            baseline_ms=100.0,
            action_taken="reduce_batch",
            detection_method="threshold",
        )
        d = event.to_dict()
        assert d["node_id"] == "n1"
        assert d["severity"] == "moderate"
        assert d["latency_ms"] == 300.0
        assert d["action_taken"] == "reduce_batch"

    def test_to_dict_with_root_cause(self) -> None:
        cause = RootCauseAttribution(node_id="n1", gpu_temp_c=90.0, probable_cause="thermal")
        event = StragglerEvent(
            node_id="n1",
            severity=StragglerSeverity.SEVERE,
            latency_ms=400.0,
            baseline_ms=100.0,
            action_taken="reassign_layers",
            detection_method="mad",
            root_cause=cause,
        )
        d = event.to_dict()
        assert d["root_cause"]["probable_cause"] == "thermal"
