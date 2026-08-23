"""Tests for adaptive multi-level compression hierarchy."""

from __future__ import annotations

from distllm.core.adaptive_compression_hierarchy import (
    AdaptiveCompressionHierarchy,
    CompressionLevel,
    RequestQualityReport,
    SessionQualityMonitor,
    _COMPRESSION_HIERARCHY,
)


class TestCompressionLevel:
    def test_hierarchy_ordering(self):
        bits = [l.bits for l in _COMPRESSION_HIERARCHY]
        assert bits == sorted(bits, reverse=True)

    def test_all_levels_have_expected_properties(self):
        for level in _COMPRESSION_HIERARCHY:
            assert level.name
            assert level.method
            assert level.ratio >= 1.0


class TestSessionQualityMonitor:
    def test_init(self):
        m = SessionQualityMonitor()
        assert m.disabled_levels == set()

    def test_baseline_established_after_three_reports(self):
        m = SessionQualityMonitor()
        for i in range(3):
            m.report(RequestQualityReport(f"r-{i}", "INT4", psnr=45.0, accept_rate=0.9))
        assert "INT4" in m._baseline

    def test_regression_disables_after_limit(self):
        m = SessionQualityMonitor(regression_limit=3)
        for i in range(3):
            m.report(RequestQualityReport(f"r-{i}", "INT4", psnr=45.0, accept_rate=0.9))
        for i in range(3):
            m.report(RequestQualityReport(f"bad-{i}", "INT4", psnr=20.0, accept_rate=0.3))
        assert "INT4" in m.disabled_levels

    def test_no_disable_with_acceptable_quality(self):
        m = SessionQualityMonitor(regression_limit=5)
        for i in range(3):
            m.report(RequestQualityReport(f"r-{i}", "INT4", psnr=45.0, accept_rate=0.9))
        for i in range(4):
            m.report(RequestQualityReport(f"ok-{i}", "INT4", psnr=43.0, accept_rate=0.8))
        assert "INT4" not in m.disabled_levels

    def test_best_level_falls_back_when_disabled(self):
        m = SessionQualityMonitor()
        m._disabled.add("INT4")
        m._disabled.add("2-bit")
        level = m.best_level("INT4")
        assert level in ("FP8", "FP16")

    def test_best_level_promotes_when_quality_exceeds_baseline(self):
        m = SessionQualityMonitor()
        m._baseline["FP8"] = 50.0
        for _ in range(5):
            m._history.setdefault("FP8", []).append(50.0)
        level = m.best_level("FP8")
        assert level == "INT4"

    def test_best_level_stays_when_baseline_not_exceeded(self):
        m = SessionQualityMonitor()
        m._baseline["FP8"] = 40.0
        for _ in range(5):
            m._history.setdefault("FP8", []).append(40.0)
        level = m.best_level("FP8")
        assert level == "FP8"


class TestAdaptiveCompressionHierarchy:
    def test_init(self):
        ach = AdaptiveCompressionHierarchy(initial_level="INT4")
        assert ach.current_level == "INT4"
        assert ach.disabled_levels == set()

    def test_select_level_returns_valid_level(self):
        ach = AdaptiveCompressionHierarchy()
        level = ach.select_level("req-1")
        assert level in [l.name for l in _COMPRESSION_HIERARCHY]

    def test_select_level_tracks_request(self):
        ach = AdaptiveCompressionHierarchy()
        ach.select_level("req-1")
        assert ach.get_level_for_request("req-1") is not None

    def test_report_cleans_up_request(self):
        ach = AdaptiveCompressionHierarchy()
        ach.select_level("req-1")
        ach.report_quality("req-1", "INT4", psnr=45.0, accept_rate=0.9)
        assert ach.get_level_for_request("req-1") is None

    def test_full_regression_and_fallback_cycle(self):
        ach = AdaptiveCompressionHierarchy(initial_level="INT4", regression_limit=3)
        for i in range(3):
            ach.select_level(f"g-{i}")
            ach.report_quality(f"g-{i}", "INT4", psnr=45.0, accept_rate=0.9)
        for i in range(4):
            ach.select_level(f"b-{i}")
            ach.report_quality(f"b-{i}", "INT4", psnr=20.0, accept_rate=0.3)
        assert "INT4" in ach.disabled_levels
        new_level = ach.select_level("recovery")
        assert new_level != "INT4"
