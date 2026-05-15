"""Tests for Feature 24: Memory Profiler CI."""

import json
import tempfile
from pathlib import Path

import pytest

from distllm.profiling.ci_profiler import (
    MemoryProfiler,
    MemorySnapshot,
    MemoryReport,
    LeakDetector,
)


class TestMemorySnapshot:
    def test_snapshot_has_defaults(self):
        snap = MemorySnapshot()
        assert snap.label == ""
        assert snap.cpu_current_mb == 0.0

    def test_snapshot_with_label(self):
        snap = MemorySnapshot(label="test_label", cpu_current_mb=42.0)
        assert snap.label == "test_label"
        assert snap.cpu_current_mb == 42.0


class TestMemoryReport:
    def test_report_to_dict(self):
        report = MemoryReport(
            operation="test_op",
            iterations=5,
            leak_detected=False,
            actual_peak_mb=100.0,
        )
        d = report.to_dict()
        assert d["operation"] == "test_op"
        assert d["iterations"] == 5
        assert d["leak_detected"] is False
        assert d["actual_peak_mb"] == 100.0

    def test_report_to_json(self):
        report = MemoryReport(operation="json_test")
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["operation"] == "json_test"


class TestLeakDetector:
    def test_no_leak_stable_memory(self):
        # Stable memory values — no leak
        values = [100.0, 100.1, 99.9, 100.0, 100.1]
        leak, slope, r_sq = LeakDetector.detect(values)
        assert leak is False

    def test_no_leak_few_values(self):
        # Too few values to detect
        values = [100.0, 101.0]
        leak, slope, r_sq = LeakDetector.detect(values)
        assert leak is False

    def test_leak_detected(self):
        # Clear linear increase
        values = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0]
        leak, slope, r_sq = LeakDetector.detect(values)
        assert leak is True
        assert slope > 1.0  # > 1MB per iteration
        assert r_sq > 0.8

    def test_leak_small_slope_not_flagged(self):
        # Small increase below threshold
        values = [100.0, 100.2, 100.4, 100.6, 100.8]
        leak, slope, r_sq = LeakDetector.detect(values)
        # Slope is 0.2, below 1.0 threshold
        assert leak is False

    def test_leak_noisy_data(self):
        # Noisy data — even with positive slope, R² should be low
        values = [100.0, 150.0, 80.0, 200.0, 90.0, 130.0]
        leak, slope, r_sq = LeakDetector.detect(values)
        # High noise means low R², so no leak flagged
        assert leak is False


class TestMemoryProfiler:
    def test_profile_simple_operation(self):
        profiler = MemoryProfiler(track_gpu=False)

        def simple_op():
            data = [0] * 10000  # Allocate some memory

        report = profiler.profile("simple_alloc", simple_op, iterations=3)

        assert report.operation == "simple_alloc"
        assert report.iterations == 3
        assert len(report.snapshots) == 6  # pre + post per iteration
        assert report.duration_s >= 0

    def test_profile_with_budget(self):
        profiler = MemoryProfiler(track_gpu=False)

        def small_op():
            _ = [0] * 100

        report = profiler.profile("small", small_op, iterations=2, budget_mb=1.0)

        # Should not exceed 1MB budget
        assert report.budget_exceeded is False

    def test_profile_exceeds_budget(self):
        profiler = MemoryProfiler(track_gpu=False)

        def large_op():
            _ = [0] * 10_000_000  # Allocate ~80MB

        report = profiler.profile("large", large_op, iterations=2, budget_mb=0.001)

        assert report.budget_exceeded is True

    def test_profile_batch(self):
        profiler = MemoryProfiler(track_gpu=False)

        ops = {
            "op_a": lambda: [0] * 100,
            "op_b": lambda: [0] * 200,
        }
        reports = profiler.profile_batch(ops)

        assert len(reports) == 2
        assert reports[0].operation == "op_a"
        assert reports[1].operation == "op_b"

    def test_save_reports(self):
        profiler = MemoryProfiler(track_gpu=False)
        report = profiler.profile("save_test", lambda: None, iterations=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            profiler.save_reports([report], str(output_path))

            assert output_path.exists()
            with open(output_path) as f:
                data = json.load(f)

            assert "reports" in data
            assert "summary" in data
            assert data["summary"]["total_operations"] == 1


class TestMemoryProfilerGPU:
    def test_gpu_disabled_when_no_pynvml(self, monkeypatch):
        monkeypatch.setattr("distllm.profiling.ci_profiler.HAS_GPU", False)
        profiler = MemoryProfiler(track_gpu=True)
        assert profiler.track_gpu is False

    def test_snapshot_without_gpu(self):
        profiler = MemoryProfiler(track_gpu=False)
        snap = profiler.snapshot(label="no_gpu")
        assert snap.label == "no_gpu"
        assert snap.gpu_used_mb == 0.0
