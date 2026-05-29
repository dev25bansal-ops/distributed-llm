"""Tests for adaptive compression during idle periods."""

import importlib.util
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import pytest


@dataclass
class StubHotSwapMgr:
    loaded: list[dict] | None = None

    def list_loaded_models(self) -> list[dict]:
        if self.loaded is not None:
            return self.loaded
        return [{"name": "test-model", "path": "/models/test", "total_layers": 32}]

    def register_model(self, name: str, path: str, layers: int) -> None:
        pass


def _get_module():
    import sys
    import types

    path = os.path.join("src", "distllm", "core", "adaptive_compression.py")
    spec = importlib.util.spec_from_file_location("adaptive_compression", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["adaptive_compression"] = mod

    mod.logger = types.ModuleType("logger")
    mod.logger.info = lambda *a, **kw: None
    mod.logger.warning = lambda *a, **kw: None
    mod.logger.error = lambda *a, **kw: None
    mod.logger.exception = lambda *a, **kw: None

    spec.loader.exec_module(mod)
    return mod


class TestIdleDetector:
    @classmethod
    def setup_class(cls):
        cls.mod = _get_module()
        cls.IdleDetector = cls.mod.IdleDetector
        cls.IdleDetectorConfig = cls.mod.IdleDetectorConfig

    def test_not_idle_when_util_high(self):
        detector = self.IdleDetector(
            utilization_fn=lambda: 0.8,
            config=self.IdleDetectorConfig(idle_duration_s=0),
        )
        assert detector.is_idle is False

    def test_idle_when_util_below_threshold_long_enough(self):
        detector = self.IdleDetector(
            utilization_fn=lambda: 0.2,
            config=self.IdleDetectorConfig(utilization_threshold_pct=30, idle_duration_s=0),
        )
        assert detector.is_idle is True

    def test_not_idle_immediately_when_util_drops(self):
        detector = self.IdleDetector(
            utilization_fn=lambda: 0.2,
            config=self.IdleDetectorConfig(utilization_threshold_pct=30, idle_duration_s=60),
        )
        assert detector.is_idle is False

    def test_idle_duration_zero_when_busy(self):
        detector = self.IdleDetector(
            utilization_fn=lambda: 0.8,
        )
        assert detector.idle_duration == 0.0

    def test_idle_duration_increases_over_time(self):
        detector = self.IdleDetector(
            utilization_fn=lambda: 0.1,
            config=self.IdleDetectorConfig(utilization_threshold_pct=30, idle_duration_s=0),
        )
        assert detector.is_idle is True
        assert detector.idle_duration > 0.0

    def test_reset_clears_idle(self):
        detector = self.IdleDetector(
            utilization_fn=lambda: 0.1,
            config=self.IdleDetectorConfig(utilization_threshold_pct=30, idle_duration_s=0),
        )
        assert detector.is_idle is True
        detector.reset()
        assert detector.idle_duration == 0.0

    def test_edge_case_exact_threshold(self):
        detector = self.IdleDetector(
            utilization_fn=lambda: 0.3,
            config=self.IdleDetectorConfig(utilization_threshold_pct=30, idle_duration_s=0),
        )
        assert detector.is_idle is False

    def test_edge_case_barely_below_threshold(self):
        detector = self.IdleDetector(
            utilization_fn=lambda: 0.299,
            config=self.IdleDetectorConfig(utilization_threshold_pct=30, idle_duration_s=0),
        )
        assert detector.is_idle is True

    def test_recovers_from_idle_when_load_returns(self):
        calls = [0.2, 0.2, 0.8, 0.8]
        idx = [0]

        def util():
            v = calls[idx[0] % len(calls)]
            idx[0] += 1
            return v

        detector = self.IdleDetector(
            utilization_fn=util,
            config=self.IdleDetectorConfig(utilization_threshold_pct=30, idle_duration_s=0),
        )
        assert detector.is_idle is True
        assert detector.is_idle is True
        assert detector.is_idle is False
        assert detector.is_idle is False


class TestCompressionJob:
    @classmethod
    def setup_class(cls):
        cls.mod = _get_module()
        cls.CompressionJob = cls.mod.CompressionJob

    def test_default_state(self):
        job = self.CompressionJob(
            model_name="m1", model_path="/p1", compressed_path="",
            method="int4", started_at=time.time(),
        )
        assert job.success is False
        assert job.finished_at is None
        assert job.error is None

    def test_completed_state(self):
        job = self.CompressionJob(
            model_name="m1", model_path="/p1", compressed_path="/p1-compressed",
            method="int4", started_at=100.0, finished_at=200.0, success=True,
        )
        assert job.success is True
        assert job.finished_at == 200.0

    def test_failed_state(self):
        job = self.CompressionJob(
            model_name="m1", model_path="/p1", compressed_path="",
            method="int4", started_at=100.0, finished_at=150.0,
            success=False, error="OOM",
        )
        assert job.success is False
        assert job.error == "OOM"


class TestAdaptiveCompressionManager:
    @classmethod
    def setup_class(cls):
        cls.mod = _get_module()
        cls.AdaptiveCompressionManager = cls.mod.AdaptiveCompressionManager
        cls.AdaptiveCompressionConfig = cls.mod.AdaptiveCompressionConfig
        cls.CompressionJob = cls.mod.CompressionJob

    def test_not_compressing_after_init(self):
        mgr = self.AdaptiveCompressionManager(
            config=self.AdaptiveCompressionConfig(enabled=True),
            utilization_fn=lambda: 0.0,
        )
        assert mgr.is_compressing is False
        assert mgr.jobs == []

    def test_jobs_empty_initially(self):
        mgr = self.AdaptiveCompressionManager()
        assert mgr.jobs == []

    def test_compressed_variants_empty_initially(self):
        mgr = self.AdaptiveCompressionManager()
        assert mgr.compressed_variants == {}

    def test_start_stop_no_error(self):
        mgr = self.AdaptiveCompressionManager(
            config=self.AdaptiveCompressionConfig(enabled=True),
        )
        mgr.start()
        mgr.stop()

    def test_double_start_is_safe(self):
        mgr = self.AdaptiveCompressionManager(
            config=self.AdaptiveCompressionConfig(enabled=True),
        )
        mgr.start()
        mgr.start()
        mgr.stop()

    def test_disabled_does_not_start_thread(self):
        mgr = self.AdaptiveCompressionManager(
            config=self.AdaptiveCompressionConfig(enabled=False),
        )
        mgr.start()
        assert mgr._thread is None
        mgr.stop()

    def test_pick_candidate_returns_none_when_no_hot_swap(self):
        mgr = self.AdaptiveCompressionManager()
        assert mgr._pick_candidate() is None

    def test_pick_candidate_returns_none_when_all_compressed(self):
        mgr = self.AdaptiveCompressionManager(
            hot_swap_mgr=StubHotSwapMgr(),
        )
        with mgr._lock:
            mgr._compressed_model_variants["test-model"] = "/compressed"
        assert mgr._pick_candidate() is None

    def test_pick_candidate_returns_first_uncompressed(self):
        mgr = self.AdaptiveCompressionManager(
            hot_swap_mgr=StubHotSwapMgr(),
        )
        result = mgr._pick_candidate()
        assert result is not None
        name, path = result
        assert name == "test-model"
        assert path == "/models/test"

    def test_get_compressed_path_returns_none_when_not_compressed(self):
        mgr = self.AdaptiveCompressionManager()
        assert mgr.get_compressed_path("unknown") is None

    def test_get_compressed_path_returns_path_when_compressed(self):
        mgr = self.AdaptiveCompressionManager()
        with mgr._lock:
            mgr._compressed_model_variants["test-model"] = "/compressed/test"
        assert mgr.get_compressed_path("test-model") == "/compressed/test"

    def test_tick_skips_when_already_compressing(self):
        mgr = self.AdaptiveCompressionManager(
            config=self.AdaptiveCompressionConfig(
                enabled=True, idle_threshold_pct=100, idle_duration_s=0,
            ),
            utilization_fn=lambda: 0.0,
            hot_swap_mgr=StubHotSwapMgr(),
        )
        mgr._compressing_now = True
        mgr._tick()
        assert mgr.jobs == []

    def test_tick_skips_when_not_idle(self):
        mgr = self.AdaptiveCompressionManager(
            config=self.AdaptiveCompressionConfig(
                enabled=True, idle_threshold_pct=30, idle_duration_s=60,
            ),
            utilization_fn=lambda: 0.8,
            hot_swap_mgr=StubHotSwapMgr(),
        )
        mgr._tick()
        assert mgr.jobs == []

    def test_tick_starts_compression_when_idle(self):
        mgr = self.AdaptiveCompressionManager(
            config=self.AdaptiveCompressionConfig(
                enabled=True, idle_threshold_pct=100, idle_duration_s=0,
            ),
            utilization_fn=lambda: 0.0,
            hot_swap_mgr=StubHotSwapMgr(),
        )
        mgr._tick()
        assert len(mgr.jobs) == 1
        assert mgr.jobs[0].model_name == "test-model"

    def test_tick_honors_already_compressed_variants(self):
        mgr = self.AdaptiveCompressionManager(
            config=self.AdaptiveCompressionConfig(
                enabled=True, idle_threshold_pct=100, idle_duration_s=0,
            ),
            utilization_fn=lambda: 0.0,
            hot_swap_mgr=StubHotSwapMgr(),
        )
        with mgr._lock:
            mgr._compressed_model_variants["test-model"] = "/compressed"
        mgr._tick()
        assert mgr.jobs == []

    def test_tick_no_candidate_when_hot_swap_empty(self):
        mgr = self.AdaptiveCompressionManager(
            config=self.AdaptiveCompressionConfig(
                enabled=True, idle_threshold_pct=100, idle_duration_s=0,
            ),
            utilization_fn=lambda: 0.0,
            hot_swap_mgr=StubHotSwapMgr(loaded=[]),
        )
        mgr._tick()
        assert mgr.jobs == []

    def test_compression_completion_callback_is_called(self):
        results = []

        class FakeCompressor:
            def compress(self, name, path, tag="compressed"):
                return "/tmp/compressed"

        mgr = self.AdaptiveCompressionManager(
            config=self.AdaptiveCompressionConfig(
                enabled=True, idle_threshold_pct=100, idle_duration_s=0,
            ),
            utilization_fn=lambda: 0.0,
            hot_swap_mgr=StubHotSwapMgr(),
            compressor=FakeCompressor(),
            on_compression_complete=lambda job: results.append(job),
        )
        mgr._tick()
        time.sleep(0.3)
        assert len(results) >= 1
        assert results[0].success is True
        assert results[0].model_name == "test-model"

    def test_concurrent_compression_jobs_unique(self):
        mgr = self.AdaptiveCompressionManager(
            config=self.AdaptiveCompressionConfig(
                enabled=True, idle_threshold_pct=100, idle_duration_s=0,
            ),
            utilization_fn=lambda: 0.0,
            hot_swap_mgr=StubHotSwapMgr(),
        )
        mgr._tick()
        mgr._tick()
        assert len(mgr.jobs) == 1

    def test_job_recorded_on_compression_failure(self):
        call_count = [0]

        class FailingCompressor:
            def compress(self, name, path, tag="compressed"):
                call_count[0] += 1
                raise RuntimeError("Compression failed")

        mgr = self.AdaptiveCompressionManager(
            config=self.AdaptiveCompressionConfig(
                enabled=True, idle_threshold_pct=100, idle_duration_s=0,
            ),
            utilization_fn=lambda: 0.0,
            hot_swap_mgr=StubHotSwapMgr(),
            compressor=FailingCompressor(),
        )
        mgr._tick()
        time.sleep(0.3)
        assert len(mgr.jobs) == 1
        assert mgr.jobs[0].success is False
        assert mgr.jobs[0].error == "Compression failed"
        assert mgr.jobs[0].finished_at is not None
        assert mgr.is_compressing is False


class TestUtilizationFn:
    @classmethod
    def setup_class(cls):
        cls.mod = _get_module()
        cls.IdleDetector = cls.mod.IdleDetector

    def test_util_fn_handles_exception(self):
        def fn():
            raise RuntimeError

        detector = self.IdleDetector(utilization_fn=fn)
        assert detector.is_idle is False


def test_module_exports():
    mod = _get_module()
    assert hasattr(mod, "IdleDetector")
    assert hasattr(mod, "IdleDetectorConfig")
    assert hasattr(mod, "CompressionJob")
    assert hasattr(mod, "SimpleCompressor")
    assert hasattr(mod, "AdaptiveCompressionConfig")
    assert hasattr(mod, "AdaptiveCompressionManager")
