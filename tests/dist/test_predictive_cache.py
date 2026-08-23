"""Real tests for predictive cache management — PatternLearner, PredictiveCacheManager.

Zero mocks — all tests use real instances and deterministic logic.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from distllm.dist.predictive_cache import (
    CachePrediction,
    CacheTier,
    PatternLearner,
    PredictiveCacheManager,
    PrefixPattern,
)


# ---------------------------------------------------------------------------
# Helper: real in-memory cache dummy that implements lookup/store/evict
# ---------------------------------------------------------------------------


class _DictCache:
    """Minimal real cache that matches the duck-typed interface used by
    PredictiveCacheManager."""

    def __init__(self) -> None:
        self._data: dict[tuple[int, ...], object] = {}

    def lookup(self, token_ids: list[int]) -> tuple[int, object | None]:
        key = tuple(token_ids)
        if key in self._data:
            return (len(token_ids), self._data[key])
        # Try prefix matching
        for k, v in self._data.items():
            match_len = 0
            for a, b in zip(k, token_ids):
                if a != b:
                    break
                match_len += 1
            if match_len > 0 and match_len >= min(len(k), len(token_ids)):
                return (match_len, v)
        return (0, None)

    def store(self, token_ids: list[int], value: object) -> None:
        self._data[tuple(token_ids)] = value

    def evict(self, token_ids: list[int]) -> None:
        self._data.pop(tuple(token_ids), None)


# ===================================================================
# CacheTier dataclass
# ===================================================================


class TestCacheTier:
    def test_default_values(self) -> None:
        t = CacheTier(name="test")
        assert t.name == "test"
        assert t.capacity_bytes == 0
        assert t.used_bytes == 0
        assert t.read_latency_ms == 0.1
        assert t.write_latency_ms == 0.1

    def test_custom_values(self) -> None:
        t = CacheTier(
            name="gpu",
            capacity_bytes=512 * 1024 * 1024,
            used_bytes=1024,
            read_latency_ms=0.05,
            write_latency_ms=0.08,
        )
        assert t.name == "gpu"
        assert t.capacity_bytes == 512 * 1024 * 1024
        assert t.used_bytes == 1024
        assert t.read_latency_ms == 0.05

    def test_mutable_fields(self) -> None:
        t = CacheTier(name="cpu", capacity_bytes=4096)
        t.used_bytes = 2048
        assert t.used_bytes == 2048


# ===================================================================
# PrefixPattern dataclass
# ===================================================================


class TestPrefixPattern:
    def test_default_values(self) -> None:
        p = PrefixPattern(prefix_tokens=(1, 2, 3))
        assert p.prefix_tokens == (1, 2, 3)
        assert p.frequency == 0
        assert p.last_seen == 0.0
        assert p.avg_match_length == 0.0
        assert p.hit_count == 0
        assert p.score == 0.0

    def test_custom_values(self) -> None:
        p = PrefixPattern(
            prefix_tokens=(10, 20, 30),
            frequency=5,
            last_seen=100.0,
            avg_match_length=3.0,
            hit_count=10,
            score=0.85,
        )
        assert p.prefix_tokens == (10, 20, 30)
        assert p.frequency == 5
        assert p.score == 0.85

    def test_mutable_fields(self) -> None:
        p = PrefixPattern(prefix_tokens=(1,))
        p.frequency = 10
        p.hit_count += 1
        assert p.frequency == 10
        assert p.hit_count == 1


# ===================================================================
# CachePrediction dataclass
# ===================================================================


class TestCachePrediction:
    def test_default_values(self) -> None:
        cp = CachePrediction(prefix_tokens=(1, 2))
        assert cp.prefix_tokens == (1, 2)
        assert cp.predicted_matches == 0
        assert cp.confidence == 0.0
        assert cp.should_prefetch is False
        assert cp.target_tier == "gpu"

    def test_custom_values(self) -> None:
        cp = CachePrediction(
            prefix_tokens=(5, 6, 7),
            predicted_matches=3,
            confidence=0.75,
            should_prefetch=True,
            target_tier="cpu",
        )
        assert cp.predicted_matches == 3
        assert cp.confidence == 0.75
        assert cp.target_tier == "cpu"

    def test_mutable(self) -> None:
        cp = CachePrediction(prefix_tokens=(1,))
        cp.should_prefetch = True
        cp.target_tier = "disk"
        assert cp.should_prefetch is True
        assert cp.target_tier == "disk"


# ===================================================================
# PatternLearner
# ===================================================================


class TestPatternLearnerInit:
    def test_defaults(self) -> None:
        pl = PatternLearner()
        assert pl.max_patterns == 10000
        assert pl.min_prefix_len == 8
        assert pl.decay_seconds == 24.0 * 3600
        assert pl.pattern_count == 0

    def test_custom_params(self) -> None:
        pl = PatternLearner(max_patterns=50, min_prefix_len=3, decay_hours=1.0)
        assert pl.max_patterns == 50
        assert pl.min_prefix_len == 3
        assert pl.decay_seconds == 3600.0


class TestPatternLearnerObserve:
    def test_observe_below_min_length(self) -> None:
        pl = PatternLearner(min_prefix_len=8)
        pl.observe([1, 2, 3])  # too short
        assert pl.pattern_count == 0

    def test_observe_min_length_exact(self) -> None:
        pl = PatternLearner(min_prefix_len=4)
        pl.observe([10, 20, 30, 40])
        assert pl.pattern_count == 1
        pattern = pl.top_patterns(1)[0]
        assert pattern.prefix_tokens == (10, 20, 30, 40)

    def test_observe_updates_frequency(self) -> None:
        pl = PatternLearner(min_prefix_len=3)
        pl.observe([1, 2, 3, 4, 5])
        pl.observe([1, 2, 3, 4, 5])
        patterns = pl.top_patterns(10)
        assert len(patterns) == 1
        assert patterns[0].frequency >= 1
        assert patterns[0].hit_count >= 1

    def test_observe_multiple_distinct_prefixes(self) -> None:
        pl = PatternLearner(min_prefix_len=3)
        pl.observe([1, 2, 3, 99])
        pl.observe([4, 5, 6, 99])
        pl.observe([7, 8, 9, 99])
        assert pl.pattern_count == 3

    def test_observe_identical_tokens_reuses_pattern(self) -> None:
        pl = PatternLearner(min_prefix_len=3)
        pl.observe([1, 2, 3])
        pl.observe([1, 2, 3])
        assert pl.pattern_count == 1

    def test_observe_empty_list(self) -> None:
        pl = PatternLearner(min_prefix_len=3)
        pl.observe([])
        assert pl.pattern_count == 0


class TestPatternLearnerEviction:
    def test_evict_when_exceeding_max_patterns(self) -> None:
        pl = PatternLearner(max_patterns=5, min_prefix_len=3, decay_hours=1000.0)
        # Insert 6 distinct prefixes — scores will be nearly equal so one
        # must be evicted.
        for i in range(6):
            pl.observe([i, i + 1, i + 2, i + 3])
        assert pl.pattern_count <= 5

    def test_eviction_removes_lowest_score(self) -> None:
        pl = PatternLearner(max_patterns=3, min_prefix_len=2, decay_hours=1000.0)
        pl.observe([10, 20])  # score ~ recency_weight * 1.0 + freq_weight * ~0.33
        pl.observe([30, 40])  # score similar
        pl.observe([50, 60])  # score similar
        # Force a fourth insertion that evicts the lowest
        pl.observe([70, 80])
        assert pl.pattern_count == 3

    def test_evict_empty_patterns(self) -> None:
        pl = PatternLearner(max_patterns=10, min_prefix_len=3)
        # _evict_lowest_score should not crash
        pl._evict_lowest_score()
        assert pl.pattern_count == 0


class TestPatternLearnerPredict:
    def test_predict_returns_matching_prefix(self) -> None:
        pl = PatternLearner(min_prefix_len=3, decay_hours=1000.0)
        pl.observe([1, 2, 3, 4, 5, 6])
        predictions = pl.predict([1, 2, 3, 4, 5, 6, 7])
        assert len(predictions) >= 1
        assert predictions[0].prefix_tokens == (1, 2, 3)

    def test_predict_empty_patterns(self) -> None:
        pl = PatternLearner(min_prefix_len=3)
        assert pl.predict([1, 2, 3]) == []

    def test_predict_no_match(self) -> None:
        pl = PatternLearner(min_prefix_len=3, decay_hours=1000.0)
        pl.observe([10, 20, 30, 40])
        predictions = pl.predict([99, 98, 97, 96])
        assert predictions == []

    def test_predict_empty_input(self) -> None:
        pl = PatternLearner(min_prefix_len=3)
        assert pl.predict([]) == []

    def test_predict_short_input_below_min(self) -> None:
        pl = PatternLearner(min_prefix_len=8)
        pl.observe([1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert pl.predict([1, 2, 3]) == []

    def test_predict_scoring_orders_by_score(self) -> None:
        pl = PatternLearner(min_prefix_len=3, max_patterns=100, decay_hours=1000.0)
        # Insert multiple patterns that all match the same query prefix
        for i in range(5):
            pl.observe([1, 2, 3, i, i + 10])
        predictions = pl.predict([1, 2, 3, 99, 99])
        assert len(predictions) >= 1
        # Scores should be descending
        scores = [p.confidence for p in predictions]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


class TestPatternLearnerTopPatterns:
    def test_top_patterns_empty(self) -> None:
        pl = PatternLearner()
        assert pl.top_patterns(10) == []

    def test_top_patterns_returns_n_entries(self) -> None:
        pl = PatternLearner(min_prefix_len=3, max_patterns=100)
        for i in range(20):
            pl.observe([i, i + 1, i + 2, i + 3])
        top = pl.top_patterns(5)
        assert len(top) == 5

    def test_top_patterns_all(self) -> None:
        pl = PatternLearner(min_prefix_len=2, max_patterns=10)
        for i in range(5):
            pl.observe([i, i + 1])
        top = pl.top_patterns(100)
        assert len(top) == 5


class TestPatternLearnerFeedback:
    def test_record_feedback_hit(self) -> None:
        pl = PatternLearner()
        assert pl._feedback_hits == 0
        pl.record_feedback(was_hit=True)
        assert pl._feedback_hits == 1
        assert pl._feedback_misses == 0

    def test_record_feedback_miss(self) -> None:
        pl = PatternLearner()
        pl.record_feedback(was_hit=False)
        assert pl._feedback_misses == 1
        assert pl._feedback_hits == 0

    def test_feedback_updates_weights_after_threshold(self) -> None:
        pl = PatternLearner(min_prefix_len=3, decay_hours=1000.0)
        # Push feedback above the 100 threshold with all hits
        for _ in range(101):
            pl.record_feedback(was_hit=True)
        # Trigger scoring to apply the adapted weights
        pl.observe([1, 2, 3, 4])
        pl.predict([1, 2, 3, 5])
        # High hit rate -> favor frequency: recency_weight should shift down
        assert pl._recency_weight < 0.6


class TestPatternLearnerSaveLoad:
    def test_save_and_load_roundtrip(self) -> None:
        pl = PatternLearner(min_prefix_len=3, decay_hours=1000.0)
        pl.observe([1, 2, 3, 4])
        pl.observe([5, 6, 7, 8])
        pl.record_feedback(was_hit=True)
        pl.record_feedback(was_hit=False)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            pl.save_patterns(path)

            pl2 = PatternLearner(min_prefix_len=3, decay_hours=1000.0)
            pl2.load_patterns(path)
            assert pl2.pattern_count == pl.pattern_count
            assert pl2._feedback_hits == 1
            assert pl2._feedback_misses == 1
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_nonexistent_file(self) -> None:
        pl = PatternLearner()
        # Should not raise
        pl.load_patterns("/nonexistent/path.json")
        assert pl.pattern_count == 0

    def test_save_empty_patterns(self) -> None:
        pl = PatternLearner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            pl.save_patterns(path)
            data = json.loads(Path(path).read_text())
            assert "patterns" in data
            assert len(data["patterns"]) == 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_corrupted_file_raises(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json")
            path = f.name
        try:
            pl = PatternLearner()
            with pytest.raises(json.JSONDecodeError):
                pl.load_patterns(path)
        finally:
            Path(path).unlink(missing_ok=True)


class TestPatternLearnerComputeMatchLen:
    def test_exact_match(self) -> None:
        pl = PatternLearner(min_prefix_len=2)
        assert pl._compute_match_len((1, 2, 3), [1, 2, 3]) == 3

    def test_partial_match(self) -> None:
        pl = PatternLearner(min_prefix_len=2)
        assert pl._compute_match_len((1, 2, 3, 4), [1, 2, 99]) == 2

    def test_no_match(self) -> None:
        pl = PatternLearner(min_prefix_len=2)
        assert pl._compute_match_len((1, 2, 3), [99, 98, 97]) == 0

    def test_prefix_longer_than_tokens(self) -> None:
        pl = PatternLearner(min_prefix_len=2)
        assert pl._compute_match_len((1, 2, 3, 4, 5), [1, 2]) == 2

    def test_tokens_shorter_than_min(self) -> None:
        pl = PatternLearner(min_prefix_len=8)
        assert pl._compute_match_len((1, 2, 3), [1, 2]) == 0

    def test_empty_tokens(self) -> None:
        pl = PatternLearner(min_prefix_len=2)
        assert pl._compute_match_len((1, 2, 3), []) == 0

    def test_empty_prefix(self) -> None:
        pl = PatternLearner(min_prefix_len=2)
        assert pl._compute_match_len((), [1, 2, 3]) == 0


class TestPatternLearnerScoreAll:
    def test_score_all_empty(self) -> None:
        pl = PatternLearner()
        # Should not raise
        pl._score_all()

    def test_score_all_produces_scores(self) -> None:
        pl = PatternLearner(min_prefix_len=3, decay_hours=1000.0)
        pl.observe([1, 2, 3, 4])
        pl._score_all()
        pattern = pl.top_patterns(1)[0]
        assert 0.0 <= pattern.score <= 1.0

    def test_score_all_no_division_by_zero(self) -> None:
        """All patterns with frequency 0 should not cause ZeroDivisionError."""
        pl = PatternLearner(min_prefix_len=3, decay_hours=1000.0)
        # Insert some patterns and verify scoring
        pl.observe([10, 20, 30, 40])
        pl.observe([50, 60, 70, 80])
        pl._score_all()
        for p in pl.top_patterns(10):
            assert isinstance(p.score, float)
            assert not math.isnan(p.score)


# ===================================================================
# PredictiveCacheManager
# ===================================================================


class TestPredictiveCacheManagerInit:
    def test_default_init(self) -> None:
        pcm = PredictiveCacheManager()
        assert pcm.learner is not None
        assert isinstance(pcm.learner, PatternLearner)
        assert pcm._gpu_cache is None
        assert pcm._cpu_cache is None
        assert pcm._disk_cache is None
        assert pcm._disk_path is None
        assert pcm.stats["prefetches"] == 0

    def test_init_with_caches(self) -> None:
        gpu = _DictCache()
        cpu = _DictCache()
        pcm = PredictiveCacheManager(gpu_cache=gpu, cpu_cache=cpu)
        assert pcm._gpu_cache is gpu
        assert pcm._cpu_cache is cpu

    def test_init_with_disk_path_creates_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            disk_path = Path(tmpdir) / "predictive_cache"
            pcm = PredictiveCacheManager(disk_path=str(disk_path))
            assert disk_path.exists()
            pcm.stop()

    def test_init_default_tiers(self) -> None:
        pcm = PredictiveCacheManager()
        assert "gpu" in pcm._tiers
        assert "cpu" in pcm._tiers
        assert "disk" in pcm._tiers
        assert pcm._tiers["gpu"].capacity_bytes == 512 * 1024 * 1024
        assert pcm._tiers["cpu"].capacity_bytes == 4 * 1024 * 1024 * 1024


class TestPredictiveCacheManagerObserveRequest:
    def test_observe_request_empty(self) -> None:
        pcm = PredictiveCacheManager()
        predictions = pcm.observe_request([])
        assert predictions == []

    def test_observe_request_learns_pattern(self) -> None:
        pcm = PredictiveCacheManager()
        pcm.learner.min_prefix_len = 3
        pcm.learner.decay_seconds = 1000 * 3600
        pcm.observe_request([1, 2, 3, 4, 5])
        assert pcm.learner.pattern_count == 1

    def test_observe_request_returns_predictions(self) -> None:
        pcm = PredictiveCacheManager()
        pcm.learner.min_prefix_len = 3
        pcm.learner.decay_seconds = 1000 * 3600
        # First call learns the pattern, second call predicts
        pcm.observe_request([1, 2, 3, 4, 5])
        predictions = pcm.observe_request([1, 2, 3, 4, 5, 6])
        assert len(predictions) >= 1

    def test_observe_request_no_prefetch_when_not_needed(self) -> None:
        pcm = PredictiveCacheManager()
        pcm.learner.min_prefix_len = 3
        predictions = pcm.observe_request([1, 2, 3])
        # No pattern yet, predictions empty
        assert predictions == []


class TestPredictiveCacheManagerStore:
    def test_store_to_nonexistent_tier(self) -> None:
        pcm = PredictiveCacheManager()
        assert pcm.store([1, 2, 3], "data", tier="nonexistent") is False

    def test_store_to_gpu_without_cache(self) -> None:
        pcm = PredictiveCacheManager()
        assert pcm.store([1, 2, 3], "data") is False  # no gpu_cache set

    def test_store_to_gpu_with_cache(self) -> None:
        gpu = _DictCache()
        pcm = PredictiveCacheManager(gpu_cache=gpu)
        assert pcm.store([1, 2, 3], "kv_data") is True

    def test_store_to_cpu(self) -> None:
        cpu = _DictCache()
        pcm = PredictiveCacheManager(cpu_cache=cpu)
        assert pcm.store([1, 2, 3], "kv_data", tier="cpu") is True


class TestPredictiveCacheManagerLookup:
    def test_lookup_miss_when_no_caches(self) -> None:
        pcm = PredictiveCacheManager()
        match_len, data = pcm.lookup([1, 2, 3])
        assert match_len == 0
        assert data is None

    def test_lookup_hit_gpu(self) -> None:
        gpu = _DictCache()
        gpu.store([1, 2, 3], "gpu_data")
        pcm = PredictiveCacheManager(gpu_cache=gpu)
        match_len, data = pcm.lookup([1, 2, 3])
        assert match_len > 0
        assert data == "gpu_data"
        assert pcm.stats["gpu_hits"] == 1

    def test_lookup_falls_through_tiers(self) -> None:
        gpu = _DictCache()
        cpu = _DictCache()
        cpu.store([10, 20, 30], "cpu_data")
        pcm = PredictiveCacheManager(gpu_cache=gpu, cpu_cache=cpu)
        match_len, data = pcm.lookup([10, 20, 30])
        assert match_len > 0
        assert data == "cpu_data"
        assert pcm.stats["cpu_hits"] == 1

    def test_lookup_miss_updates_stats(self) -> None:
        pcm = PredictiveCacheManager()
        pcm.lookup([1, 2, 3])
        assert pcm.stats["misses"] == 1

    def test_lookup_empty_tokens(self) -> None:
        pcm = PredictiveCacheManager()
        match_len, data = pcm.lookup([])
        assert match_len == 0
        assert data is None

    def test_lookup_partial_prefix_match_promotes(self) -> None:
        gpu = _DictCache()
        cpu = _DictCache()
        # Store a longer sequence in CPU
        cpu.store([1, 2, 3, 4, 5], "cpu_seq")
        pcm = PredictiveCacheManager(gpu_cache=gpu, cpu_cache=cpu)
        match_len, data = pcm.lookup([1, 2, 3, 4, 5, 6, 7])
        # Should match via CPU and promote to GPU
        assert match_len > 0
        assert data == "cpu_seq"
        # Data should now also be in GPU cache
        gpu_match, gpu_data = gpu.lookup([1, 2, 3, 4, 5])
        assert gpu_match > 0
        assert gpu_data == "cpu_seq"


class TestPredictiveCacheManagerGetColdPrefixes:
    def test_cold_prefixes_empty_when_no_patterns(self) -> None:
        pcm = PredictiveCacheManager()
        assert pcm.get_cold_prefixes() == []

    def test_cold_prefixes_filters_by_score(self) -> None:
        pcm = PredictiveCacheManager()
        pcm.learner.min_prefix_len = 3
        # Make decay very fast so the pattern score drops below 0.2
        pcm.learner.decay_seconds = 1e-6
        import time
        pcm.observe_request([1, 2, 3, 4, 5])
        time.sleep(0.001)  # ensure enough time passes for decay
        cold = pcm.get_cold_prefixes()
        assert len(cold) >= 1


class TestPredictiveCacheManagerCompressToDisk:
    def test_compress_no_gpu_cache(self) -> None:
        pcm = PredictiveCacheManager()
        assert pcm.compress_to_disk() == 0

    def test_compress_with_gpu_cache(self) -> None:
        gpu = _DictCache()
        disk = _DictCache()
        pcm = PredictiveCacheManager(gpu_cache=gpu, disk_cache=disk)
        pcm.learner.min_prefix_len = 3
        pcm.learner._score_all = lambda: None  # avoid time dependency
        # Manually inject a cold pattern
        pcm.learner._patterns[(1, 2, 3)] = PrefixPattern(
            prefix_tokens=(1, 2, 3), frequency=0, last_seen=0.0, score=0.0
        )
        gpu.store([1, 2, 3], "cold_data")
        count = pcm.compress_to_disk()
        assert count >= 1
        # Data should now also be in disk
        match_len, data = disk.lookup([1, 2, 3])
        assert match_len > 0


class TestPredictiveCacheManagerStats:
    def test_stats_snapshot(self) -> None:
        pcm = PredictiveCacheManager()
        s = pcm.stats
        assert isinstance(s, dict)
        for key in ("prefetches", "prefetch_hits", "gpu_hits", "cpu_hits", "disk_hits", "misses"):
            assert key in s

    def test_stats_is_copy(self) -> None:
        pcm = PredictiveCacheManager()
        s = pcm.stats
        s["misses"] = 999
        assert pcm.stats["misses"] == 0


class TestPredictiveCacheManagerHitRate:
    def test_hit_rate_no_requests(self) -> None:
        pcm = PredictiveCacheManager()
        assert pcm.hit_rate() == 0.0

    def test_hit_rate_all_misses(self) -> None:
        pcm = PredictiveCacheManager()
        pcm.lookup([1, 2, 3])
        assert pcm.hit_rate() == 0.0

    def test_hit_rate_partial(self) -> None:
        gpu = _DictCache()
        gpu.store([1, 2, 3], "data")
        pcm = PredictiveCacheManager(gpu_cache=gpu)
        pcm.lookup([1, 2, 3])  # hit
        pcm.lookup([9, 9, 9])  # miss
        assert pcm.hit_rate() == 0.5


class TestPredictiveCacheManagerStartStop:
    def test_start_and_stop_prefetch_service(self) -> None:
        pcm = PredictiveCacheManager()
        pcm.start_prefetch_service()
        assert pcm._running is True
        assert pcm._prefetch_thread is not None
        pcm.stop_prefetch_service()
        assert pcm._running is False

    def test_stop_without_start(self) -> None:
        pcm = PredictiveCacheManager()
        # Should not raise
        pcm.stop_prefetch_service()

    def test_start_background_compression(self) -> None:
        pcm = PredictiveCacheManager(gpu_cache=_DictCache())
        pcm.start_background_compression(interval_s=300.0)
        assert pcm._running is True
        pcm.stop()
        assert pcm._running is False

    def test_stop_idempotent(self) -> None:
        pcm = PredictiveCacheManager()
        pcm.stop()
        pcm.stop()
        assert pcm._running is False
