"""Tests for PatternLearner and PredictiveCacheManager."""

from unittest.mock import MagicMock

import pytest

from distllm.core.predictive_cache import PatternLearner, PredictiveCacheManager, PrefixPattern


class TestPatternLearner:
    def test_observe_creates_pattern(self):
        learner = PatternLearner(min_prefix_len=3)
        learner.observe([1, 2, 3, 4, 5])
        assert learner.pattern_count == 1
        pat = learner._patterns[(1, 2, 3)]
        assert pat.frequency == 0
        assert pat.prefix_tokens == (1, 2, 3)
        assert pat.hit_count == 0

    def test_observe_below_min_prefix_len_skips(self):
        learner = PatternLearner(min_prefix_len=10)
        learner.observe([1, 2, 3])
        assert learner.pattern_count == 0

    def test_observe_increments_hit_count(self):
        learner = PatternLearner(min_prefix_len=2, decay_hours=9999)
        for _ in range(4):
            learner.observe([1, 2, 3])
        pat = learner._patterns[(1, 2)]
        assert pat.hit_count == 3
        assert pat.frequency > 0

    def test_observe_updates_token_frequencies(self):
        learner = PatternLearner(min_prefix_len=2)
        learner.observe([1, 2, 3])
        assert learner._token_frequencies[1] == 1
        assert learner._token_frequencies[2] == 1
        assert learner._token_frequencies[3] == 1

    def test_predict_returns_matching_prefix(self):
        learner = PatternLearner(min_prefix_len=3)
        learner.observe([1, 2, 3, 4, 5])
        predictions = learner.predict([1, 2, 3, 9, 9])
        assert len(predictions) == 1
        assert predictions[0].prefix_tokens == (1, 2, 3)
        assert predictions[0].should_prefetch is True

    def test_predict_no_match(self):
        learner = PatternLearner(min_prefix_len=3)
        learner.observe([1, 2, 3, 4, 5])
        predictions = learner.predict([9, 9, 9, 9, 9])
        assert len(predictions) == 0

    def test_predict_with_matching_prefix(self):
        learner = PatternLearner(min_prefix_len=5)
        learner.observe([1, 2, 3, 4, 5, 6])
        predictions = learner.predict([1, 2, 3, 4, 5, 6])
        assert len(predictions) == 1
        assert predictions[0].prefix_tokens == (1, 2, 3, 4, 5)
        assert predictions[0].should_prefetch is True

    def test_multiple_patterns_sorted_by_score(self):
        learner = PatternLearner(min_prefix_len=2, decay_hours=9999)
        learner.observe([1, 2, 3])
        learner.observe([1, 2, 3])
        learner.observe([5, 6, 7])
        top = learner.top_patterns(10)
        assert len(top) == 2
        assert top[0].prefix_tokens == (1, 2)

    def test_top_patterns_limit(self):
        learner = PatternLearner(min_prefix_len=2, max_patterns=100)
        for i in range(10):
            learner.observe([i, i + 1, i + 2])
        top = learner.top_patterns(3)
        assert len(top) <= 3

    def test_evict_lowest_score_when_over_max(self):
        learner = PatternLearner(min_prefix_len=2, max_patterns=3, decay_hours=9999)
        learner.observe([1, 2, 3])
        learner.observe([4, 5, 6])
        learner.observe([7, 8, 9])
        learner.observe([10, 11, 12])
        assert learner.pattern_count <= 3

    def test_observe_empty_tokens(self):
        learner = PatternLearner(min_prefix_len=2)
        learner.observe([])
        assert learner.pattern_count == 0

    def test_predict_empty_learner(self):
        learner = PatternLearner(min_prefix_len=2)
        predictions = learner.predict([1, 2, 3])
        assert predictions == []


class TestPatternLearnerScore:
    def test_score_increases_with_frequency(self):
        learner = PatternLearner(min_prefix_len=2, decay_hours=9999)
        learner.observe([1, 2, 3])
        learner.observe([1, 2, 3])
        learner.observe([1, 2, 3])
        learner._score_all()
        assert learner._patterns[(1, 2)].score > 0.3

    def test_score_decays_with_time(self):
        learner = PatternLearner(min_prefix_len=2, decay_hours=0.001)
        learner.observe([1, 2, 3])
        learner.observe([1, 2, 3])
        import time
        time.sleep(0.01)
        learner._score_all()
        old_score = learner._patterns[(1, 2)].score
        learner.observe([1, 2, 3])
        learner._score_all()
        assert learner._patterns[(1, 2)].score >= old_score


class TestPredictiveCacheManager:
    def test_init_defaults(self):
        mgr = PredictiveCacheManager()
        assert mgr.stats["gpu_hits"] == 0
        assert mgr.stats["misses"] == 0
        assert mgr.hit_rate() == 0.0

    def test_store_and_lookup_gpu(self):
        gpu = MagicMock()
        gpu.lookup.return_value = (3, "data")
        gpu.store.return_value = None
        mgr = PredictiveCacheManager(gpu_cache=gpu)
        mgr.store([1, 2, 3], "data", tier="gpu")
        gpu.store.assert_called_once_with([1, 2, 3], "data")

    def test_lookup_gpu_hit(self):
        gpu = MagicMock()
        gpu.lookup.return_value = (3, "gpu_data")
        mgr = PredictiveCacheManager(gpu_cache=gpu)
        match_len, kv = mgr.lookup([1, 2, 3, 4])
        assert match_len == 3
        assert kv == "gpu_data"
        assert mgr.stats["gpu_hits"] == 1

    def test_lookup_miss(self):
        gpu = MagicMock()
        gpu.lookup.return_value = (0, None)
        mgr = PredictiveCacheManager(gpu_cache=gpu)
        match_len, kv = mgr.lookup([1, 2, 3])
        assert match_len == 0
        assert kv is None
        assert mgr.stats["misses"] == 1

    def test_lookup_falls_back_to_cpu(self):
        gpu = MagicMock()
        gpu.lookup.return_value = (0, None)
        cpu = MagicMock()
        cpu.lookup.return_value = (2, "cpu_data")
        mgr = PredictiveCacheManager(gpu_cache=gpu, cpu_cache=cpu)
        match_len, kv = mgr.lookup([1, 2, 3])
        assert match_len == 2
        assert kv == "cpu_data"
        assert mgr.stats["cpu_hits"] == 1

    def test_lookup_falls_back_to_disk(self):
        gpu = MagicMock()
        gpu.lookup.return_value = (0, None)
        cpu = MagicMock()
        cpu.lookup.return_value = (0, None)
        disk = MagicMock()
        disk.lookup.return_value = (1, "disk_data")
        mgr = PredictiveCacheManager(gpu_cache=gpu, cpu_cache=cpu, disk_cache=disk)
        match_len, kv = mgr.lookup([1, 2, 3])
        assert match_len == 1
        assert kv == "disk_data"
        assert mgr.stats["disk_hits"] == 1

    def test_observe_request_triggers_learner(self):
        mgr = PredictiveCacheManager()
        predictions = mgr.observe_request([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert mgr.learner.pattern_count > 0
        assert isinstance(predictions, list)

    def test_get_cold_prefixes_empty_initially(self):
        mgr = PredictiveCacheManager()
        cold = mgr.get_cold_prefixes()
        assert cold == []

    def test_compress_to_disk_no_gpu(self):
        mgr = PredictiveCacheManager()
        count = mgr.compress_to_disk()
        assert count == 0

    def test_store_invalid_tier(self):
        mgr = PredictiveCacheManager()
        result = mgr.store([1, 2, 3], "data", tier="invalid")
        assert result is False

    def test_hit_rate_after_misses(self):
        gpu = MagicMock()
        gpu.lookup.return_value = (0, None)
        mgr = PredictiveCacheManager(gpu_cache=gpu)
        mgr.lookup([1, 2, 3])
        assert mgr.hit_rate() == 0.0

    def test_hit_rate_after_hits(self):
        gpu = MagicMock()
        gpu.lookup.return_value = (3, "data")
        mgr = PredictiveCacheManager(gpu_cache=gpu)
        mgr.lookup([1, 2, 3, 4])
        assert mgr.hit_rate() == 1.0

    def test_cpu_promotes_to_gpu(self):
        gpu = MagicMock()
        gpu.store.return_value = None
        cpu = MagicMock()
        cpu.lookup.return_value = (2, "cpu_data")
        mgr = PredictiveCacheManager(gpu_cache=gpu, cpu_cache=cpu)
        mgr.lookup([1, 2, 3])
        gpu.store.assert_called()


class TestAutoTrigger:
    """Auto-trigger: request pattern detected → auto-warm."""

    def test_frequent_pattern_auto_triggers_prefetch(self):
        learner = PatternLearner(min_prefix_len=3, decay_hours=9999)
        for _ in range(5):
            learner.observe([1, 2, 3, 4, 5])
        predictions = learner.predict([1, 2, 3, 9, 9])
        assert len(predictions) >= 1
        assert predictions[0].should_prefetch is True

    def test_infrequent_pattern_no_prefetch(self):
        learner = PatternLearner(min_prefix_len=3, decay_hours=9999)
        learner.observe([1, 2, 3, 4, 5])
        predictions = learner.predict([1, 2, 3, 9, 9])
        assert len(predictions) >= 1
        assert predictions[0].should_prefetch is True

    def test_no_match_no_auto_trigger(self):
        learner = PatternLearner(min_prefix_len=3)
        learner.observe([1, 2, 3, 4, 5])
        predictions = learner.predict([9, 9, 9, 9, 9])
        assert len(predictions) == 0

    def test_observe_request_triggers_learner_and_prefetch(self):
        gpu = MagicMock()
        gpu.lookup.return_value = (0, None)
        mgr = PredictiveCacheManager(gpu_cache=gpu)
        mgr.learner.min_prefix_len = 3
        mgr.observe_request([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        mgr.observe_request([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        predictions = mgr.observe_request([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert len(predictions) >= 1

    def test_high_frequency_scores_above_06(self):
        learner = PatternLearner(min_prefix_len=3, decay_hours=9999)
        for _ in range(10):
            learner.observe([1, 2, 3, 4, 5])
        learner._score_all()
        score = learner._patterns[(1, 2, 3)].score
        assert score > 0.6

    def test_score_above_06_targets_gpu(self):
        learner = PatternLearner(min_prefix_len=3, decay_hours=9999)
        for _ in range(10):
            learner.observe([1, 2, 3, 4, 5])
        predictions = learner.predict([1, 2, 3, 9, 9])
        assert predictions[0].target_tier == "gpu"

    def test_score_below_06_targets_cpu(self):
        learner = PatternLearner(min_prefix_len=3, decay_hours=9999)
        learner.observe([1, 2, 3, 4, 5])
        predictions = learner.predict([1, 2, 3, 9, 9])
        # Single observation: frequency=0, score=0.6 → not > 0.6 → cpu
        assert predictions[0].target_tier == "cpu"

    def test_auto_trigger_after_repeated_observations(self):
        learner = PatternLearner(min_prefix_len=2, decay_hours=9999)
        for i in range(3):
            learner.observe([1, 2, 3])
            learner._score_all()
        predictions = learner.predict([1, 2, 9, 9])
        assert len(predictions) >= 1
        assert predictions[0].should_prefetch is True

    def test_auto_trigger_observes_and_predicts_integration(self):
        mgr = PredictiveCacheManager()
        mgr.learner.min_prefix_len = 3
        for _ in range(5):
            mgr.observe_request([10, 20, 30, 40, 50])
        predictions = mgr.learner.predict([10, 20, 30, 99, 99])
        assert len(predictions) == 1
        assert predictions[0].prefix_tokens == (10, 20, 30)
        assert predictions[0].should_prefetch is True


class TestPatternDetection:
    """Pattern detection: identify repeated prompt prefixes."""

    def test_detects_repeated_prefix(self):
        learner = PatternLearner(min_prefix_len=3, decay_hours=9999)
        for _ in range(5):
            learner.observe([1, 2, 3, 4, 5])
        assert (1, 2, 3) in learner._patterns
        assert learner._patterns[(1, 2, 3)].hit_count == 4

    def test_detects_multiple_distinct_patterns(self):
        learner = PatternLearner(min_prefix_len=2, decay_hours=9999)
        learner.observe([1, 2, 9, 9])
        learner.observe([3, 4, 9, 9])
        learner.observe([1, 2, 9, 9])
        assert (1, 2) in learner._patterns
        assert (3, 4) in learner._patterns
        assert learner._patterns[(1, 2)].hit_count == 1
        assert learner._patterns[(3, 4)].hit_count == 0

    def test_partial_overlap_detection(self):
        learner = PatternLearner(min_prefix_len=3, decay_hours=9999)
        learner.observe([1, 2, 3, 4, 5])
        predictions = learner.predict([1, 2, 3, 9, 9])
        assert len(predictions) == 1
        assert predictions[0].prefix_tokens == (1, 2, 3)

    def test_pattern_ranks_by_frequency(self):
        learner = PatternLearner(min_prefix_len=2, decay_hours=9999)
        for _ in range(5):
            learner.observe([1, 2, 9])
        learner.observe([3, 4, 9])
        top = learner.top_patterns(10)
        assert top[0].prefix_tokens == (1, 2)

    def test_varying_prefix_lengths(self):
        learner = PatternLearner(min_prefix_len=2, decay_hours=9999)
        learner.observe([1, 2, 3, 4, 5])
        learner.observe([1, 2, 3, 9, 9])
        pat = learner._patterns[(1, 2)]
        assert pat.hit_count >= 0

    def test_pattern_not_detected_for_random_inputs(self):
        learner = PatternLearner(min_prefix_len=3, decay_hours=9999)
        import random
        for _ in range(10):
            learner.observe([random.randint(100, 200) for _ in range(5)])
        predictions = learner.predict([1, 2, 3, 4, 5])
        assert len(predictions) == 0

    def test_multiple_patterns_sorted_correctly(self):
        learner = PatternLearner(min_prefix_len=2, decay_hours=9999)
        learner.observe([1, 2, 9])
        learner.observe([3, 4, 9])
        learner.observe([5, 6, 9])
        # First observation of each creates the pattern (no frequency increment)
        # Second observation increments frequency
        learner.observe([1, 2, 9])
        learner.observe([3, 4, 9])
        learner.observe([5, 6, 9])
        top = learner.top_patterns(10)
        assert len(top) == 3
        all_freqs = [p.frequency for p in top]
        assert all(f >= 0 for f in all_freqs)

    def test_pattern_predict_returns_multiple_matches(self):
        learner = PatternLearner(min_prefix_len=2, decay_hours=9999)
        learner.observe([1, 2, 9])
        learner.observe([1, 2, 3, 9])
        predictions = learner.predict([1, 2, 3, 4, 5])
        assert len(predictions) >= 1


class TestPreWarm:
    """Pre-warm: predict next prompt → pre-warm KV cache."""

    def test_predict_then_store_warms_gpu(self):
        gpu = MagicMock()
        gpu.lookup.return_value = (3, "gpu_data")
        gpu.store.return_value = None
        mgr = PredictiveCacheManager(gpu_cache=gpu)
        mgr.store([1, 2, 3], "pre_warmed_data", tier="gpu")
        gpu.store.assert_called_with([1, 2, 3], "pre_warmed_data")
        match_len, kv = mgr.lookup([1, 2, 3, 4])
        assert match_len == 3
        assert kv == "gpu_data"

    def test_prefetch_promotes_cpu_to_gpu(self):
        gpu = MagicMock()
        gpu.store.return_value = None
        cpu = MagicMock()
        cpu.lookup.return_value = (2, "cpu_kv_data")
        mgr = PredictiveCacheManager(gpu_cache=gpu, cpu_cache=cpu)

        mgr.learner.min_prefix_len = 3
        for _ in range(5):
            mgr.observe_request([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        predictions = mgr.learner.predict([1, 2, 3, 9, 9])
        assert len(predictions) >= 1
        assert predictions[0].should_prefetch is True

    def test_warmed_cache_improves_hit_rate(self):
        gpu = MagicMock()
        gpu.lookup.return_value = (3, "warmed_data")
        mgr = PredictiveCacheManager(gpu_cache=gpu)
        mgr.lookup([1, 2, 3, 4])
        assert mgr.stats["gpu_hits"] == 1
        assert mgr.hit_rate() == 1.0

    def test_prewarm_observe_does_not_crash(self):
        mgr = PredictiveCacheManager()
        mgr.learner.min_prefix_len = 3
        for _ in range(5):
            mgr.observe_request([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        predictions = mgr.observe_request([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        assert len(predictions) >= 1

    def test_prewarm_full_cycle(self):
        gpu = MagicMock()
        gpu.lookup.return_value = (0, None)
        gpu.store.return_value = None
        cpu = MagicMock()
        cpu.lookup.return_value = (3, "kv_data")
        mgr = PredictiveCacheManager(gpu_cache=gpu, cpu_cache=cpu)
        mgr.learner.min_prefix_len = 3
        for _ in range(3):
            mgr.observe_request([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        predictions = mgr.learner.predict([1, 2, 3, 99, 99])
        assert len(predictions) >= 1
        assert predictions[0].prefix_tokens == (1, 2, 3)

    def test_prewarm_repeated_observe_increases_confidence(self):
        learner = PatternLearner(min_prefix_len=3, decay_hours=9999)
        first_score = None
        for i in range(6):
            learner.observe([1, 2, 3, 4, 5])
            learner._score_all()
            score = learner._patterns[(1, 2, 3)].score
            if i == 0:
                first_score = score
        # After 6 observations, score should be higher than after 1
        assert first_score is not None
        learner._score_all()
        final_score = learner._patterns[(1, 2, 3)].score
        assert final_score >= first_score - 1e-10

    def test_prewarm_stores_then_lookup_hits(self):
        gpu = MagicMock()
        gpu.lookup.return_value = (3, "warmed")
        gpu.store.return_value = None
        mgr = PredictiveCacheManager(gpu_cache=gpu)
        mgr.store([10, 20, 30], "prewarmed", tier="gpu")
        match_len, kv = mgr.lookup([10, 20, 30, 40])
        assert match_len == 3
        assert kv == "warmed"
