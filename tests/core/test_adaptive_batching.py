"""Adaptive batch sizing from latency SLO targets."""

import time

from distllm.core.adaptive_batching import AdaptiveBatchingEngine, SLOConfig


class TestAdaptiveBatchingDefaults:
    """Default configuration and initial state."""

    def test_default_config(self):
        engine = AdaptiveBatchingEngine()
        assert engine._default_config.min_batch_size == 1
        assert engine._default_config.max_batch_size == 64
        assert engine._default_config.adjustment_step == 1
        assert engine._default_config.cooldown_s == 5.0

    def test_get_batch_size_unknown_model_returns_min(self):
        engine = AdaptiveBatchingEngine()
        assert engine.get_batch_size("unknown-model") == 1

    def test_get_batch_size_after_record_returns_at_least_min(self):
        engine = AdaptiveBatchingEngine()
        engine.record_batch("m1", batch_size=4, latencies=[100.0, 200.0])
        assert engine.get_batch_size("m1") >= 1

    def test_get_batch_size_unknown_default_config(self):
        config = SLOConfig(min_batch_size=2, max_batch_size=16)
        engine = AdaptiveBatchingEngine(default_config=config)
        assert engine.get_batch_size("unknown") == 2


class TestAdaptiveBatchingSetSLO:
    """Configuring SLO targets."""

    def test_set_slo_basic(self):
        engine = AdaptiveBatchingEngine()
        engine.set_slo("gpt-4", p50=300, p99=1000, max_batch=32)
        config = engine._configs["gpt-4"]
        assert config.p50_latency_ms == 300
        assert config.p99_latency_ms == 1000
        assert config.max_batch_size == 32

    def test_set_slo_partial_update(self):
        engine = AdaptiveBatchingEngine()
        engine.set_slo("gpt-4", p50=300)
        engine.set_slo("gpt-4", p99=1000)
        config = engine._configs["gpt-4"]
        assert config.p50_latency_ms == 300
        assert config.p99_latency_ms == 1000

    def test_set_slo_initializes_deques(self):
        engine = AdaptiveBatchingEngine()
        engine.set_slo("m1")
        assert "m1" in engine._latencies
        assert "m1" in engine._batch_sizes
        assert "m1" in engine._current_batch

    def test_record_batch_auto_sets_slo(self):
        engine = AdaptiveBatchingEngine()
        engine.record_batch("auto", batch_size=4, latencies=[100.0])
        assert "auto" in engine._configs


class TestAdaptiveBatchingAdjust:
    """Batch size adjusts based on latency vs SLO."""

    def test_low_latency_increases_batch(self):
        engine = AdaptiveBatchingEngine()
        config = SLOConfig(p50_latency_ms=500, p99_latency_ms=2000, adjustment_step=2, cooldown_s=0)
        engine._configs["m1"] = config
        engine.set_slo("m1")
        initial = engine._current_batch["m1"]
        engine.record_batch("m1", batch_size=4, latencies=[10.0] * 50)
        assert engine._current_batch["m1"] > initial

    def test_high_latency_decreases_batch(self):
        engine = AdaptiveBatchingEngine()
        config = SLOConfig(p50_latency_ms=500, p99_latency_ms=1000, adjustment_step=2, cooldown_s=0)
        engine._configs["m1"] = config
        engine.set_slo("m1")
        engine._current_batch["m1"] = 16
        engine.record_batch("m1", batch_size=16, latencies=[5000.0] * 50)
        current = engine._current_batch["m1"]
        assert current < 16

    def test_no_adjustment_within_cooldown(self):
        engine = AdaptiveBatchingEngine()
        config = SLOConfig(p50_latency_ms=500, p99_latency_ms=2000, cooldown_s=60)
        engine._configs["m1"] = config
        engine.set_slo("m1")
        engine._current_batch["m1"] = 8
        engine._last_adjustment["m1"] = time.time()
        engine.record_batch("m1", batch_size=8, latencies=[10.0] * 50)
        assert engine._current_batch["m1"] == 8

    def test_no_adjustment_with_few_samples(self):
        engine = AdaptiveBatchingEngine()
        config = SLOConfig(p50_latency_ms=500, p99_latency_ms=2000, cooldown_s=0)
        engine._configs["m1"] = config
        engine.set_slo("m1")
        engine.record_batch("m1", batch_size=4, latencies=[10.0] * 5)
        assert engine._current_batch["m1"] == 1

    def test_adjustment_does_not_go_below_min(self):
        engine = AdaptiveBatchingEngine()
        config = SLOConfig(p50_latency_ms=1, p99_latency_ms=2, min_batch_size=2, adjustment_step=2, cooldown_s=0)
        engine._configs["m1"] = config
        engine.set_slo("m1")
        engine._current_batch["m1"] = 4
        engine.record_batch("m1", batch_size=4, latencies=[5000.0] * 50)
        assert engine._current_batch["m1"] >= 2

    def test_adjustment_does_not_exceed_max(self):
        engine = AdaptiveBatchingEngine()
        config = SLOConfig(p50_latency_ms=500, max_batch_size=8, adjustment_step=4, cooldown_s=0)
        engine._configs["m1"] = config
        engine.set_slo("m1")
        engine._current_batch["m1"] = 6
        engine.record_batch("m1", batch_size=6, latencies=[1.0] * 50)
        assert engine._current_batch["m1"] <= 8

    def test_get_batch_size_returns_adjusted_value(self):
        engine = AdaptiveBatchingEngine()
        config = SLOConfig(p50_latency_ms=500, p99_latency_ms=2000, cooldown_s=0)
        engine._configs["m1"] = config
        engine.set_slo("m1")
        engine.record_batch("m1", batch_size=2, latencies=[10.0] * 50)
        recommended = engine.get_batch_size("m1")
        assert recommended > 1

    def test_stays_at_current_when_latency_in_band(self):
        engine = AdaptiveBatchingEngine()
        config = SLOConfig(p50_latency_ms=500, p99_latency_ms=2000, adjustment_step=1, cooldown_s=0)
        engine._configs["m1"] = config
        engine.set_slo("m1")
        engine._current_batch["m1"] = 8
        engine.record_batch("m1", batch_size=8, latencies=[400.0] * 50)
        assert engine._current_batch["m1"] == 8


class TestAdaptiveBatchingStats:
    """Statistics and model tracking."""

    def test_get_stats_empty(self):
        engine = AdaptiveBatchingEngine()
        stats = engine.get_stats("nonexistent")
        assert stats.sample_count == 0

    def test_get_stats_after_records(self):
        engine = AdaptiveBatchingEngine()
        engine.record_batch("m1", batch_size=4, latencies=[100.0, 200.0, 150.0])
        stats = engine.get_stats("m1")
        assert stats.sample_count >= 3

    def test_get_stats_p50_p99(self):
        engine = AdaptiveBatchingEngine()
        engine.record_batch("m1", batch_size=4, latencies=[10.0, 20.0, 30.0, 40.0, 50.0] * 20)
        stats = engine.get_stats("m1")
        assert 0 < stats.p50_latency_ms <= stats.p99_latency_ms

    def test_all_stats_returns_all_models(self):
        engine = AdaptiveBatchingEngine()
        engine.record_batch("m1", batch_size=2, latencies=[100.0])
        engine.record_batch("m2", batch_size=4, latencies=[200.0])
        stats = engine.all_stats()
        assert "m1" in stats
        assert "m2" in stats

    def test_batch_sizes_tracked_in_stats(self):
        engine = AdaptiveBatchingEngine()
        engine.record_batch("m1", batch_size=4, latencies=[100.0] * 10)
        stats = engine.get_stats("m1")
        assert 4 in stats.batch_sizes

    def test_concurrent_record_batch(self):
        import threading
        engine = AdaptiveBatchingEngine()
        errors = []

        def record(i):
            try:
                for _ in range(50):
                    engine.record_batch(f"m{i % 3}", batch_size=4, latencies=[100.0])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record, args=(i,), daemon=True) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert not errors
