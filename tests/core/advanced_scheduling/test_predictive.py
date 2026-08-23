"""Tests for PredictiveBatchScheduler."""

from __future__ import annotations

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_predictive = load_module("distllm/core/advanced_scheduling/predictive.py")
PredictiveBatchScheduler = _predictive.PredictiveBatchScheduler


class TestPredictiveBatchScheduler:
    """Test suite for PredictiveBatchScheduler."""

    def test_default_construction(self) -> None:
        scheduler = PredictiveBatchScheduler()
        assert scheduler._latency_history.maxlen == 100
        assert scheduler._batch_history.maxlen == 100
        assert len(scheduler._latency_history) == 0

    def test_custom_history_size(self) -> None:
        scheduler = PredictiveBatchScheduler(history_size=50)
        assert scheduler._latency_history.maxlen == 50

    def test_record(self) -> None:
        scheduler = PredictiveBatchScheduler()
        scheduler.record(batch_size=8, latency_ms=42.0)
        assert len(scheduler._latency_history) == 1
        assert scheduler._latency_history[0] == 42.0
        assert scheduler._batch_history[0] == 8

    def test_record_multiple_entries(self) -> None:
        scheduler = PredictiveBatchScheduler()
        for i in range(5):
            scheduler.record(batch_size=(i + 1) * 8, latency_ms=(i + 1) * 20.0)
        assert len(scheduler._latency_history) == 5

    def test_predict_optimal_batch_size_insufficient_data(self) -> None:
        """With fewer than 10 data points, returns default of 8."""
        scheduler = PredictiveBatchScheduler()
        for i in range(9):
            scheduler.record(batch_size=8, latency_ms=50.0)

        result = scheduler.predict_optimal_batch_size(target_latency_ms=100.0)
        assert result == 8

    def test_predict_optimal_batch_size_sufficient_data(self) -> None:
        """With sufficient data, performs linear regression."""
        scheduler = PredictiveBatchScheduler()
        # batch_size -> latency: roughly latency = batch_size * 5 + 10
        for bs in range(1, 15):
            scheduler.record(batch_size=bs, latency_ms=bs * 5 + 10)

        result = scheduler.predict_optimal_batch_size(target_latency_ms=100.0)
        # 100 = 5 * bs + 10 => bs = 18, clamped to max(1, min(18, 64)) => 18
        assert result == 18

    def test_predict_optimal_batch_size_clamps_to_max_64(self) -> None:
        scheduler = PredictiveBatchScheduler()
        # Latency barely increases with batch size: latency = bs * 0.5 + 5
        for bs in range(1, 15):
            scheduler.record(batch_size=bs, latency_ms=bs * 0.5 + 5)

        result = scheduler.predict_optimal_batch_size(target_latency_ms=1000.0)
        # 1000 = 0.5 * bs + 5 => bs = 1990, clamped to 64
        assert result == 64

    def test_predict_optimal_batch_size_clamps_to_min_1(self) -> None:
        scheduler = PredictiveBatchScheduler()
        # Fast latency growth: latency = bs * 200 + 100
        for bs in range(1, 15):
            scheduler.record(batch_size=bs, latency_ms=bs * 200 + 100)

        result = scheduler.predict_optimal_batch_size(target_latency_ms=50.0)
        # 50 = 200*bs + 100 => bs = -0.25, clamped to 1
        assert result == 1

    def test_predict_optimal_batch_size_zero_denominator(self) -> None:
        """When all batch sizes are identical, denominator is 0 -> default."""
        scheduler = PredictiveBatchScheduler()
        for _ in range(15):
            scheduler.record(batch_size=8, latency_ms=50.0)

        result = scheduler.predict_optimal_batch_size(target_latency_ms=100.0)
        assert result == 8

    def test_predict_optimal_batch_size_flat_line(self) -> None:
        """When a <= 0 (latency doesn't increase with batch), returns 32."""
        scheduler = PredictiveBatchScheduler()
        # Latency decreases with batch size (negative slope)
        for bs in range(1, 15):
            scheduler.record(batch_size=bs, latency_ms=200 - bs * 10)

        result = scheduler.predict_optimal_batch_size(target_latency_ms=100.0)
        assert result == 32

    def test_history_bounded(self) -> None:
        scheduler = PredictiveBatchScheduler(history_size=10)
        for i in range(20):
            scheduler.record(batch_size=i, latency_ms=float(i))
        assert len(scheduler._latency_history) == 10
        # Oldest entries should be evicted
        assert scheduler._latency_history[0] == 10.0
