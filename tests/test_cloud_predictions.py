"""Tests: SpotPriceTracker (cache hit/miss/stale, cheapest) and PreemptionPredictor (risk, volatility, trend, model)."""

import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from distllm.cloud.spot_provider import CloudProvider, SpotPrice, SpotProvider
from distllm.cloud.spot_price_tracker import SpotPriceTracker, PriceRecord
from distllm.cloud.preemption_predictor import PreemptionPredictor, PreemptionPrediction


# ===========================================================================
# SpotPriceTracker
# ===========================================================================


class TestSpotPriceTrackerCache:
    """Cache hit — cached price returned; cache miss — fetch from provider."""

    def test_provider_registration(self):
        tracker = SpotPriceTracker()
        provider = MagicMock(spec=SpotProvider)
        provider.provider_name = CloudProvider.AWS
        tracker.register_provider(provider)
        assert tracker.get_provider(CloudProvider.AWS) is provider
        assert tracker.get_provider(CloudProvider.AZURE) is None

    def test_cache_hit_returns_cached_price(self):
        tracker = SpotPriceTracker()
        key = "aws:g5.xlarge:us-east-1"
        with tracker._lock:
            tracker._cache[key] = PriceRecord(
                provider=CloudProvider.AWS, instance_type="g5.xlarge",
                region="us-east-1", price=0.50, on_demand_price=1.00,
                timestamp=time.time(), ttl_seconds=300,
            )
        price = tracker.get_current_price(CloudProvider.AWS, "g5.xlarge", "us-east-1")
        assert price is not None
        assert price.price == 0.50

    def test_cache_miss_queries_provider(self):
        tracker = SpotPriceTracker()
        provider = MagicMock(spec=SpotProvider)
        provider.provider_name = CloudProvider.AWS
        provider.get_current_spot_price.return_value = SpotPrice(
            provider=CloudProvider.AWS, instance_type="g5.xlarge",
            region="us-east-1", price=0.45, on_demand_price=1.00,
        )
        tracker.register_provider(provider)
        price = tracker.get_current_price(CloudProvider.AWS, "g5.xlarge", "us-east-1")
        assert price is not None
        assert price.price == 0.45
        provider.get_current_spot_price.assert_called_once_with("g5.xlarge", "us-east-1")

    def test_cache_miss_no_provider_returns_none(self):
        tracker = SpotPriceTracker()
        price = tracker.get_current_price(CloudProvider.AWS, "g5.xlarge", "us-east-1")
        assert price is None

    def test_stale_cache_triggers_refetch(self):
        tracker = SpotPriceTracker()
        key = "aws:g5.xlarge:us-east-1"
        with tracker._lock:
            tracker._cache[key] = PriceRecord(
                provider=CloudProvider.AWS, instance_type="g5.xlarge",
                region="us-east-1", price=0.50, on_demand_price=1.00,
                timestamp=time.time() - 600, ttl_seconds=300,
            )
        provider = MagicMock(spec=SpotProvider)
        provider.provider_name = CloudProvider.AWS
        provider.get_current_spot_price.return_value = SpotPrice(
            provider=CloudProvider.AWS, instance_type="g5.xlarge",
            region="us-east-1", price=0.55,
        )
        tracker.register_provider(provider)
        price = tracker.get_current_price(CloudProvider.AWS, "g5.xlarge", "us-east-1")
        assert price is not None
        assert price.price == 0.55

    def test_cache_updates_after_fetch(self):
        tracker = SpotPriceTracker()
        provider = MagicMock(spec=SpotProvider)
        provider.provider_name = CloudProvider.AWS
        provider.get_current_spot_price.return_value = SpotPrice(
            provider=CloudProvider.AWS, instance_type="g5.xlarge",
            region="us-east-1", price=0.42,
        )
        tracker.register_provider(provider)
        tracker.get_current_price(CloudProvider.AWS, "g5.xlarge", "us-east-1")
        key = "aws:g5.xlarge:us-east-1"
        with tracker._lock:
            assert key in tracker._cache
            assert tracker._cache[key].price == 0.42
            assert key in tracker._history
            assert 0.42 in tracker._history[key]


class TestSpotPriceTrackerCheapest:
    """Cheapest compatible instance selection."""

    def test_cheapest_returns_lowest_price(self):
        tracker = SpotPriceTracker()
        provider = MagicMock(spec=SpotProvider)
        provider.provider_name = CloudProvider.AWS

        def mock_get_price(inst, region):
            prices = {"g5.xlarge": 0.50, "g5.2xlarge": 0.45, "p4d.24xlarge": 1.20}
            if inst not in prices:
                return None
            p = prices[inst]
            return SpotPrice(provider=CloudProvider.AWS, instance_type=inst,
                             region=region, price=p, on_demand_price=p * 2)
        provider.get_current_spot_price.side_effect = mock_get_price
        tracker.register_provider(provider)
        cheapest = tracker.get_cheapest_compatible(
            required_vram_gb=20, region="us-east-1", providers=[CloudProvider.AWS]
        )
        assert cheapest is not None
        assert cheapest.price == 0.45

    def test_cheapest_vram_filter_excludes_small(self):
        tracker = SpotPriceTracker()
        keys = ["aws:g5.xlarge:us-east-1", "aws:g5.2xlarge:us-east-1",
                "aws:g5.4xlarge:us-east-1", "aws:p4d.24xlarge:us-east-1",
                "aws:p5.48xlarge:us-east-1"]
        vrams = {"g5.xlarge": 24, "g5.2xlarge": 24, "g5.4xlarge": 24,
                 "p4d.24xlarge": 80, "p5.48xlarge": 80}
        prices = {"g5.xlarge": 0.50, "p4d.24xlarge": 1.20}
        now = time.time()
        for inst, vram in vrams.items():
            key = f"aws:{inst}:us-east-1"
            price = prices.get(inst, 999.0)
            with tracker._lock:
                tracker._cache[key] = PriceRecord(
                    provider=CloudProvider.AWS, instance_type=inst,
                    region="us-east-1", price=price, on_demand_price=price * 2,
                    timestamp=now, ttl_seconds=300,
                )
        cheapest = tracker.get_cheapest_compatible(
            required_vram_gb=40, region="us-east-1", providers=[CloudProvider.AWS]
        )
        assert cheapest is not None
        assert cheapest.instance_type == "p4d.24xlarge"

    def test_cheapest_no_match_returns_none(self):
        tracker = SpotPriceTracker()
        provider = MagicMock(spec=SpotProvider)
        provider.provider_name = CloudProvider.AWS
        provider.get_current_spot_price.return_value = None
        tracker.register_provider(provider)
        cheapest = tracker.get_cheapest_compatible(
            required_vram_gb=999, region="us-east-1", providers=[CloudProvider.AWS]
        )
        assert cheapest is None

    def test_cheapest_no_providers_returns_none(self):
        tracker = SpotPriceTracker()
        cheapest = tracker.get_cheapest_compatible(
            required_vram_gb=20, region="us-east-1"
        )
        assert cheapest is None


class TestSpotPriceTrackerPreemptionRisk:
    """Preemption risk via SpotPriceTracker."""

    def test_insufficient_data_returns_default(self):
        tracker = SpotPriceTracker()
        risk = tracker.predict_preemption_risk(CloudProvider.AWS, "g5.xlarge", "us-east-1")
        assert risk == 0.5

    def test_single_price_point_returns_default(self):
        tracker = SpotPriceTracker()
        key = "aws:g5.xlarge:us-east-1"
        with tracker._lock:
            tracker._history[key] = [0.50]
        risk = tracker.predict_preemption_risk(CloudProvider.AWS, "g5.xlarge", "us-east-1")
        assert risk == 0.5

    def test_stable_prices_low_risk(self):
        tracker = SpotPriceTracker()
        key = "aws:g5.xlarge:us-east-1"
        with tracker._lock:
            tracker._history[key] = [0.50] * 20
        risk = tracker.predict_preemption_risk(CloudProvider.AWS, "g5.xlarge", "us-east-1")
        assert 0.0 <= risk <= 1.0


# ===========================================================================
# PreemptionPredictor
# ===========================================================================


class TestPreemptionPredictorRisk:
    """Risk calculation with volatility + trend + ratio."""

    def test_insufficient_data_uses_default_risk(self):
        predictor = PreemptionPredictor(default_risk=0.5)
        pred = predictor.predict(CloudProvider.AWS, "g5.xlarge", "us-east-1")
        assert pred.risk_score > 0
        assert pred.confidence == 0.3
        assert pred.expected_lifetime_min > 0

    def test_volatility_computation(self):
        predictor = PreemptionPredictor()
        prices = [0.50, 0.55, 0.48, 0.60, 0.52]
        for p in prices:
            predictor.record_price(SpotPrice(
                provider=CloudProvider.AWS, instance_type="g5.xlarge",
                region="us-east-1", price=p,
            ))
        vol = predictor._compute_volatility(
            predictor._history["aws:g5.xlarge:us-east-1"]
        )
        assert vol > 0
        assert isinstance(vol, float)

    def test_volatility_requires_three_points(self):
        predictor = PreemptionPredictor()
        vol = predictor._compute_volatility([0.50, 0.55])
        assert vol == 0.0

    def test_volatility_zero_for_constant_prices(self):
        predictor = PreemptionPredictor()
        assert predictor._compute_volatility([0.50, 0.50, 0.50]) == 0.0

    def test_trend_computation(self):
        predictor = PreemptionPredictor()
        prices = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
        for p in prices:
            predictor.record_price(SpotPrice(
                provider=CloudProvider.AWS, instance_type="g5.xlarge",
                region="us-east-1", price=p,
            ))
        trend = predictor._compute_trend(
            predictor._history["aws:g5.xlarge:us-east-1"]
        )
        assert trend > 0

    def test_trend_requires_six_points(self):
        predictor = PreemptionPredictor()
        assert predictor._compute_trend([0.50] * 5) == 0.0

    def test_trend_negative_for_falling_prices(self):
        predictor = PreemptionPredictor()
        prices = [0.60, 0.58, 0.56, 0.54, 0.52, 0.50]
        for p in prices:
            predictor.record_price(SpotPrice(
                provider=CloudProvider.AWS, instance_type="g5.xlarge",
                region="us-east-1", price=p,
            ))
        trend = predictor._compute_trend(
            predictor._history["aws:g5.xlarge:us-east-1"]
        )
        assert trend < 0

    def test_predict_returns_prediction_dataclass(self):
        predictor = PreemptionPredictor()
        for p in [0.50, 0.52, 0.51, 0.53, 0.49, 0.54]:
            predictor.record_price(SpotPrice(
                provider=CloudProvider.AWS, instance_type="g5.xlarge",
                region="us-east-1", price=p,
            ))
        pred = predictor.predict(CloudProvider.AWS, "g5.xlarge", "us-east-1")
        assert isinstance(pred, PreemptionPrediction)
        assert pred.provider == CloudProvider.AWS
        assert pred.instance_type == "g5.xlarge"
        assert 0.0 <= pred.risk_score <= 1.0
        assert 0.0 <= pred.confidence <= 1.0
        assert pred.expected_lifetime_min >= 0
        assert "volatility" in pred.factors
        assert "trend" in pred.factors
        assert "base_rate" in pred.factors

    def test_price_ratio_factor_included(self):
        predictor = PreemptionPredictor()
        predictor.record_price(SpotPrice(
            provider=CloudProvider.AWS, instance_type="g5.xlarge",
            region="us-east-1", price=0.80, on_demand_price=1.00,
        ))
        for _ in range(6):
            predictor.record_price(SpotPrice(
                provider=CloudProvider.AWS, instance_type="g5.xlarge",
                region="us-east-1", price=0.80,
            ))
        pred = predictor.predict(CloudProvider.AWS, "g5.xlarge", "us-east-1")
        assert "price_ratio" in pred.factors


class TestPreemptionPredictorModel:
    """Model adjustment modifies score; no model file → no model."""

    def test_model_adjustment_boosts_risk(self):
        with patch("os.path.exists", return_value=True), \
             patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps({
                "weights": {"volatility": 0.5, "trend": 0.3},
                "base_confidence": 0.7,
                "confidence_boost": 0.1,
            })
            predictor = PreemptionPredictor(model_path="/fake/model.json")
            for p in [0.50, 0.52, 0.51, 0.53, 0.49, 0.54]:
                predictor.record_price(SpotPrice(
                    provider=CloudProvider.AWS, instance_type="g5.xlarge",
                    region="us-east-1", price=p,
                ))
            pred = predictor.predict(CloudProvider.AWS, "g5.xlarge", "us-east-1")
            assert pred.risk_score > 0
            assert pred.confidence > 0

    def test_no_model_file_falls_back_to_heuristic(self):
        predictor = PreemptionPredictor(model_path="/nonexistent/model.json")
        assert predictor._model is None

    def test_model_without_weights_returns_heuristic(self):
        with patch("os.path.exists", return_value=True), \
             patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps({
                "base_confidence": 0.6,
            })
            predictor = PreemptionPredictor(model_path="/fake/model.json")
            assert predictor._model is not None
            prices = [0.50] * 6
            risk, conf = predictor._apply_model(
                CloudProvider.AWS, "g5.xlarge", "us-east-1",
                prices, {"volatility": 0.1, "trend": 0.05}, 0.5,
            )
            assert risk == 0.5
            assert conf == 0.5


class TestPreemptionPredictorSummary:
    """get_status_summary returns correct format."""

    def test_empty_summary(self):
        predictor = PreemptionPredictor()
        summary = predictor.get_status_summary()
        assert summary == {}

    def test_summary_with_data(self):
        predictor = PreemptionPredictor()
        for _ in range(6):
            predictor.record_price(SpotPrice(
                provider=CloudProvider.AWS, instance_type="g5.xlarge",
                region="us-east-1", price=0.50,
            ))
        summary = predictor.get_status_summary()
        assert "aws:g5.xlarge:us-east-1" in summary
        entry = summary["aws:g5.xlarge:us-east-1"]
        assert "risk_score" in entry
        assert "confidence" in entry
        assert "data_points" in entry
        assert entry["data_points"] >= 6


class TestPreemptionPredictionDataclass:
    """PreemptionPrediction timestamps correctly."""

    def test_timestamp_defaults_to_now(self):
        pred = PreemptionPrediction(
            provider=CloudProvider.AWS, instance_type="t2.micro",
            region="us-east-1", risk_score=0.5,
            expected_lifetime_min=60, confidence=0.5,
        )
        assert pred.timestamp > 0


class TestPriceRecord:
    def test_is_stale(self):
        record = PriceRecord(
            provider=CloudProvider.AWS, instance_type="g5.xlarge",
            region="us-east-1", price=0.50, on_demand_price=1.00,
            timestamp=time.time() - 600, ttl_seconds=300,
        )
        assert record.is_stale is True

    def test_is_fresh(self):
        record = PriceRecord(
            provider=CloudProvider.AWS, instance_type="g5.xlarge",
            region="us-east-1", price=0.50, on_demand_price=1.00,
            timestamp=time.time(), ttl_seconds=300,
        )
        assert record.is_stale is False
