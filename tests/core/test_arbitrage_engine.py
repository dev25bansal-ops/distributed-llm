"""Tests for ArbitrageEngine and SpotEnsembleManager.

Covers:
- PriceHistory: add, current, mean, stddev, min_price, trend_pct
- PricePoint and ArbitrageOpportunity dataclasses
- ArbitrageEngine: pricing updates, opportunity detection (price drop, region, provider, carbon), migration recommendations, trends, summary
- SpotEnsembleManager: provider management, leader election, interruption handling, stats
"""

from __future__ import annotations

import math

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_ae = load_module("distllm/core/arbitrage_engine.py")
ArbitrageEngine = _ae.ArbitrageEngine
SpotEnsembleManager = _ae.SpotEnsembleManager
PriceHistory = _ae.PriceHistory
PricePoint = _ae.PricePoint
ArbitrageOpportunity = _ae.ArbitrageOpportunity
MigrationRecommendation = _ae.MigrationRecommendation
OpportunityType = _ae.OpportunityType
MigrationRisk = _ae.MigrationRisk


# ── PriceHistory ──────────────────────────────────────────────────────────────


class TestPriceHistory:
    def test_empty_defaults(self):
        ph = PriceHistory(provider="aws", instance_type="p4d.24xlarge", region="us-east-1")
        assert ph.current == 0.0
        assert ph.mean == 0.0
        assert ph.stddev == 0.0
        assert ph.min_price == 0.0
        assert ph.trend_pct == 0.0

    def test_add_single_observation(self):
        ph = PriceHistory(provider="aws", instance_type="p4d.24xlarge", region="us-east-1")
        ph.add(10.0)
        assert ph.current == 10.0
        assert ph.mean == 10.0
        assert ph.min_price == 10.0
        assert ph.stddev == 0.0
        assert ph.trend_pct == 0.0

    def test_multiple_observations_stats(self):
        ph = PriceHistory(provider="aws", instance_type="p4d.24xlarge", region="us-east-1")
        for price in (10.0, 12.0, 8.0, 9.0, 11.0):
            ph.add(price)
        assert ph.current == 11.0
        assert ph.min_price == 8.0
        assert len(ph.observations) == 5

    def test_trend_pct_positive(self):
        ph = PriceHistory(provider="aws", instance_type="p4d.24xlarge", region="us-east-1")
        ph.add(10.0)
        ph.add(12.0)
        assert ph.trend_pct == pytest.approx(20.0)

    def test_trend_pct_negative(self):
        ph = PriceHistory(provider="aws", instance_type="p4d.24xlarge", region="us-east-1")
        ph.add(12.0)
        ph.add(9.0)
        assert ph.trend_pct == pytest.approx(-25.0)

    def test_window_size_enforced(self):
        ph = PriceHistory(
            provider="aws", instance_type="p4d.24xlarge", region="us-east-1",
            window_size=3,
        )
        for i in range(10):
            ph.add(float(i))
        assert len(ph.observations) == 3
        assert ph.observations[-1].price == 9.0

    def test_trend_pct_zero_when_first_is_zero(self):
        ph = PriceHistory(provider="aws", instance_type="p4d.24xlarge", region="us-east-1")
        ph.add(0.0)
        ph.add(5.0)
        assert ph.trend_pct == 0.0

    def test_stddev_single_observation(self):
        ph = PriceHistory(provider="aws", instance_type="p4d.24xlarge", region="us-east-1")
        ph.add(10.0)
        assert ph.stddev == 0.0

    def test_stddev_multiple(self):
        ph = PriceHistory(provider="aws", instance_type="p4d.24xlarge", region="us-east-1")
        for v in (10.0, 12.0, 8.0):
            ph.add(v)
        assert ph.stddev > 0


# ── ArbitrageOpportunity ──────────────────────────────────────────────────────


class TestArbitrageOpportunity:
    def test_to_dict(self):
        opp = ArbitrageOpportunity(
            opportunity_type=OpportunityType.PRICE_DROP,
            current_provider="aws",
            current_instance="p4d.24xlarge",
            current_region="us-east-1",
            current_price=10.0,
            recommended_provider="gcp",
            recommended_instance="a2-highgpu-8g",
            recommended_region="us-central1",
            recommended_price=7.0,
            savings_per_hour=3.0,
            savings_pct=30.0,
            migration_risk=MigrationRisk.LOW,
            confidence=0.9,
            reason="price drop detected",
        )
        d = opp.to_dict()
        assert d["type"] == "price_drop"
        assert d["current_price"] == 10.0
        assert d["savings_per_hour"] == 3.0
        assert d["migration_risk"] == "low"
        assert d["confidence"] == 0.9


# ── ArbitrageEngine ───────────────────────────────────────────────────────────


class TestArbitrageEngineConstruction:
    def test_default_values(self):
        engine = ArbitrageEngine()
        assert engine._price_drop_threshold == 15.0
        assert engine._region_savings_threshold == 20.0
        assert engine._provider_savings_threshold == 25.0
        assert engine._history_window == 100
        assert engine._on_opportunity is None

    def test_custom_values(self):
        cb = lambda opp: None
        engine = ArbitrageEngine(
            price_drop_threshold_pct=10.0,
            region_savings_threshold_pct=15.0,
            provider_savings_threshold_pct=20.0,
            history_window=50,
            on_opportunity=cb,
        )
        assert engine._price_drop_threshold == 10.0
        assert engine._on_opportunity is cb


class TestArbitrageEnginePricing:
    def test_update_pricing_creates_history(self):
        engine = ArbitrageEngine()
        engine.update_pricing("aws", "p4d.24xlarge", "us-east-1", 10.0)
        key = "aws:p4d.24xlarge:us-east-1"
        assert key in engine._histories
        assert engine._histories[key].current == 10.0

    def test_update_pricing_batch(self):
        engine = ArbitrageEngine()
        prices = [
            {"provider": "aws", "instance_type": "p4d.24xlarge", "region": "us-east-1", "price": 10.0},
            {"provider": "gcp", "instance_type": "a2-highgpu-8g", "region": "us-central1", "price": 8.0},
        ]
        engine.update_pricing_batch(prices)
        assert "aws:p4d.24xlarge:us-east-1" in engine._histories
        assert "gcp:a2-highgpu-8g:us-central1" in engine._histories

    def test_set_active_location(self):
        engine = ArbitrageEngine()
        engine.set_active_location("aws", "p4d.24xlarge", "us-east-1")
        assert engine._active_provider == "aws"
        assert engine._active_instance == "p4d.24xlarge"
        assert engine._active_region == "us-east-1"

    def test_set_carbon_data(self):
        engine = ArbitrageEngine()
        data = {"us-east-1": 500.0, "us-west-1": 200.0}
        engine.set_carbon_data(data)
        assert engine._carbon_data == data


class TestArbitrageEngineOpportunityDetection:
    def test_no_opportunities_without_data(self):
        engine = ArbitrageEngine()
        engine.set_active_location("aws", "p4d.24xlarge", "us-east-1")
        opps = engine.detect_opportunities()
        assert opps == []

    def test_price_drop_detection(self):
        engine = ArbitrageEngine(price_drop_threshold_pct=10.0)
        # Add enough drop data: mean ~10, current ~5 => 50% drop
        for p in (10.0, 10.0, 10.0):
            engine.update_pricing("aws", "p4d.24xlarge", "us-east-1", p)
        engine.update_pricing("aws", "p4d.24xlarge", "us-east-1", 5.0)
        opps = engine.detect_opportunities()
        assert any(o.opportunity_type == OpportunityType.PRICE_DROP for o in opps)

    def test_price_drop_below_threshold_skipped(self):
        engine = ArbitrageEngine(price_drop_threshold_pct=50.0)
        for p in (10.0, 10.0, 10.0):
            engine.update_pricing("aws", "p4d.24xlarge", "us-east-1", p)
        engine.update_pricing("aws", "p4d.24xlarge", "us-east-1", 8.0)
        opps = engine.detect_opportunities()
        assert not any(o.opportunity_type == OpportunityType.PRICE_DROP for o in opps)

    def test_cheaper_region_detected(self):
        engine = ArbitrageEngine(region_savings_threshold_pct=10.0)
        engine.set_active_location("aws", "p4d.24xlarge", "us-east-1")
        # Current region price
        engine.update_pricing("aws", "p4d.24xlarge", "us-east-1", 10.0)
        # Cheaper region
        engine.update_pricing("aws", "p4d.24xlarge", "us-west-2", 7.0)
        opps = engine.detect_opportunities()
        assert any(o.opportunity_type == OpportunityType.CHEAPER_REGION for o in opps)

    def test_cheaper_region_skipped_when_not_set(self):
        engine = ArbitrageEngine()
        # Don't set active location
        engine.update_pricing("aws", "p4d.24xlarge", "us-east-1", 10.0)
        opps = engine.detect_opportunities()
        assert not any(o.opportunity_type == OpportunityType.CHEAPER_REGION for o in opps)

    def test_cheaper_provider_detected(self):
        engine = ArbitrageEngine(provider_savings_threshold_pct=10.0)
        engine.set_active_location("aws", "p4d.24xlarge", "us-east-1")
        engine.update_pricing("aws", "p4d.24xlarge", "us-east-1", 10.0)
        engine.update_pricing("gcp", "a2-highgpu-8g", "us-central1", 7.0)
        opps = engine.detect_opportunities()
        assert any(o.opportunity_type == OpportunityType.CHEAPER_PROVIDER for o in opps)

    def test_carbon_switch_detected(self):
        engine = ArbitrageEngine()
        engine.set_active_location("aws", "p4d.24xlarge", "us-east-1")
        engine.set_carbon_data({"us-east-1": 500.0, "us-west-2": 100.0})
        # Same provider, same instance, different region at similar price
        engine.update_pricing("aws", "p4d.24xlarge", "us-east-1", 10.0)
        engine.update_pricing("aws", "p4d.24xlarge", "us-west-2", 9.5)  # Within 10%
        opps = engine.detect_opportunities()
        assert any(o.opportunity_type == OpportunityType.CARBON_SWITCH for o in opps)

    def test_carbon_switch_skipped_when_price_diff_too_large(self):
        engine = ArbitrageEngine()
        engine.set_active_location("aws", "p4d.24xlarge", "us-east-1")
        engine.set_carbon_data({"us-east-1": 500.0, "us-west-2": 100.0})
        engine.update_pricing("aws", "p4d.24xlarge", "us-east-1", 10.0)
        engine.update_pricing("aws", "p4d.24xlarge", "us-west-2", 15.0)  # 50% diff
        opps = engine.detect_opportunities()
        assert not any(o.opportunity_type == OpportunityType.CARBON_SWITCH for o in opps)

    def test_carbon_skipped_when_no_carbon_data(self):
        engine = ArbitrageEngine()
        engine.set_active_location("aws", "p4d.24xlarge", "us-east-1")
        engine.update_pricing("aws", "p4d.24xlarge", "us-east-1", 10.0)
        engine.update_pricing("aws", "p4d.24xlarge", "us-west-2", 9.5)
        opps = engine.detect_opportunities()
        assert not any(o.opportunity_type == OpportunityType.CARBON_SWITCH for o in opps)

    def test_on_opportunity_callback_invoked(self):
        fired = []

        def callback(opp):
            fired.append(opp)

        engine = ArbitrageEngine(
            price_drop_threshold_pct=1.0,
            on_opportunity=callback,
        )
        for p in (10.0, 10.0, 10.0):
            engine.update_pricing("aws", "p4d.24xlarge", "us-east-1", p)
        engine.update_pricing("aws", "p4d.24xlarge", "us-east-1", 1.0)
        engine.detect_opportunities()
        assert len(fired) >= 1

    def test_on_opportunity_callback_exception_handled(self):
        def callback(opp):
            raise ValueError("boom")

        engine = ArbitrageEngine(
            price_drop_threshold_pct=1.0,
            on_opportunity=callback,
        )
        for p in (10.0, 10.0, 10.0):
            engine.update_pricing("aws", "p4d.24xlarge", "us-east-1", p)
        engine.update_pricing("aws", "p4d.24xlarge", "us-east-1", 1.0)
        opps = engine.detect_opportunities()
        assert len(opps) >= 1


class TestArbitrageEngineMigrationRecommendations:
    def test_generates_recommendations(self):
        engine = ArbitrageEngine(price_drop_threshold_pct=1.0)
        for p in (10.0, 10.0, 10.0):
            engine.update_pricing("aws", "p4d.24xlarge", "us-east-1", p)
        engine.update_pricing("aws", "p4d.24xlarge", "us-east-1", 1.0)
        engine.detect_opportunities()
        recs = engine.generate_migration_recommendations()
        assert len(recs) >= 1
        assert recs[0].action in ("migrate", "checkpoint_and_migrate", "plan_migration")

    def test_recommendation_risk_mapping(self):
        engine = ArbitrageEngine()
        engine.set_active_location("aws", "p4d.24xlarge", "us-east-1")
        engine.update_pricing("aws", "p4d.24xlarge", "us-east-1", 10.0)
        engine.update_pricing("gcp", "a2-highgpu-8g", "us-central1", 1.0)
        # Very high savings => different provider => HIGH risk
        engine._provider_savings_threshold = 1.0
        opps = engine.detect_opportunities()
        recs = engine.generate_migration_recommendations()
        high_recs = [r for r in recs if r.risk == MigrationRisk.HIGH]
        if high_recs:
            assert high_recs[0].action == "plan_migration"


class TestArbitrageEngineQueryMethods:
    def test_get_price_trends(self):
        engine = ArbitrageEngine()
        engine.update_pricing("aws", "p4d.24xlarge", "us-east-1", 10.0)
        engine.update_pricing("gcp", "a2-highgpu-8g", "us-central1", 8.0)
        trends = engine.get_price_trends()
        assert len(trends) == 2
        # Sorted by current_price ascending
        assert trends[0]["current_price"] <= trends[1]["current_price"]

    def test_get_summary(self):
        engine = ArbitrageEngine()
        engine.set_active_location("aws", "p4d.24xlarge", "us-east-1")
        engine.update_pricing("aws", "p4d.24xlarge", "us-east-1", 10.0)
        engine.detect_opportunities()
        summary = engine.get_summary()
        assert summary["tracked_instances"] == 1
        assert "active_location" in summary
        assert "opportunities_detected" in summary


# ── SpotEnsembleManager ───────────────────────────────────────────────────────


class TestSpotEnsembleManager:
    def test_default_values(self):
        mgr = SpotEnsembleManager()
        assert mgr._active_leader is None
        assert mgr._check_interval == 30.0
        assert mgr._spare_sync_interval == 60.0
        assert mgr._running is False
        assert mgr._migration_callback is None

    def test_add_first_provider_becomes_leader(self):
        mgr = SpotEnsembleManager()
        mgr.add_provider("aws", "p4d.24xlarge", "us-east-1", spot_price=3.0)
        assert mgr.get_leader() == "aws:p4d.24xlarge:us-east-1"

    def test_add_provider_does_not_change_leader(self):
        mgr = SpotEnsembleManager()
        mgr.add_provider("aws", "p4d.24xlarge", "us-east-1", spot_price=3.0)
        mgr.add_provider("gcp", "a2-highgpu-8g", "us-central1", spot_price=2.0)
        # Leader stays as first added (cheaper does not auto-promote)
        assert mgr.get_leader() == "aws:p4d.24xlarge:us-east-1"

    def test_get_healthy_spares(self):
        mgr = SpotEnsembleManager()
        mgr.add_provider("aws", "p4d.24xlarge", "us-east-1", spot_price=3.0)
        mgr.add_provider("gcp", "a2-highgpu-8g", "us-central1", spot_price=2.0)
        mgr.add_provider("azure", "nd96asr_v4", "eastus", spot_price=2.5)
        spares = mgr.get_healthy_spares()
        assert len(spares) == 2
        assert mgr.get_leader() not in spares

    def test_remove_provider(self):
        mgr = SpotEnsembleManager()
        mgr.add_provider("aws", "p4d.24xlarge", "us-east-1")
        mgr.add_provider("gcp", "a2-highgpu-8g", "us-central1")
        mgr.remove_provider("aws:p4d.24xlarge:us-east-1")
        assert mgr.get_leader() == "gcp:a2-highgpu-8g:us-central1"
        assert mgr.get_leader() is not None

    def test_remove_unknown_provider_safe(self):
        mgr = SpotEnsembleManager()
        mgr.add_provider("aws", "p4d.24xlarge", "us-east-1")
        mgr.remove_provider("nonexistent")  # Should not raise

    def test_report_interruption_promotes_cheapest_healthy(self):
        mgr = SpotEnsembleManager()
        mgr.add_provider("aws", "p4d.24xlarge", "us-east-1", spot_price=5.0)
        mgr.add_provider("gcp", "a2-highgpu-8g", "us-central1", spot_price=2.0)
        mgr.add_provider("azure", "nd96asr_v4", "eastus", spot_price=3.0)
        # Current leader is aws (first added, not cheapest)
        assert mgr.get_leader() == "aws:p4d.24xlarge:us-east-1"
        # Report interruption on leader
        mgr.report_interruption("aws:p4d.24xlarge:us-east-1")
        # Cheapest healthy should be promoted
        assert mgr.get_leader() == "gcp:a2-highgpu-8g:us-central1"
        assert mgr._interruptions == 1
        assert mgr._leader_changes == 1

    def test_report_interruption_marks_unhealthy(self):
        mgr = SpotEnsembleManager()
        mgr.add_provider("aws", "p4d.24xlarge", "us-east-1")
        mgr.report_interruption("aws:p4d.24xlarge:us-east-1")
        assert mgr._providers["aws:p4d.24xlarge:us-east-1"]["healthy"] is False

    def test_report_interruption_unknown_provider_safe(self):
        mgr = SpotEnsembleManager()
        mgr.report_interruption("nonexistent")  # Should not raise

    def test_interruption_callback_invoked(self):
        calls = []

        def callback(from_key, to_key, preconditions):
            calls.append((from_key, to_key))

        mgr = SpotEnsembleManager(migration_callback=callback)
        mgr.add_provider("aws", "p4d.24xlarge", "us-east-1", spot_price=5.0)
        mgr.add_provider("gcp", "a2-highgpu-8g", "us-central1", spot_price=2.0)
        mgr.report_interruption("aws:p4d.24xlarge:us-east-1")
        assert len(calls) == 1
        assert calls[0][0] == "aws:p4d.24xlarge:us-east-1"
        assert calls[0][1] == "gcp:a2-highgpu-8g:us-central1"

    def test_interruption_callback_exception_handled(self):
        def callback(from_key, to_key, preconditions):
            raise ValueError("boom")

        mgr = SpotEnsembleManager(migration_callback=callback)
        mgr.add_provider("aws", "p4d.24xlarge", "us-east-1", spot_price=5.0)
        mgr.add_provider("gcp", "a2-highgpu-8g", "us-central1", spot_price=2.0)
        mgr.report_interruption("aws:p4d.24xlarge:us-east-1")

    def test_elect_leader_when_all_unhealthy_returns_none(self):
        mgr = SpotEnsembleManager()
        mgr.add_provider("aws", "p4d.24xlarge", "us-east-1")
        mgr.report_interruption("aws:p4d.24xlarge:us-east-1")
        assert mgr.get_leader() is None

    def test_stats(self):
        mgr = SpotEnsembleManager()
        mgr.add_provider("aws", "p4d.24xlarge", "us-east-1")
        mgr.add_provider("gcp", "a2-highgpu-8g", "us-central1")
        s = mgr.stats
        assert s["providers"] == 2
        assert s["active_leader"] is not None
        assert s["healthy_spares"] == 1
        assert s["interruptions"] == 0
        assert s["leader_changes"] == 0

    def test_start_stop_no_error(self):
        mgr = SpotEnsembleManager(check_interval_s=0.01)
        mgr.start()
        assert mgr._running is True
        mgr.stop()
        assert mgr._running is False

    def test_start_twice_is_idempotent(self):
        mgr = SpotEnsembleManager(check_interval_s=0.01)
        mgr.start()
        thread_id = id(mgr._thread)
        mgr.start()  # Should be no-op
        assert id(mgr._thread) == thread_id
        mgr.stop()


# ── Edge Cases ────────────────────────────────────────────────────────────────


class TestArbitrageEngineEdgeCases:
    def test_empty_price_history_mean_stddev(self):
        ph = PriceHistory(provider="a", instance_type="b", region="c")
        assert ph.mean == 0.0
        assert ph.stddev == 0.0
        assert ph.min_price == 0.0

    def test_pricepoint_factory_defaults(self):
        pp = PricePoint(provider="a", instance_type="b", region="c", price=1.0)
        assert pp.is_spot is True

    def test_migration_recommendation_dataclass(self):
        rec = MigrationRecommendation(
            from_provider="aws",
            from_region="us-east-1",
            to_provider="gcp",
            to_region="us-central1",
            from_price=10.0,
            to_price=7.0,
            estimated_savings_hourly=3.0,
            risk=MigrationRisk.MEDIUM,
        )
        assert rec.action == ""
        assert rec.preconditions == []
