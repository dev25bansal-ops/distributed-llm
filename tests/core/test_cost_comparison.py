"""Tests for CostComparison, ComparisonReport, and ComparisonRow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_cc_mod = load_module("distllm/core/cost_comparison.py")
CostComparison = _cc_mod.CostComparison
ComparisonReport = _cc_mod.ComparisonReport
ComparisonRow = _cc_mod.ComparisonRow


@dataclass
class _FakeRouter:
    """Deterministic router stub for CostComparison tests."""
    prices: list[dict[str, Any]] = field(default_factory=list)

    def get_all_prices(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.prices


@dataclass
class _FakePricingManager:
    """Deterministic pricing-manager stub for CostComparison tests."""
    pricing: list[dict[str, Any]] = field(default_factory=list)

    def get_all_pricing(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.pricing


class TestComparisonRow:
    """ComparisonRow dataclass and best_price property."""

    def test_best_price_on_demand_only(self):
        row = ComparisonRow(
            provider="aws", instance_type="p4d", region="us-east-1",
            gpu_type="A100", gpu_count=8, gpu_memory_gb=320.0,
            on_demand_hourly=32.0, spot_hourly=0.0,
            total_on_demand=768.0, total_spot=0.0,
            latency_ms=10.0, carbon_gco2_kwh=500.0, renewable_pct=30.0,
            available=True, score=0.0,
        )
        assert row.best_price == 768.0

    def test_best_price_spot_cheaper(self):
        row = ComparisonRow(
            provider="gcp", instance_type="a2-highgpu", region="us-central1",
            gpu_type="A100", gpu_count=8, gpu_memory_gb=320.0,
            on_demand_hourly=35.0, spot_hourly=10.0,
            total_on_demand=840.0, total_spot=240.0,
            latency_ms=12.0, carbon_gco2_kwh=400.0, renewable_pct=60.0,
            available=True, score=0.0,
        )
        assert row.best_price == 240.0


class TestComparisonReport:
    """ComparisonReport construction and output formatting."""

    def test_empty_report(self):
        report = ComparisonReport(gpu_type="A100", hours=24, regions=[], rows=[])
        table = report.to_table()
        assert "No matching instances found." in table

    def test_single_row_report(self):
        rows = [
            ComparisonRow(
                provider="aws", instance_type="p4d", region="us-east-1",
                gpu_type="A100", gpu_count=8, gpu_memory_gb=320.0,
                on_demand_hourly=32.0, spot_hourly=0.0,
                total_on_demand=768.0, total_spot=0.0,
                latency_ms=10.0, carbon_gco2_kwh=500.0, renewable_pct=30.0,
                available=True, score=10.0,
            ),
        ]
        report = ComparisonReport(
            gpu_type="A100", hours=24, regions=["us-east-1"], rows=rows,
        )
        table = report.to_table()
        assert "A100" in table
        assert "p4d" in table
        assert "us-east-1" in table
        assert "$768.00" in table

    def test_report_with_regions_empty(self):
        rows = [
            ComparisonRow(
                provider="aws", instance_type="p4d", region="us-east-1",
                gpu_type="A100", gpu_count=8, gpu_memory_gb=320.0,
                on_demand_hourly=32.0, spot_hourly=0.0,
                total_on_demand=768.0, total_spot=0.0,
                latency_ms=10.0, carbon_gco2_kwh=500.0, renewable_pct=30.0,
                available=True, score=10.0,
            ),
        ]
        report = ComparisonReport(
            gpu_type="A100", hours=24, regions=[], rows=rows,
        )
        table = report.to_table()
        assert "all" in table

    def test_to_dict(self):
        rows = [
            ComparisonRow(
                provider="aws", instance_type="p4d", region="us-east-1",
                gpu_type="A100", gpu_count=8, gpu_memory_gb=320.0,
                on_demand_hourly=32.0, spot_hourly=0.0,
                total_on_demand=768.0, total_spot=0.0,
                latency_ms=10.0, carbon_gco2_kwh=500.0, renewable_pct=30.0,
                available=True, score=10.0,
            ),
        ]
        report = ComparisonReport(
            gpu_type="A100", hours=24, regions=["us-east-1"], rows=rows,
        )
        d = report.to_dict()
        assert d["gpu_type"] == "A100"
        assert d["hours"] == 24
        assert len(d["rows"]) == 1
        assert d["rows"][0]["provider"] == "aws"
        assert d["rows"][0]["total_cost"] == 768.0

    def test_multi_row_savings_summary(self):
        rows = [
            ComparisonRow(
                provider="gcp", instance_type="a2-highgpu", region="us-central1",
                gpu_type="A100", gpu_count=8, gpu_memory_gb=320.0,
                on_demand_hourly=35.0, spot_hourly=10.0,
                total_on_demand=840.0, total_spot=240.0,
                latency_ms=12.0, carbon_gco2_kwh=400.0, renewable_pct=60.0,
                available=True, score=5.0,
            ),
            ComparisonRow(
                provider="aws", instance_type="p4d", region="us-east-1",
                gpu_type="A100", gpu_count=8, gpu_memory_gb=320.0,
                on_demand_hourly=32.0, spot_hourly=0.0,
                total_on_demand=768.0, total_spot=0.0,
                latency_ms=10.0, carbon_gco2_kwh=500.0, renewable_pct=30.0,
                available=True, score=10.0,
            ),
        ]
        report = ComparisonReport(
            gpu_type="A100", hours=24, regions=["us-central1", "us-east-1"], rows=rows,
        )
        table = report.to_table()
        assert "Savings" in table
        assert "Cleanest" in table


class TestCostComparison:
    """CostComparison construction and comparison logic."""

    def test_construction(self):
        cc = CostComparison()
        assert cc._router is None
        assert cc._pricing_manager is None

    def test_construction_with_router(self):
        router = _FakeRouter()
        cc = CostComparison(router=router)
        assert cc._router is router

    def test_compare_with_router(self):
        router = _FakeRouter(prices=[
            {
                "provider": "aws", "instance_type": "p4d",
                "region": "us-east-1", "price_per_hour": 32.0,
                "spot_price": 0.0, "latency_ms": 10.0,
                "carbon_gco2_kwh": 500.0, "renewable_pct": 30.0,
                "gpu_memory_gb": 320.0, "gpu_count": 8, "available": True,
            },
        ])
        cc = CostComparison(router=router)
        report = cc.compare(gpu_type="A100", hours=24)
        assert len(report.rows) == 1
        assert report.rows[0].provider == "aws"
        assert report.rows[0].total_on_demand == 768.0

    def test_compare_filters_by_region(self):
        router = _FakeRouter(prices=[
            {
                "provider": "aws", "instance_type": "p4d",
                "region": "us-east-1", "price_per_hour": 32.0,
                "spot_price": 0.0, "latency_ms": 0.0,
                "carbon_gco2_kwh": 0.0, "renewable_pct": 0.0,
                "gpu_memory_gb": 0.0, "gpu_count": 1, "available": True,
            },
            {
                "provider": "gcp", "instance_type": "a2-highgpu",
                "region": "eu-west-1", "price_per_hour": 35.0,
                "spot_price": 0.0, "latency_ms": 0.0,
                "carbon_gco2_kwh": 0.0, "renewable_pct": 0.0,
                "gpu_memory_gb": 0.0, "gpu_count": 1, "available": True,
            },
        ])
        cc = CostComparison(router=router)
        report = cc.compare(gpu_type="A100", hours=24, regions=["us-east-1"])
        assert len(report.rows) == 1
        assert report.rows[0].provider == "aws"

    def test_compare_no_router_no_pricing(self):
        cc = CostComparison()
        report = cc.compare(gpu_type="A100", hours=24)
        assert len(report.rows) == 0

    def test_compute_score(self):
        score = CostComparison._compute_score(
            on_demand=10.0, spot=5.0, latency=50.0, carbon=400.0,
        )
        # price * 0.6 = 5.0 * 0.6 = 3.0
        # latency: (50/200) * 0.2 = 0.05
        # carbon: (400/800) * 0.2 = 0.1
        assert score == pytest.approx(3.15)

    def test_compare_with_pricing_manager(self):
        pricing = _FakePricingManager(pricing=[])
        cc = CostComparison(pricing_manager=pricing)
        report = cc.compare(gpu_type="A100", hours=24)
        assert len(report.rows) == 0
