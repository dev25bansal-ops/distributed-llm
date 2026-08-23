"""Tests for cost surface modelling and arbitrage orchestration.

Classes under test:
  - CostSurfacePoint  (distllm.dist.routing.arbitrage_engine)
  - CostSurface       (distllm.dist.routing.arbitrage_engine)
"""

from __future__ import annotations

from distllm.dist.routing.arbitrage_engine import (
    CostSurface,
    CostSurfacePoint,
)


class TestCostSurfacePoint:
    """CostSurfacePoint construction and field access."""

    def test_create_point(self) -> None:
        point = CostSurfacePoint(
            region="us-east-1",
            provider="aws",
            instance_type="p4d.24xlarge",
            spot_price=12.50,
            hour=14,
        )
        assert point.region == "us-east-1"
        assert point.provider == "aws"
        assert point.spot_price == 12.50
        assert point.hour == 14

    def test_create_multiple_points(self) -> None:
        points = [
            CostSurfacePoint("us-east-1", "aws", "p4d", spot_price=10.0),
            CostSurfacePoint("eu-west-1", "gcp", "a2", spot_price=8.0),
        ]
        assert len(points) == 2


class TestCostSurfaceQuery:
    """CostSurface query, sort, and filter operations."""

    def test_query_filters_by_price(self) -> None:
        surface = CostSurface()
        points = [
            CostSurfacePoint("us-east-1", "aws", "p4d", spot_price=5.0),
            CostSurfacePoint("eu-west-1", "gcp", "a2", spot_price=15.0),
            CostSurfacePoint("ap-south-1", "aws", "p4d", spot_price=3.0),
        ]
        for p in points:
            surface._points.append(p)

        matches = surface.query(max_price=10.0)
        assert len(matches) == 2
        assert all(p.spot_price <= 10.0 for p in matches)

    def test_sort_by_cost_ascending(self) -> None:
        surface = CostSurface()
        points = [
            CostSurfacePoint("us-east-1", "aws", "p4d", spot_price=10.0),
            CostSurfacePoint("eu-west-1", "gcp", "a2", spot_price=3.0),
            CostSurfacePoint("ap-south-1", "aws", "p4d", spot_price=7.0),
        ]
        for p in points:
            surface._points.append(p)

        sorted_ = surface.sort_by_cost(ascending=True)
        assert sorted_[0].spot_price == 3.0
        assert sorted_[-1].spot_price == 10.0

    def test_query_filters_by_provider(self) -> None:
        surface = CostSurface()
        points = [
            CostSurfacePoint("us-east-1", "aws", "p4d"),
            CostSurfacePoint("eu-west-1", "gcp", "a2"),
        ]
        for p in points:
            surface._points.append(p)

        matches = surface.query(providers=["gcp"])
        assert len(matches) == 1
        assert matches[0].provider == "gcp"
