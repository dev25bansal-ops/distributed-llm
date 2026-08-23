"""Tests for ClusterStatus, FederatedRoute, and FederatedScheduler."""

from __future__ import annotations

import time

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_fed = load_module("distllm/core/advanced_scheduling/federated.py")
ClusterStatus = _fed.ClusterStatus
FederatedRoute = _fed.FederatedRoute
FederatedScheduler = _fed.FederatedScheduler


class TestClusterStatus:
    """Test suite for ClusterStatus dataclass."""

    def test_default_construction(self) -> None:
        status = ClusterStatus(cluster_id="us-east")
        assert status.cluster_id == "us-east"
        assert status.active_requests == 0
        assert status.pending_requests == 0
        assert status.gpu_utilization == 0.0
        assert status.is_healthy is True
        assert isinstance(status.last_heartbeat, float)

    def test_custom_values(self) -> None:
        status = ClusterStatus(
            cluster_id="eu-west",
            active_requests=10,
            pending_requests=3,
            gpu_utilization=75.0,
            is_healthy=False,
        )
        assert status.active_requests == 10
        assert status.gpu_utilization == 75.0
        assert status.is_healthy is False


class TestFederatedRoute:
    """Test suite for FederatedRoute dataclass."""

    def test_default_construction(self) -> None:
        route = FederatedRoute(target_cluster="us-east", reason="test")
        assert route.target_cluster == "us-east"
        assert route.reason == "test"
        assert route.estimated_latency_ms == 0.0
        assert route.estimated_cost == 0.0

    def test_custom_values(self) -> None:
        route = FederatedRoute(
            target_cluster="eu-west",
            reason="lowest_utilization",
            estimated_latency_ms=120.0,
            estimated_cost=0.05,
        )
        assert route.estimated_latency_ms == 120.0
        assert route.estimated_cost == 0.05


class TestFederatedScheduler:
    """Test suite for FederatedScheduler."""

    def test_default_construction(self) -> None:
        scheduler = FederatedScheduler()
        assert scheduler._spillover_threshold == 80.0
        assert scheduler._clusters == {}

    def test_custom_spillover_threshold(self) -> None:
        scheduler = FederatedScheduler(spillover_threshold=90.0)
        assert scheduler._spillover_threshold == 90.0

    def test_update_cluster(self) -> None:
        scheduler = FederatedScheduler()
        status = ClusterStatus(cluster_id="us-east", gpu_utilization=45.0)
        scheduler.update_cluster(status)
        assert scheduler._clusters["us-east"] is status

    def test_update_cluster_overwrites(self) -> None:
        scheduler = FederatedScheduler()
        scheduler.update_cluster(ClusterStatus(cluster_id="us-east", gpu_utilization=45.0))
        scheduler.update_cluster(ClusterStatus(cluster_id="us-east", gpu_utilization=90.0))
        assert scheduler._clusters["us-east"].gpu_utilization == 90.0

    def test_should_spillover_below_threshold(self) -> None:
        scheduler = FederatedScheduler(spillover_threshold=80.0)
        assert scheduler.should_spillover(50.0) is False

    def test_should_spillover_at_threshold(self) -> None:
        scheduler = FederatedScheduler(spillover_threshold=80.0)
        assert scheduler.should_spillover(80.0) is False  # not strictly greater

    def test_should_spillover_above_threshold(self) -> None:
        scheduler = FederatedScheduler(spillover_threshold=80.0)
        assert scheduler.should_spillover(85.0) is True

    def test_select_target_returns_lowest_utilization(self) -> None:
        scheduler = FederatedScheduler()
        scheduler.update_cluster(ClusterStatus(cluster_id="a", gpu_utilization=90.0))
        scheduler.update_cluster(ClusterStatus(cluster_id="b", gpu_utilization=30.0))
        scheduler.update_cluster(ClusterStatus(cluster_id="c", gpu_utilization=60.0))

        route = scheduler.select_target()
        assert route is not None
        assert route.target_cluster == "b"
        assert route.reason == "lowest_utilization"
        assert route.estimated_latency_ms == 50.0

    def test_select_target_excludes_cluster(self) -> None:
        scheduler = FederatedScheduler()
        scheduler.update_cluster(ClusterStatus(cluster_id="a", gpu_utilization=90.0))
        scheduler.update_cluster(ClusterStatus(cluster_id="b", gpu_utilization=30.0))

        route = scheduler.select_target(exclude="a")
        assert route is not None
        assert route.target_cluster == "b"

    def test_select_target_excludes_unhealthy(self) -> None:
        scheduler = FederatedScheduler()
        scheduler.update_cluster(
            ClusterStatus(cluster_id="a", gpu_utilization=10.0, is_healthy=False)
        )
        scheduler.update_cluster(
            ClusterStatus(cluster_id="b", gpu_utilization=50.0, is_healthy=True)
        )

        route = scheduler.select_target()
        assert route is not None
        assert route.target_cluster == "b"

    def test_select_target_no_candidates_returns_none(self) -> None:
        scheduler = FederatedScheduler()
        route = scheduler.select_target()
        assert route is None

    def test_select_target_all_excluded_returns_none(self) -> None:
        scheduler = FederatedScheduler()
        scheduler.update_cluster(ClusterStatus(cluster_id="a", gpu_utilization=50.0))
        route = scheduler.select_target(exclude="a")
        assert route is None

    def test_select_target_all_unhealthy_returns_none(self) -> None:
        scheduler = FederatedScheduler()
        scheduler.update_cluster(
            ClusterStatus(cluster_id="a", gpu_utilization=50.0, is_healthy=False)
        )
        route = scheduler.select_target()
        assert route is None

    def test_thread_safety(self) -> None:
        """Multiple updates from different threads do not corrupt state."""
        import threading

        scheduler = FederatedScheduler()

        def add_cluster(cid: str) -> None:
            scheduler.update_cluster(ClusterStatus(cluster_id=cid, gpu_utilization=50.0))

        threads = [threading.Thread(target=add_cluster, args=(f"cluster-{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(scheduler._clusters) == 10
