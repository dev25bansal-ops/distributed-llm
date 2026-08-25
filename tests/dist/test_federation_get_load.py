"""Regression tests for federation routing's ``get_load`` contract (C4).

``FederationCoordinator`` routing methods — ``get_peer_slo_status``,
``select_peer_for_sla``, and ``get_best_peer`` (the cache-affinity
router) — call ``self._load_balancer.get_load(pid)``.  That method did
not exist on ``FederationLoadBalancer`` (only ``get_remote_load``
did), so every routing decision raised ``AttributeError`` whenever any
peer had been discovered.

These tests pin:
- ``FederationLoadBalancer.get_load`` exists and returns the current
  ``RemoteClusterLoad`` (or ``None`` for unknown clusters), matching
  what the router consumes: ``is_overloaded`` and
  ``available_capacity`` plus a ``None`` check.
- The three federation routing paths execute end-to-end against a real
  coordinator with real registered peers and pick nodes according to
  the reported load values.
"""

from __future__ import annotations

import time

import pytest

from distllm.dist.federation import FederationConfig, FederationCoordinator
from distllm.dist.p2p.discovery import PeerInfo
from distllm.dist.p2p.load_balancer import (
    FederationLoadBalancer,
    RemoteClusterLoad,
)


def _make_coord(*peer_ids: str) -> FederationCoordinator:
    """Real coordinator with *peer_ids* registered as fresh peers."""
    coord = FederationCoordinator(FederationConfig(), "local", "127.0.0.1", 1)
    for pid in peer_ids:
        coord._peers[pid] = PeerInfo(
            cluster_id=pid,
            host="10.0.0.9",
            port=50060,
            last_seen=time.time(),
        )
    return coord


# ── FederationLoadBalancer.get_load ──────────────────────────────────────


class TestFederationLoadBalancerGetLoad:
    """get_load: query current load for one cluster."""

    def test_unknown_cluster_returns_none(self) -> None:
        flb = FederationLoadBalancer()
        assert flb.get_load("never-reported") is None

    def test_reported_cluster_returns_load(self) -> None:
        flb = FederationLoadBalancer()
        flb.report_load("c1", active_requests=5, pending_requests=2,
                        gpu_utilization=40.0, queue_depth=7)
        load = flb.get_load("c1")
        assert isinstance(load, RemoteClusterLoad)
        assert load.cluster_id == "c1"
        assert load.active_requests == 5
        assert load.pending_requests == 2
        assert load.gpu_utilization == 40.0
        assert load.queue_depth == 7

    def test_matches_get_remote_load(self) -> None:
        """get_load is the router-facing alias of get_remote_load."""
        flb = FederationLoadBalancer()
        flb.report_load("c1", active_requests=1, pending_requests=0,
                        gpu_utilization=25.0, queue_depth=3)
        assert flb.get_load("c1") is flb.get_remote_load("c1")

    def test_stale_data_still_returned_but_flagged(self) -> None:
        """Consistent with get_remote_load: stale entries are flagged, not dropped."""
        flb = FederationLoadBalancer(stale_threshold_s=0.01)
        flb.report_load("c1", active_requests=1, pending_requests=0,
                        gpu_utilization=25.0, queue_depth=3)
        time.sleep(0.05)
        load = flb.get_load("c1")
        assert load is not None
        assert load.stale is True

    def test_removed_cluster_returns_none(self) -> None:
        flb = FederationLoadBalancer()
        flb.report_load("c1", active_requests=1, pending_requests=0,
                        gpu_utilization=10.0, queue_depth=0)
        flb.remove_cluster("c1")
        assert flb.get_load("c1") is None


# ── Routing paths execute without AttributeError ─────────────────────────


class TestFederationRoutingGetLoadRegression:
    """The three call sites that used to crash: 808 / 843 / 908."""

    def test_get_peer_slo_status_reports_load_values(self) -> None:
        coord = _make_coord("p-known", "p-unknown")
        coord._load_balancer.report_load("p-known", active_requests=2,
                                         pending_requests=1,
                                         gpu_utilization=30.0, queue_depth=4)

        status = coord.get_peer_slo_status()  # used to raise AttributeError
        by_id = {s["cluster_id"]: s for s in status}

        assert set(by_id) == {"p-known", "p-unknown"}
        assert by_id["p-known"]["available_capacity"] == pytest.approx(70.0)
        assert by_id["p-known"]["is_overloaded"] is False
        # No heartbeat load ever received → zero capacity surfaced.
        assert by_id["p-unknown"]["available_capacity"] == 0.0
        assert by_id["p-unknown"]["is_overloaded"] is False

    def test_select_peer_for_sla_skips_unknown_and_overloaded(self) -> None:
        coord = _make_coord("p-idle", "p-busy", "p-silent")
        coord._load_balancer.report_load("p-idle", active_requests=0,
                                         pending_requests=0,
                                         gpu_utilization=20.0, queue_depth=1)
        coord._load_balancer.report_load("p-busy", active_requests=50,
                                         pending_requests=50,
                                         gpu_utilization=90.0, queue_depth=60)
        # p-silent never reported → get_load returns None → skipped.

        picked = coord.select_peer_for_sla()  # used to raise AttributeError

        assert picked is not None
        assert picked["cluster_id"] == "p-idle"
        assert picked["available_capacity"] == pytest.approx(80.0)

    def test_select_peer_for_sla_none_when_all_unusable(self) -> None:
        """Only an overloaded peer exists → no SLA-eligible candidate."""
        coord = _make_coord("p-busy")
        coord._load_balancer.report_load("p-busy", active_requests=99,
                                         pending_requests=99,
                                         gpu_utilization=95.0, queue_depth=99)

        assert coord.select_peer_for_sla() is None

    def test_select_peer_for_sla_prefers_most_capacity(self) -> None:
        coord = _make_coord("p-light", "p-mid")
        coord._load_balancer.report_load("p-light", active_requests=0,
                                         pending_requests=0,
                                         gpu_utilization=10.0, queue_depth=0)
        coord._load_balancer.report_load("p-mid", active_requests=5,
                                         pending_requests=2,
                                         gpu_utilization=60.0, queue_depth=3)

        picked = coord.select_peer_for_sla()

        assert picked is not None
        assert picked["cluster_id"] == "p-light"

    def test_get_best_peer_routes_by_capacity_without_digest(self) -> None:
        coord = _make_coord("p-light", "p-heavy")
        coord._load_balancer.report_load("p-light", active_requests=0,
                                         pending_requests=0,
                                         gpu_utilization=15.0, queue_depth=0)
        coord._load_balancer.report_load("p-heavy", active_requests=30,
                                         pending_requests=10,
                                         gpu_utilization=75.0, queue_depth=20)

        best = coord.get_best_peer()  # used to raise AttributeError

        assert best is not None
        assert best["cluster_id"] == "p-light"
        assert best["available_capacity"] == pytest.approx(85.0)

    def test_get_best_peer_excludes_overloaded_even_if_only_peer(self) -> None:
        """An overloaded-only fleet yields no candidate rather than routing there."""
        coord = _make_coord("p-busy")
        coord._load_balancer.report_load("p-busy", active_requests=99,
                                         pending_requests=99,
                                         gpu_utilization=99.0, queue_depth=99)

        assert coord.get_best_peer() is None

    def test_get_best_peer_unknown_load_is_worst_not_best(self) -> None:
        """A peer with no load data competes with 0.0 capacity, losing to measured headroom."""
        coord = _make_coord("p-silent", "p-measured")
        coord._load_balancer.report_load("p-measured", active_requests=1,
                                         pending_requests=0,
                                         gpu_utilization=35.0, queue_depth=2)

        best = coord.get_best_peer()

        assert best is not None
        assert best["cluster_id"] == "p-measured"
