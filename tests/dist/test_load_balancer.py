"""Tests for federation load balancer for cross-cluster load reporting.

Covers:
- RemoteClusterLoad dataclass: defaults, properties, staleness
- FederationLoadBalancer: report, query, selection, removal, serialization
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time as time_module
import types

import pytest


def _get_module():
    """Load the p2p load_balancer module directly from source."""
    path = os.path.join("src", "distllm", "dist", "p2p", "load_balancer.py")
    spec = importlib.util.spec_from_file_location(
        "distllm.dist.p2p.load_balancer",
        path,
    )
    mod = importlib.util.module_from_spec(spec)

    # Stub logger to keep test output clean
    mod.logger = types.ModuleType("logger")
    for level in ("debug", "info", "warning", "error", "exception"):
        setattr(mod.logger, level, lambda *a, **kw: None)

    sys.modules["distllm.dist.p2p.load_balancer"] = mod
    spec.loader.exec_module(mod)
    return mod


MOD = _get_module()
RemoteClusterLoad = MOD.RemoteClusterLoad
FederationLoadBalancer = MOD.FederationLoadBalancer


# ── RemoteClusterLoad ─────────────────────────────────────────────────────


class TestRemoteClusterLoad:
    """RemoteClusterLoad dataclass: defaults, properties, staleness."""

    def test_default_values(self) -> None:
        load = RemoteClusterLoad(cluster_id="cluster-a")
        assert load.cluster_id == "cluster-a"
        assert load.active_requests == 0
        assert load.pending_requests == 0
        assert load.gpu_utilization == 0.0
        assert load.queue_depth == 0
        assert load.tokens_per_sec == 0.0
        assert load.last_report == 0.0
        assert load.stale is False

    # -- is_overloaded --

    def test_is_overloaded_high_gpu(self) -> None:
        load = RemoteClusterLoad(cluster_id="c", gpu_utilization=90.0)
        assert load.is_overloaded is True

    def test_is_overloaded_high_queue(self) -> None:
        load = RemoteClusterLoad(cluster_id="c", queue_depth=60)
        assert load.is_overloaded is True

    def test_is_overloaded_boundary_gpu(self) -> None:
        load = RemoteClusterLoad(cluster_id="c", gpu_utilization=85.0)
        assert load.is_overloaded is False

    def test_is_overloaded_boundary_queue(self) -> None:
        load = RemoteClusterLoad(cluster_id="c", queue_depth=50)
        assert load.is_overloaded is False

    def test_is_overloaded_both_low(self) -> None:
        load = RemoteClusterLoad(cluster_id="c", gpu_utilization=30.0, queue_depth=5)
        assert load.is_overloaded is False

    # -- available_capacity --

    def test_available_capacity_overloaded_is_zero(self) -> None:
        load = RemoteClusterLoad(cluster_id="c", gpu_utilization=90.0)
        assert load.available_capacity == 0.0

    def test_available_capacity_normal(self) -> None:
        load = RemoteClusterLoad(cluster_id="c", gpu_utilization=60.0)
        assert load.available_capacity == 40.0

    def test_available_capacity_clamps_at_zero(self) -> None:
        load = RemoteClusterLoad(cluster_id="c", gpu_utilization=110.0)
        assert load.available_capacity == 0.0

    # -- is_stale --

    def test_is_stale_old_report(self) -> None:
        load = RemoteClusterLoad(cluster_id="c", last_report=0.0)
        assert load.is_stale(threshold_s=30.0) is True

    def test_is_stale_recent_report(self) -> None:
        now = time_module.time()
        load = RemoteClusterLoad(cluster_id="c", last_report=now)
        assert load.is_stale(threshold_s=30.0) is False


# ── FederationLoadBalancer ────────────────────────────────────────────────


class TestFederationLoadBalancerInit:
    """FederationLoadBalancer construction."""

    def test_defaults(self) -> None:
        flb = FederationLoadBalancer()
        assert flb.stale_threshold_s == 30.0
        assert flb.ema_alpha == 0.3
        assert flb._loads == {}

    def test_custom_params(self) -> None:
        flb = FederationLoadBalancer(stale_threshold_s=10.0, ema_alpha=0.5)
        assert flb.stale_threshold_s == 10.0
        assert flb.ema_alpha == 0.5


class TestFederationLoadBalancerReportLoad:
    """report_load -- new cluster creation and EMA updates."""

    def test_new_cluster(self) -> None:
        flb = FederationLoadBalancer()
        flb.report_load(
            "c1",
            active_requests=5,
            pending_requests=2,
            gpu_utilization=70.0,
            queue_depth=10,
        )
        load = flb._loads["c1"]
        assert load.active_requests == 5
        assert load.pending_requests == 2
        assert load.gpu_utilization == 70.0
        assert load.queue_depth == 10
        assert load.tokens_per_sec == 0.0
        assert load.stale is False

    def test_ema_update(self) -> None:
        flb = FederationLoadBalancer(ema_alpha=0.4)
        flb.report_load(
            "c1",
            active_requests=10,
            pending_requests=5,
            gpu_utilization=50.0,
            queue_depth=20,
            tokens_per_sec=100.0,
        )
        flb.report_load(
            "c1",
            active_requests=20,
            pending_requests=10,
            gpu_utilization=80.0,
            queue_depth=30,
            tokens_per_sec=200.0,
        )
        load = flb._loads["c1"]
        # active: 0.4*20 + 0.6*10 = 8 + 6 = 14
        assert load.active_requests == 14
        # pending: 0.4*10 + 0.6*5 = 4 + 3 = 7
        assert load.pending_requests == 7
        # gpu_util: 0.4*80 + 0.6*50 = 32 + 30 = 62
        assert load.gpu_utilization == 62.0
        # queue: 0.4*30 + 0.6*20 = 12 + 12 = 24
        assert load.queue_depth == 24
        # tokens: 0.4*200 + 0.6*100 = 80 + 60 = 140
        assert load.tokens_per_sec == 140.0


class TestFederationLoadBalancerGetRemoteLoad:
    """get_remote_load -- lookup and staleness marking."""

    def test_unknown_cluster(self) -> None:
        flb = FederationLoadBalancer()
        assert flb.get_remote_load("unknown") is None

    def test_known_cluster(self) -> None:
        flb = FederationLoadBalancer()
        flb.report_load(
            "c1",
            active_requests=5,
            pending_requests=2,
            gpu_utilization=70.0,
            queue_depth=10,
        )
        load = flb.get_remote_load("c1")
        assert load is not None
        assert load.cluster_id == "c1"
        assert load.active_requests == 5

    def test_stale_report(self) -> None:
        flb = FederationLoadBalancer(stale_threshold_s=0.01)
        flb.report_load(
            "c1",
            active_requests=5,
            pending_requests=2,
            gpu_utilization=70.0,
            queue_depth=10,
        )
        # Force old timestamp to guarantee staleness
        flb._loads["c1"].last_report = 0.0
        load = flb.get_remote_load("c1")
        assert load is not None
        assert load.stale is True


class TestFederationLoadBalancerGetAllLoads:
    """get_all_loads -- returns all clusters."""

    def test_empty(self) -> None:
        flb = FederationLoadBalancer()
        assert flb.get_all_loads() == {}

    def test_multiple_clusters(self) -> None:
        flb = FederationLoadBalancer()
        flb.report_load(
            "c1",
            active_requests=1,
            pending_requests=0,
            gpu_utilization=10.0,
            queue_depth=5,
        )
        flb.report_load(
            "c2",
            active_requests=2,
            pending_requests=1,
            gpu_utilization=20.0,
            queue_depth=10,
        )
        loads = flb.get_all_loads()
        assert set(loads.keys()) == {"c1", "c2"}
        assert loads["c1"].active_requests == 1
        assert loads["c2"].active_requests == 2


class TestFederationLoadBalancerGetBestCluster:
    """get_best_cluster -- selection logic with edge cases."""

    def test_empty_list_returns_none(self) -> None:
        flb = FederationLoadBalancer()
        assert flb.get_best_cluster([]) is None

    def test_unknown_clusters_returns_first(self) -> None:
        flb = FederationLoadBalancer()
        # All unknown -> score 0.0 -> first in list wins
        assert flb.get_best_cluster(["a", "b"]) == "a"

    def test_single_healthy(self) -> None:
        flb = FederationLoadBalancer()
        flb.report_load(
            "c1",
            active_requests=5,
            pending_requests=2,
            gpu_utilization=40.0,
            queue_depth=10,
        )
        assert flb.get_best_cluster(["c1"]) == "c1"

    def test_chooses_lowest_score(self) -> None:
        flb = FederationLoadBalancer()
        flb.report_load(
            "high",
            active_requests=10,
            pending_requests=5,
            gpu_utilization=80.0,
            queue_depth=30,
        )
        flb.report_load(
            "low",
            active_requests=2,
            pending_requests=1,
            gpu_utilization=20.0,
            queue_depth=5,
        )
        # low score = 20+5 = 25, high score = 80+30 = 110
        assert flb.get_best_cluster(["high", "low"]) == "low"

    def test_unknown_beats_healthy(self) -> None:
        """Unknown clusters get score 0 which is always lowest."""
        flb = FederationLoadBalancer()
        flb.report_load(
            "busy",
            active_requests=10,
            pending_requests=5,
            gpu_utilization=50.0,
            queue_depth=20,
        )
        # Unknown cluster "fresh" gets score 0, beating busy's 50+20=70
        assert flb.get_best_cluster(["busy", "fresh"]) == "fresh"

    def test_all_stale_returns_none(self) -> None:
        flb = FederationLoadBalancer(stale_threshold_s=0.01)
        flb.report_load(
            "c1",
            active_requests=5,
            pending_requests=2,
            gpu_utilization=40.0,
            queue_depth=10,
        )
        flb._loads["c1"].last_report = 0.0
        assert flb.get_best_cluster(["c1"]) is None

    def test_all_overloaded_returns_none(self) -> None:
        flb = FederationLoadBalancer()
        flb.report_load(
            "c1",
            active_requests=5,
            pending_requests=2,
            gpu_utilization=90.0,
            queue_depth=60,
        )
        assert flb.get_best_cluster(["c1"]) is None

    def test_skips_stale_and_overloaded_picks_healthy(self) -> None:
        flb = FederationLoadBalancer(stale_threshold_s=0.01)
        flb.report_load(
            "stale",
            active_requests=5,
            pending_requests=2,
            gpu_utilization=10.0,
            queue_depth=5,
        )
        flb._loads["stale"].last_report = 0.0
        flb.report_load(
            "overloaded",
            active_requests=5,
            pending_requests=2,
            gpu_utilization=95.0,
            queue_depth=60,
        )
        flb.report_load(
            "good",
            active_requests=2,
            pending_requests=1,
            gpu_utilization=30.0,
            queue_depth=10,
        )
        # Only "good" is eligible (stale and overloaded are skipped)
        assert flb.get_best_cluster(["stale", "overloaded", "good"]) == "good"


class TestFederationLoadBalancerRemoveCluster:
    """remove_cluster -- removal and absence after removal."""

    def test_remove_existing(self) -> None:
        flb = FederationLoadBalancer()
        flb.report_load(
            "c1",
            active_requests=5,
            pending_requests=2,
            gpu_utilization=70.0,
            queue_depth=10,
        )
        flb.remove_cluster("c1")
        assert "c1" not in flb._loads

    def test_remove_unknown_does_not_raise(self) -> None:
        flb = FederationLoadBalancer()
        # Should not raise
        flb.remove_cluster("nonexistent")

    def test_get_after_remove(self) -> None:
        flb = FederationLoadBalancer()
        flb.report_load(
            "c1",
            active_requests=5,
            pending_requests=2,
            gpu_utilization=70.0,
            queue_depth=10,
        )
        flb.remove_cluster("c1")
        assert flb.get_remote_load("c1") is None


class TestFederationLoadBalancerToDict:
    """to_dict -- serialization."""

    def test_empty(self) -> None:
        flb = FederationLoadBalancer()
        assert flb.to_dict() == {}

    def test_populated(self) -> None:
        flb = FederationLoadBalancer()
        flb.report_load(
            "c1",
            active_requests=5,
            pending_requests=2,
            gpu_utilization=70.0,
            queue_depth=10,
            tokens_per_sec=50.0,
        )
        flb.report_load(
            "c2",
            active_requests=3,
            pending_requests=1,
            gpu_utilization=30.0,
            queue_depth=8,
        )
        d = flb.to_dict()
        assert set(d.keys()) == {"c1", "c2"}
        assert d["c1"]["active"] == 5
        assert d["c1"]["pending"] == 2
        assert d["c1"]["gpu_util"] == 70.0
        assert d["c1"]["queue_depth"] == 10
        assert d["c1"]["tokens_per_sec"] == 50.0
        assert d["c1"]["stale"] is False
        assert d["c2"]["active"] == 3
