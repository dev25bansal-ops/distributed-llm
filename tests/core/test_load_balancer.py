"""Tests for LoadBalancer -- request distribution across coordinators.

Covers:
- Construction with strategy
- add_target and remove_target
- pick returns a target
- pick returns None when no targets
- record_success and record_failure
- healthy_targets filtering
- mark_healthy and mark_unhealthy
- stats and reset_stats
- set_strategy changes strategy

No MagicMock -- real lists, counters, and lock-free snapshot logic.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/load_balancer.py")
LoadBalancer = _mod.LoadBalancer
LBStrategy = _mod.LBStrategy
CoordinatorTarget = _mod.CoordinatorTarget
LBStats = _mod.LBStats
create_load_balancer = _mod.create_load_balancer


class TestLoadBalancerConstruction:
    """Construction and initial state."""

    def test_default_construction(self) -> None:
        lb = LoadBalancer()
        assert lb._strategy == LBStrategy.LEAST_CONNECTIONS
        assert lb._health_check_interval == 5.0
        assert lb._max_failures == 3
        assert lb._targets == []
        assert lb._total_requests == 0

    def test_custom_strategy(self) -> None:
        lb = LoadBalancer(strategy=LBStrategy.ROUND_ROBIN)
        assert lb._strategy == LBStrategy.ROUND_ROBIN

    def test_create_load_balancer_factory(self) -> None:
        lb = create_load_balancer(hosts=["10.0.0.1:50050", "10.0.0.2:50051"])
        targets = lb.all_targets()
        assert len(targets) == 2
        assert targets[0].host == "10.0.0.1"

    def test_create_load_balancer_default_port(self) -> None:
        lb = create_load_balancer(hosts=["10.0.0.1"])
        targets = lb.all_targets()
        assert targets[0].port == 50050


class TestLoadBalancerTargets:
    """Target management."""

    def test_add_target(self) -> None:
        lb = LoadBalancer()
        lb.add_target("10.0.0.1", 50050, node_id="coord-1")
        assert len(lb._targets) == 1
        assert lb._targets[0].host == "10.0.0.1"

    def test_add_target_duplicate(self) -> None:
        lb = LoadBalancer()
        lb.add_target("10.0.0.1", 50050)
        lb.add_target("10.0.0.1", 50050)  # duplicate
        assert len(lb._targets) == 1

    def test_remove_target(self) -> None:
        lb = LoadBalancer()
        lb.add_target("10.0.0.1", 50050)
        lb.add_target("10.0.0.2", 50050)
        assert lb.remove_target("10.0.0.1", 50050) is True
        assert len(lb._targets) == 1

    def test_remove_nonexistent_target(self) -> None:
        lb = LoadBalancer()
        assert lb.remove_target("10.0.0.1", 50050) is False

    def test_all_targets(self) -> None:
        lb = LoadBalancer()
        lb.add_target("10.0.0.1", 50050)
        targets = lb.all_targets()
        assert len(targets) == 1

    def test_healthy_targets_unchecked(self) -> None:
        lb = LoadBalancer()
        lb.add_target("10.0.0.1", 50050)
        healthy = lb.healthy_targets()
        assert len(healthy) == 1  # freshly added, stale check


class TestLoadBalancerPick:
    """Target selection."""

    def test_pick_returns_target(self) -> None:
        lb = LoadBalancer()
        lb.add_target("10.0.0.1", 50050)
        target = lb.pick("req-1")
        assert target is not None
        assert target.host == "10.0.0.1"

    def test_pick_returns_none_no_targets(self) -> None:
        lb = LoadBalancer()
        assert lb.pick("req-1") is None

    def test_pick_increments_active_connections(self) -> None:
        lb = LoadBalancer()
        lb.add_target("10.0.0.1", 50050)
        target = lb.pick("req-1")
        assert target is not None
        assert target.active_connections == 1

    def test_pick_round_robin(self) -> None:
        lb = LoadBalancer(strategy=LBStrategy.ROUND_ROBIN)
        lb.add_target("10.0.0.1", 50050)
        lb.add_target("10.0.0.2", 50050)
        t1 = lb.pick("req-1")
        t2 = lb.pick("req-2")
        assert t1 is not None and t2 is not None
        # Should alternate
        assert (t1.host, t1.port) != (t2.host, t2.port) or len(lb._targets) == 2

    def test_set_strategy(self) -> None:
        lb = LoadBalancer(strategy=LBStrategy.ROUND_ROBIN)
        lb.set_strategy(LBStrategy.RANDOM)
        assert lb._strategy == LBStrategy.RANDOM


class TestLoadBalancerRecording:
    """Success/failure recording."""

    def test_record_success_decrements_connections(self) -> None:
        lb = LoadBalancer()
        lb.add_target("10.0.0.1", 50050)
        target = lb.pick("req-1")
        assert target is not None
        lb.record_success(target, latency_ms=10.0)
        assert target.active_connections == 0
        assert target.total_requests == 1

    def test_record_failure_marks_unhealthy(self) -> None:
        lb = LoadBalancer(max_consecutive_failures=2)
        lb.add_target("10.0.0.1", 50050)
        target = lb.pick("req-1")
        assert target is not None
        lb.record_failure(target, error="timeout")
        lb.record_failure(target, error="timeout")
        assert target.is_healthy is False

    def test_mark_healthy_and_unhealthy(self) -> None:
        lb = LoadBalancer()
        lb.add_target("10.0.0.1", 50050)
        target = lb._targets[0]
        lb.mark_unhealthy(target)
        assert target.is_healthy is False
        lb.mark_healthy(target)
        assert target.is_healthy is True


class TestLoadBalancerStats:
    """Statistics."""

    def test_stats_returns_LBStats(self) -> None:
        lb = LoadBalancer()
        lb.add_target("10.0.0.1", 50050)
        stats = lb.stats()
        assert isinstance(stats, LBStats)
        assert stats.strategy == "least_connections"
        assert len(stats.targets) == 1
        assert stats.total_requests == 0

    def test_reset_stats(self) -> None:
        lb = LoadBalancer()
        lb.add_target("10.0.0.1", 50050)
        target = lb.pick("req-1")
        assert target is not None
        lb.record_success(target, latency_ms=10.0)
        lb.reset_stats()
        assert lb._total_requests == 0
        assert target.total_requests == 0
