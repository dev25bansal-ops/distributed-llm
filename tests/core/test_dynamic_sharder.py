"""Tests for DynamicSharder.

Covers: initial partition, node join/leave, migration plan
generation, active migrations, stats, edge cases.
"""

from __future__ import annotations

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/dynamic_sharder.py")
DynamicSharder = _mod.DynamicSharder
LayerMigration = _mod.LayerMigration
MigrationState = _mod.MigrationState
ReshardPlan = _mod.ReshardPlan


class TestDynamicSharderInit:
    def test_default_values(self):
        sharder = DynamicSharder()
        assert sharder._current_partition == {}
        assert sharder._active_migrations == []
        s = sharder.stats()
        assert s["reshards"] == 0
        assert s["migrations_completed"] == 0
        assert s["migrations_failed"] == 0

    def test_initial_partition(self):
        sharder = DynamicSharder()
        sharder.set_initial_partition({"node-a": [0, 1, 2], "node-b": [3, 4, 5]})
        part = sharder.get_current_partition()
        assert part["node-a"] == [0, 1, 2]
        assert part["node-b"] == [3, 4, 5]


class TestOnNodeJoin:
    def test_join_creates_reshard_plan(self):
        sharder = DynamicSharder()
        sharder.set_initial_partition({"node-a": [0, 1, 2, 3]})
        plan = sharder.on_node_join("node-b")
        assert plan is not None
        assert len(plan.migrations) >= 1  # some layers move to node-b
        assert "node-b" in plan.new_partition

    def test_join_updates_partition(self):
        # F-048: partition install now requires successful transfers, so
        # provide a real (succeeding) transfer callback.
        sharder = DynamicSharder(on_layer_transfer=lambda layer, src, dst: True)
        sharder.set_initial_partition({"node-a": [0, 1, 2, 3]})
        sharder.on_node_join("node-b")
        part = sharder.get_current_partition()
        assert "node-b" in part
        # Total layers preserved: 4 layers across 2 nodes
        total = sum(len(v) for v in part.values())
        assert total == 4

    def test_join_without_initial_partition_creates_default(self):
        sharder = DynamicSharder()
        plan = sharder.on_node_join("node-a")
        # When no partition exists, defaults to 32 layers and creates a plan
        assert plan is not None
        assert "node-a" in plan.new_partition


class TestOnNodeLeave:
    def test_leave_redistributes_layers(self):
        # F-048: partition install now requires successful transfers, so
        # provide a real (succeeding) transfer callback.
        sharder = DynamicSharder(on_layer_transfer=lambda layer, src, dst: True)
        sharder.set_initial_partition({"node-a": [0, 1], "node-b": [2, 3]})
        plan = sharder.on_node_leave("node-a")
        assert plan is not None
        part = sharder.get_current_partition()
        assert "node-a" not in part
        # All layers should be on node-b
        total = sum(len(v) for v in part.values())
        assert total == 4

    def test_leave_unknown_node_returns_none(self):
        sharder = DynamicSharder()
        sharder.set_initial_partition({"node-a": [0, 1]})
        plan = sharder.on_node_leave("node-nonexistent")
        assert plan is None

    def test_leave_last_node_returns_empty_partition(self):
        sharder = DynamicSharder()
        sharder.set_initial_partition({"node-a": [0, 1]})
        plan = sharder.on_node_leave("node-a")
        assert plan is not None
        part = sharder.get_current_partition()
        assert part == {}


class TestMigrationPlan:
    def test_generate_migration_plan(self):
        sharder = DynamicSharder()
        old = {"node-a": [0, 1, 2], "node-b": [3, 4]}
        new = {"node-a": [0, 1], "node-b": [2, 3, 4]}
        plan = sharder._generate_migration_plan(old, new)
        assert isinstance(plan, ReshardPlan)
        assert len(plan.migrations) == 1  # layer 2 moves from a to b
        assert plan.migrations[0].layer_id == 2
        assert plan.migrations[0].source_node == "node-a"
        assert plan.migrations[0].target_node == "node-b"

    def test_empty_plan_when_no_change(self):
        sharder = DynamicSharder()
        old = {"node-a": [0, 1]}
        plan = sharder._generate_migration_plan(old, old)
        assert len(plan.migrations) == 0

    def test_estimated_downtime_positive(self):
        sharder = DynamicSharder()
        old = {"node-a": [0, 1]}
        new = {"node-a": [], "node-b": [0, 1]}
        plan = sharder._generate_migration_plan(old, new)
        assert plan.estimated_downtime_ms > 0
        assert plan.total_bytes > 0


class TestActiveMigrations:
    def test_get_active_migrations(self, monkeypatch):
        sharder = DynamicSharder()
        sharder.set_initial_partition({"node-a": [0, 1, 2, 3]})

        # Subvert _migrate_layer to avoid sleep/transfer call
        monkeypatch.setattr(sharder, "_migrate_layer", lambda m: None)

        plan = sharder.on_node_join("node-b")
        assert plan is not None
        # After execution, active migrations should be empty
        assert sharder.get_active_migrations() == []


class TestStats:
    def test_stats_after_reshard(self):
        # F-048: reshard counts as complete only when transfers succeed.
        sharder = DynamicSharder(on_layer_transfer=lambda layer, src, dst: True)
        sharder.set_initial_partition({"node-a": [0, 1, 2, 3]})
        plan = sharder.on_node_join("node-b")
        s = sharder.stats()
        assert s["reshards"] == (1 if plan else 0)
        assert s["current_nodes"] > 0
        assert s["active_migrations"] == 0
