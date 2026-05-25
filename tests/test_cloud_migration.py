"""Tests: MigrationPlanner (proactive, reactive, checkpoint, target selection, execute) and WorkloadMigrator."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from distllm.cloud.migration_planner import (
    MigrationPlanner, MigrationPlan, MigrationStep, MigrationStrategy,
)
from distllm.cloud.workload_migrator import WorkloadMigrator
from distllm.cloud.spot_provider import CloudProvider
from distllm.cloud.preemption_predictor import PreemptionPrediction


# ===========================================================================
# MigrationPlanner
# ===========================================================================


class TestMigrationPlannerProactive:
    def test_proactive_when_risk_above_threshold(self):
        planner = MigrationPlanner(preemption_threshold=0.7)
        planner.register_node("node-1", CloudProvider.AWS, "g5.xlarge", "us-east-1")
        planner.register_node("node-2", CloudProvider.AWS, "g5.xlarge", "us-east-1")
        pred = PreemptionPrediction(
            provider=CloudProvider.AWS, instance_type="g5.xlarge",
            region="us-east-1", risk_score=0.85, expected_lifetime_min=10, confidence=0.8,
        )
        plan = planner.plan_for_preemption("node-1", pred)
        assert plan is not None
        assert plan.strategy == MigrationStrategy.PROACTIVE


class TestMigrationPlannerReactive:
    def test_reactive_when_low_risk(self):
        planner = MigrationPlanner(preemption_threshold=0.7)
        planner.register_node("node-1", CloudProvider.AWS, "g5.xlarge", "us-east-1")
        planner.register_node("node-2", CloudProvider.AWS, "g5.xlarge", "us-east-1")
        pred = PreemptionPrediction(
            provider=CloudProvider.AWS, instance_type="g5.xlarge",
            region="us-east-1", risk_score=0.3, expected_lifetime_min=120, confidence=0.9,
        )
        plan = planner.plan_for_preemption("node-1", pred)
        assert plan is not None
        assert plan.strategy == MigrationStrategy.REACTIVE


class TestMigrationPlannerLive:
    def test_live_when_high_risk_and_small_kv_cache(self):
        planner = MigrationPlanner(preemption_threshold=0.7)
        planner.register_node("node-1", CloudProvider.AWS, "g5.xlarge", "us-east-1",
                              kv_cache_bytes=512 * 1024)  # < 1MB
        planner.register_node("node-2", CloudProvider.AWS, "g5.xlarge", "us-east-1")
        pred = PreemptionPrediction(
            provider=CloudProvider.AWS, instance_type="g5.xlarge",
            region="us-east-1", risk_score=0.96, expected_lifetime_min=1, confidence=0.9,
        )
        plan = planner.plan_for_preemption("node-1", pred)
        assert plan is not None
        assert plan.strategy == MigrationStrategy.LIVE

    def test_live_steps_include_sync_and_cutover(self):
        planner = MigrationPlanner(preemption_threshold=0.7)
        planner.register_node("node-1", CloudProvider.AWS, "g5.xlarge", "us-east-1",
                              kv_cache_bytes=256 * 1024, active_requests=3)
        planner.register_node("node-2", CloudProvider.AWS, "g5.xlarge", "us-east-1")
        pred = PreemptionPrediction(
            provider=CloudProvider.AWS, instance_type="g5.xlarge",
            region="us-east-1", risk_score=0.98, expected_lifetime_min=1, confidence=0.9,
        )
        plan = planner.plan_for_preemption("node-1", pred)
        assert plan is not None
        step_names = [s.name for s in plan.steps]
        assert "sync_kv_cache" in step_names
        assert "sync_requests" in step_names
        assert "cutover" in step_names


class TestMigrationPlannerCheckpoint:
    def test_checkpoint_when_no_target_node(self):
        planner = MigrationPlanner()
        planner.register_node("node-1", CloudProvider.AWS, "g5.xlarge", "us-east-1")
        planner.register_node("node-2", CloudProvider.AWS, "g5.xlarge", "us-east-1")
        planner._node_registry.pop("node-2")
        pred = PreemptionPrediction(
            provider=CloudProvider.AWS, instance_type="g5.xlarge",
            region="us-east-1", risk_score=0.85, expected_lifetime_min=10, confidence=0.8,
        )
        plan = planner.plan_for_preemption("node-1", pred)
        assert plan is not None
        assert plan.strategy == MigrationStrategy.CHECKPOINT

    def test_checkpoint_steps_include_dump_and_restore(self):
        planner = MigrationPlanner()
        planner.register_node("node-1", CloudProvider.AWS, "g5.xlarge", "us-east-1",
                              kv_cache_bytes=1024 * 1024)
        pred = PreemptionPrediction(
            provider=CloudProvider.AWS, instance_type="g5.xlarge",
            region="us-east-1", risk_score=0.85, expected_lifetime_min=10, confidence=0.8,
        )
        plan = planner.plan_for_preemption("node-1", pred)
        step_names = [s.name for s in plan.steps]
        assert "dump_kv_cache" in step_names
        assert "restore_kv_cache" in step_names


class TestMigrationPlannerTargetSelection:
    def test_same_provider_and_type_preferred(self):
        planner = MigrationPlanner()
        planner.register_node("node-1", CloudProvider.AWS, "g5.xlarge", "us-east-1")
        planner.register_node("node-2", CloudProvider.AWS, "g5.xlarge", "us-east-1")
        planner.register_node("node-3", CloudProvider.GCP, "n1-standard-4", "us-central1")
        target = planner._select_target(planner._node_registry["node-1"])
        assert target is not None
        assert target == "node-2"

    def test_falls_back_to_any_other_node(self):
        planner = MigrationPlanner()
        planner.register_node("node-1", CloudProvider.AWS, "g5.xlarge", "us-east-1")
        planner.register_node("node-2", CloudProvider.GCP, "n1-standard-4", "us-central1")
        target = planner._select_target(planner._node_registry["node-1"])
        assert target == "node-2"

    def test_no_candidates_returns_none(self):
        planner = MigrationPlanner()
        planner.register_node("node-1", CloudProvider.AWS, "g5.xlarge", "us-east-1")
        target = planner._select_target(planner._node_registry["node-1"])
        assert target is None


class TestMigrationPlannerExecute:
    @pytest.mark.asyncio
    async def test_execute_steps_in_order(self):
        async def dump_fn(nid):
            return {"key": "value"}
        async def restore_fn(nid, data):
            pass
        planner = MigrationPlanner(
            kv_cache_dump_fn=dump_fn,
            kv_cache_restore_fn=restore_fn,
        )
        planner.register_node("node-1", CloudProvider.AWS, "g5.xlarge", "us-east-1",
                              kv_cache_bytes=1024)
        planner.register_node("node-2", CloudProvider.AWS, "g5.xlarge", "us-east-1")

    @pytest.mark.asyncio
    async def test_execute_step_failure_returns_false(self):
        async def dump_fn(nid):
            raise RuntimeError("KV cache dump failed")
        planner = MigrationPlanner(kv_cache_dump_fn=dump_fn)
        planner.register_node("node-1", CloudProvider.AWS, "g5.xlarge", "us-east-1",
                              kv_cache_bytes=1024)
        planner.register_node("node-2", CloudProvider.AWS, "g5.xlarge", "us-east-1")

    @pytest.mark.asyncio
    async def test_execute_no_callbacks_succeeds(self):
        planner = MigrationPlanner()
        planner.register_node("node-1", CloudProvider.AWS, "g5.xlarge", "us-east-1",
                              kv_cache_bytes=1024)
        planner.register_node("node-2", CloudProvider.AWS, "g5.xlarge", "us-east-1")
        pred = PreemptionPrediction(
            provider=CloudProvider.AWS, instance_type="g5.xlarge",
            region="us-east-1", risk_score=0.85, expected_lifetime_min=10, confidence=0.8,
        )
        plan = planner.plan_for_preemption("node-1", pred)
        assert plan is not None
        result = await planner.execute_plan(plan)
        assert result is True
        completed = planner.get_completed_plans()
        assert len(completed) == 1
        assert completed[0].source_node_id == "node-1"

    @pytest.mark.asyncio
    async def test_execute_step_failure_returns_false(self):
        async def dump_fn(nid):
            raise RuntimeError("KV cache dump failed")
        planner = MigrationPlanner(kv_cache_dump_fn=dump_fn)
        planner.register_node("node-1", CloudProvider.AWS, "g5.xlarge", "us-east-1",
                              kv_cache_bytes=1024)
        pred = PreemptionPrediction(
            provider=CloudProvider.AWS, instance_type="g5.xlarge",
            region="us-east-1", risk_score=0.85, expected_lifetime_min=10, confidence=0.8,
        )
        plan = planner.plan_for_preemption("node-1", pred)
        assert plan is not None
        result = await planner.execute_plan(plan)
        assert result is False

    @pytest.mark.asyncio
    async def test_execute_no_callbacks_succeeds(self):
        planner = MigrationPlanner()
        planner.register_node("node-1", CloudProvider.AWS, "g5.xlarge", "us-east-1",
                              kv_cache_bytes=1024)
        pred = PreemptionPrediction(
            provider=CloudProvider.AWS, instance_type="g5.xlarge",
            region="us-east-1", risk_score=0.85, expected_lifetime_min=10, confidence=0.8,
        )
        plan = planner.plan_for_preemption("node-1", pred)
        assert plan is not None
        result = await planner.execute_plan(plan)
        assert result is True


class TestMigrationPlannerEdgeCases:
    def test_unregistered_node_returns_none(self):
        planner = MigrationPlanner()
        plan = planner.plan_for_preemption("nonexistent")
        assert plan is None

    def test_get_active_plans(self):
        planner = MigrationPlanner()
        planner.register_node("n1", CloudProvider.AWS, "g5.xlarge", "us-east-1")
        planner.register_node("n2", CloudProvider.AWS, "g5.xlarge", "us-east-1")
        pred = PreemptionPrediction(
            provider=CloudProvider.AWS, instance_type="g5.xlarge",
            region="us-east-1", risk_score=0.85, expected_lifetime_min=10, confidence=0.8,
        )
        planner.plan_for_preemption("n1", pred)
        plans = planner.get_active_plans()
        assert len(plans) >= 1

    def test_plan_dataclass_computes_total_duration(self):
        steps = [
            MigrationStep("step1", "desc1", 1.0),
            MigrationStep("step2", "desc2", 2.5),
        ]
        plan = MigrationPlan(
            source_node_id="src", target_node_id="dst",
            strategy=MigrationStrategy.PROACTIVE, steps=steps,
        )
        assert plan.total_estimated_duration_s == 3.5
        assert plan.timestamp > 0


class TestMigrationPlannerNoNodes:
    def test_no_candidates_results_in_checkpoint(self):
        planner = MigrationPlanner()
        planner.register_node("node-1", CloudProvider.AWS, "g5.xlarge", "us-east-1")
        planner.register_node("node-2", CloudProvider.AWS, "g5.xlarge", "us-east-1")
        planner._node_registry.pop("node-2")
        pred = PreemptionPrediction(
            provider=CloudProvider.AWS, instance_type="g5.xlarge",
            region="us-east-1", risk_score=0.85, expected_lifetime_min=10, confidence=0.8,
        )
        plan = planner.plan_for_preemption("node-1", pred)
        assert plan is not None
        assert plan.strategy == MigrationStrategy.CHECKPOINT
        assert "replacement" in plan.target_node_id


# ===========================================================================
# WorkloadMigrator
# ===========================================================================


class TestWorkloadMigratorWithKVState:
    def test_migrate_with_kv_state_succeeds(self):
        migrator = WorkloadMigrator(migration_timeout_s=60.0)
        result = migrator.migrate_node_workload(
            "node-1", "node-2", kv_state={"seq_1": "data"}
        )
        assert result is True

    def test_migrate_with_kv_state_records_status(self):
        migrator = WorkloadMigrator()
        mid = "node-1->node-2"
        migrator.migrate_node_workload("node-1", "node-2", kv_state={"k": "v"})
        status = migrator.get_migration_status(mid)
        assert status is not None
        assert status["status"] == "completed"
        assert status["elapsed_s"] >= 0


class TestWorkloadMigratorWithoutKVState:
    def test_migrate_without_kv_state_completes_no_state(self):
        migrator = WorkloadMigrator()
        result = migrator.migrate_node_workload(
            "node-1", "node-2", kv_state=None
        )
        assert result is True

    def test_no_kv_state_records_completed_no_state_status(self):
        migrator = WorkloadMigrator()
        mid = "node-1->node-2"
        migrator.migrate_node_workload("node-1", "node-2")
        status = migrator.get_migration_status(mid)
        assert status is not None
        assert status["status"] == "completed_no_state"


class TestWorkloadMigratorTimeout:
    def test_timeout_returns_false(self):
        migrator = WorkloadMigrator(migration_timeout_s=0.0)
        import time
        result = migrator.migrate_node_workload(
            "node-1", "node-2", kv_state={"k": "v"}
        )
        assert result is False

    def test_timeout_without_kv_state(self):
        migrator = WorkloadMigrator(migration_timeout_s=0.0)
        result = migrator.migrate_node_workload("node-1", "node-2")
        assert result is True  # no-kv-state path returns True before timeout check


class TestWorkloadMigratorTracking:
    def test_get_migration_status_nonexistent(self):
        migrator = WorkloadMigrator()
        status = migrator.get_migration_status("nonexistent")
        assert status is None

    def test_list_active_migrations(self):
        migrator = WorkloadMigrator()
        active = migrator.list_active_migrations()
        assert isinstance(active, list)

    def test_migration_id_format(self):
        migrator = WorkloadMigrator()
        mid = "node-1->node-2"
        migrator.migrate_node_workload("node-1", "node-2", kv_state={})
        status = migrator.get_migration_status(mid)
        assert status is not None
        assert status["src"] == "node-1"
        assert status["dst"] == "node-2"

    def test_error_records_failed_status(self):
        migrator = WorkloadMigrator()
        mid = "failing->dst"
        with patch.object(migrator, "_active_migrations", side_effect=RuntimeError("fail")):
            pass
        result = migrator.migrate_node_workload(
            "failing", "dst", kv_state={"k": "v"}
        )
        assert result is True  # without exception
