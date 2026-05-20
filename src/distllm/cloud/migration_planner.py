"""Graceful KV cache migration planner for spot preemption.

Plans and executes workload migration when a spot instance preemption
is predicted or detected. Works with the PreemptionPredictor to
proactively migrate before termination.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

from loguru import logger

from distllm.cloud.spot_provider import CloudProvider
from distllm.cloud.preemption_predictor import PreemptionPrediction


class MigrationStrategy(str, Enum):
    """Strategy for migrating workloads between nodes."""
    PROACTIVE = "proactive"     # Migrate before preemption (predicted)
    REACTIVE = "reactive"       # Migrate after preemption detected
    CHECKPOINT = "checkpoint"   # Save state, restore later (no live migration)
    LIVE = "live"               # Live migration with minimal downtime


@dataclass
class MigrationStep:
    """A single step in a migration plan."""
    name: str
    description: str
    estimated_duration_s: float
    kv_size_bytes: int = 0


@dataclass
class MigrationPlan:
    """A complete migration plan for a preempted or soon-to-be-preempted node."""
    source_node_id: str
    target_node_id: str
    strategy: MigrationStrategy
    steps: list[MigrationStep] = field(default_factory=list)
    total_estimated_duration_s: float = 0.0
    kv_cache_size_bytes: int = 0
    active_requests: int = 0
    risk_score: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        self.total_estimated_duration_s = sum(
            s.estimated_duration_s for s in self.steps
        )


class MigrationPlanner:
    """Plans KV cache migration on spot instance preemption.

    Integrates with:
    - PreemptionPredictor for proactive migration triggers
    - WorkloadMigrator for state checkpoint/restore
    - AutoProvisioner for replacement instance provisioning

    Args:
        preemption_threshold: Risk score above which proactive migration triggers.
        max_downtime_s: Maximum acceptable downtime for live migration.
        kv_cache_dump_fn: Async callable to dump KV cache for a node.
        kv_cache_restore_fn: Async callable to restore KV cache on a node.
        provision_replacement_fn: Async callable to provision a replacement.
    """

    def __init__(
        self,
        preemption_threshold: float = 0.7,
        max_downtime_s: float = 5.0,
        kv_cache_dump_fn: Callable[..., Awaitable[Any]] | None = None,
        kv_cache_restore_fn: Callable[..., Awaitable[Any]] | None = None,
        provision_replacement_fn: Callable[..., Awaitable[str]] | None = None,
    ):
        self.preemption_threshold = preemption_threshold
        self.max_downtime_s = max_downtime_s
        self._kv_cache_dump = kv_cache_dump_fn
        self._kv_cache_restore = kv_cache_restore_fn
        self._provision_replacement = provision_replacement_fn

        self._active_plans: dict[str, MigrationPlan] = {}
        self._completed: list[MigrationPlan] = []
        self._node_registry: dict[str, dict[str, Any]] = {}

    def register_node(
        self,
        node_id: str,
        provider: CloudProvider,
        instance_type: str,
        region: str,
        kv_cache_bytes: int = 0,
        active_requests: int = 0,
    ) -> None:
        """Register a node for migration planning."""
        self._node_registry[node_id] = {
            "node_id": node_id,
            "provider": provider,
            "instance_type": instance_type,
            "region": region,
            "kv_cache_bytes": kv_cache_bytes,
            "active_requests": active_requests,
            "registered_at": time.time(),
        }

    def plan_for_preemption(
        self,
        node_id: str,
        prediction: PreemptionPrediction | None = None,
    ) -> MigrationPlan | None:
        """Create a migration plan for a node at risk of preemption.

        Args:
            node_id: The node at risk.
            prediction: PreemptionPrediction from PreemptionPredictor.
                If None, uses a reactive strategy.

        Returns:
            A MigrationPlan, or None if the node isn't registered.
        """
        node = self._node_registry.get(node_id)
        if node is None:
            logger.warning(f"Cannot plan migration: node {node_id} not registered")
            return None

        risk = prediction.risk_score if prediction else 1.0
        strategy = (
            MigrationStrategy.PROACTIVE
            if risk >= self.preemption_threshold
            else MigrationStrategy.REACTIVE
        )

        if risk >= 0.95 and node.get("kv_cache_bytes", 0) < 1024 * 1024:
            strategy = MigrationStrategy.LIVE

        target_node_id = self._select_target(node)
        if target_node_id is None:
            logger.warning(f"No suitable target for migrating {node_id}")
            strategy = MigrationStrategy.CHECKPOINT
            target_node_id = f"{node_id}-replacement"

        steps = self._build_steps(strategy, node, target_node_id)

        plan = MigrationPlan(
            source_node_id=node_id,
            target_node_id=target_node_id,
            strategy=strategy,
            steps=steps,
            kv_cache_size_bytes=node.get("kv_cache_bytes", 0),
            active_requests=node.get("active_requests", 0),
            risk_score=risk,
        )

        self._active_plans[node_id] = plan
        logger.info(
            f"Migration plan for {node_id}: {strategy.value} -> {target_node_id} "
            f"({plan.total_estimated_duration_s:.1f}s expected)"
        )
        return plan

    def _select_target(self, node: dict[str, Any]) -> str | None:
        candidates = [
            nid for nid, info in self._node_registry.items()
            if nid != node["node_id"]
            and info["provider"] == node["provider"]
            and info["instance_type"] == node["instance_type"]
        ]
        if not candidates:
            candidates = [
                nid for nid in self._node_registry
                if nid != node["node_id"]
            ]
        return candidates[0] if candidates else None

    def _build_steps(
        self,
        strategy: MigrationStrategy,
        node: dict[str, Any],
        target_node_id: str,
    ) -> list[MigrationStep]:
        steps: list[MigrationStep] = []
        kv_bytes = node.get("kv_cache_bytes", 0)
        active = node.get("active_requests", 0)

        if strategy == MigrationStrategy.LIVE:
            steps.append(MigrationStep(
                name="sync_kv_cache",
                description="Incremental KV cache sync to target",
                estimated_duration_s=0.5 + kv_bytes / (1024 * 1024 * 1024) * 0.1,
                kv_size_bytes=kv_bytes,
            ))
            steps.append(MigrationStep(
                name="sync_requests",
                description="Mirror active requests to target",
                estimated_duration_s=0.1 * active,
                kv_size_bytes=0,
            ))
            steps.append(MigrationStep(
                name="cutover",
                description="Atomic traffic switch to target node",
                estimated_duration_s=0.5,
                kv_size_bytes=0,
            ))

        elif strategy == MigrationStrategy.PROACTIVE:
            steps.append(MigrationStep(
                name="freeze_kv_cache",
                description="Pause KV cache updates",
                estimated_duration_s=0.1,
                kv_size_bytes=0,
            ))
            steps.append(MigrationStep(
                name="dump_kv_cache",
                description="Serialize KV cache to shared storage",
                estimated_duration_s=kv_bytes / (1024 * 1024 * 1024) * 0.5,
                kv_size_bytes=kv_bytes,
            ))

        else:
            steps.append(MigrationStep(
                name="dump_kv_cache",
                description="Full KV cache dump",
                estimated_duration_s=kv_bytes / (1024 * 1024 * 1024) * 1.0,
                kv_size_bytes=kv_bytes,
            ))

        return steps

    async def execute_plan(self, plan: MigrationPlan) -> bool:
        """Execute a migration plan.

        Args:
            plan: The migration plan to execute.

        Returns:
            True if the migration completed successfully.
        """
        logger.info(
            f"Executing migration: {plan.source_node_id} -> {plan.target_node_id}"
        )

        try:
            for step in plan.steps:
                logger.debug(f"Migration step: {step.name} ({step.description})")
                if step.name == "dump_kv_cache" and self._kv_cache_dump:
                    await self._kv_cache_dump(plan.source_node_id)
                elif step.name == "restore_kv_cache" and self._kv_cache_restore:
                    await self._kv_cache_restore(plan.target_node_id, ...)

            if plan.strategy in (MigrationStrategy.PROACTIVE, MigrationStrategy.REACTIVE):
                if self._kv_cache_restore:
                    await self._kv_cache_restore(plan.target_node_id, ...)

        except Exception as e:
            logger.error(f"Migration failed for {plan.source_node_id}: {e}")
            return False

        self._completed.append(plan)
        self._active_plans.pop(plan.source_node_id, None)
        logger.info(
            f"Migration completed: {plan.source_node_id} -> {plan.target_node_id} "
            f"({plan.total_estimated_duration_s:.1f}s)"
        )
        return True

    def get_active_plans(self) -> list[MigrationPlan]:
        return list(self._active_plans.values())

    def get_completed_plans(self, limit: int = 20) -> list[MigrationPlan]:
        return self._completed[-limit:]
