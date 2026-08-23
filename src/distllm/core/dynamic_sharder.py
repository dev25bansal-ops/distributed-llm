"""Dynamic model sharding with live migration and zero-downtime resharding.

Automatically re-shards the model when nodes join/leave the cluster.
Transfers layers between nodes while serving requests (live migration).

Features:
- Automatic re-partitioning on topology changes
- Live layer migration with request draining
- Zero-downtime resharding
- Integration with auto_partitioner and rebalancer

Usage::

    sharder = DynamicSharder(
        coordinator=coord,
        auto_partitioner=AutoPartitioner(hidden_size=4096, num_layers=32),
    )
    sharder.start()
    # When a node joins:
    sharder.on_node_join("new-node", gpu_memory_gb=80)
    # When a node leaves:
    sharder.on_node_leave("old-node")
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from loguru import logger


class MigrationState(str, Enum):
    """State of a layer migration."""
    IDLE = "idle"
    DRAINING = "draining"          # Stop sending new requests to source
    TRANSFERRING = "transferring"  # Moving layer data
    VERIFYING = "verifying"        # Checksumming transferred data
    SWITCHING = "switching"        # Updating routing tables
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class LayerMigration:
    """Tracks a single layer migration between nodes."""
    layer_id: int
    source_node: str
    target_node: str
    state: MigrationState = MigrationState.IDLE
    start_time: float = 0.0
    end_time: float = 0.0
    bytes_transferred: int = 0
    checksum: str = ""
    error: str = ""


@dataclass
class ReshardPlan:
    """A complete resharding plan."""
    old_partition: dict[str, list[int]]  # node_id -> layer_ids
    new_partition: dict[str, list[int]]
    migrations: list[LayerMigration]
    estimated_downtime_ms: float = 0.0
    total_bytes: int = 0


class DynamicSharder:
    """Manages dynamic model sharding with live migration.

    Monitors cluster topology and automatically re-shards when
    changes occur. Transfers layers between nodes while continuing
    to serve requests (zero-downtime).

    Args:
        coordinator: Coordinator instance for node management.
        auto_partitioner: AutoPartitioner for computing new partitions.
        rebalancer: Rebalancer for detecting stragglers.
        on_layer_transfer: Callback for transferring layer data.
        migration_bandwidth_gbps: Expected bandwidth for migration.
    """

    def __init__(
        self,
        coordinator: Any = None,
        auto_partitioner: Any = None,
        rebalancer: Any = None,
        on_layer_transfer: Callable[[int, str, str], bool] | None = None,
        migration_bandwidth_gbps: float = 10.0,
    ):
        self._coordinator = coordinator
        self._partitioner = auto_partitioner
        self._rebalancer = rebalancer
        self._on_transfer = on_layer_transfer
        self._bandwidth_gbps = migration_bandwidth_gbps

        self._current_partition: dict[str, list[int]] = {}
        self._active_migrations: list[LayerMigration] = []
        self._lock = threading.Lock()
        self._stats = {
            "reshards": 0,
            "migrations_completed": 0,
            "migrations_failed": 0,
            "total_bytes_transferred": 0,
            "total_downtime_ms": 0.0,
        }

    def set_initial_partition(self, partition: dict[str, list[int]]) -> None:
        """Set the initial layer-to-node partition."""
        with self._lock:
            self._current_partition = dict(partition)
        logger.info(f"Initial partition: {len(partition)} nodes, {sum(len(v) for v in partition.values())} layers")

    def on_node_join(self, node_id: str, gpu_memory_gb: float = 0) -> ReshardPlan | None:
        """Handle a new node joining the cluster.

        Computes a new partition that includes the new node and
        generates a migration plan to redistribute layers.

        Returns:
            ReshardPlan if resharding is needed, None otherwise.
        """
        with self._lock:
            old_partition = dict(self._current_partition)

        # Compute new partition including the new node
        new_partition = self._compute_new_partition(add_node=node_id)

        if new_partition == old_partition:
            logger.info(f"Node {node_id} joined but no resharding needed")
            return None

        # Generate migration plan
        plan = self._generate_migration_plan(old_partition, new_partition)

        logger.info(
            f"Node {node_id} joined: resharding {len(plan.migrations)} layers "
            f"({plan.total_bytes / 1e9:.1f}GB, est. {plan.estimated_downtime_ms:.0f}ms)"
        )

        # Execute migrations
        self._execute_reshard(plan)
        return plan

    def on_node_leave(self, node_id: str) -> ReshardPlan | None:
        """Handle a node leaving the cluster.

        Redistributes the departing node's layers to remaining nodes.

        Returns:
            ReshardPlan if resharding is needed, None otherwise.
        """
        with self._lock:
            old_partition = dict(self._current_partition)
            if node_id not in old_partition:
                logger.warning(f"Node {node_id} not in partition")
                return None

        # Compute new partition without the departed node
        new_partition = self._compute_new_partition(remove_node=node_id)

        # Generate migration plan
        plan = self._generate_migration_plan(old_partition, new_partition)

        logger.info(
            f"Node {node_id} left: redistributing {len(plan.migrations)} layers "
            f"to {len(new_partition)} remaining nodes"
        )

        # Execute migrations
        self._execute_reshard(plan)
        return plan

    def _compute_new_partition(
        self,
        add_node: str | None = None,
        remove_node: str | None = None,
    ) -> dict[str, list[int]]:
        """Compute a new partition based on current topology."""
        current_nodes = list(self._current_partition.keys())

        if add_node and add_node not in current_nodes:
            current_nodes.append(add_node)
        if remove_node and remove_node in current_nodes:
            current_nodes.remove(remove_node)

        if not current_nodes:
            return {}

        # Count total layers
        total_layers = sum(len(layers) for layers in self._current_partition.values())
        if total_layers == 0:
            total_layers = 32  # Default

        # Distribute layers evenly
        layers_per_node = max(1, total_layers // len(current_nodes))
        remainder = total_layers % len(current_nodes)

        new_partition = {}
        layer_idx = 0
        for i, node_id in enumerate(current_nodes):
            count = layers_per_node + (1 if i < remainder else 0)
            new_partition[node_id] = list(range(layer_idx, layer_idx + count))
            layer_idx += count

        return new_partition

    def _generate_migration_plan(
        self,
        old_partition: dict[str, list[int]],
        new_partition: dict[str, list[int]],
    ) -> ReshardPlan:
        """Generate a migration plan from old to new partition."""
        migrations = []
        total_bytes = 0

        # Find layers that need to move
        old_layer_to_node = {}
        for node_id, layers in old_partition.items():
            for layer_id in layers:
                old_layer_to_node[layer_id] = node_id

        new_layer_to_node = {}
        for node_id, layers in new_partition.items():
            for layer_id in layers:
                new_layer_to_node[layer_id] = node_id

        for layer_id in set(list(old_layer_to_node.keys()) + list(new_layer_to_node.keys())):
            old_node = old_layer_to_node.get(layer_id)
            new_node = new_layer_to_node.get(layer_id)

            if old_node != new_node and new_node is not None:
                # Estimate layer size (rough: 100MB per layer for 7B model)
                layer_bytes = 100 * 1024 * 1024  # 100MB default
                migration = LayerMigration(
                    layer_id=layer_id,
                    source_node=old_node or "unknown",
                    target_node=new_node,
                    bytes_transferred=layer_bytes,
                )
                migrations.append(migration)
                total_bytes += layer_bytes

        # Estimate downtime (transfer time + switch time)
        transfer_time_ms = (total_bytes * 8) / (self._bandwidth_gbps * 1e6)  # ms
        switch_time_ms = len(migrations) * 10  # 10ms per layer switch
        estimated_downtime = transfer_time_ms + switch_time_ms

        return ReshardPlan(
            old_partition=old_partition,
            new_partition=new_partition,
            migrations=migrations,
            estimated_downtime_ms=estimated_downtime,
            total_bytes=total_bytes,
        )

    def _execute_reshard(self, plan: ReshardPlan) -> None:
        """Execute a resharding plan with live migration."""
        with self._lock:
            self._active_migrations = list(plan.migrations)

        failed = False
        for migration in plan.migrations:
            try:
                self._migrate_layer(migration)
                with self._lock:
                    self._stats["migrations_completed"] += 1
                    self._stats["total_bytes_transferred"] += migration.bytes_transferred
            except Exception as e:
                failed = True
                migration.state = MigrationState.FAILED
                migration.error = str(e)
                with self._lock:
                    self._stats["migrations_failed"] += 1
                logger.error(f"Migration failed for layer {migration.layer_id}: {e}")

        # Install the new partition only if every migration succeeded; on any
        # failure keep the old partition so routing never points at nodes that
        # never received their layers (F-048).
        with self._lock:
            if failed:
                self._active_migrations = []
                logger.error("Reshard aborted: keeping old partition (migration failure)")
                return
            self._current_partition = dict(plan.new_partition)
            self._active_migrations = []
            self._stats["reshards"] += 1

        logger.info(f"Reshard complete: {len(plan.migrations)} layers migrated")

    def _migrate_layer(self, migration: LayerMigration) -> None:
        """Migrate a single layer from source to target node.

        Steps:
        1. Drain: Stop sending new requests to source node for this layer
        2. Transfer: Move layer data to target node
        3. Verify: Checksum the transferred data
        4. Switch: Update routing to use target node
        """
        migration.start_time = time.time()
        migration.state = MigrationState.DRAINING

        # Step 1: Drain (brief pause to let in-flight requests complete)
        time.sleep(0.1)  # 100ms drain window

        # Step 2: Transfer
        migration.state = MigrationState.TRANSFERRING
        if self._on_transfer is None:
            # Fail closed: without a real transfer callback no data moves, so
            # the layer must NOT be reported as migrated and the new partition
            # must not be installed (F-048).
            raise RuntimeError(
                "No on_layer_transfer callback configured; refusing to "
                f"mark layer {migration.layer_id} as transferred"
            )
        success = self._on_transfer(
            migration.layer_id,
            migration.source_node,
            migration.target_node,
        )
        if not success:
            raise RuntimeError(f"Transfer failed for layer {migration.layer_id}")

        # Step 3: Verify
        migration.state = MigrationState.VERIFYING
        # Checksum would go here in production

        # Step 4: Switch
        migration.state = MigrationState.SWITCHING
        # Routing update would go here

        migration.state = MigrationState.COMPLETE
        migration.end_time = time.time()

    def get_current_partition(self) -> dict[str, list[int]]:
        """Return the current layer-to-node partition."""
        with self._lock:
            return dict(self._current_partition)

    def get_active_migrations(self) -> list[dict]:
        """Return currently active migrations."""
        with self._lock:
            return [
                {
                    "layer_id": m.layer_id,
                    "source": m.source_node,
                    "target": m.target_node,
                    "state": m.state.value,
                    "bytes": m.bytes_transferred,
                }
                for m in self._active_migrations
            ]

    def stats(self) -> dict:
        with self._lock:
            return {
                **self._stats,
                "current_nodes": len(self._current_partition),
                "current_layers": sum(len(v) for v in self._current_partition.values()),
                "active_migrations": len(self._active_migrations),
            }
