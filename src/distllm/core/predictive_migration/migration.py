from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from loguru import logger


class MigrationStatus(Enum):
    PENDING = "pending"
    IN_FLIGHT = "in_flight"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MigrationTask:
    """A single KV cache migration from source to target node."""
    content_hash: str
    source_node: str
    target_node: str
    size_bytes: int = 0
    status: MigrationStatus = MigrationStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    duration_ms: float | None = None
    error: str | None = None


class PreMigrationScheduler:
    """Schedules KV cache pre-migration based on predictions.

    Given predicted next prefixes from the Markov model, determines
    which KV cache entries to migrate, to which nodes, and manages
    the migration lifecycle. Batches migrations, respects bandwidth
    limits, and handles TTL for migrated entries.

    Usage:
        scheduler = PreMigrationScheduler(transfer_fn=my_transfer)
        scheduler.schedule(predictions, current_node="node-a")
        await scheduler.execute_batch()
    """

    def __init__(
        self,
        transfer_fn: Callable[[str, str, str], bool] | None = None,
        max_concurrent: int = 4,
        max_bandwidth_mbps: float = 1000.0,
        migrated_ttl_secs: float = 600.0,
    ):
        self._transfer_fn = transfer_fn
        self._max_concurrent = max_concurrent
        self._max_bandwidth_mbps = max_bandwidth_mbps
        self._migrated_ttl = migrated_ttl_secs

        self._pending: list[MigrationTask] = []
        self._in_flight: list[MigrationTask] = []
        self._completed: list[MigrationTask] = []
        self._failed: list[MigrationTask] = []
        self._recent_migrations: dict[str, float] = {}  # hash -> timestamp

        self._total_bytes_transferred: int = 0
        self._bandwidth_window: list[tuple[float, int]] = []

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def schedule(
        self,
        predictions: list[Any],
        content_store,
        source_node: str,
        target_nodes: list[str],
        confidence_threshold: float = 0.3,
    ) -> list[MigrationTask]:
        """Schedule migrations for predicted prefixes.

        Args:
            predictions: List of Prediction objects from MarkovChainPredictor.
            content_store: ContentAddressableStore to look up entries.
            source_node: Node ID where the KV cache currently resides.
            target_nodes: Candidate target node IDs for placement.
            confidence_threshold: Minimum confidence to trigger migration.

        Returns:
            List of newly created MigrationTask objects.
        """
        created: list[MigrationTask] = []

        for pred in predictions:
            if pred.confidence < confidence_threshold:
                continue

            if pred.prefix_hash in self._recent_migrations:
                continue

            entry = content_store.get_entry(pred.prefix_hash)
            if entry is None:
                continue

            target = self._select_target_node(
                pred.prefix_hash, target_nodes
            )
            if target == source_node:
                continue

            task = MigrationTask(
                content_hash=pred.prefix_hash,
                source_node=source_node,
                target_node=target,
                size_bytes=entry.size_bytes,
            )
            self._pending.append(task)
            self._recent_migrations[pred.prefix_hash] = time.time()
            created.append(task)

        if created:
            logger.info(
                f"Scheduled {len(created)} migrations "
                f"(pending: {len(self._pending)})"
            )

        return created

    def _select_target_node(
        self, content_hash: str, candidates: list[str]
    ) -> str:
        if not candidates:
            return "unknown"
        bucket = int(content_hash[:8], 16) % len(candidates)
        return candidates[bucket]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute_batch(self) -> list[MigrationTask]:
        """Execute pending migrations up to max_concurrent.

        Returns completed tasks from this batch.
        """
        while self._pending and len(self._in_flight) < self._max_concurrent:
            task = self._pending.pop(0)
            task.status = MigrationStatus.IN_FLIGHT
            task.started_at = time.time()
            self._in_flight.append(task)

        if not self._in_flight:
            return []

        completed_batch: list[MigrationTask] = []
        for task in self._in_flight[:]:
            try:
                if self._transfer_fn is None:
                    logger.debug(
                        f"Simulating migration: "
                        f"{task.content_hash} "
                        f"{task.source_node} -> {task.target_node}"
                    )
                    success = True
                else:
                    success = self._transfer_fn(
                        task.content_hash,
                        task.source_node,
                        task.target_node,
                    )

                task.duration_ms = (
                    time.time() - task.started_at
                ) * 1000
                task.completed_at = time.time()

                if success:
                    task.status = MigrationStatus.COMPLETED
                    self._completed.append(task)
                    self._record_transfer(task.size_bytes)
                else:
                    task.status = MigrationStatus.FAILED
                    task.error = "transfer_fn returned False"
                    self._failed.append(task)

                self._in_flight.remove(task)
                completed_batch.append(task)

            except Exception as e:
                task.status = MigrationStatus.FAILED
                task.error = str(e)
                task.completed_at = time.time()
                self._in_flight.remove(task)
                self._failed.append(task)
                completed_batch.append(task)

        return completed_batch

    def _record_transfer(self, size_bytes: int) -> None:
        self._total_bytes_transferred += size_bytes
        self._bandwidth_window.append((time.time(), size_bytes))
        cutoff = time.time() - 10
        self._bandwidth_window = [
            (t, s) for t, s in self._bandwidth_window if t > cutoff
        ]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_stale_recent(self, max_age_secs: float = 300.0) -> int:
        now = time.time()
        stale = [
            h
            for h, ts in self._recent_migrations.items()
            if now - ts > max_age_secs
        ]
        for h in stale:
            self._recent_migrations.pop(h, None)
        return len(stale)

    def cleanup_old_completed(
        self, max_age_secs: float = 3600.0
    ) -> int:
        now = time.time()
        before = len(self._completed)
        self._completed = [
            t
            for t in self._completed
            if t.completed_at and now - t.completed_at < max_age_secs
        ]
        return before - len(self._completed)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def in_flight_count(self) -> int:
        return len(self._in_flight)

    @property
    def completed_count(self) -> int:
        return len(self._completed)

    @property
    def failed_count(self) -> int:
        return len(self._failed)

    def current_bandwidth_mbps(self) -> float:
        cutoff = time.time() - 10
        recent = [
            s for t, s in self._bandwidth_window if t > cutoff
        ]
        if not recent:
            return 0.0
        total_bytes = sum(recent)
        return (total_bytes * 8) / (10 * 1_000_000)

    def stats(self) -> dict[str, Any]:
        return {
            "pending": self.pending_count,
            "in_flight": self.in_flight_count,
            "completed": self.completed_count,
            "failed": self.failed_count,
            "total_bytes_transferred": self._total_bytes_transferred,
            "total_mb_transferred": round(
                self._total_bytes_transferred / (1024 * 1024), 2
            ),
            "current_bandwidth_mbps": round(
                self.current_bandwidth_mbps(), 2
            ),
            "max_concurrent": self._max_concurrent,
            "migrated_ttl_secs": self._migrated_ttl,
            "recent_migrations": len(self._recent_migrations),
        }

    def reset(self) -> None:
        self._pending.clear()
        self._in_flight.clear()
        self._completed.clear()
        self._failed.clear()
        self._recent_migrations.clear()
        self._total_bytes_transferred = 0
        self._bandwidth_window.clear()
