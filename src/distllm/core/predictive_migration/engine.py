from __future__ import annotations

import threading
import time
from typing import Any, Callable

from loguru import logger

from distllm.core.predictive_migration.tracker import (
    PrefixFrequencyTracker,
)
from distllm.core.predictive_migration.predictor import (
    MarkovChainPredictor,
    Prediction,
)
from distllm.core.predictive_migration.store import (
    ContentAddressableStore,
)
from distllm.core.predictive_migration.migration import (
    PreMigrationScheduler,
)


class PredictiveMigrationEngine:
    """Orchestrates the full predictive KV cache migration lifecycle.

    Runs a background loop that:
    1. Observes incoming prompt prefixes
    2. Updates the Markov chain predictor
    3. Predicts the next likely prefixes
    4. Schedules and executes pre-migration of KV cache

    Usage:
        engine = PredictiveMigrationEngine()
        engine.start()

        # On each request:
        engine.observe(prompt_tokens)

        # Stop:
        engine.stop()
    """

    def __init__(
        self,
        tracker: PrefixFrequencyTracker | None = None,
        predictor: MarkovChainPredictor | None = None,
        store: ContentAddressableStore | None = None,
        scheduler: PreMigrationScheduler | None = None,
        observe_interval: float = 10.0,
        predict_interval: float = 30.0,
        migrate_interval: float = 15.0,
        source_node: str = "local",
        target_nodes: list[str] | None = None,
        confidence_threshold: float = 0.3,
        top_k_predictions: int = 10,
        hash_fn: Callable[[list[int]], str] | None = None,
    ):
        self._tracker = tracker or PrefixFrequencyTracker()
        self._predictor = predictor or MarkovChainPredictor()
        self._store = store or ContentAddressableStore()
        self._scheduler = scheduler or PreMigrationScheduler()

        self._observe_interval = observe_interval
        self._predict_interval = predict_interval
        self._migrate_interval = migrate_interval
        self._source_node = source_node
        self._target_nodes = target_nodes or ["node-a", "node-b"]
        self._confidence_threshold = confidence_threshold
        self._top_k = top_k_predictions
        self._hash_fn = hash_fn or self._default_hash

        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        self._total_observed: int = 0
        self._total_predictions: int = 0
        self._total_migrations: int = 0
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(self, token_ids: list[int], cluster: str = "default") -> None:
        """Observe a prompt and update tracker + predictor.

        Call this on each incoming request with the prompt tokens.
        """
        prefix_hash = self._tracker.observe(token_ids, cluster)
        if prefix_hash:
            self._predictor.observe(prefix_hash)
            with self._lock:
                self._total_observed += 1

    def start(self) -> None:
        """Start the background migration loop."""
        if self._running:
            return
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._loop, daemon=True
        )
        self._thread.start()
        logger.info("PredictiveMigrationEngine started")

    def stop(self) -> None:
        """Stop the background loop."""
        self._running = False
        logger.info("PredictiveMigrationEngine stopped")

    def set_transfer_fn(
        self,
        transfer_fn: Callable[[str, str, str], bool],
    ) -> None:
        """Set the KV cache transfer function for the scheduler."""
        self._scheduler._transfer_fn = transfer_fn

    def set_target_nodes(self, nodes: list[str]) -> None:
        self._target_nodes = list(nodes)

    def set_source_node(self, node_id: str) -> None:
        self._source_node = node_id

    def predict(self, top_k: int | None = None) -> list[Prediction]:
        """Get current predictions for the next likely prefixes.

        Uses the most recently observed prefix as the current state.
        """
        top = self._tracker.top_prefixes(1)
        if not top:
            return []
        current = top[0].prefix_hash
        k = top_k or self._top_k
        return self._predictor.predict(current_hash=current, top_k=k)

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        last_observe = 0.0
        last_predict = 0.0
        last_migrate = 0.0
        last_cleanup = 0.0

        while self._running:
            now = time.time()

            # Phase 1: Process observations (flushes buffer)
            if now - last_observe >= self._observe_interval:
                last_observe = now

            # Phase 2: Predict next prefixes
            if now - last_predict >= self._predict_interval:
                last_predict = now
                self._run_prediction_phase()

            # Phase 3: Execute scheduled migrations
            if now - last_migrate >= self._migrate_interval:
                last_migrate = now
                self._run_migration_phase()

            # Periodic cleanup
            if now - last_cleanup >= 300.0:
                last_cleanup = now
                self._run_cleanup()

            time.sleep(1.0)

    def _run_prediction_phase(self) -> None:
        predictions = self.predict()
        if not predictions:
            return

        with self._lock:
            self._total_predictions += len(predictions)

        logger.debug(
            f"Predicted {len(predictions)} next prefixes "
            f"(top confidence: {predictions[0].confidence:.3f})"
        )

        tasks = self._scheduler.schedule(
            predictions=predictions,
            content_store=self._store,
            source_node=self._source_node,
            target_nodes=self._target_nodes,
            confidence_threshold=self._confidence_threshold,
        )
        if tasks:
            logger.info(
                f"Scheduled {len(tasks)} pre-migrations from predictions"
            )

    def _run_migration_phase(self) -> None:
        import asyncio

        try:
            loop = asyncio.new_event_loop()
            completed = loop.run_until_complete(
                self._scheduler.execute_batch()
            )
            loop.close()
        except Exception:
            # Fallback: synchronous execution is handled inside scheduler
            # for the no-async case
            logger.debug("Migration phase: no async loop available")
            completed = []

        if completed:
            successful = [
                t for t in completed if t.status.name == "COMPLETED"
            ]
            with self._lock:
                self._total_migrations += len(successful)
            for task in successful:
                logger.debug(
                    f"Migrated {task.content_hash}: "
                    f"{task.source_node} -> {task.target_node} "
                    f"({task.duration_ms:.0f}ms)"
                )

    def _run_cleanup(self) -> None:
        expired = self._store.sweep_expired()
        stale_migrations = self._scheduler.cleanup_stale_recent()
        old_completed = self._scheduler.cleanup_old_completed()
        if any([expired, stale_migrations, old_completed]):
            logger.debug(
                f"Cleanup: {expired} expired cache entries, "
                f"{stale_migrations} stale migration records, "
                f"{old_completed} old completed tasks"
            )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def uptime_seconds(self) -> float:
        if self._start_time == 0:
            return 0.0
        return time.time() - self._start_time

    def _default_hash(self, token_ids: list[int]) -> str:
        import hashlib

        raw = ",".join(str(t) for t in token_ids)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "uptime_seconds": int(self.uptime_seconds),
            "total_observed": self._total_observed,
            "total_predictions": self._total_predictions,
            "total_migrations": self._total_migrations,
            "tracker": self._tracker.stats(),
            "predictor": self._predictor.stats(),
            "store": self._store.stats(),
            "scheduler": self._scheduler.stats(),
            "source_node": self._source_node,
            "target_nodes": self._target_nodes,
            "confidence_threshold": self._confidence_threshold,
        }

    def summary(self) -> str:
        s = self.stats()
        lines = [
            f"PredictiveMigrationEngine: {'RUNNING' if s['running'] else 'STOPPED'}",
            f"  Uptime: {s['uptime_seconds']}s",
            f"  Observed: {s['total_observed']} prefixes",
            f"  Predictions made: {s['total_predictions']}",
            f"  Migrations completed: {s['total_migrations']}",
            f"  Cache entries: {s['store']['entries']}",
            f"  Cache hit rate: {s['store']['hit_rate']:.1%}",
            f"  Scheduler pending: {s['scheduler']['pending']}",
            f"  Scheduler in-flight: {s['scheduler']['in_flight']}",
            f"  Predictor states: {s['predictor']['total_states']}",
        ]
        return "\n".join(lines)
