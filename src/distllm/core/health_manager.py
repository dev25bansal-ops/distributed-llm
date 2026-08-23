"""Health probing, node recovery, and straggler detection.

Integrates:
- FailoverEngine state machine (HEALTHY → DEGRADED → UNHEALTHY → OFFLINE)
- NodeRecoveryManager for checkpoint-based self-healing
- StragglerDetector for slow node detection
- ReputationSystem for quality tracking
"""

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

from loguru import logger

from distllm.core.resource_manager import ResourceManager
from distllm.dist.pipeline import PipelineOrchestrator
from distllm.dist.recovery import (
    NodeRecoveryManager,
)
from distllm.dist.reputation import ReputationSystem
from distllm.dist.straggler import DetectionMethod, StragglerDetector
from distllm.health.failover import FailoverEngine
from distllm.health.state import HealthRecord, HealthStateStore, NodeState


class HealthManager:
    """Periodic health probes, node failure recovery, straggler detection.

    Uses FailoverEngine state machine for progressive degradation:
    HEALTHY → DEGRADED → UNHEALTHY → OFFLINE

    Runs a background thread that pings all registered nodes, updates
    reputations, detects stragglers, and triggers recovery callbacks.
    """

    def __init__(
        self,
        pipeline: PipelineOrchestrator,
        resource_mgr: ResourceManager,
        reputation: ReputationSystem | None = None,
        recovery_manager: NodeRecoveryManager | None = None,
        straggler_detector: StragglerDetector | None = None,
        check_interval_s: float = 10.0,
        health_check_timeout_s: float = 5.0,
        failover_engine: FailoverEngine | None = None,
    ):
        self._pipeline = pipeline
        self._resource_mgr = resource_mgr
        self._reputation = reputation
        self._recovery_manager = recovery_manager or NodeRecoveryManager()
        self._straggler_detector = straggler_detector or StragglerDetector(
            detection_method=DetectionMethod.MAD,
            on_straggler_cb=self._on_straggler_detected,
        )
        self._check_interval_s = check_interval_s
        self._health_check_timeout_s = health_check_timeout_s

        self._failover = failover_engine or FailoverEngine()
        self._health_store = HealthStateStore()
        self._failover.on_state_change(self._on_state_change)

        self._running = threading.Event()
        self._health_event = threading.Event()
        self._health_thread: threading.Thread | None = None

        cpu_count = os.cpu_count() or 1
        self._executor = ThreadPoolExecutor(
            max_workers=min(32, cpu_count * 4),
            thread_name_prefix="health-probe",
        )

        self._setup_recovery_callbacks()

    def start(self) -> None:
        self._running.set()
        self._health_event.clear()
        self._health_thread = threading.Thread(
            target=self._health_probe_loop,
            daemon=True,
            name="health-probe",
        )
        self._health_thread.start()
        logger.info(f"Health manager started (check every {self._check_interval_s}s)")

    def stop(self) -> None:
        self._running.clear()
        self._health_event.set()
        if self._health_thread and self._health_thread.is_alive():
            self._health_thread.join(timeout=3.0)
        # Shut down the thread pool executor to release worker threads.
        # Without this, worker threads remain alive until the process
        # exits, causing a thread leak on restart.
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass

    @property
    def straggler_detector(self) -> StragglerDetector:
        return self._straggler_detector

    @property
    def recovery_manager(self) -> NodeRecoveryManager:
        return self._recovery_manager

    @property
    def failover_engine(self) -> FailoverEngine:
        return self._failover

    @property
    def health_store(self) -> HealthStateStore:
        return self._health_store

    def is_healthy(self) -> bool:
        """Return whether the coordinator's health manager is running.

        Used by the HA replication heartbeat as a lightweight liveness signal
        (a manager that has been started with a live pipeline is considered
        healthy).
        """
        return self._running.is_set() and self._pipeline is not None

    # --- State transition callback ---

    def _on_state_change(
        self, node_id: str, old_state: NodeState, new_state: NodeState
    ) -> None:
        logger.info(f"Node {node_id}: {old_state.value} → {new_state.value}")

        if new_state == NodeState.UNHEALTHY or new_state == NodeState.OFFLINE:
            self._resource_mgr.record_failure(node_id)
            self._recovery_manager.on_node_failure(node_id)

        elif new_state == NodeState.DEGRADED and old_state in (
            NodeState.OFFLINE, NodeState.UNHEALTHY
        ):
            # Node recovering — mark alive, reset straggler baseline
            self._recovery_manager.mark_alive(node_id)
            self._straggler_detector.reset_baseline(node_id)

        elif new_state == NodeState.HEALTHY and old_state == NodeState.DEGRADED:
            self._recovery_manager.mark_alive(node_id)
            self._straggler_detector.reset_baseline(node_id)

    # --- Straggler callback ---

    def _on_straggler_detected(self, report) -> None:
        logger.warning(
            f"Straggler {report.node_id}: {report.slowdown_factor}x slower "
            f"(action: {report.recommended_action})"
        )
        if report.recommended_action == "reassign_layers":
            self._recovery_manager.on_node_failure(report.node_id)

    # --- Recovery callbacks ---

    def _setup_recovery_callbacks(self) -> None:
        if self._recovery_manager is None:
            return

        def on_drain(node_id: str) -> None:
            logger.info(f"Recovery: draining node {node_id}")
            node = self._pipeline.nodes.get(node_id)
            if node:
                node.healthy = False
            self._resource_mgr.record_failure(node_id)

        def on_redistribute(node_id: str, plan) -> None:
            logger.info(f"Recovery: redistributing layers from {node_id}")
            for redistribution in plan.redistributions:
                surviving = self._pipeline.nodes.get(redistribution.surviving_node_id)
                if surviving:
                    surviving.start_layer = redistribution.new_start_layer
                    surviving.end_layer = redistribution.new_end_layer
                    logger.info(
                        f"  {redistribution.surviving_node_id}: now layers "
                        f"{redistribution.new_start_layer}-{redistribution.new_end_layer}"
                    )

        def on_recover(node_id: str, seq_ids: list[str]) -> list:
            """Recover sequences from failed node onto surviving nodes."""
            logger.info(f"Recovery: recovering {len(seq_ids)} sequences from {node_id}")
            recovered = []
            surviving_nodes = [
                nid for nid, n in self._pipeline.nodes.items()
                if nid != node_id and getattr(n, "healthy", False)
            ]
            if not surviving_nodes:
                logger.warning("No surviving nodes for sequence recovery")
                return recovered

            target_id = surviving_nodes[0]
            target_node = self._pipeline.nodes.get(target_id)

            for seq_id in seq_ids:
                ckpt = self._recovery_manager.get_checkpoint(seq_id)
                if ckpt is None:
                    continue
                try:
                    # Transfer KV cache to surviving node if possible
                    if target_node and hasattr(target_node, "client") and target_node.client:
                        if ckpt.kv_cache is not None:
                            target_node.client.transfer_kv_cache(ckpt.kv_cache)
                    recovered.append(seq_id)
                    logger.info(f"  Recovered {seq_id} → {target_id}")
                except Exception as e:
                    logger.error(f"  Failed to recover {seq_id}: {e}")

            return recovered

        def on_mark_dead(node_id: str) -> None:
            logger.info(f"Recovery: marking node {node_id} as dead")
            self._pipeline.nodes.pop(node_id, None)
            self._pipeline.node_order = sorted(
                self._pipeline.nodes.keys(),
                key=lambda nid: self._pipeline.nodes[nid].start_layer,
            )

        self._recovery_manager.set_drain_callback(on_drain)
        self._recovery_manager.set_redistribute_layers_callback(on_redistribute)
        self._recovery_manager.set_recover_sequences_callback(on_recover)
        self._recovery_manager.set_mark_dead_callback(on_mark_dead)

    # --- Per-node health probe (runs in thread pool) ---

    def _probe_single_node(self, node_id: str, node) -> None:
        """Probe one node and update health store.  Runs in executor thread."""
        try:
            # Get or create health record
            record = self._health_store.get(node_id)
            if record is None:
                record = HealthRecord(node_id=node_id)
                self._health_store.set(node_id, record)

            _t0 = time.monotonic()
            alive = node.health_check()
            latency_ms = (time.monotonic() - _t0) * 1000.0

            if self._reputation:
                self._reputation.record_health(node_id, alive)

            # Use FailoverEngine state machine
            new_state = self._failover.evaluate(record, alive, latency_ms)
            self._health_store.update_state(node_id, new_state)

        except Exception:
            raise   # let the caller (as_completed) handle logging

    # --- Health probe loop ---

    def _health_probe_loop(self) -> None:
        consecutive_errors = 0
        max_backoff_s = 60.0

        while not self._health_event.is_set():
            try:
                self._health_event.wait(self._check_interval_s)
                if self._health_event.is_set() or not self._running.is_set():
                    break

                # Evict stale checkpoints periodically
                self._recovery_manager.evict_stale_checkpoints()

                nodes_snapshot = self._pipeline.snapshot_nodes()

                futures = {}
                for node_id, node in nodes_snapshot.items():
                    if node is None or node.client is None:
                        continue
                    future = self._executor.submit(
                        self._probe_single_node, node_id, node
                    )
                    futures[future] = node_id

                for future in as_completed(futures):
                    node_id = futures[future]
                    try:
                        future.result(timeout=self._health_check_timeout_s)
                    except TimeoutError:
                        logger.error(
                            f"Health check timed out for node {node_id}"
                        )
                        consecutive_errors += 1
                    except Exception as e:
                        logger.error(
                            f"Health check failed for node {node_id}: {e}"
                        )
                        consecutive_errors += 1

                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                logger.error(
                    f"Health probe loop error ({consecutive_errors}x): {e}",
                    exc_info=True,
                )
                backoff = min(consecutive_errors * 2.0, max_backoff_s)
                if self._health_event.wait(backoff):
                    break

    def get_node_status(self) -> dict:
        nodes_status = {}
        for node_id, node in self._pipeline.nodes.items():
            try:
                record = self._health_store.get(node_id)
                state = record.state.value if record else "unknown"
                nodes_status[node_id] = {
                    "healthy": getattr(node, "healthy", False),
                    "state": state,
                    "start_layer": node.start_layer,
                    "end_layer": node.end_layer,
                    "gpu_name": getattr(node, "gpu_name", ""),
                    "gpu_memory_free": getattr(node, "gpu_memory_free", 0),
                }
            except Exception:
                nodes_status[node_id] = {"healthy": False, "state": "unknown"}
        return nodes_status
