"""SubsystemManager — manages lifecycle of coordinator subsystems.

Extracted from ``coordinator.py`` to reduce class size.  Pure code move with
no logic changes.  Each method operates on the coordinator instance stored
as ``self.coordinator``.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger
from transformers import AutoTokenizer

from distllm.core.memory_defragmenter import TieredCompactionLevel

if TYPE_CHECKING:
    from distllm.core.coordinator import Coordinator


class SubsystemManager:
    """Manages subsystem lifecycle for a Coordinator instance.

    The subsystem manager stores a reference to the coordinator and manages
    start/stop lifecycle, optional subsystem loading, health tracking,
    defragmentation, state replication, and related background tasks.
    """

    def __init__(self, coordinator: Coordinator) -> None:
        self.coordinator = coordinator

    # ── Straggler callback ──

    def _on_straggler_detected(self, report: Any) -> None:
        """Callback invoked when a straggler node is detected."""
        logger.warning(
            f"Straggler {report.node_id}: {report.slowdown_factor}x slower "
            f"(action: {report.recommended_action})"
        )
        if report.recommended_action == "reassign_layers" and self.coordinator._recovery_manager is not None:
            self.coordinator._recovery_manager.on_node_failure(report.node_id)

    # ── HA State Replication (delegated to CoordinatorElection) ──

    def _start_state_replication(self) -> None:
        """Start continuous state replication to HA peer coordinators.

        Delegated to :meth:`CoordinatorElection._start_state_replication`.
        """
        self.coordinator._election._start_state_replication()

    def _replication_loop(self) -> None:
        """Continuously push state snapshots to HA peers.

        Delegated to :meth:`CoordinatorElection._replication_loop`.
        """
        self.coordinator._election._replication_loop()

    def set_replication_peers(self, peer_urls: list[str]) -> None:
        """Set HA peer coordinator URLs for state replication.

        Delegated to :meth:`CoordinatorElection.set_replication_peers`.
        """
        self.coordinator._election.set_replication_peers(peer_urls)

    # ── Utilization ──

    def _default_utilization_fn(self) -> float:
        """Compute cluster utilization fraction (0.0 idle, 1.0 max)."""
        try:
            bs = self.coordinator._batch_scheduler
            if bs is not None:
                stats = bs.stats()
                active = stats.get("active_requests", 0)
                pending = stats.get("pending_requests", 0)
                max_batch = stats.get("max_batch_size", 4)
                total = float(active + pending)
                return min(total / max(max_batch, 1), 1.0)
        except Exception as e:
            logger.warning(f"Failed to compute cluster utilization: {e}")
        return 0.0

    # ── Async defragmentation ──

    async def _defrag_loop(self) -> None:
        """Background loop that periodically checks and runs defragmentation."""
        if self.coordinator._defragmenter is None:
            return

        interval = self.coordinator._defragmenter.config.interval_seconds
        logger.debug(f"Defrag background loop started (interval={interval}s)")

        while not self.coordinator._shutting_down:
            try:
                await asyncio.sleep(interval)

                if self.coordinator._shutting_down:
                    break

                # Find PagedAttentionManager instances to defrag
                for backend in self._get_paged_backends():
                    if self.coordinator._defragmenter.should_defragment(backend._blocks):
                        # Determine tier
                        ratio = self.coordinator._defragmenter._compute_fragmentation_ratio(backend._blocks)
                        tier = TieredCompactionLevel.L1_HOT
                        if self.coordinator._defragmenter.config.tiered_compaction:
                            if ratio > self.coordinator._defragmenter.config.l3_nvme_swap_threshold:
                                tier = TieredCompactionLevel.L3_COLD
                            elif ratio > self.coordinator._defragmenter.config.l2_cpu_swap_threshold:
                                tier = TieredCompactionLevel.L2_WARM

                        # Temperature-aware defragmentation: skip or reduce
                        # aggressiveness when active requests have high
                        # temperatures (>1.0), since high-temperature sampling
                        # is more sensitive to cache state changes.  Under
                        # high-temperature workloads, L1 compaction only.
                        active_temp = self._get_active_temperature()
                        if active_temp is not None and active_temp > 1.0:
                            if tier != TieredCompactionLevel.L1_HOT:
                                logger.debug(
                                    f"Temperature-aware defrag: reducing tier from {tier.value} "
                                    f"to L1_HOT (active temp={active_temp:.2f})"
                                )
                                tier = TieredCompactionLevel.L1_HOT

                        result = await self.coordinator._defragmenter.defragment_with_tier_async(backend, tier)
                        self.record_metric("defrag_blocks_moved", result.blocks_moved)
                        self.record_metric("defrag_duration_ms", result.time_ms)

                        if result.fragmentation_after < result.fragmentation_before:
                            logger.info(
                                f"Defrag improved fragmentation: "
                                f"{result.fragmentation_before:.1%} -> {result.fragmentation_after:.1%}"
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Defrag loop error: {e}")

    def _get_active_temperature(self) -> float | None:
        """Return the average temperature across active generation requests.

        Used by temperature-aware defragmentation to avoid aggressive
        cache reorganisation when high-temperature sampling is active,
        since high-temp outputs are more sensitive to cache state changes.

        Returns None if no active requests or no scheduler available.
        """
        bs = self.coordinator._batch_scheduler
        if bs is None:
            return None
        try:
            temps = bs.get_active_temperatures()
            return sum(temps) / len(temps) if temps else None
        except Exception:
            return None

    def _get_paged_backends(self) -> list[Any]:
        """Collect PagedAttentionManager instances from all backends."""
        backends: list[Any] = []
        engine = getattr(self.coordinator, "_inference_engine", None)
        if engine is not None:
            if hasattr(engine, "_paged_mgr") and engine._paged_mgr is not None:
                backends.append(engine._paged_mgr)
            if hasattr(engine, "backends"):
                for be in engine.backends:
                    if hasattr(be, "_paged_mgr") and be._paged_mgr is not None:
                        backends.append(be._paged_mgr)
        return backends

    # ── Metrics ──

    def record_metric(self, name: str, value: float = 1.0) -> None:
        """Record a metric (used by RequestPipeline)."""
        self.coordinator._request_handler.record_metric(name, value)

    def _cleanup_stale_results(self) -> None:
        """Remove stale entries from _request_results to prevent memory leaks."""
        self.coordinator._request_handler._cleanup_stale_results()

    # ── Subsystem startup helper ──

    def _start_subsystem(
        self,
        name: str,
        module_path: str,
        class_name: str,
        attrs_name: str,
        constructor_kwargs: dict | None = None,
        post_init: Callable | None = None,
    ) -> Any | None:
        """Start an optional subsystem, returning the instance or None.

        All 9 optional subsystems follow the same try/except pattern.
        This helper eliminates ~150 lines of near-identical boilerplate.

        Args:
            name: Subsystem name for health tracking (e.g. ``"discovery"``).
            module_path: Dot-separated module path.
            class_name: Class to import from *module_path*.
            attrs_name: Attribute name to store the instance on ``self``.
            constructor_kwargs: Dict of kwargs for the constructor.
            post_init: Optional callable ``fn(instance)`` called after init.

        Returns:
            The instance, or None if import failed.
        """
        try:
            mod = __import__(module_path, fromlist=[class_name])
            cls = getattr(mod, class_name)
            instance = cls(**(constructor_kwargs or {}))
            setattr(self.coordinator, attrs_name, instance)
            self.coordinator._subsystem_health[name] = {"status": "ok", "error": None}
            if post_init:
                post_init(instance)
            logger.info("{} initialized", name.replace("_", " ").title())
            return instance
        except ImportError as e:
            self.coordinator._subsystem_health[name] = {"status": "missing_deps", "error": str(e)}
            setattr(self.coordinator, attrs_name, None)
            logger.debug("{} not available: {}", name, e)
            return None
        except Exception as e:
            self.coordinator._subsystem_health[name] = {"status": "failed", "error": str(e)}
            setattr(self.coordinator, attrs_name, None)
            logger.error("{} failed to start: {}", name, e)
            return None

    # ── Start ──

    def start(
        self,
        blocking: bool = True,
        on_stop: Callable | None = None,
        health_check_interval_s: float = 10.0,
    ) -> None:
        """Start the coordinator and all subsystems.

        Args:
            blocking: If True, blocks until shutdown. If False, runs
                asynchronously and calls *on_stop* when done.
            on_stop: Optional callback invoked after async shutdown.
            health_check_interval_s: Seconds between health checks.
        """
        coord = self.coordinator

        if coord.tokenizer is None:
            coord.tokenizer = AutoTokenizer.from_pretrained(
                coord.model_name,
                trust_remote_code=coord.trust_remote_code,
                revision=coord.model_revision,
            )
        coord._health_check_interval_s = health_check_interval_s
        coord._running.set()
        coord._health_event.clear()
        coord._health_mgr.start()

        # Start HA state replication if peers configured
        coord._election._start_state_replication()

        if hasattr(coord, '_adaptive_compression_mgr') and coord._adaptive_compression_mgr:
            coord._adaptive_compression_mgr.start()

        if coord._defragmenter is not None:
            # BUG-007: Check for a running event loop before using ensure_future
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None and loop.is_running():
                coord._defrag_task = asyncio.ensure_future(self._defrag_loop())
                logger.info("Defrag background loop started (async)")
            else:
                # No running loop — start defrag in a background thread instead
                def _run_defrag_loop() -> None:
                    import asyncio as _asyncio
                    _loop = _asyncio.new_event_loop()
                    _asyncio.set_event_loop(_loop)
                    _loop.run_until_complete(self._defrag_loop())
                    _loop.close()
                t = threading.Thread(target=_run_defrag_loop, daemon=True, name="defrag-loop")
                t.start()
                logger.info("Defrag background loop started (threaded fallback)")

        # --- Optional subsystems (via _start_subsystem helper) ---
        self._start_subsystem(
            "discovery", "distllm.dist.discovery", "DiscoveryService", "_discovery",
            constructor_kwargs={"port": coord.port, "service_id": "distllm-coordinator"},
        )

        if (hasattr(coord.config, 'federation_config')
                and coord.config.federation_config
                and coord.config.federation_config.enabled):
            self._start_subsystem(
                "federation", "distllm.dist.federation", "FederationCoordinator", "_federation",
                constructor_kwargs={
                    "config": coord.config.federation_config,
                    "local_cluster_id": coord.config.federation_config.cluster_id,
                    "local_host": 'localhost',
                    "local_port": coord.port,
                    "coordinator_ref": coord,
                },
            )

        self._start_subsystem(
            "smart_router", "distllm.core.smart_model_router", "SmartModelRouter", "_smart_router",
        )

        self._start_subsystem(
            "semantic_cache", "distllm.core.semantic_cache", "SemanticCache", "_semantic_cache",
            constructor_kwargs={
                "similarity_threshold": float(os.environ.get("DISTLLM_SEMANTIC_CACHE_THRESHOLD", "0.92")),
                "max_entries": 10000,
            },
        )

        self._start_subsystem(
            "disaggregated_scheduler", "distllm.core.advanced_scheduling.disaggregated",
            "DisaggregatedBatchScheduler", "_disaggregated_scheduler",
        )

        self._start_subsystem(
            "carbon_engine", "distllm.core.carbon_migration", "CarbonMigrationEngine", "_carbon_engine",
            constructor_kwargs={
                "threshold": float(os.environ.get("DISTLLM_CARBON_THRESHOLD", "400.0")),
                "check_interval_s": 300.0,
            },
        )

        self._start_subsystem(
            "arbitrage_engine", "distllm.core.arbitrage_engine", "ArbitrageEngine", "_arbitrage_engine",
            constructor_kwargs={
                "min_savings_pct": float(os.environ.get("DISTLLM_ARBITRAGE_MIN_SAVINGS", "15.0")),
                "check_interval_s": float(os.environ.get("DISTLLM_ARBITRAGE_INTERVAL", "60.0")),
            },
        )

        self._start_subsystem(
            "cost_tracker", "distllm.core.cost_tracker", "CostTracker", "_cost_tracker",
        )

        # --- Wire new modules into production paths ---

        # AutonomousHealer: integrates with node failure callbacks
        coord._auto_healer = None
        try:
            from distllm.core.autonomous_healer import AutonomousHealer
            coord._auto_healer = AutonomousHealer(
                on_drain_callback=coord._on_node_drain,
                on_recover_callback=coord._on_node_recover,
                dry_run=os.environ.get("DISTLLM_AUTO_HEAL_DRY_RUN", "1") == "1",
            )
            coord._auto_healer.start()
            logger.info("Autonomous healer started (dry_run={})", os.environ.get("DISTLLM_AUTO_HEAL_DRY_RUN", "1"))
        except Exception as e:
            logger.debug("Autonomous healer not available: {}", e)

        # SpotEnsembleManager: multi-provider spot instance pooling
        coord._spot_ensemble = None
        try:
            from distllm.core.arbitrage_engine import SpotEnsembleManager
            coord._spot_ensemble = SpotEnsembleManager()
            logger.info("Spot ensemble manager initialized")
        except Exception as e:
            logger.debug("Spot ensemble not available: {}", e)

        # AgenticRouter: LLM-as-judge routing layer
        coord._agentic_router = None
        try:
            from distllm.core.agentic_router import AgenticRouter
            router_models = [{"name": coord.model_name, "quantization": "int4"}]
            coord._agentic_router = AgenticRouter(available_models=router_models)
            logger.info("Agentic router initialized")
        except Exception as e:
            logger.debug("Agentic router not available: {}", e)

        # Autoscaler needs special handling (ScalingMetrics init)
        autoscaler = self._start_subsystem(
            "autoscaler", "distllm.core.intelligent_autoscaler",
            "IntelligentAutoscaler", "_autoscaler",
            constructor_kwargs={"min_nodes": 1, "max_nodes": 20, "target_utilization": 0.7},
        )
        if autoscaler is not None:
            # F-014: feed real metrics and actuate periodically instead of a
            # single startup record_metrics call that was never evaluated.
            coord._start_autoscaler_loop()

        # Startup self-check: log which subsystems are active vs degraded/missing
        failed_subsystems = [
            name for name, info in coord._subsystem_health.items()
            if info["status"] == "failed"
        ]
        missing_subsystems = [
            name for name, info in coord._subsystem_health.items()
            if info["status"] == "missing_deps"
        ]
        active_subsystems = [
            name for name, info in coord._subsystem_health.items()
            if info["status"] == "ok"
        ]
        logger.info(
            f"Startup subsystem check: "
            f"{len(active_subsystems)} active, "
            f"{len(missing_subsystems)} missing deps, "
            f"{len(failed_subsystems)} failed"
        )
        if active_subsystems:
            logger.info(f"  Active: {', '.join(sorted(active_subsystems))}")
        if missing_subsystems:
            logger.info(f"  Missing deps (optional): {', '.join(sorted(missing_subsystems))}")
        if failed_subsystems:
            logger.warning(f"  FAILED (degraded): {', '.join(sorted(failed_subsystems))}")

        logger.info(f"Coordinator started on port {coord.port} "
                     f"(health check every {health_check_interval_s}s)")
        if blocking:
            try:
                coord._running.wait()
            except KeyboardInterrupt:
                logger.info("Coordinator shutting down...")
                self.stop()
        else:
            async def _wait_and_callback_async() -> None:
                try:
                    await coord._async_shutdown.wait()
                except Exception:
                    pass
                finally:
                    if on_stop:
                        on_stop()
            # CRIT-003 fix: Check for a running event loop before using ensure_future
            try:
                _loop = asyncio.get_running_loop()
            except RuntimeError:
                _loop = None
            if _loop is not None and _loop.is_running():
                asyncio.ensure_future(_wait_and_callback_async())
            else:
                # No running loop — run callback in a background thread
                def _run_shutdown_callback() -> None:
                    import asyncio as _asyncio
                    __loop = _asyncio.new_event_loop()
                    _asyncio.set_event_loop(__loop)
                    __loop.run_until_complete(_wait_and_callback_async())
                    __loop.close()
                t = threading.Thread(
                    target=_run_shutdown_callback,
                    daemon=True,
                    name="shutdown-callback",
                )
                t.start()

    # ── Stop ──

    def stop(self, timeout: float = 30.0) -> None:
        """Graceful shutdown with in-flight request draining.

        Args:
            timeout: Maximum seconds to wait for in-flight requests to complete.

        Shutdown sequence:
        1. Set _shutting_down flag (rejects new requests)
        2. Cordon all nodes (mark as draining — stop sending new work)
        3. Stop accepting new requests (clear running flag)
        4. Wait for in-flight requests to complete (with timeout)
        5. Checkpoint all active sequences
        6. Release GPU memory
        7. Close gRPC connections
        8. Save state to disk
        9. Stop background threads
        """
        coord = self.coordinator

        logger.info("Initiating graceful shutdown...")
        coord._shutting_down = True

        # 1. Cordon all nodes — mark as draining so scheduler stops sending work
        for nid in list(coord.nodes.keys()):
            try:
                if hasattr(coord._resource_mgr, 'mark_node_draining'):
                    coord._resource_mgr.mark_node_draining(nid)
                logger.debug(f"Cordoned node {nid}")
            except Exception as e:
                logger.warning(f"Failed to cordon node {nid}: {e}")

        # 2. Stop accepting new requests
        coord._running.clear()
        coord._health_event.set()
        coord._async_shutdown.set()

        # 2. Wait for in-flight requests to complete
        if coord._batch_scheduler is not None:
            active_count = coord._batch_scheduler.active_count
            if active_count > 0:
                logger.info(f"Waiting for {active_count} in-flight requests to complete (timeout={timeout}s)...")
                deadline = time.time() + timeout
                remaining = active_count
                while remaining > 0 and time.time() < deadline:
                    time.sleep(0.25)
                    remaining = coord._batch_scheduler.active_count
                if remaining > 0:
                    logger.warning(f"{remaining} requests still in-flight after timeout, forcing shutdown")

        # 3. Checkpoint active sequences
        if coord._batch_scheduler is not None and coord._batch_scheduler.active_count > 0:
            active_snapshot = coord._batch_scheduler.snapshot_active()
            for req_id, seq in active_snapshot:
                try:
                    if coord._recovery_manager is not None:
                        coord._recovery_manager.save_checkpoint(
                            request_id=req_id,
                            kv_cache=None,
                            prompt_tokens=getattr(seq, 'prompt_tokens', []),
                            generated_tokens=getattr(seq, 'generated_tokens', []),
                            node_id="coordinator",
                        )
                except Exception as e:
                    logger.warning(f"Failed to checkpoint {req_id}: {e}")

        # 4. Release GPU memory
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU memory released")
        except ImportError:
            pass

        # 5. Close gRPC connections and nodes
        for nid, node in list(coord.nodes.items()):
            try:
                node.close()
            except Exception as e:
                logger.warning(f"Error closing node {nid}: {e}")

        # 6. Close connection pools and thread pools
        if hasattr(coord, '_resource_mgr') and coord._resource_mgr is not None:
            try:
                coord._resource_mgr._health_check_pool.shutdown(wait=False)
            except Exception as e:
                logger.warning(f"Error shutting down health check pool: {e}")
        if hasattr(coord, '_resource_mgr') and coord._resource_mgr is not None:
            try:
                coord._resource_mgr._conn_pool.close_all()
            except Exception as e:
                logger.warning(f"Error closing sync connection pool: {e}")
            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop is not None and loop.is_running():
                    asyncio.ensure_future(coord._resource_mgr._async_conn_pool.close_all())
                else:
                    # No running loop — close synchronously in a new event loop
                    asyncio.run(coord._resource_mgr._async_conn_pool.close_all())
            except Exception as e:
                logger.warning(f"Error closing async connection pool: {e}")

        # 7. Stop background services
        coord._stop_autoscaler_loop()
        coord._health_mgr.stop()
        if hasattr(coord, '_adaptive_compression_mgr') and coord._adaptive_compression_mgr:
            coord._adaptive_compression_mgr.stop()
        if coord._defrag_task is not None:
            coord._defrag_task.cancel()
            coord._defrag_task = None
            logger.debug("Defrag background loop stopped")
        if hasattr(coord, '_federation') and coord._federation:
            coord._federation.stop()
        if hasattr(coord, '_discovery') and coord._discovery:
            coord._discovery.stop()

        # 8. Shutdown pipeline
        coord._pipeline.shutdown()

        # 9. Save state to disk
        try:
            self._save_shutdown_state()
        except Exception as e:
            logger.warning(f"Failed to save shutdown state: {e}")

        logger.info("Graceful shutdown complete")

    def _save_shutdown_state(self) -> None:
        """Save coordinator state to disk for recovery after restart."""
        import json
        # Only persist safe, JSON-serialisable fields.  Avoid ``default=str``
        # which both leaks sensitive data through string coercion and produces
        # write-only state files that cannot be reliably deserialised.
        coord = self.coordinator
        state: dict[str, Any] = {
            "model_name": coord.model_name,
            "shutdown_time": time.time(),
            "nodes": {
                nid: {"host": n.host, "port": n.port, "healthy": n.healthy}
                for nid, n in coord.nodes.items()
            },
        }
        # H-17: Write to a protected path — use data dir with restricted perms
        state_dir = os.environ.get("DISTLLM_DATA_DIR", os.path.expanduser("~/.distllm"))
        os.makedirs(state_dir, exist_ok=True)
        state_path = os.path.join(state_dir, "shutdown_state.json")
        fd = os.open(state_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        logger.debug(f"Shutdown state saved to {state_path}")
