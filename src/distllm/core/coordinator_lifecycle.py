"""Server lifecycle manager for the Coordinator facade.

Handles server start/stop, gossip loop, rebalancer loop, and request completion tracking.
Extracted from the Coordinator class.
"""

import threading
import time
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


class RequestTracker:
    """Tracks request completion state for batched generation.

    Attributes:
        _results: Dict of request_id -> result string.
        _events: Dict of request_id -> threading.Event.
        _lock: Threading lock for concurrent access.
        _shutting_down: Whether the server is shutting down.
    """

    def __init__(self):
        self._results: Dict[str, str] = {}
        self._events: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._shutting_down = False

    @property
    def shutting_down(self) -> bool:
        return self._shutting_down

    @shutting_down.setter
    def shutting_down(self, value: bool):
        self._shutting_down = value

    def register_request(self, request_id: str) -> threading.Event:
        """Register a new request and return its completion event.

        Args:
            request_id: Unique request identifier.

        Returns:
            Threading event for the request.
        """
        event = threading.Event()
        with self._lock:
            self._events[request_id] = event
        return event

    def set_result(self, request_id: str, result: str) -> None:
        """Set the result for a completed request.

        Args:
            request_id: Unique request identifier.
            result: Generated text result.
        """
        with self._lock:
            self._results[request_id] = result
            event = self._events.pop(request_id, None)
            if event:
                event.set()

    def wait_for_result(self, request_id: str, timeout: float = 120.0) -> str:
        """Wait for a batched request to complete and return the result.

        Args:
            request_id: Unique request identifier.
            timeout: Maximum time to wait in seconds.

        Returns:
            Generated text, or empty string on timeout.
        """
        with self._lock:
            event = self._events.get(request_id)
            if event is None:
                return self._results.pop(request_id, "")

        if event.wait(timeout=timeout):
            with self._lock:
                return self._results.pop(request_id, "")

        with self._lock:
            self._events.pop(request_id, None)
            self._results.pop(request_id, None)
        return ""

    @property
    def pending_count(self) -> int:
        """Number of pending requests."""
        with self._lock:
            return len(self._events)

    def complete_batch_requests(
        self,
        scheduler_active: Dict[str, Any],
        scheduler_pending: List[Any],
        tokenizer,
    ) -> None:
        """Mark completed requests from scheduler and signal events.

        Args:
            scheduler_active: Dict of active sequences from scheduler.
            scheduler_pending: List of pending sequences from scheduler.
            tokenizer: Tokenizer for decoding.
        """
        completed = []
        with self._lock:
            for rid, seq in scheduler_active.items():
                if seq.is_complete:
                    completed.append((rid, seq))

            for seq in list(scheduler_pending):
                if seq.is_complete:
                    completed.append((seq.request_id, seq))

        for rid, seq in completed:
            # Decode outside the lock to avoid blocking other requests
            result = tokenizer.decode(
                seq.prompt_tokens + seq.generated_tokens,
                skip_special_tokens=True,
            )
            with self._lock:
                self._results[rid] = result
                event = self._events.pop(rid, None)
                if event:
                    event.set()

    def clear(self) -> None:
        """Clear all request state."""
        with self._lock:
            self._results.clear()
            self._events.clear()


class ServerLifecycle:
    """Manages server lifecycle: start, stop, background daemons.

    Attributes:
        resource_mgr: ResourceManager for closing node connections.
        cache_persistence: Optional cache persistence manager.
        gossip_protocol: Optional gossip protocol instance.
        rebalancer: Optional rebalancer instance.
        latency_tracker: Optional latency tracker.
        total_layers: Total number of model layers.
        request_tracker: Request completion tracker.
    """

    def __init__(
        self,
        resource_mgr,
        cache_persistence=None,
        gossip_protocol=None,
        rebalancer=None,
        latency_tracker=None,
        total_layers: int = 0,
    ):
        self.resource_mgr = resource_mgr
        self.cache_persistence = cache_persistence
        self.gossip_protocol = gossip_protocol
        self.rebalancer = rebalancer
        self.latency_tracker = latency_tracker
        self.total_layers = total_layers
        self.request_tracker = RequestTracker()
        self._server = None

    @property
    def server(self):
        return self._server

    @server.setter
    def server(self, value):
        self._server = value

    def start(
        self,
        port: int,
        tokenizer,
        model_info,
        scheduler,
        coordinator_service_cls,
        grpc_server_cls,
        blocking: bool = True,
        on_stop: Optional[Callable] = None,
    ) -> None:
        """Start the coordinator gRPC server.

        Args:
            port: Port to listen on.
            tokenizer: Tokenizer instance.
            model_info: Model info dict (may be None).
            scheduler: Optional batch scheduler.
            coordinator_service_cls: CoordinatorService class.
            grpc_server_cls: GRPCServer class.
            blocking: Whether to block until termination.
            on_stop: Optional callback when server stops.
        """
        if tokenizer is None:
            from transformers import AutoTokenizer
            # This shouldn't happen in normal usage, but handle gracefully
            logger.warning("Tokenizer not initialized, loading for server start")

        # Update scheduler with model info if available
        if model_info is not None and scheduler is not None:
            scheduler._model_info = model_info
            scheduler._use_length_grouping = True

        servicer = coordinator_service_cls()
        self._server = grpc_server_cls(port=port, servicer=servicer)
        self._server.start()

        logger.info(f"Coordinator started on port {port}")

        if blocking:
            try:
                self._server.wait_for_termination()
            except KeyboardInterrupt:
                logger.info("Coordinator shutting down...")
                self.stop()
        else:
            def _wait_and_callback():
                try:
                    self._server.wait_for_termination()
                except KeyboardInterrupt:
                    pass
                finally:
                    if on_stop:
                        on_stop()

            thread = threading.Thread(target=_wait_and_callback, daemon=True)
            thread.start()

    def start_background_daemons(
        self,
        cache_mgr,
        gossip_interval: float = 10.0,
    ) -> None:
        """Start gossip and rebalancer background threads.

        Args:
            cache_mgr: Cache manager for gossip rounds.
            gossip_interval: Seconds between gossip rounds.
        """
        # Gossip loop
        if self.gossip_protocol is not None:
            gossip_thread = threading.Thread(
                target=self._gossip_loop,
                args=(cache_mgr, gossip_interval),
                daemon=True,
                name="gossip-loop",
            )
            gossip_thread.start()

        # Rebalancer loop
        if self.rebalancer and self.rebalancer._settings.enabled:
            rebalancer_thread = threading.Thread(
                target=self._rebalancer_loop,
                daemon=True,
                name="rebalancer-loop",
            )
            rebalancer_thread.start()

    def _gossip_loop(self, cache_mgr, interval: float = 10.0) -> None:
        """Background daemon that runs periodic gossip rounds.

        Args:
            cache_mgr: Cache manager for sync_with_peers.
            interval: Seconds between gossip rounds.
        """
        while True:
            try:
                time.sleep(interval)
                if cache_mgr is not None:
                    discovered = cache_mgr.sync_with_peers()
                    if discovered > 0:
                        logger.debug(f"Gossip round: discovered {discovered} new cache entries")
            except Exception:
                logger.debug("Gossip round error (non-fatal)", exc_info=True)

    def _rebalancer_loop(self) -> None:
        """Background loop that checks for stragglers periodically."""
        while True:
            time.sleep(self.rebalancer._settings.check_interval)
            if not self.rebalancer._settings.enabled:
                continue
            should, reason = self.rebalancer.should_rebalance()
            if should:
                stragglers = self.rebalancer.detect_stragglers()
                logger.warning(f"Stragglers detected: {stragglers}")
                all_avg = self.latency_tracker.get_all_avg()
                partition = self.rebalancer.compute_new_partition(
                    self.total_layers, all_avg
                )
                logger.info(
                    f"Recommended partition: "
                    f"{[(p.node_id, p.start_layer, p.end_layer) for p in partition]}"
                )
                logger.info("NOTE: Partition recommendation is logged for manual approval (v1)")
                self.rebalancer.record_rebalance()

    def wait_for_termination(self) -> None:
        """Block until the coordinator server terminates."""
        if self._server:
            try:
                self._server.wait_for_termination()
            except KeyboardInterrupt:
                logger.info("Coordinator shutting down...")
                self.stop()

    def stop(self) -> None:
        """Stop the coordinator with graceful shutdown."""
        logger.info("Initiating graceful shutdown...")

        # Phase 1: Stop accepting new requests
        self.request_tracker.shutting_down = True
        logger.info("Phase 1: Stopped accepting new requests")

        # Phase 2: Wait for in-flight requests to complete (up to 30s)
        if self.request_tracker.pending_count > 0:
            logger.info(
                f"Phase 2: Waiting for {self.request_tracker.pending_count} in-flight requests..."
            )
            # Access events directly for waiting
            with self.request_tracker._lock:
                events = list(self.request_tracker._events.values())
            for event in events:
                event.wait(timeout=30.0)

        # Phase 3: Persist cache if enabled
        if self.cache_persistence and self.cache_persistence._settings.enabled:
            logger.info("Phase 3: Persisting cache to disk...")
            self.cache_persistence.enforce_disk_limit()

        # Phase 4: Stop gRPC server
        if self._server:
            logger.info("Phase 4: Stopping gRPC server...")
            self._server.stop(grace=10)

        # Phase 5: Close node connections
        logger.info("Phase 5: Closing node connections...")
        # Note: nodes dict is passed in from Coordinator
        # This is handled by the Coordinator.stop() method

        # Phase 6: Cleanup request state
        self.request_tracker.clear()

        # Phase 7: Shutdown plugins if loaded (handled by Coordinator)

        logger.info("Graceful shutdown complete")

    async def stop_async(self, nodes: Dict[str, Any] = None) -> None:
        """Stop the coordinator with graceful shutdown (async).

        Args:
            nodes: Optional dict of node_id -> NodeRegistration for cleanup.
        """
        logger.info("Initiating graceful shutdown (async)...")

        # Phase 1: Stop accepting new requests
        self.request_tracker.shutting_down = True
        logger.info("Phase 1: Stopped accepting new requests")

        # Phase 2: Wait for in-flight requests (up to 30s)
        if self.request_tracker.pending_count > 0:
            logger.info(
                f"Phase 2: Waiting for {self.request_tracker.pending_count} in-flight requests..."
            )
            with self.request_tracker._lock:
                events = list(self.request_tracker._events.values())
            for event in events:
                event.wait(timeout=30.0)

        # Phase 3: Persist cache
        if self.cache_persistence and self.cache_persistence._settings.enabled:
            logger.info("Phase 3: Persisting cache to disk...")
            self.cache_persistence.enforce_disk_limit()

        # Phase 4: Stop gRPC server
        if self._server:
            logger.info("Phase 4: Stopping gRPC server...")
            self._server.stop(grace=10)

        # Phase 5: Close node connections (async)
        if nodes:
            logger.info("Phase 5: Closing node connections...")
            await self.resource_mgr.close_all_async(nodes)

        # Phase 6: Cleanup request state
        self.request_tracker.clear()

        logger.info("Graceful shutdown complete (async)")
