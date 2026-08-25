"""Wide-area pipeline for distributed LLM inference over high-latency links.

Extends PipelineOrchestrator with:
1. Async execution — non-blocking pipeline with asyncio
2. Token accumulation — batch multiple decode steps per WAN traversal
3. Adaptive batching — adjust batch size based on measured RTT
4. Graceful degradation — fallback to local-only on WAN failures
"""


from __future__ import annotations
import asyncio
import time
import statistics
import threading
from typing import TYPE_CHECKING, Any, Callable

import torch
from loguru import logger

from distllm.config.loader import NodeRole
from distllm.core.kv_cache import KVCache
from distllm.dist.pipeline import PipelineOrchestrator
from distllm.dist import node_pb2
from distllm.dist.pipeline.serialization import (
    process_forward_response_pb,
    set_kv_cache_proto,
    to_proto_tensor,
)
from distllm.dist.config import WideAreaConfig

if TYPE_CHECKING:
    from distllm.core.resource_manager import NodeRegistration, ResourceManager
from distllm.errors.types import (
    InputValidationError,
    NodeUnreachableError,
    OOMError,
)


class _NullResourceManager:
    """No-op stand-in when no ResourceManager is configured.

    Matches the subset of ResourceManager used by
    ``process_forward_response_pb`` so the shared helper works unchanged
    for pipelines constructed without a manager (parent guards every use
    with ``if self._resource_mgr:``).
    """

    def record_success(self, node_id: str) -> None:  # noqa: D102
        pass

    def record_failure(self, node_id: str) -> None:  # noqa: D102
        pass

    def check_circuit_breaker(self, node_id: str) -> bool:  # noqa: D102
        return False


_null_resource_mgr = _NullResourceManager()


class WideAreaPipeline(PipelineOrchestrator):
    """Wide-area pipeline for high-latency distributed inference.


    Extends PipelineOrchestrator with:
    - Token accumulation: batch N decode steps per WAN traversal
    - Adaptive batching: adjusts accumulation window based on measured RTT
    - WAN timeouts: per-link timeout config (default 120s vs 30s LAN)
    - QUIC transport: UDP-based alternative to gRPC for WAN links
    - Graceful degradation: falls back to local-only on persistent WAN failure
    """


    def __init__(
        self,
        resource_mgr=None,
        total_layers: int = 0,
        wan_config: WideAreaConfig | None = None,
        quic_transport: Any | None = None,
        latency_tracker: Any | None = None,
    ):
        super().__init__(resource_mgr=resource_mgr)
        self.total_layers = total_layers
        # Wire the latency tracker so _calibrate_decode_ms() uses real
        # measurements instead of the hardcoded 50 ms default.
        if latency_tracker is not None:
            self._latency_tracker = latency_tracker
        self.wan = wan_config or WideAreaConfig()
        self._quic_transport = quic_transport

        self._link_latencies: dict[tuple[str, str], list[float]] = {}
        self._measuring_lock = threading.Lock()

        # Topology snapshot lock for the async WAN path.  Distinct from the
        # parent's ``_lock`` so run_pipeline_async_p2p can take a consistent
        # node/order snapshot without contending with sync-path mutations;
        # RLock because _discovery_loop registers nodes while holding it and
        # register_node itself acquires the parent lock.
        self._topology_lock = threading.RLock()

        self._current_window = self.wan.accumulation_window
        self._last_adjustment = time.time()

        self._auto_discovery_running = False
        self._discovery_lock = threading.Lock()

        # Auto-initialize QUIC transport for WAN if configured
        if self._quic_transport is None and self.wan.enabled:
            self._auto_init_quic()

    @property
    def resource_mgr(self) -> Any:
        """ResourceManager for circuit-breaker state.

        The parent stores it as ``_resource_mgr`` (private); the WAN path
        reads circuit-breaker state per hop, so expose a read/write view
        (tests assign ``pipeline.resource_mgr = rm`` to swap managers).
        """
        return self._resource_mgr

    @resource_mgr.setter
    def resource_mgr(self, value: Any) -> None:
        self._resource_mgr = value

    def _auto_init_quic(self) -> None:
        """Auto-initialize QUIC transport based on WAN config.


        When transport is "auto", selects QUIC if aioquic is available.
        When transport is "quic", fails if aioquic is not installed.
        When transport is "grpc", skips QUIC initialization.
        """

        transport = getattr(self.wan, "transport", "auto")

        if transport == "grpc":
            logger.info("WAN transport: gRPC (explicitly configured)")
            return

        try:
            from distllm.dist.quic_transport import is_quic_available, QuicTransportClient

            if transport == "quic" and not is_quic_available():
                raise RuntimeError(
                    "QUIC transport requested but aioquic is not installed. "
                    "Install with: pip install aioquic"
                )

            if is_quic_available():
                self._quic_transport = QuicTransportClient()
                logger.info(
                    "WAN transport: QUIC (0-RTT, no head-of-line blocking, "
                    "better packet-loss recovery)"
                )
            else:
                logger.info("WAN transport: gRPC (aioquic not available, install for QUIC)")
        except ImportError:
            if transport == "quic":
                raise
            logger.info("WAN transport: gRPC (quic_transport module not available)")

    @property
    def quic_transport(self) -> Any | None:
        return self._quic_transport

    @quic_transport.setter
    def quic_transport(self, qt: Any | None) -> None:
        self._quic_transport = qt

    def _use_quic(self) -> bool:
        """Whether a usable QUIC transport is attached for this hop.

        Injected doubles may expose an ``is_available`` flag; the real
        :class:`QuicTransportClient` reports connectivity via
        ``is_connected``.
        """
        qt = self._quic_transport
        if qt is None:
            return False
        if isinstance(qt, str):  # test double sentinel values
            return False
        available = getattr(qt, "is_available", None)
        if available is not None:
            return bool(available)
        return bool(getattr(qt, "is_connected", True))

    def get_estimated_link_rtt_ms(self, from_node: str, to_node: str) -> float:
        key = (from_node, to_node)
        with self._measuring_lock:
            samples = self._link_latencies.get(key, [])
        if not samples:
            return -1.0
        return statistics.median(samples)

    def _adjust_accumulation_window(self) -> int:
        if not self.wan.adaptive_batching:
            return self._current_window

        now = time.time()
        if now - self._last_adjustment < self.wan.latency_sample_interval:
            return self._current_window

        # H-04: use the WAN topology lock (defined in __init__ above);
        # fall back to the parent's _lock for any legacy subclass.
        lock = getattr(self, '_topology_lock', None) or getattr(self, '_lock', None)
        if lock is None:
            node_ids = list(self.node_order)
        else:
            with lock:
                node_ids = list(self.node_order)

        max_rtt = 0.0
        for i in range(len(node_ids) - 1):
            rtt = self.get_estimated_link_rtt_ms(node_ids[i], node_ids[i + 1])
            if rtt > max_rtt:
                max_rtt = rtt

        self._last_adjustment = now

        if max_rtt <= 0:
            return self._current_window

        # Adaptive decode_ms_per_token: calibrate from recent measurements
        decode_ms_per_token = self._calibrate_decode_ms()
        optimal = max(1, int(max_rtt / decode_ms_per_token))
        self._current_window = min(optimal, self.wan.accumulation_window)
        logger.debug(
            f"WAN adaptive window: RTT={max_rtt:.0f}ms, "
            f"decode_ms={decode_ms_per_token:.1f}ms, window={self._current_window}"
        )
        return self._current_window

    def _calibrate_decode_ms(self) -> float:
        """Calibrate decode_ms_per_token from recent latency measurements.


        Uses the latency tracker's recent data if available, otherwise
        falls back to the default estimate.
        """

        default = 50.0
        if not hasattr(self, '_latency_tracker') or self._latency_tracker is None:
            return default

        try:
            recent = self._latency_tracker.get_recent_latencies(limit=20)
            if recent and len(recent) >= 5:
                # Use median of recent decode latencies
                import statistics
                return max(1.0, statistics.median(recent))
        except Exception:
            pass
        return default

    async def run_pipeline_async_p2p(
        self,
        input_ids: torch.Tensor,
        node_kv_caches: dict[str, list | None],
        request_id: str,
        draft_tokens: list[int] | None = None,
        is_prefill: bool = False,
    ) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]

        with self._topology_lock:
            node_order_snapshot = list(self.node_order)
            nodes_snapshot = dict(self.nodes)
        total_nodes = len(node_order_snapshot)

        if total_nodes == 0:
            raise RuntimeError("No nodes registered in WAN pipeline")

        for node_id in node_order_snapshot:
            node = nodes_snapshot[node_id]
            rm = self.resource_mgr
            if rm is not None and rm.check_circuit_breaker(node_id):
                fallback = self._find_fallback_node(node_id, node)
                if fallback is not None:
                    continue
                if self.wan.fallback_to_local:
                    logger.warning(f"WAN pipeline: node {node_id} CB open, falling back to local")
                    return await self._run_local_fallback(input_ids, request_id)
                raise NodeUnreachableError(
                    node_id=node_id, host=node.host, port=node.port,
                    original_error=Exception(f"Circuit breaker open for {node_id}"),
                )

        current_hidden: torch.Tensor | None = None
        timeout = self.wan.wan_timeout_seconds

        for i, node_id in enumerate(node_order_snapshot):
            node = nodes_snapshot[node_id]
            # Route around hops whose circuit breaker opened between the
            # pre-flight check and execution.  The pre-flight loop above
            # guarantees a covering fallback exists here (otherwise it
            # already raised or fell back to local).
            rm = self.resource_mgr
            if rm is not None and rm.check_circuit_breaker(node_id):
                fallback = self._find_fallback_node(node_id, node)
                if fallback is None:
                    raise NodeUnreachableError(
                        node_id=node_id, host=node.host, port=node.port,
                        original_error=Exception(f"Circuit breaker open for {node_id}"),
                    )
                logger.info(
                    f"WAN pipeline: rerouting stage for {node_id} "
                    f"through fallback {fallback.node_id}"
                )
                node = fallback
            past_kv = node_kv_caches.get(node_id)
            is_first = (i == 0)
            is_last = (i == total_nodes - 1)

            current_hidden = await self._async_execute_node(
                node_id, node, node_kv_caches,
                is_first, is_last, seq_len, batch_size,
                current_hidden, request_id, draft_tokens,
                input_ids, timeout,
            )

        return current_hidden

    def _find_fallback_node(
        self, node_id: str, node: "NodeRegistration",
    ) -> "NodeRegistration | None":
        """Find a healthy node whose layer range fully covers ``node``'s.

        Used when a hop's circuit breaker is open: if a redundant copy of
        the same layers exists on another healthy node, the pipeline can
        route around the failed stage.  Returns None when no substitute
        exists (the caller then degrades to local inference or raises).
        """
        if self._resource_mgr is None:
            return None
        with self._topology_lock:
            candidates = [
                n for nid, n in self._nodes.items()
                if nid != node_id
                and getattr(n, "is_healthy", True)
                and not (
                    hasattr(self._resource_mgr, "check_circuit_breaker")
                    and self._resource_mgr.check_circuit_breaker(nid)
                )
                and n.start_layer <= node.start_layer
                and n.end_layer >= node.end_layer
            ]
        if not candidates:
            return None
        # Prefer the candidate with the tightest fit (smallest span), then
        # lowest measured latency.
        return min(
            candidates,
            key=lambda n: ((n.end_layer - n.start_layer), n.latency_ms),
        )

    async def _async_execute_node(
        self,
        node_id: str,
        node: "NodeRegistration",
        node_kv_caches: dict[str, list | None],
        is_first: bool,
        is_last: bool,
        seq_len: int,
        batch_size: int,
        current_hidden: torch.Tensor | None,
        request_id: str,
        draft_tokens: list[int] | None,
        input_ids: torch.Tensor,
        timeout: float,
    ) -> torch.Tensor:
        past_kv = node_kv_caches.get(node_id)
        request = self._prepare_forward_request(
            node_id, node, is_first, is_last,
            seq_len, batch_size, current_hidden, past_kv,
            request_id, draft_tokens, input_ids,
        )

        try:
            # QUIC transport (alternative to gRPC for WAN links)
            if self._use_quic():
                qt = self._quic_transport
                serialized_data = request.SerializeToString()
                # Injected transports expose send_forward_pass(); the real
                # QuicTransportClient names it forward_pass().
                send = getattr(qt, "send_forward_pass", None) or qt.forward_pass
                response_data = await asyncio.wait_for(
                    send(serialized_data),
                    timeout=timeout,
                )
                response_pb = node_pb2.ForwardPassResponse()
                response_pb.ParseFromString(response_data)
                return self._process_forward_response(response_pb, node_id, node, node_kv_caches)

            # gRPC (standard path)
            if hasattr(node, 'async_client') and node.async_client is not None:
                response = await asyncio.wait_for(
                    node.async_client.stub.ForwardPass(request),
                    timeout=timeout,
                )
            else:
                response = await asyncio.wait_for(
                    asyncio.to_thread(node.client.stub.ForwardPass, request),
                    timeout=timeout,
                )
            return self._process_forward_response(response, node_id, node, node_kv_caches)
        except asyncio.TimeoutError as e:
            if self.resource_mgr is not None:
                self.resource_mgr.record_failure(node_id)
            # PipelineNode's health field is ``is_healthy``; setting
            # ``healthy`` would create a dead attribute that no routing
            # filter reads.
            node.is_healthy = False
            node_kv_caches.pop(node_id, None)
            raise NodeUnreachableError(
                node_id=node_id, host=node.host, port=node.port, original_error=e
            ) from e
        except (NodeUnreachableError, OOMError, InputValidationError):
            node_kv_caches.pop(node_id, None)
            raise

    def _prepare_forward_request(
        self,
        node_id: str,
        node: "NodeRegistration",
        is_first: bool,
        is_last: bool,
        seq_len: int,
        batch_size: int,
        current_hidden: torch.Tensor | None,
        past_kv: list | None,
        request_id: str,
        draft_tokens: list[int] | None,
        input_ids: torch.Tensor,
    ):
        """Build the ForwardPassRequest proto for one WAN hop.

        Mirrors the sync path's wire format (node_client.forward_request):
        first hop carries input_ids, later hops carry hidden_states; KV
        cache rides along when present so the worker can use_cache.

        Raises:
            InputValidationError: On a non-first hop without hidden states
                (the pipeline chain is broken) or a malformed shape.
        """
        request = node_pb2.ForwardPassRequest(
            request_id=request_id,
            batch_size=batch_size,
            seq_len=seq_len,
            use_cache=past_kv is not None,
            is_first_pass=is_first,
            is_last_pass=is_last,
        )

        if is_first:
            # First hop sends token ids flattened to 1-D per proto spec.
            ids = input_ids.detach().to("cpu").reshape(-1).tolist()
            if not ids:
                raise InputValidationError(
                    "input_ids must be non-empty", field="input_ids",
                )
            request.input_ids.extend(int(t) for t in ids)
        else:
            if current_hidden is None:
                raise InputValidationError(
                    f"pipeline chain broken at node {node_id}: "
                    "no hidden states from previous hop",
                    field="hidden_states",
                )
            if current_hidden.dim() < 2:
                raise InputValidationError(
                    f"hidden_states must be at least 2-D, got shape "
                    f"{tuple(current_hidden.shape)}",
                    field="hidden_states",
                )
            request.hidden_states.CopyFrom(to_proto_tensor(current_hidden))

        if past_kv:
            set_kv_cache_proto(request.kv_cache, past_kv)

        if draft_tokens:
            request.draft_tokens.extend(int(t) for t in draft_tokens)

        logger.debug(
            f"WAN forward request for {node_id}: first={is_first} "
            f"last={is_last} seq_len={seq_len} batch_size={batch_size} "
            f"kv_layers={len(past_kv) if past_kv else 0}"
        )
        return request

    def _process_forward_response(
        self,
        response,
        node_id: str,
        node: "NodeRegistration",
        node_kv_caches: dict[str, list | None],
    ) -> torch.Tensor:
        """Handle a ForwardPassResponse from one WAN hop.

        Delegates to the shared serialization helper
        (``process_forward_response_pb``) so error handling, circuit
        breaker accounting, and KV-cache extraction match the sync
        pipeline exactly.  Also flips the real ``is_healthy`` field —
        the shared helper writes ``node.healthy``, which no routing
        filter on PipelineNode reads.
        """
        try:
            return process_forward_response_pb(
                response, node_id, node, node_kv_caches,
                _null_resource_mgr if self.resource_mgr is None else self.resource_mgr,
            )
        except NodeUnreachableError:
            # process_forward_response_pb sets ``node.healthy = False``
            # (a dead attribute on PipelineNode); mirror onto is_healthy.
            node.is_healthy = False
            raise

    async def _run_local_fallback(
        self, input_ids: torch.Tensor, request_id: str
    ) -> torch.Tensor:
        local_model = getattr(self, '_local_model', None)
        local_tokenizer = getattr(self, '_local_tokenizer', None)
        if local_model is None:
            raise RuntimeError(
                "WAN fallback requested but no local model available. "
                "Set fallback_to_local=False or load a local model."
            )
        import torch
        with torch.no_grad():
            outputs = local_model(
                input_ids=input_ids,
                use_cache=True,
            )
        logger.info(f"WAN fallback: ran {request_id} locally")
        return outputs.logits

    def set_local_fallback_model(self, model, tokenizer) -> None:
        self._local_model = model
        self._local_tokenizer = tokenizer

    async def run_pipeline_accumulated(
        self,
        input_ids: torch.Tensor,
        node_kv_caches: dict[str, list | None],
        request_id: str,
        draft_model_fn: Callable | None = None,
        prefill_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if draft_model_fn is None or not self.wan.token_accumulation:
            return await self.run_pipeline_async_p2p(
                input_ids, node_kv_caches, request_id,
            )

        window = self._adjust_accumulation_window()
        total_tokens = input_ids.shape[1]

        draft_ids = draft_model_fn(prefill_logits, window) if prefill_logits is not None else None
        if draft_ids is None or len(draft_ids) == 0:
            return await self.run_pipeline_async_p2p(
                input_ids, node_kv_caches, request_id,
            )

        draft_tensor = torch.tensor(
            [draft_ids], dtype=input_ids.dtype, device=input_ids.device
        )
        accumulated_input = torch.cat([input_ids, draft_tensor], dim=1)

        logits = await self.run_pipeline_async_p2p(
            accumulated_input, node_kv_caches, request_id,
        )

        return logits

    async def run_pipeline_speculative(
        self,
        input_ids: torch.Tensor,
        node_kv_caches: dict[str, list | None],
        request_id: str,
        draft_model_fn: Callable,
        num_candidates: int = 8,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """WAN speculative decoding: draft locally, verify remotely.


        Generates N draft tokens locally using a fast draft model,
        then sends all candidates across WAN for batch verification.
        Accepts tokens that match the target distribution, re-drafts
        from the first mismatch.

        This reduces WAN round-trips from O(max_new_tokens) to
        O(max_new_tokens / acceptance_rate * num_candidates).

        Args:
            input_ids: Prompt token IDs.
            node_kv_caches: Per-node KV cache state.
            request_id: Request identifier.
            draft_model_fn: Local draft model function.
            num_candidates: Draft tokens per WAN round-trip.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated token IDs including the prompt.
        """

        from distllm.dist.wan_speculative import WANSpeculativeDecoder

        # Target forward: runs across WAN via pipeline
        async def target_forward(tokens, **kwargs):
            return await self.run_pipeline_async_p2p(
                tokens, node_kv_caches, request_id,
            )

        # Draft forward: runs locally (fast)
        def draft_forward(prefix, num_tokens, **kwargs):
            return draft_model_fn(prefix, num_tokens)

        decoder = WANSpeculativeDecoder(
            target_forward=target_forward,
            draft_forward=draft_forward,
            num_candidates=num_candidates,
            temperature=temperature,
        )

        result = await decoder.generate(input_ids, max_new_tokens=max_new_tokens)

        stats = decoder.stats
        if stats["wan_rounds"] > 0:
            logger.info(
                f"WAN speculative decoding: "
                f"{stats['tokens_accepted']} accepted, "
                f"{stats['tokens_rejected']} rejected, "
                f"acceptance_rate={stats['acceptance_rate']:.2f}, "
                f"wan_speedup={stats['wan_speedup']:.1f}x, "
                f"wan_rounds={stats['wan_rounds']}"
            )

        return result

    def start_auto_discovery(self, discovery_port: int = 5353, service_type: str = "_distllm._tcp") -> None:
        if self._auto_discovery_running:
            return
        self._auto_discovery_running = True
        thread = threading.Thread(
            target=self._discovery_loop,
            args=(service_type, discovery_port),
            daemon=True,
            name="wan-auto-discovery",
        )
        thread.start()
        logger.info(f"WAN auto-discovery started for {service_type}")

    def _discovery_loop(self, service_type: str, port: int) -> None:
        try:
            import socket as _socket
        except ImportError:
            logger.warning("Cannot start auto-discovery: socket module unavailable")
            self._auto_discovery_running = False
            return

        while self._auto_discovery_running:
            try:
                discovered = self._discover_nodes(service_type, port)
                for host, svc_port, layers, node_id in discovered:
                    with self._topology_lock:
                        if node_id not in self.nodes:
                            logger.info(f"Auto-discovered node {node_id} at {host}:{svc_port}")
                            self.register_node(
                                node_id=node_id, host=host, port=svc_port,
                                start_layer=layers[0], end_layer=layers[1],
                            )
            except Exception as e:
                logger.debug(f"Auto-discovery cycle error: {e}")
            time.sleep(self.wan.heartbeat_interval_seconds)

    @staticmethod
    def _discover_nodes(service_type: str, port: int) -> list[tuple[str, int, tuple[int, int], str]]:
        """Discover coordinator nodes on the LAN via mDNS/zeroconf.


        Returns:
            List of (host, port, (start_layer, end_layer), node_id) tuples.
        """

        try:
            from distllm.dist.discovery import DiscoveryClient
            client = DiscoveryClient(timeout=3.0)
            found = client.discover()
            results = []
            for svc in found:
                host = svc.get("host", "")
                svc_port = svc.get("port", port)
                name = svc.get("name", "unknown")
                props = svc.get("properties", {})
                model = props.get("model", "")
                if host:
                    results.append((host, svc_port, (0, 0), name))
            if results:
                logger.info(f"Discovered {len(results)} nodes via mDNS: {[r[0] for r in results]}")
            return results
        except ImportError:
            logger.debug("zeroconf not installed, mDNS discovery unavailable")
            return []
        except Exception as e:
            logger.debug(f"mDNS discovery failed: {e}")
            return []

    def get_measured_latency(self, from_node: str, to_node: str) -> float | None:
        key = (from_node, to_node)
        with self._measuring_lock:
            samples = self._link_latencies.get(key)
            if not samples:
                key = (to_node, from_node)
                samples = self._link_latencies.get(key)
            if samples:
                return statistics.median(samples[-10:])
            return None

    def get_latency_stats(self) -> dict:
        stats = {}
        with self._measuring_lock:
            for (a, b), samples in self._link_latencies.items():
                if len(samples) > 0:
                    stats[f"{a}→{b}"] = {
                        "median_ms": round(statistics.median(samples[-20:]), 1),
                        "min_ms": round(min(samples[-20:]), 1),
                        "max_ms": round(max(samples[-20:]), 1),
                        "samples": len(samples),
                    }
        return stats

    def shutdown(self) -> None:
        super().shutdown()
