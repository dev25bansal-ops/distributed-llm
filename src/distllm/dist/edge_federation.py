"""Edge Inference Federation — connect lightweight devices for small model inference.

Allows mobile devices, browsers (via WebRTC), and IoT devices to participate
in the cluster as lightweight inference nodes. Only small models (1B-3B params)
are routed to edge nodes — larger models still run on the GPU cluster.

Architecture:
    Browser ──WebRTC──┐
    Mobile  ──WebSocket┼── EdgeFederationManager ──► Cluster Coordinator
    IoT     ──MQTT────┘          │
                                 ├── Routes small models (1B-3B) to edge
                                 ├── Falls back to cluster for large models
                                 └── Tracks latency/reliability per device

NAT traversal via WebRTC ICE/STUN/TURN (existing infrastructure).
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger

from distllm.api.routes.webrtc import WebRTCSessionManager


# Maximum parameter count for models that can be routed to edge nodes
# Edge nodes are assumed to have limited compute (phone CPU, browser WASM)
EDGE_MODEL_SIZE_LIMIT = 3_000_000_000  # 3B parameters


@dataclass
class EdgeNodeProfile:
    """Profile of a connected edge device.

    Attributes:
        node_id: Unique identifier for this edge node.
        device_type: Type of device ('mobile', 'browser', 'iot').
        model_name: Model this node is configured to run.
        max_tokens_per_request: Maximum output tokens this node can handle.
        max_batch_size: Maximum batch size for this node.
        avg_latency_ms: Rolling average latency in milliseconds.
        last_seen: Timestamp of last successful health check.
        is_online: Whether the node is currently connected and available.
        transport: Connection transport ('webrtc', 'websocket', 'mqtt').
        session_id: WebRTC or WebSocket session ID.
        total_requests_served: Counter for requests handled.
        total_errors: Counter for errors encountered.
    """
    node_id: str
    device_type: str = "browser"
    model_name: str = ""
    max_tokens_per_request: int = 256
    max_batch_size: int = 1
    avg_latency_ms: float = 0.0
    last_seen: float = field(default_factory=time.time)
    is_online: bool = True
    transport: str = "webrtc"
    session_id: str = ""
    total_requests_served: int = 0
    total_errors: int = 0


class EdgeFederationManager:
    """Manages lightweight edge devices as inference nodes.

    Routes small model inference (1B-3B params) to connected edge devices.
    Falls back to the GPU cluster if no edge nodes are available or if the
    model is too large.

    Thread-safe: uses threading.Lock for node registry access.
    """

    def __init__(
        self,
        coordinator: Any = None,
        model_size_limit: int = EDGE_MODEL_SIZE_LIMIT,
        webrtc_manager: WebRTCSessionManager | None = None,
        node_timeout_seconds: float = 30.0,
    ):
        self._coordinator = coordinator
        self._model_size_limit = model_size_limit
        self._webrtc_manager = webrtc_manager or WebRTCSessionManager()
        self._node_timeout = node_timeout_seconds

        # node_id -> EdgeNodeProfile
        self._nodes: dict[str, EdgeNodeProfile] = {}
        # M-03: Persistent WebRTC connection pool to avoid RTCPeerConnection per call
        self._webrtc_connections: dict[str, Any] = {}
        self._lock = threading.Lock()

        # Stats
        self._requests_routed = 0
        self._requests_failed = 0
        self._total_edge_latency_ms = 0.0

        # Background health check
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the edge federation manager and health check loop."""
        self._running = True
        self._thread = threading.Thread(target=self._health_loop, daemon=True, name="edge-health")
        self._thread.start()
        logger.info("Edge federation manager started")

    def stop(self) -> None:
        """Stop the edge federation manager."""
        self._running = False
        logger.info("Edge federation manager stopped")

    # ── Node Registration ────────────────────────────────────────────────

    def register_node(
        self,
        node_id: str,
        device_type: str = "browser",
        model_name: str = "",
        transport: str = "webrtc",
        session_id: str = "",
    ) -> EdgeNodeProfile:
        """Register or update an edge node.

        Returns the EdgeNodeProfile for the node.
        """
        with self._lock:
            profile = self._nodes.get(node_id)
            if profile is None:
                profile = EdgeNodeProfile(
                    node_id=node_id,
                    device_type=device_type,
                    model_name=model_name,
                    transport=transport,
                    session_id=session_id or f"edge-{uuid.uuid4().hex[:8]}",
                )
                self._nodes[node_id] = profile
                logger.info(f"Edge node registered: {node_id} ({device_type} via {transport})")
            else:
                profile.is_online = True
                profile.last_seen = time.time()
                profile.transport = transport
                if session_id:
                    profile.session_id = session_id
                if model_name:
                    profile.model_name = model_name
        return profile

    def unregister_node(self, node_id: str) -> bool:
        """Remove an edge node from the registry."""
        with self._lock:
            return self._nodes.pop(node_id, None) is not None

    def get_node(self, node_id: str) -> EdgeNodeProfile | None:
        with self._lock:
            return self._nodes.get(node_id)

    def get_online_nodes(self) -> list[EdgeNodeProfile]:
        """Get all currently online edge nodes."""
        now = time.time()
        with self._lock:
            return [
                n for n in self._nodes.values()
                if n.is_online and (now - n.last_seen) < self._node_timeout
            ]

    # ── Inference Routing ────────────────────────────────────────────────

    def should_route_to_edge(self, model_params_b: int) -> bool:
        """Check if a model is small enough for edge inference.

        Args:
            model_params_b: Model size in billions of parameters.

        Returns:
            True if the model can be routed to edge nodes.
        """
        return model_params_b * 1_000_000_000 <= self._model_size_limit

    def route_inference(
        self,
        prompt: str,
        model_name: str,
        max_tokens: int = 256,
        fallback_fn: Callable | None = None,
    ) -> str | None:
        """Route an inference request to the best available edge node.

        Falls back to the cluster (via fallback_fn) if no edge node is
        available or suitable.

        Args:
            prompt: Input prompt.
            model_name: Model to use for inference.
            max_tokens: Maximum output tokens.
            fallback_fn: Callable(prompt, model_name, max_tokens) for fallback.

        Returns:
            Generated text, or None on failure.
        """
        # Find best edge node (lowest latency, online)
        online = self.get_online_nodes()
        if not online:
            if fallback_fn:
                return fallback_fn(prompt, model_name, max_tokens)
            return None

        # Pick lowest-latency node
        best = min(online, key=lambda n: n.avg_latency_ms)

        t0 = time.monotonic()
        try:
            result = self._send_to_edge_node(best, prompt, max_tokens)
            elapsed = (time.monotonic() - t0) * 1000

            with self._lock:
                best.total_requests_served += 1
                best.avg_latency_ms = best.avg_latency_ms * 0.8 + elapsed * 0.2
                best.last_seen = time.time()
                self._requests_routed += 1
                self._total_edge_latency_ms += elapsed

            return result
        except Exception as e:
            logger.warning(f"Edge inference failed on {best.node_id}: {e}")
            with self._lock:
                best.total_errors += 1
                self._requests_failed += 1
                best.is_online = False  # Mark offline on failure

            if fallback_fn:
                return fallback_fn(prompt, model_name, max_tokens)
            return None

    def _send_to_edge_node(
        self, node: EdgeNodeProfile, prompt: str, max_tokens: int,
    ) -> str:
        """Send an inference request to an edge node via its transport.

        Raises on connection/failure — caller handles fallback.
        """
        if node.transport == "webrtc":
            return self._send_via_webrtc(node, prompt, max_tokens)
        elif node.transport == "websocket":
            return self._send_via_websocket(node, prompt, max_tokens)
        else:
            raise ValueError(f"Unsupported edge transport: {node.transport}")

    def _send_via_webrtc(
        self, node: EdgeNodeProfile, prompt: str, max_tokens: int,
    ) -> str:
        """Send inference request via WebRTC data channel.

        Uses aiortc to establish a data channel to the browser/node
        and sends the prompt as a JSON message. The response is
        received asynchronously via the data channel.
        """
        logger.debug(f"WebRTC inference to {node.node_id} (session={node.session_id})")
        try:
            import asyncio
            import json

            # Build the inference request payload
            payload = json.dumps({
                "type": "inference",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "request_id": getattr(self, '_request_id', ''),
            })

            # Try aiortc data channel if available
            try:
                from aiortc import RTCPeerConnection, RTCSessionDescription
                # M-03: Reuse persistent WebRTC connections instead of per-call
                pc = self._webrtc_connections.get(node.node_id)
                if pc is None:
                    pc = RTCPeerConnection()
                    self._webrtc_connections[node.node_id] = pc
                channel = pc.createDataChannel("inference")

                # Set up channel events
                result_future: asyncio.Future = asyncio.Future()

                @channel.on("message")
                def on_message(message):
                    if not result_future.done():
                        result_future.set_result(message)

                @channel.on("error")
                def on_error(err):
                    if not result_future.done():
                        result_future.set_exception(
                            ConnectionError(f"WebRTC data channel error: {err}")
                        )

                # Send prompt
                channel.send(payload)
                channel.send("__END__")

                # Wait for response with timeout
                import asyncio as _asyncio
                try:
                    result = _asyncio.wait_for(
                        asyncio.shield(result_future),
                        timeout=float(os.environ.get("DISTLLM_EDGE_TIMEOUT", "30.0")),
                    )
                    return str(result)
                except _asyncio.TimeoutError:
                    raise TimeoutError("WebRTC inference timed out")

            except ImportError:
                logger.warning("aiortc not installed — WebRTC edge inference unavailable")
                raise ConnectionError("aiortc not available")

        except Exception as e:
            logger.warning(f"WebRTC inference failed for {node.node_id}: {e}")
            raise

    def _send_via_websocket(
        self, node: EdgeNodeProfile, prompt: str, max_tokens: int,
    ) -> str:
        """Send inference request via WebSocket.

        Connects to the edge node's WebSocket endpoint, sends the
        prompt as a JSON message, and streams the response back.
        """
        logger.debug(f"WebSocket inference to {node.node_id} (session={node.session_id})")
        try:
            import asyncio
            import json
            import httpx

            # Build payload
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "stream": False,
            }

            async def _do_ws_inference() -> str:
                # Use httpx to POST to the edge node's endpoint
                # The edge node exposes a simple HTTP endpoint for inference
                async with httpx.AsyncClient(timeout=float(
                    os.environ.get("DISTLLM_EDGE_TIMEOUT", "30.0")
                )) as client:
                    try:
                        ws_url = f"ws://{node.node_id}/infer"
                        # For HTTP-based fallback:
                        http_url = f"http://{node.node_id}/v1/chat/completions"
                        resp = await client.post(
                            http_url,
                            json={"messages": [{"role": "user", "content": prompt}],
                                  "max_tokens": max_tokens},
                            timeout=float(os.environ.get("DISTLLM_EDGE_TIMEOUT", "30.0")),
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        else:
                            raise ConnectionError(
                                f"Edge node returned {resp.status_code}: {resp.text[:200]}"
                            )
                    except Exception as e:
                        raise ConnectionError(f"WebSocket edge inference failed: {e}")

            # Run async in a new event loop (this method is sync)
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(_do_ws_inference())
                loop.close()
                return result
            except Exception as e:
                raise ConnectionError(f"WebSocket inference failed: {e}")

        except Exception as e:
            logger.warning(f"WebSocket inference failed for {node.node_id}: {e}")
            raise

    # ── Health Checks ────────────────────────────────────────────────────

    def _health_loop(self) -> None:
        """Background loop that marks stale nodes as offline."""
        while self._running:
            time.sleep(15)
            now = time.time()
            stale_count = 0
            with self._lock:
                for node in self._nodes.values():
                    if node.is_online and (now - node.last_seen) > self._node_timeout:
                        node.is_online = False
                        stale_count += 1
            if stale_count:
                logger.debug(f"Marked {stale_count} edge nodes offline (timeout)")

    # ── Stats ────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            online = sum(1 for n in self._nodes.values() if n.is_online)
            total = self._requests_routed + self._requests_failed
            return {
                "connected_nodes": len(self._nodes),
                "online_nodes": online,
                "offline_nodes": len(self._nodes) - online,
                "requests_routed": self._requests_routed,
                "requests_failed": self._requests_failed,
                "success_rate": self._requests_routed / total if total > 0 else 1.0,
                "avg_edge_latency_ms": round(
                    self._total_edge_latency_ms / max(self._requests_routed, 1), 1
                ),
                "model_size_limit_b": self._model_size_limit / 1_000_000_000,
                "devices": {
                    "mobile": sum(1 for n in self._nodes.values() if n.device_type == "mobile"),
                    "browser": sum(1 for n in self._nodes.values() if n.device_type == "browser"),
                    "iot": sum(1 for n in self._nodes.values() if n.device_type == "iot"),
                },
            }
