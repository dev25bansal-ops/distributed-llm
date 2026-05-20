"""GatewayRouter: routes requests to backends with health awareness.

Supports multi-cluster federation through ``MultiClusterRouter``:
when a ``cluster_router`` is attached, the gateway first resolves
the best cluster+coordinator, then routes the request to that
coordinator's backend.
"""

import asyncio
import hashlib
import time
from typing import Any, Optional

from loguru import logger

from distllm.gateway.backend import ModelBackend, create_backend
from distllm.gateway.fallback import FallbackManager
from distllm.gateway.models import (
    BackendConfig,
    BackendHealth,
    GatewayConfig,
    ModelRoute,
)


class GatewayRouter:
    """Routes chat completion requests to configured backends.

    Supports model-based routing, weighted random selection among
    healthy backends, health checking, fallback chains, and optional
    multi-cluster federation routing.

    When a ``cluster_router`` is attached, the route method first
    resolves the best cluster for the user (via affinity ring and
    latency spillover), then routes to that cluster's coordinator.
    """

    def __init__(self, config: GatewayConfig):
        self.config = config
        self._backends: dict[str, ModelBackend] = {}
        self._routes: dict[str, ModelRoute] = {}
        self._health_cache: dict[str, BackendHealth] = {}
        self._fallback = FallbackManager()
        self._health_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._cluster_router: Any = None  # MultiClusterRouter (optional)

    async def start(self):
        """Initialize backends and start health checker."""
        for bc in self.config.backends:
            backend = create_backend(bc)
            self._backends[bc.name] = backend
            self._health_cache[bc.name] = BackendHealth(backend_name=bc.name)
            logger.info(f"Gateway backend registered: {bc.name} ({bc.backend_type.value})")

        for route in self.config.routes:
            self._routes[route.model_name] = route
            logger.info(f"Gateway route registered: {route.model_name} -> {route.primary_backend}")

        self._health_task = asyncio.create_task(self._health_loop())

    async def stop(self):
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

    def set_cluster_router(self, cluster_router: Any) -> None:
        """Attach a ``MultiClusterRouter`` for cross-cluster routing.

        When set, the gateway will route to a coordinator backend
        resolved by the cluster router instead of picking among
        locally-configured backends.

        Args:
            cluster_router: A ``MultiClusterRouter`` instance.
        """
        self._cluster_router = cluster_router

    def _resolve_user_id(self, body: dict, headers: dict | None = None) -> str:
        """Extract a stable user identifier from the request.

        Priority: ``user`` field in body → ``X-User-ID`` header →
        ``X-API-Key`` prefix → MD5 of last message content.
        """
        uid = body.get("user", "")
        if uid:
            return str(uid)
        if headers:
            uid = headers.get("X-User-ID", "") or headers.get("x-user-id", "")
            if uid:
                return str(uid)
            api_key = headers.get("Authorization", "") or headers.get("authorization", "")
            if api_key.startswith("Bearer "):
                return api_key[len("Bearer "):]
        messages = body.get("messages", [])
        if messages:
            content = messages[-1].get("content", "")
            if content:
                return hashlib.md5(content.encode()).hexdigest()
        return "anonymous"

    def _resolve_user_region(self, body: dict, headers: dict | None = None) -> str | None:
        """Extract user region hint from request headers."""
        if headers:
            region = (headers.get("X-Region", "") or
                      headers.get("x-region", "") or
                      headers.get("X-User-Region", ""))
            return region or None
        return None

    async def route_chat_completion(self, body: dict, headers: dict | None = None) -> dict:
        """Route a chat completion to the best backend.

        If a ``cluster_router`` is attached, first resolves the
        optimal cluster and coordinator for the user, then routes
        to that coordinator's URL.
        """
        # Multi-cluster routing path
        if self._cluster_router is not None:
            return await self._route_via_cluster(body, headers)

        # Local backend routing path
        model = body.get("model", "")
        route = self._routes.get(model)
        if route:
            return await self._route_with_fallback(
                route.primary_backend, route.fallback_chain, body, headers
            )

        backends = self._healthy_backends()
        if not backends:
            raise RuntimeError("No healthy backends available")

        total = sum(b.config.weight for b in backends)
        if total == 0:
            selected = backends[0]
        else:
            r = time.monotonic() % total
            cumulative = 0
            selected = backends[0]
            for b in backends:
                cumulative += b.config.weight
                if r < cumulative:
                    selected = b
                    break

        return await selected.chat_completion(body, headers)

    async def route_chat_completion_stream(self, body: dict, headers: dict | None = None):
        """Route a streaming chat completion."""
        if self._cluster_router is not None:
            async for chunk in self._stream_via_cluster(body, headers):
                yield chunk
            return

        model = body.get("model", "")
        route = self._routes.get(model)
        if route:
            async for chunk in self._stream_with_fallback(
                route.primary_backend, route.fallback_chain, body, headers
            ):
                yield chunk
            return

        backends = self._healthy_backends()
        if not backends:
            raise RuntimeError("No healthy backends available")

        total = sum(b.config.weight for b in backends)
        r = time.monotonic() % (total or 1)
        cumulative = 0
        selected = backends[0]
        for b in backends:
            cumulative += b.config.weight or 1
            if r < cumulative:
                selected = b
                break

        async for chunk in selected.chat_completion_stream(body, headers):
            yield chunk

    async def list_models(self) -> list[str]:
        """Aggregate models from all healthy backends."""
        models = set()
        for backend in self._backends.values():
            if backend.healthy:
                try:
                    m = await backend.list_models()
                    models.update(m)
                except Exception:
                    continue
        return sorted(models)

    def get_health(self) -> dict[str, BackendHealth]:
        return dict(self._health_cache)

    async def _route_with_fallback(
        self, primary: str, fallback_chain: list[str], body: dict, headers: dict | None = None
    ) -> dict:
        """Try primary, then fallback chain on error."""
        candidates = [primary] + fallback_chain
        for name in candidates:
            backend = self._backends.get(name)
            if backend is None or not backend.healthy:
                continue
            try:
                return await backend.chat_completion(body, headers)
            except Exception as e:
                logger.warning(f"Fallback from {name}: {e}")
                self._fallback.record_failure(name)
                continue
        raise RuntimeError(f"No backend succeeded for {body.get('model', 'unknown')}")

    async def _stream_with_fallback(self, primary, fallback_chain, body, headers):
        candidates = [primary] + fallback_chain
        for name in candidates:
            backend = self._backends.get(name)
            if backend is None or not backend.healthy:
                continue
            try:
                async for chunk in backend.chat_completion_stream(body, headers):
                    yield chunk
                return
            except Exception as e:
                logger.warning(f"Stream fallback from {name}: {e}")
                self._fallback.record_failure(name)
                continue
        raise RuntimeError(f"No backend succeeded for stream {body.get('model', 'unknown')}")

    def _healthy_backends(self) -> list[ModelBackend]:
        return [b for b in self._backends.values() if b.healthy]

    # -- Multi-cluster federation routing --

    async def _route_via_cluster(self, body: dict, headers: dict | None = None) -> dict:
        """Route a request via the multi-cluster router.

        1. Resolve user_id (for affinity) + region hint from request
        2. Route to the best cluster + coordinator
        3. Forward the request to that coordinator's backend URL

        Injects ``X-Cluster-Id`` and ``X-Route-Reason`` into the
        response for observability.
        """
        user_id = self._resolve_user_id(body, headers)
        user_region = self._resolve_user_region(body, headers)

        coord, reason = await self._cluster_router.route(
            user_id=user_id, user_region=user_region,
        )
        if coord is None:
            raise RuntimeError(
                f"No cluster available for user {user_id}: {reason}"
            )

        # Build a temporary backend pointing at the chosen coordinator
        from distllm.gateway.backend import BackendConfig, BackendType, NativeBackend

        backend = NativeBackend(
            BackendConfig(
                name=coord.node_id,
                base_url=coord.url,
                timeout_s=120.0,
            ),
        )

        result = await backend.chat_completion(body, headers)

        if isinstance(result, dict):
            result.setdefault("x-distllm", {})
            result["x-distllm"]["cluster_id"] = (
                self._cluster_router.discovery.get_cluster(coord.node_id.split("-coord")[0])
                if hasattr(self._cluster_router, "discovery") else "unknown"
            )
            result["x-distllm"]["route_reason"] = reason

        return result

    async def _stream_via_cluster(self, body: dict, headers: dict | None = None):
        """Stream a request via the multi-cluster router."""
        user_id = self._resolve_user_id(body, headers)
        user_region = self._resolve_user_region(body, headers)

        coord, reason = await self._cluster_router.route(
            user_id=user_id, user_region=user_region,
        )
        if coord is None:
            raise RuntimeError(
                f"No cluster available for user {user_id}: {reason}"
            )

        from distllm.gateway.backend import BackendConfig, BackendType, NativeBackend

        backend = NativeBackend(
            BackendConfig(
                name=coord.node_id,
                base_url=coord.url,
                timeout_s=120.0,
            ),
        )

        first = True
        async for chunk in backend.chat_completion_stream(body, headers):
            if first:
                import json as _json
                try:
                    payload = _json.loads(chunk.removeprefix("data: "))
                    payload.setdefault("x-distllm", {})
                    payload["x-distllm"]["cluster_id"] = (
                        self._cluster_router.discovery.get_cluster(coord.node_id.split("-coord")[0])
                        if hasattr(self._cluster_router, "discovery") else "unknown"
                    )
                    payload["x-distllm"]["route_reason"] = reason
                    yield f"data: {_json.dumps(payload)}\n\n"
                except Exception:
                    yield chunk
                first = False
            else:
                yield chunk

    async def _health_loop(self):
        """Periodic health check of all backends."""
        interval = self.config.health_check_interval_s
        while True:
            await asyncio.sleep(interval)
            for name, backend in self._backends.items():
                try:
                    healthy, lat, err = await backend.health()
                    self._health_cache[name] = BackendHealth(
                        backend_name=name,
                        healthy=healthy,
                        latency_ms=lat,
                        active_requests=backend.active_requests,
                        last_check=time.time(),
                        error=err,
                        models_available=backend.models_available,
                    )
                except Exception as e:
                    self._health_cache[name] = BackendHealth(
                        backend_name=name,
                        healthy=False,
                        error=str(e),
                    )
