"""OpenAI-compatible REST API for distributed LLM inference."""

from __future__ import annotations

import argparse
import asyncio
import hmac
import os
import threading
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware

from distllm.core.coordinator import Coordinator
from distllm.core.plugin_system import PluginSystem
from distllm.plugins.builtin import RateLimitPlugin, AuditLogPlugin, MetricsPlugin
from distllm.api.middleware import AuthMiddleware, RequestIDMiddleware, RequestRateLimitMiddleware
from distllm.config.resolver import ConfigResolver
from distllm.dashboard.ws_handler import (
    manager,
    metrics_broadcaster,
    get_collector,
    parse_client_message,
    stream_metrics_sse,
    KNOWN_METRIC_CATEGORIES,
)
from fastapi.responses import StreamingResponse
from starlette.responses import Response
from distllm.core.monitor import SystemMonitor
from distllm.config.settings import DistLLMSettings
from distllm.core.debug import set_debug_mode
from distllm.observability.tracing import setup_tracing
from distllm.observability.logging import setup_logging
from distllm.observability.exporter import DistLLMPrometheusExporter
from loguru import logger
from distllm.constants import HSTS_MAX_AGE

# Re-export route routers
from distllm.api.api_state import g as _g
from distllm.api.errors import error_response, error_response_from_request
from distllm.api.routes import (
    chat_router,
    chat_v2_router,
    completion_router,
    embeddings_router,
    health_router,
    gossip_router,
    admin_router,
    marketplace_router,
    federated_router,
    webrtc_router,
    leaderboard_router,
    prompts_router,
    model_registry_router,
    router_admin_router,
    defrag_router,
    batch_router,
)

# Re-export models from route modules for backward compatibility
from distllm.api.routes.chat import (
    ChatMessage,
    ChatCompletionRequest,
    ChatChoice,
    ChatCompletionResponse,
)
from distllm.api.routes.completion import (
    CompletionRequest,
    CompletionChoice,
    CompletionResponse,
)
from distllm.api.routes.embeddings import (
    EmbeddingRequest,
    EmbeddingObject,
    EmbeddingResponse,
)
from distllm.api.routes.health import ModelInfo, ModelList, ParamUpdateRequest

# Re-export streaming helpers for backward compatibility
from distllm.api.streaming import (
    _build_chunk,
    _generate_tokens,
    _stream_response,
)


def _get_cors_origins() -> list[str]:
    """Get CORS origins from env var (falls back to settings default).

    Security: Rejects wildcard origins unless DISTLLM_DEV_MODE=1 is set.
    Validates that all origins are well-formed URLs.
    Returns safe defaults if configuration is invalid.
    """
    DEFAULT_ORIGINS = ["http://localhost:3000", "http://localhost:8080"]

    raw = os.environ.get("DISTLLM_CORS_ORIGINS")
    origins: list[str] = []
    if raw:
        origins = [o.strip() for o in raw.split(",") if o.strip()]
    else:
        try:
            settings_val = DistLLMSettings().coordinator.cors_origins
            origins = [o.strip() for o in settings_val.split(",") if o.strip()]
        except Exception:
            origins = list(DEFAULT_ORIGINS)

    if not origins:
        origins = list(DEFAULT_ORIGINS)

    valid = []
    for origin in origins:
        if origin == "*" and os.environ.get("DISTLLM_DEV_MODE") != "1":
            valid.extend(DEFAULT_ORIGINS)
            continue
        valid.append(origin)
    return valid


# Lazy-initialized CORS origins (avoids import-time side effects)
_CORS_ORIGINS: list[str] | None = None
_cors_origins_lock = threading.Lock()


def _get_cors_origins_lazy() -> list[str]:
    """Get CORS origins, initializing on first call."""
    global _CORS_ORIGINS
    if _CORS_ORIGINS is None:
        with _cors_origins_lock:
            if _CORS_ORIGINS is None:
                _CORS_ORIGINS = _get_cors_origins()
    return _CORS_ORIGINS


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: initialize on startup, clean up on shutdown."""
    import signal
    import os

    _init_observability()

    # Initialize plugin system and register built-in plugins
    state.plugin_system = PluginSystem()
    _init_plugins(state.plugin_system)

    # Register SIGHUP handler for configuration hot-reload (Unix only)
    if hasattr(signal, 'SIGHUP'):
        def _reload_config(signum, frame):
            """Reload configuration on SIGHUP without restarting."""
            try:
                from distllm.config.settings import DistLLMSettings
                config_path = os.environ.get("DISTLLM_CONFIG", "config.yaml")
                if os.path.exists(config_path):
                    new_settings = DistLLMSettings.from_yaml(config_path=config_path)
                    # Update coordinator settings if available
                    coord = getattr(state, 'coordinator', None)
                    if coord is not None:
                        logger.info(f"Config reloaded from {config_path}")
                        # Update rate limits, timeouts, etc. from new settings
                        # Use lock to prevent race condition with scheduler
                        if hasattr(coord, '_batch_scheduler') and coord._batch_scheduler is not None:
                            scheduler = coord._batch_scheduler
                            if hasattr(new_settings, 'batching'):
                                with scheduler._lock if hasattr(scheduler, '_lock') else None:
                                    scheduler.max_batch_size = new_settings.batching.max_batch_size
                                    scheduler.max_tokens_per_batch = new_settings.batching.max_tokens_per_batch
                    else:
                        logger.info(f"Config reloaded (no coordinator to update)")
                else:
                    logger.warning(f"Config file not found: {config_path}")
            except Exception as e:
                logger.error(f"Config reload failed: {e}")

        signal.signal(signal.SIGHUP, _reload_config)
        logger.info("SIGHUP handler registered for config hot-reload")

    # Security warning when TLS is disabled
    if not os.environ.get("DISTLLM_TLS_ENABLED", "").lower() in ("1", "true"):
        logger.warning(
            "TLS is DISABLED. API keys and data are transmitted in plaintext. "
            "Set DISTLLM_TLS_ENABLED=true for production deployments."
        )

    # Display API key for users
    from distllm.core.api_key_store import get_api_key_store
    store = get_api_key_store()
    display_key = store.get_display_key()
    if display_key:
        logger.info(f"API Key: {display_key}")
        logger.info(f"Use: curl -H 'Authorization: Bearer {display_key}' http://localhost:8000/health")
    else:
        logger.info("API keys loaded from config file. Use 'distllm config keys' to manage.")

    _start_ws_broadcaster()
    yield
    if state.plugin_system:
        state.plugin_system.stop_all()
    if state.ws_broadcast_task:
        state.ws_broadcast_task.cancel()


# Disable OpenAPI docs in production (set DISTLLM_ENABLE_DOCS=1 to enable)
_enable_docs = os.environ.get("DISTLLM_ENABLE_DOCS", "0").lower() in ("1", "true")

app = FastAPI(
    lifespan=lifespan,
    title="Distributed LLM API",
    description="OpenAI-compatible REST API for distributed LLM inference across multiple machines using pipeline parallelism",
    version="0.4.0",
    docs_url="/docs" if _enable_docs else None,
    redoc_url="/redoc" if _enable_docs else None,
    openapi_tags=[
        {"name": "chat", "description": "Chat completion endpoints with streaming support across distributed nodes"},
        {"name": "completion", "description": "Text completion endpoints with streaming support across distributed nodes"},
        {"name": "embedding", "description": "Text embedding and document reranking"},
        {"name": "system", "description": "Health checks, metrics, and cluster status"},
        {"name": "gossip", "description": "P2P gossip protocol for distributed KV cache discovery between nodes"},
    ],
)

# Security: Configure CORS with explicit allowed origins
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_origins_lazy(),
    allow_credentials=False,  # Security: Disable credentials for CORS
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-Request-Timeout", "X-Priority"],
    max_age=600,  # Cache preflight for 10 minutes
)

# Security headers middleware
_API_VERSIONS: dict[str, str] = {
    "v1": "2024-01-01",
    "v2": "2025-03-01",
}
_API_VERSION_HEADER = "X-API-Version"
_API_SUNSET_HEADER = "Sunset"
_API_DEPRECATION_HEADER = "X-API-Deprecation"
# Map version -> sunset date (RFC 3339) when that version will be removed
_API_SUNSET_DATES: dict[str, str | None] = {
    "v1": None,         # current stable, no sunset
    "v2": None,         # latest stable, no sunset
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers and API versioning to all responses.

    Version contract:
      - Every versioned response includes ``X-API-Version`` with a date string.
      - Endpoints under a deprecated version get ``Sunset`` and
        ``X-API-Deprecation`` headers.
      - Unversioned endpoints (``/health``, ``/dashboard``, ``/api/*``, etc.)
        are considered internal and do not receive version headers.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        tls_enabled = os.environ.get("DISTLLM_TLS_ENABLED", "false").lower() == "true"
        if tls_enabled:
            response.headers["Strict-Transport-Security"] = f"max-age={HSTS_MAX_AGE}; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

        # API versioning: tag versioned paths with the API version
        path = request.url.path
        for ver_prefix, ver_date in _API_VERSIONS.items():
            if path.startswith(f"/{ver_prefix}/"):
                response.headers[_API_VERSION_HEADER] = ver_date
                sunset = _API_SUNSET_DATES.get(ver_prefix)
                if sunset is not None:
                    response.headers[_API_SUNSET_HEADER] = sunset
                    response.headers[_API_DEPRECATION_HEADER] = (
                        f"Version {ver_prefix} will be removed after {sunset}. "
                        f"See https://docs.distllm.dev/api-versions for migration."
                    )
                break

        return response


app.add_middleware(SecurityHeadersMiddleware)


# Structured error responses
class ErrorResponse(BaseModel):
    """Standardized error response format."""
    error: str
    message: str
    type: str = "api_error"
    code: str | None = None
    request_id: str | None = None


def _error_response(
    status_code: int,
    error: str,
    message: str,
    type: str = "api_error",
    request: Request | None = None,
    exc: Exception | None = None,
) -> JSONResponse:
    """Build a standardized error response."""
    return error_response_from_request(status_code, error, message, type, request, exc=exc)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Convert Pydantic validation errors to OpenAI-compatible 422."""
    messages = []
    for err in exc.errors():
        loc = " -> ".join(str(p) for p in err.get("loc", []))
        messages.append(f"{loc}: {err.get('msg', '')}" if loc else err.get("msg", ""))
    return _error_response(
        status_code=422,
        error="Invalid Request",
        message="; ".join(messages),
        type="invalid_request_error",
        request=request,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Convert HTTPException to structured error response."""
    return _error_response(
        status_code=exc.status_code,
        error=f"HTTP {exc.status_code}",
        message=exc.detail,
        type="http_error",
        request=request,
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled exceptions with structured response."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return _error_response(
        status_code=500,
        error="Internal Server Error",
        message="An unexpected error occurred. Please try again later.",
        type="internal_error",
        request=request,
        exc=exc,
    )


# ── Single source of truth for application state ─────────────────────
# Routes use `g` from api_state.py. Server code uses `state` below.
# Both MUST reference the same AppState instance to stay in sync.
from distllm.api.api_state import _state as _shared_state


class _ServerState:
    """Server-side state proxy that delegates to the shared AppState.

    This ensures ``state.coordinator`` and ``g.coordinator`` always
    return the same object — no dual bookkeeping.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(_shared_state, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(_shared_state, name, value)


state = _ServerState()





def _init_observability() -> None:
    """Initialize tracing, logging, metrics exporter."""
    setup_logging(level="INFO", json_format=True)

    setup_tracing(
        service_name="distllm-api",
        sampling_strategy="head",
        sampling_ratio=1.0,
    )

    state.metrics_exporter = DistLLMPrometheusExporter()


# Request timeout middleware
# NOTE: Registered BEFORE AuthMiddleware so that AuthMiddleware runs first
# (FastAPI executes middleware in reverse order of registration).
class TimeoutMiddleware(BaseHTTPMiddleware):
    """Cancel requests that exceed the timeout limit."""

    DEFAULT_TIMEOUT = 120.0  # 2 minutes default

    # H-20: Use rstrip('/') for path matching to catch trailing slashes
    ENDPOINT_TIMEOUTS = {
        "/v1/chat/completions": 300.0,
        "/v1/completions": 300.0,
        "/v1/embeddings": 60.0,
    }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # H-20: Normalize path before lookup — strip trailing slash and query params
        path = request.url.path.rstrip("/")
        timeout = self.ENDPOINT_TIMEOUTS.get(
            path, self.DEFAULT_TIMEOUT
        )
        # Per-request timeout via X-Request-Timeout header
        per_request = getattr(request.state, "request_timeout", None)
        if per_request is not None:
            timeout = per_request

        try:
            async with asyncio.timeout(timeout):
                response = await call_next(request)
                return response
        except asyncio.TimeoutError:
            return _error_response(
                status_code=504,
                error="Gateway Timeout",
                message=f"Request exceeded {timeout:.0f}s timeout limit",
                type="timeout_error",
                request=request,
            )


app.add_middleware(TimeoutMiddleware)

# AuthMiddleware registered AFTER TimeoutMiddleware so it runs first (outermost)
app.add_middleware(AuthMiddleware)

# RequestIDMiddleware registered after AuthMiddleware so it runs before it
# on incoming requests, ensuring request.state.request_id is set before any
# middleware that reads it (e.g. AuthMiddleware error responses).
app.add_middleware(RequestIDMiddleware)

# RequestRateLimitMiddleware — per-IP request rate limiting
# Registered after RequestIDMiddleware so request.state.request_id is
# available for rate-limit error responses.
app.add_middleware(RequestRateLimitMiddleware)

# DedupMiddleware — collapses identical concurrent POST requests
# Only applies to /v1/chat/completions; uses content fingerprinting.
from distllm.api.dedup import DedupMiddleware
app.add_middleware(DedupMiddleware)




class _RequestTooLarge(Exception):
    pass


# Request size limiting middleware
class RequestSizeLimitMiddleware:
    """Limit maximum request body size to prevent OOM."""

    MAX_REQUEST_SIZE = 32 * 1024 * 1024  # 32 MB default

    def __init__(self, app: Any, max_size: int | None = None) -> None:
        self.app = app
        if max_size is not None:
            self.MAX_REQUEST_SIZE = max_size

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope.get("method") not in ("POST", "PUT", "PATCH"):
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        cl = headers.get(b"content-length")
        if cl:
            try:
                if int(cl) > self.MAX_REQUEST_SIZE:
                    response = JSONResponse(
                        status_code=413,
                        content={
                            "error": {
                                "message": f"Request exceeds maximum size of {self.MAX_REQUEST_SIZE // (1024*1024)} MB",
                                "type": "request_too_large",
                                "code": "413",
                            },
                        },
                    )
                    await response(scope, receive, send)
                    return
            except (ValueError, TypeError):
                pass

        total = 0

        async def limited_receive():
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.MAX_REQUEST_SIZE:
                    raise _RequestTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestTooLarge:
            response = JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "message": f"Request exceeds maximum size of {self.MAX_REQUEST_SIZE // (1024*1024)} MB",
                        "type": "request_too_large",
                        "code": "413",
                    },
                },
            )
            await response(scope, receive, send)
            return


app.add_middleware(RequestSizeLimitMiddleware, max_size=100_000_000)


# Backpressure middleware
class BackpressureMiddleware(BaseHTTPMiddleware):
    """Reject requests when system is under heavy load."""

    MAX_PENDING_REQUESTS = 1000  # Max pending requests before rejecting

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip backpressure for health/metrics endpoints
        if request.url.path in ("/health", "/ready", "/live", "/metrics", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        # Check if scheduler is overloaded
        if state.coordinator and state.coordinator.scheduler:
            try:
                stats = state.coordinator.scheduler.stats()
                pending = stats.get("pending_requests", 0)
                if isinstance(pending, (int, float)) and pending >= self.MAX_PENDING_REQUESTS:
                    return _error_response(
                        status_code=503,
                        error="Service Unavailable",
                        message=f"System overloaded: {pending} pending requests",
                        type="backpressure_error",
                        request=request,
                    )
            except (AttributeError, TypeError, KeyError):
                pass  # Scheduler stats unavailable or malformed, skip check

        # Check if shutting down
        if state.coordinator and getattr(state.coordinator, "_shutting_down", False):
            return _error_response(
                status_code=503,
                error="Service Unavailable",
                message="Service is shutting down",
                type="shutdown_error",
                request=request,
            )

        return await call_next(request)


app.add_middleware(BackpressureMiddleware)


# ── Plugin system ───────────────────────────────────────────────────────────

def _init_plugins(ps: PluginSystem) -> None:
    """Register + load + init + start built-in plugins."""
    for cls in (RateLimitPlugin, AuditLogPlugin, MetricsPlugin):
        ps.register(cls)
    ps.load_all()
    ps.init_all()
    ps.start_all()
    logger.info(f"Plugin system ready: {len(ps.list_plugins())} plugins active")


class PluginHookMiddleware(BaseHTTPMiddleware):
    """Dispatch plugin hooks around every request.

    Runs after ``BackpressureMiddleware`` (outermost) and before all other
    middleware.  Captures request context, dispatches ``on_request``,
    delegates to the next handler, then dispatches ``on_response`` or
    ``on_error`` based on the outcome.
    """

    SKIP_PATHS = {"/health", "/ready", "/live", "/metrics", "/docs", "/openapi.json",
                  "/redoc", "/ws", "/dashboard"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        plugin_sys: PluginSystem | None = getattr(state, "plugin_system", None)
        if plugin_sys is None or request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        # Build plugin context from the request
        req_ctx = {
            "method": request.method,
            "path": request.url.path,
            "query": str(request.query_params),
            "request_id": getattr(request.state, "request_id", ""),
            "tenant": getattr(request.state, "tenant", "default"),
            "model": getattr(request.state, "model", ""),
            "client_ip": request.client.host if request.client else "",
            "user_agent": request.headers.get("user-agent", ""),
            "api_key_role": getattr(request.state, "api_key_role", ""),
            "api_key_id": getattr(request.state, "api_key_id", ""),
        }

        # Allow plugins to modify/reject the request
        plugin_ctx = plugin_sys.dispatch_on_request(req_ctx)
        reject = plugin_ctx.get("_reject")
        if reject:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": reject.get("reason", "rate_limit_exceeded"),
                        "type": "rate_limit",
                        "retry_after": reject.get("retry_after", 60),
                    },
                },
            )

        # Process the request
        start = time.time()
        try:
            response = await call_next(request)
        except Exception as exc:
            plugin_sys.dispatch("on_error", req_ctx, exc)
            raise

        # Post-process: dispatch on_response for non-streaming responses
        duration_ms = (time.time() - start) * 1000
        resp_ctx = {
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        }
        plugin_sys.dispatch("on_response", req_ctx, resp_ctx)

        return response


app.add_middleware(PluginHookMiddleware)


# Include route routers
app.include_router(chat_router)
app.include_router(chat_v2_router)
app.include_router(completion_router)
app.include_router(embeddings_router)
app.include_router(health_router)
app.include_router(gossip_router)
app.include_router(admin_router)
app.include_router(marketplace_router)
app.include_router(federated_router)
app.include_router(webrtc_router)
app.include_router(leaderboard_router)
app.include_router(prompts_router)
app.include_router(model_registry_router)
app.include_router(router_admin_router)
app.include_router(batch_router)

# Cost tracking middleware
try:
    from distllm.api.cost_middleware import CostTrackingMiddleware
    app.add_middleware(CostTrackingMiddleware)
except ImportError:
    pass

# --- Dashboard & WebSocket ---

from pathlib import Path


@app.websocket("/ws")
async def dashboard_websocket(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time dashboard metrics.

    Client may send JSON commands:
      - ``{"type":"subscribe","metrics":["latency","gpu"],"interval":2.0}``
      - ``{"type":"ping"}``

    Supported metric categories: latency, ttft, throughput, tokens_per_sec,
    kv_cache, speculative, cost, queue_depth, active_requests, scheduler,
    nodes, gpu, prefix_cache, spec_decoder, topology, tenants.
    """
    # Authenticate WebSocket connection via query param or first message
    auth_token = websocket.query_params.get("token", "")
    if not auth_token:
        # Try to get from Authorization header
        auth_header = websocket.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            auth_token = auth_header[7:]

    # Validate API key if auth is configured
    from distllm.core.api_key_store import get_api_key_store
    store = get_api_key_store()
    if store.get_key_count() > 0:
        if not auth_token:
            logger.warning("WebSocket connection rejected: missing API key")
            await websocket.close(code=4001, reason="API key required")
            return
        result = store.authenticate(auth_token)
        if result is None:
            await websocket.close(code=4001, reason="Invalid API key")
            return

    await manager.connect(websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            cmd = parse_client_message(raw)
            cmd_type = cmd.get("type", "error")

            if cmd_type == "subscribe":
                manager.subscribe(
                    websocket,
                    metric_types=cmd.get("metrics"),
                    interval=cmd.get("interval", 1.0),
                )
                await manager.send_to(websocket, {
                    "type": "subscribed",
                    "metrics": cmd.get("metrics"),
                    "interval": cmd.get("interval", 1.0),
                })
            elif cmd_type == "ping":
                await manager.send_to(websocket, {"type": "pong", "timestamp": time.time()})
            elif cmd_type == "error":
                await manager.send_to(websocket, {"type": "error", "detail": cmd.get("detail", "Unknown error")})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.websocket("/ws/metrics")
async def metrics_websocket(websocket: WebSocket) -> None:
    """Dedicated WebSocket endpoint for live metrics streaming.

    Unlike /ws (which requires subscribe commands), this endpoint
    auto-streams all metrics at a configurable interval.

    SECURITY: Requires API key authentication (same as /ws endpoint).

    Query params:
        interval: Stream interval in seconds (default: 1.0)
        categories: Comma-separated metric categories to include
        token: API key for authentication
    """
    # SECURITY: Authenticate WebSocket connection
    from distllm.core.api_key_store import get_api_key_store
    store = get_api_key_store()
    auth_token = websocket.query_params.get("token", "")
    if not auth_token:
        auth_header = websocket.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            auth_token = auth_header[7:]

    if store.get_key_count() > 0:
        if not auth_token:
            logger.warning("Metrics WebSocket rejected: missing API key")
            await websocket.close(code=4001, reason="API key required")
            return
        result = store.authenticate(auth_token)
        if result is None:
            await websocket.close(code=4001, reason="Invalid API key")
            return

    await websocket.accept()
    # Clamp interval to safe range (0.2s - 10.0s) to prevent DoS
    interval = max(0.2, min(float(websocket.query_params.get("interval", "1.0")), 10.0))
    categories = websocket.query_params.get("categories", "")
    requested = [c.strip() for c in categories.split(",") if c.strip()] or None

    collector = get_collector()
    try:
        while True:
            coord = state.coordinator
            if coord is None:
                await websocket.send_json({"type": "status", "coordinator": "not_loaded"})
                await asyncio.sleep(interval)
                continue

            # Build metrics snapshot
            snapshot = {"type": "metrics", "timestamp": time.time()}

            # Coordinator metrics
            try:
                coord_metrics = coord.get_metrics()
                if not requested or "coordinator" in requested:
                    snapshot["coordinator"] = coord_metrics
            except Exception:
                pass

            # Scheduler metrics
            try:
                if coord.scheduler:
                    sched_stats = coord.scheduler.stats()
                    if not requested or "scheduler" in requested:
                        snapshot["scheduler"] = sched_stats
            except Exception:
                pass

            # Prometheus metrics snapshot (if available)
            try:
                from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
                prom_data = generate_latest()
                if prom_data and (not requested or "prometheus" in requested):
                    snapshot["prometheus"] = {
                        "gpu_util": _extract_prom_gauge(prom_data, "distllm_gpu_utilization"),
                        "requests_active": _extract_prom_gauge(prom_data, "distllm_active_requests"),
                        "tokens_total": _extract_prom_counter(prom_data, "distllm_tokens_total"),
                    }
            except Exception:
                pass

            # Collector metrics
            if collector:
                if not requested or "latency" in requested:
                    snapshot["latency"] = collector.summary()
                if not requested or "kv_cache" in requested:
                    snapshot["kv_cache"] = {"hit_rate": collector.kv_hit_rate()}
                if not requested or "speculative" in requested:
                    snapshot["speculative"] = {"acceptance_rate": collector.spec_acceptance_rate()}

            # GPU metrics
            try:
                mon = state.monitor
                if mon and (not requested or "gpu" in requested):
                    sys_metrics = mon.collect()
                    snapshot["gpu"] = sys_metrics.get("gpu", {})
                    snapshot["cpu"] = sys_metrics.get("cpu", {})
            except Exception:
                pass

            await websocket.send_json(snapshot)
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        pass


@app.get(
    "/dashboard",
    response_class=HTMLResponse,
    summary="Dashboard page",
    description="Serve the real-time monitoring dashboard HTML page. The dashboard displays live metrics, request throughput, latency charts, and system health via WebSocket connection.",
    response_description="Dashboard HTML page",
    include_in_schema=False,
)
async def dashboard_page() -> HTMLResponse:
    """Serve the real-time dashboard HTML."""
    html_path = Path(__file__).parent.parent / "dashboard" / "static_v2" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>Dashboard not found</h1>")


@app.get(
    "/dashboard/leaderboard",
    response_class=HTMLResponse,
    summary="Benchmark leaderboard page",
    description="Serve the benchmark leaderboard HTML page for comparing results across models, hardware, and frameworks.",
    include_in_schema=False,
)
async def dashboard_leaderboard_page() -> HTMLResponse:
    """Serve the benchmark leaderboard HTML."""
    html_path = Path(__file__).parent.parent / "dashboard" / "static_v2" / "leaderboard.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>Leaderboard not found</h1>")


@app.get(
    "/dashboard/models",
    response_class=HTMLResponse,
    summary="Model Registry page",
    include_in_schema=False,
)
async def model_registry_page() -> HTMLResponse:
    """Serve the model registry dashboard HTML."""
    html_path = Path(__file__).parent.parent / "dashboard" / "static_v2" / "models.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>Model Registry not found</h1>")


@app.get(
    "/api/cluster/nodes",
    summary="Cluster node topology",
    description="Return all registered worker nodes with their GPU info, health status, and layer assignments.",
    response_description="List of cluster nodes with capabilities",
)
async def api_cluster_nodes() -> dict:
    """Return current cluster node topology."""
    coord = state.coordinator
    if coord is None:
        return {"nodes": []}
    nodes_list = []
    for node_id, node in coord.nodes.items():
        if isinstance(node, dict):
            nodes_list.append({
                "node_id": node_id,
                "host": node.get("host", ""),
                "port": node.get("port", 0),
                "healthy": node.get("healthy", False),
                "start_layer": node.get("start_layer", 0),
                "end_layer": node.get("end_layer", 0),
                "gpu_name": node.get("gpu_name", ""),
            })
        else:
            nodes_list.append({
                "node_id": node_id,
                "host": getattr(node, "host", ""),
                "port": getattr(node, "port", 0),
                "healthy": getattr(node, 'healthy', False),
                "start_layer": getattr(node, 'start_layer', 0),
                "end_layer": getattr(node, 'end_layer', 0),
                "gpu_name": getattr(node, 'gpu_name', ''),
                "gpu_memory_free": getattr(node, 'gpu_memory_free', 0),
                "gpu_memory_total": getattr(node, 'gpu_memory_total', 0),
                "gpu_sm_count": getattr(node, 'gpu_sm_count', 0),
            })
    return {"nodes": nodes_list, "total_layers": coord.total_layers}


@app.post(
    "/v1/federation/heartbeat",
    summary="Federation heartbeat",
    description="Receive heartbeat from a federated peer coordinator with load metrics. "
                "Authenticated via cluster key in X-Cluster-Key header.",
    include_in_schema=False,
)
async def federation_heartbeat(request: Request) -> dict:
    """Receive and store heartbeat from a federated peer.

    Requires a valid ``X-Cluster-Key`` header matching the local coordinator's
    cluster key.  Prevents spoofed heartbeats from untrusted sources.
    """
    coord = state.coordinator
    if coord is None:
        return JSONResponse(status_code=503, content={"status": "unavailable"})

    # Verify cluster key
    local_key = getattr(coord.config, 'cluster_key', None)
    if local_key:
        received_key = request.headers.get("X-Cluster-Key", "")
        if not received_key or not hmac.compare_digest(received_key, local_key):
            return JSONResponse(
                status_code=401,
                content={"status": "unauthorized", "error": "invalid cluster key"},
            )

    body = await request.json()
    if hasattr(coord, '_federation') and coord._federation:
        try:
            for pid, peer in coord._federation._peers.items():
                if pid != coord._federation.config.cluster_id:
                    coord._federation._load_balancer.report_load(
                        cluster_id=pid,
                        active_requests=body.get("active_requests", 0),
                        pending_requests=body.get("pending_requests", 0),
                        gpu_utilization=body.get("gpu_utilization", 0.0),
                        queue_depth=body.get("pending_requests", 0),
                    )
        except Exception:
            pass
    return {"status": "ok"}


@app.get(
    "/api/pipeline/health",
    summary="Pipeline orchestrator health and metrics",
    description="Return pipeline health status, node execution metrics, transport info, and configuration.",
    response_description="Pipeline health and metrics",
)
async def api_pipeline_health() -> dict:
    """Return pipeline orchestrator health and metrics."""
    coord = state.coordinator
    if coord is None:
        return {"status": "no_coordinator"}
    pipeline = getattr(coord, '_pipeline', None)
    if pipeline is None:
        return {"status": "no_pipeline"}
    metrics = pipeline.get_pipeline_metrics()

    # Add per-node health and latency
    nodes = {}
    latency_tracker = getattr(pipeline, '_latency_tracker', None)
    for node_id, node in pipeline.nodes.items():
        node_latency = None
        if latency_tracker is not None:
            node_latency = latency_tracker.get_avg(node_id) if hasattr(latency_tracker, 'get_avg') else None
        nodes[node_id] = {
            "healthy": getattr(node, 'healthy', False),
            "latency_ms": node_latency,
            "start_layer": getattr(node, 'start_layer', 0),
            "end_layer": getattr(node, 'end_layer', 0),
            "gpu_name": getattr(node, 'gpu_name', ''),
        }
    metrics["nodes"] = nodes
    return metrics


@app.get(
    "/api/cluster/reputation",
    summary="Node reputation scores",
    description="Return reputation scores for all registered nodes based on reliability, speed, uptime, and health.",
    response_description="Reputation scores per node",
)
async def api_cluster_reputation() -> dict:
    """Return node reputation scores."""
    coord = state.coordinator
    if coord is None or not hasattr(coord, '_reputation'):
        return {"reputation": {}}
    return coord._reputation.get_summary()


@app.get(
    "/api/metrics/collector",
    summary="Metrics collector snapshot",
    description="Return a snapshot of all collected metrics from the observability collector, including raw counters and gauges for instrumentation debugging.",
    response_description="Collector metrics snapshot",
)
async def api_collector_metrics() -> dict:
    """Return current collector metrics snapshot."""
    return get_collector().summary()


@app.get(
    "/api/metrics/stream",
    summary="Metrics SSE stream",
    description="Subscribe to a real-time metrics stream via Server-Sent Events. Use query parameters to filter metric categories and set update interval.",
    response_description="Event stream of structured metrics JSON.",
)
async def api_metrics_stream(
    metrics: str = "",
    interval: float = 1.0,
) -> StreamingResponse:
    """SSE endpoint for real-time dashboard metrics.

    Query parameters:
      - ``metrics``: Comma-separated list of categories to subscribe to
                     (omit for all). e.g. ``metrics=latency,gpu,nodes``
      - ``interval``: Update interval in seconds (0.2-10.0, default 1.0)

    Supported categories: latency, ttft, throughput, tokens_per_sec,
    kv_cache, speculative, cost, queue_depth, active_requests, scheduler,
    nodes, gpu, prefix_cache, spec_decoder, topology, tenants.
    """
    interval = max(0.2, min(interval, 10.0))
    requested = None
    if metrics:
        requested = {m.strip() for m in metrics.split(",") if m.strip()}
        unknown = requested - KNOWN_METRIC_CATEGORIES
        if unknown:
            return JSONResponse(
                status_code=400,
                content={
                    "detail": f"Unknown metric categories: {', '.join(sorted(unknown))}",
                    "valid_categories": sorted(KNOWN_METRIC_CATEGORIES),
                },
            )

    return StreamingResponse(
        stream_metrics_sse(state.coordinator, requested_metrics=requested, interval=interval),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get(
    "/api/requests/waterfall",
    summary="Recent request waterfall data",
    description="Return recent request lifecycle data (queue → prefill → decode) for the dashboard waterfall chart.",
    response_description="List of request timing entries with elapsed_ms, ttft_ms, and request_id",
)
async def api_waterfall(limit: int = 50) -> list:
    """Return recent request waterfall entries showing lifecycle phases."""
    coord = state.coordinator
    if coord is None:
        return []
    try:
        scheduler = getattr(coord, "scheduler", None)
        if scheduler is None:
            return []
        tracker = getattr(scheduler, "latency_tracker", None) or getattr(scheduler, "_latency_tracker", None)
        if tracker is None:
            return []
        return tracker.get_recent_metrics(limit=limit)
    except (AttributeError, RuntimeError):
        return []


@app.get(
    "/api/continuum/stats",
    summary="Edge-to-cloud continuum statistics",
    description="Return statistics about the edge-to-cloud device continuum including device types, transports, and layer assignments.",
    response_description="Continuum statistics",
)
async def api_continuum_stats() -> dict:
    """Return edge-to-cloud continuum statistics."""
    continuum = getattr(state, "continuum", None)
    if continuum is None:
        return {"status": "not_initialized"}
    return continuum.get_stats()


@app.get(
    "/api/cost/summary",
    summary="Cost tracking summary",
    description="Return cost tracking summary including per-request costs, savings vs cloud APIs, and throughput metrics.",
    response_description="Cost summary",
)
async def api_cost_summary(tenant_id: str = "") -> dict:
    """Return cost tracking summary."""
    try:
        from distllm.core.cost_tracker import get_cost_tracker
        return get_cost_tracker().get_cost_summary(tenant_id)
    except ImportError:
        return {"status": "not_available"}


@app.get(
    "/api/cost/history",
    summary="Cost history",
    description="Return recent cost tracking history.",
    response_description="Cost history entries",
)
async def api_cost_history(limit: int = 100) -> list:
    """Return recent cost history."""
    try:
        from distllm.core.cost_tracker import get_cost_tracker
        return get_cost_tracker().get_history(limit)
    except ImportError:
        return []


@app.get(
    "/api/streaming-cost/stats",
    summary="Streaming cost statistics",
    description="Return real-time streaming cost tracker statistics.",
    response_description="Streaming cost stats",
)
async def api_streaming_cost_stats() -> dict:
    """Return streaming cost tracker statistics."""
    try:
        from distllm.core.streaming_cost import get_streaming_cost_tracker
        return get_streaming_cost_tracker().get_stats()
    except ImportError:
        return {"status": "not_available"}


def _extract_prom_gauge(data: bytes, metric_name: str) -> float | None:
    """Extract a single gauge value from Prometheus text format."""
    try:
        for line in data.decode("utf-8").splitlines():
            if line.startswith(metric_name) and " " in line:
                parts = line.rsplit(" ", 1)
                return float(parts[-1]) if len(parts) == 2 else None
    except Exception:
        return None
    return None


def _extract_prom_counter(data: bytes, metric_name: str) -> float | None:
    """Extract a counter value from Prometheus text format."""
    try:
        for line in data.decode("utf-8").splitlines():
            if line.startswith(metric_name) and not line.startswith("#"):
                parts = line.rsplit(" ", 1)
                if len(parts) == 2:
                    return float(parts[-1])
    except Exception:
        return None
    return None


def _start_ws_broadcaster() -> None:
    """Start the WebSocket metrics broadcaster background task."""
    if state.coordinator is not None:
        state.ws_broadcast_task = asyncio.create_task(metrics_broadcaster(state.coordinator))





def create_coordinator(
    model_name: str,
    dtype: str = "float16",
    local: bool = False,
    max_batch_size: int = 1,
    max_tokens_per_batch: int = 4096,
    settings: DistLLMSettings | None = None,
) -> Coordinator:
    """Create and configure the coordinator."""
    if settings:
        max_batch_size = settings.batching.max_batch_size
        max_tokens_per_batch = settings.batching.max_tokens_per_batch

    try:
        from distllm.dist.config import WideAreaConfig
    except ImportError:
        WideAreaConfig = None

    wide_area_config = None
    if settings and settings.wide_area.enabled:
        wa = settings.wide_area
        wide_area_config = WideAreaConfig(
            enabled=wa.enabled,
            p2p_forwarding=wa.p2p_forwarding,
            tokens_before_forward=wa.tokens_before_forward,
            wan_timeout_seconds=wa.wan_timeout_seconds,
            max_retries=wa.max_retries,
            backoff_base_seconds=wa.backoff_base_seconds,
        )

    coord = Coordinator(
        model_name=model_name,
        dtype=dtype,
        max_batch_size=max_batch_size,
        max_tokens_per_batch=max_tokens_per_batch,
        metrics_exporter=state.metrics_exporter,
        wide_area_config=wide_area_config,
        plugin_system=getattr(state, "plugin_system", None),
    )

    if local:
        coord.load_local_model()
        logger.info(f"Coordinator loaded model locally: {model_name}")
    else:
        logger.info(f"Coordinator ready for distributed mode: {model_name}")

    # Start gRPC server for worker connections
    # Create a minimal node-like object for the gRPC server
    coord_port = 50050  # Default coordinator gRPC port
    try:
        from distllm.dist.node_service import NodeServer

        # Create a wrapper that provides the interface NodeServer expects
        class _CoordinatorNode:
            def __init__(self, coordinator: Coordinator) -> None:
                self._coord = coordinator
                self.node_id = "coordinator"
                self.host = "0.0.0.0"
                self.port = coord_port
                self.start_layer = 0
                self.end_layer = 0
                self.total_layers = 0
                self.healthy = True
                self.partitioner = None

            def forward_fn(self, **kwargs: Any) -> Any:
                return self._coord.generate(**kwargs)

            def health_check(self) -> bool:
                return True

        coord._node_wrapper = _CoordinatorNode(coord)
        coord._node_server = NodeServer(coord._node_wrapper, port=coord_port, max_workers=4)
        coord._node_server.start(use_tls=False)
        logger.info(f"Coordinator gRPC server started on port {coord_port} for worker connections")
    except Exception as e:
        logger.warning(f"Could not start gRPC server on port {coord_port}: {e}")
        logger.warning("Workers will not be able to connect. Run 'system coordinator' separately.")

    monitor_inst = SystemMonitor()
    state.coordinator = coord
    state.monitor = monitor_inst

    return coord


def _load_settings(args: Any) -> DistLLMSettings:
    """Load settings via :class:`ConfigResolver` with full precedence.

    Precedence (lowest to highest):
    1. Pydantic defaults
    2. YAML config file
    3. Environment variables (``DISTLLM__*``) — handled by pydantic-settings
    4. CLI arguments (highest)
    """
    # Resolve config path
    config_path = args.config
    if config_path is None:
        # M-08: Use public resolve_config_path instead of private _resolve_config_path
        config_path = ConfigResolver.resolve_config_path("api", args)

    # Build CLI overrides
    cli_overrides = {}
    if args.model:
        cli_overrides.setdefault("model", {})["name"] = args.model
    if args.dtype:
        cli_overrides.setdefault("model", {})["dtype"] = args.dtype
    if args.host:
        cli_overrides.setdefault("coordinator", {})["host"] = args.host
    if args.port:
        cli_overrides.setdefault("coordinator", {})["api_port"] = args.port
    if args.quantization and args.quantization != "none":
        cli_overrides.setdefault("quantization", {})["method"] = args.quantization

    return DistLLMSettings.from_yaml(
        config_path=config_path,
        cli_overrides=cli_overrides or None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed LLM REST API")
    parser.add_argument("--model", type=str, default=None, help="Model name (overrides config)")
    parser.add_argument("--host", type=str, default=None, help="Server host (overrides config)")
    parser.add_argument("--port", type=int, default=None, help="Server port (overrides config)")
    parser.add_argument("--dtype", type=str, default=None, help="Model dtype (overrides config)")
    parser.add_argument("--local", action="store_true", help="Load model locally")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with tensor shape logging")
    parser.add_argument("--validate-config", action="store_true", help="Validate configuration at startup and exit")
    parser.add_argument("--quantization", type=str, default="none",
        choices=["none", "bitsandbytes_4bit", "bitsandbytes_8bit", "gptq"],
        help="Quantization method for model loading")
    parser.add_argument("--nodes", type=str, nargs="+", help="host:port:start:end per node")
    parser.add_argument("--total-layers", type=int, help="Total layers in model")

    args = parser.parse_args()

    if args.validate_config:
        DistLLMSettings.validate_startup()
        print("✅ Config validation passed")
        return

    settings = _load_settings(args)

    if args.debug:
        set_debug_mode(True)
        logger.warning(
            "DEBUG MODE ENABLED: Tensor shape logging is active. "
            "This may leak sensitive information about model architecture. "
            "Do not use in production."
        )
        logger.info("Debug mode enabled: tensor shape logging active")

    coord = create_coordinator(
        model_name=settings.model.name,
        dtype=settings.model.dtype,
        local=args.local,
        settings=settings,
    )

    # Register distributed nodes from CLI args
    if args.nodes and not args.local:
        for i, node_str in enumerate(args.nodes):
            parts = node_str.split(":")
            host = parts[0]
            port = int(parts[1])
            start = int(parts[2])
            end = int(parts[3])
            coord.manual_register(
                node_id=f"node_{i}",
                host=host, port=port,
                start_layer=start, end_layer=end,
                total_layers=args.total_layers,
            )
        logger.info(f"Registered {len(args.nodes)} distributed nodes")

    logger.info("Distributed inference pipeline initialized")

    logger.info(f"Starting API server on {settings.coordinator.host}:{settings.coordinator.api_port}")

    # Register graceful shutdown handler
    async def _shutdown() -> None:
        logger.info("Shutdown requested, draining connections...")
        if state.coordinator:
            in_flight = 0
            if state.coordinator.scheduler:
                try:
                    stats = state.coordinator.scheduler.stats()
                    in_flight = stats.get("pending_requests", 0) + stats.get("active_requests", 0)
                except (AttributeError, TypeError, KeyError):
                    pass
            if in_flight > 0:
                logger.info(f"Waiting for {in_flight} in-flight request(s) to complete...")
                await asyncio.sleep(min(in_flight * 0.5, 10.0))
        logger.info("Shutdown complete")

    app.add_event_handler("shutdown", _shutdown)

    config = uvicorn.Config(
        app,
        host=settings.coordinator.host,
        port=settings.coordinator.api_port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    main()
