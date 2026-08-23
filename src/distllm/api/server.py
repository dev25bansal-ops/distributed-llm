"""OpenAI-compatible REST API for distributed LLM inference."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import os
import threading
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, nullcontext
from typing import Any, Callable

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware

from distllm.core.coordinator import Coordinator
from distllm.core.plugin_system import PluginSystem
from distllm.plugins.builtin import RateLimitPlugin, AuditLogPlugin, MetricsPlugin, AuthPlugin
from distllm.plugins.health_plugin import HealthPlugin
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
from distllm.api.observability_middleware import ObservabilityMiddleware
from loguru import logger
from distllm.constants import HSTS_MAX_AGE

# Re-export route routers
from distllm.api.api_state import g as _g
from distllm.api.auth_deps import require_role
from distllm.api.errors import error_response, error_response_from_request, error_openapi_entry
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
    eval_router,
    metrics_history_router,
)
from distllm.api.routes.tools import router as tools_router

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


# ── Named constants ──────────────────────────────────────────────
CORS_MAX_AGE = 600                       # CORS preflight cache TTL (seconds)
REQUEST_SIZE_LIMIT = 100_000_000         # Max request body size (100 MB)
KEY_ROTATION_RATE_LIMIT = 60.0          # Min interval between key rotations (seconds)
CLUSTER_KEY_ROTATION_GRACE_PERIOD = 300  # Old key grace period after rotation (seconds, 5 min)
DEFAULT_COORD_PORT = 50050               # Default gRPC port for coordinator node server

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
        if origin == "*":
            if os.environ.get("DISTLLM_CORS_ALLOW_ALL", "").lower() in ("1", "true"):
                logger.critical(
                    "SECURITY: Wildcard CORS origins are enabled via DISTLLM_CORS_ALLOW_ALL. "
                    "This allows ANY origin to make cross-origin requests. "
                    "Do NOT use in production. Set DISTLLM_CORS_ALLOW_ALL=0 or unset to disable."
                )
                valid.append(origin)
            else:
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
    plugin_config = {"verify_plugins": getattr(state, "verify_plugins", False)}
    state.plugin_system = PluginSystem(config=plugin_config)
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
                                with scheduler._lock if hasattr(scheduler, '_lock') else nullcontext():
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

    # Log API key presence (never log the raw key)
    from distllm.core.api_key_store import get_api_key_store
    store = get_api_key_store()
    display_key = store.get_display_key()
    if display_key:
        fingerprint = hashlib.sha256(display_key.encode()).hexdigest()[:12]
        logger.info("API key configured (fingerprint: %s...)", fingerprint)
        logger.info("Use 'distllm config keys' to view or rotate keys.")
    else:
        logger.info("API keys loaded from config file. Use 'distllm config keys' to manage.")

    _start_ws_broadcaster()
    yield
    if state.plugin_system:
        state.plugin_system.stop_all()
    if state.ws_broadcast_task:
        state.ws_broadcast_task.cancel()


# OpenAPI docs enabled by default (set DISTLLM_DISABLE_DOCS=1 to disable)
_enable_docs = os.environ.get("DISTLLM_DISABLE_DOCS", "0").lower() not in ("1", "true")

app = FastAPI(
    lifespan=lifespan,
    title="Distributed LLM API",
    description="OpenAI-compatible REST API for distributed LLM inference across multiple machines using pipeline parallelism",
    version="0.4.0",
    docs_url="/docs" if _enable_docs else None,
    redoc_url="/redoc" if _enable_docs else None,
    openapi_url="/openapi.json" if _enable_docs else None,
    openapi_tags=[
        {"name": "chat", "description": "Chat completion endpoints with streaming support across distributed nodes"},
        {"name": "completion", "description": "Text completion endpoints with streaming support across distributed nodes"},
        {"name": "embedding", "description": "Text embedding and document reranking"},
        {"name": "system", "description": "Health checks, metrics, and cluster status"},
        {"name": "gossip", "description": "P2P gossip protocol for distributed KV cache discovery between nodes"},
        {"name": "models", "description": "Model registry: load, unload, inspect, warm up"},
        {"name": "auth", "description": "API-key and SSO token issuance/refresh/revoke"},
        {"name": "admin", "description": "Cluster administration (admin role required): nodes, config, logs"},
        {"name": "batch", "description": "Offline batch inference jobs with status polling and result streaming"},
        {"name": "evaluation", "description": "Benchmark evaluation runs and stored reports"},
        {"name": "federated", "description": "Federated fine-tuning rounds across participating nodes"},
        {"name": "leaderboard", "description": "Community benchmark leaderboard: submit, verify, vote, comment"},
        {"name": "marketplace", "description": "Compute marketplace: listings, jobs, provider earnings"},
        {"name": "monitoring", "description": "Metrics history, trends, thresholds, topology, waterfall traces"},
        {"name": "prompts", "description": "Prompt library: templates, versioning, sharing, forking"},
        {"name": "router", "description": "Model routing rules and hybrid-routing capabilities"},
        {"name": "tools", "description": "Registered tool/function handlers callable by the model"},
        {"name": "webrtc", "description": "WebRTC session negotiation for low-latency edge clients"},
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
    max_age=CORS_MAX_AGE,  # Cache preflight for 10 minutes
)

# ── OpenAPI security scheme ───────────────────────────────────────────────────
#
# Auth is enforced globally by AuthMiddleware (Bearer API key) rather than via
# per-route dependencies, so FastAPI cannot infer it. We post-process the
# schema to declare the scheme once and mark every authenticated operation so
# Swagger UI shows its "Authorize" button and lock icons per route.
#
# Paths listed here are exactly the AuthMiddleware exemptions — keep in sync
# with middleware.py.
_UNAUTHENTICATED_PATHS = frozenset({
    "/health",
    "/v1/health",
    "/v1/health/readiness",
    "/v1/health/liveness",
    "/ready",
    "/live",
    "/healthz",
    "/readyz",
    "/metrics",
})


def custom_openapi() -> dict:
    """Build the OpenAPI schema with an ``ApiKeyAuth`` security scheme."""
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
    )
    schema.setdefault("components", {})["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "http",
            "scheme": "bearer",
            "description": (
                "DistLLM API key sent as `Authorization: Bearer <key>`. "
                "Create keys with `distllm config keys`."
            ),
        },
    }
    for path, methods in schema.get("paths", {}).items():
        if path in _UNAUTHENTICATED_PATHS:
            continue
        for method, op in methods.items():
            if isinstance(op, dict) and method in ("get", "post", "put", "patch", "delete"):
                op.setdefault("security", [{"ApiKeyAuth": []}])

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi  # type: ignore[method-assign]

# Security headers middleware
# Version identifiers mirror the canonical APIVersion registry in
# api_versioning.py (date-based YYYY-MM strings).
_API_VERSIONS: dict[str, str] = {
    "v1": "2026-01",
    "v2": "2026-07",
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
        trust_proxy_tls = os.environ.get("DISTLLM_TRUST_PROXY_TLS", "").lower() in ("1", "true")
        if tls_enabled or trust_proxy_tls:
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

# CSRF Same-Origin middleware — validates Origin/Referer on state-changing
# requests to prevent Cross-Site Request Forgery. Runs after CORS and security
# headers but before AuthMiddleware so CSRF violations are caught early.
from distllm.api.csrf_middleware import CSRFSameOriginMiddleware
app.add_middleware(CSRFSameOriginMiddleware)


# Structured error responses
class ErrorResponse(BaseModel):
    """Standardized error response format."""
    error: str
    message: str
    type: str = "api_error"
    code: str | None = None
    request_id: str | None = None


# ── API Response Models ────────────────────────────────────────────


class ClusterNodeInfo(BaseModel):
    """Individual cluster node information."""
    node_id: str = ""
    host: str = ""
    port: int = 0
    healthy: bool = False
    start_layer: int = 0
    end_layer: int = 0
    gpu_name: str = ""
    gpu_memory_free: int = 0
    gpu_memory_total: int = 0
    gpu_sm_count: int = 0


class ClusterNodesResponse(BaseModel):
    """Cluster node topology response."""
    nodes: list[ClusterNodeInfo] = []
    total_layers: int = 0


class PipelineNodeHealth(BaseModel):
    """Per-node health info within a pipeline."""
    healthy: bool = False
    latency_ms: float | None = None
    start_layer: int = 0
    end_layer: int = 0
    gpu_name: str = ""


class PipelineHealthResponse(BaseModel):
    """Pipeline orchestrator health and metrics response."""
    status: str = ""
    nodes: dict[str, PipelineNodeHealth] = {}


class ChangelogEntry(BaseModel):
    """Single changelog entry."""
    date: str = ""
    change: str = ""


class ChangelogResponse(BaseModel):
    """API changelog response."""
    version: str = ""
    date: str = ""
    changes: list[ChangelogEntry] = []


class RotateClusterKeyResponse(BaseModel):
    """Successful cluster-key rotation response."""
    status: str = "ok"
    new_key: str = Field(default="", description="The new cluster key — distribute to all nodes immediately")
    detail: str = ""


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

# Created at module load so middleware mounted below (ObservabilityMiddleware)
# can hold a direct reference; _init_observability reuses it at startup.
state.metrics_exporter = DistLLMPrometheusExporter()


class PluginHookMiddleware(BaseHTTPMiddleware):
    """Dispatch plugin hooks around every request.

    Registered before ``AuthMiddleware`` in code so that AuthMiddleware (which
    is prepended later and therefore runs outermost/first) populates
    ``request.state`` (api_key_role, api_key_id, tenant...) before the plugin
    hooks dispatch.  Captures request context, dispatches ``on_request``,
    delegates to the next handler, then dispatches ``on_response`` or
    ``on_error`` based on the outcome.
    """

    SKIP_PATHS = {"/health", "/ready", "/live", "/healthz", "/readyz", "/metrics",
                  "/docs", "/openapi.json", "/redoc", "/ws", "/dashboard"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        plugin_sys: PluginSystem | None = getattr(state, "plugin_system", None)
        if plugin_sys is None or request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        # Build plugin context from the request
        # SECURITY: Only pass a redacted auth fingerprint in the general context
        # to prevent arbitrary plugins from leaking credentials via logs
        # or external storage. The auth plugin receives the full header
        # separately via its own dispatch hook.
        auth_header_raw = request.headers.get("authorization", "")
        auth_fingerprint = ""
        if auth_header_raw.startswith("Bearer "):
            token = auth_header_raw[7:]
            if len(token) > 8:
                import hashlib
                auth_fingerprint = f"Bearer {token[:8]}...{hashlib.sha256(token.encode()).hexdigest()[:8]}"
            else:
                auth_fingerprint = "Bearer (invalid)"
        elif auth_header_raw:
            auth_fingerprint = "(non-bearer)"
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
            "auth_fingerprint": auth_fingerprint,
            # Internal: full auth header for auth plugin JWT validation.
            # WARNING: Do not log or persist this value. It contains the
            # full bearer token / API key.
            "_auth_header": auth_header_raw,
        }

        # Allow plugins to modify/reject the request
        plugin_ctx = plugin_sys.dispatch_on_request(req_ctx)
        reject = plugin_ctx.get("_reject")
        if reject:
            status = reject.get("status", 429)
            error_body: dict[str, Any] = {
                "message": reject.get("reason", "Request rejected"),
                "type": "auth_error" if status in (401, 403) else "rate_limit",
            }
            if "retry_after" in reject:
                error_body["retry_after"] = reject["retry_after"]
            return JSONResponse(
                status_code=status,
                content={"error": error_body},
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






def _init_observability() -> None:
    """Initialize tracing, logging, metrics exporter."""
    setup_logging(level="INFO", json_format=True)

    # Tracing sampling: default to 10% (head-based) in production.
    # Set DISTLLM_TRACE_SAMPLE_RATE=1.0 for full traces during debugging.
    import os as _os
    _trace_sample_rate = float(_os.environ.get("DISTLLM_TRACE_SAMPLE_RATE", "0.1"))
    setup_tracing(
        service_name="distllm-api",
        sampling_strategy="head",
        sampling_ratio=min(1.0, max(0.0, _trace_sample_rate)),
    )

    # Reuse the exporter created at module load so the middleware mounted above
    # keeps recording into the same registry.
    if getattr(state, "metrics_exporter", None) is None:
        state.metrics_exporter = DistLLMPrometheusExporter()


# ── Middleware Stack ─────────────────────────────────────────────────────────
#
# STARLETTE MIDDLEWARE vs PLUGIN SYSTEM — when to use each
#
# Starlette Middleware (app.add_middleware):
#   Use for low-level, ordering-sensitive request/response pipeline processing.
#   - Runs in the FastAPI/Starlette pipeline (reverse registration order).
#   - Has direct access to the raw ASGI scope, Request, and Response objects.
#   - Best for infrastructure cross-cutting concerns that MUST be at a
#     specific position in the pipeline (auth before rate-limit, etc.).
#   - Examples: auth, CORS, security headers, timeouts, backpressure,
#     circuit breakers, request size limits, request IDs, deduplication,
#     prompt injection detection.
#   - Drawback: third parties cannot inject middleware; order is fragile.
#
# Plugin System (PluginHookMiddleware + PluginSystem):
#   Use for extensible, swappable, lifecycle-managed behavior.
#   - Runs via a single middleware entry point (PluginHookMiddleware, see below)
#     that dispatches ``on_request`` / ``on_response`` / ``on_error`` hooks
#     to all active plugins.
#   - Has lifecycle management (init -> start -> stop).
#   - Can be discovered from the filesystem, installed from PyPI, or
#     registered programmatically — third-party extensible.
#   - Best for: custom auth schemes, custom audit logging, custom metrics,
#     custom health probes, anything that should be swappable/installable.
#   - Drawback: does NOT have raw ASGI scope access; runs at a fixed point
#     in the middleware stack (PluginHookMiddleware position).
#
# Rule of thumb:
#   - If the concern MUST live at a precise position in the pipeline (e.g.
#     auth must run before rate-limiting uses request.state), use Starlette
#     middleware.
#   - If the concern should be swappable, installable, or addable by users
#     without touching the codebase, use a plugin.
#   - If the concern needs lifecycle management beyond request/response
#     (e.g. background loops, model load/unload hooks), use a plugin.
# ─────────────────────────────────────────────────────────────────────────────

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

# DedupMiddleware — collapses identical concurrent POST requests. Registered
# BEFORE AuthMiddleware so it runs AFTER auth (add_middleware prepends, so the
# first-registered middleware executes last/innermost): an unauthenticated
# replay must never hit a cached response, and the fingerprint is namespaced by
# the authenticated api_key_id (see dedup.py).
from distllm.api.dedup import DedupMiddleware
app.add_middleware(DedupMiddleware)

# PluginHookMiddleware runs via its own ASGI dispatch; it must execute
# AFTER AuthMiddleware populates request.state (auth fingerprint, role,
# tenant), so it is registered BEFORE AuthMiddleware in code (add_middleware
# prepends, so the last-registered middleware runs first/outermost and the
# first-registered runs last/innermost — AuthMiddleware runs before
# PluginHookMiddleware on the way in).
app.add_middleware(PluginHookMiddleware)

# AuthMiddleware registered AFTER PluginHookMiddleware so it runs first
# (outermost) and populates request.state before plugin hooks dispatch.
app.add_middleware(AuthMiddleware)

# RequestIDMiddleware registered after AuthMiddleware so it runs before it
# on incoming requests, ensuring request.state.request_id is set before any
# middleware that reads it (e.g. AuthMiddleware error responses).
app.add_middleware(RequestIDMiddleware)

# RequestRateLimitMiddleware — per-IP request rate limiting
# Registered after RequestIDMiddleware so request.state.request_id is
# available for rate-limit error responses.
app.add_middleware(RequestRateLimitMiddleware)

# Prompt Injection Detection Middleware — detects and mitigates prompt
# injection attacks (BLOCK / SANITIZE / FLAG). Runs after auth so blocked
# requests don't consume resources, but before the main route handlers.
from distllm.api.prompt_injection import PromptInjectionMiddleware
app.add_middleware(PromptInjectionMiddleware)

# ContentModerationMiddleware — intercepts requests/responses for toxicity,
# PII, jailbreak, and topic-policy enforcement.  Enabled when the
# DISTLLM_MODERATION=1 environment variable is set.
#
# Runs after AuthMiddleware so request.state is populated, and before
# route handlers so blocked requests are rejected early.
if os.environ.get("DISTLLM_MODERATION", "0") == "1":
    from distllm.api.middleware import ContentModerationMiddleware
    app.add_middleware(ContentModerationMiddleware)
    logger.info("ContentModerationMiddleware registered (DISTLLM_MODERATION=1)")

# Docs auth middleware — requires admin role for OpenAPI documentation pages
class DocsAuthMiddleware(BaseHTTPMiddleware):
    """Require admin role to access OpenAPI documentation.

    Protects ``/docs``, ``/redoc``, and ``/openapi.json`` from
    unauthorized access.  Must be registered **after** AuthMiddleware
    so that ``request.state.api_key_role`` is populated.
    """

    DOCS_PATHS = frozenset({"/docs", "/redoc", "/openapi.json"})

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.DOCS_PATHS:
            role = getattr(request.state, "api_key_role", None)
            if role != "admin":
                return _error_response(
                    status_code=403,
                    error="Forbidden",
                    message="Admin access required for API documentation. "
                            "Authenticate with an admin API key.",
                    type="auth_error",
                    request=request,
                )
        return await call_next(request)

# DocsAuthMiddleware — requires admin role on /docs, /redoc, /openapi.json
# Registered after AuthMiddleware so request.state.api_key_role is set.
app.add_middleware(DocsAuthMiddleware)


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


app.add_middleware(RequestSizeLimitMiddleware, max_size=REQUEST_SIZE_LIMIT)


# Backpressure middleware
class BackpressureMiddleware(BaseHTTPMiddleware):
    """Reject requests when system is under heavy load.

    Graduated backpressure tiers:
      - 500-800 pending:  Retry-After: 1s, process normally
      - 800-1000 pending: Retry-After: 5s, shed low-priority requests
      - 1000+ pending:    503 with Retry-After: 30s
    """

    # Paths exempt from backpressure
    EXEMPT_PATHS = frozenset({
        "/health", "/ready", "/live", "/metrics",
        "/docs", "/openapi.json", "/redoc",
    })

    # Low-priority paths shed first under moderate load
    LOW_PRIORITY_PATHS = frozenset({
        "/v1/embeddings",
        "/v1/batch",
        "/api/defrag",
    })

    _TIER_LOW = 500       # start adding Retry-After headers
    _TIER_MED = 800       # start shedding low-priority
    _TIER_HIGH = 1000     # hard reject all non-exempt

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip backpressure for health/metrics endpoints
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        # Check if scheduler is overloaded
        if state.coordinator and state.coordinator.scheduler:
            try:
                stats = state.coordinator.scheduler.stats()
                pending = stats.get("pending_requests", 0)
                if isinstance(pending, (int, float)):
                    if pending >= self._TIER_HIGH:
                        # Hard shed: reject with 503
                        resp = _error_response(
                            status_code=503,
                            error="Service Unavailable",
                            message=f"System overloaded: {int(pending)} pending requests",
                            type="backpressure_error",
                            request=request,
                        )
                        resp.headers["Retry-After"] = "30"
                        return resp

                    if pending >= self._TIER_MED:
                        # Shed low-priority requests
                        path = request.url.path.rstrip("/")
                        is_low_priority = any(
                            path.startswith(lp) for lp in self.LOW_PRIORITY_PATHS
                        )
                        if is_low_priority:
                            resp = _error_response(
                                status_code=503,
                                error="Service Unavailable",
                                message=f"Low-priority request shed: {int(pending)} pending requests",
                                type="backpressure_error",
                                request=request,
                            )
                            resp.headers["Retry-After"] = "5"
                            return resp

                    if pending >= self._TIER_LOW:
                        # Advisory: add Retry-After header but process normally
                        response = await call_next(request)
                        response.headers["Retry-After"] = "1"
                        return response
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

# CircuitBreakerMiddleware — protects downstream from cascade failures.
# Registered after BackpressureMiddleware so it runs between
# BackpressureMiddleware and the route handlers.
from distllm.api.circuit_breaker_middleware import CircuitBreakerMiddleware
app.add_middleware(CircuitBreakerMiddleware)


# ── Plugin system ───────────────────────────────────────────────────────────
#
# Plugins are for extensible, swappable behavior (custom auth, custom logging,
# custom metrics, health probes, etc.) that users or third parties can add or
# remove without modifying the core codebase.  They run through the single
# PluginHookMiddleware entry point below.
#
# See the "Middleware Stack" guide above for a full comparison between
# Starlette middleware and the plugin system.

def _init_plugins(ps: PluginSystem) -> None:
    """Register + load + init + start built-in plugins."""
    for cls in (RateLimitPlugin, AuditLogPlugin, MetricsPlugin, HealthPlugin, AuthPlugin):
        ps.register(cls)
    ps.load_all()
    ps.init_all()
    ps.start_all()
    logger.info(f"Plugin system ready: {len(ps.list_plugins())} plugins active")


# Include route routers
app.include_router(chat_router)
app.include_router(chat_v2_router)
app.include_router(completion_router)
app.include_router(embeddings_router)
app.include_router(tools_router)
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
app.include_router(eval_router)
app.include_router(metrics_history_router)
# Cost tracking middleware
try:
    from distllm.api.cost_middleware import CostTrackingMiddleware
    app.add_middleware(CostTrackingMiddleware)
except ImportError:
    pass

# Body cache middleware — outermost, reads request body ONCE and caches it so
# downstream middlewares (auth, rate-limit, prompt-injection) do not each
# re-read and re-parse the body independently.
# Registered after CostTrackingMiddleware so it wraps all inner middleware.
from distllm.api.body_cache_middleware import BodyCacheMiddleware
app.add_middleware(BodyCacheMiddleware)

# Tracing middleware — outermost, W3C Trace-Context propagation.
# Registered LAST so it wraps the entire pipeline and captures the full
# request lifecycle (auth, rate-limiting, inference phases).
from distllm.api.tracing_middleware import TracingMiddleware
app.add_middleware(TracingMiddleware)

# Observability middleware — records RED (rate/errors/duration) metrics for
# every request.  Registered outermost so it sees the full lifecycle including
# auth rejections.
app.add_middleware(
    ObservabilityMiddleware,
    metrics_exporter=state.metrics_exporter,
)

# SSO middleware — outer SSO/JWT authentication that falls through to
# API-key auth (AuthMiddleware) when no valid SSO token is present.
# Enables POST /v1/auth/{token,refresh,revoke}; a no-op unless DISTLLM_SSO_*
# env vars configure a provider. Registered after tracing so it runs first,
# matching AuthMiddleware's `request.state.auth_method == "sso"` skip path.
from distllm.api.sso_middleware import setup_sso
setup_sso(app)

# --- Dashboard & WebSocket ---

from pathlib import Path
from fastapi.staticfiles import StaticFiles


# Mount dashboard static files (CSS, JS)
dashboard_static = Path(__file__).parent.parent / "dashboard" / "static"
dashboard_static.mkdir(parents=True, exist_ok=True)
app.mount("/dashboard/static", StaticFiles(directory=str(dashboard_static)), name="dashboard_static")


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
    # SECURITY FIX: Removed query param token support — tokens in URLs are logged
    # in server access logs, proxy logs, browser history, and analytics.
    # Token from Authorization header, or (browser WS clients) from a
    # Sec-WebSocket-Protocol subprotocol pair: ["Bearer", "<key>"].
    auth_token = None
    auth_header = websocket.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        auth_token = auth_header[7:]
    else:
        sec_ws = websocket.headers.get("sec-websocket-protocol", "")
        parts = [p.strip() for p in sec_ws.split(",") if p.strip()]
        if len(parts) >= 2 and parts[0] == "Bearer":
            auth_token = parts[1]

    # SECURITY: Reject connection when no API keys are configured
    from distllm.core.api_key_store import get_api_key_store
    store = get_api_key_store()
    if store.get_key_count() == 0:
        logger.warning("WebSocket connection rejected: no API keys configured (auth disabled)")
        await websocket.close(code=4001, reason="Server authentication not configured")
        return
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
    # SECURITY: Authenticate WebSocket connection via header, or (browser
    # WS clients) via Sec-WebSocket-Protocol subprotocol pair.
    # Query param tokens are insecure (logged in URLs, proxies, browser history)
    auth_token = None
    auth_header = websocket.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        auth_token = auth_header[7:]
    else:
        sec_ws = websocket.headers.get("sec-websocket-protocol", "")
        parts = [p.strip() for p in sec_ws.split(",") if p.strip()]
        if len(parts) >= 2 and parts[0] == "Bearer":
            auth_token = parts[1]

    # SECURITY: Reject connection when no API keys are configured
    from distllm.core.api_key_store import get_api_key_store
    store = get_api_key_store()
    if store.get_key_count() == 0:
        logger.warning("Metrics WebSocket rejected: no API keys configured (auth disabled)")
        await websocket.close(code=4001, reason="Server authentication not configured")
        return
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
    html_path = Path(__file__).parent.parent / "dashboard" / "static" / "index.html"
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
    response_model=ClusterNodesResponse,
    dependencies=[Depends(require_role("auditor"))],
)
async def api_cluster_nodes(request: Request) -> dict:
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

    # SECURITY: Cluster key is required
    local_key = getattr(coord.config, 'cluster_key', None)
    if not local_key:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "error": "Federation disabled: no cluster_key configured"},
        )
    received_key = request.headers.get("X-Cluster-Key", "")
    if not received_key:
        return JSONResponse(
            status_code=401,
            content={"status": "unauthorized", "error": "missing cluster key"},
        )

    # Accept current key or pending old key (grace period during rotation)
    key_valid = hmac.compare_digest(received_key, local_key)
    if not key_valid:
        pending_old = getattr(coord, "_pending_old_cluster_key", None)
        rotation_time = getattr(coord, "_key_rotation_time", 0)
        grace_expiry = rotation_time + CLUSTER_KEY_ROTATION_GRACE_PERIOD  # 5-minute grace period
        if pending_old and time.time() < grace_expiry:
            key_valid = hmac.compare_digest(received_key, pending_old)
    if not key_valid:
        return JSONResponse(
            status_code=401,
            content={"status": "unauthorized", "error": "invalid cluster key"},
        )

    # Validate heartbeat body with Pydantic
    from pydantic import BaseModel, Field, field_validator

    class FederationHeartbeat(BaseModel):
        active_requests: int = Field(default=0, ge=0)
        pending_requests: int = Field(default=0, ge=0)
        gpu_utilization: float = Field(default=0.0, ge=0.0, le=100.0)
        queue_depth: int | None = Field(default=None, ge=0)

        @field_validator("gpu_utilization")
        @classmethod
        def validate_utilization_range(cls, v: float) -> float:
            return max(0.0, min(100.0, v))

    raw_body = await request.json()
    heartbeat = FederationHeartbeat(**raw_body)

    if hasattr(coord, '_federation') and coord._federation:
        try:
            for pid, peer in coord._federation._peers.items():
                if pid != coord._federation.config.cluster_id:
                    coord._federation._load_balancer.report_load(
                        cluster_id=pid,
                        active_requests=heartbeat.active_requests,
                        pending_requests=heartbeat.pending_requests,
                        gpu_utilization=heartbeat.gpu_utilization,
                        queue_depth=heartbeat.queue_depth or heartbeat.pending_requests,
                    )
        except Exception:
            logger.opt(exception=True).debug("Federation heartbeat processing failed")
    return {"status": "ok"}


@app.post(
    "/api/cluster/rotate-key",
    summary="Rotate cluster key",
    description="Rotate the cluster authentication key. The new key is applied immediately "
                "and must be distributed to all nodes. Supports a grace period during which "
                "both the old and new key are accepted. Rate-limited to one rotation per 60 seconds.",
    response_description="The new cluster key and grace-period details",
    response_model=RotateClusterKeyResponse,
    responses={
        401: error_openapi_entry("Missing or invalid API key", type_="auth_error", code="authentication_error"),
        403: error_openapi_entry("Admin role required", type_="auth_error", code="forbidden"),
        429: {
            "description": "Rotation rate-limited (max once per 60s)",
            "content": {
                "application/json": {
                    "example": {
                        "status": "rate_limited",
                        "detail": "Key rotation rate limited. Retry in 42s.",
                        "retry_after_s": 42,
                    },
                },
            },
        },
    },
    dependencies=[Depends(require_role("admin"))],
)
async def rotate_cluster_key(request: Request) -> dict:
    """Rotate the cluster authentication key.

    Generates a new cryptographically random key, applies it with a
    configurable grace period during which both old and new keys are
    accepted for HMAC verification.

    Rate-limited: max 1 rotation per 60 seconds to prevent
    rolling-DoS attacks (CWE-799).
    """
    # Rate limit: max 1 rotation per 60 seconds
    now = time.time()
    last_rotation = getattr(state, "_last_key_rotation_time", 0.0)
    if now - last_rotation < KEY_ROTATION_RATE_LIMIT:
        remaining = int(60.0 - (now - last_rotation))
        return JSONResponse(
            status_code=429,
            content={
                "status": "rate_limited",
                "detail": f"Key rotation rate limited. Retry in {remaining}s.",
                "retry_after_s": remaining,
            },
        )
    state._last_key_rotation_time = now

    import secrets
    new_key = secrets.token_urlsafe(32)

    coord = state.coordinator
    if coord is None:
        return {"status": "error", "detail": "coordinator not available"}

    old_key = getattr(coord.config, 'cluster_key', None)
    # Store old key as pending_old_key for grace period validation
    if old_key:
        coord._pending_old_cluster_key = old_key
        coord._key_rotation_time = time.time()

    coord.config.cluster_key = new_key
    logger.warning(
        f"Cluster key rotated by admin. "
        f"New key fingerprint: {hashlib.sha256(new_key.encode()).hexdigest()[:16]}"
    )
    return {
        "status": "ok",
        "new_key": new_key,
        "detail": "Save this key and distribute it to all nodes. "
                  "The previous key will be accepted for 5 minutes.",
    }


def _ha_secret_rejection(request: Request) -> JSONResponse | None:
    """Return a 403 rejection when the HA shared secret is absent/mismatched.

    HA replication endpoints fail CLOSED: with ``DISTLLM_HA_SECRET`` unset there
    is no way to authenticate the sender, so the endpoint refuses rather than
    accepting arbitrary leader-election or state-snapshot input from an
    unauthenticated socket.
    """
    expected_secret = os.environ.get("DISTLLM_HA_SECRET", "")
    if not expected_secret:
        return JSONResponse(
            status_code=403,
            content={"status": "error", "detail": "HA shared secret not configured"},
        )
    received_secret = request.headers.get("X-HA-Secret", "")
    if not hmac.compare_digest(received_secret, expected_secret):
        return JSONResponse(
            status_code=403,
            content={"status": "error", "detail": "invalid HA secret"},
        )
    return None


@app.post(
    "/api/v1/ha/snapshot",
    summary="HA state snapshot",
    description="Receive a coordinator state snapshot from the leader for HA standby replication.",
    include_in_schema=False,
)
async def ha_state_snapshot(request: Request) -> dict:
    """Receive and apply a coordinator state snapshot from the HA leader.

    Authenticated via a shared HA secret header that both leader and
    standby coordinators use, preventing arbitrary state injection.
    """
    coord = state.coordinator
    if coord is None:
        return {"status": "error", "detail": "coordinator not available"}

    # SECURITY: Require HA shared secret (fail closed when not configured)
    rejection = _ha_secret_rejection(request)
    if rejection is not None:
        return rejection

    try:
        raw = await request.json()
        if not isinstance(raw, dict):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "detail": "request body must be a JSON object"},
            )

        nodes = raw.get("nodes", {})
        if not isinstance(nodes, dict):
            nodes = {}
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        validated: dict = {
            "nodes": nodes,
            "metadata": metadata,
        }
        coord.apply_state_snapshot(validated)
        return {"status": "ok", "applied_nodes": len(nodes)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post(
    "/api/v1/ha/heartbeat",
    summary="HA leader-election heartbeat",
    description="Receive a leader-election heartbeat from a peer coordinator, "
                "refreshing its last-seen time and election term.",
    include_in_schema=False,
)
async def ha_heartbeat(request: Request) -> dict:
    """Receive a leader-election heartbeat from a peer coordinator.

    Refreshes the sender's liveness and adopts its term if higher, which is
    how the Raft-like leader election in ``RayFaultTolerance`` stays
    consistent across coordinators. Authenticated with the same shared HA
    secret as the snapshot endpoint when ``DISTLLM_HA_SECRET`` is configured.
    """
    coord = state.coordinator
    if coord is None:
        return {"status": "error", "detail": "coordinator not available"}

    # SECURITY: Require the same HA shared secret as the snapshot endpoint
    # (fail closed when not configured).
    rejection = _ha_secret_rejection(request)
    if rejection is not None:
        return rejection

    try:
        raw = await request.json()
        if not isinstance(raw, dict):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "detail": "request body must be a JSON object"},
            )

        sender_id = str(raw.get("coordinator_id", ""))
        if not sender_id:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "detail": "missing coordinator_id"},
            )
        try:
            term = int(raw.get("term", 0))
        except (TypeError, ValueError):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "detail": "term must be an integer"},
            )
        state_raw = raw.get("state")
        peer_state = state_raw if isinstance(state_raw, dict) else None

        election = getattr(getattr(coord, "_election", None), "_ha_election", None)
        if election is None:
            return {"status": "error", "detail": "HA election not enabled"}

        peer = election.handle_heartbeat_request(sender_id, term, peer_state)
        return {"status": "ok", "peer": peer}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get(
    "/api/pipeline/health",
    summary="Pipeline orchestrator health and metrics",
    description="Return pipeline health status, node execution metrics, transport info, and configuration.",
    response_description="Pipeline health and metrics",
    response_model=PipelineHealthResponse,
    dependencies=[Depends(require_role("auditor"))],
)
async def api_pipeline_health(request: Request) -> dict:
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
    response_model=dict,
    dependencies=[Depends(require_role("auditor"))],
)
async def api_cluster_reputation(request: Request) -> dict:
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
    dependencies=[Depends(require_role("auditor"))],
)
async def api_collector_metrics(request: Request) -> dict:
    """Return current collector metrics snapshot."""
    return get_collector().summary()


@app.get(
    "/api/metrics/stream",
    summary="Metrics SSE stream",
    description="Subscribe to a real-time metrics stream via Server-Sent Events. Use query parameters to filter metric categories and set update interval.",
    response_description="Event stream of structured metrics JSON.",
    dependencies=[Depends(require_role("auditor"))],
)
async def api_metrics_stream(
    request: Request,
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
    dependencies=[Depends(require_role("auditor"))],
)
async def api_waterfall(request: Request, limit: int = 50) -> list:
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
    response_model=dict,
    dependencies=[Depends(require_role("auditor"))],
)
async def api_continuum_stats(request: Request) -> dict:
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
    response_model=dict,
    dependencies=[Depends(require_role("auditor"))],
)
async def api_cost_summary(request: Request, tenant_id: str = "") -> dict:
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
    dependencies=[Depends(require_role("auditor"))],
)
async def api_cost_history(request: Request, limit: int = 100) -> list:
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
    dependencies=[Depends(require_role("auditor"))],
)
async def api_streaming_cost_stats(request: Request) -> dict:
    """Return streaming cost tracker statistics."""
    try:
        from distllm.core.streaming_cost import get_streaming_cost_tracker
        return get_streaming_cost_tracker().get_stats()
    except ImportError:
        return {"status": "not_available"}


# ── API Changelog ─────────────────────────────────────────────────

_API_CHANGELOG_DATA: dict[str, str | list[dict[str, str]]] = {
    "version": "0.4.0",
    "date": "2026-07-20",
    "changes": [
        {"date": "2026-07-20", "change": "OpenAPI docs enabled by default with admin-level auth gate on /docs"},
        {"date": "2026-07-18", "change": "Added cluster key rotation with grace period"},
        {"date": "2026-07-15", "change": "Added HA state snapshot endpoint for standby replication"},
        {"date": "2026-07-12", "change": "Added cost tracking and streaming cost endpoints"},
        {"date": "2026-07-10", "change": "Added edge-to-cloud continuum statistics endpoint"},
        {"date": "2026-07-08", "change": "Added backpressure middleware with graduated shedding tiers"},
        {"date": "2026-07-05", "change": "Added prompt injection detection and mitigation middleware"},
        {"date": "2026-07-03", "change": "Added circuit breaker middleware for downstream protection"},
        {"date": "2026-07-01", "change": "Extended federation with load-balancer and heartbeat protocol"},
        {"date": "2026-06-28", "change": "Added WebSocket metrics streaming at /ws/metrics"},
        {"date": "2026-06-25", "change": "Added SSE metrics stream at /api/metrics/stream"},
    ],
}


@app.get(
    "/api/changelog",
    summary="API changelog",
    description="Return recent API changes and version history.",
    response_description="API changelog with version, date, and list of changes",
    response_model=ChangelogResponse,
    dependencies=[Depends(require_role("auditor"))],
)
async def api_changelog(request: Request) -> dict:
    """Return recent API changelog entries."""
    return _API_CHANGELOG_DATA


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
    coord_port = DEFAULT_COORD_PORT  # Default coordinator gRPC port
    try:
        from distllm.dist.node_service import NodeServer

        # Determine TLS config from settings
        coord_tls = False
        coord_cert_file: str | None = None
        coord_key_file: str | None = None
        coord_ca_cert: str | None = None
        if settings and settings.tls.enabled:
            coord_tls = True
            coord_cert_file = settings.tls.cert_file
            coord_key_file = settings.tls.key_file
            coord_ca_cert = settings.tls.ca_cert_file

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
        coord._node_server = NodeServer(
            coord._node_wrapper,
            port=coord_port,
            max_workers=4,
            cluster_key=getattr(coord.config, "cluster_key", None),
        )
        coord._node_server.start(
            use_tls=coord_tls,
            cert_file=coord_cert_file,
            key_file=coord_key_file,
            ca_cert=coord_ca_cert,
        )
        if coord_tls:
            logger.info(f"Coordinator gRPC server started on port {coord_port} with TLS for worker connections")
        else:
            logger.info(f"Coordinator gRPC server started on port {coord_port} for worker connections (no TLS)")
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
    parser.add_argument("--verify-plugins", action="store_true", help="Require SHA-256 hash verification for all plugins (fail-closed)")
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

    # Store verify-plugins flag in shared state for lifespan to read
    state.verify_plugins = getattr(args, "verify_plugins", False)

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
