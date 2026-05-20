"""OpenAI-compatible REST API for distributed LLM inference."""

import argparse
import asyncio
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware

from distllm.core.coordinator import Coordinator
from distllm.api.middleware import AuthMiddleware, RequestIDMiddleware
from distllm.api.rate_limiter import RateLimiter
from distllm.api.rate_limit_middleware import RateLimitMiddleware
from distllm.api.observability_middleware import ObservabilityMiddleware
from distllm.dashboard.ws_handler import (
    manager,
    metrics_broadcaster,
    get_collector,
    parse_client_message,
    stream_metrics_sse,
    KNOWN_METRIC_CATEGORIES,
)
from fastapi.responses import StreamingResponse
from distllm.core.monitor import SystemMonitor
from distllm.config.settings import DistLLMSettings
from distllm.config.loader import load_config_file
from distllm.communication.grpc import set_debug_mode
from distllm.observability.tracing import setup_tracing
from distllm.observability.logging import setup_logging
from distllm.observability.exporter import DistLLMPrometheusExporter
from distllm.monitoring.anomaly_detector import AnomalyDetector
from distllm.scheduling.cost_aware_scaler import GPUCostTracker
from loguru import logger
from distllm.constants import HSTS_MAX_AGE
from distllm.core.quantization_selector import build_quantization_config
from distllm.tenant.store import TenantStore
from distllm.tenant.middleware import TenantMiddleware
from distllm.tenant.rate_limiter import TenantRateLimiter
from distllm.tenant.billing import UsageMeter
from distllm.tenant.router import TenantModelRouter

# Re-export route routers
from distllm.api.errors import error_response, error_response_from_request
from distllm.api.routes import (
    chat_router,
    completion_router,
    embeddings_router,
    adapters_router,
    health_router,
    versions_router,
    multi_model_router,
    batch_router,
    audio_router,
    images_router,
    moderations_router,
    files_router,
    fine_tuning_router,
    gossip_router,
    optimization_router,
    rag_router,
    agent_router,
    disagg_router,
    pipeline_router,
    debug_router,
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
from distllm.api.routes.adapters import AdapterLoadRequest
from distllm.api.routes.health import ModelInfo, ModelList, ParamUpdateRequest

# Re-export streaming helpers for backward compatibility
from distllm.api.streaming import (
    _stream_event,
    _stream_start_event,
    _stream_stop_event,
    _generate_tokens,
    _stream_response,
)


def _get_cors_origins() -> list[str]:
    """Get CORS origins from env var (falls back to settings default).
    
    Security: Rejects wildcard origins unless DISTLLM_DEV_MODE=1 is set.
    Validates that all origins are well-formed URLs.
    """
    raw = os.environ.get("DISTLLM_CORS_ORIGINS")
    origins = []
    if raw:
        origins = [o.strip() for o in raw.split(",") if o.strip()]
    else:
        origins = [o.strip() for o in DistLLMSettings().coordinator.cors_origins.split(",") if o.strip()]

    valid = []
    for origin in origins:
        if origin == "*" and os.environ.get("DISTLLM_DEV_MODE") != "1":
            allowed = "http://localhost:3000,http://localhost:8080"
            valid.extend(o.strip() for o in allowed.split(",") if o.strip())
            continue
        valid.append(origin)
    return valid


ALLOWED_ORIGINS = _get_cors_origins()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize services on startup, clean up on shutdown."""
    _init_observability()
    _start_ws_broadcaster()
    yield
    if state.ws_broadcast_task:
        state.ws_broadcast_task.cancel()


app = FastAPI(
    lifespan=lifespan,
    title="Distributed LLM API",
    description="OpenAI-compatible REST API for distributed LLM inference across multiple machines using pipeline parallelism",
    version="0.4.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "chat", "description": "Chat completion endpoints with streaming support"},
        {"name": "completion", "description": "Text completion endpoints with streaming support"},
        {"name": "embedding", "description": "Text embedding and document reranking for RAG pipelines"},
        {"name": "adapters", "description": "LoRA adapter management for multi-adapter inference"},
        {"name": "system", "description": "Health checks, Kubernetes probes, metrics, and generation parameter updates"},
        {"name": "versions", "description": "Model versioning, A/B testing, shadow deployments, and blue-green rollouts"},
        {"name": "multi-model", "description": "Multi-model hot-swap: load, unload, and manage multiple models concurrently"},
        {"name": "batch", "description": "Async batch processing of chat and completion requests via JSONL files"},
        {"name": "audio", "description": "Speech-to-text transcription and text-to-speech synthesis"},
        {"name": "images", "description": "Image generation, editing, and variation using diffusion models"},
        {"name": "moderations", "description": "Content moderation: classify text for harmful content across multiple categories"},
        {"name": "files", "description": "File upload and management for fine-tuning, batch processing, and RAG"},
        {"name": "fine-tuning", "description": "Model fine-tuning job creation, monitoring, and management"},
        {"name": "gossip", "description": "P2P gossip protocol for distributed KV cache discovery between nodes"},
        {"name": "optimization", "description": "Self-optimizing engine: performance tuning suggestions and auto-tuning"},
        {"name": "rag", "description": "Retrieval-Augmented Generation: document ingestion, retrieval, and prompt enrichment"},
        {"name": "agent", "description": "ReAct agent execution: run goal-oriented agents with custom tool definitions"},
        {"name": "disagg", "description": "Disaggregated serving: separated prefill/decode node pools for improved throughput"},
        {"name": "pipeline", "description": "Model pipeline composition: chain multiple models in sequential workflows"},
        {"name": "debug", "description": "Debug and replay tools: request inspection, replay, and deterministic generation"},
        {"name": "tenants", "description": "Multi-tenant management: CRUD, usage metering, billing, and quota enforcement"},
    ],
)

# Security: Configure CORS with explicit allowed origins
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # Security: Disable credentials for CORS
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-Request-Timeout", "X-Priority"],
    max_age=600,  # Cache preflight for 10 minutes
)

# Security headers middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        # HSTS only makes sense when TLS is enabled
        tls_enabled = os.environ.get("DISTLLM_TLS_ENABLED", "false").lower() == "true"
        if tls_enabled:
            response.headers["Strict-Transport-Security"] = f"max-age={HSTS_MAX_AGE}; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
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
) -> JSONResponse:
    """Build a standardized error response."""
    return error_response_from_request(status_code, error, message, type, request)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Convert HTTPException to structured error response."""
    return _error_response(
        status_code=exc.status_code,
        error=f"HTTP {exc.status_code}",
        message=exc.detail,
        type="http_error",
        request=request,
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions with structured response."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return _error_response(
        status_code=500,
        error="Internal Server Error",
        message="An unexpected error occurred. Please try again later.",
        type="internal_error",
        request=request,
    )


# Application state — encapsulates all shared mutable state for lifecycle management
class AppState:
    """Manages shared application state across the API server.

    All mutable module-level state is consolidated here to prevent shared
    mutable globals that could be mutated across requests.
    """

    def __init__(self):
        self.coordinator: Coordinator | None = None
        self.monitor: SystemMonitor | None = None
        self.startup_time: float = time.time()
        self.metrics_exporter: DistLLMPrometheusExporter | None = None
        self.cost_tracker: GPUCostTracker | None = None
        self.anomaly_detector: AnomalyDetector | None = None
        self.tenant_store: TenantStore | None = None
        self.tenant_rate_limiter: TenantRateLimiter | None = None
        self.usage_meter: UsageMeter | None = None
        self.tenant_router: TenantModelRouter | None = None
        self.rate_limiter: RateLimiter | None = None
        self.ws_broadcast_task = None

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.startup_time

    def set(self, coord: Coordinator, mon: SystemMonitor) -> None:
        """Set the coordinator and monitor instances."""
        self.coordinator = coord
        self.monitor = mon

    def clear(self) -> None:
        """Reset state (useful for testing)."""
        self.coordinator = None
        self.monitor = None


state = AppState()


def _init_tenants(settings: DistLLMSettings | None = None) -> None:
    """Initialize the multi-tenant system."""
    state.tenant_store = TenantStore()
    state.tenant_rate_limiter = TenantRateLimiter()
    state.usage_meter = UsageMeter(state.tenant_store)
    state.tenant_router = TenantModelRouter()

    app.state.tenant_store = state.tenant_store
    app.state.usage_meter = state.usage_meter
    app.state.tenant_router = state.tenant_router

    # Always read tenant config from settings; never from os.environ (no env injection)
    tenant_enabled = bool(settings and getattr(settings, "tenant", None) and settings.tenant.enabled)

    # Read admin API key from settings only; never write secrets to os.environ
    admin_key = None
    if settings and getattr(settings, "tenant", None) and settings.tenant.admin_api_key:
        admin_key = settings.tenant.admin_api_key.get_secret_value() if hasattr(settings.tenant.admin_api_key, "get_secret_value") else str(settings.tenant.admin_api_key)
    # Ensure no leftover env var remains (defense in depth)
    os.environ.pop("ADMIN_API_KEY", None)

    if tenant_enabled:
        app.add_middleware(TenantMiddleware, store=state.tenant_store, enabled=True)
        _init_tenant_routes()
        logger.info("Multi-tenant system initialized with middleware")
    else:
        logger.info("Tenant system enabled by default")

    # Seed default tenant for backward compatibility
    if state.tenant_store.get_tenant("default") is None:
        from distllm.tenant.models import TenantTier
        state.tenant_store.create_tenant(name="Default", tier=TenantTier.FREE)


def _init_observability():
    """Initialize tracing, logging, metrics exporter, cost tracker, anomaly detector."""
    # Structured logging with OTel trace injection
    setup_logging(level="INFO", json_format=True)

    # OpenTelemetry tracing with head-based sampling (100% by default)
    setup_tracing(
        service_name="distllm-api",
        sampling_strategy="head",
        sampling_ratio=1.0,
    )

    # Prometheus metrics exporter
    state.metrics_exporter = DistLLMPrometheusExporter()

    # Cost tracker
    state.cost_tracker = GPUCostTracker()

    # Anomaly detector
    state.anomaly_detector = AnomalyDetector(sigma_threshold=3.0)
    state.anomaly_detector.register_metric("http_request_duration", window_size=60, sigma_threshold=3.0)
    state.anomaly_detector.register_metric("http_error_rate", window_size=30, sigma_threshold=2.5)

    # Wire anomaly callbacks to increment Prometheus counter
    exporter = state.metrics_exporter

    def _on_anomaly(event):
        exporter.anomaly_detected_total.labels(
            metric=event.metric, type="statistical_deviation"
        ).inc()
        logger.warning(f"Anomaly detected: {event.metric}={event.value:.2f} "
                       f"(mean={event.mean:.2f}, sigma={event.deviation_sigma:.1f})")

    state.anomaly_detector.on_anomaly(_on_anomaly)

    # Add ObservabilityMiddleware
    app.add_middleware(
        ObservabilityMiddleware,
        metrics_exporter=state.metrics_exporter,
        cost_tracker=state.cost_tracker,
        anomaly_detector=state.anomaly_detector,
    )


# Request timeout middleware
# NOTE: Registered BEFORE AuthMiddleware so that AuthMiddleware runs first
# (FastAPI executes middleware in reverse order of registration).
class TimeoutMiddleware(BaseHTTPMiddleware):
    """Cancel requests that exceed the timeout limit."""

    DEFAULT_TIMEOUT = 120.0  # 2 minutes default
    ENDPOINT_TIMEOUTS = {
        "/v1/chat/completions": 300.0,  # 5 minutes for chat
        "/v1/completions": 300.0,  # 5 minutes for completions
        "/v1/embeddings": 60.0,  # 1 minute for embeddings
    }

    async def dispatch(self, request: Request, call_next):
        timeout = self.ENDPOINT_TIMEOUTS.get(
            request.url.path, self.DEFAULT_TIMEOUT
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


# Request size limiting middleware
class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Limit maximum request body size to prevent OOM."""

    MAX_REQUEST_SIZE = 32 * 1024 * 1024  # 32 MB default

    async def dispatch(self, request: Request, call_next):
        if request.method in ("POST", "PUT", "PATCH"):
            cl = request.headers.get("content-length")
            if cl:
                try:
                    if int(cl) > self.MAX_REQUEST_SIZE:
                        return _error_response(
                            status_code=413,
                            error="Request Entity Too Large",
                            message=f"Request exceeds maximum size of {self.MAX_REQUEST_SIZE // (1024*1024)} MB",
                            type="request_too_large",
                            request=request,
                        )
                except (ValueError, TypeError):
                    pass
        return await call_next(request)


app.add_middleware(RequestSizeLimitMiddleware)


# Backpressure middleware
class BackpressureMiddleware(BaseHTTPMiddleware):
    """Reject requests when system is under heavy load."""

    MAX_PENDING_REQUESTS = 1000  # Max pending requests before rejecting

    async def dispatch(self, request: Request, call_next):
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


def _init_rate_limiter(settings: DistLLMSettings | None = None) -> None:
    """Initialize the rate limiter from settings."""
    from distllm.config.settings import RateLimitSettings
    rl_settings = settings.rate_limit if settings else RateLimitSettings()
    state.rate_limiter = RateLimiter(
        default_rpm=rl_settings.default_rpm,
        endpoint_limits=rl_settings.endpoint_limits,
        burst_multiplier=rl_settings.burst_multiplier,
        auth_rpm_multiplier=rl_settings.auth_rpm_multiplier,
    )
    app.add_middleware(
        RateLimitMiddleware,
        rate_limiter=state.rate_limiter,
        enabled=rl_settings.enabled,
    )


# Include route routers
app.include_router(chat_router)
app.include_router(completion_router)
app.include_router(embeddings_router)
app.include_router(adapters_router)
app.include_router(health_router)
app.include_router(versions_router)
app.include_router(multi_model_router)
app.include_router(batch_router)
app.include_router(audio_router)
app.include_router(images_router)
app.include_router(moderations_router)
app.include_router(files_router)
app.include_router(fine_tuning_router)
app.include_router(gossip_router)
app.include_router(optimization_router)
app.include_router(rag_router)
app.include_router(agent_router)
app.include_router(disagg_router)
app.include_router(pipeline_router)
app.include_router(debug_router, include_in_schema=False)

# --- Dashboard & WebSocket ---

from pathlib import Path


@app.websocket("/ws")
async def dashboard_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard metrics.

    Client may send JSON commands:
      - ``{"type":"subscribe","metrics":["latency","gpu"],"interval":2.0}``
      - ``{"type":"ping"}``

    Supported metric categories: latency, ttft, throughput, tokens_per_sec,
    kv_cache, speculative, cost, queue_depth, active_requests, scheduler,
    nodes, gpu, prefix_cache, spec_decoder, topology, tenants.
    """
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


@app.get(
    "/dashboard",
    response_class=HTMLResponse,
    summary="Dashboard page",
    description="Serve the real-time monitoring dashboard HTML page. The dashboard displays live metrics, request throughput, latency charts, and system health via WebSocket connection.",
    response_description="Dashboard HTML page",
    include_in_schema=False,
)
async def dashboard_page():
    """Serve the real-time dashboard HTML."""
    html_path = Path(__file__).parent.parent / "dashboard" / "static_v2" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>Dashboard not found</h1>")


@app.get(
    "/api/metrics/collector",
    summary="Metrics collector snapshot",
    description="Return a snapshot of all collected metrics from the observability collector, including raw counters and gauges for instrumentation debugging.",
    response_description="Collector metrics snapshot",
)
async def api_collector_metrics():
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
):
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


def _start_ws_broadcaster():
    """Start the WebSocket metrics broadcaster background task."""
    if state.coordinator is not None:
        state.ws_broadcast_task = asyncio.create_task(metrics_broadcaster(state.coordinator))


def _init_tenant_routes():
    """Include tenant router lazily to avoid circular import."""
    from distllm.api.routes.tenants import router as t_router
    app.include_router(t_router)


def _build_quantization_config(settings: DistLLMSettings):
    """Build quantization config from settings for the Coordinator."""
    q = settings.quantization
    if q.method == "none":
        return None

    if q.method in ("gptq", "awq", "fp8"):
        return build_quantization_config(
            q.method,
            gptq_bits=q.gptq_bits,
            gptq_group_size=q.gptq_group_size,
            gptq_desc_act=q.gptq_desc_act,
            gptq_use_marlin=q.gptq_use_marlin,
            awq_bits=q.awq_bits,
            awq_group_size=q.awq_group_size,
            fp8_scheme=q.fp8_scheme,
            fp8_dynamic=q.fp8_dynamic,
        )

    # BitsAndBytes
    try:
        from transformers import BitsAndBytesConfig
        import torch
    except ImportError:
        logger.warning("BitsAndBytes not available, skipping quantization config")
        return None

    if "4bit" in q.method:
        compute_dtype = getattr(torch, q.bnb_4bit_compute_dtype, torch.float16)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=q.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=q.bnb_4bit_use_double_quant,
        )
    elif "8bit" in q.method:
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=q.llm_int8_threshold)
    return None


def _build_speculative_config(settings: DistLLMSettings):
    """Build speculative decoding config from settings."""
    s = settings.speculative
    if not s.draft_model and s.method == "draft_model":
        return None
    return {
        "draft_model": s.draft_model,
        "num_assistant_tokens": s.num_assistant_tokens,
        "min_acceptance_rate": s.min_acceptance_rate,
        "warmup_steps": s.warmup_steps,
        "method": s.method,
        "medusa_num_heads": s.medusa_num_heads,
        "medusa_num_tokens_per_head": s.medusa_num_tokens_per_head,
        "eagle_checkpoint": s.eagle_checkpoint,
        "eagle_variant": s.eagle_variant,
        "eagle_hidden_size": s.eagle_hidden_size,
        "eagle_vocab_size": s.eagle_vocab_size,
        "eagle_num_layers": s.eagle_num_layers,
        "ngram_min_match": s.ngram_min_match,
    }


def _build_lora_config(settings: DistLLMSettings):
    """Build LoRA config from settings."""
    l = settings.lora
    if not l.enabled:
        return None
    return {
        "adapters": l.adapters,
    }


def create_coordinator(
    model_name: str,
    dtype: str = "float16",
    local: bool = False,
    max_batch_size: int = 1,
    max_tokens_per_batch: int = 4096,
    prefix_cache_enabled: bool = False,
    prefix_cache_max_entries: int = 1024,
    prefix_cache_min_prefix_len: int = 16,
    radix_tree_cache_enabled: bool = False,
    chunked_prefill_enabled: bool = True,
    chunked_prefill_chunk_size: int = 512,
    quantization_config=None,
    speculative_config=None,
    lora_config=None,
    settings: DistLLMSettings | None = None,
) -> Coordinator:
    """Create and configure the coordinator and monitor."""
    if settings:
        max_batch_size = settings.batching.max_batch_size
        max_tokens_per_batch = settings.batching.max_tokens_per_batch
        prefix_cache_enabled = settings.prefix_cache.enabled
        prefix_cache_max_entries = settings.prefix_cache.max_entries
        prefix_cache_min_prefix_len = settings.prefix_cache.min_prefix_len
        radix_tree_cache_enabled = settings.prefix_cache.radix_tree_enabled
        chunked_prefill_enabled = settings.chunked_prefill.enabled
        chunked_prefill_chunk_size = settings.chunked_prefill.chunk_size

    coord = Coordinator(
        model_name=model_name,
        dtype=dtype,
        max_batch_size=max_batch_size,
        max_tokens_per_batch=max_tokens_per_batch,
        prefix_cache_enabled=prefix_cache_enabled,
        prefix_cache_max_entries=prefix_cache_max_entries,
        prefix_cache_min_prefix_len=prefix_cache_min_prefix_len,
        radix_tree_cache_enabled=radix_tree_cache_enabled,
        chunked_prefill_enabled=chunked_prefill_enabled,
        chunked_prefill_chunk_size=chunked_prefill_chunk_size,
        quantization_config=quantization_config,
        speculative_config=speculative_config,
        lora_config=lora_config,
        embedding_config=getattr(settings, "embedding", None) if settings else None,
        version_config=getattr(settings, "version", None) if settings else None,
        metrics_exporter=state.metrics_exporter,
    )

    # Initialize multi-model chat router from settings
    if settings and settings.chat_router.enabled:
        chat_router_settings = settings.chat_router
        from distllm.core.chat_router import ModelRouter
        model_router = ModelRouter(chat_router_settings)
        # Register the hybrid model name so the chat route recognises it
        model_router.register_hybrid_name(chat_router_settings.name)
        coord._chat_router = model_router

    if local:
        coord.load_local_model()
        logger.info(f"Coordinator loaded model locally: {model_name}")
    else:
        logger.info(f"Coordinator ready for distributed mode: {model_name}")

    monitor_inst = SystemMonitor()
    state.set(coord, monitor_inst)

    return coord


def _load_settings(args) -> DistLLMSettings:
    """Load settings with precedence: CLI > env > YAML > defaults.

    Precedence order (lowest to highest):
    1. Pydantic defaults (lowest)
    2. YAML config file
    3. Environment variables (DISTLLM_*)
    4. CLI arguments (highest)
    """
    # Find YAML config path
    config_path = args.config
    if config_path is None:
        for candidate in ["config.yaml", os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yaml")]:
            if os.path.exists(candidate):
                config_path = candidate
                break

    # Step 1: Start with pydantic defaults
    base = DistLLMSettings().model_dump()

    # Step 2: Layer YAML config on top of defaults
    if config_path and os.path.exists(config_path):
        yaml_data = load_config_file(config_path)
        if yaml_data:
            base = _deep_merge(base, yaml_data)

    # Step 3: Layer environment variables on top (env > YAML)
    # Parse DISTLLM_<KEY>=value into the top-level dict,
    # and DISTLLM_<KEY>__<SUBKEY>=value into nested dicts
    for env_key, env_val in os.environ.items():
        if not env_key.startswith("DISTLLM_"):
            continue
        suffix = env_key[len("DISTLLM_"):]
        if not suffix:
            continue
        parts = suffix.split("__")
        target = base
        for part in parts[:-1]:
            key = part.lower()
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        target[parts[-1].lower()] = _parse_env_value(env_val)

    # Step 4: CLI overrides (highest precedence)
    if args.model:
        base.setdefault("model", {})["name"] = args.model
    if args.dtype:
        base.setdefault("model", {})["dtype"] = args.dtype
    if args.host:
        base.setdefault("coordinator", {})["host"] = args.host
    if args.port:
        base.setdefault("coordinator", {})["api_port"] = args.port
    if args.quantization and args.quantization != "none":
        base.setdefault("quantization", {})["method"] = args.quantization

    return DistLLMSettings.model_validate(base)


def _parse_env_value(value: str):
    """Parse an environment variable string into an appropriate Python type."""
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def main():
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

    # Settings is the source of truth (env + YAML + CLI already merged)
    create_coordinator(
        model_name=settings.model.name,
        dtype=settings.model.dtype,
        local=args.local,
        settings=settings,
        quantization_config=_build_quantization_config(settings),
        speculative_config=_build_speculative_config(settings),
        lora_config=_build_lora_config(settings),
    )

    # Initialize rate limiter from settings
    _init_rate_limiter(settings)

    # Initialize tenant system
    _init_tenants(settings)

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
    uvicorn.run(app, host=settings.coordinator.host, port=settings.coordinator.api_port, log_level="info", shutdown_timeout=30)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into a copy of base dict.

    Returns a new dict without mutating inputs.
    """
    result = {}
    result.update(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


if __name__ == "__main__":
    main()
