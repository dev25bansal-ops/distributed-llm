"""OpenAI-compatible REST API for distributed LLM inference."""

import os
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware

from distllm.core.coordinator import Coordinator
from distllm.api.middleware import AuthMiddleware, RequestIDMiddleware
from distllm.api.rate_limiter import RateLimiter
from distllm.api.rate_limit_middleware import RateLimitMiddleware
from distllm.api.observability_middleware import ObservabilityMiddleware
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

# Re-export route routers
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
    _sample_token,
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


app = FastAPI(
    title="Distributed LLM API",
    description="OpenAI-compatible REST API for distributed LLM inference across multiple machines using pipeline parallelism",
    version="0.4.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "chat", "description": "Chat completion endpoints with streaming support"},
        {"name": "completion", "description": "Text completion endpoints with streaming support"},
        {"name": "models", "description": "Model listing and management"},
        {"name": "adapters", "description": "LoRA adapter management for multi-adapter inference"},
        {"name": "system", "description": "Health checks, metrics, and system status"},
    ],
)

# Security: Configure CORS with explicit allowed origins
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # Security: Disable credentials for CORS
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
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
        from distllm.constants import HSTS_MAX_AGE
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


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Convert HTTPException to structured error response."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP {exc.status_code}",
            message=exc.detail,
            type="http_error",
            request_id=getattr(request.state, "request_id", None),
        ).model_dump(exclude_none=True),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions with structured response."""
    import logging
    logger = logging.getLogger("distllm.api")
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred. Please try again later.",
            type="internal_error",
            request_id=getattr(request.state, "request_id", None),
        ).model_dump(exclude_none=True),
    )


# Application state — encapsulates global mutable state for lifecycle management
class AppState:
    """Manages shared application state (coordinator, monitor, startup time).

    Provides set/clear methods for lifecycle management and testing.
    Module-level globals (coordinator, monitor, _startup_time) are still
    accessible for backwards compatibility with endpoint code.
    """

    def __init__(self):
        self.coordinator: Coordinator | None = None
        self.monitor: SystemMonitor | None = None
        self.startup_time: float = time.time()

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.startup_time

    def set(self, coord: Coordinator, mon: SystemMonitor) -> None:
        """Set the coordinator and monitor instances."""
        self.coordinator = coord
        self.monitor = mon
        global coordinator, monitor, _startup_time
        coordinator = self.coordinator
        monitor = self.monitor
        _startup_time = self.startup_time

    def clear(self) -> None:
        """Reset state (useful for testing)."""
        self.coordinator = None
        self.monitor = None
        global coordinator, monitor
        coordinator = None
        monitor = None


state = AppState()

# Module-level globals for backwards compatibility with endpoint code
coordinator: Coordinator | None = None
monitor: SystemMonitor | None = None
_startup_time: float = time.time()

app.add_middleware(RequestIDMiddleware)

# Observability globals
metrics_exporter: DistLLMPrometheusExporter | None = None
cost_tracker: GPUCostTracker | None = None
anomaly_detector: AnomalyDetector | None = None

# Observability middleware (after RequestIDMiddleware, before AuthMiddleware)
def _init_observability():
    """Initialize tracing, logging, metrics exporter, cost tracker, anomaly detector."""
    global metrics_exporter, cost_tracker, anomaly_detector

    # Structured logging with OTel trace injection
    setup_logging(level="INFO", json_format=True)

    # OpenTelemetry tracing with head-based sampling (100% by default)
    setup_tracing(
        service_name="distllm-api",
        sampling_strategy="head",
        sampling_ratio=1.0,
    )

    # Prometheus metrics exporter
    metrics_exporter = DistLLMPrometheusExporter()

    # Cost tracker
    cost_tracker = GPUCostTracker()

    # Anomaly detector
    anomaly_detector = AnomalyDetector(sigma_threshold=3.0)
    anomaly_detector.register_metric("http_request_duration", window_size=60, sigma_threshold=3.0)
    anomaly_detector.register_metric("http_error_rate", window_size=30, sigma_threshold=2.5)

    # Wire anomaly callbacks to increment Prometheus counter
    def _on_anomaly(event):
        metrics_exporter.anomaly_detected_total.labels(
            metric=event.metric, type="statistical_deviation"
        ).inc()
        logger.warning(f"Anomaly detected: {event.metric}={event.value:.2f} "
                       f"(mean={event.mean:.2f}, sigma={event.deviation_sigma:.1f})")

    anomaly_detector.on_anomaly(_on_anomaly)

    # Add ObservabilityMiddleware
    app.add_middleware(
        ObservabilityMiddleware,
        metrics_exporter=metrics_exporter,
        cost_tracker=cost_tracker,
        anomaly_detector=anomaly_detector,
    )


_init_observability()

app.add_middleware(AuthMiddleware)


# Request timeout middleware
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

        try:
            import asyncio
            async with asyncio.timeout(timeout):
                response = await call_next(request)
                return response
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content=ErrorResponse(
                    error="Gateway Timeout",
                    message=f"Request exceeded {timeout:.0f}s timeout limit",
                    type="timeout_error",
                    request_id=getattr(request.state, "request_id", None),
                ).model_dump(exclude_none=True),
            )


app.add_middleware(TimeoutMiddleware)


# Backpressure middleware
class BackpressureMiddleware(BaseHTTPMiddleware):
    """Reject requests when system is under heavy load."""

    MAX_PENDING_REQUESTS = 1000  # Max pending requests before rejecting

    async def dispatch(self, request: Request, call_next):
        # Skip backpressure for health/metrics endpoints
        if request.url.path in ("/health", "/ready", "/live", "/metrics", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        # Check if scheduler is overloaded
        if coordinator and coordinator.scheduler:
            try:
                stats = coordinator.scheduler.stats()
                pending = stats.get("pending_requests", 0)
                if isinstance(pending, (int, float)) and pending >= self.MAX_PENDING_REQUESTS:
                    return JSONResponse(
                        status_code=503,
                        content=ErrorResponse(
                            error="Service Unavailable",
                            message=f"System overloaded: {pending} pending requests",
                            type="backpressure_error",
                            request_id=getattr(request.state, "request_id", None),
                        ).model_dump(exclude_none=True),
                        headers={"Retry-After": "5"},
                    )
            except (AttributeError, TypeError, KeyError):
                pass  # Scheduler stats unavailable or malformed, skip check

        # Check if shutting down
        if coordinator and getattr(coordinator, "_shutting_down", False):
            return JSONResponse(
                status_code=503,
                content=ErrorResponse(
                    error="Service Unavailable",
                    message="Service is shutting down",
                    type="shutdown_error",
                    request_id=getattr(request.state, "request_id", None),
                ).model_dump(exclude_none=True),
                headers={"Retry-After": "10"},
            )

        return await call_next(request)


app.add_middleware(BackpressureMiddleware)


# Rate limiter (initialized with defaults, configured on startup)
_rate_limiter: RateLimiter | None = None


def _init_rate_limiter(settings: DistLLMSettings | None = None) -> None:
    """Initialize the rate limiter from settings."""
    global _rate_limiter
    from distllm.config.settings import RateLimitSettings
    rl_settings = settings.rate_limit if settings else RateLimitSettings()
    _rate_limiter = RateLimiter(
        default_rpm=rl_settings.default_rpm,
        endpoint_limits=rl_settings.endpoint_limits,
        burst_multiplier=rl_settings.burst_multiplier,
    )
    app.add_middleware(
        RateLimitMiddleware,
        rate_limiter=_rate_limiter,
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


def _build_quantization_config(settings: DistLLMSettings):
    """Build quantization config from settings for the Coordinator."""
    q = settings.quantization
    if q.method == "none":
        return None

    if q.method in ("gptq", "awq", "fp8"):
        # These use dict configs passed to model loader
        from distllm.core.quantization_selector import build_quantization_config
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
        "method": s.method,
        "medusa_num_heads": s.medusa_num_heads,
        "medusa_num_tokens_per_head": s.medusa_num_tokens_per_head,
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
        metrics_exporter=metrics_exporter,
    )

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

    Always starts with DistLLMSettings() which reads env vars automatically
    via pydantic-settings, then layers YAML config, then CLI overrides.
    Merges all layers together before a single validation pass.
    """
    settings = DistLLMSettings()
    overrides: dict = {}

    # Find YAML config path
    config_path = args.config
    if config_path is None:
        for candidate in ["config.yaml", os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yaml")]:
            if os.path.exists(candidate):
                config_path = candidate
                break

    if config_path and os.path.exists(config_path):
        yaml_data = load_config_file(config_path)
        if yaml_data:
            overrides = _deep_merge({}, yaml_data)

    # CLI overrides (highest precedence) — layer on top of YAML
    if args.model:
        overrides.setdefault("model", {})["name"] = args.model
    if args.dtype:
        overrides.setdefault("model", {})["dtype"] = args.dtype
    if args.host:
        overrides.setdefault("coordinator", {})["host"] = args.host
    if args.port:
        overrides.setdefault("coordinator", {})["api_port"] = args.port
    if args.quantization and args.quantization != "none":
        overrides.setdefault("quantization", {})["method"] = args.quantization

    if overrides:
        merged = _deep_merge(settings.model_dump(), overrides)
        settings = DistLLMSettings.model_validate(merged)

    return settings


def main():
    import argparse

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
        import logging
        logging.getLogger("distllm.security").warning(
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

    logger.info(f"Starting API server on {settings.coordinator.host}:{settings.coordinator.api_port}")
    uvicorn.run(app, host=settings.coordinator.host, port=settings.coordinator.api_port, log_level="info")


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
