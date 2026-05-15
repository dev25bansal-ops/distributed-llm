"""OpenAI-compatible REST API for distributed LLM inference."""

import asyncio
import os
import time
import uuid
import json
from typing import List, Optional, Dict, Any

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, ConfigDict
import uvicorn
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from distllm.core.coordinator import Coordinator
from distllm.api.middleware import AuthMiddleware, RequestIDMiddleware
from distllm.api.rate_limiter import RateLimiter
from distllm.api.rate_limit_middleware import RateLimitMiddleware
from distllm.core.monitor import SystemMonitor
from distllm.config.settings import DistLLMSettings
from distllm.config.loader import load_config_file, dict_to_config
from distllm.communication.grpc import set_debug_mode

# CORS configuration
ALLOWED_ORIGINS = os.environ.get(
    "DISTLLM_CORS_ORIGINS",
    "http://localhost:3000,http://localhost:8080"
).split(",")


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

        # Content Security Policy
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        # MIME type sniffing protection
        response.headers["X-Content-Type-Options"] = "nosniff"
        # XSS protection (legacy browsers)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        # Strict Transport Security (for HTTPS deployments)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Permissions policy
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

        return response


app.add_middleware(SecurityHeadersMiddleware)


# Structured error responses
class ErrorResponse(BaseModel):
    """Standardized error response format."""
    error: str
    message: str
    type: str = "api_error"
    code: Optional[str] = None
    request_id: Optional[str] = None


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

# Global coordinator and monitor instances
coordinator: Optional[Coordinator] = None
monitor: Optional[SystemMonitor] = None
_startup_time = time.time()  # For liveness probe uptime calculation

app.add_middleware(RequestIDMiddleware)
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
_rate_limiter: Optional[RateLimiter] = None


def _init_rate_limiter(settings: Optional[DistLLMSettings] = None) -> None:
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


# --- Request/Response Models ---

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender", examples=["user", "assistant", "system"])
    content: Optional[str] = Field(default=None, description="Content of the message", examples=["Hello, how are you?"])
    name: Optional[str] = Field(default=None, description="Name for function messages or multi-agent scenarios")
    tool_calls: Optional[List[dict]] = Field(default=None, description="Tool calls generated by the assistant")
    tool_call_id: Optional[str] = Field(default=None, description="Tool call ID for tool response messages")
    function_call: Optional[dict] = Field(default=None, description="Deprecated: function call generated by the assistant")


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Explain distributed inference"}],
                "max_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "stream": False,
            }]
        }
    )

    model: str = Field(default="distributed-llm", description="Model identifier")
    messages: List[ChatMessage] = Field(..., description="List of messages in the conversation")
    temperature: float = Field(default=0.7, ge=0, le=2.0, description="Sampling temperature (0-2.0)")
    top_p: float = Field(default=0.9, ge=0, le=1.0, description="Nucleus sampling threshold (0-1)")
    top_k: int = Field(default=0, ge=0, description="Top-k sampling (0 = disabled)")
    max_tokens: int = Field(default=256, ge=1, le=8192, description="Maximum tokens to generate (1-8192)")
    stream: bool = Field(default=False, description="Whether to stream the response")
    stream_options: Optional[dict] = Field(default=None, description="Options for streaming response, e.g. {'include_usage': true}")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences to halt generation")
    n: int = Field(default=1, ge=1, le=128, description="Number of completions to generate")
    logprobs: Optional[bool] = Field(default=None, description="Whether to return log probabilities")
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=20, description="Number of top logprobs to return")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Penalty for new tokens based on presence in text")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Penalty for tokens based on frequency in text")
    seed: Optional[int] = Field(default=None, description="Random seed for deterministic sampling")
    user: Optional[str] = Field(default=None, description="End-user identifier for monitoring and abuse detection")
    logit_bias: Optional[Dict[str, float]] = Field(default=None, description="Modify likelihood of specified tokens")
    response_format: Optional[dict] = Field(default=None, description="Response format constraint, e.g. {'type': 'json_object'}")
    adapter: Optional[str] = Field(default=None, description="LoRA adapter ID to use for this request")
    priority: int = Field(default=2, ge=0, le=3, description="Request priority: 0=critical, 1=high, 2=normal, 3=low")
    tools: Optional[List[dict]] = Field(default=None, description="List of tools the model may call")
    tool_choice: Optional[str] = Field(default=None, description="Controls tool calling: 'none', 'auto', or 'required'")
    functions: Optional[List[dict]] = Field(default=None, description="Deprecated: list of functions for the model to call")
    function_call: Optional[str] = Field(default=None, description="Deprecated: controls function calling behavior")


class ChatChoice(BaseModel):
    index: int = 0
    message: Optional[ChatMessage] = None
    delta: Optional[Dict[str, str]] = None
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "distributed-llm"
    choices: List[ChatChoice]
    usage: Optional[Dict[str, int]] = None
    generation_time: Optional[float] = None


class CompletionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "model": "distributed-llm",
                "prompt": "Once upon a time",
                "max_tokens": 128,
                "temperature": 0.8,
                "stream": False,
            }]
        }
    )
    model: str = Field(default="distributed-llm", description="Model identifier")
    prompt: str = Field(..., description="The prompt text to generate from")
    max_tokens: int = Field(default=256, ge=1, le=8192, description="Maximum tokens to generate (1-8192)")
    temperature: float = Field(default=0.7, ge=0, le=2.0, description="Sampling temperature (0-2.0)")
    top_p: float = Field(default=0.9, ge=0, le=1.0, description="Nucleus sampling threshold (0-1)")
    top_k: int = Field(default=0, ge=0, description="Top-k sampling (0 = disabled)")
    stream: bool = Field(default=False, description="Whether to stream the response")
    priority: int = Field(default=2, ge=0, le=3, description="Request priority: 0=critical, 1=high, 2=normal, 3=low")


class CompletionChoice(BaseModel):
    index: int = 0
    text: str = ""
    delta: Optional[str] = None
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:12]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "distributed-llm"
    choices: List[CompletionChoice]
    generation_time: Optional[float] = None


class ModelInfo(BaseModel):
    id: str = Field(..., description="Model identifier")
    object: str = "model"
    created: int = Field(..., description="Unix timestamp of model creation")
    owned_by: str = "distributed-llm"
    root: Optional[str] = Field(default=None, description="Root model for fine-tuned models")
    archived: bool = Field(default=False, description="Whether the model is archived")


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "model": "distributed-llm",
                "input": ["Hello world", "Test sentence"],
                "encoding_format": "float",
            }]
        }
    )
    model: str = Field(default="distributed-llm", description="Model identifier")
    input: List[str] = Field(..., description="Input text(s) to embed")
    encoding_format: str = Field(default="float", description="Output format: 'float' or 'base64'")
    dimensions: Optional[int] = Field(default=None, ge=1, description="Number of dimensions for the embedding")
    user: Optional[str] = Field(default=None, description="End-user identifier")


class EmbeddingObject(BaseModel):
    index: int = Field(..., description="Index of the embedding in the input list")
    object: str = "embedding"
    embedding: List[float] = Field(..., description="The embedding vector")


class EmbeddingResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"embed-{uuid.uuid4().hex[:12]}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "distributed-llm"
    data: List[EmbeddingObject]
    usage: Dict[str, int] = Field(default_factory=dict)


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class ParamUpdateRequest(BaseModel):
    """Request to update generation parameters mid-stream."""
    temperature: Optional[float] = Field(default=None, ge=0, le=2.0, description="New sampling temperature")
    top_p: Optional[float] = Field(default=None, ge=0, le=1.0, description="New nucleus sampling threshold")
    top_k: Optional[int] = Field(default=None, ge=0, description="New top-k sampling value")


class AdapterLoadRequest(BaseModel):
    action: str  # "load", "set", "list"
    id: Optional[str] = None
    path: Optional[str] = None


# --- API Endpoints ---

@app.get("/v1/models", tags=["models"])
async def list_models():
    """List available models."""
    if coordinator is None:
        return ModelList(data=[])
    if hasattr(coordinator, 'list_models'):
        model_names = coordinator.list_models()
    else:
        model_names = [coordinator.model_name]
    return ModelList(
        data=[
            ModelInfo(id=name, created=int(time.time()))
            for name in model_names
        ]
    )


@app.post("/v1/embeddings", tags=["embedding"])
async def create_embeddings(request: EmbeddingRequest):
    """Create embeddings for input text.

    Generates dense vector embeddings for each input string.
    If no embedding model is loaded, falls back to pooling the last hidden state.
    """
    if coordinator is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    if not coordinator.tokenizer:
        raise HTTPException(status_code=503, detail="Tokenizer not available")

    start_time = time.time()
    embeddings = []
    total_tokens = 0

    for idx, text in enumerate(request.input):
        input_ids = coordinator.tokenizer.encode(text, return_tensors="pt")
        if hasattr(coordinator, "local_partitioner") and coordinator.local_partitioner:
            model = coordinator.local_partitioner.full_model
            device = next(model.parameters()).device
            input_ids = input_ids.to(device)

            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
                # Use mean pooling over last hidden state
                last_hidden = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else outputs.last_hidden_state
                attention_mask = torch.ones_like(input_ids)
                masked = last_hidden * attention_mask.unsqueeze(-1)
                embedding = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                embeddings.append(embedding[0].tolist())
        else:
            raise HTTPException(
                status_code=503,
                detail="Embedding generation requires a loaded model. Use --local flag or connect to worker nodes.",
            )

        total_tokens += input_ids.shape[-1]

    elapsed = time.time() - start_time

    return EmbeddingResponse(
        model=request.model,
        data=[
            EmbeddingObject(index=i, embedding=emb)
            for i, emb in enumerate(embeddings)
        ],
        usage={
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    )


@app.get("/v1/adapters", tags=["adapters"])
async def list_adapters():
    """List loaded LoRA adapters."""
    if coordinator is None or not hasattr(coordinator, 'adapter_manager') or coordinator.adapter_manager is None:
        raise HTTPException(status_code=503, detail="LoRA not enabled")

    adapters = coordinator.adapter_manager.list_adapters()
    return {
        "active": coordinator.adapter_manager.active_adapter,
        "adapters": adapters,
    }


@app.post("/v1/adapters", tags=["adapters"])
async def manage_adapters(request: AdapterLoadRequest):
    """Load, set, or list LoRA adapters."""
    if coordinator is None or not hasattr(coordinator, 'adapter_manager') or coordinator.adapter_manager is None:
        raise HTTPException(status_code=503, detail="LoRA not enabled")

    if request.action == "load":
        if not request.id or not request.path:
            raise HTTPException(status_code=400, detail="id and path required for load action")

        # Security: Prevent path traversal in adapter paths
        import os
        adapter_path = os.path.normpath(request.path)
        # Block absolute paths and parent directory traversal
        if os.path.isabs(adapter_path) and not adapter_path.startswith("/app/adapters"):
            raise HTTPException(
                status_code=403,
                detail="Adapter path must be relative or within /app/adapters directory"
            )
        if ".." in adapter_path.split(os.sep):
            raise HTTPException(
                status_code=403,
                detail="Adapter path cannot contain parent directory traversal"
            )

        coordinator.adapter_manager.load_adapter(request.id, request.path)
        return {"status": "loaded", "id": request.id}
    elif request.action == "set":
        if not request.id:
            raise HTTPException(status_code=400, detail="id required for set action")
        coordinator.adapter_manager.set_active(request.id)
        return {"status": "active", "id": request.id}
    elif request.action == "list":
        return {
            "active": coordinator.adapter_manager.active_adapter,
            "adapters": coordinator.adapter_manager.list_adapters(),
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")


@app.post("/v1/chat/completions", tags=["chat"])
async def chat_completions(request: ChatCompletionRequest):
    """Chat completions endpoint."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    # Validate requested model against registry
    if hasattr(coordinator, 'list_models') and request.model not in ("distributed-llm", ""):
        available = coordinator.list_models()
        if request.model not in available:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' not found. Available: {available}"
            )

    # Switch adapter if requested
    if request.adapter is not None and hasattr(coordinator, 'adapter_manager') and coordinator.adapter_manager is not None:
        coordinator.adapter_manager.set_active(request.adapter)

    prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])

    # Build schema constraint for structured output
    schema = None
    if request.response_format:
        fmt_type = request.response_format.get("type", "")
        if fmt_type == "json_object":
            schema = {}  # Simple JSON constraint
        elif fmt_type == "json_schema" and "schema" in request.response_format:
            schema = request.response_format["schema"]

    if request.stream:
        return StreamingResponse(
            _stream_chat(prompt, request),
            media_type="text/event-stream",
        )

    start_time = time.time()

    # Use batch scheduler if available
    if coordinator.scheduler is not None:
        request_id = coordinator.generate_async(
            prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            schema=schema,
            priority=request.priority,
        )
        result = await asyncio.to_thread(coordinator.wait_for_result, request_id)
    else:
        result = await asyncio.to_thread(
            coordinator.generate,
            prompt,
            request.max_tokens,
            request.temperature,
            request.top_p,
        )

    elapsed = time.time() - start_time

    generated = result[len(prompt):] if result.startswith(prompt) else result

    # Compute token counts without re-encoding (prompt already tokenized in generate())
    prompt_tokens = len(coordinator.tokenizer.encode(prompt))
    completion_tokens = len(coordinator.tokenizer.encode(generated))

    return ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatChoice(
                message=ChatMessage(role="assistant", content=generated.strip()),
                finish_reason="stop",
            )
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        generation_time=round(elapsed, 3),
    )


@app.post("/v1/completions", tags=["completion"])
async def completions(request: CompletionRequest):
    """Text completions endpoint."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    if request.stream:
        return StreamingResponse(
            _stream_completion(request.prompt, request),
            media_type="text/event-stream",
        )

    start_time = time.time()
    result = await asyncio.to_thread(
        coordinator.generate,
        request.prompt,
        request.max_tokens,
        request.temperature,
        request.top_p,
    )
    elapsed = time.time() - start_time

    generated = result[len(request.prompt):] if result.startswith(request.prompt) else result

    return CompletionResponse(
        model=request.model,
        choices=[
            CompletionChoice(
                text=generated,
                finish_reason="stop",
            )
        ],
        generation_time=round(elapsed, 3),
    )


@app.get("/health", tags=["system"])
async def health_check():
    """Health check endpoint."""
    if coordinator is None:
        return {"status": "unhealthy", "reason": "No model loaded"}

    node_health = coordinator.health_check() if coordinator.nodes else {}
    health = {
        "status": "healthy",
        "model": coordinator.model_name,
        "nodes": len(coordinator.nodes),
        "node_health": node_health,
    }

    if monitor and coordinator.scheduler:
        health.update(monitor.health_check(coordinator.scheduler))

    return health


@app.get("/ready", tags=["system"])
async def readiness_check():
    """Kubernetes readiness probe.

    Returns 200 only when the service can accept traffic.
    """
    if coordinator is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "No model loaded"},
        )

    if getattr(coordinator, "_shutting_down", False):
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "Service is shutting down"},
        )

    # Check if at least one node is healthy (for distributed mode)
    if coordinator.nodes:
        node_health = coordinator.health_check()
        healthy_nodes = sum(1 for h in node_health.values() if h.get("healthy"))
        if healthy_nodes == 0:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "reason": "No healthy nodes available",
                    "healthy_nodes": 0,
                    "total_nodes": len(coordinator.nodes),
                },
            )

    return {"status": "ready"}


@app.get("/live", tags=["system"])
async def liveness_check():
    """Kubernetes liveness probe.

    Returns 200 if the process is alive and not deadlocked.
    """
    # Simple check: if we can reach this endpoint, we're alive
    return {"status": "alive", "uptime_seconds": time.time() - _startup_time}


@app.get("/metrics", tags=["system"])
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    lines = []

    if coordinator is None:
        # Return service status even when not initialized
        lines.append("# TYPE distllm_service_up gauge")
        lines.append("distllm_service_up 0")
        lines.append("# TYPE distllm_coordinator_loaded gauge")
        lines.append("distllm_coordinator_loaded 0")
        return "\n".join(lines)

    # Use Prometheus exporter if available
    if coordinator.metrics_exporter:
        return coordinator.metrics_exporter.generate_metrics()

    # Fallback: dict-based text format
    m = coordinator.get_metrics()

    # Add service status
    lines.append("# TYPE distllm_service_up gauge")
    lines.append("distllm_service_up 1")
    lines.append("# TYPE distllm_coordinator_loaded gauge")
    lines.append("distllm_coordinator_loaded 1")

    for name, value in m.items():
        if isinstance(value, (int, float)):
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        else:
            lines.append(f"# {name} {value}")

    # Add scheduler metrics
    if coordinator.scheduler:
        stats = coordinator.scheduler.stats()
        lines.append("# TYPE distllm_active_requests gauge")
        lines.append(f"distllm_active_requests {stats['active_requests']}")
        lines.append("# TYPE distllm_pending_requests gauge")
        lines.append(f"distllm_pending_requests {stats['pending_requests']}")

    # Add prefix cache metrics
    if coordinator.prefix_cache:
        pc_stats = coordinator.prefix_cache.stats()
        for name, value in pc_stats.items():
            if isinstance(value, (int, float)):
                lines.append(f"# TYPE distllm_{name} gauge")
                lines.append(f"distllm_{name} {value}")

    # Add system metrics
    if monitor:
        sys_metrics = monitor.collect()
        if "cpu" in sys_metrics:
            lines.append("# TYPE distllm_cpu_percent gauge")
            lines.append(f"distllm_cpu_percent {sys_metrics['cpu']['percent']}")
        if "gpu" in sys_metrics:
            gpu = sys_metrics["gpu"]
            if "memory_percent" in gpu:
                lines.append("# TYPE distllm_gpu_memory_percent gauge")
                lines.append(f"distllm_gpu_memory_percent {gpu['memory_percent']}")
            if "temperature_c" in gpu:
                lines.append("# TYPE distllm_gpu_temperature_c gauge")
                lines.append(f"distllm_gpu_temperature_c {gpu['temperature_c']}")

    # Add readiness status
    lines.append("# TYPE distllm_ready gauge")
    lines.append(f"distllm_ready {1 if not getattr(coordinator, '_shutting_down', False) else 0}")

    return "\n".join(lines)


@app.post("/v1/update-params/{request_id}", tags=["generation"])
async def update_generation_params(request_id: str, params: ParamUpdateRequest):
    """Update generation parameters for an in-progress request.

    Allows changing temperature, top_p, and top_k mid-generation for streaming requests.
    """
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")

    updated = coordinator._param_update_channel.update(
        request_id,
        temperature=params.temperature if params.temperature is not None else None,
        top_p=params.top_p if params.top_p is not None else None,
        top_k=params.top_k if params.top_k is not None else None,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found or already completed")

    return {
        "request_id": request_id,
        "temperature": updated.temperature,
        "top_p": updated.top_p,
        "top_k": updated.top_k,
    }


# --- Sampling Helper ---

def _sample_token(logits: torch.Tensor, temperature: float, top_p: float, top_k: int = 0) -> torch.Tensor:
    """Sample next token from logits with temperature, top-k, and top-p filtering.

    Args:
        logits: Logits tensor of shape [1, vocab_size]
        temperature: Sampling temperature (0 = greedy)
        top_p: Nucleus sampling threshold (1.0 = disabled)
        top_k: Top-k sampling (0 = disabled)

    Returns:
        Next token tensor of shape [1, 1]
    """
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        # Apply top-k before top-p
        if top_k > 0:
            top_k_indices = torch.topk(probs[0], top_k, dim=-1).indices
            vocab_size = probs.shape[-1]
            mask = torch.zeros(vocab_size, dtype=torch.bool, device=probs.device)
            mask[top_k_indices] = True
            probs = probs.masked_fill(~mask, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs = probs.masked_fill(indices_to_remove, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        return torch.multinomial(probs, 1)
    else:
        return torch.argmax(logits, dim=-1, keepdim=True)


# --- Streaming Helpers ---

async def _stream_chat(prompt: str, request: ChatCompletionRequest):
    """Stream chat completion tokens with KV cache for efficient incremental decoding."""
    import asyncio

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if coordinator:
        coordinator._param_update_channel.register(request_id)

    yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}}]})}\n\n"

    if not coordinator:
        yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'finish_reason': 'stop', 'delta': {}}]})}\n\n"
        yield "data: [DONE]\n\n"
        return

    if coordinator.local_partitioner:
        model = coordinator.local_partitioner.full_model
        tokenizer = coordinator.tokenizer
        device = next(model.parameters()).device

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        past_key_values = None

        with torch.no_grad():
            for step in range(request.max_tokens):
                # Check for dynamic param updates
                temperature = request.temperature
                top_p = request.top_p
                top_k = request.top_k
                if coordinator:
                    from distllm.core.param_update_channel import GenerationParams
                    params = coordinator._param_update_channel.get(request_id)
                    if params is not None and isinstance(params, GenerationParams):
                        temperature = params.temperature
                        top_p = params.top_p
                        top_k = params.top_k

                if step == 0:
                    outputs = await asyncio.to_thread(model, input_ids, use_cache=True)
                else:
                    outputs = await asyncio.to_thread(model, next_token, past_key_values=past_key_values, use_cache=True)

                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                next_token = _sample_token(logits, temperature, top_p, top_k)

                token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
                yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {'content': token_text}}]})}\n\n"

                if next_token.item() == tokenizer.eos_token_id:
                    break

    elif coordinator.node_order:
        # Distributed streaming mode
        tokenizer = coordinator.tokenizer
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        generated_ids = input_ids.clone()

        node_kv_caches: Dict[str, Optional[List]] = {
            nid: None for nid in coordinator.node_order
        }

        for step in range(request.max_tokens):
            temperature = request.temperature
            top_p = request.top_p
            top_k = request.top_k
            if coordinator:
                from distllm.core.param_update_channel import GenerationParams
                params = coordinator._param_update_channel.get(request_id)
                if params is not None and isinstance(params, GenerationParams):
                    temperature = params.temperature
                    top_p = params.top_p
                    top_k = params.top_k

            step_input = generated_ids if step == 0 else generated_ids[:, -1:]

            logits = await asyncio.to_thread(
                coordinator._pipeline.run_pipeline,
                step_input, node_kv_caches, request_id,
            )

            next_token = _sample_token(logits[:, -1, :], temperature, top_p, top_k)
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)

            token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {'content': token_text}}]})}\n\n"

            if next_token.item() == tokenizer.eos_token_id:
                break
    else:
        yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'finish_reason': 'stop', 'delta': {'content': 'No model loaded for streaming'}}]})}\n\n"

    yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'finish_reason': 'stop', 'delta': {}}]})}\n\n"
    yield "data: [DONE]\n\n"

    if coordinator:
        coordinator._param_update_channel.unregister(request_id)


async def _stream_completion(prompt: str, request: CompletionRequest):
    """Stream completion tokens with KV cache for efficient incremental decoding."""
    import asyncio

    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"

    if coordinator:
        coordinator._param_update_channel.register(request_id)

    yield f"data: {json.dumps({'id': request_id, 'object': 'text_completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'text': ''}]})}\n\n"

    if not coordinator:
        yield f"data: {json.dumps({'id': request_id, 'object': 'text_completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'finish_reason': 'stop', 'text': ''}]})}\n\n"
        yield "data: [DONE]\n\n"
        if coordinator:
            coordinator._param_update_channel.unregister(request_id)
        return

    if coordinator.local_partitioner:
        model = coordinator.local_partitioner.full_model
        tokenizer = coordinator.tokenizer
        device = next(model.parameters()).device

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        past_key_values = None

        with torch.no_grad():
            for step in range(request.max_tokens):
                # Check for dynamic param updates
                temperature = request.temperature
                top_p = request.top_p
                top_k = request.top_k
                if coordinator:
                    from distllm.core.param_update_channel import GenerationParams
                    params = coordinator._param_update_channel.get(request_id)
                    if params is not None and isinstance(params, GenerationParams):
                        temperature = params.temperature
                        top_p = params.top_p
                        top_k = params.top_k

                if step == 0:
                    outputs = await asyncio.to_thread(model, input_ids, use_cache=True)
                else:
                    outputs = await asyncio.to_thread(model, next_token, past_key_values=past_key_values, use_cache=True)

                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                next_token = _sample_token(logits, temperature, top_p, top_k)

                token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
                yield f"data: {json.dumps({'id': request_id, 'object': 'text_completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'text': token_text}]})}\n\n"

                if next_token.item() == tokenizer.eos_token_id:
                    break

    elif coordinator.node_order:
        # Distributed streaming mode
        tokenizer = coordinator.tokenizer
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        generated_ids = input_ids.clone()

        node_kv_caches: Dict[str, Optional[List]] = {
            nid: None for nid in coordinator.node_order
        }

        for step in range(request.max_tokens):
            temperature = request.temperature
            top_p = request.top_p
            top_k = request.top_k
            if coordinator:
                from distllm.core.param_update_channel import GenerationParams
                params = coordinator._param_update_channel.get(request_id)
                if params is not None and isinstance(params, GenerationParams):
                    temperature = params.temperature
                    top_p = params.top_p
                    top_k = params.top_k

            step_input = generated_ids if step == 0 else generated_ids[:, -1:]

            logits = await asyncio.to_thread(
                coordinator._pipeline.run_pipeline,
                step_input, node_kv_caches, request_id,
            )

            next_token = _sample_token(logits[:, -1, :], temperature, top_p, top_k)
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)

            token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            yield f"data: {json.dumps({'id': request_id, 'object': 'text_completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'text': token_text}]})}\n\n"

            if next_token.item() == tokenizer.eos_token_id:
                break
    else:
        yield f"data: {json.dumps({'id': request_id, 'object': 'text_completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'finish_reason': 'stop', 'text': 'No model loaded for streaming'}]})}\n\n"

    yield f"data: {json.dumps({'id': request_id, 'object': 'text_completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'finish_reason': 'stop', 'text': ''}]})}\n\n"
    yield "data: [DONE]\n\n"

    if coordinator:
        coordinator._param_update_channel.unregister(request_id)


def create_coordinator(
    model_name: str,
    dtype: str = "float16",
    local: bool = False,
    max_batch_size: int = 1,
    max_tokens_per_batch: int = 4096,
    prefix_cache_enabled: bool = False,
    prefix_cache_max_entries: int = 1024,
    prefix_cache_min_prefix_len: int = 16,
    chunked_prefill_enabled: bool = True,
    chunked_prefill_chunk_size: int = 512,
    quantization_config=None,
    speculative_config=None,
    lora_config=None,
    settings: Optional[DistLLMSettings] = None,
) -> Coordinator:
    """Create and configure the coordinator and monitor."""
    global coordinator, monitor

    # Use settings object if provided, otherwise use defaults
    if settings:
        max_batch_size = settings.batching.max_batch_size
        max_tokens_per_batch = settings.batching.max_tokens_per_batch
        prefix_cache_enabled = settings.prefix_cache.enabled
        prefix_cache_max_entries = settings.prefix_cache.max_entries
        prefix_cache_min_prefix_len = settings.prefix_cache.min_prefix_len
        chunked_prefill_enabled = settings.chunked_prefill.enabled
        chunked_prefill_chunk_size = settings.chunked_prefill.chunk_size

    coordinator = Coordinator(
        model_name=model_name,
        dtype=dtype,
        max_batch_size=max_batch_size,
        max_tokens_per_batch=max_tokens_per_batch,
        prefix_cache_enabled=prefix_cache_enabled,
        prefix_cache_max_entries=prefix_cache_max_entries,
        prefix_cache_min_prefix_len=prefix_cache_min_prefix_len,
        chunked_prefill_enabled=chunked_prefill_enabled,
        chunked_prefill_chunk_size=chunked_prefill_chunk_size,
        quantization_config=quantization_config,
        speculative_config=speculative_config,
        lora_config=lora_config,
    )

    if local:
        coordinator.load_local_model()
        logger.info(f"Coordinator loaded model locally: {model_name}")
    else:
        logger.info(f"Coordinator ready for distributed mode: {model_name}")

    # Initialize system monitor
    monitor = SystemMonitor()

    # Initialize rate limiter from settings
    _init_rate_limiter(settings)

    return coordinator


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Distributed LLM REST API")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--local", action="store_true", help="Load model locally")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with tensor shape logging")
    parser.add_argument("--validate-config", action="store_true", help="Validate configuration at startup and exit")

    args = parser.parse_args()

    # Optional: validate config and exit
    if args.validate_config:
        DistLLMSettings.validate_startup()
        print("✅ Config validation passed")
        return

    # Load configuration: YAML -> env vars -> CLI args
    settings = _load_settings(args)

    # Override model name from CLI if provided (CLI takes precedence)
    model_name = args.model or settings.model.name
    dtype = args.dtype or settings.model.dtype

    if args.debug:
        set_debug_mode(True)
        import logging
        logging.getLogger("distllm.security").warning(
            "DEBUG MODE ENABLED: Tensor shape logging is active. "
            "This may leak sensitive information about model architecture. "
            "Do not use in production."
        )
        logger.info("Debug mode enabled: tensor shape logging active")

    create_coordinator(
        model_name=model_name,
        dtype=dtype,
        local=args.local,
        settings=settings,
    )

    host = args.host or settings.coordinator.host
    port = args.port or settings.coordinator.api_port
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


def _load_settings(args) -> DistLLMSettings:
    """Load settings from YAML config, environment variables, and CLI args."""
    # Determine config file path
    config_path = args.config
    if config_path is None:
        for candidate in ["config.yaml", os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yaml")]:
            if os.path.exists(candidate):
                config_path = candidate
                break

    # Load YAML if available
    yaml_data = load_config_file(config_path) if config_path else {}

    # Apply CLI overrides
    cli_overrides = {}
    if args.model:
        cli_overrides.setdefault("model", {})["name"] = args.model
    if args.dtype:
        cli_overrides.setdefault("model", {})["dtype"] = args.dtype
    if args.host:
        cli_overrides.setdefault("coordinator", {})["host"] = args.host
    if args.port:
        cli_overrides.setdefault("coordinator", {})["api_port"] = args.port

    if cli_overrides:
        _deep_merge(yaml_data, cli_overrides)

    return dict_to_config(yaml_data) if yaml_data else DistLLMSettings()


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


if __name__ == "__main__":
    main()
