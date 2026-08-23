"""Text completion routes: POST /v1/completions."""

import asyncio
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ConfigDict

from ..api_state import g
from ..streaming import _get_client_id, _stream_response


router = APIRouter(tags=["completion"])


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
    prompt: str = Field(..., min_length=1, max_length=131072, description="The prompt text to generate from")
    max_tokens: int = Field(default=256, ge=0, le=8192, description="Maximum tokens to generate (0=return immediately, 1-8192)")
    temperature: float = Field(default=0.7, ge=0, le=2.0, description="Sampling temperature (0-2.0)")
    top_p: float = Field(default=0.9, ge=0, le=1.0, description="Nucleus sampling threshold (0-1)")
    top_k: int = Field(default=0, ge=0, description="Top-k sampling (0 = disabled)")
    stream: bool = Field(default=False, description="Whether to stream the response")
    response_format: dict | None = Field(default=None, description="Response format constraint, e.g. {'type': 'json_object'} or {'type': 'json_schema', 'schema': {...}}")
    priority: int = Field(default=2, ge=0, le=3, description="Request priority: 0=critical, 1=high, 2=normal, 3=low")
    user: str | None = Field(default=None, description="Tenant/user identifier for rate limiting")


class CompletionChoice(BaseModel):
    index: int = 0
    text: str = ""
    delta: str | None = None
    finish_reason: str | None = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:12]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "distributed-llm"
    choices: list[CompletionChoice]
    generation_time: float | None = None


@router.post(
    "/v1/completions",
    summary="Create text completion",
    description="Generate text completions from a prompt. Supports streaming, structured output via response_format, and priority scheduling. OpenAI-compatible request/response format.",
    response_description="Text completion response with generated text and usage statistics",
    responses={
        503: {"description": "No model loaded"},
    },
)
async def completions(request: Request, body: CompletionRequest):
    """Text completions endpoint."""
    # Set observability state for middleware
    request.state.model = body.model
    request.state.tenant = body.user or "default"

    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    # Build schema constraint for structured output
    schema = None
    if body.response_format:
        fmt_type = body.response_format.get("type", "")
        if fmt_type == "json_object":
            schema = {}
        elif fmt_type == "json_schema" and "schema" in body.response_format:
            schema = body.response_format["schema"]

    if body.max_tokens == 0:
        return CompletionResponse(
            model=body.model,
            choices=[CompletionChoice(text="", finish_reason="length")],
        )

    if body.stream:
        client_id = _get_client_id(request)
        return StreamingResponse(
            _stream_response(
                body.prompt, body, "text_completion.chunk", "cmpl-",
                response_format=body.response_format,
                client_id=client_id, endpoint="/v1/completions",
            ),
            media_type="text/event-stream",
        )

    start_time = time.time()

    result = await asyncio.to_thread(
        coord.generate,
        body.prompt,
        body.max_tokens,
        body.temperature,
        body.top_p,
        user_id=getattr(request.state, "tenant", "default"),
    )
    elapsed = time.time() - start_time

    generated = result

    # Validate structured output if response_format specified
    if body.response_format and generated:
        from distllm.core.structured_output import validate_structured_output
        fmt_type = body.response_format.get("type", "")
        validation_schema = None
        if fmt_type == "json_schema" and "schema" in body.response_format:
            validation_schema = body.response_format["schema"]
        elif fmt_type == "json_object":
            validation_schema = {}
        if validation_schema is not None:
            validated = validate_structured_output(generated, validation_schema)
            if validated is None:
                from loguru import logger
                logger.warning(f"Structured output validation failed for response_format={fmt_type}")

    return CompletionResponse(
        model=body.model,
        choices=[
            CompletionChoice(
                text=generated,
                finish_reason="stop",
            )
        ],
        generation_time=round(elapsed, 3),
    )
