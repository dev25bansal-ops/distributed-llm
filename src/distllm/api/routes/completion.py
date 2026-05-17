"""Text completion routes: POST /v1/completions."""

import asyncio
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ConfigDict

from ..api_state import g
from ..streaming import _stream_response


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
    prompt: str = Field(..., max_length=131072, description="The prompt text to generate from")
    max_tokens: int = Field(default=256, ge=1, le=8192, description="Maximum tokens to generate (1-8192)")
    temperature: float = Field(default=0.7, ge=0, le=2.0, description="Sampling temperature (0-2.0)")
    top_p: float = Field(default=0.9, ge=0, le=1.0, description="Nucleus sampling threshold (0-1)")
    top_k: int = Field(default=0, ge=0, description="Top-k sampling (0 = disabled)")
    stream: bool = Field(default=False, description="Whether to stream the response")
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


@router.post("/v1/completions")
async def completions(request: Request, body: CompletionRequest):
    """Text completions endpoint."""
    # Set observability state for middleware
    request.state.model = body.model
    request.state.tenant = body.user or "default"

    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    # Read speculative decoding headers (override config defaults)
    spec_num_tokens = request.headers.get("x-speculative-num-tokens")
    spec_method = request.headers.get("x-speculative-method")
    original_num = None
    original_method = None

    if spec_num_tokens and coord._spec_decoder:
        original_num = coord._spec_decoder.num_assistant_tokens
        coord._spec_decoder.num_assistant_tokens = int(spec_num_tokens)
    if spec_method and coord._spec_decoder:
        original_method = coord._spec_decoder.method
        coord._spec_decoder.method = spec_method

    try:
        if body.stream:
            return StreamingResponse(
                _stream_response(body.prompt, body, "text_completion.chunk", "cmpl-"),
                media_type="text/event-stream",
            )

        start_time = time.time()
        result = await asyncio.to_thread(
            coord.generate,
            body.prompt,
            body.max_tokens,
            body.temperature,
            body.top_p,
        )
        elapsed = time.time() - start_time

        generated = result[len(body.prompt):] if result.startswith(body.prompt) else result

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
    finally:
        # Restore original speculative settings
        if spec_num_tokens and coord._spec_decoder:
            coord._spec_decoder.num_assistant_tokens = original_num
        if spec_method and coord._spec_decoder:
            coord._spec_decoder.method = original_method
