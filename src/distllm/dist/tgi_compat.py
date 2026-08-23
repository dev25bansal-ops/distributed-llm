"""TGI-compatible API endpoint — wraps the coordinator in a HuggingFace TGI-compatible interface.

Provides a drop-in replacement endpoint that speaks the same protocol as
`text-generation-inference <https://github.com/huggingface/text-generation-inference>`_,
so existing TGI clients, inference libraries, and monitoring tools work
without modification.

Endpoints implemented:
    POST /generate            — Single-step generation (streaming via SSE)
    POST /generate_stream     — Streaming generation (SSE, TGI-flavored)
    GET  /health              — Health check
    GET  /info                — Model info (model id, version, limits)

Usage:
    # In your coordinator startup (FastAPI app), mount the router:
    from distllm.dist.tgi_compat import tgi_router
    app.include_router(tgi_router, prefix="/tgi")

    # Or use standalone on a separate port:
    uvicorn distllm.dist.tgi_compat:app --port 8080
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field


# ── Request / response models ────────────────────────────────────────


class TGIRequest(BaseModel):
    inputs: str = Field(..., description="Input prompt")
    parameters: dict[str, Any] | None = Field(default=None, description="Generation parameters")
    stream: bool = Field(default=False, description="Whether to stream the response")


class TGIGenerateResponse(BaseModel):
    generated_text: str = ""
    details: dict[str, Any] | None = None


class TGIStreamChunk(BaseModel):
    token: dict[str, Any] | None = None
    generated_text: str | None = None
    details: dict[str, Any] | None = None


class TGIHealthResponse(BaseModel):
    status: str = "healthy"


class TGIInfoResponse(BaseModel):
    model_id: str = "distributed-llm"
    model_dtype: str = "float16"
    sha: str = ""
    max_input_length: int = 131072
    max_total_tokens: int = 139264
    version: str = "2.0.0"


# ── Router ────────────────────────────────────────────────────────────

tgi_router = APIRouter(tags=["tgi"])


def _get_coordinator():
    """Lazy import of the global coordinator reference."""
    try:
        from distllm.api.api_state import g
        return g.coordinator
    except (ImportError, AttributeError):
        return None


def _default_params() -> dict[str, Any]:
    return {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "do_sample": True,
        "seed": None,
        "stop": [],
    }


# ── OpenAI-compatible wrapper ─────────────────────────────────────────

def _to_openai_params(tgi_params: dict[str, Any] | None) -> dict[str, Any]:
    """Convert TGI parameter names to the internal OpenAI-compatible names."""
    params = dict(_default_params())
    if tgi_params is None:
        return params

    mapping = {
        "max_new_tokens": "max_tokens",
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "repetition_penalty": "frequency_penalty",
        "seed": "seed",
        "stop": "stop",
    }

    for tgi_key, internal_key in mapping.items():
        if tgi_key in tgi_params:
            params[internal_key] = tgi_params[tgi_key]

    # TGI stop sequences can be a list or single string.
    stop = tgi_params.get("stop", [])
    if isinstance(stop, str):
        params["stop"] = [stop]
    elif isinstance(stop, list):
        params["stop"] = stop

    return params


# ── Endpoints ─────────────────────────────────────────────────────────


@tgi_router.post(
    "/generate",
    summary="TGI Generate",
    description="Single-step text generation endpoint compatible with HuggingFace TGI.",
    response_model=TGIGenerateResponse,
)
async def tgi_generate(body: TGIRequest) -> TGIGenerateResponse | StreamingResponse:
    """Generate text (non-streaming)."""
    if body.stream:
        return await tgi_generate_stream(body)

    coord = _get_coordinator()
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    params = _to_openai_params(body.parameters)

    # Build the internal request payload.
    internal_request = {
        "model": getattr(coord, "model_name", "distributed-llm"),
        "messages": [{"role": "user", "content": body.inputs}],
        "max_tokens": params["max_tokens"],
        "temperature": params["temperature"],
        "stream": False,
    }

    try:
        result = await coord.process_request(internal_request)
        # Extract generated text from the response.
        if isinstance(result, dict):
            choices = result.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                generated = message.get("content", "")
            else:
                generated = result.get("text", "")
        else:
            generated = str(result)

        return TGIGenerateResponse(generated_text=generated)

    except Exception as e:
        logger.error(f"TGI generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@tgi_router.post(
    "/generate_stream",
    summary="TGI Stream Generate",
    description="Streaming generation endpoint compatible with HuggingFace TGI.",
)
async def tgi_generate_stream(body: TGIRequest) -> StreamingResponse:
    """Generate text (streaming, TGI-flavored SSE)."""
    coord = _get_coordinator()
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    params = _to_openai_params(body.parameters)

    internal_request = {
        "model": getattr(coord, "model_name", "distributed-llm"),
        "messages": [{"role": "user", "content": body.inputs}],
        "max_tokens": params["max_tokens"],
        "temperature": params["temperature"],
        "stream": True,
    }

    async def event_stream():
        """Yield TGI-flavored SSE events."""
        try:
            async for chunk in coord.process_streaming(internal_request):
                # Convert internal chunk to TGI format.
                text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                if text:
                    tgi_chunk = {"token": {"text": text, "logprob": None, "special": False}}
                    yield f"data: {json.dumps(tgi_chunk)}\n\n"

            # Final done chunk with full generated text.
            yield f"data: {json.dumps({'generated_text': ''})}\n\n"

        except Exception as e:
            logger.error(f"TGI stream failed: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@tgi_router.get("/health", response_model=TGIHealthResponse)
async def tgi_health() -> TGIHealthResponse:
    """Health check endpoint (TGI-compatible)."""
    return TGIHealthResponse(status="healthy")


@tgi_router.get("/info", response_model=TGIInfoResponse)
async def tgi_info() -> TGIInfoResponse:
    """Model info endpoint (TGI-compatible)."""
    coord = _get_coordinator()
    model_id = "distributed-llm"
    if coord is not None:
        model_id = getattr(coord, "model_name", model_id)
    return TGIInfoResponse(model_id=model_id)


# ── Standalone app ────────────────────────────────────────────────────

app = None


def create_standalone_app() -> Any:
    """Create a standalone FastAPI app for the TGI server (runs without
    the coordinator on port 8080 for testing/compatibility)."""
    from fastapi import FastAPI
    app = FastAPI(title="DistLLM TGI-compatible API")
    app.include_router(tgi_router)
    return app
