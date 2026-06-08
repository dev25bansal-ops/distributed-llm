"""FastAPI router that adds OpenAI-compatible endpoints backed by DistLLM."""

from __future__ import annotations

from typing import Any, Optional

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse


def create_distllm_router(
    distllm_url: str = "http://localhost:8000",
    prefix: str = "/v1",
    tags: Optional[list[str]] = None,
) -> APIRouter:
    """Create a FastAPI ``APIRouter`` with OpenAI-compatible endpoints.

    Instead of middleware interception, this adds explicit routes that
    forward to DistLLM.  Use when you want the proxy to appear in your
    OpenAPI docs.

    Usage::

        from fastapi import FastAPI
        from distllm_fastapi import create_distllm_router

        app = FastAPI()
        router = create_distllm_router(distllm_url="http://localhost:8000")
        app.include_router(router)
    """
    router = APIRouter(prefix=prefix, tags=tags or ["distllm"])
    client = httpx.AsyncClient(base_url=distllm_url, timeout=120.0)

    @router.post("/chat/completions")
    async def chat_completions(request: Request):
        body = await request.body()
        payload = await request.json()

        if payload.get("stream"):
            async def _stream():
                async with client.stream("POST", "/v1/chat/completions", content=body) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk

            return StreamingResponse(_stream(), media_type="text/event-stream")

        resp = await client.post("/v1/chat/completions", content=body)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)

    @router.post("/completions")
    async def completions(request: Request):
        body = await request.body()
        resp = await client.post("/v1/completions", content=body)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)

    @router.post("/embeddings")
    async def embeddings(request: Request):
        body = await request.body()
        resp = await client.post("/v1/embeddings", content=body)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)

    @router.get("/models")
    async def list_models():
        resp = await client.get("/v1/models")
        return JSONResponse(content=resp.json(), status_code=resp.status_code)

    return router
