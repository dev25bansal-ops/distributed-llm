"""FastAPI proxy server for the distributed-llm router.

Exposes OpenAI-compatible endpoints and proxies requests to
the appropriate coordinator using sticky session routing.
"""

import hashlib
import time
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from distllm.router.service import RouterService, compute_session_key

router = APIRouter()

_router_service: Optional[RouterService] = None


def set_router_service(service: RouterService) -> None:
    global _router_service
    _router_service = service


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Proxy chat completions to the appropriate coordinator."""
    if _router_service is None:
        raise HTTPException(503, "Router not initialized")

    body = await request.json()
    session_key = compute_session_key(body, request.client.host if request.client else None)

    coordinator = _router_service.get_coordinator(session_key)
    if not coordinator:
        raise HTTPException(503, "No healthy coordinators available")

    url = f"{coordinator.url}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-ID": request.headers.get("X-Request-ID", ""),
    }

    is_stream = body.get("stream", False)

    if is_stream:
        async def stream_proxy():
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("POST", url, json=body, headers=headers) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        yield f"data: {error_body.decode()}\n\n"
                        return
                    async for chunk in resp.aiter_text():
                        yield chunk

        return StreamingResponse(stream_proxy(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(url, json=body, headers=headers)
            if resp.status_code != 200:
                raise HTTPException(resp.status_code, detail=resp.text)
            return resp.json()


@router.post("/v1/completions")
async def completions(request: Request):
    """Proxy text completions to the appropriate coordinator."""
    if _router_service is None:
        raise HTTPException(503, "Router not initialized")

    body = await request.json()
    session_key = compute_session_key(body, request.client.host if request.client else None)

    coordinator = _router_service.get_coordinator(session_key)
    if not coordinator:
        raise HTTPException(503, "No healthy coordinators available")

    url = f"{coordinator.url}/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-ID": request.headers.get("X-Request-ID", ""),
    }

    is_stream = body.get("stream", False)

    if is_stream:
        async def stream_proxy():
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("POST", url, json=body, headers=headers) as resp:
                    if resp.status_code != 200:
                        return
                    async for chunk in resp.aiter_text():
                        yield chunk

        return StreamingResponse(stream_proxy(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(url, json=body, headers=headers)
            if resp.status_code != 200:
                raise HTTPException(resp.status_code, detail=resp.text)
            return resp.json()


@router.get("/v1/models")
async def list_models():
    """Aggregate model info from all coordinators."""
    if _router_service is None:
        raise HTTPException(503, "Router not initialized")

    coordinators = _router_service.discovery.get_healthy()
    if not coordinators:
        raise HTTPException(503, "No healthy coordinators available")

    # Return models from the first healthy coordinator
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{coordinators[0].url}/v1/models")
        if resp.status_code == 200:
            return resp.json()
        raise HTTPException(resp.status_code, detail=resp.text)


@router.get("/health")
async def router_health():
    """Router health and coordinator status summary."""
    if _router_service is None:
        return {"status": "unhealthy", "reason": "Router not initialized"}

    coordinators = _router_service.discovery.get_all()
    healthy = [c for c in coordinators.values() if c.healthy]

    return {
        "status": "healthy" if healthy else "degraded",
        "coordinators_total": len(coordinators),
        "coordinators_healthy": len(healthy),
        "ring_size": _router_service.ring.node_count,
    }
