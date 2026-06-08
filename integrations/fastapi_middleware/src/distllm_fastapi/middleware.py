"""FastAPI middleware that proxies OpenAI-compatible requests to DistLLM."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

import httpx
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

logger = logging.getLogger("distllm_fastapi")

_DISTLLM_PATHS = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/models",
}


class DistLLMMiddleware(BaseHTTPMiddleware):
    """Middleware that intercepts OpenAI-compatible API calls and forwards
    them to a DistLLM cluster.

    Usage::

        from fastapi import FastAPI
        from distllm_fastapi import DistLLMMiddleware

        app = FastAPI()
        app.add_middleware(
            DistLLMMiddleware,
            distllm_url="http://localhost:8000",
        )

    Parameters
    ----------
    app : FastAPI
        The FastAPI application.
    distllm_url : str
        DistLLM coordinator URL.
    prefix : str
        URL prefix to intercept (default: ``"/v1"``).
    passthrough_paths : set[str], optional
        Additional paths to intercept.
    """

    def __init__(
        self,
        app: Any,
        distllm_url: str = "http://localhost:8000",
        prefix: str = "/v1",
        passthrough_paths: Optional[set[str]] = None,
        **kwargs: Any,
    ):
        super().__init__(app, **kwargs)
        self.distllm_url = distllm_url.rstrip("/")
        self.prefix = prefix.rstrip("/")
        self.intercept_paths = _DISTLLM_PATHS | (passthrough_paths or set())
        self._client = httpx.AsyncClient(base_url=distllm_url, timeout=120.0)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        # Only intercept /v1/* paths
        if not path.startswith(self.prefix):
            return await call_next(request)

        # Check if it's a path we should proxy
        should_proxy = any(path.endswith(p.removeprefix("/v1")) for p in self.intercept_paths)
        if not should_proxy:
            return await call_next(request)

        # Forward to DistLLM
        body = await request.body()
        headers = dict(request.headers)
        headers.pop("host", None)

        method = request.method.upper()
        target_path = path  # e.g. /v1/chat/completions

        try:
            if method == "GET":
                resp = await self._client.get(target_path, headers=headers)
            else:
                # Check if streaming
                is_stream = False
                if body:
                    try:
                        payload = json.loads(body)
                        is_stream = payload.get("stream", False)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass

                if is_stream:
                    return StreamingResponse(
                        self._stream_forward(method, target_path, body, headers),
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache"},
                    )

                resp = await self._client.request(
                    method, target_path, content=body, headers=headers
                )

            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers),
            )
        except Exception as e:
            logger.error("DistLLM proxy error: %s", e)
            return Response(
                content=json.dumps({"error": str(e)}),
                status_code=502,
                media_type="application/json",
            )

    async def _stream_forward(
        self, method: str, path: str, body: bytes, headers: dict
    ) -> AsyncIterator[bytes]:
        async with self._client.stream(method, path, content=body, headers=headers) as resp:
            async for chunk in resp.aiter_bytes():
                yield chunk
