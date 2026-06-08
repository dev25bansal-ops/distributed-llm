"""Server middleware stack for DistLLM API.

Extracted from server.py to improve maintainability.
Contains all middleware classes that wrap the FastAPI application.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from distllm.constants import HSTS_MAX_AGE

# API versioning constants
_API_VERSIONS: dict[str, str] = {
    "v1": "2024-01-01",
    "v2": "2025-03-01",
}
_API_VERSION_HEADER = "X-API-Version"
_API_SUNSET_HEADER = "Sunset"
_API_DEPRECATION_HEADER = "X-API-Deprecation"
_API_SUNSET_DATES: dict[str, str | None] = {
    "v1": None,
    "v2": None,
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers and API versioning to all responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        tls_enabled = os.environ.get("DISTLLM_TLS_ENABLED", "false").lower() == "true"
        if tls_enabled:
            response.headers["Strict-Transport-Security"] = f"max-age={HSTS_MAX_AGE}; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

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


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Cancel requests that exceed the timeout limit."""

    DEFAULT_TIMEOUT = 120.0
    ENDPOINT_TIMEOUTS = {
        "/v1/chat/completions": 300.0,
        "/v1/completions": 300.0,
        "/v1/embeddings": 60.0,
    }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # H-20: Normalize path before lookup — strip trailing slash
        path = request.url.path.rstrip("/")
        timeout = self.ENDPOINT_TIMEOUTS.get(path, self.DEFAULT_TIMEOUT)
        per_request = getattr(request.state, "request_timeout", None)
        if per_request is not None:
            timeout = per_request

        try:
            async with asyncio.timeout(timeout):
                response = await call_next(request)
                return response
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={
                    "error": {
                        "message": f"Request exceeded {timeout:.0f}s timeout limit",
                        "type": "timeout_error",
                        "code": "504",
                    }
                },
            )


class BackpressureMiddleware(BaseHTTPMiddleware):
    """Reject requests when system is under heavy load."""

    MAX_PENDING_REQUESTS = 1000

    def __init__(self, app: Any, get_coordinator: Callable = lambda: None) -> None:
        super().__init__(app)
        self._get_coordinator = get_coordinator

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in ("/health", "/ready", "/live", "/metrics", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        coord = self._get_coordinator()
        if coord and getattr(coord, "_shutting_down", False):
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "Service is shutting down",
                        "type": "shutdown_error",
                        "code": "503",
                    }
                },
            )

        if coord and coord.scheduler:
            try:
                stats = coord.scheduler.stats()
                pending = stats.get("pending_requests", 0)
                if isinstance(pending, (int, float)) and pending >= self.MAX_PENDING_REQUESTS:
                    return JSONResponse(
                        status_code=503,
                        content={
                            "error": {
                                "message": f"System overloaded: {pending} pending requests",
                                "type": "backpressure_error",
                                "code": "503",
                            }
                        },
                    )
            except (AttributeError, TypeError, KeyError):
                pass

        return await call_next(request)


class RequestSizeLimitMiddleware:
    """Limit maximum request body size to prevent OOM."""

    MAX_REQUEST_SIZE = 32 * 1024 * 1024  # 32 MB

    def __init__(self, app: Any) -> None:
        self.app = app

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
                            }
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
                    }
                },
            )
            await response(scope, receive, send)


class _RequestTooLarge(Exception):
    pass
