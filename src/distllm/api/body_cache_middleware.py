"""Middleware that reads and caches the request body once for downstream reuse.

Problem
-------
At least three middlewares (PromptInjectionMiddleware, DedupMiddleware,
CostTrackingMiddleware) independently read and parse the request body via
``await request.body()`` or ``await request.json()``.  Each re-read triggers a
re-parse of the JSON payload, wasting CPU on large bodies.

Solution
--------
``BodyCacheMiddleware`` runs **before** all body-reading middleware in the
pipeline.  It reads the raw body once (which Starlette then caches on
``request._body``), attempts to parse it as JSON, and stores both the raw
bytes and the parsed result in ``request.state``::

    request.state.raw_body     -> bytes | None
    request.state.parsed_body  -> dict | list | None

Downstream middleware should check ``request.state.parsed_body`` first
before calling ``await request.json()`` or ``await request.body()``::

    body = getattr(request.state, "parsed_body", None)
    if body is not None:
        prompt = body.get("messages", []) ...
    else:
        # fallback — no body or non-JSON content-type

Registration
------------
Register this middleware **last** (after every other ``add_middleware`` call)
so it becomes the outermost middleware and reads the body before anything
else touches the ASGI receive stream::

    app.add_middleware(...)   # all existing middleware
    app.add_middleware(...)
    app.add_middleware(BodyCacheMiddleware)  # outermost — reads body first
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import Request
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response


class BodyCacheMiddleware(BaseHTTPMiddleware):
    """Read and cache the request body once for the entire middleware chain.

    On every request (regardless of HTTP method) the middleware:

    1. Reads ``await request.body()`` — the first call caches the raw bytes
       in Starlette's ``request._body`` so subsequent calls return immediately.
    2. Attempts to parse the bytes as UTF-8 JSON.
    3. Stores the result on ``request.state``:

       - ``request.state.raw_body`` (``bytes | None``)
       - ``request.state.parsed_body`` (``dict | list | None``)

    GET, HEAD, DELETE and other bodiless requests set both fields to ``None``.

    JSON parse failures are logged at DEBUG level and result in
    ``parsed_body = None`` — the raw bytes are still available for
    downstream middleware that may accept non-JSON content types.
    """

    # HTTP methods that are expected to carry a request body.
    _BODY_METHODS: frozenset[str] = frozenset({"POST", "PUT", "PATCH"})

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Read, cache, and expose the request body on ``request.state``."""
        if request.method in self._BODY_METHODS:
            await self._cache_body(request)
        else:
            # Bodiless requests — ensure the state keys exist so downstream
            # code can safely ``getattr(request.state, "parsed_body", None)``.
            request.state.raw_body = None
            request.state.parsed_body = None

        return await call_next(request)

    async def _cache_body(self, request: Request) -> None:
        """Read the body once, parse as JSON, and store on ``request.state``."""
        try:
            raw: bytes = await request.body()
        except Exception as exc:
            logger.opt(exception=True).debug(
                "BodyCacheMiddleware: failed to read request body: {}", exc
            )
            request.state.raw_body = None
            request.state.parsed_body = None
            return

        request.state.raw_body = raw

        if not raw:
            request.state.parsed_body = None
            return

        try:
            parsed: Any = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.opt(exception=True).debug(
                "BodyCacheMiddleware: failed to parse request body as JSON: {}", exc
            )
            request.state.parsed_body = None
            return

        request.state.parsed_body = parsed
