"""Request deduplication middleware using content fingerprinting.

Identical concurrent non-streaming requests (same prompt + params) are
collapsed: only the first is processed; subsequent requests wait for
and return the same result. Also caches responses for duplicate
requests within a configurable TTL.

Streaming requests are NEVER deduplicated to avoid breaking SSE.
"""

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from typing import Any

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


class _FingerprintCache:
    """Lightweight in-memory cache with LRU eviction for dedup results."""

    def __init__(self, max_size: int = 5000, ttl_s: float = 3600.0):
        self._max_size = max_size
        self._ttl = ttl_s
        self._cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._in_flight: dict[str, list[str]] = {}
        self._results: OrderedDict[str, tuple[float, str | None]] = OrderedDict()
        self._wait_events: dict[str, set[asyncio.Event]] = {}
        self._max_results = max_size * 2  # Cap results dict separately

    def fingerprint(self, body: bytes) -> str:
        return hashlib.sha256(body).hexdigest()

    def is_in_flight(self, fp: str) -> bool:
        return fp in self._in_flight

    def mark_in_flight(self, fp: str, request_id: str) -> None:
        if fp not in self._in_flight:
            self._in_flight[fp] = []
        self._in_flight[fp].append(request_id)
        self._results.pop(fp, None)

    def clear_in_flight(self, fp: str, request_id: str) -> None:
        ids = self._in_flight.get(fp)
        if ids:
            ids[:] = [rid for rid in ids if rid != request_id]
            if not ids:
                self._in_flight.pop(fp, None)
                self._results.pop(fp, None)
                self._signal_waiters(fp)

    def _signal_waiters(self, fp: str) -> None:
        events = self._wait_events.pop(fp, set())
        for evt in events:
            evt.set()

    def store(self, fp: str, response: str) -> None:
        self._cache[fp] = (time.time(), response)
        self._cache.move_to_end(fp)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
        self._results[fp] = (time.time(), response)
        # Evict stale results
        while len(self._results) > self._max_results:
            self._results.popitem(last=False)
        self._signal_waiters(fp)

    def lookup(self, fp: str) -> str | None:
        entry = self._cache.get(fp)
        if entry is None:
            return None
        created, response = entry
        if time.time() - created > self._ttl:
            self._cache.pop(fp, None)
            return None
        self._cache.move_to_end(fp)
        return response

    async def wait_for_result(self, fp: str, poll: float = 0.05, timeout: float = 30.0) -> str | None:
        # Check if already available
        entry = self._results.get(fp)
        if entry is not None:
            _, result = entry
            if result is not None:
                return result
        if fp not in self._in_flight:
            return None

        # Register for notification
        event = asyncio.Event()
        self._wait_events.setdefault(fp, set()).add(event)

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass

        entry = self._results.get(fp)
        return entry[1] if entry is not None else None


_cache = _FingerprintCache()


def _is_streaming_request(body_bytes: bytes) -> bool:
    """Check if a request body indicates SSE streaming."""
    try:
        body = json.loads(body_bytes)
        return bool(body.get("stream", False))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False


class DedupMiddleware(BaseHTTPMiddleware):
    """Middleware that deduplicates identical concurrent POST requests.

    Only applies to /v1/chat/completions for non-streaming requests.
    Streaming requests pass through without deduplication to preserve SSE.
    """

    async def dispatch(self, request: Request, call_next):
        if request.method != "POST" or not request.url.path.startswith("/v1/chat/completions"):
            return await call_next(request)

        body_bytes = await request.body()
        if not body_bytes:
            return await call_next(request)

        if _is_streaming_request(body_bytes):
            return await call_next(request)

        fp = _cache.fingerprint(body_bytes)
        request.state.dedup_fingerprint = fp

        cached = _cache.lookup(fp)
        if cached is not None:
            logger.debug(f"Dedup cache hit for {fp[:16]}")
            return Response(content=cached, media_type="application/json")

        blocked = await _cache.wait_for_result(fp)
        if blocked is not None:
            logger.debug(f"Dedup wait completed for {fp[:16]}")
            return Response(content=blocked, media_type="application/json")

        # H-10: Use fp (content fingerprint) as identifier instead of id(request)
        req_id = fp
        _cache.mark_in_flight(fp, req_id)

        try:
            response = await call_next(request)
            # H-11: For streaming responses, pass through without buffering
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type or "stream" in content_type:
                return response

            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
            if response.status_code == 200:
                _cache.store(fp, response_body.decode())
            return Response(content=response_body, status_code=response.status_code,
                            media_type=response.media_type, headers=dict(response.headers))
        finally:
            _cache.clear_in_flight(fp, req_id)
