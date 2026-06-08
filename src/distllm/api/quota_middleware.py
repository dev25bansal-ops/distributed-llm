"""Quota enforcement middleware for the API layer.

Tracks token usage per request, enforces per-tenant quotas
(tokens/day, requests/minute, max concurrency), and records
usage for billing.

Integrates with the ``UsageMeter`` and the role-based auth system
to attribute usage to API keys and tenants.
"""

from __future__ import annotations

import os
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from distllm.core.usage_meter import UsageMeter, create_usage_meter

from loguru import logger


_ENABLED = os.environ.get("DISTLLM_QUOTA_ENABLED", "0") == "1"

# Module-level singleton (initialized on first use)
_meter: UsageMeter | None = None


def get_usage_meter() -> UsageMeter:
    """Get or create the singleton usage meter."""
    global _meter
    if _meter is None:
        db = os.environ.get("DISTLLM_USAGE_DB", ".usage.db")
        _meter = create_usage_meter(storage_path=db)
    return _meter


def _estimate_token_count(text: str) -> int:
    """Estimate token count for a string.

    Uses tiktoken (cl100k_base) if available for accurate counting,
    falls back to len(text) // 4 heuristic.
    """
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text, disallowed_special=()))
    except ImportError:
        return max(1, len(text) // 4)
    except Exception:
        return max(1, len(text) // 4)


class QuotaMiddleware(BaseHTTPMiddleware):
    """Middleware that records usage and enforces per-tenant quotas.

    Requires ``request.state.tenant_id`` and ``request.state.api_key_role``
    to be set by earlier middleware (e.g. AuthMiddleware).

    To enable, set ``DISTLLM_QUOTA_ENABLED=1`` or pass ``enable=True``.
    """

    def __init__(self, app, enable: bool | None = None) -> None:
        super().__init__(app)
        self._enabled = _ENABLED if enable is None else enable

    async def dispatch(self, request: Request, call_next):
        if not self._enabled:
            return await call_next(request)

        meter = get_usage_meter()

        # Skip non-tracked paths
        path = request.url.path
        if not self._should_track(path):
            return await call_next(request)

        tenant_id = getattr(request.state, "tenant_id", None) or getattr(
            request.state, "api_key_id", "anonymous"
        )
        key_id = getattr(request.state, "api_key_id", "")

        # Check quota before processing
        allowed, reason = meter.enforce_quota(tenant_id)
        if not allowed:
            logger.warning(f"Quota exceeded for {tenant_id}: {reason}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": f"Quota exceeded: {reason}",
                        "type": "quota_exceeded",
                        "code": "429",
                    }
                },
            )

        start_time = time.monotonic()
        try:
            response = await call_next(request)
            return response
        finally:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            meter.release_quota(tenant_id)

            # Record usage for successful streaming endpoints
            if response.status_code < 400:
                self._record_usage(request, response, meter, tenant_id, key_id, elapsed_ms)

    def _should_track(self, path: str) -> bool:
        """Return True if the path should be tracked for usage."""
        tracked_prefixes = ("/v1/chat/completions", "/v1/completions", "/v1/embeddings")
        return any(path.startswith(p) for p in tracked_prefixes)

    def _record_usage(
        self,
        request: Request,
        response,
        meter: UsageMeter,
        tenant_id: str,
        key_id: str,
        elapsed_ms: float,
    ) -> None:
        """Estimate and record token usage from request/response."""
        model = getattr(request.state, "model", "unknown")

        # Estimate input tokens from request body
        input_tokens = 0
        output_tokens = 0

        content_type = request.headers.get("content-type", "")
        if "json" in content_type:
            try:
                body = getattr(request, "_json", None) or getattr(request.state, "parsed_body", None)
                if body is None and hasattr(request, "body"):
                    # C-06: Use asyncio.get_event_loop() instead of None
                    try:
                        loop = asyncio.get_event_loop()
                        raw = loop.create_task(request.body())
                    except RuntimeError:
                        # No running loop — use run_coroutine_threadsafe with the main loop
                        try:
                            loop = asyncio.get_running_loop()
                            raw = asyncio.run_coroutine_threadsafe(request.body(), loop)
                        except RuntimeError:
                            logger.debug("No event loop available for quota body read")
                            raw = None
                prompt = ""
                if body:
                    if isinstance(body, dict):
                        prompt = body.get("prompt", "") or ""
                        messages = body.get("messages", [])
                        if messages:
                            prompt = " ".join(
                                m.get("content", "") for m in messages if isinstance(m, dict)
                            )
                input_tokens = _estimate_token_count(prompt)
            except (ValueError, TypeError, AttributeError) as e:
                logger.debug(f"Failed to estimate input tokens: {e}")
                input_tokens = 0

        # Estimate output tokens from response
        try:
            if hasattr(response, "body"):
                raw_body = response.body
                if raw_body:
                    import json
                    try:
                        data = json.loads(raw_body)
                        if isinstance(data, dict):
                            choices = data.get("choices", [])
                            if choices:
                                text = choices[0].get("text", "") or choices[0].get("message", {}).get("content", "")
                                output_tokens = _estimate_token_count(text)
                    except (json.JSONDecodeError, TypeError):
                        pass
        except (AttributeError, ValueError) as e:
            logger.debug(f"Failed to estimate output tokens: {e}")
            output_tokens = 0

        if input_tokens or output_tokens:
            meter.record_request(
                tenant_id=tenant_id,
                model_name=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens or 1,
                duration_ms=elapsed_ms,
                endpoint=request.url.path,
                key_id=key_id,
            )
