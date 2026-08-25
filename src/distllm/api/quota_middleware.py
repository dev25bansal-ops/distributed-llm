"""Quota enforcement middleware for the API layer.

Tracks token usage per request, enforces per-tenant quotas
(tokens/day, requests/minute, max concurrency), and records
usage for billing.

Integrates with the ``UsageMeter`` and the role-based auth system
to attribute usage to API keys and tenants.

Mounted in ``server.py`` immediately after AuthMiddleware executes
(registered before it in code so Starlette makes it inner), gated by
``DISTLLM_QUOTA_ENABLED`` (default "1" = on).
"""

from __future__ import annotations

import os
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from distllm.api.errors import error_response
from distllm.core.usage_meter import UsageMeter, create_usage_meter

from loguru import logger


QUOTA_ENABLED_ENV = "DISTLLM_QUOTA_ENABLED"


def quota_enabled_from_env() -> bool:
    """Read the quota gate from the environment at call time.

    Reading at mount/construct time (instead of once at module import)
    keeps the gate testable and honors env changes made before the app
    object is built.
    """
    return os.environ.get(QUOTA_ENABLED_ENV, "1") == "1"


# Import-time snapshot kept for backwards compatibility; QuotaMiddleware
# itself re-reads the env when constructed with ``enable=None``.
_ENABLED = quota_enabled_from_env()

# Module-level singleton (initialized on first use)
_meter: UsageMeter | None = None


def get_usage_meter() -> UsageMeter:
    """Get or create the singleton usage meter."""
    global _meter
    if _meter is None:
        db = os.environ.get("DISTLLM_USAGE_DB", ".usage.db")
        _meter = create_usage_meter(storage_path=db)
    return _meter


def reset_usage_meter() -> None:
    """Drop the singleton so the next request builds a fresh meter.

    Used by tests to isolate per-test UsageMeter state; production code
    never needs to call this.
    """
    global _meter
    if _meter is not None:
        _meter.close()
    _meter = None


# ── TenantBillingManager backend (tiered plans) ────────────────────────────
#
# Alternative enforcement backend backed by ``core.tenant_billing``
# (free/pro/enterprise tiers, rolling req/min, tokens/day, monthly cost cap).
# Selected per-middleware-instance via ``tenant_billing=True`` or the
# ``DISTLLM_TENANT_BILLING=1`` env var.  When active, quota decisions come
# from ``TenantBillingManager.check()`` (AllowDeny) instead of the
# UsageMeter's per-tenant QuotaLimit rows.

_billing_manager = None


def get_billing_manager():
    """Get or create the module-level TenantBillingManager singleton."""
    global _billing_manager
    if _billing_manager is None:
        from distllm.core.tenant_billing import get_tenant_billing_manager
        _billing_manager = get_tenant_billing_manager()
    return _billing_manager


def reset_billing_manager() -> None:
    """Drop the billing-manager singleton (test isolation helper)."""
    global _billing_manager
    _billing_manager = None


TENANT_BILLING_ENV = "DISTLLM_TENANT_BILLING"


def _tenant_billing_from_env() -> bool:
    return os.environ.get(TENANT_BILLING_ENV, "0") == "1"


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

    Requires ``request.state.tenant_id`` (SSO requests) or
    ``request.state.api_key_id`` (API-key requests) to be set by earlier
    middleware — i.e. it must execute AFTER AuthMiddleware.  In server.py
    it is registered BEFORE ``add_middleware(AuthMiddleware)``, which under
    Starlette's prepend semantics places it inside (after) auth.

    Tenant attribution falls back ``tenant_id`` -> ``api_key_id`` ->
    ``"anonymous"`` because AuthMiddleware sets only ``api_key_*`` fields.

    To disable, set ``DISTLLM_QUOTA_ENABLED=0`` or pass ``enable=False``.
    """

    def __init__(
        self,
        app,
        enable: bool | None = None,
        tenant_billing: bool | None = None,
    ) -> None:
        super().__init__(app)
        self._enabled = (
            quota_enabled_from_env() if enable is None else enable
        )
        self._tenant_billing = (
            _tenant_billing_from_env() if tenant_billing is None else tenant_billing
        )

    async def dispatch(self, request: Request, call_next):
        if not self._enabled:
            return await call_next(request)

        # Skip non-tracked paths BEFORE touching the meter so untracked
        # traffic never constructs the SQLite-backed singleton.
        path = request.url.path
        if not self._should_track(path):
            return await call_next(request)

        tenant_id = getattr(request.state, "tenant_id", None) or getattr(
            request.state, "api_key_id", "anonymous"
        )
        key_id = getattr(request.state, "api_key_id", "")

        if self._tenant_billing:
            return await self._dispatch_tenant_billing(
                request, call_next, tenant_id, key_id
            )

        meter = get_usage_meter()

        # Check quota before processing
        allowed, reason = meter.enforce_quota(tenant_id)
        if not allowed:
            logger.warning(f"Quota exceeded for {tenant_id}: {reason}")
            # Best-effort Retry-After: daily-token limits reset at midnight,
            # rate/concurrency limits refill per minute.  Without parsing
            # `reason` we advertise the conservative 60s window (matching the
            # limiter's shortest refill period).
            return self._quota_429(request, reason)

        start_time = time.monotonic()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            meter.release_quota(tenant_id)

            # Record usage for successful streaming endpoints
            if response is not None and response.status_code < 400:
                self._record_usage(request, response, meter, tenant_id, key_id, elapsed_ms)

    async def _dispatch_tenant_billing(self, request, call_next, tenant_id: str, key_id: str):
        """Enforce via TenantBillingManager (tiered plans) instead of UsageMeter.

        The manager's rolling windows are consumed by ``check(consume=True)``
        on the request path; token/cost aggregation happens post-response so a
        rejected request never inflates the tenant's usage.
        """
        mgr = get_billing_manager()

        decision = mgr.check(tenant_id)
        if not decision.allowed:
            logger.warning(f"Quota exceeded for {tenant_id}: {decision.reason}")
            return self._quota_429(request, decision.reason)

        start_time = time.monotonic()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            # Record usage for successful requests (tokens aggregated from
            # the same estimation logic used by the UsageMeter backend).
            if response is not None and response.status_code < 400:
                input_tokens, output_tokens = self._estimate_usage(request, response)
                if input_tokens or output_tokens:
                    try:
                        cost_usd, _compute_s = mgr.estimate_request_cost(
                            input_tokens, output_tokens,
                            getattr(request.state, "model", ""),
                        )
                    except Exception as e:  # cost tracker unavailable — don't fail the request
                        logger.debug(f"Cost estimate failed: {e}")
                        cost_usd = 0.0
                    mgr.record_usage(
                        tenant_id=tenant_id,
                        tokens_in=input_tokens,
                        tokens_out=output_tokens,
                        model_name=getattr(request.state, "model", ""),
                        endpoint=request.url.path,
                        key_id=key_id,
                        cost_usd=cost_usd,
                    )

    def _quota_429(self, request: Request, reason: str):
        """Build the standard 429 quota response (envelope + Retry-After header)."""
        resp = error_response(
            status_code=429,
            error="Too Many Requests",
            message=f"Quota exceeded: {reason}",
            type="quota_exceeded",
            code="429",
            request_id=getattr(request.state, "request_id", None),
            retry_after=60,
        )
        # error_response embeds retry_after in the body only; the
        # Retry-After HTTP header is expected on 429s (RFC 7231/6585).
        resp.headers["Retry-After"] = "60"
        return resp

    def _should_track(self, path: str) -> bool:
        """Return True if the path should be tracked for usage."""
        tracked_prefixes = ("/v1/chat/completions", "/v1/completions", "/v1/embeddings")
        return any(path.startswith(p) for p in tracked_prefixes)

    def _estimate_usage(self, request: Request, response) -> tuple[int, int]:
        """Estimate ``(input_tokens, output_tokens)`` from request/response."""
        input_tokens = 0
        output_tokens = 0

        content_type = request.headers.get("content-type", "")
        if "json" in content_type:
            # Prefer BodyCacheMiddleware's parsed snapshot; fall back to
            # FastAPI's cached body.  If neither is available, skip token
            # estimation rather than attempting a fragile async body read
            # that can deadlock.
            body = getattr(request.state, "parsed_body", None) or getattr(request, "_json", None)
            prompt = ""
            try:
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

        return input_tokens, output_tokens

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

        input_tokens, output_tokens = self._estimate_usage(request, response)

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
