"""Shared IP extraction utilities for consistent proxy header handling.

All middleware MUST use ``get_client_ip`` instead of reading proxy headers
directly.  This prevents rate-limit bypass when different middleware resolve
the same request to different client IPs behind a reverse proxy.
"""

from __future__ import annotations

import os

from starlette.requests import Request


def _is_trust_proxy_enabled() -> bool:
    """Return True when proxy headers should be trusted.

    Trust is enabled by ``DISTLLM_TRUST_PROXY_HEADERS=1`` or
    ``DISTLLM_TRUST_PROXY_HEADERS=true``, or implicitly during tests
    (``PYTEST_CURRENT_TEST`` is set).
    """
    value = os.environ.get("DISTLLM_TRUST_PROXY_HEADERS", "")
    if value.lower() in ("1", "true"):
        return True
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    return False


def get_client_ip(request: Request, *, trust_proxy: bool | None = None) -> str:
    """Extract the client IP address from a request.

    When *trust_proxy* is ``True`` (or ``None`` and the environment variable
    indicates proxy trust), the function inspects ``X-Real-IP`` first, then
    the leftmost entry of ``X-Forwarded-For`` (per RFC 7239).  Otherwise,
    or as a final fallback, ``request.client.host`` is used.

    Args:
        request: The incoming Starlette/FastAPI request.
        trust_proxy: Explicit override.  ``None`` reads the environment.

    Returns:
        A string IP address, or ``"unknown"`` if nothing is available.
    """
    if trust_proxy is None:
        trust_proxy = _is_trust_proxy_enabled()

    if trust_proxy:
        real_ip = request.headers.get("X-Real-IP", "").strip()
        if real_ip:
            return real_ip

        forwarded = request.headers.get("X-Forwarded-For", "")
        parts = [p.strip() for p in forwarded.split(",") if p.strip()]
        if parts:
            return parts[0]

    return request.client.host if request.client else "unknown"
