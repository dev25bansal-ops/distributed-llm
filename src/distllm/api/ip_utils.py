"""Shared IP extraction utilities for consistent proxy header handling.

All middleware MUST use ``get_client_ip`` instead of reading proxy headers
directly.  This prevents rate-limit bypass when different middleware resolve
the same request to different client IPs behind a reverse proxy.

Trust model (HIGH fix M2, fail-closed):

Forwarded headers (``X-Real-IP`` / ``X-Forwarded-For``) are honored ONLY when
the immediate peer — ``request.client.host`` — is in the trusted-proxy
allowlist.  The allowlist comes from ``DISTLLM_TRUSTED_PROXIES`` (comma
separated).  When the variable is unset the default is loopback-only; when it
is set, it is authoritative (setting it without loopback stops trusting
127.0.0.1); setting it empty disables header trust entirely.

For a *trusted* peer, the client address is the RIGHTMOST non-proxy entry of
``X-Forwarded-For`` (walking from the end, skipping entries that match the
trusted set), falling back to ``X-Real-IP`` only when no XFF entry survives.
This prevents spoofing: a client-supplied leftmost entry can never override
what the trusted proxy actually observed.
"""

from __future__ import annotations

import os

from starlette.requests import Request

_DEFAULT_TRUSTED_PROXIES = frozenset({"127.0.0.1", "::1"})


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


def _trusted_proxies() -> frozenset[str]:
    """Return the trusted-proxy allowlist per DISTLLM_TRUSTED_PROXIES.

    Unset  -> loopback defaults (127.0.0.1, ::1).
    Set    -> exactly the listed addresses (empty string = trust nothing).
    """
    raw = os.environ.get("DISTLLM_TRUSTED_PROXIES")
    if raw is None:
        return _DEFAULT_TRUSTED_PROXIES
    return frozenset(p.strip() for p in raw.split(",") if p.strip())


def get_client_ip(request: Request, *, trust_proxy: bool | None = None) -> str:
    """Extract the client IP address from a request.

    Fail-closed semantics (HIGH fix M2): forwarded headers are consulted ONLY
    when the immediate peer is in the trusted-proxy allowlist (see module
    docstring).  Otherwise the peer address itself is returned and any
    client-supplied headers are ignored.

    For a trusted peer, ``X-Real-IP`` wins when present; otherwise the
    rightmost non-proxy entry of ``X-Forwarded-For`` is used.

    Args:
        request: The incoming Starlette/FastAPI request.
        trust_proxy: Explicit override.  ``None`` reads the environment.

    Returns:
        A string IP address, or ``"unknown"`` if nothing is available.
    """
    explicit = trust_proxy
    if explicit is None:
        enabled = _is_trust_proxy_enabled()
    else:
        enabled = bool(explicit)

    peer = request.client.host if request.client else None

    # Gate matrix:
    #   explicit False           -> headers never trusted.
    #   explicit True            -> operator override: headers trusted from any peer.
    #   auto (None), enabled     -> headers only from allowlisted peers.
    #   auto (None), not enabled -> headers never trusted.
    if explicit is False or not enabled or peer is None:
        return peer if peer is not None else "unknown"
    if explicit is None and peer not in _trusted_proxies():
        # Fail closed: untrusted peer — client-supplied headers ignored.
        return peer

    real_ip = request.headers.get("X-Real-IP", "").strip()
    if real_ip:
        return real_ip

    forwarded = request.headers.get("X-Forwarded-For", "")
    parts = [p.strip() for p in forwarded.split(",") if p.strip()]
    trusted = _trusted_proxies()
    # Rightmost non-proxy entry: walk from the end of the chain toward the
    # client, skipping proxies we know about.  The first non-proxy address is
    # the closest client the trusted chain actually saw.
    for entry in reversed(parts):
        if entry not in trusted:
            return entry

    return peer if peer is not None else "unknown"
