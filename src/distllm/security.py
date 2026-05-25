"""Security helpers shared by runtime integrations."""

from __future__ import annotations

import ipaddress
import os
import socket
import urllib.request
from typing import Any
from urllib.parse import urlparse


def hf_revision(revision: str | None = None) -> str:
    """Return the Hugging Face revision to use for model/dataset downloads.

    Production deployments should set DISTLLM_MODEL_REVISION or pass an
    immutable commit SHA. For developer convenience we default to "main" unless
    DISTLLM_REQUIRE_MODEL_REVISION=1 is set.
    """
    resolved = revision or os.getenv("DISTLLM_MODEL_REVISION") or os.getenv("HF_MODEL_REVISION")
    if resolved:
        return resolved
    if os.getenv("DISTLLM_REQUIRE_MODEL_REVISION", "").lower() in {"1", "true", "yes"}:
        raise RuntimeError(
            "Hugging Face downloads require an explicit immutable revision. "
            "Set DISTLLM_MODEL_REVISION or HF_MODEL_REVISION."
        )
    return "main"


def validate_http_url(url: str, *, allow_private_hosts: bool = False) -> str:
    """Validate that a URL uses HTTP(S) and, by default, resolves publicly."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r}")
    if not parsed.hostname:
        raise ValueError("URL must include a hostname")

    if allow_private_hosts:
        return url

    addresses = socket.getaddrinfo(parsed.hostname, parsed.port, type=socket.SOCK_STREAM)
    for *_, sockaddr in addresses:
        ip = ipaddress.ip_address(sockaddr[0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError(f"URL resolves to a non-public address: {ip}")
    return url


def safe_urlopen(
    request_or_url: urllib.request.Request | str,
    *,
    timeout: float,
    allow_private_hosts: bool = False,
) -> Any:
    """Open a URL only after scheme/host validation."""
    url = (
        request_or_url.full_url
        if isinstance(request_or_url, urllib.request.Request)
        else request_or_url
    )
    validate_http_url(url, allow_private_hosts=allow_private_hosts)
    return urllib.request.urlopen(request_or_url, timeout=timeout)  # nosec B310  # noqa: S310
