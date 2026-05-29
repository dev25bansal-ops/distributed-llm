"""Security helpers shared by runtime integrations."""

from __future__ import annotations

import ipaddress
import os
import socket
import urllib.request
import warnings
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
    warnings.warn(
        "Using unpinned HuggingFace revision 'main' — model weights may change "
        "without notice. Set DISTLLM_MODEL_REVISION to a specific commit SHA "
        "for reproducible, secure deployments.",
        UserWarning,
        stacklevel=2,
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
    """Open a URL safely, with DNS rebinding protection.

    Resolves the hostname to an IP once, validates the IP against private
    ranges, then connects directly to the IP (bypassing DNS on the actual
    TCP connection). This prevents DNS rebinding attacks where an attacker
    changes DNS between validation and connection.

    Args:
        request_or_url: URL string or ``urllib.request.Request``.
        timeout: Connection timeout in seconds.
        allow_private_hosts: If True, skip the private IP check.

    Returns:
        HTTP response object.

    Raises:
        ValueError: If the URL or resolved IP fails validation.
        urllib.error.URLError: If the connection fails.
    """
    url = (
        request_or_url.full_url
        if isinstance(request_or_url, urllib.request.Request)
        else request_or_url
    )

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r}")
    if not parsed.hostname:
        raise ValueError("URL must include a hostname")

    # Resolve hostname to IP addresses once
    addresses = socket.getaddrinfo(
        parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80),
        type=socket.SOCK_STREAM,
    )

    if not allow_private_hosts:
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

    # Pick the first resolved address
    family, socktype, proto, canonname, sockaddr = addresses[0]
    resolved_ip = sockaddr[0]

    # Reconstruct URL with resolved IP, preserving the original Host header
    resolved_url = parsed._replace(netloc=f"{resolved_ip}:{sockaddr[1]}" if sockaddr[1] else resolved_ip).geturl()
    host_header = parsed.hostname
    if parsed.port and parsed.port != (443 if parsed.scheme == "https" else 80):
        host_header = f"{parsed.hostname}:{parsed.port}"

    req = urllib.request.Request(
        resolved_url,
        data=request_or_url.data if isinstance(request_or_url, urllib.request.Request) else None,
        headers=dict(request_or_url.headers) if isinstance(request_or_url, urllib.request.Request) else {},
        origin_req_host=parsed.hostname,
    )
    req.add_unredirected_header("Host", host_header)

    return urllib.request.urlopen(req, timeout=timeout)  # nosec B310  # noqa: S310
