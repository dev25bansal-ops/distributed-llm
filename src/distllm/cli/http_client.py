"""Shared HTTP client construction for CLI commands.

Every CLI module used to build its own ``httpx.Client`` with no auth header, so
any command that talks to the API returned 401 against a cluster the user had
just started with ``distllm cluster start``. The server mints an API key on
first boot and persists it (``core.api_key_store`` writes it to
``$DISTLLM_KEY_FILE`` or ``.distllm/api_key``); this module is the single place
that reads it back.

Use :func:`client_for` instead of constructing ``httpx.Client`` directly, and
:func:`auth_headers` for the rare call site that needs a bare ``httpx.get``.
"""

from __future__ import annotations

import os

import httpx

DEFAULT_KEY_FILE = os.path.join(".distllm", "api_key")


def resolve_api_key() -> str | None:
    """Return the API key the local server expects, if one is available.

    Precedence: ``DISTLLM_API_KEY``, then ``API_KEY`` (the server's own env
    var), then the persisted key file. Environment wins so an explicit key
    always overrides a stale file.
    """
    for name in ("DISTLLM_API_KEY", "API_KEY"):
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()

    path = os.environ.get("DISTLLM_KEY_FILE") or DEFAULT_KEY_FILE
    try:
        with open(path, encoding="utf-8") as fh:
            key = fh.read().strip()
    except OSError:
        return None
    return key or None


def auth_headers() -> dict[str, str]:
    """Authorization header for the local server, or an empty dict."""
    key = resolve_api_key()
    return {"Authorization": f"Bearer {key}"} if key else {}


def client_for(
    host: str,
    port: int,
    *,
    timeout: float = 30.0,
    base_url: str | None = None,
) -> httpx.Client:
    """An ``httpx.Client`` pointed at the API, carrying the local key."""
    return httpx.Client(
        base_url=base_url or f"http://{host}:{port}",
        timeout=timeout,
        headers=auth_headers(),
    )


def format_http_error(exc: httpx.HTTPStatusError, *, max_len: int = 200) -> str:
    """Render an HTTP error as one readable line.

    Commands used to print ``{status} {response.text}``, which dumped a whole
    HTML error page into the terminal when the address was not a distllm
    server (a plain 404 from any other service listening on that port). This
    keeps the JSON error message the API returns, collapses whitespace, and
    replaces HTML with an explanation.
    """
    response = exc.response
    status = response.status_code
    body = ""

    try:
        payload = response.json()
    except (ValueError, UnicodeDecodeError):
        payload = None

    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            body = str(err.get("message") or err.get("code") or "")
        elif err:
            body = str(err)
        body = body or str(payload.get("detail") or payload.get("message") or "")

    if not body:
        text = " ".join((response.text or "").split())
        if text.startswith("<"):
            body = (
                "the server returned an HTML page, not a distllm response "
                "(is something else listening on this port?)"
            )
        else:
            body = text[:max_len]

    return f"HTTP {status}: {body}" if body else f"HTTP {status}"
