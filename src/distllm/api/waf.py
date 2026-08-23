"""Edge Web Application Firewall (WAF) middleware for the distributed-LLM API.

``WAFMiddleware`` is a raw ASGI middleware (no dependency on ``BaseHTTPMiddleware``
so it can inspect and rewrite the request *body* before any downstream handler —
including the prompt-injection detector and the routers — ever sees it).

It enforces, at the edge:

1. Maximum request-body size (``max_body_size``).
2. Content-Type allowlist (``content_type_allowlist``).
3. Header allowlist / denylist (``header_allowlist`` / ``header_denylist``).
4. A deny-pattern scanner (SQLi / XSS / path-traversal) over the request
   body **and** the raw query string.

Clean requests pass straight through (the buffered body is re-fed to the
downstream app so nothing downstream needs to know the WAF touched it).
Violations return ``403 Forbidden`` with a JSON body naming the reason.

The policy is fully tunable via :class:`WAFConfig`.  Wire it up with
:func:`add_waf_middleware`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from starlette.responses import JSONResponse

# ── Default deny-pattern ruleset ──────────────────────────────────────────
# Each entry is ``(regex, reason)``.  Patterns are matched case-insensitively
# against the decoded request body and the raw query string.
_DEFAULT_DENY_PATTERNS: list[tuple[str, str]] = [
    # ── SQL injection ──
    (r"(?i)(?:^|\W)(?:select|union|insert|update|delete|drop|alter|create|truncate|exec|execute)\b[\s\S]{0,40}?\b(?:from|into|table|database)\b", "sql_injection"),
    (r"(?i)(?:--|#)\s*(?:$|\w)", "sql_comment"),
    (r"(?i)\b(or|and)\b\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+", "sql_injection"),
    (r"(?i);\s*(?:drop|delete|update|insert|select|truncate)\b", "sql_injection"),
    (r"(?i)\bunion\b[\s\S]{0,40}?\bselect\b", "sql_injection"),
    (r"(?i)\b(sleep|benchmark|pg_sleep|waitfor)\s*\(", "sql_timeblind"),
    (r"(?i)/\*\s*!?\d*", "sql_inline_comment"),
    # ── XSS ──
    (r"(?i)<\s*script\b", "xss_script_tag"),
    (r"(?i)<\s*iframe\b", "xss_iframe_tag"),
    (r"(?i)<\s*img\b[^>]*\bsrc\s*=\s*['\"]?\s*(?:javascript|x):", "xss_img"),
    (r"(?i)\bjavascript\s*:", "xss_javascript_uri"),
    (r"(?i)\bon(?:error|load|click|mouseover|focus|blur|submit)\s*=", "xss_event_handler"),
    (r"(?i)<\s*(?:svg|object|embed|math|link|meta)\b", "xss_dangerous_tag"),
    (r"(?i)\b(?:alert|confirm|prompt)\s*\(", "xss_popup"),
    (r"(?i)<\s*style\b[^>]*expression\s*\(", "xss_css_expression"),
    # ── Path traversal / LFI ──
    (r"(?:\.\./|\.\.\\)", "path_traversal"),
    (r"(?:%2e%2e%2f|%2e%2e/|%2e%2e%5c|%2e%2e\\)", "path_traversal"),
    (r"(?i)(?:etc/passwd|windows/win.ini|boot.ini|/proc/self/environ)", "lfi_sensitive_path"),
    (r"(?i)\b(?:file|php)://", "scheme_smuggling"),
]


@dataclass
class WAFConfig:
    """Tunable WAF policy.

    Attributes:
        max_body_size: Max request-body bytes. ``0``/negative disables the check.
        content_type_allowlist: Lower-cased content types permitted (params in
            the request are ignored). ``None``/empty disables the check.
        header_allowlist: If non-empty, *only* these lower-cased header names are
            permitted. ``None`` disables the check.
        header_denylist: Lower-cased header names that must never appear.
        deny_patterns: ``(regex, reason)`` list; ``None`` uses the built-in set.
        deny_methods: HTTP methods whose body is scanned.
        skip_paths: Paths exempt from all WAF checks (e.g. health probes).
        body_scan_limit: Max bytes of the body scanned (decoded); ``0`` = no cap.
    """

    max_body_size: int = 1 * 1024 * 1024  # 1 MiB
    content_type_allowlist: list[str] | None = None
    header_allowlist: list[str] | None = None
    header_denylist: list[str] | None = None
    deny_patterns: list[tuple[str, str]] | None = None
    deny_methods: set[str] = field(default_factory=lambda: {"POST", "PUT", "PATCH", "DELETE"})
    skip_paths: set[str] = field(
        default_factory=lambda: {"/health", "/ready", "/live", "/healthz", "/readyz", "/metrics"}
    )
    body_scan_limit: int = 0
    # Header names that are always permitted even when an allowlist is set
    # (framing + common well-behaved client headers; never attacker-controlled
    # in a way that bypasses the policy).
    _ALWAYS_ALLOWED_HEADERS: tuple[str, ...] = (
        "content-length",
        "host",
        "content-type",
        "connection",
        "accept",
        "accept-encoding",
        "accept-language",
        "user-agent",
        "cache-control",
        "pragma",
    )

    def __post_init__(self) -> None:
        if self.content_type_allowlist is not None:
            self.content_type_allowlist = [c.lower().strip() for c in self.content_type_allowlist]
        if self.header_allowlist is not None:
            self.header_allowlist = [h.lower() for h in self.header_allowlist]
        if self.header_denylist is not None:
            self.header_denylist = [h.lower() for h in self.header_denylist]
        self.deny_methods = {m.upper() for m in self.deny_methods}
        self.skip_paths = {p for p in self.skip_paths}
        if self.deny_patterns is None:
            self.deny_patterns = list(_DEFAULT_DENY_PATTERNS)
        # Pre-compile deny patterns once.
        self._compiled: list[tuple[re.Pattern[str], str]] = [
            (re.compile(p), reason) for p, reason in self.deny_patterns
        ]


class WAFViolation(Exception):
    """Raised internally when a request violates the WAF policy."""

    def __init__(self, status_code: int, reason: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.reason = reason
        self.message = message


def _violation_response(status_code: int, reason: str, message: str) -> JSONResponse:
    """Build a standardized 403/413 error body (OpenAI-compatible shape)."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "waf_rejected",
                "param": None,
                "code": reason,
            }
        },
    )


def _scan(text: str, compiled: list[tuple[re.Pattern[str], str]]) -> str | None:
    """Return the reason of the first matching deny pattern, else ``None``."""
    if not text:
        return None
    for pattern, reason in compiled:
        if pattern.search(text):
            return reason
    return None


class WAFMiddleware:
    """Raw ASGI middleware enforcing the edge WAF policy.

    Reads and buffers the request body (for scanned methods), applies all
    checks, and — on success — re-feeds the buffered body downstream so the
    rest of the stack is transparent to the inspection.
    """

    def __init__(self, app: Any, config: WAFConfig | None = None) -> None:
        self.app = app
        self.config = config or WAFConfig()

    async def __call__(self, scope: dict, receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "GET").upper()

        # Health/metrics probes are fully exempt from all checks.
        if path in self.config.skip_paths:
            await self.app(scope, receive, send)
            return

        # Header policy applies to EVERY request (lightweight, no body read).
        try:
            self._check_headers(scope)
        except WAFViolation as exc:
            await _violation_response(exc.status_code, exc.reason, exc.message)(scope, receive, send)
            return

        # Query-string pattern scan applies to EVERY method (SQLi/XSS in the
        # URL is method-independent). No body read required.
        try:
            self._check_query_patterns(scope)
        except WAFViolation as exc:
            await _violation_response(exc.status_code, exc.reason, exc.message)(scope, receive, send)
            return

        # Body-bearing methods additionally get content-type, body-size, and
        # body pattern checks.
        if method not in self.config.deny_methods:
            await self.app(scope, receive, send)
            return

        try:
            body = await self._read_body(receive)
            self._check_content_type(scope)
            self._check_body_size(body)
            self._check_body_patterns(body)
        except WAFViolation as exc:
            await _violation_response(exc.status_code, exc.reason, exc.message)(scope, receive, send)
            return

        # Re-feed the buffered body to the downstream app exactly once.
        await self.app(scope, self._make_receive(body), send)

    # ── helpers ──

    async def _read_body(self, receive) -> bytes:
        """Read the full request body from the ASGI receive channel."""
        body = b""
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                break
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break
        return body

    def _make_receive(self, body: bytes):
        """Return a ``receive`` coroutine that replays the buffered *body* once."""

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        return receive

    def _get_headers(self, scope: dict) -> dict[str, str]:
        """Return lower-cased header-name -> joined value map from the scope."""
        out: dict[str, str] = {}
        for raw_name, raw_value in scope.get("headers", []):
            name = raw_name.decode("latin-1").lower()
            value = raw_value.decode("latin-1", errors="replace")
            out[name] = value
        return out

    def _check_headers(self, scope: dict) -> None:
        headers = self._get_headers(scope)
        cfg = self.config

        if cfg.header_denylist:
            for banned in cfg.header_denylist:
                if banned in headers:
                    raise WAFViolation(
                        403,
                        "blocked_header",
                        f"Request rejected by WAF: disallowed header '{banned}'.",
                    )

        if cfg.header_allowlist is not None:
            for name in headers:
                # Framing + well-behaved client headers are always permitted.
                if name in self.config._ALWAYS_ALLOWED_HEADERS:
                    continue
                if name not in cfg.header_allowlist:
                    raise WAFViolation(
                        403,
                        "header_not_allowed",
                        f"Request rejected by WAF: header '{name}' is not in the allowlist.",
                    )

    def _check_content_type(self, scope: dict) -> None:
        cfg = self.config
        if not cfg.content_type_allowlist:
            return
        headers = self._get_headers(scope)
        ctype = headers.get("content-type", "")
        # Strip parameters (e.g. '; charset=utf-8').
        ctype_base = ctype.split(";", 1)[0].strip().lower()
        if not ctype_base:
            # Missing content-type still fails the allowlist unless '*' allowed.
            if "*" not in cfg.content_type_allowlist:
                raise WAFViolation(
                    415,
                    "content_type_not_allowed",
                    "Request rejected by WAF: missing Content-Type is not permitted.",
                )
            return
        if ctype_base not in cfg.content_type_allowlist and "*" not in cfg.content_type_allowlist:
            raise WAFViolation(
                415,
                "content_type_not_allowed",
                f"Request rejected by WAF: Content-Type '{ctype_base}' is not permitted.",
            )

    def _check_body_size(self, body: bytes) -> None:
        cfg = self.config
        if cfg.max_body_size and len(body) > cfg.max_body_size:
            raise WAFViolation(
                413,
                "body_too_large",
                f"Request rejected by WAF: body size {len(body)} exceeds limit "
                f"{cfg.max_body_size}.",
            )

    def _check_query_patterns(self, scope: dict) -> None:
        cfg = self.config
        if not cfg._compiled:
            return
        raw = scope.get("query_string", b"").decode("latin-1", errors="replace")
        # ASGI delivers the query string percent-encoded; decode it so patterns
        # match the human-visible value (e.g. "1 OR 1=1" not "1%20OR%201%3D1").
        from urllib.parse import unquote

        query = unquote(raw)
        reason = _scan(query, cfg._compiled)
        if reason:
            raise WAFViolation(
                403,
                reason,
                f"Request rejected by WAF: deny-pattern '{reason}' matched in query string.",
            )

    def _check_body_patterns(self, body: bytes) -> None:
        cfg = self.config
        if not cfg._compiled:
            return
        text = body.decode("utf-8", errors="ignore")
        if cfg.body_scan_limit and len(text) > cfg.body_scan_limit:
            text = text[: cfg.body_scan_limit]
        reason = _scan(text, cfg._compiled)
        if reason:
            raise WAFViolation(
                403,
                reason,
                f"Request rejected by WAF: deny-pattern '{reason}' matched in request body.",
            )


def add_waf_middleware(app: Any, config: WAFConfig | None = None) -> None:
    """Attach the WAF middleware to a Starlette/FastAPI ``app``.

    Args:
        app: The ASGI app (FastAPI or Starlette instance).
        config: Optional :class:`WAFConfig`; defaults to the built-in policy.
    """
    app.add_middleware(WAFMiddleware, config=config)
