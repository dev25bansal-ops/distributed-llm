"""Regression: cost middleware returns 400, not 500, on bad bodies.

Wave 4 fuzzer found that the inference endpoints (chat completions,
completions, embeddings) returned HTTP 500 when the request body was
malformed JSON or a non-object JSON value. The cost middleware was the
layer that called ``data.get(...)`` on the parsed body without first
checking the type, and the downstream handler crashed when it tried to
interpret the bad payload.

This test pins the contract: any body that is not a JSON object
(``[]``, ``42``, ``"a string"``, ``null``, etc.) must be rejected with
``400 Bad Request`` *before* reaching the downstream handler. The
existing fuzzer-pinned cases are covered, plus an additional case for
a JSON object whose ``messages`` field is the wrong type.
"""

from __future__ import annotations

import json
from typing import Any

import pytest


# Bodies the fuzzer pinned as causing 500.
FUZZER_BAD_BODIES: list[tuple[str, Any]] = [
    ("empty_list", []),
    ("int_42", 42),
    ("string_body", "a string"),
    ("null_body", None),
]
# Plus a structurally-broken-but-still-an-object case.
EXTRA_BAD_BODIES: list[tuple[str, Any]] = [
    ("messages_not_list", {"messages": "not a list"}),
    ("messages_int", {"messages": 7}),
    ("messages_null", {"messages": None}),
]


def _is_400_response(resp: Any) -> bool:
    """Return True if the response looks like the cost-middleware 400."""
    # The cost middleware returns a JSONResponse with status_code == 400
    # and content.error.type == "invalid_request_body".
    status = getattr(resp, "status_code", None)
    if status != 400:
        return False
    body = getattr(resp, "body", None)
    if body is None:
        return True  # JSONResponse constructed directly — accept
    try:
        decoded = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    err = decoded.get("error") if isinstance(decoded, dict) else None
    if not isinstance(err, dict):
        return False
    return err.get("type") == "invalid_request_body"


def test_parse_request_body_rejects_non_object_payloads() -> None:
    """Each fuzzer-pinned body must decode to None (treated as invalid)."""
    from distllm.api.cost_middleware import _parse_request_body

    for label, body_value in FUZZER_BAD_BODIES:
        encoded = json.dumps(body_value).encode("utf-8")
        result = _parse_request_body(encoded)
        assert result is None, (
            f"fuzzer body {label}={body_value!r} must be rejected, got {result!r}"
        )


def test_parse_request_body_accepts_well_formed_object() -> None:
    """A well-formed JSON object must pass through unchanged."""
    from distllm.api.cost_middleware import _parse_request_body

    obj = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
    encoded = json.dumps(obj).encode("utf-8")
    result = _parse_request_body(encoded)
    assert result == obj


def test_parse_request_body_rejects_empty_body() -> None:
    """Empty body → None (caller treats as no body, no estimation)."""
    from distllm.api.cost_middleware import _parse_request_body
    assert _parse_request_body(b"") is None


def test_parse_request_body_rejects_malformed_json() -> None:
    """Malformed JSON → None (downstream validation handles 400 message)."""
    from distllm.api.cost_middleware import _parse_request_body
    assert _parse_request_body(b"{not json}") is None


@pytest.mark.parametrize("label,bad_body", FUZZER_BAD_BODIES)
def test_dispatch_returns_400_on_fuzzer_pinned_body(label: str, bad_body: Any) -> None:
    """End-to-end: cost middleware.dispatch returns 400 for each fuzzer body.

    Constructs the ASGI scope manually so we exercise the middleware in
    isolation against the fuzzer-pinned payloads — no FastAPI app, no
    downstream handler, no global state.
    """
    import pytest
    import asyncio
    from starlette.requests import Request as StarletteRequest

    from distllm.api.cost_middleware import CostTrackingMiddleware

    encoded = json.dumps(bad_body).encode("utf-8")
    scope: dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "headers": [(b"content-type", b"application/json")],
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
        "http_version": "1.1",
        "root_path": "",
        "app": None,
    }

    async def receive():
        return {"type": "http.request", "body": encoded, "more_body": False}

    request = StarletteRequest(scope, receive=receive)

    async def call_next(_req):
        # If the cost middleware does its job, call_next is never reached
        # for these bodies. Raise loud to make that obvious.
        raise AssertionError(
            "call_next should not be reached for invalid body "
            f"({label}={bad_body!r})"
        )

    middleware = CostTrackingMiddleware(app=None)

    @pytest.mark.asyncio
    async def _run() -> Any:
        return await middleware.dispatch(request, call_next)

    response = asyncio.new_event_loop().run_until_complete(_run())
    assert _is_400_response(response), (
        f"cost middleware must return 400 invalid_request_body for "
        f"{label}={bad_body!r}, got status={getattr(response, 'status_code', '?')}, "
        f"body={getattr(response, 'body', b'')!r}"
    )


@pytest.mark.parametrize("label,bad_body", EXTRA_BAD_BODIES)
def test_dispatch_passes_through_extra_bad_bodies(label: str, bad_body: Any) -> None:
    """Bodies that *are* JSON objects but have wrong-typed fields pass through.

    These are not the cost middleware's responsibility — the downstream
    handler validates field types. The cost middleware should NOT 400 on
    a malformed-but-object body. Pinning this prevents the cost middleware
    from over-eagerly rejecting payloads the inference endpoint actually
    understands.
    """
    import pytest
    import asyncio
    from starlette.requests import Request as StarletteRequest
    from starlette.responses import JSONResponse

    from distllm.api.cost_middleware import CostTrackingMiddleware

    encoded = json.dumps(bad_body).encode("utf-8")
    scope: dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "headers": [(b"content-type", b"application/json")],
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
        "http_version": "1.1",
        "root_path": "",
        "app": None,
    }

    async def receive():
        return {"type": "http.request", "body": encoded, "more_body": False}

    request = StarletteRequest(scope, receive=receive)

    downstream_called = {"v": False}

    async def call_next(_req):
        downstream_called["v"] = True
        return JSONResponse({"ok": True}, status_code=200)

    middleware = CostTrackingMiddleware(app=None)

    @pytest.mark.asyncio
    async def _run() -> Any:
        return await middleware.dispatch(request, call_next)

    response = asyncio.new_event_loop().run_until_complete(_run())
    assert downstream_called["v"], (
        f"call_next must run for object-typed body {label}={bad_body!r}"
    )
    assert getattr(response, "status_code", None) != 400, (
        f"cost middleware must NOT 400 an object body just because a field is wrong-typed "
        f"({label}={bad_body!r})"
    )


def test_dispatch_ignores_non_inference_paths() -> None:
    """Paths outside /v1/{chat,completions,embeddings} skip cost-tracking entirely."""
    import pytest
    import asyncio
    from starlette.requests import Request as StarletteRequest
    from starlette.responses import JSONResponse

    from distllm.api.cost_middleware import CostTrackingMiddleware

    scope: dict[str, Any] = {
        "type": "http",
        "method": "GET",
        "path": "/health",
        "raw_path": b"/health",
        "headers": [],
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
        "http_version": "1.1",
        "root_path": "",
        "app": None,
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    request = StarletteRequest(scope, receive=receive)

    downstream_called = {"v": False}

    async def call_next(_req):
        downstream_called["v"] = True
        return JSONResponse({"ok": True}, status_code=200)

    middleware = CostTrackingMiddleware(app=None)

    @pytest.mark.asyncio
    async def _run() -> Any:
        return await middleware.dispatch(request, call_next)

    response = asyncio.new_event_loop().run_until_complete(_run())
    assert downstream_called["v"], "/health should bypass cost middleware"
    assert getattr(response, "status_code", None) == 200