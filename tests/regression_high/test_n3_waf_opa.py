"""Regression tests for N3: WAF edge policy + OPA/Rego authz engine (Cat5).

Covers both deliverables via a minimal Starlette ASGI app + TestClient:

WAF (src/distllm/api/waf.py)
  (1) blocks an oversized request body (413);
  (2) blocks SQLi / XSS / path-traversal patterns in body + query (403);
  (3) blocks disallowed content-type and disallowed headers (415 / 403);
  (4) allows a clean request to pass through (200).

OPA authz (src/distllm/api/authz/opa.py)
  (5) returns allow for a permitted action and deny for a forbidden one
      using the pure-Python fallback (the `opa` binary is absent here);
  (6) load_policy works and is reflected in authorize() decisions.
"""

from __future__ import annotations

import json

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient


# ── Minimal ASGI app under test ────────────────────────────────────────────
async def _echo(request):
    try:
        body = await request.body()
    except Exception:
        body = b""
    return JSONResponse({"ok": True, "bytes": len(body)})


def _make_app(config=None):
    app = Starlette(
        routes=[
            Route("/echo", _echo, methods=["GET", "POST", "PUT", "PATCH", "DELETE"]),
        ]
    )
    # Wire WAF directly (not via add_waf_middleware to keep the test isolated
    # from the full server import graph).
    from distllm.api.waf import WAFMiddleware

    app.add_middleware(WAFMiddleware, config=config)
    return app


@pytest.fixture
def clean_client():
    return TestClient(_make_app())


@pytest.fixture
def strict_config():
    from distllm.api.waf import WAFConfig

    return WAFConfig(
        max_body_size=1024,
        content_type_allowlist=["application/json"],
        header_allowlist=["content-type", "content-length", "host", "x-custom"],
        header_denylist=["x-malicious"],
    )


@pytest.fixture
def strict_client(strict_config):
    return TestClient(_make_app(strict_config))


# ── (1) oversized body ─────────────────────────────────────────────────────
def test_waf_blocks_oversized_body(strict_client):
    big = "x" * 2048  # > 1024 limit
    resp = strict_client.post(
        "/echo",
        data=big,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 413
    assert resp.json()["error"]["type"] == "waf_rejected"
    assert resp.json()["error"]["code"] == "body_too_large"


# ── (2) SQLi / XSS / path-traversal patterns ───────────────────────────────
@pytest.mark.parametrize(
    "payload,reason",
    [
        ('{"q": "1 OR 1=1"}', "sql_injection"),
        ('{"q": "union select * from users"}', "sql_injection"),
        ('{"q": "<script>alert(1)</script>"}', "xss_script_tag"),
        ('{"q": "../../etc/passwd"}', "path_traversal"),
        ('{"q": "%2e%2e%2fetc%2fpasswd"}', "path_traversal"),
        ('{"q": "<img src=javascript:alert(1)>"}', "xss_img"),
    ],
)
def test_waf_blocks_injection_patterns_in_body(strict_client, payload, reason):
    resp = strict_client.post(
        "/echo",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["error"]["code"] == reason


def test_waf_blocks_pattern_in_query_string(clean_client):
    resp = clean_client.get("/echo?q=1%20OR%201%3D1")
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "sql_injection"


# ── (3) disallowed content-type / headers ──────────────────────────────────
def test_waf_blocks_disallowed_content_type(strict_client):
    resp = strict_client.post(
        "/echo",
        data='{"q": "hi"}',
        headers={"Content-Type": "text/plain"},
    )
    assert resp.status_code == 415
    assert resp.json()["error"]["code"] == "content_type_not_allowed"


def test_waf_blocks_disallowed_header(clean_client, strict_client):
    # header_denylist is set in strict_config -> x-malicious must be blocked.
    resp = strict_client.post(
        "/echo",
        data='{"q": "hi"}',
        headers={"Content-Type": "application/json", "X-Malicious": "1"},
    )
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "blocked_header"


def test_waf_blocks_header_not_in_allowlist(strict_client):
    # 'x-other' is not in the allowlist -> blocked.
    resp = strict_client.post(
        "/echo",
        data='{"q": "hi"}',
        headers={"Content-Type": "application/json", "X-Other": "1"},
    )
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "header_not_allowed"


# ── (4) clean request passes ───────────────────────────────────────────────
def test_waf_allows_clean_request(strict_client):
    resp = strict_client.post(
        "/echo",
        data='{"q": "hello world"}',
        headers={"Content-Type": "application/json", "X-Custom": "ok"},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


# ── (5) OPA authz allow / deny (pure-python fallback) ──────────────────────
def test_opa_authz_allow_and_deny():
    from distllm.api.authz import authorize

    # Allowed: admin may do anything.
    allow = authorize({"subject": {"role": "admin"}, "action": "delete", "resource": "svc:x"})
    assert allow.allow is True
    assert allow.source == "python"

    # Allowed: user may read.
    read = authorize({"subject": {"role": "user"}, "action": "read", "resource": "doc:1"})
    assert read.allow is True

    # Denied: user may NOT write.
    deny = authorize({"subject": {"role": "user"}, "action": "write", "resource": "doc:1"})
    assert deny.allow is False
    assert deny.source == "python"

    # Denied: unknown role -> default deny.
    unknown = authorize({"subject": {"role": "guest"}, "action": "read", "resource": "doc:1"})
    assert unknown.allow is False


def test_opa_authz_explicit_grant():
    from distllm.api.authz import authorize, load_policy

    load_policy()  # reset to default
    # No explicit grant for user:write on doc:2 initially.
    before = authorize({"subject": {"role": "user"}, "action": "write", "resource": "doc:2"})
    assert before.allow is False

    load_policy(
        None
    )  # re-load default; explicit-grant behaviour proven in (6) via file.


# ── (6) policy load works ──────────────────────────────────────────────────
def test_opa_load_policy_from_file(tmp_path):
    from distllm.api.authz import authorize, load_policy

    policy_file = tmp_path / "policy.json"
    policy_file.write_text(
        json.dumps(
            {
                "default": "deny",
                "roles": {
                    "admin": {"allow_all": True},
                    "service": {"prefix": "svc:"},
                    "user": {"actions": ["read"]},
                },
                # Per-subject grant: a "user" may write doc:2 even though the
                # role default only permits "read".
                "role_grants": {
                    "user": [{"resource": "doc:2", "action": "write"}]
                },
                "grants": [],
            }
        )
    )
    loaded = load_policy(str(policy_file))
    assert loaded["role_grants"]["user"][0]["resource"] == "doc:2"

    # Default deny for a resource/action with no grant.
    denied = authorize({"subject": {"role": "user"}, "action": "write", "resource": "doc:9"})
    assert denied.allow is False

    # Now the per-subject grant permits user:write on doc:2.
    granted = authorize({"subject": {"role": "user"}, "action": "write", "resource": "doc:2"})
    assert granted.allow is True
    assert granted.source == "python"


def test_opa_backend_reporting():
    from distllm.api.authz import OPA_AVAILABLE, OPAClient

    client = OPAClient()
    # On this host `opa` is absent -> fallback backend.
    assert client.backend_name in ("opa", "python")
    assert client.authorize({"subject": {"role": "admin"}, "action": "read", "resource": "x"}).allow is True
    if not OPA_AVAILABLE:
        assert client.backend_name == "python"
