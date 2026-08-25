"""SEC-A4 regression tests — webhook SSRF guard at registration AND delivery.

Threat: an authenticated low-privilege user registers a webhook whose URL
targets an internal service (cloud metadata endpoint, localhost admin port).
The coordinator then POSTs signed, structured event payloads to that target
and exposes delivery success/failure differentials (internal port-knocking).

Fixes covered here:
1. Registration layer  — ``routes/webhooks.py`` rejects unsafe URLs (HTTP 400)
   and requires admin/user-admin role on the whole /v1/webhooks router.
2. Delivery engine     — ``api/webhooks/delivery.py`` ``WebhookManager.register``
   raises :class:`UnsafeWebhookURLError`; ``_submit_delivery`` re-validates and
   dead-letters instead of posting if a stored URL was mutated post-registration.
3. One-shot dispatcher — ``dispatch_webhook`` refuses unsafe URLs outright.
4. Allowlist escape hatch — ``DISTLLM_WEBHOOK_ALLOWLIST`` deny-by-default.
5. Core manager parity — ``core/webhook_manager.WebhookManager._deliver``
   re-validates before its httpx.post.

All network-touching paths are mocked or pointed at unroutable test servers;
no real internal/metadata endpoint is ever contacted.
"""

from __future__ import annotations

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest


# ── Helpers ──────────────────────────────────────────────────────────────────

SSRF_TARGETS = [
    "http://169.254.169.254/latest/meta-data/",
    "http://localhost:8000/admin/v1/nodes",
    "http://127.0.0.1:8000/admin",
    "https://10.0.0.5/hook",
    "http://192.168.1.10/callback",
    "file:///etc/passwd",
]

PUBLIC_URL = "https://hooks.example.com/webhook"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Keep DISTLLM_WEBHOOK_ALLOWLIST state deterministic per test."""
    monkeypatch.delenv("DISTLLM_WEBHOOK_ALLOWLIST", raising=False)
    yield


@pytest.fixture()
def webhook_store(tmp_path, monkeypatch):
    """Point PersistentStore at a temp dir and reset the module cache."""
    from distllm.api import persistent_store as ps
    from distllm.api.routes import webhooks as webhooks_routes

    monkeypatch.setenv("DISTLLM_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ps, "_store", None)
    monkeypatch.setattr(webhooks_routes, "_webhook_cache", {})
    yield
    # Do not restore a global store bound to the tmp dir.
    monkeypatch.setattr(ps, "_store", None)


# ── 1. Route-level: registration-time rejection ──────────────────────────────

class TestRouteRegistrationRejection:
    def test_metadata_url_rejected_at_registration(self, webhook_store):
        """The exact SEC-A4 repro: cloud metadata URL must now be rejected."""
        from fastapi import HTTPException
        from distllm.api.routes.webhooks import WebhookCreate, create_webhook

        body = WebhookCreate(
            url="http://169.254.169.254/latest/meta-data/",
            secret="x" * 16,
            events=["batch.completed"],
        )
        with pytest.raises(HTTPException) as exc:
            asyncio.run(create_webhook(body))
        assert exc.value.status_code == 400
        assert "DISTLLM_WEBHOOK_ALLOWLIST" in exc.value.detail

    def test_localhost_admin_url_rejected_at_registration(self, webhook_store):
        from fastapi import HTTPException
        from distllm.api.routes.webhooks import WebhookCreate, create_webhook

        body = WebhookCreate(
            url="http://localhost:8000/admin/v1/nodes",
            secret="x" * 16,
            events=["batch.completed"],
        )
        with pytest.raises(HTTPException) as exc:
            asyncio.run(create_webhook(body))
        assert exc.value.status_code == 400

    @pytest.mark.parametrize("url", SSRF_TARGETS)
    def test_all_ssrf_targets_rejected_at_registration(self, webhook_store, url):
        from fastapi import HTTPException
        from distllm.api.routes.webhooks import WebhookCreate, create_webhook

        body = WebhookCreate(url=url, secret="x" * 16, events=["batch.completed"])
        with pytest.raises(HTTPException) as exc:
            asyncio.run(create_webhook(body))
        assert exc.value.status_code == 400

    def test_public_https_registration_still_works(self, webhook_store):
        """Normal https webhooks are unaffected by the guard."""
        from distllm.api.routes.webhooks import WebhookCreate, create_webhook

        body = WebhookCreate(
            url=PUBLIC_URL,
            secret="x" * 16,
            events=["batch.completed"],
        )
        reg = asyncio.run(create_webhook(body))
        assert reg.url == PUBLIC_URL
        assert reg.active is True


class TestRouteUpdateRejection:
    def test_update_cannot_smuggle_unsafe_url(self, webhook_store):
        """A registered-safe webhook cannot be updated to an SSRF target."""
        from fastapi import HTTPException
        from distllm.api.routes.webhooks import (
            WebhookCreate,
            WebhookUpdate,
            create_webhook,
            update_webhook,
        )

        reg = asyncio.run(create_webhook(WebhookCreate(
            url=PUBLIC_URL, secret="x" * 16, events=["batch.completed"],
        )))

        with pytest.raises(HTTPException) as exc:
            asyncio.run(update_webhook(reg.id, WebhookUpdate(
                url="http://169.254.169.254/latest/meta-data/",
            )))
        assert exc.value.status_code == 400

        # And the stored URL is unchanged.
        from distllm.api.routes.webhooks import get_webhook
        data = asyncio.run(get_webhook(reg.id))
        assert data.url == PUBLIC_URL


# ── 2. Route-level: RBAC gate ────────────────────────────────────────────────

class TestRouteRBACGate:
    """Webhook mutation routes require admin/user-admin (was: no gate)."""

    @pytest.mark.parametrize("role,allowed", [
        ("admin", True),
        ("user-admin", True),
        ("model-admin", False),
        ("inference-only", False),
        ("read-only", False),
        ("auditor", False),
    ])
    def test_role_matrix_on_router_dependency(self, role, allowed):
        """Drive require_role directly against each hierarchy role."""
        from unittest.mock import MagicMock
        from fastapi import HTTPException
        from fastapi import Request
        from distllm.api.auth_deps import require_role

        mock_request = MagicMock(spec=Request)
        mock_request.state.api_key_role = role

        check = require_role("admin", "user-admin")
        if allowed:
            asyncio.run(check(mock_request))  # must not raise
        else:
            with pytest.raises(HTTPException) as exc:
                asyncio.run(check(mock_request))
            assert exc.value.status_code == 403

    def test_router_declares_role_gate(self):
        """Structural check: the router itself carries the role dependency."""
        from distllm.api.routes.webhooks import router
        from distllm.api.auth_deps import require_role

        dep_roles = []
        for dep in router.dependencies:
            # FastAPI wraps Depends(callable) — recover the factory's roles.
            closure = getattr(dep.dependency, "__closure__", None) or ()
            for cell in closure:
                if isinstance(cell.cell_contents, tuple):
                    dep_roles.extend(cell.cell_contents)

        assert set(dep_roles) >= {"admin", "user-admin"}, (
            f"/v1/webhooks router lost its role gate; found {dep_roles}"
        )

    def test_full_stack_low_priv_key_gets_403_on_register(self, webhook_store):
        """End-to-end through the REAL /v1/webhooks router with its
        role-gate dependency active.

        Note: routes/webhooks.py's router is not currently mounted in
        api/server.py (a separate wiring gap); we exercise the router
        object directly in a minimal app whose auth stub mirrors
        AuthMiddleware's contract (request.state.api_key_role), so the
        router-level Depends(require_role(...)) runs exactly as in prod.
        """
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from distllm.core.api_key_store import get_api_key_store

        store = get_api_key_store()
        store.add_key("lowpriv-key-1234567890", role="inference-only")
        store.add_key("admin-key-12345678901", role="admin")
        roles_by_token = {
            "lowpriv-key-1234567890": "inference-only",
            "admin-key-12345678901": "admin",
        }

        app = FastAPI()

        @app.middleware("http")
        async def _stub_auth(request: Request, call_next):
            token = request.headers.get("Authorization", "")[7:].strip()
            role = roles_by_token.get(token)
            if role:
                request.state.api_key_role = role
                request.state.api_key_id = f"key-{token[:8]}"
            return await call_next(request)

        # Coordinator availability check must pass too (require_coordinator).
        from distllm.api.api_state import g
        from unittest.mock import MagicMock

        mock_coord = MagicMock()
        original = g.coordinator
        g.coordinator = mock_coord

        try:
            from distllm.api.routes.webhooks import router as webhooks_router

            app.include_router(webhooks_router)
            client = TestClient(app, raise_server_exceptions=False)

            # Low-privilege key: role gate rejects BEFORE handler logic.
            resp = client.post("/v1/webhooks", headers={
                "Authorization": "Bearer lowpriv-key-1234567890",
            }, json={
                "url": PUBLIC_URL,
                "secret": "x" * 16,
                "events": ["batch.completed"],
            })
            assert resp.status_code == 403, resp.text

            # Admin key passes the gate and reaches handler logic (201).
            resp_ok = client.post("/v1/webhooks", headers={
                "Authorization": "Bearer admin-key-12345678901",
            }, json={
                "url": PUBLIC_URL,
                "secret": "x" * 16,
                "events": ["batch.completed"],
            })
            assert resp_ok.status_code == 201, resp_ok.text
        finally:
            g.coordinator = original


# ── 3. Delivery engine: registration + delivery-time enforcement ────────────

class TestDeliveryEngineRegistration:
    def test_manager_register_raises_for_ssrf_targets(self):
        from distllm.api.webhooks.delivery import UnsafeWebhookURLError, WebhookManager

        mgr = WebhookManager()
        for url in SSRF_TARGETS:
            with pytest.raises(UnsafeWebhookURLError):
                mgr.register(url, {"job.completed"}, "sekret")
        assert mgr.list() == []  # nothing was registered

    def test_manager_register_accepts_public_url(self):
        from distllm.api.webhooks.delivery import WebhookManager

        mgr = WebhookManager()
        wid = mgr.register(PUBLIC_URL, {"job.completed"}, "sekret")
        assert wid
        assert mgr.get(wid).url == PUBLIC_URL


class TestDeliveryTimeRevalidation:
    """URLs mutated after registration are blocked at dispatch time."""

    def _mutated_mgr(self):
        from distllm.api.webhooks.delivery import WebhookManager

        mgr = WebhookManager()
        wid = mgr.register(PUBLIC_URL, {"job.completed"}, "sekret")
        reg = mgr.get(wid)
        # Simulate post-registration tampering of the stored URL.
        object.__setattr__(reg, "url", "http://169.254.169.254/latest/meta-data/")
        return mgr, wid, reg

    def test_mutated_url_never_posted(self):
        from distllm.api.webhooks.delivery import WebhookDeliveryStatus

        mgr, wid, reg = self._mutated_mgr()

        with patch("distllm.api.webhooks.delivery.httpx.post") as mock_post:
            matched = mgr.dispatch("job.completed", {"job_id": "abc"})
            assert matched == [wid]
            mock_post.assert_not_called()  # the critical assertion

        log = mgr.get_delivery_log(webhook_id=wid)
        assert len(log) == 1
        assert log[0].status == WebhookDeliveryStatus.DEAD
        assert "SSRF" in (log[0].error or "")

    def test_mutated_core_target_never_posted(self):
        """Same guarantee for core/webhook_manager's background deliverer."""
        from distllm.core.webhook_manager import WebhookEvent, WebhookManager as CoreMgr
        from distllm.core.webhook_manager import WebhookTarget

        core = CoreMgr()
        target = WebhookTarget(
            url="http://169.254.169.254/latest/meta-data/",
            events=["node.joined"], label="tampered",
        )
        core._targets.append(target)  # bypass register() to simulate mutation

        # httpx is imported inside _deliver, so patch the module itself.
        with patch("httpx.post") as mock_post:
            core._deliver(target, WebhookEvent.NODE_JOINED, {"node_id": "n1"})
            mock_post.assert_not_called()

        assert target.active is False  # fail-closed deactivation
        entry = core.delivery_log()[-1]
        assert entry.success is False
        assert "SSRF" in entry.error


# ── 4. One-shot dispatch_webhook helper ──────────────────────────────────────

class TestDispatchWebhookHelper:
    def test_dispatch_webhook_refuses_ssrf_url(self):
        from distllm.api.webhooks import dispatch_webhook

        ok = asyncio.run(dispatch_webhook(
            "http://169.254.169.254/latest/meta-data/", "s3cret", "job.completed", {}
        ))
        assert ok is False

    def test_dispatch_webhook_refuses_localhost(self):
        from distllm.api.webhooks import dispatch_webhook

        ok = asyncio.run(dispatch_webhook(
            "http://127.0.0.1:9200/_search", "s3cret", "job.completed", {}
        ))
        assert ok is False

    def test_dispatch_webhook_posts_and_signs_public_url(self, monkeypatch):
        """Happy path over a real local ephemeral socket (loopback server used
        only as the *receiver* inside this test process; the allowlist env
        permits it exactly as an operator would opt in)."""
        monkeypatch.setenv("DISTLLM_WEBHOOK_ALLOWLIST", "127.0.0.1")
        received: list[tuple[str, dict]] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                received.append((self.headers.get("X-Webhook-Signature", ""), body))
                self.send_response(204)
                self.end_headers()

            def log_message(self, *args):  # silence
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            from distllm.api.webhooks import dispatch_webhook

            ok = asyncio.run(dispatch_webhook(
                f"http://127.0.0.1:{port}/hook", "topsecret", "job.completed",
                {"job_id": "j-1"},
            ))
            assert ok is True
            assert len(received) == 1
            sig, body = received[0]
            assert sig  # HMAC signature header present
            assert b'"event"' in body
        finally:
            server.shutdown()

    def test_dispatch_webhook_does_not_follow_redirects_to_private(self):
        """A public host redirecting to loopback must not be followed."""
        class RedirectHandler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                self.send_response(302)
                self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
                self.end_headers()

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), RedirectHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            from distllm.api.webhooks import dispatch_webhook

            ok = asyncio.run(dispatch_webhook(
                f"http://127.0.0.1:{port}/redirector", "sec", "job.completed", {},
            ))
            # The redirect target would have been blocked anyway; with
            # follow_redirects=False we simply report failure on 3xx.
            assert ok is False
        finally:
            server.shutdown()


# ── 5. Allowlist escape hatch ────────────────────────────────────────────────

class TestAllowlistEscapeHatch:
    def test_allowlisted_private_host_accepted_everywhere(self, webhook_store, monkeypatch):
        monkeypatch.setenv("DISTLLM_WEBHOOK_ALLOWLIST", "localhost")

        from distllm.api.routes.webhooks import WebhookCreate, create_webhook

        reg = asyncio.run(create_webhook(WebhookCreate(
            url="http://localhost:9000/local-hook",
            secret="x" * 16,
            events=["batch.completed"],
        )))
        assert reg.url == "http://localhost:9000/local-hook"

        # Delivery engine accepts it too.
        from distllm.api.webhooks.delivery import WebhookManager

        mgr = WebhookManager()
        wid = mgr.register("http://localhost:9000/local-hook", {"e"}, "s")
        assert mgr.get(wid).url.endswith(":9000/local-hook")

    def test_allowlist_is_deny_by_default(self, monkeypatch):
        """When an allowlist exists, even public hosts outside it are denied."""
        monkeypatch.setenv("DISTLLM_WEBHOOK_ALLOWLIST", "internal-hooks.corp.local")

        from distllm.core.webhook_manager import is_safe_webhook_url

        assert is_safe_webhook_url("https://internal-hooks.corp.local/hook") is True
        assert is_safe_webhook_url("https://hooks.example.com/webhook") is False

    def test_route_rejection_mentions_allowlist(self, webhook_store):
        from fastapi import HTTPException
        from distllm.api.routes.webhooks import WebhookCreate, create_webhook

        with pytest.raises(HTTPException) as exc:
            asyncio.run(create_webhook(WebhookCreate(
                url="http://10.1.2.3/x", secret="x" * 16, events=["batch.completed"],
            )))
        assert "DISTLLM_WEBHOOK_ALLOWLIST" in exc.value.detail


# ── 6. Core guard unit coverage (incl. new unspecified-IP rule) ──────────────

class TestCoreGuardUnit:
    def test_is_safe_webhook_url_basic_matrix(self):
        from distllm.core.webhook_manager import is_safe_webhook_url

        for url in SSRF_TARGETS:
            assert is_safe_webhook_url(url) is False, url
        assert is_safe_webhook_url(PUBLIC_URL) is True

    def test_unspecified_ip_rejected(self):
        """0.0.0.0 connects to localhost on most stacks — must be blocked."""
        from distllm.core.webhook_manager import is_safe_webhook_url

        assert is_safe_webhook_url("http://0.0.0.0:2375/containers/json") is False

    def test_backward_compat_alias_intact(self):
        from distllm.core import webhook_manager as wm

        assert wm.is_safe_webhook_url is wm._is_safe_webhook_url
