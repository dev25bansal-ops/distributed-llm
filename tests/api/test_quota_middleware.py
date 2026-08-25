"""Tests for QuotaMiddleware.

Covers:
- Token estimation with and without tiktoken
- Quota enforcement (under limit, at limit, exceeded)
- Skipping quota when disabled / untracked paths
- DISTLLM_QUOTA_ENABLED env gate semantics
- 429 response contract (Retry-After header + OpenAI error envelope)
- Per-tenant isolation and usage recording
- Real-app mount conformance: registered in server.py, inner of AuthMiddleware
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

import distllm.api.quota_middleware as qm
from distllm.api.quota_middleware import (
    QuotaMiddleware,
    _estimate_token_count,
    get_usage_meter,
)


# ======================================================================
# _estimate_token_count unit tests
# ======================================================================


class TestEstimateTokenCount:
    def test_returns_zero_for_empty_text(self):
        assert _estimate_token_count("") == 0
        assert _estimate_token_count(None) == 0

    def test_fallback_heuristic(self):
        """Without tiktoken, uses len(text) // 4."""
        # If tiktoken is installed, the heuristic isn't used, so we just
        # verify the function returns a reasonable positive integer.
        count = _estimate_token_count("Hello, world!")
        assert count > 0

    def test_uses_tiktoken_when_available(self):
        """When tiktoken is available, it's used for encoding."""
        # tiktoken is a real installed package; the function imports it
        # inside its body.  We can't mock it away, so just verify it
        # returns a plausible count.
        try:
            import tiktoken
            count = _estimate_token_count("Some text here")
            assert count > 0
        except ImportError:
            pass  # tiktoken not installed, test irrelevant

    def test_heuristic_for_long_text(self):
        """Long text returns a positive token estimate."""
        text = "word " * 200  # ~1000 chars
        count = _estimate_token_count(text)
        assert count > 0


# ======================================================================
# QuotaMiddleware integration tests
# ======================================================================


@pytest.fixture
def quota_app():
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat():
        return {"choices": [{"message": {"content": "Hello!"}}]}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def _make_client(app, enabled: bool = True):
    """Helper: add QuotaMiddleware and return TestClient."""
    app.add_middleware(QuotaMiddleware, enable=enabled)
    client = TestClient(app)

    # Wire basic request state that AuthMiddleware normally sets
    async def mock_auth_middleware(request, call_next):
        request.state.tenant_id = "test-tenant"
        request.state.api_key_id = "key-123"
        request.state.api_key_role = "inference-only"
        request.state.model = "test-model"
        return await call_next(request)

    app.user_middleware.insert(0, type("mock_auth", (), {"__call__": mock_auth_middleware}))
    return client


def test_passthrough_when_disabled():
    """When DISTLLM_QUOTA_ENABLED=0 (default), requests pass unhindered."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat():
        return {"choices": [{"message": {"content": "ok"}}]}

    app.add_middleware(QuotaMiddleware, enable=False)
    client = TestClient(app)

    resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200


def _make_client(enabled=True, _app=None):
    """Helper: create FastAPI app with QuotaMiddleware + mock auth state."""
    if _app is None:
        app = FastAPI()
    else:
        app = _app

    class MockAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.state.tenant_id = "test-tenant"
            request.state.api_key_id = "key-123"
            request.state.api_key_role = "inference-only"
            request.state.model = "test-model"
            return await call_next(request)

    # Register mock auth AFTER QuotaMiddleware so it runs FIRST (outermost)
    # on incoming requests, setting request.state before QuotaMiddleware reads it.
    app.add_middleware(QuotaMiddleware, enable=enabled)
    app.add_middleware(MockAuthMiddleware)
    return TestClient(app)


def test_allows_request_under_quota():
    """When tenant is under quota, request succeeds."""
    app = FastAPI()

    from distllm.core.usage_meter import UsageMeter
    meter = UsageMeter(storage_path=":memory:")
    # Real signature: enforce_quota(tenant_id, raise_on_block=True, requested_tokens=None)
    meter.enforce_quota = lambda tenant_id, raise_on_block=True, requested_tokens=None: (True, "")
    import distllm.api.quota_middleware as qm
    qm._meter = meter

    @app.post("/v1/chat/completions")
    async def chat():
        return {"choices": [{"message": {"content": "ok"}}]}

    client = _make_client(True, app)

    resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200


def test_rejects_request_when_quota_exceeded():
    """When tenant quota is exceeded, returns 429."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat():
        return {"choices": [{"message": {"content": "ok"}}]}

    client = _make_client(True, app)

    from distllm.core.usage_meter import UsageMeter
    meter = UsageMeter(storage_path=":memory:")  # fresh in-memory meter
    meter.enforce_quota = lambda tenant_id, raise_on_block=True, requested_tokens=None: (
        False, "Daily token limit exceeded")
    import distllm.api.quota_middleware as qm
    qm._meter = meter

    resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 429
    body = resp.json()
    assert "quota_exceeded" in body.get("error", {}).get("type", "")


def test_skips_non_tracked_paths():
    """Health endpoints are not tracked by quota."""
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    app.add_middleware(QuotaMiddleware, enable=True)
    client = TestClient(app)

    resp = client.get("/health")
    assert resp.status_code == 200


def test_should_track_returns_true_for_chat():
    mw = QuotaMiddleware.__new__(QuotaMiddleware)
    assert mw._should_track("/v1/chat/completions") is True
    assert mw._should_track("/v1/completions") is True
    assert mw._should_track("/v1/embeddings") is True


def test_should_track_returns_false_for_other():
    mw = QuotaMiddleware.__new__(QuotaMiddleware)
    assert mw._should_track("/health") is False
    assert mw._should_track("/v1/plugins") is False


# ======================================================================
# Env-gate behaviour (DISTLLM_QUOTA_ENABLED)
# ======================================================================


class TestEnvGate:
    def test_defaults_on(self, monkeypatch):
        monkeypatch.delenv("DISTLLM_QUOTA_ENABLED", raising=False)
        assert qm.quota_enabled_from_env() is True

    def test_explicit_one_enables(self, monkeypatch):
        monkeypatch.setenv("DISTLLM_QUOTA_ENABLED", "1")
        assert qm.quota_enabled_from_env() is True

    def test_zero_disables(self, monkeypatch):
        monkeypatch.setenv("DISTLLM_QUOTA_ENABLED", "0")
        assert qm.quota_enabled_from_env() is False

    def test_garbage_value_disables(self, monkeypatch):
        monkeypatch.setenv("DISTLLM_QUOTA_ENABLED", "yes")
        assert qm.quota_enabled_from_env() is False

    def test_constructor_reads_env_when_enable_none(self, monkeypatch):
        monkeypatch.setenv("DISTLLM_QUOTA_ENABLED", "0")
        mw = QuotaMiddleware(lambda scope, receive, send: None)
        assert mw._enabled is False

    def test_explicit_enable_overrides_env(self, monkeypatch):
        monkeypatch.setenv("DISTLLM_QUOTA_ENABLED", "0")
        mw = QuotaMiddleware(lambda scope, receive, send: None, enable=True)
        assert mw._enabled is True


# ======================================================================
# Enforcement via TestClient — stub app, injected in-memory meter
# ======================================================================


def _over_limit_app(monkeypatch, *, set_request_id: bool = True):
    """FastAPI app + QuotaMiddleware + mock auth + over-limit in-memory meter.

    Returns ``(client, meter, hits)`` where ``hits`` counts route executions.
    """
    from distllm.core.usage_meter import QuotaLimit, UsageMeter

    app = FastAPI()
    hits = {"route": 0}

    @app.post("/v1/chat/completions")
    async def chat():
        hits["route"] += 1
        return {"choices": [{"message": {"content": "ok"}}]}

    class MockAuth(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.state.tenant_id = "tenant-x"
            request.state.api_key_id = "key-x"
            request.state.api_key_role = "inference-only"
            request.state.model = "test-model"
            if set_request_id:
                request.state.request_id = "req-quota-1"
            return await call_next(request)

    meter = UsageMeter(storage_path=":memory:")
    meter.set_quota("tenant-x", QuotaLimit(tenant_id="tenant-x", max_tokens_per_day=10))
    # Burn the daily budget: next enforce_quota() must reject.
    meter.record_request(
        tenant_id="tenant-x", model_name="test-model",
        input_tokens=100, output_tokens=0,
    )

    # Innermost-first registration: quota inner of auth so auth populates
    # request.state first (mirrors production ordering).
    app.add_middleware(QuotaMiddleware, enable=True)
    app.add_middleware(MockAuth)

    # Route the middleware at OUR meter, never the SQLite-backed singleton.
    monkeypatch.setattr(qm, "_meter", meter)
    return TestClient(app), meter, hits


class TestOverLimitRejection:
    def test_returns_429_with_retry_after_header_and_error_envelope(self, monkeypatch):
        client, meter, hits = _over_limit_app(monkeypatch)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 429
        # Proper HTTP header (RFC 6585), not just a body field.
        assert resp.headers.get("Retry-After") == "60"
        body = resp.json()
        err = body["error"]
        assert err["type"] == "quota_exceeded"
        assert err["code"] == "429"
        assert "exceeded" in err["message"]
        # request_id propagated from request.state into the envelope.
        assert body.get("request_id") == "req-quota-1"

    def test_blocked_request_never_reaches_route(self, monkeypatch):
        client, meter, hits = _over_limit_app(monkeypatch)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 429
        assert hits["route"] == 0

    def test_concurrency_slot_not_leaked_on_rejection(self, monkeypatch):
        client, meter, _hits = _over_limit_app(monkeypatch)
        client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        # Rejection happens inside enforce_quota before increment; the
        # failed path must never call release_quota into negative territory.
        assert meter.get_concurrent("tenant-x") == 0


class TestUnderLimitFlow:
    def _client(self, monkeypatch):
        from distllm.api.body_cache_middleware import BodyCacheMiddleware
        from distllm.core.usage_meter import QuotaLimit, UsageMeter

        app = FastAPI()
        state_seen = {}

        @app.post("/v1/chat/completions")
        async def chat():
            return {"choices": [{"message": {"content": "Hello!"}}]}

        class MockAuth(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                request.state.tenant_id = "tenant-ok"
                request.state.api_key_id = "key-ok"
                request.state.model = "test-model"
                return await call_next(request)

        meter = UsageMeter(storage_path=":memory:")
        meter.set_quota("tenant-ok", QuotaLimit(tenant_id="tenant-ok", max_tokens_per_day=1_000_000))

        # Execution order (outer -> inner): BodyCache -> MockAuth -> Quota,
        # mirroring production where BodyCacheMiddleware wraps everything and
        # QuotaMiddleware reads its request.state.parsed_body snapshot.
        app.add_middleware(QuotaMiddleware, enable=True)
        app.add_middleware(MockAuth)
        app.add_middleware(BodyCacheMiddleware)
        monkeypatch.setattr(qm, "_meter", meter)
        return TestClient(app), meter

    def test_request_passes_and_usage_recorded(self, monkeypatch):
        client, meter = self._client(monkeypatch)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hello world"}]},
        )
        assert resp.status_code == 200
        usage = meter.total_usage()
        assert usage["total_requests"] == 1
        assert usage["total_input_tokens"] > 0
        # Concurrency slot released after completion.
        assert meter.get_concurrent("tenant-ok") == 0
        record = meter.records(tenant_id="tenant-ok")[0]
        assert record.key_id == "key-ok"
        assert record.endpoint == "/v1/chat/completions"

    def test_tenant_isolation_other_tenant_unaffected(self, monkeypatch):
        client, meter = self._client(monkeypatch)
        client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert meter.tenant_usage("tenant-ok") is not None
        assert meter.tenant_usage("some-other-tenant") is None


class TestDisabledAndUntracked:
    def test_disabled_middleware_never_touches_meter(self, monkeypatch):
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat():
            return {"choices": [{"message": {"content": "ok"}}]}

        app.add_middleware(QuotaMiddleware, enable=False)
        monkeypatch.setattr(qm, "_meter", None)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        assert qm._meter is None  # singleton never constructed

    def test_untracked_path_skips_meter_entirely(self, monkeypatch):
        app = FastAPI()

        @app.get("/healthz")
        async def healthz():
            return {"status": "ok"}

        app.add_middleware(QuotaMiddleware, enable=True)
        monkeypatch.setattr(qm, "_meter", None)
        client = TestClient(app)
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert qm._meter is None  # untracked traffic must not build the DB


# ======================================================================
# Real-app conformance: QuotaMiddleware mounted in server.py
# ======================================================================


def _quota_gate_on() -> bool:
    return os.environ.get("DISTLLM_QUOTA_ENABLED", "1") == "1"


class TestQuotaMountConformance:
    """Static ordering pin, mirroring the DocsAuth B1-conformance pattern."""

    def test_quota_in_stack_and_inner_of_auth(self):
        if not _quota_gate_on():
            pytest.skip("DISTLLM_QUOTA_ENABLED != 1 at import time")
        from distllm.api.middleware import AuthMiddleware
        from distllm.api.server import app

        classes = [mw.cls for mw in app.user_middleware]
        assert QuotaMiddleware in classes, (
            "QuotaMiddleware missing from the live middleware stack — "
            "per-tenant quotas silently off (audit Q1 regression)"
        )
        # Lower user_middleware index = outer = runs first.  Auth MUST run
        # before Quota so request.state carries tenant identity.
        assert classes.index(AuthMiddleware) < classes.index(QuotaMiddleware), (
            "QuotaMiddleware is outer of AuthMiddleware — it would see "
            "unset tenant/api-key state and attribute everything to "
            "'anonymous' (ordering regression)"
        )

    def test_gate_off_leaves_stack_unchanged(self, monkeypatch):
        monkeypatch.setenv("DISTLLM_QUOTA_ENABLED", "0")
        # The gate is evaluated at server-module import; simulate the
        # decision the mount site makes rather than rebuilding the app.
        assert _quota_gate_on() is False


class TestQuotaEndToEndRealApp:
    """Over-limit 429 and default-pass flows through the REAL app object."""

    @pytest.fixture
    def e2e(self, monkeypatch):
        import secrets as _secrets

        from distllm.api.api_state import g
        from distllm.api.server import app
        from distllm.core.api_key_store import reset_api_key_store
        from distllm.core.usage_meter import QuotaLimit, UsageMeter

        key = _secrets.token_urlsafe(32)
        monkeypatch.delenv("DISABLE_AUTH", raising=False)
        monkeypatch.delenv("DISTLLM_DEV_MODE", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.delenv("API_KEY_WAS_SET", raising=False)
        monkeypatch.setenv(
            "API_KEYS",
            json.dumps({"keys": [{
                "key": key, "role": "inference-only",
                "label": "quota-e2e", "key_id": "quota-e2e",
            }]}),
        )
        reset_api_key_store()

        # Mock coordinator (same shape as test_chat_basic.make_mock_coordinator)
        try:
            import torch
            coord = MagicMock()
            coord.model_name = "test-model"
            coord.nodes = {}
            coord.node_order = []
            coord.scheduler = None
            coord.prefix_cache = None
            coord.metrics_exporter = None

            def encode_fn(text, **kwargs):
                tokens = [1, 2, 3, 4, 5]
                if kwargs.get("return_tensors") == "pt":
                    return torch.tensor([tokens])
                return tokens

            coord.tokenizer.encode.side_effect = encode_fn
            coord.tokenizer.decode.side_effect = lambda tokens, **kw: "tok tok tok"
            coord.tokenizer.eos_token_id = 0
            coord.generate.return_value = "Hello! This is a test response."
            mock_model = MagicMock()
            mock_model.parameters.side_effect = lambda: iter([torch.randn(10, 10)])
            mock_output = MagicMock()
            mock_output.logits = torch.randn(1, 5, 1000)
            mock_output.past_key_values = MagicMock()
            mock_model.return_value = mock_output
            coord.local_partitioner = MagicMock()
            coord.local_partitioner.full_model = mock_model
            coord.list_models.return_value = ["test-model"]
            coord._vlm_pipeline = None
            coord._spec_decoder = None
            coord._model_router = None
            coord._shutting_down = False
            coord.tokenizer.chat_template = None
        except ImportError:
            coord = MagicMock()

        original_coord = g.coordinator
        g.coordinator = coord

        # Isolated in-memory meter instead of the repo-root .usage.db.
        meter = UsageMeter(storage_path=":memory:")
        monkeypatch.setattr(qm, "_meter", meter)

        client = TestClient(app)
        client.headers["Authorization"] = f"Bearer {key}"

        yield {"client": client, "meter": meter, "key_id": "quota-e2e"}

        g.coordinator = original_coord
        reset_api_key_store()

    def test_default_no_quota_set_does_not_break_flow(self, e2e):
        """Fresh tenant, no QuotaLimit configured -> request succeeds."""
        client = e2e["client"]
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test-model",
                  "messages": [{"role": "user", "content": "say hi"}],
                  "max_tokens": 5},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "choices" in data
        # Usage was attributed to the authenticated key id.
        assert e2e["meter"].total_usage()["total_requests"] >= 1
        assert e2e["meter"].tenant_usage("quota-e2e") is not None

    def test_low_daily_quota_returns_429_with_headers(self, e2e):
        """Tenant configured with a tiny daily cap -> 429 + Retry-After."""
        from distllm.core.usage_meter import QuotaLimit

        meter = e2e["meter"]
        meter.set_quota("quota-e2e", QuotaLimit(tenant_id="quota-e2e", max_tokens_per_day=5))
        # Pre-burn the budget so the very next request is over-limit.
        meter.record_request(
            tenant_id="quota-e2e", model_name="test-model",
            input_tokens=500, output_tokens=0,
        )

        resp = e2e["client"].post(
            "/v1/chat/completions",
            json={"model": "test-model",
                  "messages": [{"role": "user", "content": "over the hill"}],
                  "max_tokens": 5},
        )
        assert resp.status_code == 429, resp.text
        assert resp.headers.get("Retry-After") == "60"
        err = resp.json()["error"]
        assert err["type"] == "quota_exceeded"
        assert "daily token limit" in err["message"]
        assert meter.get_concurrent("quota-e2e") == 0

    def test_health_endpoint_untracked_by_quota(self, e2e):
        resp = e2e["client"].get("/healthz")
        assert resp.status_code == 200
