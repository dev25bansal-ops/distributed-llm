"""Regression: streaming must not report a successful empty completion when
there is no backend that can generate.

Defect: ``_generate_tokens`` found ``local_partitioner=None`` and
``node_order=[]`` (the state of a distributed-mode coordinator before any
worker registers, or after all workers left) and fell off the end without
raising. ``_stream_response`` then emitted a role chunk, an empty
``finish_reason="stop"`` chunk and ``[DONE]`` with HTTP 200 — a
successful-looking empty completion. The identical non-streaming request
raised ``ConfigError("No nodes registered in the pipeline")`` (HTTP 500), so
only streaming silently "succeeded".

Starlette sends the 200 status line before the first iteration of the body
iterator, so the routes now call ``ensure_streaming_available()`` *before*
returning the StreamingResponse, and ``_generate_tokens`` raises the same
ConfigError if nodes vanish between that check and the first token.
"""

from __future__ import annotations

import os
import secrets

import pytest
from fastapi.testclient import TestClient

from distllm.api.api_state import g
from distllm.api.server import app
from distllm.api.streaming import _generate_tokens, ensure_streaming_available
from distllm.core.api_key_store import reset_api_key_store
from distllm.core.coordinator import Coordinator
from distllm.errors import ConfigError


@pytest.fixture(autouse=True)
def _isolated_quota_meter():
    """Point QuotaMiddleware at a scratch meter so these HTTP tests never
    observe another module's leaked singleton (a sibling test file leaves a
    permanently-rejecting ``qm._meter`` mock installed, which turned these
    requests into 429s when the full suite ran in one session)."""
    import distllm.api.quota_middleware as qm
    from distllm.core.usage_meter import UsageMeter

    original = qm._meter
    qm._meter = UsageMeter(storage_path=":memory:")
    yield
    qm._meter.close()
    qm._meter = original


def _make_client() -> TestClient:
    test_api_key = secrets.token_urlsafe(32)
    os.environ.pop("API_KEY_WAS_SET", None)
    os.environ.pop("API_KEYS", None)
    os.environ["API_KEY"] = test_api_key
    reset_api_key_store()
    client = TestClient(app, raise_server_exceptions=False)
    client.headers["Authorization"] = f"Bearer {test_api_key}"
    return client


def _cleanup_auth() -> None:
    os.environ.pop("API_KEY", None)
    os.environ.pop("API_KEY_WAS_SET", None)
    reset_api_key_store()


@pytest.fixture
def distributed_mode_coordinator():
    """A real Coordinator in distributed mode: no partitioner, no nodes."""
    coord = Coordinator(model_name="stub-model", dtype="float16")
    assert coord.local_partitioner is None
    assert not coord.node_order
    original = g.coordinator
    g.coordinator = coord
    yield coord
    g.coordinator = original


# ---------------------------------------------------------------------------
# The helper itself
# ---------------------------------------------------------------------------


class TestEnsureStreamingAvailable:
    def test_raises_when_coordinator_has_no_backend(self, distributed_mode_coordinator):
        with pytest.raises(ConfigError, match="No nodes registered"):
            ensure_streaming_available(distributed_mode_coordinator)

    def test_passes_with_local_partitioner(self):
        class _P:
            full_model = object()

        class _C:
            local_partitioner = _P()
            node_order: list = []

        ensure_streaming_available(_C())  # must not raise

    def test_passes_with_partitioner_on_inference_engine(self):
        """Single-node installs load weights into the engine's partitioner."""

        class _Engine:
            local_partitioner = object()

        class _C:
            local_partitioner = None
            node_order: list = []
            _inference_engine = _Engine()

        ensure_streaming_available(_C())  # must not raise

    def test_passes_with_registered_nodes(self):
        class _C:
            local_partitioner = None
            node_order = ["node-0"]

        ensure_streaming_available(_C())  # must not raise

    def test_none_coordinator_left_to_routes(self):
        """No coordinator at all is the routes' 503, not this guard's job."""
        ensure_streaming_available(None)  # must not raise


# ---------------------------------------------------------------------------
# The streaming generator's fall-through guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_tokens_raises_instead_of_falling_through(
    distributed_mode_coordinator,
):
    """Direct generator callers get the ConfigError, not an empty result."""
    gen = _generate_tokens(
        "hi", "req-1", max_tokens=4, temperature=0.7, top_p=0.9, top_k=0,
    )
    with pytest.raises(ConfigError, match="No nodes registered"):
        async for _ in gen:
            pass


# ---------------------------------------------------------------------------
# The HTTP contract: a real 5xx, never a silent empty 200
# ---------------------------------------------------------------------------


class TestStreamingWithNoBackend:
    @pytest.fixture(autouse=True)
    def _client(self, distributed_mode_coordinator):
        self.coord = distributed_mode_coordinator
        self.client = _make_client()
        yield
        _cleanup_auth()

    def test_streaming_chat_returns_error_not_empty_200(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "stub-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code >= 500, resp.text[:400]
        body = resp.json()
        err = body["error"]
        assert err["code"] == "config_error"
        assert "No nodes registered in the pipeline" in err["message"]
        # None of the silent-empty-200 stream shape may appear.
        assert "[DONE]" not in resp.text
        assert "text/event-stream" not in resp.headers.get("content-type", "")

    def test_streaming_completion_returns_error_not_empty_200(self):
        resp = self.client.post(
            "/v1/completions",
            json={
                "model": "stub-model",
                "prompt": "hi",
                "max_tokens": 4,
                "stream": True,
            },
        )
        assert resp.status_code >= 500, resp.text[:400]
        assert resp.json()["error"]["code"] == "config_error"
        assert "[DONE]" not in resp.text

    def test_parity_with_non_streaming(self):
        """The streaming and non-streaming statuses now agree."""
        payload = {
            "model": "stub-model",
            "messages": [{"role": "user", "content": "hi"}],
        }
        non_stream = self.client.post(
            "/v1/chat/completions", json={**payload, "stream": False}
        )
        stream = self.client.post(
            "/v1/chat/completions", json={**payload, "stream": True}
        )
        assert non_stream.status_code >= 500
        assert stream.status_code == non_stream.status_code
        assert (
            stream.json()["error"]["code"] == non_stream.json()["error"]["code"]
        )


# ---------------------------------------------------------------------------
# The catch-all 500 handler must keep typed errors' actionable message
# ---------------------------------------------------------------------------


class TestGeneralExceptionHandlerMessage:
    """Defect: the catch-all replaced every unhandled exception's message
    with "An unexpected error occurred. Please try again later.", telling the
    caller to retry a configuration problem that will never succeed until an
    operator acts. The cause existed only in the server log."""

    @pytest.mark.asyncio
    async def test_distllm_error_keeps_actionable_message(self):
        import json as _json

        from distllm.api.server_errors import general_exception_handler

        exc = ConfigError(
            "No nodes registered in the pipeline", context={"action": "generate"}
        )
        resp = await general_exception_handler(None, exc)  # type: ignore[arg-type]
        body = _json.loads(resp.body)
        assert "No nodes registered in the pipeline" in body["error"]["message"]
        assert "try again later" not in body["error"]["message"]
        assert body["error"]["code"] == "config_error"

    @pytest.mark.asyncio
    async def test_untyped_error_stays_masked(self):
        import json as _json

        from distllm.api.server_errors import general_exception_handler

        exc = RuntimeError("secret internal detail /tmp/path")
        resp = await general_exception_handler(None, exc)  # type: ignore[arg-type]
        body = _json.loads(resp.body)
        assert body["error"]["message"] == (
            "An unexpected error occurred. Please try again later."
        )
        assert "secret internal detail" not in body["error"]["message"]

    def test_config_error_message_reaches_the_caller_end_to_end(
        self, distributed_mode_coordinator
    ):
        """The non-streaming repro now carries the real cause to the client."""
        client = _make_client()
        try:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "stub-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )
            assert resp.status_code == 500
            err = resp.json()["error"]
            assert "No nodes registered in the pipeline" in err["message"]
            assert err["code"] == "config_error"
        finally:
            _cleanup_auth()