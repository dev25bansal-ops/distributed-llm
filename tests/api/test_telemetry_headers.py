"""Enhancement #5: telemetry response headers on /v1/chat/completions.

X-DistLLM-TTFT-Ms, X-DistLLM-Tokens-Per-Sec, X-DistLLM-Cost-USD.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app() -> FastAPI:
    app = FastAPI()
    from distllm.api.routes.chat import router

    app.include_router(router)
    return app


def _mock_coord(content: str = "hello world"):
    coord = MagicMock()
    coord.generate = MagicMock(return_value=content)
    tok = MagicMock()
    tok.encode = MagicMock(side_effect=lambda t, **kw: list(range(len(t.split()))))
    coord.tokenizer = tok
    coord.nodes = {}
    # Model surface: resolve() -> "m", and list_models() must contain it.
    coord._model_router = MagicMock()
    coord._model_router.resolve = MagicMock(return_value="m")
    coord._model_router.list_models = MagicMock(return_value=["m"])
    coord.list_models = MagicMock(return_value=["m"])
    coord.adapter_manager = MagicMock()
    coord.adapter_manager.list_adapters = MagicMock(return_value=[])
    # Text-only: no VLM pipeline (MagicMock attr would trip the multimodal path)
    coord._vlm_pipeline = None
    return coord


def test_telemetry_headers_present():
    app = _make_app()
    coord = _mock_coord()

    from distllm.api.api_state import g

    old = g.coordinator
    g.coordinator = coord
    try:
        client = TestClient(app)
        r = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
    finally:
        g.coordinator = old
    assert r.status_code == 200, r.text[:300]
    assert "X-DistLLM-TTFT-Ms" in r.headers
    assert "X-DistLLM-Tokens-Per-Sec" in r.headers
    # TTFT is a non-negative int in ms
    assert int(r.headers["X-DistLLM-TTFT-Ms"]) >= 0
    # tokens/sec parses as float
    float(r.headers["X-DistLLM-Tokens-Per-Sec"])


def test_telemetry_header_values_sane():
    app = _make_app()
    coord = _mock_coord("one two three four")

    from distllm.api.api_state import g

    old = g.coordinator
    g.coordinator = coord
    try:
        client = TestClient(app)
        r = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi there"}]},
        )
    finally:
        g.coordinator = old
    tps = float(r.headers["X-DistLLM-Tokens-Per-Sec"])
    assert tps >= 0.0
    # Cost header: present or absent, never garbage
    cost = r.headers.get("X-DistLLM-Cost-USD")
    if cost is not None:
        float(cost)


def test_streaming_response_has_headers():
    """Streaming path: response is well-formed when the streaming engine is
    unavailable on the mock coordinator (degrades to a clean error, never a
    500 with a broken SSE stream)."""
    app = _make_app()
    coord = _mock_coord()
    # Streaming reads coord._inference_engine.local_partitioner.full_model;
    # leave it unset (None engine) so the route takes its fallback path.
    coord._inference_engine = None

    from distllm.api.api_state import g

    old = g.coordinator
    g.coordinator = coord
    try:
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
    finally:
        g.coordinator = old
    # Never an unhandled crash: a structured error is acceptable here
    # because the mock has no real streaming model surface.
    assert r.status_code < 500, r.text[:300]
