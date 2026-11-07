"""Regression test for AF-04-STREAM-METER: streaming metering.

Quota records written for SSE responses used to carry output_tokens=0
because ``_estimate_usage`` only inspects ``response.body``, which a
``StreamingResponse`` never has.  The middleware now wraps the SSE
generator so the token counter increments per chunk and the quota
record is written on generator close with the real output token count.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

import distllm.api.quota_middleware as qm
from distllm.api.quota_middleware import QuotaMiddleware
from distllm.core.usage_meter import UsageMeter


def _sse_chunks(n_tokens: int):
    """Yield n SSE data chunks, each carrying exactly one 1-token delta."""
    for i in range(n_tokens):
        chunk = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "a"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"


@pytest.fixture
def streaming_app(monkeypatch):
    """FastAPI app whose endpoint streams a 5-token SSE response."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat():
        return StreamingResponse(
            _sse_chunks(5),
            media_type="text/event-stream",
        )

    class MockAuth(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.state.tenant_id = "tenant-stream"
            request.state.api_key_id = "key-stream"
            request.state.model = "test-model"
            return await call_next(request)

    meter = UsageMeter(storage_path=":memory:")
    app.add_middleware(QuotaMiddleware, enable=True)
    app.add_middleware(MockAuth)
    monkeypatch.setattr(qm, "_meter", meter)
    return TestClient(app), meter


def test_streaming_quota_record_has_real_output_tokens(streaming_app):
    """Streaming a fake 5-token SSE response must record output_tokens==5."""
    client, meter = streaming_app
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert resp.headers.get("content-type", "").startswith("text/event-stream")

    records = meter.records(tenant_id="tenant-stream")
    assert len(records) == 1, f"expected exactly 1 quota record, got {len(records)}"
    record = records[0]
    assert record.output_tokens == 5, (
        f"SSE quota record must carry real output tokens, got {record.output_tokens}"
    )
