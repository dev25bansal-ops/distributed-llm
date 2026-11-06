"""RED metrics must carry real model/tenant labels, not constants.

Regression for the audit finding that /metrics exported distllm_requests_total
with hardcoded model="distributed-llm" tenant="default" on every request —
the labels existed but were dead for attribution. _record_metrics now reads
request.state.model (set by chat routes) and request.state.tenant_id (SSO)
with api_key_id fallback and the old constants as last resort.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from distllm.api.observability_middleware import ObservabilityMiddleware


def _make_request(**state_attrs):
    req = MagicMock()
    # real request.state returns None for unset attrs; a MagicMock returns
    # child mocks, which would poison the getattr(...) or ... fallbacks.
    req.state = SimpleNamespace(tenant_id=None, model=None, api_key_id=None)
    for k, v in state_attrs.items():
        setattr(req.state, k, v)
    return req


def test_record_metrics_uses_real_model_and_tenant():
    exporter = MagicMock()
    mw = ObservabilityMiddleware(app=MagicMock(), metrics_exporter=exporter)

    req = _make_request(model="llama-3.1-70b", tenant_id="tenant-42")
    mw._record_metrics(req, "GET", "/v1/chat/completions", 200, 0.5, None)

    calls = exporter.requests_total.labels.call_args_list
    assert any(c.kwargs.get("model") == "llama-3.1-70b" for c in calls), calls
    assert any(c.kwargs.get("tenant") == "tenant-42" for c in calls), calls


def test_record_metrics_falls_back_to_api_key_id_then_default():
    exporter = MagicMock()
    mw = ObservabilityMiddleware(app=MagicMock(), metrics_exporter=exporter)

    req = _make_request(api_key_id="key-7")
    mw._record_metrics(req, "POST", "/v1/embeddings", 200, 0.1, None)
    tenant_calls = [c.kwargs.get("tenant") for c in exporter.requests_total.labels.call_args_list]
    assert "key-7" in tenant_calls, tenant_calls

    # nothing on state -> constants
    exporter.reset_mock()
    mw._record_metrics(_make_request(), "GET", "/health", 200, 0.01, None)
    defaults = [c.kwargs for c in exporter.requests_total.labels.call_args_list][0]
    assert defaults["model"] == "distributed-llm"
    assert defaults["tenant"] == "default"
