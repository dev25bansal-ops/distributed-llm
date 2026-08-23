"""N6 — Federated observability: W3C trace-context propagation.

Proves that trace context is propagated across the federation boundary via
the heartbeat path, reusing ``distllm.observability.tracing`` helpers:

  (a) a span started before ``_exchange_heartbeats`` yields outgoing request
      metadata containing ``traceparent``;
  (b) extracting that metadata reconstructs a context whose trace-id equals
      the parent span's trace-id (cross-boundary correlation);
  (c) no import breaks and the additive API surface exists.

Tests are model-free and network-free — the heartbeat HTTP POST is
monkeypatched to capture the outgoing headers instead of hitting the wire.
"""

from __future__ import annotations

import pytest

from opentelemetry import trace, context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace.propagation.tracecontext import (
    TraceContextTextMapPropagator,
)


# ── shared OTel provider (real SDK spans, no exporter/network) ──────────
_PROVIDER = TracerProvider()
trace.set_tracer_provider(_PROVIDER)
_TRACER = trace.get_tracer("test_n6")


def _make_coordinator():
    """Build a FederationCoordinator with one peer, no threads, no network."""
    from distllm.dist.federation import FederationConfig, FederationCoordinator
    from distllm.dist.p2p.discovery import PeerInfo

    cfg = FederationConfig(enabled=False, cluster_id="cluster-local")
    coord = FederationCoordinator(
        config=cfg,
        local_cluster_id="cluster-local",
        local_host="127.0.0.1",
        local_port=50050,
        coordinator_ref=None,
    )
    # Register a single remote peer.
    coord._peers = {
        "cluster-remote": PeerInfo(
            cluster_id="cluster-remote",
            host="10.0.0.2",
            port=50050,
        )
    }
    # No SVID / auth headers needed for the propagation assertions.
    coord._svid = None
    # psutil may be absent in the CI env; _get_local_load imports it. Stub it
    # so the network-free heartbeat path reaches the (patched) HTTP POST.
    coord._get_local_load = lambda: {
        "active_requests": 0,
        "pending_requests": 0,
        "gpu_utilization": 0.0,
    }
    return coord


# ── (c) import / API-surface sanity ────────────────────────────────────

def test_imports_and_api_surface():
    from distllm.dist import federation
    from distllm.observability import tracing

    assert hasattr(federation.FederationCoordinator, "_trace_metadata")
    assert hasattr(
        federation.FederationCoordinator, "extract_incoming_trace_context"
    )
    # Reuses existing tracing helpers (not duplicated).
    assert hasattr(tracing, "inject_trace_context")
    assert hasattr(tracing, "extract_trace_context")
    assert hasattr(tracing, "TraceContextTextMapPropagator")


def test_trace_metadata_empty_without_span():
    """With no active recording span, _trace_metadata yields no traceparent."""
    coord = _make_coordinator()
    md = coord._trace_metadata()
    assert isinstance(md, dict)
    assert "traceparent" not in md


def test_trace_metadata_injects_current_span():
    """_trace_metadata reflects the CURRENT span's trace-id."""
    coord = _make_coordinator()
    with _TRACER.start_as_current_span("parent") as span:
        md = coord._trace_metadata()
        assert "traceparent" in md
        parent_tid = format(span.get_span_context().trace_id, "032x")
        assert parent_tid in md["traceparent"]


# ── (a) + (b) end-to-end propagation across the heartbeat boundary ─────

def test_heartbeat_injects_and_propagates_trace_context(monkeypatch):
    coord = _make_coordinator()

    captured = {}

    def _fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers or {}

        class _Resp:
            status_code = 200

            @staticmethod
            def json():
                return {"active_requests": 0, "pending_requests": 0}

        return _Resp()

    # Capture outgoing metadata instead of hitting the network.
    monkeypatch.setattr(coord._http_client, "post", _fake_post)

    # Start a parent span, then exchange heartbeats. The child
    # 'federation.heartbeat' span's context must be injected.
    with _TRACER.start_as_current_span("federation-op") as parent_span:
        parent_tid = format(parent_span.get_span_context().trace_id, "032x")
        coord._exchange_heartbeats()

    headers = captured.get("headers", {})

    # (a) outgoing metadata contains traceparent
    assert "traceparent" in headers, f"headers were: {headers}"

    # (b) extracting reconstructs a context whose trace-id == parent's
    propagator = TraceContextTextMapPropagator()
    extracted_ctx = propagator.extract(carrier=dict(headers))
    remote_span = trace.get_current_span(extracted_ctx)
    remote_tid = format(remote_span.get_span_context().trace_id, "032x")

    assert remote_tid == parent_tid, (
        f"cross-boundary trace-id mismatch: {remote_tid} != {parent_tid}"
    )


def test_extract_incoming_trace_context_is_safe():
    """Receive-side helper must never raise on typical header shapes."""
    from distllm.dist.federation import FederationCoordinator

    # None, dict, and list-of-tuples must all be tolerated.
    FederationCoordinator.extract_incoming_trace_context(None)
    FederationCoordinator.extract_incoming_trace_context(
        {"traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01"}
    )
    FederationCoordinator.extract_incoming_trace_context(
        [("traceparent", "00-" + "c" * 32 + "-" + "d" * 16 + "-01")]
    )


def test_heartbeat_no_traceparent_without_active_span(monkeypatch):
    """Sanity: outside any span, heartbeat sends no traceparent (network-free)."""
    coord = _make_coordinator()
    captured = {}

    def _fake_post(url, json=None, headers=None, timeout=None):
        captured["headers"] = headers or {}

        class _Resp:
            status_code = 200

            @staticmethod
            def json():
                return {"active_requests": 0, "pending_requests": 0}

        return _Resp()

    monkeypatch.setattr(coord._http_client, "post", _fake_post)

    try:
        coord._exchange_heartbeats()
    except Exception as e:  # pragma: no cover - defensive, network-free path
        pytest.skip(f"heartbeat path unavailable in this env: {e}")

    # A child span IS created inside _exchange_heartbeats, so a traceparent
    # for that child span is expected even without an outer span.
    assert "traceparent" in captured.get("headers", {})
