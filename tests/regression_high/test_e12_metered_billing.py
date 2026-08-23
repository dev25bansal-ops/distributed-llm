"""E12 regression: multi-tenant metered billing (reuse cost_tracker).

This test exercises the thin metering layer built on top of the EXISTING
:mod:`distllm.core.cost_tracker`:

  * :class:`~distllm.api.metering.UsageRecord` — billing-oriented record.
  * :class:`~distllm.api.metering.MeteringStore` — in-memory store with
    per-tenant tallies (optional JSONL backend, pluggable).
  * :class:`~distllm.api.metering.MeteringMiddleware` — taps the request flow
    and records a UsageRecord, REUSING the singleton ``CostTracker`` (no
    duplicated quota/cost logic).
  * :class:`~distllm.api.metering.BillingExporter` — invoice JSON; Stripe is a
    documented STUB (no network, no real billing).

Required assertions (per task):
  (1) two tenants get *separate* tallies;
  (2) cost is computed (reuses cost_tracker's estimate, proportional to tokens);
  (3) export_invoice produces a valid JSON invoice AND, with STRIPE_API_KEY
      unset, runs the stub (no network).
"""

from __future__ import annotations

import json

import pytest

from distllm.core.cost_tracker import get_cost_tracker
from distllm.core.metering import (
    BillingExporter,
    JsonlBackend,
    MeteringMiddleware,
    MeteringStore,
    UsageRecord,
    get_metering_store,
    reset_metering_store,
)


@pytest.fixture
def store():
    """Fresh, isolated, in-memory store (no file backend)."""
    reset_metering_store()
    s = MeteringStore()  # no backend -> pure memory
    yield s
    s.reset()


@pytest.fixture
def no_stripe(monkeypatch):
    """Guarantee STRIPE_API_KEY is unset so we hit the documented stub."""
    monkeypatch.delenv("STRIPE_API_KEY", raising=False)
    yield


class TestMeteringStore:
    def test_two_tenants_get_separate_tallies(self, store):
        store.record_request(
            tenant_id="acme", tokens_in=100, tokens_out=50,
            compute_s=1.2, cost_usd=0.01, model_name="gpt-4o-mini",
        )
        store.record_request(
            tenant_id="acme", tokens_in=200, tokens_out=80,
            compute_s=2.0, cost_usd=0.02, model_name="gpt-4o-mini",
        )
        store.record_request(
            tenant_id="globex", tokens_in=10, tokens_out=5,
            compute_s=0.3, cost_usd=0.001, model_name="gpt-4o-mini",
        )

        acme = store.tally("acme")
        globex = store.tally("globex")

        # Separate per-tenant rollups, not summed together.
        assert acme["requests"] == 2
        assert globex["requests"] == 1
        assert acme["tokens_in"] == 300
        assert acme["tokens_out"] == 130
        assert acme["total_tokens"] == 430
        assert globex["tokens_in"] == 10
        assert globex["cost_usd"] != acme["cost_usd"]
        # acme cost is exactly the sum of its two records
        assert acme["cost_usd"] == pytest.approx(0.03)
        assert globex["cost_usd"] == pytest.approx(0.001)

    def test_cost_is_computed_not_hand_waved(self, store):
        # Reuse the existing CostTracker to assert the recorded cost matches
        # the canonical estimate (cost math lives in cost_tracker, not here).
        tracker = get_cost_tracker()
        est = tracker.estimate_cost(
            input_tokens=1000, output_tokens=500, model_name="gpt-4o-mini",
        )
        assert est.estimated_cost_usd > 0  # cost was actually computed

        rec = store.record_request(
            tenant_id="acme",
            tokens_in=1000,
            tokens_out=500,
            compute_s=est.estimated_gpu_seconds,
            cost_usd=est.estimated_cost_usd,
            model_name="gpt-4o-mini",
        )
        # The stored cost equals the reused cost_tracker estimate (proportional
        # to token count, not a constant stub).
        assert rec.cost_usd == pytest.approx(est.estimated_cost_usd)
        assert rec.cost_usd > 0
        # Bigger request -> bigger cost (proportionality to tokens).
        est_big = tracker.estimate_cost(
            input_tokens=10000, output_tokens=5000, model_name="gpt-4o-mini",
        )
        assert est_big.estimated_cost_usd > est.estimated_cost_usd
        t = store.tally("acme")
        assert t["cost_usd"] == pytest.approx(est.estimated_cost_usd)

    def test_jsonl_backend_persists_roundtrip(self, tmp_path):
        path = str(tmp_path / "usage.jsonl")
        backend = JsonlBackend(path)
        s1 = MeteringStore(backend=backend)
        s1.record_request(tenant_id="t", tokens_in=5, tokens_out=5,
                          compute_s=0.1, cost_usd=0.001)
        # Reload from disk via a second store sharing the same backend.
        s2 = MeteringStore(backend=JsonlBackend(path))
        assert len(s2.all_records()) == 1
        assert s2.tally("t")["requests"] == 1

    def test_usage_record_is_immutable_friendly(self):
        r = UsageRecord(
            tenant_id="x", timestamp=1.0, tokens_in=2, tokens_out=3,
            compute_s=0.5, cost_usd=0.01,
        )
        assert r.total_tokens == 5
        d = r.to_dict()
        assert d["tenant_id"] == "x"
        assert isinstance(d, dict)


class TestBillingExporterStub:
    def test_export_invoice_is_valid_json_via_stub(self, store, no_stripe):
        store.record_request(tenant_id="acme", tokens_in=100, tokens_out=50,
                             compute_s=1.0, cost_usd=0.05, model_name="gpt-4o-mini")
        store.record_request(tenant_id="acme", tokens_in=40, tokens_out=20,
                             compute_s=0.5, cost_usd=0.02, model_name="gpt-4o-mini")

        exporter = BillingExporter()
        invoice = exporter.export_invoice("acme", store.records_for_tenant("acme"),
                                          period="2026-07")

        # Must be serializeable/valid JSON and round-trippable.
        blob = json.loads(json.dumps(invoice))
        assert blob["tenant_id"] == "acme"
        assert blob["period"] == "2026-07"
        assert blob["line_item_count"] == 2
        assert blob["subtotal_usd"] == pytest.approx(0.07)
        assert blob["amount_due_usd"] == pytest.approx(0.07)
        # STRIPE_API_KEY unset -> stub mode, NO network.
        assert blob["mode"] == "stub"
        assert "Stripe" in blob["note"]

    def test_export_invoice_json_string_is_parseable(self, store, no_stripe):
        store.record_request(tenant_id="acme", tokens_in=10, tokens_out=10,
                             compute_s=0.2, cost_usd=0.003)
        exporter = BillingExporter()
        txt = exporter.export_invoice_json("acme", store.records_for_tenant("acme"))
        parsed = json.loads(txt)  # raises if not valid JSON
        assert parsed["mode"] == "stub"
        assert parsed["line_item_count"] == 1

    def test_stub_used_when_stripe_missing(self, store, no_stripe, monkeypatch):
        # Even if a key is provided, without the `stripe` package installed we
        # must fall back to the stub (no network, no crash).
        monkeypatch.setenv("STRIPE_API_KEY", "sk_test_xxx")
        exporter = BillingExporter()
        assert exporter._api_key == "sk_test_xxx"
        invoice = exporter.export_invoice("acme", store.all_records())
        # stripe is not a dependency -> stub fallback
        assert invoice["mode"] == "stub"

    def test_explicit_api_key_param_overrides_env(self, store, no_stripe, monkeypatch):
        monkeypatch.setenv("STRIPE_API_KEY", "sk_env")
        # Explicit None => behaves like no key (stub).
        exporter = BillingExporter(api_key=None)
        assert exporter._api_key is None
        invoice = exporter.export_invoice("acme", store.all_records())
        assert invoice["mode"] == "stub"


class TestMeteringMiddlewareReusesCostTracker:
    """The middleware must ADD a metering step without re-inventing cost math."""

    def test_middleware_records_into_store_reusing_cost_tracker(self, store, no_stripe):
        # Build a fake ASGI app + request/response that mimic what
        # cost_middleware already attached (X-DistLLM-* headers).
        mw = MeteringMiddleware(app=None, store=store, enable=True)
        assert mw._tracker is get_cost_tracker()  # reuse singleton, not a new one

        req = _FakeRequest(tenant_id="acme", model="gpt-4o-mini",
                           path="/v1/chat/completions")
        resp = _FakeResponse(tokens_in=100, tokens_out=25)

        import asyncio
        asyncio.run(mw.dispatch(req, _make_call_next(resp)))

        recs = store.records_for_tenant("acme")
        assert len(recs) == 1
        r = recs[0]
        assert r.tokens_in == 100 and r.tokens_out == 25
        # cost/compute come from the reused CostTracker (not re-estimated here).
        assert r.cost_usd > 0
        assert r.compute_s > 0
        assert r.model_name == "gpt-4o-mini"

    def test_middleware_passthrough_when_disabled(self, store, no_stripe):
        mw = MeteringMiddleware(app=None, store=store, enable=False)
        req = _FakeRequest(tenant_id="acme", model="gpt-4o-mini",
                           path="/v1/chat/completions")
        resp = _FakeResponse(tokens_in=10, tokens_out=10)
        import asyncio
        asyncio.run(mw.dispatch(req, _make_call_next(resp)))
        # Disabled => no records written.
        assert store.all_records() == []

    def test_middleware_skips_non_inference_paths(self, store, no_stripe):
        mw = MeteringMiddleware(app=None, store=store, enable=True)
        req = _FakeRequest(tenant_id="acme", model="gpt-4o-mini", path="/health")
        resp = _FakeResponse(tokens_in=10, tokens_out=10)
        import asyncio
        asyncio.run(mw.dispatch(req, _make_call_next(resp)))
        assert store.all_records() == []


# ── Tiny fakes so we can exercise the middleware without a real server ──

class _FakeRequest:
    def __init__(self, tenant_id, model, path):
        class _State:
            pass
        self.state = _State()
        self.state.tenant_id = tenant_id
        self.state.api_key_id = tenant_id
        self.state.model = model
        self.state.request_id = "req-" + str(uuid4())

        class _Url:
            pass
        self.url = _Url()
        self.url.path = path


def uuid4():
    import uuid as _u
    return str(_u.uuid4())


class _FakeResponse:
    def __init__(self, tokens_in, tokens_out):
        # Mirror the headers cost_middleware attaches.
        self.headers = {
            "X-DistLLM-Tokens": f"{tokens_in}/{tokens_out}/{tokens_in + tokens_out}",
            "X-DistLLM-GPU-Time": "0.5",
        }
        # Cost header will be filled by reusing cost_tracker in middleware;
        # if absent the middleware falls back to the tracker estimate.

    @property
    def status_code(self):
        return 200


async def _call_next(response, request):
    return response


def _make_call_next(response):
    """Return an async call_next(request) that just yields `response`."""
    async def _inner(request):
        return response
    return _inner
