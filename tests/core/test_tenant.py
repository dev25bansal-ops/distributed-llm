"""Tests: TenantStore — CRUD, API key resolution/regeneration, concurrent access, usage recording/reporting."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from distllm.tenant.store import TenantStore
from distllm.tenant.models import Tenant, TenantTier, TenantUsageRecord, ResourceQuota


# ===========================================================================
# TenantStore — CRUD
# ===========================================================================


class TestTenantStoreCRUD:
    def test_create_tenant(self):
        store = TenantStore()
        tenant = store.create_tenant("test-tenant", tier=TenantTier.STARTER)
        assert tenant.name == "test-tenant"
        assert tenant.tier == TenantTier.STARTER
        assert tenant.tenant_id.startswith("tnt_")
        assert tenant.is_active is True
        assert tenant.api_key != ""

    def test_get_tenant(self):
        store = TenantStore()
        created = store.create_tenant("get-test")
        fetched = store.get_tenant(created.tenant_id)
        assert fetched is not None
        assert fetched.name == "get-test"
        assert fetched.tenant_id == created.tenant_id

    def test_get_tenant_not_found(self):
        store = TenantStore()
        assert store.get_tenant("nonexistent") is None

    def test_update_tenant_name(self):
        store = TenantStore()
        t = store.create_tenant("original-name")
        updated = store.update_tenant(t.tenant_id, name="new-name")
        assert updated is not None
        assert updated.name == "new-name"
        fetched = store.get_tenant(t.tenant_id)
        assert fetched.name == "new-name"

    def test_update_tenant_tier(self):
        store = TenantStore()
        t = store.create_tenant("tier-test")
        updated = store.update_tenant(t.tenant_id, tier=TenantTier.BUSINESS)
        assert updated.tier == TenantTier.BUSINESS

    def test_update_tenant_deactivate(self):
        store = TenantStore()
        t = store.create_tenant("active-tenant")
        updated = store.update_tenant(t.tenant_id, is_active=False)
        assert updated.is_active is False

    def test_update_tenant_not_found(self):
        store = TenantStore()
        assert store.update_tenant("nonexistent", name="x") is None

    def test_delete_tenant(self):
        store = TenantStore()
        t = store.create_tenant("delete-me")
        assert store.delete_tenant(t.tenant_id) is True
        assert store.get_tenant(t.tenant_id) is None

    def test_delete_tenant_not_found(self):
        store = TenantStore()
        assert store.delete_tenant("nonexistent") is False

    def test_list_tenants(self):
        store = TenantStore()
        store.create_tenant("a")
        store.create_tenant("b")
        store.create_tenant("c")
        tenants = store.list_tenants()
        assert len(tenants) == 3

    def test_create_tenant_with_custom_quota(self):
        store = TenantStore()
        quota = ResourceQuota(max_rpm=200, max_tpm=50000)
        t = store.create_tenant("quota-test", tier=TenantTier.BUSINESS, quota=quota)
        assert t.quota.max_rpm == 200
        assert t.quota.max_tpm == 50000


# ===========================================================================
# TenantStore — API key resolution
# ===========================================================================


class TestTenantStoreApiKey:
    def test_get_tenant_by_api_key(self):
        store = TenantStore()
        t = store.create_tenant("key-test")
        found = store.get_tenant_by_api_key(t.api_key)
        assert found is not None
        assert found.tenant_id == t.tenant_id

    def test_get_tenant_by_api_key_invalid(self):
        store = TenantStore()
        assert store.get_tenant_by_api_key("bad-key") is None

    def test_get_tenant_id_by_api_key(self):
        store = TenantStore()
        t = store.create_tenant("id-test")
        tid = store.get_tenant_id_by_api_key(t.api_key)
        assert tid == t.tenant_id

    def test_get_tenant_id_by_api_key_invalid(self):
        store = TenantStore()
        assert store.get_tenant_id_by_api_key("bad") is None

    def test_api_key_unique_per_tenant(self):
        store = TenantStore()
        t1 = store.create_tenant("t1")
        t2 = store.create_tenant("t2")
        assert t1.api_key != t2.api_key


# ===========================================================================
# TenantStore — API key regeneration
# ===========================================================================


class TestTenantStoreRegenerateKey:
    def test_regenerate_returns_new_key(self):
        store = TenantStore()
        t = store.create_tenant("regen-test")
        old_key = t.api_key
        new_key = store.regenerate_api_key(t.tenant_id)
        assert new_key is not None
        assert new_key != old_key

    def test_old_key_invalidated(self):
        store = TenantStore()
        t = store.create_tenant("invalidation-test")
        old_key = t.api_key
        store.regenerate_api_key(t.tenant_id)
        assert store.get_tenant_by_api_key(old_key) is None

    def test_new_key_works(self):
        store = TenantStore()
        t = store.create_tenant("new-key-test")
        new_key = store.regenerate_api_key(t.tenant_id)
        found = store.get_tenant_by_api_key(new_key)
        assert found is not None
        assert found.tenant_id == t.tenant_id

    def test_regenerate_not_found(self):
        store = TenantStore()
        assert store.regenerate_api_key("nonexistent") is None


# ===========================================================================
# TenantStore — concurrent access
# ===========================================================================


class TestTenantStoreConcurrent:
    def test_concurrent_create(self):
        store = TenantStore()
        errors = []
        def create(n):
            try:
                store.create_tenant(f"concurrent-{n}")
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=create, args=(i,)) for i in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len(errors) == 0
        assert len(store.list_tenants()) == 20

    def test_concurrent_update_same_tenant(self):
        store = TenantStore()
        t = store.create_tenant("concurrent-update")
        errors = []
        def set_inactive():
            try:
                store.update_tenant(t.tenant_id, is_active=False)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=set_inactive) for _ in range(10)]
        for th in threads: th.start()
        for th in threads: th.join()
        assert len(errors) == 0
        fetched = store.get_tenant(t.tenant_id)
        assert fetched is not None
        assert fetched.is_active is False

    def test_concurrent_regenerate_key(self):
        store = TenantStore()
        t = store.create_tenant("concurrent-regen")
        keys = []
        errors = []
        def regen():
            try:
                k = store.regenerate_api_key(t.tenant_id)
                if k:
                    keys.append(k)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=regen) for _ in range(5)]
        for th in threads: th.start()
        for th in threads: th.join()
        assert len(errors) == 0
        assert len(keys) >= 1


# ===========================================================================
# TenantStore — usage recording
# ===========================================================================


class TestTenantStoreUsageRecording:
    def test_record_usage(self):
        store = TenantStore()
        t = store.create_tenant("usage-test")
        record = TenantUsageRecord(tenant_id=t.tenant_id, input_tokens=50, output_tokens=100,
                                   requests=1, model="default", endpoint="/v1/chat/completions")
        store.record_usage(record)
        report = store.get_usage_report(t.tenant_id)
        assert report.total_requests == 1
        assert report.total_input_tokens == 50
        assert report.total_output_tokens == 100

    def test_record_multiple_usage(self):
        store = TenantStore()
        t = store.create_tenant("multi-usage")
        for i in range(5):
            store.record_usage(TenantUsageRecord(
                tenant_id=t.tenant_id, input_tokens=10, output_tokens=20,
                requests=1, model="default",
            ))
        report = store.get_usage_report(t.tenant_id)
        assert report.total_requests == 5
        assert report.total_input_tokens == 50
        assert report.total_output_tokens == 100

    def test_record_usage_other_tenant(self):
        store = TenantStore()
        t1 = store.create_tenant("t1")
        t2 = store.create_tenant("t2")
        store.record_usage(TenantUsageRecord(tenant_id=t1.tenant_id, requests=1))
        store.record_usage(TenantUsageRecord(tenant_id=t2.tenant_id, requests=3))
        r1 = store.get_usage_report(t1.tenant_id)
        r2 = store.get_usage_report(t2.tenant_id)
        assert r1.total_requests == 1
        assert r2.total_requests == 3

    def test_usage_report_not_found(self):
        store = TenantStore()
        with pytest.raises(ValueError):
            store.get_usage_report("nonexistent")

    def test_usage_report_since_filter(self):
        store = TenantStore()
        t = store.create_tenant("since-test")
        store.record_usage(TenantUsageRecord(
            tenant_id=t.tenant_id, input_tokens=10, requests=1, timestamp=time.time() - 100,
        ))
        store.record_usage(TenantUsageRecord(
            tenant_id=t.tenant_id, input_tokens=20, requests=1, timestamp=time.time(),
        ))
        recent = store.get_usage_report(t.tenant_id, since=time.time() - 50)
        assert recent.total_input_tokens == 20
        assert recent.total_requests == 1


# ===========================================================================
# TenantStore — usage report aggregation
# ===========================================================================


class TestTenantStoreUsageAggregation:
    def test_aggregation_correct_totals(self):
        store = TenantStore()
        t = store.create_tenant("aggregation-test")
        for i in range(10):
            store.record_usage(TenantUsageRecord(
                tenant_id=t.tenant_id, input_tokens=100, output_tokens=50,
                requests=2, model="default", latency_ms=30.0, cost=0.001,
            ))
        report = store.get_usage_report(t.tenant_id)
        assert report.total_requests == 20
        assert report.total_input_tokens == 1000
        assert report.total_output_tokens == 500
        assert report.total_cost == pytest.approx(0.01, rel=1e-6)
        assert report.avg_latency_ms == pytest.approx(30.0, rel=0.1)

    def test_endpoint_breakdown(self):
        store = TenantStore()
        t = store.create_tenant("endpoint-test")
        store.record_usage(TenantUsageRecord(
            tenant_id=t.tenant_id, requests=3, endpoint="/v1/chat/completions",
        ))
        store.record_usage(TenantUsageRecord(
            tenant_id=t.tenant_id, requests=1, endpoint="/v1/completions",
        ))
        report = store.get_usage_report(t.tenant_id)
        assert "/v1/chat/completions" in report.endpoint_breakdown
        assert "/v1/completions" in report.endpoint_breakdown
        assert report.endpoint_breakdown["/v1/chat/completions"]["requests"] == 3
        assert report.endpoint_breakdown["/v1/completions"]["requests"] == 1

    def test_usage_report_no_data(self):
        store = TenantStore()
        t = store.create_tenant("no-data")
        report = store.get_usage_report(t.tenant_id)
        assert report.total_requests == 0
        assert report.total_input_tokens == 0
        assert report.total_output_tokens == 0
        assert report.model_breakdown == {}
        assert report.endpoint_breakdown == {}

    def test_max_usage_records_trim(self):
        store = TenantStore()
        store._max_usage_records = 10
        t = store.create_tenant("trim-test")
        for i in range(20):
            store.record_usage(TenantUsageRecord(
                tenant_id=t.tenant_id, requests=1, input_tokens=i,
            ))
        assert len(store._usage) <= 10

    def test_model_breakdown(self):
        store = TenantStore()
        t = store.create_tenant("breakdown")
        store.record_usage(TenantUsageRecord(
            tenant_id=t.tenant_id, input_tokens=50, output_tokens=30, requests=2,
            model="premium", cost=0.005,
        ))
        store.record_usage(TenantUsageRecord(
            tenant_id=t.tenant_id, input_tokens=10, output_tokens=5, requests=1,
            model="fast", cost=0.002,
        ))
        report = store.get_usage_report(t.tenant_id)
        assert "premium" in report.model_breakdown
        assert "fast" in report.model_breakdown
        assert report.model_breakdown["premium"]["requests"] == 2
        assert report.model_breakdown["premium"]["input_tokens"] == 50
        assert report.model_breakdown["fast"]["requests"] == 1


# ===========================================================================
# UsageMeter
# ===========================================================================


class TestUsageMeter:
    def test_record_creates_usage_record(self):
        from distllm.tenant.billing import UsageMeter
        store = TenantStore()
        meter = UsageMeter(store)
        t = store.create_tenant("meter-test")
        record = meter.record(t.tenant_id, input_tokens=100, output_tokens=50,
                              model="default", endpoint="/v1/chat", latency_ms=45.0)
        assert record.input_tokens == 100
        assert record.output_tokens == 50
        assert record.cost > 0

    def test_live_snapshot(self):
        from distllm.tenant.billing import UsageMeter
        store = TenantStore()
        meter = UsageMeter(store)
        t = store.create_tenant("snapshot-test")
        meter.record(t.tenant_id, input_tokens=10, output_tokens=5)
        snap = meter.get_live_snapshot(t.tenant_id)
        assert isinstance(snap, dict)


# ===========================================================================
# TenantMiddleware
# ===========================================================================


class TestTenantMiddlewareDisabled:
    def test_disabled_mode_bypasses_auth(self):
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from distllm.tenant.middleware import TenantMiddleware
        from distllm.tenant.store import TenantStore

        app = FastAPI()
        store = TenantStore()
        store.create_tenant("test-tenant")
        app.add_middleware(TenantMiddleware, store=store, enabled=False)

        @app.get("/test")
        async def handler(request: Request):
            return {"tenant_id": request.state.tenant_id}

        with TestClient(app) as client:
            resp = client.get("/test")
            assert resp.status_code == 200
            assert resp.json()["tenant_id"] == "default"


class TestTenantMiddlewareHeaderResolution:
    def test_x_tenant_id_header(self):
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from distllm.tenant.middleware import TenantMiddleware
        from distllm.tenant.store import TenantStore

        app = FastAPI()
        store = TenantStore()
        t = store.create_tenant("header-test")
        app.add_middleware(TenantMiddleware, store=store, enabled=True)

        @app.get("/test")
        async def handler(request: Request):
            return {"tenant_id": request.state.tenant_id}

        with TestClient(app) as client:
            resp = client.get("/test", headers={"X-Tenant-ID": t.tenant_id})
            assert resp.status_code == 200
            assert resp.json()["tenant_id"] == t.tenant_id


class TestTenantMiddlewareApiKeyResolution:
    def test_x_tenant_api_key_header(self):
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from distllm.tenant.middleware import TenantMiddleware
        from distllm.tenant.store import TenantStore

        app = FastAPI()
        store = TenantStore()
        t = store.create_tenant("api-key-test")
        app.add_middleware(TenantMiddleware, store=store, enabled=True)

        @app.get("/test")
        async def handler(request: Request):
            return {"tenant_id": request.state.tenant_id}

        with TestClient(app) as client:
            resp = client.get("/test", headers={"X-Tenant-API-Key": t.api_key})
            assert resp.status_code == 200
            assert resp.json()["tenant_id"] == t.tenant_id


class TestTenantMiddlewareBearerToken:
    def test_bearer_token_with_tnt_prefix(self):
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from distllm.tenant.middleware import TenantMiddleware
        from distllm.tenant.store import TenantStore

        app = FastAPI()
        store = TenantStore()
        t = store.create_tenant("bearer-test")
        app.add_middleware(TenantMiddleware, store=store, enabled=True)

        @app.get("/test")
        async def handler(request: Request):
            return {"tenant_id": request.state.tenant_id}

        with TestClient(app) as client:
            resp = client.get("/test", headers={"Authorization": f"Bearer {t.api_key}"})
            assert resp.status_code == 200
            assert resp.json()["tenant_id"] == t.tenant_id

    def test_bearer_non_tnt_token_returns_401(self):
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from distllm.tenant.middleware import TenantMiddleware
        from distllm.tenant.store import TenantStore

        app = FastAPI()
        store = TenantStore()
        app.add_middleware(TenantMiddleware, store=store, enabled=True)

        @app.get("/test")
        async def handler(request: Request):
            return {"ok": True}

        with TestClient(app) as client:
            resp = client.get("/test", headers={"Authorization": "Bearer some-other-token"})
            assert resp.status_code == 401


class TestTenantMiddlewareInactive:
    def test_inactive_tenant_returns_403(self):
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from distllm.tenant.middleware import TenantMiddleware
        from distllm.tenant.store import TenantStore

        app = FastAPI()
        store = TenantStore()
        t = store.create_tenant("inactive-tenant")
        store.update_tenant(t.tenant_id, is_active=False)
        app.add_middleware(TenantMiddleware, store=store, enabled=True)

        @app.get("/test")
        async def handler(request: Request):
            return {"ok": True}

        with TestClient(app) as client:
            resp = client.get("/test", headers={"X-Tenant-ID": t.tenant_id})
            assert resp.status_code == 403


class TestTenantMiddlewareConcurrentLimit:
    def test_concurrent_limit_exceeded_returns_429(self):
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from distllm.tenant.middleware import TenantMiddleware
        from distllm.tenant.store import TenantStore
        from distllm.tenant.models import ResourceQuota

        app = FastAPI()
        store = TenantStore()
        t = store.create_tenant("concurrent-test", quota=ResourceQuota(max_concurrent_requests=1))
        app.add_middleware(TenantMiddleware, store=store, enabled=True)

        @app.get("/slow")
        async def slow(request: Request):
            import asyncio
            await asyncio.sleep(0.1)
            return {"ok": True}

        with TestClient(app) as client:
            import threading
            results = []
            def hit():
                r = client.get("/slow", headers={"X-Tenant-ID": t.tenant_id})
                results.append(r.status_code)
            threads = [threading.Thread(target=hit) for _ in range(3)]
            for th in threads: th.start()
            for th in threads: th.join()
            assert 429 in results


class TestTenantMiddlewareCleanupAfterError:
    def test_concurrent_counter_decremented_after_error(self):
        from distllm.tenant.middleware import TenantMiddleware
        store = TenantStore()
        t = store.create_tenant("error-cleanup")
        middleware = TenantMiddleware(MagicMock(), store=store, enabled=True)
        middleware._concurrent[t.tenant_id] = 1
        middleware._concurrent[t.tenant_id] = max(0, middleware._concurrent[t.tenant_id] - 1)
        assert middleware._concurrent[t.tenant_id] == 0


# ===========================================================================
# TenantRateLimiter
# ===========================================================================


class TestTenantRateLimiter:
    def test_check_request_exhausts_bucket(self):
        from distllm.tenant.rate_limiter import TenantRateLimiter
        from distllm.tenant.models import ResourceQuota
        rl = TenantRateLimiter()
        quota = ResourceQuota(max_rpm=3)
        burst = int(3 * 1.5)
        for _ in range(burst):
            assert rl.check_request("t1", quota) is True
        assert rl.check_request("t1", quota) is False

    def test_token_bucket_default_burst(self):
        from distllm.tenant.rate_limiter import TokenBucket
        tb = TokenBucket(3)
        assert tb.max_tokens == int(3 * 1.5)

    def test_check_tokens_consumes_multiple(self):
        from distllm.tenant.rate_limiter import TenantRateLimiter
        from distllm.tenant.models import ResourceQuota
        rl = TenantRateLimiter()
        quota = ResourceQuota(max_tpm=3)
        burst = int(3 * 1.5)
        for _ in range(burst):
            assert rl.check_tokens("t2", quota, estimated_tokens=1) is True
        assert rl.check_tokens("t2", quota, estimated_tokens=1) is False

    def test_retry_after_computed(self):
        from distllm.tenant.rate_limiter import TenantRateLimiter
        from distllm.tenant.models import ResourceQuota
        rl = TenantRateLimiter()
        quota = ResourceQuota(max_rpm=1)
        rl.check_request("t3", quota)
        limits = rl.get_limits("t3", quota)
        assert limits["retry_after_seconds"] > 0

    def test_reset_tenant_clears_buckets(self):
        from distllm.tenant.rate_limiter import TenantRateLimiter
        from distllm.tenant.models import ResourceQuota
        rl = TenantRateLimiter()
        quota = ResourceQuota(max_rpm=3)
        burst = int(3 * 1.5)
        for _ in range(burst):
            rl.check_request("t4", quota)
        assert rl.check_request("t4", quota) is False
        rl.reset_tenant("t4")
        assert rl.check_request("t4", quota) is True

    def test_get_limits_returns_all_fields(self):
        from distllm.tenant.rate_limiter import TenantRateLimiter
        from distllm.tenant.models import ResourceQuota
        rl = TenantRateLimiter()
        quota = ResourceQuota(max_rpm=10, max_tpm=1000)
        limits = rl.get_limits("t5", quota)
        assert limits["rpm_limit"] == 10
        assert limits["tpm_limit"] == 1000


# ===========================================================================
# UsageMeter — cost calculation
# ===========================================================================


class TestUsageMeterCostCalculation:
    def test_default_model_cost(self):
        from distllm.tenant.billing import UsageMeter, _lookup_model_cost
        store = TenantStore()
        meter = UsageMeter(store)
        t = store.create_tenant("cost-test")
        record = meter.record(t.tenant_id, input_tokens=1000, output_tokens=500,
                              model="default")
        expected = (1000 / 1000) * 0.001 + (500 / 1000) * 0.002
        assert record.cost == pytest.approx(expected)

    def test_premium_model_cost(self):
        from distllm.tenant.billing import UsageMeter
        store = TenantStore()
        meter = UsageMeter(store)
        t = store.create_tenant("premium-cost")
        record = meter.record(t.tenant_id, input_tokens=2000, output_tokens=1000,
                              model="premium")
        expected = (2000 / 1000) * 0.005 + (1000 / 1000) * 0.010
        assert record.cost == pytest.approx(expected)

    def test_enterprise_model_cost(self):
        from distllm.tenant.billing import UsageMeter
        store = TenantStore()
        meter = UsageMeter(store)
        t = store.create_tenant("enterprise-cost")
        record = meter.record(t.tenant_id, input_tokens=5000, output_tokens=3000,
                              model="enterprise")
        expected = (5000 / 1000) * 0.010 + (3000 / 1000) * 0.020
        assert record.cost == pytest.approx(expected)

    def test_lookup_model_cost(self):
        from distllm.tenant.billing import _lookup_model_cost
        inp, out = _lookup_model_cost("premium")
        assert inp == 0.005
        assert out == 0.010

    def test_lookup_model_cost_unknown_defaults(self):
        from distllm.tenant.billing import _lookup_model_cost
        inp, out = _lookup_model_cost("nonexistent")
        assert inp == 0.001
        assert out == 0.002

    def test_billing_report_generation(self):
        from distllm.tenant.billing import UsageMeter, BillingReport
        store = TenantStore()
        meter = UsageMeter(store)
        report_gen = BillingReport(meter)
        t = store.create_tenant("billing-test")
        meter.record(t.tenant_id, input_tokens=100, output_tokens=50, model="default")
        report = report_gen.generate_report(t.tenant_id, since=0)
        assert "summary" in report
        assert "cost_breakdown" in report
        assert report["summary"]["total_requests"] >= 1


# ===========================================================================
# UsageMeter — record flow
# ===========================================================================


class TestUsageMeterRecordFlow:
    def test_record_appends_to_store(self):
        from distllm.tenant.billing import UsageMeter
        store = TenantStore()
        meter = UsageMeter(store)
        t = store.create_tenant("flow-test")
        rec = meter.record(t.tenant_id, input_tokens=10, output_tokens=5)
        assert len(store._usage) == 1
        assert store._usage[0].tenant_id == t.tenant_id
        assert store._usage[0].input_tokens == 10


# ===========================================================================
# BillingReport — format
# ===========================================================================


class TestBillingReportFormat:
    def test_summary_breakdown_structure(self):
        from distllm.tenant.billing import UsageMeter, BillingReport
        store = TenantStore()
        meter = UsageMeter(store)
        gen = BillingReport(meter)
        t = store.create_tenant("format-test")
        meter.record(t.tenant_id, input_tokens=50, output_tokens=25, model="default", endpoint="/v1/chat")
        report = gen.generate_report(t.tenant_id, since=0)
        assert set(report.keys()) == {"tenant_id", "tier", "period_start", "period_end", "summary", "cost_breakdown"}
        assert set(report["summary"].keys()) == {"total_requests", "total_input_tokens", "total_output_tokens", "total_cost", "avg_latency_ms"}
        assert "by_model" in report["cost_breakdown"]
        assert "by_endpoint" in report["cost_breakdown"]


# ===========================================================================
# TenantModelRouter
# ===========================================================================


class TestTenantModelRouterTierAccess:
    def test_enterprise_can_access_all(self):
        from distllm.tenant.router import TenantModelRouter
        from distllm.tenant.models import Tenant, TenantTier
        tenant = Tenant(tenant_id="", name="ent", tier=TenantTier.ENTERPRISE, quota=ResourceQuota(allowed_models=["default", "fast", "premium", "enterprise"]))
        router = TenantModelRouter()
        models = router.get_available_models(tenant)
        assert "enterprise" in models
        assert "default" in models

    def test_free_cannot_access_premium(self):
        from distllm.tenant.router import TenantModelRouter
        from distllm.tenant.models import Tenant, TenantTier
        tenant = Tenant(tenant_id="", name="free", tier=TenantTier.FREE, quota=ResourceQuota(allowed_models=["default"]))
        router = TenantModelRouter()
        models = router.get_available_models(tenant)
        assert "default" in models
        assert "premium" not in models

    def test_get_tier_for_model(self):
        from distllm.tenant.router import TenantModelRouter
        router = TenantModelRouter()
        tiers = router.get_tier_for_model("default")
        assert "free" in tiers
        tiers = router.get_tier_for_model("enterprise")
        assert "enterprise" in tiers

    def test_get_tier_for_unknown_model(self):
        from distllm.tenant.router import TenantModelRouter
        router = TenantModelRouter()
        tiers = router.get_tier_for_model("unknown")
        assert tiers == ["free"]


class TestTenantModelRouterFallback:
    def test_unallowed_model_falls_back_to_default(self):
        from distllm.tenant.router import TenantModelRouter
        from distllm.tenant.models import Tenant, TenantTier
        tenant = Tenant(tenant_id="", name="t1", tier=TenantTier.FREE, quota=ResourceQuota(allowed_models=["default"]))
        router = TenantModelRouter()
        model = router.resolve_model("premium", tenant)
        assert model == "default"

    def test_allowed_model_returns_requested(self):
        from distllm.tenant.router import TenantModelRouter
        from distllm.tenant.models import Tenant, TenantTier
        tenant = Tenant(tenant_id="", name="t2", tier=TenantTier.BUSINESS, quota=ResourceQuota(allowed_models=["default", "fast"]))
        router = TenantModelRouter()
        model = router.resolve_model("fast", tenant)
        assert model == "fast"


class TestTenantModelRouterNoTenant:
    def test_none_tenant_gets_default(self):
        from distllm.tenant.router import TenantModelRouter
        router = TenantModelRouter()
        models = router.get_available_models(None)
        assert models == ["default"]

    def test_none_tenant_request_unknown_falls_back(self):
        from distllm.tenant.router import TenantModelRouter
        router = TenantModelRouter()
        model = router.resolve_model("premium", None)
        assert model == "default"


# ===========================================================================
# Tenant — post_init and tier quotas
# ===========================================================================


class TestTenantPostInit:
    def test_tenant_id_generated(self):
        from distllm.tenant.models import Tenant
        t = Tenant(tenant_id="", name="auto-id")
        assert t.tenant_id.startswith("tnt_")
        assert len(t.tenant_id) > 4

    def test_tier_default_free(self):
        from distllm.tenant.models import Tenant
        t = Tenant(tenant_id="", name="tier-test")
        assert t.tier == TenantTier.FREE

    def test_quota_merged_with_tier_defaults(self):
        from distllm.tenant.models import Tenant
        t = Tenant(tenant_id="", name="q-merge", tier=TenantTier.BUSINESS)
        assert t.quota.max_rpm == 300
        assert t.quota.max_concurrent_requests == 20

    def test_custom_quota_overrides_tier(self):
        from distllm.tenant.models import Tenant, ResourceQuota
        t = Tenant(tenant_id="", name="custom", tier=TenantTier.FREE, quota=ResourceQuota(max_rpm=999))
        assert t.quota.max_rpm == 999


class TestTenantTierQuotas:
    def test_free_tier_limits(self):
        from distllm.tenant.models import TIER_QUOTAS, TenantTier
        q = TIER_QUOTAS[TenantTier.FREE]
        assert q.max_rpm == 10

    def test_starter_tier_limits(self):
        from distllm.tenant.models import TIER_QUOTAS, TenantTier
        q = TIER_QUOTAS[TenantTier.STARTER]
        assert q.max_rpm == 60
        assert q.kv_cache_size_mb == 512

    def test_business_tier_limits(self):
        from distllm.tenant.models import TIER_QUOTAS, TenantTier
        q = TIER_QUOTAS[TenantTier.BUSINESS]
        assert q.max_rpm == 300
        assert q.max_concurrent_requests == 20
        assert q.kv_cache_size_mb == 2048

    def test_enterprise_tier_highest_limits(self):
        from distllm.tenant.models import TIER_QUOTAS, TenantTier
        q = TIER_QUOTAS[TenantTier.ENTERPRISE]
        assert q.max_rpm == 3000
        assert q.kv_cache_size_mb == 8192
        assert q.max_concurrent_requests == 100
