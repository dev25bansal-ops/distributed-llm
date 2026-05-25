"""Tests for tenant management API routes."""

import os
import time
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from distllm.tenant.store import TenantStore
from distllm.tenant.billing import UsageMeter
from distllm.tenant.models import TenantTier
from distllm.api.routes.tenants import router


@pytest.fixture
def app():
    app = FastAPI()
    store = TenantStore()
    store.create_tenant(name="Default", tier=TenantTier.FREE)
    app.state.tenant_store = store
    app.state.usage_meter = UsageMeter(store)
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    os.environ["ADMIN_API_KEY"] = "test-admin-key-12345"
    with TestClient(app) as c:
        yield c
    os.environ.pop("ADMIN_API_KEY", None)


def _auth_header():
    return {"Authorization": "Bearer test-admin-key-12345"}


class TestCreateTenant:
    def test_create_tenant_with_quota_override(self, client):
        resp = client.post("/v1/tenants", json={
            "name": "Quota Tenant",
            "tier": "free",
            "quota": {"max_rpm": 100, "max_tpm": 5000},
        }, headers=_auth_header())
        assert resp.status_code == 200
        data = resp.json()
        assert data["quota"]["max_rpm"] == 100
        assert data["quota"]["max_tpm"] == 5000
    def test_create_tenant_success(self, client):
        resp = client.post("/v1/tenants", json={"name": "Acme Corp", "tier": "business"}, headers=_auth_header())
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Acme Corp"
        assert data["tier"] == "business"
        assert data["tenant_id"].startswith("tnt_")
        assert data["api_key"].startswith("tnt_")

    def test_create_tenant_invalid_tier(self, client):
        resp = client.post("/v1/tenants", json={"name": "Bad", "tier": "ultra"}, headers=_auth_header())
        assert resp.status_code == 400

    def test_create_tenant_no_auth(self, client):
        resp = client.post("/v1/tenants", json={"name": "NoAuth"})
        assert resp.status_code == 401

    def test_create_tenant_wrong_auth(self, client):
        resp = client.post("/v1/tenants", json={"name": "Wrong"}, headers={"Authorization": "Bearer bad-key"})
        assert resp.status_code == 401


class TestListTenants:
    def test_list_tenants(self, client):
        resp = client.get("/v1/tenants", headers=_auth_header())
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert any(t["name"] == "Default" for t in data)

    def test_list_tenants_empty(self):
        app = FastAPI()
        app.state.tenant_store = TenantStore()
        from distllm.tenant.billing import UsageMeter
        app.state.usage_meter = UsageMeter(app.state.tenant_store)
        app.include_router(router)
        os.environ["ADMIN_API_KEY"] = "test-admin-key-12345"
        with TestClient(app) as c:
            resp = c.get("/v1/tenants", headers={"Authorization": "Bearer test-admin-key-12345"})
            assert resp.status_code == 200
            assert resp.json() == []
        os.environ.pop("ADMIN_API_KEY", None)

    def test_list_tenants_no_auth(self, client):
        resp = client.get("/v1/tenants")
        assert resp.status_code == 401


class TestGetTenant:
    def test_get_tenant_success(self, client):
        list_resp = client.get("/v1/tenants", headers=_auth_header())
        tenant_id = list_resp.json()[0]["tenant_id"]
        resp = client.get(f"/v1/tenants/{tenant_id}", headers=_auth_header())
        assert resp.status_code == 200
        assert resp.json()["tenant_id"] == tenant_id

    def test_get_tenant_not_found(self, client):
        resp = client.get("/v1/tenants/nonexistent", headers=_auth_header())
        assert resp.status_code == 404


class TestUpdateTenant:
    def test_update_tenant_name(self, client):
        list_resp = client.get("/v1/tenants", headers=_auth_header())
        tenant_id = list_resp.json()[0]["tenant_id"]
        resp = client.put(f"/v1/tenants/{tenant_id}", json={"name": "Updated Corp"}, headers=_auth_header())
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_update_tenant_tier(self, client):
        list_resp = client.get("/v1/tenants", headers=_auth_header())
        tenant_id = list_resp.json()[0]["tenant_id"]
        resp = client.put(f"/v1/tenants/{tenant_id}", json={"tier": "enterprise"}, headers=_auth_header())
        assert resp.status_code == 200

    def test_update_tenant_invalid_tier(self, client):
        list_resp = client.get("/v1/tenants", headers=_auth_header())
        tenant_id = list_resp.json()[0]["tenant_id"]
        resp = client.put(f"/v1/tenants/{tenant_id}", json={"tier": "mega"}, headers=_auth_header())
        assert resp.status_code == 400

    def test_update_tenant_is_active(self, client):
        list_resp = client.get("/v1/tenants", headers=_auth_header())
        tenant_id = list_resp.json()[0]["tenant_id"]
        resp = client.put(f"/v1/tenants/{tenant_id}", json={"is_active": False}, headers=_auth_header())
        assert resp.status_code == 200

    def test_update_tenant_not_found(self, client):
        resp = client.put("/v1/tenants/nonexistent", json={"name": "Nope"}, headers=_auth_header())
        assert resp.status_code == 404


class TestDeleteTenant:
    def test_delete_tenant_success(self, client):
        resp = client.post("/v1/tenants", json={"name": "Delete Me", "tier": "free"}, headers=_auth_header())
        tenant_id = resp.json()["tenant_id"]
        del_resp = client.delete(f"/v1/tenants/{tenant_id}", headers=_auth_header())
        assert del_resp.status_code == 200
        get_resp = client.get(f"/v1/tenants/{tenant_id}", headers=_auth_header())
        assert get_resp.status_code == 404

    def test_delete_tenant_not_found(self, client):
        resp = client.delete("/v1/tenants/nonexistent", headers=_auth_header())
        assert resp.status_code == 404


class TestRegenerateKey:
    def test_regenerate_key_success(self, client):
        resp = client.post("/v1/tenants", json={"name": "Key Test", "tier": "starter"}, headers=_auth_header())
        tenant_id = resp.json()["tenant_id"]
        old_key = resp.json()["api_key"]
        regen_resp = client.post(f"/v1/tenants/{tenant_id}/regenerate-key", headers=_auth_header())
        assert regen_resp.status_code == 200
        assert regen_resp.json()["api_key"] != old_key

    def test_regenerate_key_not_found(self, client):
        resp = client.post("/v1/tenants/nonexistent/regenerate-key", headers=_auth_header())
        assert resp.status_code == 404


class TestUsage:
    def test_get_usage_report(self, client):
        store = TenantStore()
        tenant = store.create_tenant(name="Usage Test", tier=TenantTier.STARTER)
        meter = UsageMeter(store)
        meter.record(tenant.tenant_id, input_tokens=100, output_tokens=50, model="default", endpoint="/chat")
        report = meter.get_report(tenant.tenant_id)
        assert report.total_requests == 1
        assert report.total_input_tokens == 100
        assert report.total_output_tokens == 50

    def test_get_usage_report_via_http(self, client):
        resp = client.post("/v1/tenants", json={"name": "Usage HTTP", "tier": "starter"}, headers=_auth_header())
        tenant_id = resp.json()["tenant_id"]
        store = client.app.state.tenant_store
        meter = client.app.state.usage_meter
        meter.record(tenant_id, input_tokens=200, output_tokens=75, model="default", endpoint="/chat")
        resp = client.get(f"/v1/tenants/{tenant_id}/usage", headers=_auth_header())
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_requests"] == 1
        assert data["total_input_tokens"] == 200

    def test_get_usage_report_no_auth(self, client):
        resp = client.post("/v1/tenants", json={"name": "Usage NoAuth", "tier": "free"}, headers=_auth_header())
        tenant_id = resp.json()["tenant_id"]
        resp = client.get(f"/v1/tenants/{tenant_id}/usage")
        assert resp.status_code == 401

    def test_get_usage_report_not_found(self, client):
        resp = client.get("/v1/tenants/nonexistent/usage", headers=_auth_header())
        assert resp.status_code == 404

    def test_get_usage_with_since_param(self, client):
        resp = client.post("/v1/tenants", json={"name": "Usage Since", "tier": "business"}, headers=_auth_header())
        tenant_id = resp.json()["tenant_id"]
        store = client.app.state.tenant_store
        meter = client.app.state.usage_meter
        meter.record(tenant_id, input_tokens=50, output_tokens=25)
        resp = client.get(f"/v1/tenants/{tenant_id}/usage?since=0", headers=_auth_header())
        assert resp.status_code == 200
        assert resp.json()["total_requests"] >= 1

    def test_get_billing_with_since_param(self, client):
        list_resp = client.get("/v1/tenants", headers=_auth_header())
        tenant_id = list_resp.json()[0]["tenant_id"]
        resp = client.get(f"/v1/tenants/{tenant_id}/billing?since=0", headers=_auth_header())
        assert resp.status_code == 200
        assert "summary" in resp.json()

    def test_live_snapshot(self, client):
        store = TenantStore()
        tenant = store.create_tenant(name="Live Test", tier=TenantTier.FREE)
        meter = UsageMeter(store)
        meter.record(tenant.tenant_id, input_tokens=10, output_tokens=5)
        snapshot = meter.get_live_snapshot(tenant.tenant_id, window_seconds=60)
        assert snapshot["requests_1m"] == 1
        assert snapshot["input_tokens_1m"] == 10


class TestBilling:
    def test_get_billing(self, client):
        list_resp = client.get("/v1/tenants", headers=_auth_header())
        tenant_id = list_resp.json()[0]["tenant_id"]
        resp = client.get(f"/v1/tenants/{tenant_id}/billing", headers=_auth_header())
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data
        assert "cost_breakdown" in data

    def test_get_billing_not_found(self, client):
        resp = client.get("/v1/tenants/nonexistent/billing", headers=_auth_header())
        assert resp.status_code == 404

    def test_get_billing_no_auth(self, client):
        list_resp = client.get("/v1/tenants", headers=_auth_header())
        tenant_id = list_resp.json()[0]["tenant_id"]
        resp = client.get(f"/v1/tenants/{tenant_id}/billing")
        assert resp.status_code == 401


class TestLiveUsage:
    def test_get_live_usage_public(self, client):
        resp = client.post("/v1/tenants", json={"name": "Live", "tier": "free"}, headers=_auth_header())
        tenant_id = resp.json()["tenant_id"]
        resp = client.get(f"/v1/tenants/{tenant_id}/live")
        assert resp.status_code == 200

    def test_get_live_usage_response_shape(self, client):
        resp = client.post("/v1/tenants", json={"name": "Live Shape", "tier": "free"}, headers=_auth_header())
        tenant_id = resp.json()["tenant_id"]
        meter = client.app.state.usage_meter
        meter.record(tenant_id, input_tokens=10, output_tokens=5)
        resp = client.get(f"/v1/tenants/{tenant_id}/live")
        assert resp.status_code == 200
        data = resp.json()
        assert "requests_1m" in data
        assert "input_tokens_1m" in data
        assert "output_tokens_1m" in data
        assert "cost_1m" in data

    def test_get_live_usage_with_window(self, client):
        resp = client.post("/v1/tenants", json={"name": "Live Window", "tier": "free"}, headers=_auth_header())
        tenant_id = resp.json()["tenant_id"]
        resp = client.get(f"/v1/tenants/{tenant_id}/live?window=30")
        assert resp.status_code == 200

    def test_get_live_usage_not_found(self, client):
        resp = client.get("/v1/tenants/nonexistent/live")
        assert resp.status_code == 404


class TestAuthorization:
    def test_no_admin_key_configured(self):
        os.environ.pop("ADMIN_API_KEY", None)
        app = FastAPI()
        app.state.tenant_store = TenantStore()
        from distllm.tenant.billing import UsageMeter
        app.state.usage_meter = UsageMeter(app.state.tenant_store)
        app.include_router(router)
        with TestClient(app) as c:
            resp = c.get("/v1/tenants", headers={"Authorization": "Bearer any-key"})
            assert resp.status_code == 503

    def test_malformed_auth_header(self, client):
        resp = client.get("/v1/tenants", headers={"Authorization": "NotBearer something"})
        assert resp.status_code == 401

    def test_empty_auth_header(self, client):
        resp = client.get("/v1/tenants", headers={"Authorization": ""})
        assert resp.status_code == 401
