"""Regression test for audit finding F-011.

AzureAvailabilityChecker.check_availability used to POST to
``https://management.azure.com/subscriptions/{subscriptionId}/providers/...``
with LITERAL braces (not an f-string), no bearer token, and a bare
``except`` that failed OPEN (``available = instance_type in
_AZURE_GPU_INSTANCES``, always True for known SKUs).  The scheduler could
therefore believe any known SKU was available in ANY region, including
regions that do not exist.

The fix:
  1. Interpolates the real subscription id (from ``AZURE_SUBSCRIPTION_ID``
     or the constructor) and the region into the ARM URL.
  2. Attaches a bearer token from ``DefaultAzureCredential``.
  3. Fails CLOSED on every error path (missing subscription id, missing
     credential, HTTP error, malformed response): ``available=False``.
"""

from __future__ import annotations

import sys
import types

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_azure_mod = load_module("distllm/cloud/azure.py")
AzureAvailabilityChecker = _azure_mod.AzureAvailabilityChecker


class _FakeToken:
    token = "fake-token"


def _install_fakes(monkeypatch, *, credential=None, cred_error=None):
    """Install fake azure.identity + httpx modules in sys.modules."""
    fake_identity = types.ModuleType("azure.identity")

    class _FakeDefaultAzureCredential:
        def __init__(self, *a, **kw):
            if cred_error is not None:
                raise cred_error

        def get_token(self, scope):
            return _FakeToken()

    fake_identity.DefaultAzureCredential = _FakeDefaultAzureCredential
    monkeypatch.setitem(sys.modules, "azure.identity", fake_identity)

    calls: dict = {}

    class _FakeResponse:
        def __init__(self, status_code=200, payload=None):
            self.status_code = status_code
            self._payload = payload or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return self._payload

    def _fake_get(url, **kwargs):
        calls["url"] = url
        calls["headers"] = kwargs.get("headers", {})
        calls["params"] = kwargs.get("params", {})
        return _FakeResponse(status_code=calls.get("_status", 200),
                             payload={"value": [{"name": "Standard_NC24ads_A100_v4"}]})

    fake_httpx = types.ModuleType("httpx")
    fake_httpx.get = _fake_get
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    return calls


class TestF011AzureAvailabilityFailClosed:
    """F-011: Azure availability check must be real and fail closed."""

    def test_missing_subscription_id_fails_closed(self, monkeypatch):
        """No AZURE_SUBSCRIPTION_ID -> available=False (was fail-open True)."""
        monkeypatch.delenv("AZURE_SUBSCRIPTION_ID", raising=False)
        _install_fakes(monkeypatch)
        checker = AzureAvailabilityChecker()
        info = checker.check_availability("Standard_NC24ads_A100_v4", "eastus")
        assert info.available is False

    def test_credential_failure_fails_closed(self, monkeypatch):
        """DefaultAzureCredential failure -> available=False for known SKU."""
        monkeypatch.setenv("AZURE_SUBSCRIPTION_ID", "sub-123")
        _install_fakes(
            monkeypatch,
            cred_error=RuntimeError("no credentials available"),
        )
        checker = AzureAvailabilityChecker()
        # Known SKU that the old code reported as available=True on error.
        info = checker.check_availability("Standard_NC24ads_A100_v4", "eastus")
        assert info.available is False

    def test_http_error_fails_closed(self, monkeypatch):
        """ARM returning an HTTP error -> available=False."""
        monkeypatch.setenv("AZURE_SUBSCRIPTION_ID", "sub-123")
        calls = _install_fakes(monkeypatch)
        calls["_status"] = 401
        checker = AzureAvailabilityChecker()
        info = checker.check_availability("Standard_NC24ads_A100_v4", "eastus")
        assert info.available is False

    def test_success_reports_real_result(self, monkeypatch):
        """Happy path: SKU present in region -> available=True."""
        monkeypatch.setenv("AZURE_SUBSCRIPTION_ID", "sub-123")
        calls = _install_fakes(monkeypatch)
        checker = AzureAvailabilityChecker()
        info = checker.check_availability("Standard_NC24ads_A100_v4", "westeurope")
        assert info.available is True
        # Region and subscription must actually be interpolated into the URL.
        assert "/subscriptions/sub-123/providers/Microsoft.Compute/" in calls["url"]
        assert "/locations/westeurope/vmSizes" in calls["url"]
        assert "{subscriptionId}" not in calls["url"]
        assert "{region}" not in calls["url"]
        # Auth header must carry a bearer token.
        assert calls["headers"]["Authorization"].startswith("Bearer ")
        # API version still set.
        assert calls["params"]["api-version"] == "2023-07-01"

    def test_sku_absent_in_region_reports_false(self, monkeypatch):
        """SKU exists but not offered in this region -> available=False."""
        monkeypatch.setenv("AZURE_SUBSCRIPTION_ID", "sub-123")
        calls = _install_fakes(monkeypatch)
        calls["_status"] = 200
        # Fake returns only Standard_NC24ads_A100_v4; ask for another SKU.
        checker = AzureAvailabilityChecker()
        info = checker.check_availability("Standard_ND96asr_v4", "japaneast")
        assert info.available is False

    def test_constructor_subscription_id_wins(self, monkeypatch):
        """Explicit constructor arg overrides env var."""
        monkeypatch.setenv("AZURE_SUBSCRIPTION_ID", "env-sub")
        calls = _install_fakes(monkeypatch)
        checker = AzureAvailabilityChecker(subscription_id="ctor-sub")
        checker.check_availability("Standard_NC24ads_A100_v4", "eastus")
        assert "/subscriptions/ctor-sub/" in calls["url"]
