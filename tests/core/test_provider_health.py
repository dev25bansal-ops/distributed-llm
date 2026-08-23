"""Tests for ProviderHealthProber, RegionHealth, ProbeResult.

Uses the import-helper pattern to avoid circular imports.
"""

from __future__ import annotations

import time

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_health_mod = load_module("distllm/core/provider_health.py")
ProbeResult = _health_mod.ProbeResult
RegionHealth = _health_mod.RegionHealth
ProviderHealthProber = _health_mod.ProviderHealthProber
_HEALTH_ENDPOINTS = _health_mod._HEALTH_ENDPOINTS


class TestProbeResult:
    def test_defaults(self):
        r = ProbeResult(provider="aws", region="us-east-1", endpoint="https://example.com/ping")
        assert r.healthy is True
        assert r.latency_ms == 0.0
        assert r.status_code == 0
        assert r.error == ""
        assert r.timestamp > 0

    def test_custom_values(self):
        r = ProbeResult(
            provider="gcp", region="us-central1",
            endpoint="https://compute.googleapis.com/ping",
            healthy=False, latency_ms=150.0, status_code=503,
            error="Service Unavailable",
        )
        assert r.healthy is False
        assert r.latency_ms == 150.0
        assert r.status_code == 503
        assert r.error == "Service Unavailable"


class TestRegionHealth:
    def test_defaults(self):
        h = RegionHealth(provider="aws", region="us-east-1")
        assert h.healthy is True
        assert h.avg_latency_ms == 0.0
        assert h.success_rate == 1.0
        assert h.consecutive_failures == 0

    def test_to_dict(self):
        h = RegionHealth(provider="aws", region="us-east-1")
        d = h.to_dict()
        assert d["provider"] == "aws"
        assert d["region"] == "us-east-1"
        assert d["healthy"] is True
        assert "avg_latency_ms" in d
        assert "success_rate" in d
        assert "last_check" in d
        assert "consecutive_failures" in d


class TestProviderHealthProber:
    def test_add_region_and_get_health(self):
        prober = ProviderHealthProber()
        prober.add_region("aws", "us-east-1")
        prober.add_region("aws", "us-west-2")
        health = prober.get_health()
        assert len(health) == 2
        assert health[0]["provider"] == "aws"

    def test_get_health_with_filters(self):
        prober = ProviderHealthProber()
        prober.add_region("aws", "us-east-1")
        prober.add_region("gcp", "us-central1")
        results = prober.get_health(provider="aws")
        assert len(results) == 1
        assert results[0]["region"] == "us-east-1"

    def test_is_healthy_default_true(self):
        prober = ProviderHealthProber()
        assert prober.is_healthy("aws", "nonexistent") is True

    def test_is_healthy_registered(self):
        prober = ProviderHealthProber()
        prober.add_region("aws", "us-east-1")
        assert prober.is_healthy("aws", "us-east-1") is True

    def test_get_unhealthy_regions_empty(self):
        prober = ProviderHealthProber()
        prober.add_region("aws", "us-east-1")
        assert prober.get_unhealthy_regions() == []

    def test_get_endpoint(self):
        prober = ProviderHealthProber()
        ep = prober._get_endpoint("aws", "us-east-1")
        assert ep == "https://ec2.us-east-1.amazonaws.com/ping"

    def test_get_endpoint_fallback(self):
        prober = ProviderHealthProber()
        ep = prober._get_endpoint("unknown_provider", "unknown-region")
        assert ep == "https://unknown_provider.com"

    def test_add_region_dedup(self):
        prober = ProviderHealthProber()
        prober.add_region("aws", "us-east-1")
        prober.add_region("aws", "us-east-1")
        assert len(prober._regions) == 1

    def test_start_stop(self):
        prober = ProviderHealthProber(probe_interval_s=9999)
        prober.add_region("aws", "us-east-1")
        prober.start()
        assert prober._running is True
        assert prober._thread is not None
        assert prober._thread.name == "provider-health-prober"
        prober.stop()
        assert prober._running is False
        assert prober._thread is None

    def test_health_endpoints_defined(self):
        assert "aws" in _HEALTH_ENDPOINTS
        assert "gcp" in _HEALTH_ENDPOINTS
        assert "azure" in _HEALTH_ENDPOINTS
        assert "us-east-1" in _HEALTH_ENDPOINTS["aws"]
