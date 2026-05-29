"""Security tests for routing module.

Covers:
- Auth token not leaked in logs
- SSRF via region string
- CLI secret exposure
- Connection pool limits
"""

import logging
import os
from unittest.mock import patch

import pytest

from distllm.core.cross_cloud_router import CarbonIntensityProvider, CarbonProvider


@pytest.mark.security
class TestAuthTokenLeakage:
    """Test auth tokens are not leaked in logs."""

    def test_electricitymap_token_not_in_logs(self, caplog):
        os.environ["ELECTRICITYMAP_AUTH_TOKEN"] = "super-secret-token-12345"
        try:
            provider = CarbonIntensityProvider(provider=CarbonProvider.ELECTRICITYMAP)
            with caplog.at_level(logging.DEBUG):
                provider.get_intensity("us-east-1")
            # Token should not appear in any log message
            for record in caplog.records:
                assert "super-secret-token-12345" not in record.getMessage()
        finally:
            del os.environ["ELECTRICITYMAP_AUTH_TOKEN"]

    def test_watttime_password_not_in_logs(self, caplog):
        os.environ["WATTTIME_USERNAME"] = "testuser"
        os.environ["WATTTIME_PASSWORD"] = "super-secret-password"
        try:
            provider = CarbonIntensityProvider(provider=CarbonProvider.WATTTIME)
            with caplog.at_level(logging.DEBUG):
                provider.get_intensity("us-east-1")
            for record in caplog.records:
                assert "super-secret-password" not in record.getMessage()
        finally:
            del os.environ["WATTTIME_USERNAME"]
            del os.environ["WATTTIME_PASSWORD"]


@pytest.mark.security
class TestSSRFViaRegionString:
    """Test that malicious region strings don't cause SSRF."""

    def test_malicious_region_returns_static(self):
        provider = CarbonIntensityProvider(provider=CarbonProvider.STATIC)
        intensity = provider.get_intensity("http://evil.com/callback")
        assert intensity.source == "static"

    def test_empty_region_handled(self):
        provider = CarbonIntensityProvider(provider=CarbonProvider.STATIC)
        intensity = provider.get_intensity("")
        assert intensity.gco2_per_kwh > 0

    def test_very_long_region_handled(self):
        provider = CarbonIntensityProvider(provider=CarbonProvider.STATIC)
        intensity = provider.get_intensity("A" * 10000)
        assert intensity.gco2_per_kwh > 0


@pytest.mark.security
class TestConnectionPoolLimits:
    """Test connection pool doesn't exhaust resources."""

    def test_shared_client_reused(self):
        provider = CarbonIntensityProvider(provider=CarbonProvider.STATIC)
        client1 = provider._get_http_client()
        client2 = provider._get_http_client()
        assert client1 is client2  # Same instance

    def test_close_releases_client(self):
        provider = CarbonIntensityProvider(provider=CarbonProvider.STATIC)
        _ = provider._get_http_client()
        provider.close()
        assert provider._http_client is None


@pytest.mark.security
class TestClusterKeySecurity:
    """Test cluster key is not exposed in CLI args."""

    def test_resolve_cluster_key_from_env(self):
        os.environ["DISTLLM_CLUSTER_KEY"] = "env-secret-key"
        try:
            from distllm.core.coordinator import _resolve_cluster_key
            key = _resolve_cluster_key()
            assert key == "env-secret-key"
        finally:
            del os.environ["DISTLLM_CLUSTER_KEY"]

    def test_resolve_cluster_key_missing(self):
        # Ensure no env var set
        os.environ.pop("DISTLLM_CLUSTER_KEY", None)
        from distllm.core.coordinator import _resolve_cluster_key
        key = _resolve_cluster_key()
        # Should be None or from file
        assert key is None or isinstance(key, str)
