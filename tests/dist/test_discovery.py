"""Tests for distllm.dist.discovery module.

Tests DiscoveryService and DiscoveryClient using ONLY real objects
from the module.  No unittest.mock, no network-dependent assertions,
no timing-dependent assertions.
"""

from __future__ import annotations

import socket
from types import SimpleNamespace

import pytest

from distllm.dist.discovery import DiscoveryService, DiscoveryClient


# ---------------------------------------------------------------------------
# DiscoveryService
# ---------------------------------------------------------------------------


class TestDiscoveryService:
    """Test DiscoveryService -- advertises coordinator on LAN via mDNS."""

    # -- init edge cases ---------------------------------------------------

    def test_init_defaults(self) -> None:
        service = DiscoveryService()
        assert service._port == 50050
        assert service._service_id == "distllm-cluster"
        assert isinstance(service._host, str)
        assert len(service._host) > 0
        assert service._properties == {}
        assert service._running is False
        assert service._zeroconf is None

    def test_init_custom_all_params(self) -> None:
        service = DiscoveryService(
            port=12345,
            service_id="my-test",
            host="test-host",
            properties={"region": "us-east"},
        )
        assert service._port == 12345
        assert service._service_id == "my-test"
        assert service._host == "test-host"
        assert service._properties == {"region": "us-east"}

    def test_init_host_none_uses_gethostname(self) -> None:
        service = DiscoveryService(host=None)
        assert isinstance(service._host, str)
        assert len(service._host) > 0

    def test_init_properties_none_becomes_empty_dict(self) -> None:
        assert DiscoveryService(properties=None)._properties == {}

    def test_init_properties_empty_dict(self) -> None:
        assert DiscoveryService(properties={})._properties == {}

    def test_init_service_id_empty_string(self) -> None:
        assert DiscoveryService(service_id="")._service_id == ""

    def test_init_port_zero(self) -> None:
        assert DiscoveryService(port=0)._port == 0

    def test_init_port_max(self) -> None:
        assert DiscoveryService(port=65535)._port == 65535

    # -- lifecycle tests ---------------------------------------------------

    def test_stop_without_start_is_safe(self) -> None:
        service = DiscoveryService()
        service.stop()
        assert service._running is False
        assert service._zeroconf is None

    def test_start_and_stop(self) -> None:
        """start() then stop() -- no crash regardless of mDNS availability."""
        service = DiscoveryService(port=19999, service_id="test-roundtrip")
        service.start()
        service.stop()
        assert service._running is False
        assert service._zeroconf is None

    def test_start_stop_multiple_cycles(self) -> None:
        """start()/stop() is idempotent across multiple calls."""
        service = DiscoveryService(port=19998, service_id="test-multi")
        for _ in range(3):
            service.start()
            service.stop()
        assert service._running is False

    # -- internal helpers --------------------------------------------------

    def test_get_local_ip_returns_valid_or_none(self) -> None:
        ip = DiscoveryService._get_local_ip()
        if ip is not None:
            parts = ip.split(".")
            assert len(parts) == 4
            for p in parts:
                v = int(p)
                assert 0 <= v <= 255

    def test_get_local_ip_no_exception(self) -> None:
        """Must never raise, regardless of environment."""
        DiscoveryService._get_local_ip()  # no exception expected


# ---------------------------------------------------------------------------
# DiscoveryClient
# ---------------------------------------------------------------------------


class TestDiscoveryClient:
    """Test DiscoveryClient -- discovers coordinators on LAN via mDNS."""

    # -- init edge cases ---------------------------------------------------

    def test_init_defaults(self) -> None:
        client = DiscoveryClient()
        assert client._timeout == 3.0
        assert client._found_services == []

    def test_init_custom_timeout(self) -> None:
        assert DiscoveryClient(timeout=0.5)._timeout == 0.5

    def test_init_timeout_zero(self) -> None:
        assert DiscoveryClient(timeout=0.0)._timeout == 0.0

    def test_init_timeout_negative(self) -> None:
        assert DiscoveryClient(timeout=-5.0)._timeout == -5.0

    def test_initial_found_services_empty(self) -> None:
        assert DiscoveryClient()._found_services == []

    # -- _on_service: non-Added state changes ------------------------------

    def test_on_service_ignores_removed(self) -> None:
        """Removed state change must not add to found_services."""
        from zeroconf import ServiceStateChange

        client = DiscoveryClient()
        client._on_service(
            None, "_distllm._tcp.local.", "some-service", ServiceStateChange.Removed
        )
        assert client._found_services == []

    def test_on_service_ignores_updated(self) -> None:
        """Updated state change must not add to found_services."""
        from zeroconf import ServiceStateChange

        client = DiscoveryClient()
        client._on_service(
            None, "_distllm._tcp.local.", "some-service", ServiceStateChange.Updated
        )
        assert client._found_services == []

    # -- _on_service: Added with synthetic zeroconf doubles ----------------
    # These tests simulate a zeroconf responder *without* binding a real UDP
    # socket, keeping the suite deterministic and network-free.

    def _make_fake_zc(
        self,
        addresses: list[bytes] | None = None,
        port: int = 50050,
        properties: dict[bytes, bytes] | None = None,
    ) -> SimpleNamespace:
        """Build a minimal zeroconf-like double for testing _on_service."""
        if addresses is None:
            addresses = [socket.inet_aton("127.0.0.1")]
        info = SimpleNamespace(
            addresses=addresses,
            port=port,
            properties=properties if properties is not None else {},
        )
        return SimpleNamespace(get_service_info=lambda _t, _n: info)

    def test_on_service_adds_entry(self) -> None:
        from zeroconf import ServiceStateChange

        from distllm.dist.discovery import _SERVICE_TYPE

        zc = self._make_fake_zc()
        client = DiscoveryClient()
        client._on_service(zc, _SERVICE_TYPE, "hello-node", ServiceStateChange.Added)

        assert len(client._found_services) == 1
        entry = client._found_services[0]
        assert entry["host"] == "127.0.0.1"
        assert entry["port"] == 50050
        assert entry["name"] == "hello-node"
        assert entry["properties"] == {}

    def test_on_service_strips_service_type_from_name(self) -> None:
        from zeroconf import ServiceStateChange

        from distllm.dist.discovery import _SERVICE_TYPE

        zc = self._make_fake_zc()
        client = DiscoveryClient()
        client._on_service(
            zc,
            _SERVICE_TYPE,
            f"stripped-dot.{_SERVICE_TYPE}",
            ServiceStateChange.Added,
        )

        assert len(client._found_services) == 1
        assert client._found_services[0]["name"] == "stripped-dot"

    def test_on_service_ignores_duplicates(self) -> None:
        from zeroconf import ServiceStateChange

        from distllm.dist.discovery import _SERVICE_TYPE

        zc = self._make_fake_zc()
        client = DiscoveryClient()
        client._on_service(zc, _SERVICE_TYPE, "dup", ServiceStateChange.Added)
        client._on_service(zc, _SERVICE_TYPE, "dup", ServiceStateChange.Added)

        assert len(client._found_services) == 1

    def test_on_service_skips_entry_without_host(self) -> None:
        from zeroconf import ServiceStateChange

        from distllm.dist.discovery import _SERVICE_TYPE

        zc = self._make_fake_zc(addresses=[])
        client = DiscoveryClient()
        client._on_service(zc, _SERVICE_TYPE, "no-host", ServiceStateChange.Added)

        assert client._found_services == []

    def test_on_service_skips_none_info(self) -> None:
        """When get_service_info returns None the entry must be skipped."""
        from zeroconf import ServiceStateChange

        from distllm.dist.discovery import _SERVICE_TYPE

        zc = SimpleNamespace(get_service_info=lambda _t, _n: None)
        client = DiscoveryClient()
        client._on_service(zc, _SERVICE_TYPE, "missing", ServiceStateChange.Added)

        assert client._found_services == []

    def test_on_service_decodes_bytes_properties(self) -> None:
        from zeroconf import ServiceStateChange

        from distllm.dist.discovery import _SERVICE_TYPE

        info = SimpleNamespace(
            addresses=[socket.inet_aton("10.0.0.1")],
            port=50050,
            properties={b"region": b"us-east", b"model": b"llama"},
        )
        zc = SimpleNamespace(get_service_info=lambda _t, _n: info)
        client = DiscoveryClient()
        client._on_service(zc, _SERVICE_TYPE, "prop-test", ServiceStateChange.Added)

        assert len(client._found_services) == 1
        assert client._found_services[0]["properties"] == {
            "region": "us-east",
            "model": "llama",
        }

    def test_on_service_handles_mixed_byte_str_properties(self) -> None:
        from zeroconf import ServiceStateChange

        from distllm.dist.discovery import _SERVICE_TYPE

        info = SimpleNamespace(
            addresses=[socket.inet_aton("10.0.0.2")],
            port=50050,
            properties={b"key1": b"val1", "key2": "val2"},
        )
        zc = SimpleNamespace(get_service_info=lambda _t, _n: info)
        client = DiscoveryClient()
        client._on_service(zc, _SERVICE_TYPE, "mixed", ServiceStateChange.Added)

        assert client._found_services[0]["properties"] == {
            "key1": "val1",
            "key2": "val2",
        }

    # -- discover() smoke tests -------------------------------------------
    # These exercise the real Zeroconf bind path.  If mDNS port 5353 is
    # unavailable the test skips; otherwise we only check result shape
    # (no timing/network assertions).

    def test_discover_returns_list(self) -> None:
        """discover() returns a list (likely empty on a short timeout)."""
        client = DiscoveryClient(timeout=0.01)
        try:
            result = client.discover()
        except OSError:
            pytest.skip("mDNS port 5353 not available")
        assert isinstance(result, list)
        for entry in result:
            assert isinstance(entry, dict)
            assert "host" in entry
            assert "port" in entry
            assert "name" in entry
            assert "properties" in entry
            if entry["host"] is not None:
                assert isinstance(entry["host"], str)
            assert isinstance(entry["port"], int)
            assert isinstance(entry["name"], str)
            assert isinstance(entry["properties"], dict)

    def test_discover_called_twice(self) -> None:
        """Multiple discover() calls are safe."""
        client = DiscoveryClient(timeout=0.01)
        try:
            r1 = client.discover()
            r2 = client.discover()
        except OSError:
            pytest.skip("mDNS port 5353 not available")
        assert isinstance(r1, list)
        assert isinstance(r2, list)
