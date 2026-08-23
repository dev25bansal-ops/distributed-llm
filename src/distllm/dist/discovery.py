"""mDNS/Zeroconf auto-discovery for DistLLM clusters.

Allows devices on the same LAN to automatically discover each other
without manual IP:port configuration.

Usage:
    # On the coordinator:
    from distllm.dist.discovery import DiscoveryService
    service = DiscoveryService(port=50050, service_id="my_cluster")
    service.start()

    # On the worker:
    from distllm.dist.discovery import DiscoveryClient
    client = DiscoveryClient(timeout=3.0)
    coordinators = client.discover()
    # → [("192.168.1.100", 50050, "my_cluster"), ...]
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Any

from loguru import logger

_SERVICE_TYPE = "_distllm._tcp.local."


class DiscoveryService:
    """Advertises a coordinator on the LAN via mDNS.

    Registers a DNS-SD service of type ``_distllm._tcp`` so that
    ``DiscoveryClient`` instances can find it without manual config.

    Args:
        port: The coordinator's gRPC port.
        service_id: A human-readable name for this cluster.
        host: The hostname or IP to advertise (auto-detected if None).
        properties: Optional key/value metadata to include in advertisement.
    """

    def __init__(
        self,
        port: int = 50050,
        service_id: str = "distllm-cluster",
        host: str | None = None,
        properties: dict[str, str] | None = None,
    ):
        self._port = port
        self._service_id = service_id
        self._host = host or socket.gethostname()
        self._properties = properties or {}
        self._running = False
        self._zeroconf: Any = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start advertising the coordinator on the LAN."""
        try:
            from zeroconf import Zeroconf, ServiceInfo
        except ImportError:
            logger.warning("zeroconf not available. Install: pip install zeroconf")
            return

        local_ip = self._get_local_ip()
        if not local_ip:
            logger.warning("No LAN IP found — cannot advertise via mDNS")
            return

        info = ServiceInfo(
            type_=_SERVICE_TYPE,
            name=f"{self._service_id}.{_SERVICE_TYPE}",
            addresses=[socket.inet_aton(local_ip)],
            port=self._port,
            properties=self._properties,
        )

        try:
            self._zeroconf = Zeroconf(interfaces=[local_ip])
            self._zeroconf.register_service(info)
            self._running = True
            logger.info(f"mDNS: advertising {self._service_id} on {local_ip}:{self._port}")
        except Exception as e:
            logger.warning(f"mDNS registration failed: {e}")

    def stop(self) -> None:
        """Stop advertising."""
        if self._zeroconf:
            try:
                self._zeroconf.unregister_all_services()
                self._zeroconf.close()
            except Exception:
                pass
            self._zeroconf = None
            self._running = False
            logger.info("mDNS: advertisement stopped")

    @staticmethod
    def _get_local_ip() -> str | None:
        """Get the primary LAN IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.5)
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return None


class DiscoveryClient:
    """Discovers DistLLM coordinators on the LAN via mDNS.

    Args:
        timeout: Maximum time in seconds to wait for responses.
    """

    def __init__(self, timeout: float = 3.0):
        self._timeout = timeout
        self._found_services: list[dict[str, Any]] = []

    def discover(self) -> list[dict[str, Any]]:
        """Scan the LAN for DistLLM coordinators.

        Returns:
            List of dicts with keys: host, port, name, properties.
        """
        try:
            from zeroconf import Zeroconf, ServiceBrowser
        except ImportError:
            logger.warning("zeroconf not available. Install: pip install zeroconf")
            return []

        self._found_services = []
        zeroconf = Zeroconf()
        browser = ServiceBrowser(zeroconf, _SERVICE_TYPE, handlers=[self._on_service])

        time.sleep(self._timeout)
        zeroconf.close()

        return self._found_services

    def _on_service(self, zeroconf, service_type, name, state_change) -> None:
        """Callback when a service is discovered or removed."""
        from zeroconf import ServiceStateChange

        if state_change is ServiceStateChange.Added:
            info = zeroconf.get_service_info(service_type, name)
            if info:
                host = socket.inet_ntoa(info.addresses[0]) if info.addresses else None
                entry = {
                    "host": host,
                    "port": info.port,
                    "name": name.replace(f".{_SERVICE_TYPE}", ""),
                    "properties": {
                        k.decode() if isinstance(k, bytes) else k:
                            v.decode() if isinstance(v, bytes) else v
                        for k, v in (info.properties or {}).items()
                    },
                }
                if entry["host"] and entry not in self._found_services:
                    self._found_services.append(entry)
                    logger.debug(f"mDNS discovered: {entry['name']} at {entry['host']}:{entry['port']}")
