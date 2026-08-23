"""Auto-discovery of DistLLM nodes on the local network using mDNS.

Implements zero-configuration node discovery so that ``distllm cluster discover``
finds all running coordinators on the same LAN segment automatically.

Uses ``python-zeroconf`` for mDNS service registration and browsing.
Falls back gracefully if the library is not installed.

Usage::

    # On coordinator node (automatically started):
    discoverer = AutoDiscoverer(service_type="_distllm._tcp.local.", port=50050)
    discoverer.register()

    # On client node:
    discoverer = AutoDiscoverer()
    nodes = discoverer.discover(timeout=3.0)
    # -> [{"host": "10.0.0.5", "port": 50050, "name": "coordinator-1", ...}]

    # Clean shutdown:
    discoverer.unregister()
"""

from __future__ import annotations

import ipaddress
import json
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

HAS_ZEROCONF = False
try:
    from zeroconf import IPVersion, ServiceBrowser, ServiceInfo, ServiceStateChange, Zeroconf
    HAS_ZEROCONF = True
except ImportError:
    pass


# Default mDNS service type for DistLLM coordinators.
_DEFAULT_SERVICE_TYPE = "_distllm._tcp.local."
_DEFAULT_DISCOVERY_TIMEOUT = 3.0


@dataclass
class DiscoveredNode:
    """A node discovered via mDNS on the local network."""
    host: str
    port: int
    name: str = ""
    server: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    is_coordinator: bool = True
    model_name: str = ""
    num_gpus: int = 0
    total_layers: int = 0
    api_port: int = 0
    weight: float = 1.0


@dataclass
class DiscoveryConfig:
    """Configuration for auto-discovery.

    Args:
        service_type: mDNS service type string.
        port: Port the coordinator's gRPC server listens on.
        api_port: Port the coordinator's API server listens on.
        host: Explicit host address (auto-detect if empty).
        ttl: mDNS record TTL in seconds.
        properties: Additional key/value properties to advertise.
        discovery_timeout: Seconds to wait for discovery responses.
    """
    service_type: str = _DEFAULT_SERVICE_TYPE
    port: int = 50050
    api_port: int = 8000
    host: str = ""
    ttl: int = 120
    properties: dict[str, str] = field(default_factory=dict)
    discovery_timeout: float = _DEFAULT_DISCOVERY_TIMEOUT


class AutoDiscoverer:
    """mDNS-based node discovery for DistLLM clusters.

    Registers this node as a ``_distllm._tcp`` service so other nodes
    can discover it, and can browse for other registered nodes on the
    same LAN segment without any manual IP configuration.

    Thread-safe: uses ``threading.Lock`` around shared ``_found_nodes``.
    """

    def __init__(self, config: DiscoveryConfig | None = None) -> None:
        self._config = config or DiscoveryConfig()
        self._zeroconf: Zeroconf | None = None
        self._service_info: ServiceInfo | None = None
        self._browser: ServiceBrowser | None = None
        self._lock = threading.Lock()
        self._found_nodes: dict[str, DiscoveredNode] = {}
        self._registered = False
        self._running = False

    # ── Registration ──────────────────────────────────────────────────────

    def register(self) -> bool:
        """Register this node as an mDNS service.

        Advertises the coordinator's gRPC port and metadata (model name,
        API port, GPU count) so other nodes can discover it via
        ``distllm cluster discover``.

        Returns:
            True if registration succeeded, False otherwise.
        """
        if not HAS_ZEROCONF:
            logger.warning(
                "python-zeroconf not installed. "
                "Install with: pip install zeroconf\n"
                "Auto-discovery disabled."
            )
            return False

        host = self._config.host or self._resolve_local_ip()
        if not host:
            logger.error("Could not determine local IP for mDNS registration")
            return False

        props = {
            "api_port": str(self._config.api_port),
            "coordinator": "1",
            "version": "0.4.1",
        }
        props.update(self._config.properties)

        self._service_info = ServiceInfo(
            type_=self._config.service_type,
            name=f"distllm-coordinator-{id(self):x}.{self._config.service_type}",
            addresses=[socket.inet_aton(host)],
            port=self._config.port,
            weight=0,
            priority=0,
            properties=props,
            server=socket.gethostname() + ".local.",
        )

        try:
            self._zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
            self._zeroconf.register_service(
                self._service_info,
                ttl=self._config.ttl,
            )
            self._registered = True
            logger.info(f"mDNS service registered: {host}:{self._config.port}")
            return True
        except Exception as e:
            logger.error(f"mDNS registration failed: {e}")
            if self._zeroconf:
                self._zeroconf.close()
                self._zeroconf = None
            return False

    def unregister(self) -> None:
        """Unregister the mDNS service and clean up resources."""
        if self._zeroconf and self._service_info:
            try:
                self._zeroconf.unregister_service(self._service_info)
            except Exception:
                pass
        if self._browser:
            try:
                self._browser.cancel()
            except Exception:
                pass
        if self._zeroconf:
            try:
                self._zeroconf.close()
            except Exception:
                pass
        self._zeroconf = None
        self._service_info = None
        self._browser = None
        self._registered = False
        self._running = False

    def __enter__(self) -> AutoDiscoverer:
        self.register()
        return self

    def __exit__(self, *args: Any) -> None:
        self.unregister()

    # ── Discovery ─────────────────────────────────────────────────────────

    def discover(
        self,
        timeout: float | None = None,
        min_nodes: int = 0,
    ) -> list[DiscoveredNode]:
        """Discover DistLLM coordinators on the local network.

        Browses for ``_distllm._tcp`` services and collects results
        for *timeout* seconds.

        Args:
            timeout: Max seconds to wait (default: config value, 3s).
            min_nodes: Return early once this many nodes are found (0 = wait full timeout).

        Returns:
            List of discovered nodes with host, port, and metadata.
        """
        if not HAS_ZEROCONF:
            logger.warning("python-zeroconf not installed — cannot discover nodes")
            return []

        self._found_nodes.clear()
        timeout = timeout if timeout is not None else self._config.discovery_timeout

        try:
            zc = Zeroconf(ip_version=IPVersion.V4Only)

            class _Listener:
                def __init__(self, outer: AutoDiscoverer) -> None:
                    self._outer = outer

                def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
                    info = zc.get_service_info(type_, name)
                    if info is None:
                        return
                    node = self._outer._info_to_node(info)
                    if node is not None:
                        with self._outer._lock:
                            self._outer._found_nodes[name] = node

                def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
                    with self._outer._lock:
                        self._outer._found_nodes.pop(name, None)

            listener = _Listener(self)
            browser = ServiceBrowser(zc, self._config.service_type, listener)
            self._browser = browser

            # Wait for discovery with early-exit if min_nodes reached
            deadline = time.time() + timeout
            while time.time() < deadline:
                with self._lock:
                    if min_nodes > 0 and len(self._found_nodes) >= min_nodes:
                        break
                time.sleep(0.1)

            browser.cancel()
            zc.close()

            with self._lock:
                results = list(self._found_nodes.values())
            return results

        except Exception as e:
            logger.error(f"mDNS discovery failed: {e}")
            return []

    # ── Helpers ───────────────────────────────────────────────────────────

    def _info_to_node(self, info: ServiceInfo) -> DiscoveredNode | None:
        """Convert a zeroconf ServiceInfo to a DiscoveredNode."""
        if not info.addresses:
            return None
        try:
            host = socket.inet_ntoa(info.addresses[0])
        except (ValueError, OSError):
            host = str(info.addresses[0])

        props = _decode_properties(info.properties)
        return DiscoveredNode(
            host=host,
            port=info.port,
            name=info.name,
            server=info.server or "",
            properties=props,
            is_coordinator=props.get("coordinator") == "1",
            model_name=props.get("model_name", ""),
            num_gpus=int(props.get("num_gpus", 0)),
            total_layers=int(props.get("total_layers", 0)),
            api_port=int(props.get("api_port", 0)),
            weight=float(props.get("weight", 1.0)),
        )

    @staticmethod
    def _resolve_local_ip() -> str:
        """Resolve the local IP address of this machine."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("10.255.255.255", 1))
                ip = s.getsockname()[0]
                if ip and not ip.startswith("127."):
                    return ip
        except Exception:
            pass
        try:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            if ip and not ip.startswith("127."):
                return ip
        except Exception:
            pass
        return ""

    # ── Integration helpers ───────────────────────────────────────────────

    @staticmethod
    def start_discovery_background(
        config: DiscoveryConfig | None = None,
        callback: Any = None,
    ) -> AutoDiscoverer:
        """Convenience: create, register, and start background discovery in one call.

        Returns the running AutoDiscoverer instance — caller should keep
        a reference and call ``unregister()`` on shutdown.
        """
        discoverer = AutoDiscoverer(config)
        discoverer.register()
        if callback:
            discoverer._callback = callback
        return discoverer

    def discover_peers(self, timeout: float = 3.0) -> list[dict[str, Any]]:
        """Discover peers and return plain dicts (for CLI/API integration)."""
        nodes = self.discover(timeout=timeout)
        return [
            {
                "host": n.host,
                "port": n.port,
                "name": n.name,
                "api_port": n.api_port,
                "model_name": n.model_name,
                "num_gpus": n.num_gpus,
            }
            for n in nodes
        ]


def _decode_properties(raw: dict[bytes, bytes]) -> dict[str, str]:
    """Decode zeroconf byte-string properties to plain strings."""
    return {
        k.decode("utf-8", errors="replace"): v.decode("utf-8", errors="replace")
        for k, v in raw.items()
    }
