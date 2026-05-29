"""Provider Health Prober — continuous health monitoring for cloud regions.

Background service that probes all configured cloud regions for:
- HTTP/S endpoint health
- gRPC connectivity
- Latency measurements
- Spot availability

Feeds results into the CrossCloudRouter's availability and latency caches.
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger


@dataclass
class ProbeResult:
    """Result of a single health probe."""
    provider: str
    region: str
    endpoint: str
    healthy: bool = True
    latency_ms: float = 0.0
    status_code: int = 0
    error: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class RegionHealth:
    """Aggregated health status for a region."""
    provider: str
    region: str
    healthy: bool = True
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    last_check: float = 0.0
    consecutive_failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "region": self.region,
            "healthy": self.healthy,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "success_rate": round(self.success_rate, 3),
            "last_check": self.last_check,
            "consecutive_failures": self.consecutive_failures,
        }


# Default health check endpoints per provider/region
_HEALTH_ENDPOINTS: dict[str, dict[str, str]] = {
    "aws": {
        "us-east-1": "https://ec2.us-east-1.amazonaws.com/ping",
        "us-west-2": "https://ec2.us-west-2.amazonaws.com/ping",
        "eu-west-1": "https://ec2.eu-west-1.amazonaws.com/ping",
    },
    "gcp": {
        "us-central1": "https://compute.googleapis.com/compute/v1/projects",
        "europe-west1": "https://compute.googleapis.com/compute/v1/projects",
    },
    "azure": {
        "eastus": "https://management.azure.com/subscriptions?api-version=2022-12-01",
        "westeurope": "https://management.azure.com/subscriptions?api-version=2022-12-01",
    },
}


class ProviderHealthProber:
    """Continuous health prober for cloud provider regions.

    Usage::

        prober = ProviderHealthProber()
        prober.add_region("aws", "us-east-1")
        prober.add_region("gcp", "us-central1")
        prober.set_router(router)  # Feed results to CrossCloudRouter
        prober.start()
    """

    def __init__(
        self,
        probe_interval_s: float = 60.0,
        timeout_s: float = 10.0,
        failure_threshold: int = 3,
        on_health_change: Callable[[RegionHealth], None] | None = None,
    ):
        self._interval = probe_interval_s
        self._timeout = timeout_s
        self._failure_threshold = failure_threshold
        self._on_health_change = on_health_change

        self._regions: list[tuple[str, str]] = []  # (provider, region)
        self._health: dict[str, RegionHealth] = {}
        self._router: Any = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def add_region(self, provider: str, region: str) -> None:
        """Add a region to probe."""
        key = f"{provider}:{region}"
        if (provider, region) not in self._regions:
            self._regions.append((provider, region))
            self._health[key] = RegionHealth(provider=provider, region=region)

    def set_router(self, router: Any) -> None:
        """Set the CrossCloudRouter to feed health results into."""
        self._router = router

    def start(self) -> None:
        """Start the background probing loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._probe_loop,
            daemon=True,
            name="provider-health-prober",
        )
        self._thread.start()
        logger.info(f"Provider health prober started ({len(self._regions)} regions, interval={self._interval}s)")

    def stop(self) -> None:
        """Stop the probing loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _probe_loop(self) -> None:
        while self._running:
            for provider, region in self._regions:
                if not self._running:
                    break
                try:
                    self._probe_region(provider, region)
                except Exception as e:
                    logger.debug(f"Probe failed for {provider}:{region}: {e}")
            time.sleep(self._interval)

    def _probe_region(self, provider: str, region: str) -> None:
        """Probe a single region."""
        key = f"{provider}:{region}"
        endpoint = self._get_endpoint(provider, region)

        start = time.time()
        healthy = False
        latency = 0.0
        status_code = 0
        error = ""

        try:
            import httpx
            resp = httpx.get(endpoint, timeout=self._timeout, follow_redirects=True)
            latency = (time.time() - start) * 1000
            status_code = resp.status_code
            healthy = 200 <= resp.status_code < 500
        except Exception as e:
            latency = (time.time() - start) * 1000
            error = str(e)

        # Update health record
        with self._lock:
            health = self._health.get(key)
            if not health:
                health = RegionHealth(provider=provider, region=region)
                self._health[key] = health

            was_healthy = health.healthy
            if healthy:
                health.consecutive_failures = 0
                health.healthy = True
                # Exponential moving average for latency
                if health.avg_latency_ms > 0:
                    health.avg_latency_ms = 0.7 * health.avg_latency_ms + 0.3 * latency
                else:
                    health.avg_latency_ms = latency
            else:
                health.consecutive_failures += 1
                if health.consecutive_failures >= self._failure_threshold:
                    health.healthy = False

            health.last_check = time.time()
            # Update success rate (rolling window)
            total_checks = max(1, int((time.time() - health.last_check) / self._interval) + 1)
            health.success_rate = max(0, 1.0 - (health.consecutive_failures / max(total_checks, 10)))

            # Feed to router if health changed
            if self._router and was_healthy != health.healthy:
                self._router.update_availability(provider, "", health.healthy, region=region)
                if self._on_health_change:
                    try:
                        self._on_health_change(health)
                    except Exception:
                        pass
            if self._router and healthy and latency > 0:
                self._router.update_latency(f"{provider}:{region}", latency)

    def _get_endpoint(self, provider: str, region: str) -> str:
        """Get the health check endpoint for a region."""
        endpoints = _HEALTH_ENDPOINTS.get(provider, {})
        return endpoints.get(region, f"https://{provider}.com")

    def get_health(self, provider: str = "", region: str = "") -> list[dict[str, Any]]:
        """Get health status for all or specific regions."""
        with self._lock:
            results = []
            for key, health in self._health.items():
                if provider and health.provider != provider:
                    continue
                if region and health.region != region:
                    continue
                results.append(health.to_dict())
            return results

    def is_healthy(self, provider: str, region: str) -> bool:
        """Check if a specific region is healthy."""
        with self._lock:
            health = self._health.get(f"{provider}:{region}")
            return health.healthy if health else True

    def get_unhealthy_regions(self) -> list[dict[str, Any]]:
        """Get all unhealthy regions."""
        with self._lock:
            return [h.to_dict() for h in self._health.values() if not h.healthy]
