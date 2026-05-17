"""Custom HPA for distributed-llm.

Reads Prometheus metrics and scales coordinator or node pool replicas
based on throughput and queue depth targets.
"""

import asyncio

import httpx
from loguru import logger

from distllm.errors import ConfigValidationError

try:
    from kubernetes import client as k8s_client, config as k8s_config
    HAS_K8S = True
except ImportError:
    HAS_K8S = False


# Allowlisted Prometheus URLs for SSRF protection
ALLOWED_PROMETHEUS_HOSTS: set = {
    "prometheus",
    "prometheus.monitoring",
    "prometheus.monitoring.svc",
    "prometheus.monitoring.svc.cluster.local",
    "localhost",
    "127.0.0.1",
}


def _validate_prometheus_url(url: str) -> str:
    """Validate a Prometheus URL against the allowlist to prevent SSRF.

    Args:
        url: The Prometheus URL to validate.

    Returns:
        The validated URL.

    Raises:
        ValueError: If the URL is not in the allowlist.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname
    scheme = parsed.scheme

    if scheme not in ("http", "https"):
        raise ConfigValidationError("prometheus_url", f"Scheme must be http or https, got '{scheme}'")

    if host not in ALLOWED_PROMETHEUS_HOSTS and not host.endswith(".svc.cluster.local"):
        raise ConfigValidationError(
            "prometheus_url",
            f"Host '{host}' is not in the allowlist. Allowed: {', '.join(sorted(ALLOWED_PROMETHEUS_HOSTS))}"
        )

    return url


class CustomHPA:
    """Horizontal Pod Autoscaler based on Prometheus metrics."""

    def __init__(
        self,
        prometheus_url: str = "http://prometheus:9090",
        metric: str = "tokens_per_second",
        target_value: float = 100.0,
        min_replicas: int = 1,
        max_replicas: int = 10,
        scale_target: str = "coordinator",  # or "nodepool"
        scale_target_name: str = "",
        evaluation_interval: float = 30.0,
    ):
        self._prometheus_url = _validate_prometheus_url(prometheus_url).rstrip("/")
        self._metric = metric
        self._target_value = target_value
        self._min_replicas = min_replicas
        self._max_replicas = max_replicas
        self._scale_target = scale_target
        self._scale_target_name = scale_target_name
        self._interval = evaluation_interval
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._evaluation_loop())
        logger.info(f"HPA started for {self._scale_target} (target: {self._target_value})")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _evaluation_loop(self) -> None:
        while self._running:
            await self._evaluate()
            await asyncio.sleep(self._interval)

    async def _evaluate(self) -> None:
        current_value = await self._query_metric()
        if current_value is None:
            return

        if self._target_value <= 0:
            return

        desired = int((current_value / self._target_value) * self._get_current_replicas())
        desired = max(self._min_replicas, min(self._max_replicas, desired))

        current = self._get_current_replicas()
        if desired != current:
            logger.info(
                f"HPA scaling {self._scale_target}/{self._scale_target_name}: "
                f"{current} -> {desired} (metric={self._metric}, value={current_value:.1f})"
            )
            await self._scale(desired)

    async def _query_metric(self) -> float | None:
        """Query Prometheus for the current metric value."""
        query_map = {
            "tokens_per_second": "distllm_tokens_per_second",
            "queue_depth": "distllm_coordinator_queue_depth",
        }
        metric_name = query_map.get(self._metric, self._metric)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self._prometheus_url}/api/v1/query",
                    params={"query": metric_name},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("data", {}).get("result", [])
                    if results:
                        return float(results[0].get("value", [0, 0])[1])
        except httpx.HTTPError as e:
            logger.warning(f"HPA metric query failed: {e}")
        return None

    def _get_k8s_apps_client(self):
        """Get authenticated K8s AppsV1Api client."""
        if not HAS_K8S:
            return None
        try:
            k8s_config.load_incluster_config()
        except k8s_config.ConfigException:
            k8s_config.load_kube_config()
        return k8s_client.AppsV1Api()

    def _get_current_replicas(self) -> int:
        """Read current replica count from K8s StatefulSet status."""
        apps_api = self._get_k8s_apps_client()
        if apps_api is None:
            return self._min_replicas

        try:
            ss = apps_api.read_namespaced_stateful_set_status(
                self._scale_target_name, "default"
            )
            return ss.status.replicas or self._min_replicas
        except Exception:
            logger.warning(f"Could not read replicas for {self._scale_target_name}, using min")
            return self._min_replicas

    async def _scale(self, desired_replicas: int) -> None:
        """Scale the target resource to desired replicas using K8s API."""
        if not HAS_K8S:
            logger.warning("kubernetes package not available, cannot scale")
            return
        try:
            apps_api = self._get_k8s_apps_client()
            if apps_api is None:
                return
            scale = k8s_client.V1Scale(
                spec=k8s_client.V1ScaleSpec(replicas=desired_replicas)
            )
            apps_api.patch_namespaced_stateful_set_scale(
                self._scale_target_name, "default", scale
            )
            logger.info(f"HPA: scaled {self._scale_target_name} to {desired_replicas}")
        except Exception as e:
            logger.error(f"HPA scale failed: {e}")
