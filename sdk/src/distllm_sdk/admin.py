"""AdminClient for the DistLLM cluster admin API.

Provides an async client for programmatic cluster management: listing nodes,
draining/cordoning nodes, querying cluster status, managing models, retrieving
metrics, and federation configuration.

Usage::

    async with AdminClient(
        base_url="http://coordinator:8000",
        api_key="sk-...",
    ) as admin:
        nodes = await admin.list_nodes()
        status = await admin.get_cluster_status()
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from distllm_sdk.constants import DEFAULT_HTTP_TIMEOUT
from distllm_sdk.errors import (
    ApiError,
    AuthenticationError,
    RateLimitError,
    TimeoutError,
)

_logger = logging.getLogger("distllm_sdk.admin")

__all__ = ["AdminClient"]


def _map_http_error(
    status_code: int,
    body: dict[str, Any],
    request_id: str | None = None,
) -> ApiError:
    """Map an HTTP status code and response body to a typed ``ApiError``."""
    msg = (
        body.get("error", {}).get("message", "")
        if isinstance(body.get("error"), dict)
        else body.get("message", httpx.codes.get_reason_phrase(status_code))
    )
    if status_code == 401:
        return AuthenticationError(msg or "Authentication failed", request_id=request_id)
    if status_code == 429:
        retry_after = (
            body.get("retry_after") if isinstance(body.get("error"), dict) else None
        )
        return RateLimitError(
            msg or "Rate limit exceeded",
            retry_after=retry_after,
            request_id=request_id,
        )
    if status_code == 404:
        return ApiError(
            msg or "Not found",
            status_code=status_code,
            error_type="not_found",
            request_id=request_id,
        )
    if status_code == 503:
        retry_after = (
            body.get("retry_after") if isinstance(body.get("error"), dict) else None
        )
        return ApiError(
            msg or "Service unavailable",
            status_code=status_code,
            error_type="service_unavailable",
            retry_after=retry_after,
            request_id=request_id,
        )
    return ApiError(
        msg or "API error",
        status_code=status_code,
        error_type="api_error",
        request_id=request_id,
    )


class AdminClient:
    """Async HTTP client for the DistLLM cluster admin API.

    Args:
        base_url: Root URL of the DistLLM coordinator (e.g.
            ``http://coordinator:8000``).
        api_key: Optional API key for authentication. Sent as a ``Bearer``
            token in the ``Authorization`` header.
        timeout: Default HTTP request timeout in seconds.
        verify: Whether to verify TLS certificates. Pass ``False`` to disable
            (not recommended in production).

    Example::

        admin = AdminClient(
            base_url="http://10.0.0.1:8000",
            api_key="sk-my-admin-key",
        )
        async with admin:
            nodes = await admin.list_nodes()
            for node in nodes:
                print(node["node_id"], node["state"])
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
        verify: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(self._timeout),
            verify=verify,
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "AdminClient":
        """Enter the async context manager.

        Returns:
            The ``AdminClient`` instance, ready for use.
        """
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit the async context manager, closing the HTTP client."""
        await self.close()

    async def close(self) -> None:
        """Close the underlying ``httpx.AsyncClient`` and free resources."""
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Internal request helpers
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send an HTTP request and return the JSON response dict.

        Args:
            method: HTTP method (``"GET"``, ``"POST"``, ``"PUT"``, etc.).
            path: URL path relative to ``base_url`` (e.g. ``/admin/nodes``).
            **kwargs: Extra arguments forwarded to ``httpx.AsyncClient.request``.

        Returns:
            The parsed JSON response body as a dictionary.

        Raises:
            AuthenticationError: If the server returns a 401 status.
            RateLimitError: If the server returns a 429 status.
            TimeoutError: If the request times out.
            ApiError: For any other non-2xx response.
        """
        try:
            response = await self._client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            body: dict[str, Any] = {}
            try:
                body = exc.response.json()
            except (json.JSONDecodeError, httpx.DecodingError):
                pass
            request_id = body.get("request_id") if isinstance(body, dict) else None
            raise _map_http_error(
                exc.response.status_code,
                body,
                request_id=request_id,
            ) from exc
        except httpx.TimeoutException as exc:
            raise TimeoutError(
                f"Request to {method} {path} timed out after {self._timeout}s",
            ) from exc
        except httpx.ConnectError as exc:
            raise ApiError(
                f"Connection refused to {self.base_url}{path}",
                status_code=0,
                error_type="connection_error",
            ) from exc
        except httpx.RemoteProtocolError as exc:
            raise ApiError(
                f"Protocol error on {method} {path}: {exc}",
                status_code=0,
                error_type="protocol_error",
            ) from exc

    async def _request_list(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Send a request that is expected to return a JSON array.

        Some admin endpoints return a top-level list (e.g. ``/admin/nodes``)
        rather than a wrapped object. This helper unwraps ``{"data": [...]}``
        or ``{"nodes": [...]}`` envelopes automatically.

        Args:
            method: HTTP method.
            path: URL path relative to ``base_url``.
            **kwargs: Extra arguments forwarded to ``_request``.

        Returns:
            The response data as a list of dictionaries.
        """
        data = await self._request(method, path, **kwargs)
        if isinstance(data, list):
            return data
        # Common envelope keys
        for key in ("data", "nodes", "results", "items"):
            if isinstance(data.get(key), list):
                return data[key]
        _logger.warning(
            "Response from %s %s is not a list and contains no known envelope key; "
            "returning as-is",
            method,
            path,
        )
        return list(data.values()) if isinstance(data, dict) else [data]

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    async def list_nodes(self) -> list[dict[str, Any]]:
        """List all registered worker nodes with their current state.

        GET ``/admin/nodes``

        Each node dict includes ``node_id``, ``host``, ``port``, ``healthy``,
        ``draining``, ``state``, ``gpu_name``, ``gpu_memory_free``,
        ``gpu_memory_total``, and layer assignment fields.

        Returns:
            A list of node-info dictionaries.
        """
        return await self._request_list("GET", "/admin/nodes")

    async def get_node(self, node_id: str) -> dict[str, Any]:
        """Get detailed info for a single node.

        GET ``/admin/nodes/{node_id}``

        Args:
            node_id: Unique identifier of the node.

        Returns:
            A node-info dictionary.

        Raises:
            ApiError: If the node is not found (status 404).
        """
        return await self._request("GET", f"/admin/nodes/{node_id}")

    async def drain_node(self, node_id: str) -> dict[str, Any]:
        """Drain a node, stopping new request dispatch.

        POST ``/admin/nodes/{node_id}/drain``

        Existing in-flight requests are allowed to complete. The node is not
        removed from the cluster; it will be skipped during load balancing.

        Args:
            node_id: Unique identifier of the node to drain.

        Returns:
            A status dictionary with keys ``status``, ``node_id``, ``message``.
        """
        return await self._request("POST", f"/admin/nodes/{node_id}/drain")

    async def cordon_node(self, node_id: str) -> dict[str, Any]:
        """Cordon a node, marking it as unschedulable for new workloads.

        POST ``/admin/nodes/{node_id}/cordon``

        A cordoned node is excluded from new model deployments and request
        routing until explicitly uncordoned. This is stronger than draining
        and is typically used for planned maintenance.

        Args:
            node_id: Unique identifier of the node to cordon.

        Returns:
            A status dictionary with keys ``status``, ``node_id``, ``message``.
        """
        return await self._request("POST", f"/admin/nodes/{node_id}/cordon")

    async def uncordon_node(self, node_id: str) -> dict[str, Any]:
        """Uncordon a previously cordoned node, restoring it to service.

        POST ``/admin/nodes/{node_id}/uncordon``

        Reverses the effect of :meth:`cordon_node`. The node will once again
        receive new deployments and request traffic.

        Args:
            node_id: Unique identifier of the node to uncordon.

        Returns:
            A status dictionary with keys ``status``, ``node_id``, ``message``.
        """
        return await self._request("POST", f"/admin/nodes/{node_id}/uncordon")

    # ------------------------------------------------------------------
    # Cluster status
    # ------------------------------------------------------------------

    async def get_cluster_status(self) -> dict[str, Any]:
        """Get the overall cluster health and status summary.

        GET ``/admin/cluster/status``

        Returns aggregate information such as total node count, healthy node
        count, draining count, total layers, and the coordinator's current
        operating state.

        Returns:
            A cluster-status dictionary.
        """
        return await self._request("GET", "/admin/cluster/status")

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    async def list_models(self) -> list[dict[str, Any]]:
        """List all models known to the cluster with detailed metadata.

        GET ``/admin/models``

        Each model dict includes ``id``, ``loaded``, ``dtype``,
        ``memory_used_mb``, ``memory_peak_mb``, ``partition``, ``version``,
        and GPU memory information.

        Returns:
            A list of model-info dictionaries.
        """
        return await self._request_list("GET", "/admin/models")

    async def get_model_info(self, model_id: str) -> dict[str, Any]:
        """Get detailed information for a specific model.

        GET ``/admin/models/{model_id}``

        Args:
            model_id: The model identifier (e.g. ``"llama-3-8b"``).

        Returns:
            A model-info dictionary.

        Raises:
            ApiError: If the model is not found (status 404).
        """
        return await self._request("GET", f"/admin/models/{model_id}")

    async def deploy_model(
        self,
        model: str,
        replicas: int = 1,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Deploy (load) a model onto the cluster.

        POST ``/admin/models/deploy``

        Instructs the coordinator to load the given model across available
        worker nodes. The model must be available in the model registry or
        on the shared filesystem.

        Args:
            model: Model name or path (e.g. ``"meta-llama/Llama-3-8B"``).
            replicas: Number of model replicas to deploy.
            **kwargs: Additional deployment parameters (e.g. ``dtype``,
                ``trust_remote_code``, ``quantization``).

        Returns:
            A deployment-status dictionary.
        """
        payload: dict[str, Any] = {"model": model, "replicas": replicas}
        payload.update(kwargs)
        return await self._request("POST", "/admin/models/deploy", json=payload)

    async def unload_model(self, model_id: str) -> dict[str, Any]:
        """Unload (remove) a model from the cluster.

        POST ``/admin/models/{model_id}/unload``

        Frees GPU memory and removes the model from all nodes it was loaded on.

        Args:
            model_id: The model identifier to unload.

        Returns:
            A status dictionary.
        """
        return await self._request(
            "POST",
            f"/admin/models/{model_id}/unload",
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    async def get_cluster_metrics(self) -> dict[str, Any]:
        """Get aggregate cluster performance metrics.

        GET ``/admin/metrics``

        Returns metrics such as request throughput, average latency, token
        generation rate, GPU utilisation, queue depth, cache hit ratio, and
        active request count.

        Returns:
            A metrics dictionary with numeric values and timestamps.
        """
        return await self._request("GET", "/admin/metrics")

    # ------------------------------------------------------------------
    # Federation
    # ------------------------------------------------------------------

    async def update_federation(self, config: dict[str, Any]) -> dict[str, Any]:
        """Update the federation configuration.

        PUT ``/admin/federation/config``

        Applies a new federation configuration to the cluster. The configuration
        controls how the cluster participates in federated training and model
        merging with peer coordinators.

        Args:
            config: Federation configuration dictionary. Expected keys may
                include ``enabled``, ``peer_coordinators``, ``merge_strategy``,
                ``heartbeat_interval``, and ``api_key``.

        Returns:
            A status dictionary confirming the update.
        """
        return await self._request(
            "PUT",
            "/admin/federation/config",
            json=config,
        )

    async def get_federation_status(self) -> dict[str, Any]:
        """Get the current federation configuration and status.

        GET ``/admin/federation/status``

        Returns the active federation configuration, peer coordinator
        connections, round status, and training statistics.

        Returns:
            A federation-status dictionary.
        """
        return await self._request("GET", "/admin/federation/status")
