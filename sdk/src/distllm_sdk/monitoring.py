"""MonitoringClient for Prometheus metrics and health monitoring.

Provides an async HTTP client for querying Prometheus-style metrics, health
endpoints, and admin-level monitoring data from a DistLLM cluster.

Usage::

    async with MonitoringClient(
        base_url="http://coordinator:8000",
        api_key="sk-...",
    ) as mon:
        health = await mon.get_health()
        metrics = await mon.get_metrics()
        gpu = await mon.get_gpu_utilization()
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from distllm_sdk.constants import DEFAULT_HTTP_TIMEOUT
from distllm_sdk.errors import ApiError, AuthenticationError, RateLimitError, TimeoutError

_logger = logging.getLogger("distllm_sdk.monitoring")

__all__ = [
    "MonitoringClient",
    "prom_text_to_dict",
]


# ---------------------------------------------------------------------------
# Prometheus text-format parser
# ---------------------------------------------------------------------------

# Regex matches a Prometheus-style metric line, e.g.:
#   metric_name{label="value",label2="value2"} 42.0 1234567890
_METRIC_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"  # metric name
    r"(?:\{(?P<labels>[^}]*)\})?\s*"  # optional label set
    r"(?P<value>-?(?:inf|nan|[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?))"  # value
    r"(?:\s+(?P<timestamp>[0-9]+(?:\.[0-9]+)?))?\s*$",  # optional timestamp
    re.IGNORECASE,
)

# Regex to split label key=value pairs inside curly braces.
_LABEL_PAIR_RE = re.compile(r'([a-zA-Z_:][a-zA-Z0-9_:]*)=("(?:[^"\\]|\\.)*"|[^\s,]+)')


def prom_text_to_dict(text: str) -> dict[str, Any]:
    """Parse a Prometheus text-format response into a structured dictionary.

    The returned dictionary groups metric samples by metric *name* and,
    for metrics with labels, nests them under the label signature.  Helper
    and comment lines (``# TYPE``, ``# HELP``, ``# UNIT``) are captured in
    a top-level ``_metadata`` key.

    Args:
        text: The raw Prometheus text body (``Content-Type: text/plain``).

    Returns:
        A dictionary with the following shape::

            {
                "go_gc_duration_seconds": {
                    "summary": 0.0,
                    "count": 42.0,
                },
                "http_requests_total": {
                    '{method="GET",handler="/v1/models"}': 1024.0,
                    '{method="POST",handler="/v1/chat"}": 512.0,
                },
                "up": 1.0,
                "_metadata": {
                    "up": {"type": "gauge", "help": "..."},
                },
            }
    """
    result: dict[str, Any] = {}
    metadata: dict[str, dict[str, str]] = {}
    current_type: str | None = None
    current_help: str | None = None

    for line_ in text.splitlines():
        line = line_.strip()
        if not line:
            continue

        # Comment / metadata lines
        if line.startswith("#"):
            parts = line[1:].strip().split(None, 2)
            if len(parts) >= 2:
                keyword = parts[0]
                name = parts[1]
                if keyword.upper() == "TYPE":
                    current_type = parts[2] if len(parts) >= 3 else "untyped"
                    metadata.setdefault(name, {})
                    metadata[name]["type"] = current_type
                elif keyword.upper() == "HELP":
                    current_help = parts[2] if len(parts) >= 3 else ""
                    metadata.setdefault(name, {})
                    metadata[name]["help"] = current_help
                elif keyword.upper() == "UNIT":
                    unit = parts[2] if len(parts) >= 3 else ""
                    metadata.setdefault(name, {})
                    metadata[name]["unit"] = unit
            continue

        # Data line
        m = _METRIC_LINE_RE.match(line)
        if not m:
            continue

        name = m.group("name")
        raw_labels = m.group("labels")
        raw_value = m.group("value")

        # Parse value
        if raw_value.lower() in ("inf", "+inf"):
            value: float = float("inf")
        elif raw_value.lower() == "-inf":
            value = float("-inf")
        elif raw_value.lower() == "nan":
            value = float("nan")
        else:
            value = float(raw_value)

        # Build key based on whether labels exist
        if raw_labels:
            label_signature = "{" + raw_labels + "}"
            entry = result.setdefault(name, {})
            if isinstance(entry, dict):
                entry[label_signature] = value
            # If someone else stored a scalar here, don't overwrite — push into a sub-dict
            elif isinstance(entry, (int, float)):
                result[name] = {"_scalar": entry, label_signature: value}
        else:
            # Unlabelled metric — check if we already have labelled samples
            existing = result.get(name)
            if isinstance(existing, dict) and existing:
                # Merge _scalar alongside labelled entries
                existing.setdefault("_scalar", value)
            elif existing is None:
                result[name] = value  # type: ignore[assignment]

    if metadata:
        result["_metadata"] = metadata

    return result


# ---------------------------------------------------------------------------
# Grafana-style JSON parser
# ---------------------------------------------------------------------------


def _parse_grafana_like(data: Any) -> dict[str, Any]:
    """Attempt to coerce a raw response into a standardised metrics dictionary.

    Recognised shapes:

    - Top-level list: wraps it in ``{"results": [...]}``.
    - Array-based time series ``{"data": [{"values": [...], ...}]}`` —
      normalised so each series is keyed by its label name.
    - Simple key-value response (returned as-is).

    Args:
        data: Parsed JSON data (dict or list).

    Returns:
        A normalised dictionary.
    """
    if isinstance(data, list):
        # Top-level list — likely an array of metric objects
        return {"results": data}

    if not isinstance(data, dict):
        return {"data": data}

    # Check for common time-series envelope
    for key in ("data", "results", "series"):
        items = data.get(key)
        if isinstance(items, list):
            normalised: list[dict[str, Any]] = []
            for item in items:
                if not isinstance(item, dict):
                    normalised.append({"value": item})
                elif "values" in item:
                    # Array-based time series: rotate into flat dict
                    labels = item.get("labels") or item.get("metric") or {}
                    values = item["values"]
                    normalised.append(
                        {
                            "labels": labels,
                            "values": values,
                        }
                    )
                else:
                    normalised.append(item)
            return {key: normalised, **{k: v for k, v in data.items() if k != key}}

    return data


# ---------------------------------------------------------------------------
# MonitoringClient
# ---------------------------------------------------------------------------


class MonitoringClient:
    """Async HTTP client for Prometheus metrics and health monitoring.

    This client provides access to raw Prometheus metrics (``/metrics``),
    health checks (``/health``), and admin-level monitoring endpoints for
    GPU utilisation, latency histograms, error rates, cache statistics, and
    more.

    The client manages its own ``httpx.AsyncClient`` lifecycle and should
    be used as a context manager::

        async with MonitoringClient("http://localhost:8000") as mon:
            up = await mon.get_health()

    Args:
        base_url: Root URL of the DistLLM coordinator or monitoring endpoint
            (e.g. ``http://coordinator:8000``).
        api_key: Optional API key for authentication. Sent as a ``Bearer``
            token in the ``Authorization`` header.
        timeout: Default HTTP request timeout in seconds.
        verify: Whether to verify TLS certificates. Pass ``False`` to disable
            (not recommended in production).
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

    async def __aenter__(self) -> "MonitoringClient":
        """Enter the async context manager.

        Returns:
            The ``MonitoringClient`` instance, ready for use.
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
            method: HTTP method (``"GET"``, ``"POST"``, etc.).
            path: URL path relative to ``base_url``.
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

    async def _request_text(self, method: str, path: str, **kwargs: Any) -> str:
        """Send an HTTP request and return the raw text body.

        Used for endpoints that return ``Content-Type: text/plain`` (e.g.
        the Prometheus ``/metrics`` endpoint).

        Args:
            method: HTTP method.
            path: URL path relative to ``base_url``.
            **kwargs: Extra arguments forwarded to ``httpx.AsyncClient.request``.

        Returns:
            The raw text response body.

        Raises:
            Same errors as ``_request``.
        """
        try:
            response = await self._client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.text
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
        """Send a request that is expected to return a JSON list.

        Some monitoring endpoints return a top-level list or wrap it in a
        ``{"data": [...]}`` envelope. This helper unwraps common envelopes
        automatically.

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
        for key in ("data", "results", "items", "nodes", "deployments"):
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
    # Core monitoring endpoints
    # ------------------------------------------------------------------

    async def get_metrics(self) -> dict[str, Any]:
        """Fetch Prometheus-style metrics from the ``/metrics`` endpoint.

        The raw Prometheus text is parsed into a structured dictionary via
        :func:`prom_text_to_dict`. If the endpoint returns JSON instead, the
        JSON is returned as-is (with Grafana-style normalisation applied).

        GET ``/metrics``

        Returns:
            A dictionary of metric names to their values / labelled samples.
            See :func:`prom_text_to_dict` for the detailed schema.
        """
        text = await self._request_text("GET", "/metrics")
        if text.startswith("{"):
            # JSON response — parse and normalise
            try:
                data = json.loads(text)
                return _parse_grafana_like(data)
            except json.JSONDecodeError:
                pass
        return prom_text_to_dict(text)

    async def get_health(self) -> dict[str, Any]:
        """Fetch the service health status.

        GET ``/health``

        Returns:
            A dictionary with health information. Typical keys include
            ``status`` (e.g. ``"healthy"``), ``version``, ``uptime_seconds``,
            and component-level health sub-objects.
        """
        return await self._request("GET", "/health")

    # ------------------------------------------------------------------
    # Admin-level monitoring endpoints
    # ------------------------------------------------------------------

    async def get_live_connections(self) -> int:
        """Get the number of currently active live connections.

        GET ``/admin/metrics/connections``

        Returns:
            The integer count of active connections to the coordinator.
        """
        data = await self._request("GET", "/admin/metrics/connections")
        # The endpoint may return {"connections": N} or a bare number.
        if isinstance(data.get("connections"), (int, float)):
            return int(data["connections"])
        if isinstance(data.get("count"), (int, float)):
            return int(data["count"])
        if isinstance(data.get("value"), (int, float)):
            return int(data["value"])
        # Fallback: return the first numeric value found
        for v in data.values():
            if isinstance(v, (int, float)):
                return int(v)
        return int(data.get("connections", 0))

    async def get_request_rate(self, window: str = "1m") -> dict[str, Any]:
        """Get the current request rate over the given time window.

        GET ``/admin/metrics/request-rate?window={window}``

        Args:
            window: The time window over which to compute the rate (e.g.
                ``"1m"``, ``"5m"``, ``"1h"``). Defaults to ``"1m"``.

        Returns:
            A dictionary with request-rate data. Typical keys include
            ``requests_per_second``, ``window``, and ``total_requests``.
        """
        return await self._request(
            "GET",
            "/admin/metrics/request-rate",
            params={"window": window},
        )

    async def get_latency_histogram(self) -> dict[str, Any]:
        """Get request latency percentiles (p50, p95, p99).

        GET ``/admin/metrics/latency``

        Returns:
            A dictionary containing latency statistics. Typical keys include
            ``p50_ms``, ``p95_ms``, ``p99_ms``, ``mean_ms``, ``min_ms``,
            ``max_ms``, and ``sample_count``.
        """
        return await self._request("GET", "/admin/metrics/latency")

    async def get_gpu_utilization(self) -> list[dict[str, Any]]:
        """Get per-GPU utilisation metrics across the cluster.

        GET ``/admin/metrics/gpu``

        Returns:
            A list of dictionaries, one per GPU. Each dict typically contains
            ``node_id``, ``gpu_index``, ``gpu_name``, ``utilization_pct``,
            ``memory_used_mb``, ``memory_total_mb``, ``memory_utilization_pct``,
            and ``temperature_celsius``.
        """
        return await self._request_list("GET", "/admin/metrics/gpu")

    async def get_error_rate(self) -> dict[str, Any]:
        """Get error-rate metrics for the cluster.

        GET ``/admin/metrics/errors``

        Returns:
            A dictionary with error-rate data. Typical keys include
            ``errors_per_second``, ``total_errors``, and a breakdown by
            status code or error type.
        """
        return await self._request("GET", "/admin/metrics/errors")

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get KV-cache and response-cache statistics.

        GET ``/admin/metrics/cache``

        Returns:
            A dictionary with cache statistics. Typical keys include
            ``hit_ratio``, ``miss_ratio``, ``total_hits``, ``total_misses``,
            ``cached_entries``, ``memory_used_mb``, and ``eviction_count``.
        """
        return await self._request("GET", "/admin/metrics/cache")

    async def get_token_throughput(self) -> dict[str, Any]:
        """Get token generation throughput metrics.

        GET ``/admin/metrics/throughput``

        Returns:
            A dictionary with throughput data. Typical keys include
            ``tokens_per_second``, ``prompt_tokens_per_second``,
            ``generation_tokens_per_second``, ``total_prompt_tokens``,
            ``total_generation_tokens``, and ``active_requests``.
        """
        return await self._request("GET", "/admin/metrics/throughput")

    async def get_system_resources(self) -> dict[str, Any]:
        """Get coordinator system resource utilisation.

        GET ``/admin/metrics/resources``

        Returns:
            A dictionary with system resource data. Typical keys include
            ``cpu_usage_pct``, ``memory_usage_pct``, ``memory_available_mb``,
            ``disk_usage_pct``, ``disk_free_mb``, ``load_average``, and
            ``network_io``.
        """
        return await self._request("GET", "/admin/metrics/resources")

    async def get_deployment_health(self) -> list[dict[str, Any]]:
        """Get health status for all active model deployments.

        GET ``/admin/health/deployments``

        Returns:
            A list of dictionaries, one per deployment. Each dict typically
            contains ``deployment_id``, ``model``, ``status``, ``healthy``,
            ``replicas``, ``ready_replicas``, ``uptime_seconds``, and
            ``last_checked``.
        """
        return await self._request_list("GET", "/admin/health/deployments")


# ---------------------------------------------------------------------------
# Error mapping (internal)
# ---------------------------------------------------------------------------


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
