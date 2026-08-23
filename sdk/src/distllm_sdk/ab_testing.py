"""ABTestClient for A/B testing between models on a DistLLM cluster.

Provides an async client for creating, managing, and evaluating A/B tests that
compare two models side-by-side. Tests can be run server-side via the admin API
or locally on the client by sending identical prompts to both models and
collecting latency, error-rate, and throughput statistics.

Usage::

    from distllm_sdk.ab_testing import ABTestClient, ABTestConfig

    async with ABTestClient(
        base_url="http://coordinator:8000",
        api_key="sk-...",
    ) as ab:
        test_id = await ab.create_test(
            ABTestConfig(
                name="llama-vs-mistral",
                model_a="llama-3-8b",
                model_b="mistral-7b",
                traffic_ratio_a=0.5,
            )
        )
        result = await ab.get_test(test_id)
        print(f"Winner: {result.winner}")
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import dataclass
from typing import Any

import httpx

from distllm_sdk.constants import DEFAULT_HTTP_TIMEOUT
from distllm_sdk.errors import ApiError, AuthenticationError, RateLimitError, TimeoutError

_logger = logging.getLogger("distllm_sdk.ab_testing")

__all__ = [
    "ABTestConfig",
    "ABTestResult",
    "ABTestClient",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ABTestConfig:
    """Configuration for an A/B test between two models.

    Attributes:
        name: Human-readable label for the test.
        model_a: Identifier of the first (control or candidate) model.
        model_b: Identifier of the second (candidate) model.
        traffic_ratio_a: Fraction of live traffic routed to **model_a**
            (0.0 = all traffic to model B, 1.0 = all traffic to model A).
        metrics_collection_interval: How often (in seconds) the coordinator
            aggregates metrics for this test.
        minimal_sample_size: Minimum number of requests per model before a
            winner can be declared.
    """

    name: str
    model_a: str
    model_b: str
    traffic_ratio_a: float = 0.5
    metrics_collection_interval: int = 60
    minimal_sample_size: int = 100


@dataclass
class ABTestResult:
    """Result of a completed or in-progress A/B test.

    Attributes:
        test_name: Human-readable label for the test.
        model_a: Identifier of the first model.
        model_b: Identifier of the second model.
        total_requests_a: Number of requests served by **model_a**.
        total_requests_b: Number of requests served by **model_b**.
        avg_latency_a: Average request latency for **model_a** in seconds.
        avg_latency_b: Average request latency for **model_b** in seconds.
        error_rate_a: Fraction of requests that errored on **model_a** (0-1).
        error_rate_b: Fraction of requests that errored on **model_b** (0-1).
        tokens_per_second_a: Mean token generation throughput for **model_a**.
        tokens_per_second_b: Mean token generation throughput for **model_b**.
        winner: Model identifier that won, or ``None`` if the test is
            inconclusive.
        confidence: Statistical confidence in the winner (0.0-1.0), or 0.0
            when no winner has been determined.
    """

    test_name: str
    model_a: str
    model_b: str
    total_requests_a: int = 0
    total_requests_b: int = 0
    avg_latency_a: float = 0.0
    avg_latency_b: float = 0.0
    error_rate_a: float = 0.0
    error_rate_b: float = 0.0
    tokens_per_second_a: float = 0.0
    tokens_per_second_b: float = 0.0
    winner: str | None = None
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
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


def _build_headers(api_key: str | None = None) -> dict[str, str]:
    """Build standard JSON headers with optional bearer auth."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class ABTestClient:
    """Async HTTP client for the DistLLM A/B testing admin API.

    Create, monitor, and manage A/B tests that compare two models. Tests can
    be orchestrated server-side (the coordinator routes a configurable fraction
    of live traffic to each model) or run locally via
    :meth:`run_local_comparison`.

    Args:
        base_url: Root URL of the DistLLM coordinator (e.g.
            ``http://coordinator:8000``).
        api_key: Optional API key for authentication. Sent as a ``Bearer``
            token in the ``Authorization`` header.
        timeout: Default HTTP request timeout in seconds.

    Example::

        client = ABTestClient(
            base_url="http://10.0.0.1:8000",
            api_key="sk-my-key",
        )
        async with client:
            test_id = await client.create_test(
                ABTestConfig(
                    name="gpt-vs-llama",
                    model_a="gpt-4o-mini",
                    model_b="llama-3-8b",
                    traffic_ratio_a=0.5,
                )
            )
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=_build_headers(api_key),
            timeout=httpx.Timeout(self._timeout),
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> ABTestClient:
        """Enter the async context manager.

        Returns:
            The ``ABTestClient`` instance, ready for use.
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
            method: HTTP method (``"GET"``, ``"POST"``, ``"PUT"``, ``"DELETE"``).
            path: URL path relative to ``base_url`` (e.g. ``/admin/ab-tests``).
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

    # ------------------------------------------------------------------
    # A/B test CRUD
    # ------------------------------------------------------------------

    async def create_test(self, config: ABTestConfig) -> str:
        """Create a new A/B test on the coordinator.

        POST ``/admin/ab-tests``

        The coordinator will begin routing traffic to both models according to
        the ``traffic_ratio_a`` configured in *config*.

        Args:
            config: The A/B test configuration describing the two models,
                traffic split, collection interval, and sample size
                requirements.

        Returns:
            The unique identifier (``test_id``) of the created test.

        Raises:
            ApiError: If the request is rejected (e.g. invalid model names,
                duplicate test name).
        """
        payload: dict[str, Any] = {
            "name": config.name,
            "model_a": config.model_a,
            "model_b": config.model_b,
            "traffic_ratio_a": config.traffic_ratio_a,
            "metrics_collection_interval": config.metrics_collection_interval,
            "minimal_sample_size": config.minimal_sample_size,
        }
        data = await self._request("POST", "/admin/ab-tests", json=payload)
        test_id: str = data.get("test_id", data.get("id", ""))
        return test_id

    async def list_tests(self) -> list[dict[str, Any]]:
        """List all A/B tests known to the coordinator.

        GET ``/admin/ab-tests``

        Returns:
            A list of dictionaries, each representing an A/B test with its
            current configuration and aggregate metrics.
        """
        data = await self._request("GET", "/admin/ab-tests")
        if isinstance(data, list):
            return data
        # Unwrap common envelope keys
        for key in ("data", "tests", "results", "items"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]

    async def get_test(self, test_id: str) -> ABTestResult:
        """Get the current result and metrics for a specific A/B test.

        GET ``/admin/ab-tests/{test_id}``

        Args:
            test_id: The unique identifier of the test, as returned by
                :meth:`create_test`.

        Returns:
            An ``ABTestResult`` instance with aggregate statistics for both
            models and, if enough data has been collected, the inferred
            winner and confidence score.

        Raises:
            ApiError: If the test is not found (status 404).
        """
        data = await self._request("GET", f"/admin/ab-tests/{test_id}")
        return self._parse_result(data)

    async def stop_test(self, test_id: str) -> dict[str, Any]:
        """Stop an active A/B test.

        POST ``/admin/ab-tests/{test_id}/stop``

        No further traffic will be routed to either model for this test.
        Existing metrics are preserved and can still be retrieved via
        :meth:`get_test`.

        Args:
            test_id: The unique identifier of the test to stop.

        Returns:
            A status dictionary confirming the test has been stopped.
        """
        return await self._request("POST", f"/admin/ab-tests/{test_id}/stop")

    async def promote_winner(self, test_id: str, model: str) -> dict[str, Any]:
        """Promote one model as the winner of an A/B test.

        POST ``/admin/ab-tests/{test_id}/promote``

        This tells the coordinator to use the selected model as the default
        for the deployment that was under test. The promoted model is recorded
        in the test history.

        Args:
            test_id: The unique identifier of the test.
            model: The model identifier to promote (must be one of the
                ``model_a`` or ``model_b`` values from the test config).

        Returns:
            A status dictionary confirming the promotion.

        Raises:
            ApiError: If the test is not found or the model name does not
                match either arm of the test.
        """
        payload: dict[str, Any] = {"model": model}
        return await self._request(
            "POST",
            f"/admin/ab-tests/{test_id}/promote",
            json=payload,
        )

    async def delete_test(self, test_id: str) -> bool:
        """Delete an A/B test and its collected metrics.

        DELETE ``/admin/ab-tests/{test_id}``

        Args:
            test_id: The unique identifier of the test to delete.

        Returns:
            ``True`` if the test was successfully deleted, ``False`` otherwise.
        """
        data = await self._request("DELETE", f"/admin/ab-tests/{test_id}")
        return data.get("deleted", False)

    # ------------------------------------------------------------------
    # Local comparison (client-side)
    # ------------------------------------------------------------------

    async def run_local_comparison(
        self,
        model_a: str,
        model_b: str,
        prompts: list[str],
        **kwargs: Any,
    ) -> ABTestResult:
        """Run the same prompts against both models locally and compare results.

        This method creates two ephemeral :class:`distllm_sdk.DistLLMClient`
        instances (sharing the same ``base_url`` and ``api_key``) and sends each
        prompt to both models. Statistics (latency, error rate, tokens per
        second) are collected client-side and returned as an ``ABTestResult``.

        Args:
            model_a: Identifier of the first model.
            model_b: Identifier of the second model.
            prompts: List of prompt strings to send to both models.
            **kwargs: Extra keyword arguments forwarded to both models' chat
                completion calls (e.g. ``temperature``, ``max_tokens``,
                ``top_p``).

        Returns:
            An ``ABTestResult`` summarising the comparison. The ``winner``
            field is set if one model has statistically significantly better
            average latency (p < 0.05, two-sample t-test approximation).

        Raises:
            ImportError: If the ``scipy`` package is not installed (required
                for statistical significance testing).
        """
        from distllm_sdk.client import DistLLMClient

        async with DistLLMClient(
            base_url=self.base_url,
            api_key=self._api_key,
        ) as client_a, DistLLMClient(
            base_url=self.base_url,
            api_key=self._api_key,
        ) as client_b:
            stats_a = await self._collect_stats(client_a, model_a, prompts, **kwargs)
            stats_b = await self._collect_stats(client_b, model_b, prompts, **kwargs)

        winner: str | None = None
        confidence: float = 0.0

        n_a = len(stats_a["latencies"])
        n_b = len(stats_b["latencies"])

        if n_a > 1 and n_b > 1:
            mean_a = statistics.mean(stats_a["latencies"])
            mean_b = statistics.mean(stats_b["latencies"])
            var_a = statistics.variance(stats_a["latencies"]) if n_a > 1 else 0.0
            var_b = statistics.variance(stats_b["latencies"]) if n_b > 1 else 0.0

            # Welch's t-test approximation
            se = ((var_a / n_a) + (var_b / n_b)) ** 0.5
            if se > 0:
                t_stat = (mean_a - mean_b) / se
                # Approximate degrees of freedom (Welch-Satterthwaite)
                num = ((var_a / n_a) + (var_b / n_b)) ** 2
                denom = (
                    ((var_a / n_a) ** 2) / (n_a - 1)
                    + ((var_b / n_b) ** 2) / (n_b - 1)
                )
                dof = num / denom if denom > 0 else min(n_a, n_b) - 1

                # Cumulative t-distribution approximation (Abramowitz & Stegun)
                # Only used when scipy is unavailable.
                try:
                    from scipy.stats import t as t_dist

                    p_value = t_dist.sf(abs(t_stat), df=dof) * 2.0
                except ImportError:
                    p_value = self._approx_t_sf(abs(t_stat), dof) * 2.0

                confidence = 1.0 - p_value

                if confidence > 0.95 and stats_a["error_rate"] < 0.5 and stats_b["error_rate"] < 0.5:
                    winner = model_a if mean_a < mean_b else model_b
            else:
                # Identical variance -- no winner
                winner = None
                confidence = 0.0

        return ABTestResult(
            test_name=f"local:{model_a}_vs_{model_b}",
            model_a=model_a,
            model_b=model_b,
            total_requests_a=n_a,
            total_requests_b=n_b,
            avg_latency_a=statistics.mean(stats_a["latencies"]) if stats_a["latencies"] else 0.0,
            avg_latency_b=statistics.mean(stats_b["latencies"]) if stats_b["latencies"] else 0.0,
            error_rate_a=stats_a["error_rate"],
            error_rate_b=stats_b["error_rate"],
            tokens_per_second_a=stats_a["tps"],
            tokens_per_second_b=stats_b["tps"],
            winner=winner,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Stats collection helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _collect_stats(
        client: Any,
        model: str,
        prompts: list[str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run prompts against a model and collect performance statistics.

        Args:
            client: A ``DistLLMClient`` instance.
            model: Model identifier to query.
            prompts: List of prompt strings to send.
            **kwargs: Extra keyword arguments forwarded to
                ``client.chat_completions()``.

        Returns:
            A dictionary with the following keys:

            - ``latencies`` (list[float]): Per-request wall-clock latencies in
              seconds.
            - ``error_rate`` (float): Fraction of requests that failed (0-1).
            - ``tps`` (float): Mean tokens-per-second across successful
              requests.
        """
        latencies: list[float] = []
        errors: int = 0
        tps_values: list[float] = []

        for prompt in prompts:
            messages: list[dict[str, str]] = [
                {"role": "user", "content": prompt},
            ]
            start = time.monotonic()
            try:
                response = await client.chat_completions(
                    messages=messages,
                    model=model,
                    **kwargs,
                )
                elapsed = time.monotonic() - start
                latencies.append(elapsed)
                if response.usage and response.usage.tokens_per_second > 0:
                    tps_values.append(response.usage.tokens_per_second)
            except Exception:
                elapsed = time.monotonic() - start
                latencies.append(elapsed)
                errors += 1

        total = len(prompts)
        return {
            "latencies": latencies,
            "error_rate": errors / total if total > 0 else 0.0,
            "tps": statistics.mean(tps_values) if tps_values else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_result(data: dict[str, Any]) -> ABTestResult:
        """Convert a raw response dict into an ``ABTestResult``.

        Args:
            data: Response dictionary from the A/B test API endpoint.

        Returns:
            A typed ``ABTestResult`` dataclass instance.
        """
        return ABTestResult(
            test_name=data.get("test_name", data.get("name", "")),
            model_a=data.get("model_a", ""),
            model_b=data.get("model_b", ""),
            total_requests_a=data.get("total_requests_a", data.get("requests_a", 0)),
            total_requests_b=data.get("total_requests_b", data.get("requests_b", 0)),
            avg_latency_a=data.get("avg_latency_a", data.get("latency_a", 0.0)),
            avg_latency_b=data.get("avg_latency_b", data.get("latency_b", 0.0)),
            error_rate_a=data.get("error_rate_a", data.get("error_a", 0.0)),
            error_rate_b=data.get("error_rate_b", data.get("error_b", 0.0)),
            tokens_per_second_a=data.get("tokens_per_second_a", data.get("tps_a", 0.0)),
            tokens_per_second_b=data.get("tokens_per_second_b", data.get("tps_b", 0.0)),
            winner=data.get("winner"),
            confidence=data.get("confidence", 0.0),
        )

    @staticmethod
    def _approx_t_sf(x: float, dof: float) -> float:
        """Approximate the survival function of Student's t distribution.

        Uses the Abramowitz & Stegun formula 26.7.10 (Zelen & Severo) to
        approximate the standard normal CDF, then applies Hill's approximation
        to map the t-statistic to a normal z-score when the degrees of freedom
        are large. For smaller *dof* the result is a reasonable heuristic for
        confidence thresholding.

        Args:
            x: The t-statistic (positive).
            dof: Degrees of freedom (>= 1).

        Returns:
            Right-tail probability P(T > x), clamped to [0, 1].
        """
        if dof < 1.0:
            dof = 1.0

        if dof > 100:
            from math import erfc

            return 0.5 * erfc(x / 2**0.5)

        # Hill's approximation of t -> normal z-score
        from math import exp, pi, sqrt

        a = dof - 0.5
        b = 48.0 * a * a
        z = (
            (a * sqrt((1.0 + (x * x) / (2.0 * a)) / (1.0 + (x * x) / (dof))))
            - (0.5 + 1.0 / (4.0 * a))
            * (1.0 + (1.0 + 2.0 / (3.0 * a)) / b)
            / (1.0 + (0.5 + (1.0 - 1.0 / (6.0 * a)) / (2.0 * a)) / b)
        )

        if z < 0:
            return 1.0  # tail probability is large, not meaningful

        # Abramowitz & Stegun 26.2.17 for normal CDF tail
        b0 = 0.2316419
        c1, c2, c3, c4, c5 = (
            0.319381530,
            -0.356563782,
            1.781477937,
            -1.821255978,
            1.330274429,
        )
        t = 1.0 / (1.0 + b0 * z)
        phi = (1.0 / sqrt(2.0 * pi)) * exp(-z * z / 2.0)
        p = phi * (c1 * t + c2 * t**2 + c3 * t**3 + c4 * t**4 + c5 * t**5)

        return max(0.0, min(1.0, p))
