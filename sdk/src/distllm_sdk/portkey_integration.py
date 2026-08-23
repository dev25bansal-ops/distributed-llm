"""Portkey observability integration for DistLLM SDK.

Wraps DistLLM client calls with Portkey's telemetry to provide
full observability (latency, cost, error tracking).

Usage::

    from distllm_sdk import DistLLMClient
    from distllm_sdk.portkey_integration import PortkeyMonitor

    monitor = PortkeyMonitor(api_key="pk-...")
    client = DistLLMClient(base_url="http://localhost:8000")

    # Wrap for automatic observability
    monitor.wrap(client)

    response = await client.chat_completions(...)
"""

from __future__ import annotations

import functools
import logging
from typing import Any

logger = logging.getLogger("distllm_sdk")


class PortkeyMonitor:
    """Attaches Portkey observability to a DistLLM client.

    Works by monkey-patching the client's request methods to
    inject Portkey's telemetry headers and logging.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        metadata: dict[str, str] | None = None,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._metadata = metadata or {}

    def wrap(self, client: Any) -> None:
        """Wrap a DistLLM client instance with Portkey monitoring.

        Intercepts ``_request`` and ``_request_raw`` to add
        Portkey headers and log telemetry.
        """
        if not hasattr(client, "_request"):
            logger.warning("PortkeyMonitor.wrap requires an object with _request method")
            return

        original_request = client._request
        monitor = self

        @functools.wraps(original_request)
        async def wrapped_request(method: str, path: str, **kwargs: Any) -> Any:
            # Add Portkey trace headers
            if "headers" not in kwargs:
                kwargs["headers"] = {}
            if monitor._api_key:
                kwargs["headers"]["x-portkey-api-key"] = monitor._api_key
            if monitor._base_url:
                kwargs["headers"]["x-portkey-base-url"] = monitor._base_url
            for k, v in monitor._metadata.items():
                kwargs["headers"][f"x-portkey-metadata-{k}"] = v

            return await original_request(method, path, **kwargs)

        client._request = wrapped_request

        # Also wrap sync if available
        if hasattr(client, "_request_sync"):
            original_sync = client._request_sync

            @functools.wraps(original_sync)
            def wrapped_sync(method: str, path: str, **kwargs: Any) -> Any:
                if "headers" not in kwargs:
                    kwargs["headers"] = {}
                if monitor._api_key:
                    kwargs["headers"]["x-portkey-api-key"] = monitor._api_key
                if monitor._base_url:
                    kwargs["headers"]["x-portkey-base-url"] = monitor._base_url
                return original_sync(method, path, **kwargs)

            client._request_sync = wrapped_sync

        logger.info("PortkeyMonitor wrapped %s", type(client).__name__)
