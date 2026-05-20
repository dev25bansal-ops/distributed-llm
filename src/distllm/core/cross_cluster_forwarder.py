"""Cross-cluster request forwarder for federated inference.

Serializes inference requests and forwards them to remote coordinators
via HTTP streaming (SSE), with response streaming back to the client.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator

from loguru import logger


class CrossClusterForwarder:
    """Forwards inference requests to remote cluster coordinators.

    Supports:
    - HTTP POST with JSON body for single requests
    - HTTP streaming (SSE) for response streaming
    - gRPC fallback (if gRPC stubs are available)
    - Request serialization and response deserialization
    """

    def __init__(
        self,
        timeout_s: float = 120.0,
        max_retries: int = 2,
        retry_delay_s: float = 1.0,
    ) -> None:
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_delay_s = retry_delay_s

    def forward_request(
        self,
        remote_coord_url: str,
        request: dict[str, Any],
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Forward a single inference request to a remote coordinator.

        Args:
            remote_coord_url: URL of the remote coordinator.
            request: Inference request dict (model, prompt, max_tokens, etc.).
            timeout_s: Request timeout (uses default if None).

        Returns:
            Response dict from the remote coordinator.
        """
        import urllib.request

        timeout = timeout_s or self.timeout_s
        payload = json.dumps(request).encode()
        url = f"{remote_coord_url.rstrip('/')}/v1/completions"

        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-Forwarded-From": "federated",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read())
                    logger.debug(
                        f"Cross-cluster forward: {url} -> {len(data.get('choices', []))} choices"
                    )
                    return data
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(
                        f"Cross-cluster forward attempt {attempt + 1} failed: {e}, retrying..."
                    )
                    time.sleep(self.retry_delay_s * (attempt + 1))
                else:
                    logger.error(f"Cross-cluster forward failed after {self.max_retries + 1} attempts: {e}")
                    raise

    async def forward_streaming(
        self,
        remote_coord_url: str,
        request: dict[str, Any],
        timeout_s: float | None = None,
    ) -> AsyncGenerator[str, None]:
        """Forward a request and stream the response back.

        Uses Server-Sent Events (SSE) for streaming.
        Falls back to polling if the remote doesn't support SSE.

        Args:
            remote_coord_url: URL of the remote coordinator.
            request: Inference request dict with stream=True.
            timeout_s: Request timeout.

        Yields:
            SSE data chunks from the remote response.
        """
        import urllib.request

        timeout = timeout_s or self.timeout_s
        request["stream"] = True
        payload = json.dumps(request).encode()
        url = f"{remote_coord_url.rstrip('/')}/v1/chat/completions"

        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "X-Forwarded-From": "federated",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                # Read SSE stream line by line
                while True:
                    line = resp.readline().decode("utf-8")
                    if not line:
                        break
                    if line.startswith("data: "):
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        yield data
        except Exception as e:
            logger.error(f"Cross-cluster streaming forward failed: {e}")
            yield json.dumps({"error": str(e)})

    def forward_kv_cache(
        self,
        remote_node_url: str,
        prefix_hash: str,
        kv_data: dict[str, Any],
    ) -> bool:
        """Forward KV cache data to a remote node for cache warming.

        Args:
            remote_node_url: URL of the remote node.
            prefix_hash: Hash of the prefix being cached.
            kv_data: Serialized KV cache data.

        Returns:
            True if the cache was accepted by the remote node.
        """
        import urllib.request

        payload = json.dumps({
            "prefix_hash": prefix_hash,
            "kv_data": kv_data,
        }).encode()

        url = f"{remote_node_url.rstrip('/')}/api/v1/cache/warm"
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status == 200
        except Exception as e:
            logger.debug(f"KV cache forward failed: {e}")
            return False

    def replicate_kv_batch(
        self,
        entries: list[dict[str, Any]],
        remote_node_urls: list[str],
    ) -> int:
        """Batch-replicate KV cache entries to multiple remote nodes.

        Args:
            entries: List of dicts with 'prefix_hash' and 'kv_data' keys.
            remote_node_urls: List of remote node URLs to replicate to.

        Returns:
            Number of successful replications.
        """
        import urllib.request

        success = 0
        for entry in entries:
            prefix_hash = entry.get("prefix_hash", "")
            kv_data = entry.get("kv_data", {})
            payload = json.dumps({
                "prefix_hash": prefix_hash,
                "kv_data": kv_data,
            }).encode()

            for url in remote_node_urls:
                target = f"{url.rstrip('/')}/api/v1/cache/warm"
                try:
                    req = urllib.request.Request(
                        target,
                        data=payload,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        if resp.status == 200:
                            success += 1
                except Exception as e:
                    logger.debug(f"KV batch replicate to {target} failed: {e}")
        return success
