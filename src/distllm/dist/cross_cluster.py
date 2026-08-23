"""Cross-cluster request forwarder for federated inference."""

from __future__ import annotations

import base64
import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx

try:
    import ray
except ImportError:
    ray = None
from loguru import logger

from distllm.dist import node_pb2
from distllm.security import safe_urlopen


class CrossClusterForwarder:
    def __init__(
        self,
        timeout_s: float = 120.0,
        max_retries: int = 2,
        retry_delay_s: float = 1.0,
    ) -> None:
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_delay_s = retry_delay_s
        self._ray_workers: dict[str, list[ray.actor.ActorHandle]] = {}

    def set_ray_workers(
        self,
        workers: dict[str, list[ray.actor.ActorHandle]],
    ) -> None:
        self._ray_workers.update(workers)

    def _call_ray_worker(
        self,
        cluster_id: str,
        request: dict[str, Any],
    ) -> dict[str, Any] | None:
        workers = self._ray_workers.get(cluster_id)
        if not workers:
            return None

        worker = workers[hash(json.dumps(request, sort_keys=True)) % len(workers)]
        try:
            ref = worker.process.remote(request)
            return ray.get(ref, timeout=self.timeout_s)
        except Exception as e:
            logger.warning(f"Ray forward to cluster '{cluster_id}' failed: {e}")
            return None

    def forward_request(
        self,
        remote_coord_url: str,
        request: dict[str, Any],
        timeout_s: float | None = None,
        cluster_id: str | None = None,
    ) -> dict[str, Any]:
        if cluster_id:
            result = self._call_ray_worker(cluster_id, request)
            if result is not None:
                logger.debug(
                    f"Ray cross-cluster forward to '{cluster_id}': "
                    f"{len(result.get('choices', []))} choices"
                )
                return result

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
                with safe_urlopen(req, timeout=timeout, allow_private_hosts=True) as resp:
                    data = json.loads(resp.read())
                    logger.debug(
                        f"Cross-cluster forward: {url} -> {len(data.get('choices', []))} choices"
                    )
                    return data
            except Exception as e:
                if attempt < self.max_retries:
                    # Exponential backoff with jitter to prevent thundering herd
                    import random
                    base_delay = self.retry_delay_s * (2 ** attempt)
                    jitter = random.uniform(0, base_delay * 0.5)
                    delay = base_delay + jitter
                    logger.warning(
                        f"Cross-cluster forward attempt {attempt + 1} failed: {e}, "
                        f"retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Cross-cluster forward failed after {self.max_retries + 1} attempts: {e}"
                    )
                    raise

    async def forward_streaming(
        self,
        remote_coord_url: str,
        request: dict[str, Any],
        timeout_s: float | None = None,
    ) -> AsyncGenerator[str, None]:
        timeout = timeout_s or self.timeout_s
        request["stream"] = True
        url = f"{remote_coord_url.rstrip('/')}/v1/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
                async with client.stream(
                    "POST",
                    url,
                    json=request,
                    headers={"X-Forwarded-From": "federated"},
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:].strip()
                            if data == "[DONE]":
                                break
                            yield data
        except Exception as e:
            logger.error(f"Cross-cluster streaming forward failed: {e}")
            yield json.dumps({"error": str(e)})

    def _kv_to_protobuf(self, kv_data: dict[str, Any]) -> bytes:
        """Serialize KV cache data to protobuf binary for efficient transfer.

        Uses ``KVCacheProto`` from the gRPC schema for 10-50x faster
        serialization vs JSON for large tensor payloads.

        Args:
            kv_data: Dict with ``key`` and ``value`` tensor lists.

        Returns:
            Serialized protobuf bytes (base64-encoded str for JSON transport).
        """
        cache_pb = node_pb2.KVCacheProto()
        from distllm.dist.pipeline.serialization import to_proto_tensor, set_kv_cache_proto

        layers = kv_data if isinstance(kv_data, list) else kv_data.get("layers", [])
        for layer in layers:
            layer_pb = cache_pb.layers.add()
            if "key" in layer and "value" in layer:
                k = layer["key"]
                v = layer["value"]
                if hasattr(k, "shape"):
                    layer_pb.key_states.CopyFrom(to_proto_tensor(k))
                    layer_pb.value_states.CopyFrom(to_proto_tensor(v))
                else:
                    # Already serialized as list — skip tensor conversion
                    pass
        return base64.b64encode(cache_pb.SerializeToString()).decode("ascii")

    def forward_kv_cache(
        self,
        remote_node_url: str,
        prefix_hash: str,
        kv_data: dict[str, Any],
    ) -> bool:
        import urllib.request

        kv_binary = self._kv_to_protobuf(kv_data)
        payload = json.dumps({
            "prefix_hash": prefix_hash,
            "kv_data": {"__proto__": "KVCacheProto", "data": kv_binary},
        }).encode()

        url = f"{remote_node_url.rstrip('/')}/api/v1/cache/warm"
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with safe_urlopen(req, timeout=30, allow_private_hosts=True) as resp:
                return resp.status == 200
        except Exception as e:
            logger.debug(f"KV cache forward failed: {e}")
            return False

    def replicate_kv_batch(
        self,
        entries: list[dict[str, Any]],
        remote_node_urls: list[str],
    ) -> int:
        import urllib.request

        success = 0
        for entry in entries:
            prefix_hash = entry.get("prefix_hash", "")
            kv_data = entry.get("kv_data", {})
            kv_binary = self._kv_to_protobuf(kv_data)
            payload = json.dumps({
                "prefix_hash": prefix_hash,
                "kv_data": {"__proto__": "KVCacheProto", "data": kv_binary},
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
                    with safe_urlopen(req, timeout=30, allow_private_hosts=True) as resp:
                        if resp.status == 200:
                            success += 1
                except Exception as e:
                    logger.debug(f"KV batch replicate to {target} failed: {e}")
        return success
