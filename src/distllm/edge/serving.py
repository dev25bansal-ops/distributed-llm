"""Edge inference server: lightweight HTTP API for edge devices.

Serves OpenAI-compatible chat completions using quantized models
with automatic fallback to cloud cluster when overloaded.
"""

import json
import os
import time
from typing import Optional

import httpx
from loguru import logger

from distllm.edge.models import EdgeConfig, QuantizationType
from distllm.edge.quantized import QuantizedModel
from distllm.edge.routing import EdgeRouter
from distllm.edge.sharding import ModelShardManager


class EdgeInferenceServer:
    """Lightweight inference server for edge deployment.

    Loads quantized models, serves requests locally, and falls
    back to the cloud cluster when capacity is exceeded.
    """

    def __init__(self, config: EdgeConfig):
        self.config = config
        self._models: dict[str, QuantizedModel] = {}
        self._shard_manager = ModelShardManager(config.shard_dir)
        self._router = EdgeRouter(config)
        self._start_time = time.time()
        self._active_requests = 0
        self._queue_depth = 0
        self._total_requests = 0

    async def start(self) -> None:
        for model_name in self.config.models:
            model = QuantizedModel(
                model_name=model_name,
                quant_type=self.config.quantization,
                device=self.config.device,
            )
            model.load()
            self._models[model_name] = model
        logger.info(f"Edge server started: {self.config.node_id} ({len(self._models)} models)")

    async def stop(self) -> None:
        self._models.clear()

    async def chat_completion(self, body: dict, headers: dict | None = None) -> dict:
        """Handle a chat completion request with edge-first routing."""
        self._active_requests += 1
        self._total_requests += 1

        try:
            decision = self._router.decide(body, self._active_requests)

            if decision == "cloud":
                sanitized_headers = self._sanitize_headers(headers)
                return await self._cloud_fallback(body, sanitized_headers)

            model_name = body.get("model", "")
            if model_name not in self._models:
                if self.config.models:
                    model_name = self.config.models[0]
                else:
                    return await self._cloud_fallback(body, self._sanitize_headers(headers))

            model = self._models[model_name]
            return await model.generate(
                messages=body.get("messages", []),
                max_tokens=body.get("max_tokens", 128),
                temperature=body.get("temperature", 0.7),
            )
        finally:
            self._active_requests -= 1

    async def chat_completion_stream(self, body: dict, headers: dict | None = None):
        self._active_requests += 1
        self._total_requests += 1
        try:
            decision = self._router.decide(body, self._active_requests)
            if decision == "cloud":
                sanitized_headers = self._sanitize_headers(headers)
                async for chunk in self._cloud_fallback_stream(body, sanitized_headers):
                    yield chunk
                return

            model_name = body.get("model", "")
            if model_name not in self._models:
                if self.config.models:
                    model_name = self.config.models[0]
                else:
                    sanitized_headers = self._sanitize_headers(headers)
                    async for chunk in self._cloud_fallback_stream(body, sanitized_headers):
                        yield chunk
                    return

            model = self._models[model_name]
            async for chunk in model.generate_stream(
                messages=body.get("messages", []),
                max_tokens=body.get("max_tokens", 128),
                temperature=body.get("temperature", 0.7),
            ):
                yield chunk
        finally:
            self._active_requests -= 1

    def get_health(self) -> dict:
        try:
            import psutil
            cpu_pct = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            mem_pct = mem.percent
        except (ImportError, Exception):
            cpu_pct = 0.0
            mem_pct = 0.0
        return {
            "node_id": self.config.node_id,
            "healthy": True,
            "cpu_usage_pct": cpu_pct,
            "memory_usage_pct": mem_pct,
            "active_requests": self._active_requests,
            "queue_depth": self._queue_depth,
            "uptime_s": time.time() - self._start_time,
            "total_requests": self._total_requests,
            "models_deployed": list(self._models.keys()),
        }

    @staticmethod
    def _sanitize_headers(headers: dict | None) -> dict:
        if headers is None:
            return {}
        sensitive = {"authorization", "cookie", "x-api-key", "api-key", "proxy-authorization"}
        return {k: v for k, v in headers.items() if k.lower() not in sensitive}

    async def _cloud_fallback(self, body: dict, headers: dict | None = None) -> dict:
        logger.info(f"Edge fallback to cloud: {self.config.cloud_fallback_url}")
        hdrs = {"Content-Type": "application/json"}
        if headers:
            hdrs.update(self._sanitize_headers(headers))
        async with httpx.AsyncClient(timeout=self.config.cloud_fallback_timeout_s) as client:
            resp = await client.post(
                f"{self.config.cloud_fallback_url}/v1/chat/completions",
                json=body,
                headers=hdrs,
            )
            return resp.json()

    async def _cloud_fallback_stream(self, body: dict, headers: dict | None = None):
        logger.info(f"Edge stream fallback to cloud: {self.config.cloud_fallback_url}")
        hdrs = {"Content-Type": "application/json"}
        if headers:
            hdrs.update(self._sanitize_headers(headers))
        async with httpx.AsyncClient(timeout=self.config.cloud_fallback_timeout_s) as client:
            async with client.stream(
                "POST",
                f"{self.config.cloud_fallback_url}/v1/chat/completions",
                json={**body, "stream": True},
                headers=hdrs,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.strip():
                        yield f"data: {line}\n\n"
