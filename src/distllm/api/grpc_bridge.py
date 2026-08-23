"""gRPC-Web and REST bridge for browser-based clients.

Translates OpenAI-compatible REST API calls to DistLLM gRPC calls,
enabling browser-based AI apps (LangChain, AI agents) to use DistLLM
as their backend without a custom server.

Architecture::

    Browser (OpenAI SDK) ──► FastAPI ──► gRPC ──► DistLLM Node
      HTTP POST /v1/chat      REST        protobuf     inference

Endpoints:
    - ``POST /v1/chat/completions`` — OpenAI-compatible chat completion
    - ``POST /v1/completions`` — OpenAI-compatible text completion
    - ``POST /v1/embeddings`` — OpenAI-compatible embeddings
    - ``GET /v1/models`` — List available models
    - ``GET /health`` — gRPC health check via REST
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from typing import Any

from loguru import logger


# ── OpenAI-compatible request/response models ──────────────────────────

class ChatCompletionRequest:
    """OpenAI-compatible chat completion request."""

    def __init__(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 1.0,
        max_tokens: int = 2048,
        stream: bool = False,
        **kwargs: Any,
    ):
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self.extra = kwargs

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatCompletionRequest":
        return cls(
            model=data.get("model", "default"),
            messages=data.get("messages", []),
            temperature=data.get("temperature", 1.0),
            max_tokens=data.get("max_tokens", 2048),
            stream=data.get("stream", False),
        )


class CompletionRequest:
    """OpenAI-compatible text completion request."""

    def __init__(
        self,
        model: str,
        prompt: str,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ):
        self.model = model
        self.prompt = prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra = kwargs

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompletionRequest":
        prompt = data.get("prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        return cls(
            model=data.get("model", "default"),
            prompt=prompt,
            temperature=data.get("temperature", 1.0),
            max_tokens=data.get("max_tokens", 2048),
        )


# ── Bridge implementation ──────────────────────────────────────────────

class GRPCBridge:
    """Translates OpenAI-compatible REST calls to DistLLM gRPC.

    Manages a pool of gRPC channels for high concurrency instead of
    a single channel that can bottleneck under load.

    Usage behind FastAPI::

        bridge = GRPCBridge(grpc_target="localhost:50051")

        @app.post("/v1/chat/completions")
        async def chat(request: dict):
            return await bridge.chat_completion(request)
    """

    def __init__(
        self,
        grpc_target: str = "localhost:50051",
        use_tls: bool = False,
        pool_size: int = 4,
    ):
        self._grpc_target = grpc_target
        self._use_tls = use_tls
        self._pool_size = max(1, pool_size)
        self._channels: list[Any] = []
        self._next_channel = 0
        self._lock = threading.RLock()

    async def _ensure_channel(self) -> Any:
        """Lazy-init gRPC channel pool with round-robin selection."""
        import grpc
        with self._lock:
            if len(self._channels) < self._pool_size:
                for _ in range(self._pool_size - len(self._channels)):
                    if self._use_tls:
                        ch = grpc.aio.secure_channel(
                            self._grpc_target,
                            grpc.ssl_channel_credentials(),
                        )
                    else:
                        ch = grpc.aio.insecure_channel(self._grpc_target)
                    self._channels.append(ch)

        # Round-robin channel selection
        with self._lock:
            idx = self._next_channel
            self._next_channel = (idx + 1) % len(self._channels)
        return self._channels[idx]

    async def chat_completion(
        self,
        request_data: dict[str, Any],
    ) -> dict[str, Any]:
        """OpenAI-compatible ``POST /v1/chat/completions``."""
        req = ChatCompletionRequest.from_dict(request_data)

        # Convert messages to a single prompt
        prompt = self._messages_to_prompt(req.messages)

        # Call gRPC
        result = await self._infer(prompt, req)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.get("text", ""),
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("generated_tokens", 0),
                "total_tokens": result.get("prompt_tokens", 0) + result.get("generated_tokens", 0),
            },
        }

    async def completion(
        self,
        request_data: dict[str, Any],
    ) -> dict[str, Any]:
        """OpenAI-compatible ``POST /v1/completions``."""
        req = CompletionRequest.from_dict(request_data)
        result = await self._infer(req.prompt, req)

        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "text": result.get("text", ""),
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("generated_tokens", 0),
                "total_tokens": result.get("prompt_tokens", 0) + result.get("generated_tokens", 0),
            },
        }

    async def list_models(self) -> dict[str, Any]:
        """OpenAI-compatible ``GET /v1/models``."""
        return {
            "object": "list",
            "data": [
                {"id": "default", "object": "model", "created": int(time.time()), "owned_by": "distllm"},
            ],
        }

    async def health(self) -> dict[str, Any]:
        """Health check — delegates to gRPC health probe."""
        try:
            channel = await self._ensure_channel()
            import grpc
            from grpc_health.v1 import health_pb2, health_pb2_grpc
            stub = health_pb2_grpc.HealthStub(channel)
            resp = stub.Check(health_pb2.HealthCheckRequest(service=""))
            return {
                "status": "serving" if resp.status == 1 else "not_serving",
                "grpc_target": self._grpc_target,
            }
        except Exception as e:
            return {"status": "unavailable", "error": str(e)}

    async def _infer(
        self,
        prompt: str,
        req: Any,
    ) -> dict[str, Any]:
        """Send inference request via gRPC.

        Requires compiled protobuf stubs in ``distllm.dist.node_pb2``
        and ``distllm.dist.node_pb2_grpc``.  If these are not available
        (pre-compilation), falls back to a simulated response so the
        REST API can still return structured output for testing.

        To compile protos::

            python -m grpc_tools.protoc \\
                -I=proto/ --python_out=src/distllm/dist \\
                --grpc_python_out=src/distllm/dist \\
                proto/node_service.proto
        """
        channel = await self._ensure_channel()

        try:
            from distllm.dist import node_pb2, node_pb2_grpc

            stub = node_pb2_grpc.NodeServiceStub(channel)
            grpc_request = node_pb2.InferRequest(
                model=req.model,
                prompt=prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            )
            grpc_response = await stub.Infer(grpc_request, timeout=60.0)
            return {
                "text": grpc_response.text,
                "prompt_tokens": grpc_response.prompt_tokens,
                "generated_tokens": grpc_response.generated_tokens,
            }
        except ImportError:
            logger.warning(
                "gRPC proto stubs not found — falling back to simulated response. "
                "Compile protos with: python -m grpc_tools.protoc ..."
            )
            await asyncio.sleep(0.05)  # Simulate minimal network latency
            return {
                "text": f"Simulated gRPC response for: {prompt[:50]}...",
                "prompt_tokens": len(prompt.split()),
                "generated_tokens": 64,
            }
        except Exception as e:
            logger.error(f"gRPC inference call failed: {e}")
            return {
                "text": "",
                "prompt_tokens": len(prompt.split()),
                "generated_tokens": 0,
                "error": str(e),
            }

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
        """Convert OpenAI chat messages to a single prompt string."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        return "\n".join(parts)

    async def close(self) -> None:
        if self._channel:
            await self._channel.close()
