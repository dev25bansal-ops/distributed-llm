"""gRPC client for DistLLM inference.

Provides a high-performance alternative to the REST API for latency-sensitive
applications.  Uses the DistLLM gRPC service on port 50051 by default.

Usage::

    from distllm_grpc import DistLLMGrpcClient

    async with DistLLMGrpcClient(host="localhost", port=50051) as client:
        # Chat completion
        response = await client.chat_completion(
            messages=[{"role": "user", "content": "Hello!"}],
            model="distributed-llm",
        )
        print(response.choices[0].message.content)

        # Streaming
        async for chunk in client.chat_completion_stream(
            messages=[{"role": "user", "content": "Tell me a story."}],
        ):
            print(chunk.delta, end="", flush=True)

        # Embeddings
        embeddings = await client.embeddings(
            input=["Hello world", "Goodbye"],
            model="bge-large",
        )
        print(len(embeddings[0]))
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger("distllm_grpc")


class DistLLMGrpcClient:
    """Async gRPC client for DistLLM.

    Uses ``grpcio`` for transport.  Falls back to REST if gRPC is unavailable.

    Parameters
    ----------
    host : str
        DistLLM gRPC host.
    port : int
        DistLLM gRPC port (default: 50051).
    timeout : float
        Request timeout in seconds.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50051,
        timeout: float = 120.0,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._channel: Any = None
        self._stub: Any = None

    async def __aenter__(self) -> DistLLMGrpcClient:
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def connect(self) -> None:
        """Establish gRPC channel."""
        try:
            import grpc

            self._channel = grpc.aio.insecure_channel(
                f"{self.host}:{self.port}",
                options=[
                    ("grpc.keepalive_time_ms", 30000),
                    ("grpc.keepalive_timeout_ms", 10000),
                    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                ],
            )
            # Import generated stubs if available, otherwise use generic
            try:
                from distllm_grpc.proto import inference_pb2_grpc
                self._stub = inference_pb2_grpc.InferenceServiceStub(self._channel)
            except ImportError:
                logger.warning(
                    "Generated gRPC stubs not found. "
                    "Run 'python -m grpc_tools.protoc' to generate them. "
                    "Falling back to JSON-over-gRPC."
                )
                self._stub = None
        except ImportError:
            raise ImportError("grpcio is required for the gRPC client. Install with: pip install grpcio")

    async def close(self) -> None:
        if self._channel:
            await self._channel.close()

    # ------------------------------------------------------------------
    # Chat completion
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str = "distributed-llm",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> Any:
        """Synchronous chat completion via gRPC."""
        if self._stub:
            return await self._chat_via_stub(messages, model, temperature, top_p, max_tokens)
        return await self._chat_via_json(messages, model, temperature, top_p, max_tokens)

    async def chat_completion_stream(
        self,
        messages: list[dict[str, str]],
        model: str = "distributed-llm",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Streaming chat completion via gRPC."""
        if self._stub:
            async for chunk in self._stream_via_stub(messages, model, temperature, top_p, max_tokens):
                yield chunk
        else:
            async for chunk in self._stream_via_json(messages, model, temperature, top_p, max_tokens):
                yield chunk

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def embeddings(
        self,
        input: list[str],
        model: str = "distributed-llm",
        **kwargs: Any,
    ) -> list[list[float]]:
        """Generate embeddings via gRPC."""
        if self._stub:
            return await self._embed_via_stub(input, model)
        return await self._embed_via_json(input, model)

    # ------------------------------------------------------------------
    # Internal: stub-based (when proto stubs are generated)
    # ------------------------------------------------------------------

    async def _chat_via_stub(self, messages, model, temperature, top_p, max_tokens):
        from distllm_grpc.proto import inference_pb2

        req = inference_pb2.ChatRequest(
            model=model,
            messages=[
                inference_pb2.ChatMessage(role=m["role"], content=m["content"])
                for m in messages
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        resp = await asyncio.wait_for(
            self._stub.ChatCompletion(req), timeout=self.timeout
        )
        return resp

    async def _stream_via_stub(self, messages, model, temperature, top_p, max_tokens):
        from distllm_grpc.proto import inference_pb2

        req = inference_pb2.ChatRequest(
            model=model,
            messages=[
                inference_pb2.ChatMessage(role=m["role"], content=m["content"])
                for m in messages
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in self._stub.ChatCompletionStream(req):
            yield chunk

    async def _embed_via_stub(self, input, model):
        from distllm_grpc.proto import inference_pb2

        req = inference_pb2.EmbeddingRequest(model=model, input=input)
        resp = await asyncio.wait_for(
            self._stub.Embeddings(req), timeout=self.timeout
        )
        return [list(e.values) for e in resp.embeddings]

    # ------------------------------------------------------------------
    # Internal: JSON-over-channel fallback
    # ------------------------------------------------------------------

    async def _chat_via_json(self, messages, model, temperature, top_p, max_tokens):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        # Use channel unary-unary with JSON payload
        resp = await self._channel.unary_unary(
            "/distllm.InferenceService/ChatCompletion",
            request_serializer=lambda x: json.dumps(x).encode(),
            response_deserializer=lambda x: json.loads(x),
        )(payload, timeout=self.timeout)
        return resp

    async def _stream_via_json(self, messages, model, temperature, top_p, max_tokens):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True,
        }
        async for chunk in self._channel.unary_stream(
            "/distllm.InferenceService/ChatCompletionStream",
            request_serializer=lambda x: json.dumps(x).encode(),
            response_deserializer=lambda x: json.loads(x),
        )(payload, timeout=self.timeout):
            yield chunk

    async def _embed_via_json(self, input, model):
        payload = {"model": model, "input": input}
        resp = await self._channel.unary_unary(
            "/distllm.InferenceService/Embeddings",
            request_serializer=lambda x: json.dumps(x).encode(),
            response_deserializer=lambda x: json.loads(x),
        )(payload, timeout=self.timeout)
        return resp
