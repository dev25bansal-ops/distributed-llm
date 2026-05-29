"""Mock draft model server fixture for integration testing.

Provides a FastAPI test server that simulates remote draft models
with configurable latency, accuracy, and error rates.

Usage::

    @pytest.fixture
    def draft_server():
        server = MockDraftServer(model_name="test-draft", accuracy=0.8)
        return server

    def test_with_server(draft_server):
        app = draft_server.create_app()
        # Use TestClient for testing
"""

from __future__ import annotations

import random
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class MockDraftServer:
    """Configurable mock draft model server for testing.

    Args:
        model_name: Name returned in responses.
        accuracy: Probability of returning the "correct" token.
        latency_ms: Simulated latency in milliseconds.
        error_rate: Probability of returning an error (0.0-1.0).
        tokens_per_request: Default number of tokens to generate.
    """

    def __init__(
        self,
        model_name: str = "mock-draft",
        accuracy: float = 0.8,
        latency_ms: float = 0.0,
        error_rate: float = 0.0,
        tokens_per_request: int = 5,
    ) -> None:
        self.model_name = model_name
        self.accuracy = accuracy
        self.latency_ms = latency_ms
        self.error_rate = error_rate
        self.tokens_per_request = tokens_per_request
        self.request_count = 0
        self.total_tokens = 0

    def create_app(self) -> FastAPI:
        """Create a FastAPI test application."""
        app = FastAPI(title="Mock Draft Server")

        @app.post("/v1/completions")
        async def completions(request: Request) -> Any:
            return self._handle_completions(request)

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request) -> Any:
            return self._handle_chat_completions(request)

        @app.get("/health")
        async def health() -> Any:
            return {"status": "healthy", "model": self.model_name}

        @app.get("/v1/models")
        async def models() -> Any:
            return {"data": [{"id": self.model_name, "object": "model"}]}

        return app

    def _handle_completions(self, request: Request) -> JSONResponse:
        """Handle /v1/completions requests."""
        self.request_count += 1

        # Simulate latency
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)

        # Simulate errors
        if random.random() < self.error_rate:
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )

        # Generate mock tokens
        num_tokens = self.tokens_per_request
        token_ids = [random.randint(0, 32000) for _ in range(num_tokens)]
        logprobs = [random.uniform(-2.0, -0.01) for _ in range(num_tokens)]

        self.total_tokens += num_tokens

        return JSONResponse(content={
            "id": f"mock-{self.request_count}",
            "object": "text_completion",
            "model": self.model_name,
            "choices": [{
                "index": 0,
                "token_ids": token_ids,
                "logprobs": {
                    "token_ids": token_ids,
                    "token_logprobs": logprobs,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": num_tokens,
                "total_tokens": num_tokens,
            },
        })

    def _handle_chat_completions(self, request: Request) -> JSONResponse:
        """Handle /v1/chat/completions requests."""
        self.request_count += 1

        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)

        if random.random() < self.error_rate:
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )

        num_tokens = self.tokens_per_request
        token_ids = [random.randint(0, 32000) for _ in range(num_tokens)]
        logprobs = [random.uniform(-2.0, -0.01) for _ in range(num_tokens)]

        self.total_tokens += num_tokens

        return JSONResponse(content={
            "id": f"mock-chat-{self.request_count}",
            "object": "chat.completion",
            "model": self.model_name,
            "choices": [{
                "index": 0,
                "token_ids": token_ids,
                "logprobs": {
                    "token_ids": token_ids,
                    "token_logprobs": logprobs,
                },
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }],
        })

    def get_stats(self) -> dict[str, Any]:
        return {
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "error_rate": self.error_rate,
        }
