"""E2E tests for streaming completion endpoints."""

import pytest


@pytest.mark.e2e
class TestStreamingCompletionE2E:
    def test_streaming_chat_completion_returns_sse(self, e2e_api_client):
        """Streaming chat completion should return SSE content type."""
        response = e2e_api_client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        # SSE response contains "data:" lines
        assert "data:" in response.text

    def test_streaming_chat_completion_has_data_prefix(self, e2e_api_client):
        """SSE response lines should start with 'data:'."""
        response = e2e_api_client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 3,
                "stream": True,
            },
        )
        text = response.text
        lines = [l for l in text.split("\n") if l.strip()]
        data_lines = [l for l in lines if l.startswith("data:")]
        assert len(data_lines) > 0

    def test_streaming_text_completion_returns_sse(self, e2e_api_client):
        """Streaming text completion endpoint should return SSE."""
        response = e2e_api_client.post(
            "/v1/completions",
            json={
                "model": "distributed-llm",
                "prompt": "Once upon a time",
                "max_tokens": 3,
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        assert "data:" in response.text
