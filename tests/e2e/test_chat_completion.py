"""E2E tests for chat completion endpoints."""

import pytest


@pytest.mark.e2e
class TestChatCompletionE2E:
    def test_chat_completion_returns_valid_response(self, e2e_api_client):
        """Send a chat completion request and verify the response format."""
        response = e2e_api_client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]

    def test_chat_completion_with_system_message(self, e2e_api_client):
        """Chat completion with system + user messages."""
        response = e2e_api_client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi"},
                ],
                "max_tokens": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]

    def test_chat_completion_includes_model_field(self, e2e_api_client):
        """Response should include the model field."""
        response = e2e_api_client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 3,
            },
        )
        data = response.json()
        assert data.get("model") == "distributed-llm"

    def test_chat_completion_has_id_and_created(self, e2e_api_client):
        """Response should have id and created fields."""
        response = e2e_api_client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 3,
            },
        )
        data = response.json()
        assert "id" in data
        assert "created" in data
        assert isinstance(data["created"], int)
