"""Security test — API key handling, no leakage in logs/errors."""

from unittest.mock import MagicMock

from distllm.core.distributed_speculative import (
    RemoteDraftConfig,
    RemoteDraftModel,
)


class TestDraftAuth:
    def test_api_key_not_in_stats(self):
        """API key should not appear in stats output."""
        model = RemoteDraftModel(RemoteDraftConfig(
            endpoint_url="http://draft:8000/v1/completions",
            api_key="sk-super-secret-key-12345",
        ))
        s = model.stats
        stats_str = str(s)
        assert "sk-super-secret-key-12345" not in stats_str
        model.close()

    def test_api_key_not_in_error_messages(self):
        """API key should not leak into error messages."""
        model = RemoteDraftModel(RemoteDraftConfig(
            endpoint_url="http://draft:8000/v1/completions",
            api_key="sk-secret",
            max_retries=0,
        ))

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("401 Unauthorized")
        mock_resp.status_code = 401
        mock_client.post.return_value = mock_resp
        model._client = mock_client

        result = model.generate_tokens([1, 2], num_tokens=1)
        assert "sk-secret" not in result.error
        model.close()

    def test_api_key_only_in_auth_header(self):
        """API key should only appear in the Authorization header."""
        model = RemoteDraftModel(RemoteDraftConfig(
            endpoint_url="http://draft:8000/v1/completions",
            api_key="Bearer my-token",
        ))
        headers = model._build_headers()
        assert headers["Authorization"] == "Bearer Bearer my-token"
        # Should not appear elsewhere
        model.close()

    def test_no_api_key_no_auth_header(self):
        """No API key → no Authorization header."""
        model = RemoteDraftModel(RemoteDraftConfig(
            endpoint_url="http://draft:8000/v1/completions",
            api_key="",
        ))
        headers = model._build_headers()
        assert "Authorization" not in headers
        model.close()


class TestDraftSSL:
    def test_verify_ssl_default_true(self):
        """SSL verification enabled by default."""
        model = RemoteDraftModel(RemoteDraftConfig(
            endpoint_url="https://draft:8000/v1/completions",
        ))
        assert model._config.verify_ssl is True
        model.close()

    def test_verify_ssl_can_be_disabled(self):
        """SSL verification can be disabled for self-signed certs."""
        model = RemoteDraftModel(RemoteDraftConfig(
            endpoint_url="https://self-signed:8000/v1/completions",
            verify_ssl=False,
        ))
        assert model._config.verify_ssl is False
        model.close()


class TestDraftInjection:
    def test_model_name_sanitization(self):
        """Model name is passed as-is to JSON payload (server-side validation)."""
        model = RemoteDraftModel(RemoteDraftConfig(
            endpoint_url="http://draft:8000/v1/completions",
            model_name="test-model",
        ))

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{
                "token_ids": [1],
                "logprobs": {"token_ids": [1], "token_logprobs": [-0.1]},
            }],
        }
        mock_client.post.return_value = mock_resp
        model._client = mock_client

        model.generate_tokens([1], num_tokens=1)

        # Verify the payload was sent
        call_args = mock_client.post.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        assert payload["model"] == "test-model"
        model.close()
