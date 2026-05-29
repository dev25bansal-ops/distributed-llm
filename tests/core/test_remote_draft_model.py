"""Unit tests for RemoteDraftModel — HTTP response parsing, retry, errors."""

from unittest.mock import MagicMock


from distllm.core.distributed_speculative import (
    DraftTokenResult,
    RemoteDraftConfig,
    RemoteDraftModel,
    _CompletionsResponse,
    _ChatCompletionsResponse,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_config(**overrides) -> RemoteDraftConfig:
    defaults = {
        "endpoint_url": "http://draft:8000/v1/completions",
        "model_name": "test-draft",
        "max_retries": 1,
        "timeout_seconds": 5.0,
    }
    defaults.update(overrides)
    return RemoteDraftConfig(**defaults)


def _completions_response(token_ids, logprobs=None):
    """Build a valid completions-style response dict."""
    choice = {"token_ids": token_ids, "index": 0}
    if logprobs is not None:
        choice["logprobs"] = {"token_ids": token_ids, "token_logprobs": logprobs}
    return {"choices": [choice], "model": "test"}


def _chat_completions_response(token_ids, logprobs=None):
    """Build a valid chat completions-style response dict."""
    choice = {
        "token_ids": token_ids,
        "index": 0,
        "message": {"role": "assistant", "content": ""},
    }
    if logprobs is not None:
        choice["logprobs"] = {
            "token_ids": token_ids,
            "token_logprobs": logprobs,
        }
    return {"choices": [choice], "model": "test"}


def _openai_text_response(tokens, token_logprobs):
    """Build OpenAI-style text completions response with logprobs."""
    return {
        "choices": [{
            "text": "",
            "index": 0,
            "logprobs": {
                "tokens": tokens,
                "token_logprobs": token_logprobs,
            },
        }],
    }


# ── DraftTokenResult ─────────────────────────────────────────────────────


class TestDraftTokenResult:
    def test_ok_with_tokens(self):
        r = DraftTokenResult(token_ids=[1, 2], logprobs=[-0.1, -0.2])
        assert r.ok is True

    def test_not_ok_empty(self):
        r = DraftTokenResult(token_ids=[], logprobs=[])
        assert r.ok is False

    def test_not_ok_with_error(self):
        r = DraftTokenResult(token_ids=[1], logprobs=[-0.1], error="fail")
        assert r.ok is False


# ── Response parsing ─────────────────────────────────────────────────────


class TestExtractFromCompletions:
    def test_token_ids_in_choice(self):
        model = RemoteDraftModel(_make_config())
        data = _completions_response([10, 20, 30], [-0.1, -0.2, -0.3])
        result = model._extract_from_completions(data)
        assert result.ok
        assert result.token_ids == [10, 20, 30]
        assert len(result.logprobs) == 3

    def test_token_ids_in_logprobs(self):
        model = RemoteDraftModel(_make_config())
        data = {"choices": [{"logprobs": {"token_ids": [5, 6], "token_logprobs": [-0.5, -0.6]}}]}
        result = model._extract_from_completions(data)
        assert result.ok
        assert result.token_ids == [5, 6]

    def test_openai_text_format(self):
        model = RemoteDraftModel(_make_config())
        data = _openai_text_response([7, 8, 9], [-0.7, -0.8, -0.9])
        result = model._extract_from_completions(data)
        assert result.ok
        assert result.token_ids == [7, 8, 9]

    def test_top_level_tokens(self):
        model = RemoteDraftModel(_make_config())
        data = {"tokens": [1, 2], "choices": []}
        result = model._extract_from_completions(data)
        # tokens at top level with empty choices — validated model catches this
        assert result.token_ids == [1, 2] or not result.ok

    def test_no_choices(self):
        model = RemoteDraftModel(_make_config())
        data = {"error": "bad request"}
        result = model._extract_from_completions(data)
        assert not result.ok
        assert "Unexpected" in result.error or "No choices" in result.error

    def test_empty_choice(self):
        model = RemoteDraftModel(_make_config())
        data = {"choices": [{}]}
        result = model._extract_from_completions(data)
        assert not result.ok


class TestExtractFromChatResponse:
    def test_token_ids_in_choice(self):
        model = RemoteDraftModel(_make_config())
        data = _chat_completions_response([10, 20], [-0.1, -0.2])
        result = model._extract_tokens_from_chat_response(data)
        assert result.ok
        assert result.token_ids == [10, 20]

    def test_no_choices(self):
        model = RemoteDraftModel(_make_config())
        data = {"choices": []}
        result = model._extract_tokens_from_chat_response(data)
        assert not result.ok

    def test_content_only_no_token_ids(self):
        model = RemoteDraftModel(_make_config())
        data = {"choices": [{"message": {"content": "hello"}, "index": 0}]}
        result = model._extract_tokens_from_chat_response(data)
        assert not result.ok
        assert "no token IDs" in result.error.lower() or "empty" in result.error.lower()


# ── Pydantic validation ─────────────────────────────────────────────────


class TestPydanticValidation:
    def test_completions_response_valid(self):
        resp = _CompletionsResponse.model_validate(
            _completions_response([1, 2], [-0.1, -0.2])
        )
        assert len(resp.choices) == 1
        assert resp.choices[0].token_ids == [1, 2]

    def test_chat_completions_response_valid(self):
        resp = _ChatCompletionsResponse.model_validate(
            _chat_completions_response([3, 4], [-0.3, -0.4])
        )
        assert len(resp.choices) == 1
        assert resp.choices[0].token_ids == [3, 4]

    def test_completions_response_missing_choices(self):
        resp = _CompletionsResponse.model_validate({"error": "bad"})
        assert resp.choices == []


# ── generate_tokens (mocked HTTP) ────────────────────────────────────────


class TestGenerateTokens:
    def test_successful_call(self):
        model = RemoteDraftModel(_make_config())
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _completions_response([10, 20], [-0.1, -0.2])
        mock_client.post.return_value = mock_resp
        model._client = mock_client

        result = model.generate_tokens([1, 2, 3], num_tokens=2)
        assert result.ok
        assert result.token_ids == [10, 20]
        assert model._stats.total_calls == 1
        assert model._stats.total_tokens == 2

    def test_http_error_returns_empty(self):
        model = RemoteDraftModel(_make_config(max_retries=0))
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("500 Server Error")
        mock_resp.status_code = 500
        mock_client.post.return_value = mock_resp
        model._client = mock_client

        result = model.generate_tokens([1, 2, 3], num_tokens=2)
        assert not result.ok
        assert model._stats.errors == 1

    def test_retry_on_failure_then_success(self):
        model = RemoteDraftModel(_make_config(max_retries=2))
        mock_client = MagicMock()

        fail_resp = MagicMock()
        fail_resp.raise_for_status.side_effect = Exception("503")

        ok_resp = MagicMock()
        ok_resp.raise_for_status = MagicMock()
        ok_resp.json.return_value = _completions_response([42], [-0.5])

        mock_client.post.side_effect = [fail_resp, ok_resp]
        model._client = mock_client

        result = model.generate_tokens([1], num_tokens=1)
        assert result.ok
        assert result.token_ids == [42]
        assert mock_client.post.call_count == 2

    def test_api_key_in_headers(self):
        config = _make_config(api_key="sk-test-123")
        model = RemoteDraftModel(config)
        headers = model._build_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer sk-test-123"

    def test_no_api_key_no_auth_header(self):
        config = _make_config(api_key="")
        model = RemoteDraftModel(config)
        headers = model._build_headers()
        assert "Authorization" not in headers


# ── Stats ────────────────────────────────────────────────────────────────


class TestStats:
    def test_initial_stats(self):
        model = RemoteDraftModel(_make_config())
        s = model.stats
        assert s["total_calls"] == 0
        assert s["total_tokens"] == 0
        assert s["errors"] == 0

    def test_stats_after_calls(self):
        model = RemoteDraftModel(_make_config())
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _completions_response([1, 2, 3], [-0.1, -0.2, -0.3])
        mock_client.post.return_value = mock_resp
        model._client = mock_client

        model.generate_tokens([1], num_tokens=3)
        model.generate_tokens([1], num_tokens=2)

        s = model.stats
        assert s["total_calls"] == 2
        assert s["total_tokens"] == 5


# ── Close ────────────────────────────────────────────────────────────────


class TestClose:
    def test_close_sync_client(self):
        model = RemoteDraftModel(_make_config())
        mock_client = MagicMock()
        model._client = mock_client
        model.close()
        mock_client.close.assert_called_once()
        assert model._client is None
