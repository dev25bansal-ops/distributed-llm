"""Tests for TGI-compatible API endpoint (tgi_compat.py).

Tests the public API surface of the TGI compatibility layer using only
real objects — no mocks, no GPU, no network.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from distllm.dist.tgi_compat import (
    TGIRequest,
    TGIGenerateResponse,
    TGIStreamChunk,
    TGIHealthResponse,
    TGIInfoResponse,
    _default_params,
    _to_openai_params,
    create_standalone_app,
    tgi_router,
)


# ── Pydantic Models ──────────────────────────────────────────────────────


class TestTGIModels:
    """Pydantic request/response models."""

    # ── TGIRequest ───────────────────────────────────────────────────

    def test_tgi_request_defaults(self):
        """Required field only."""
        req = TGIRequest(inputs="hello")
        assert req.inputs == "hello"
        assert req.parameters is None
        assert req.stream is False

    def test_tgi_request_all_fields(self):
        """All fields populated."""
        req = TGIRequest(
            inputs="test",
            parameters={"temperature": 0.5, "max_new_tokens": 100},
            stream=True,
        )
        assert req.inputs == "test"
        assert req.parameters == {"temperature": 0.5, "max_new_tokens": 100}
        assert req.stream is True

    def test_tgi_request_empty_input(self):
        """Empty string input is allowed."""
        req = TGIRequest(inputs="")
        assert req.inputs == ""

    def test_tgi_request_missing_input_raises(self):
        """Missing required 'inputs' field raises ValidationError."""
        with pytest.raises(ValidationError):
            TGIRequest()

    def test_tgi_request_invalid_stream_type_raises(self):
        """Non-boolean stream raises ValidationError."""
        with pytest.raises(ValidationError):
            TGIRequest(inputs="x", stream="not_bool")

    def test_tgi_request_parameters_as_none(self):
        """Explicit None for parameters is equivalent to omitting it."""
        req = TGIRequest(inputs="hello", parameters=None)
        assert req.parameters is None

    # ── TGIGenerateResponse ──────────────────────────────────────────

    def test_generate_response_defaults(self):
        """All fields use their defaults."""
        resp = TGIGenerateResponse()
        assert resp.generated_text == ""
        assert resp.details is None

    def test_generate_response_all_fields(self):
        """All fields populated."""
        resp = TGIGenerateResponse(
            generated_text="Hello, world!",
            details={"tokens": 5, "finish_reason": "stop"},
        )
        assert resp.generated_text == "Hello, world!"
        assert resp.details == {"tokens": 5, "finish_reason": "stop"}

    def test_generate_response_empty_text(self):
        """Explicit empty string is stored correctly."""
        resp = TGIGenerateResponse(generated_text="")
        assert resp.generated_text == ""

    def test_generate_response_none_details(self):
        """None details is valid."""
        resp = TGIGenerateResponse(generated_text="ok", details=None)
        assert resp.details is None

    # ── TGIStreamChunk ───────────────────────────────────────────────

    def test_stream_chunk_defaults(self):
        """All fields are None by default."""
        chunk = TGIStreamChunk()
        assert chunk.token is None
        assert chunk.generated_text is None
        assert chunk.details is None

    def test_stream_chunk_with_token(self):
        """Token field populated."""
        chunk = TGIStreamChunk(
            token={"text": "hello", "logprob": None, "special": False},
        )
        assert chunk.token == {"text": "hello", "logprob": None, "special": False}

    def test_stream_chunk_with_generated_text(self):
        """Generated text field populated."""
        chunk = TGIStreamChunk(generated_text="Hello", details={"tokens": 1})
        assert chunk.generated_text == "Hello"
        assert chunk.details == {"tokens": 1}

    # ── TGIHealthResponse ────────────────────────────────────────────

    def test_health_response_defaults(self):
        """Default status is 'healthy'."""
        resp = TGIHealthResponse()
        assert resp.status == "healthy"

    def test_health_response_custom_status(self):
        """Custom status string is preserved."""
        resp = TGIHealthResponse(status="degraded")
        assert resp.status == "degraded"

    # ── TGIInfoResponse ──────────────────────────────────────────────

    def test_info_response_defaults(self):
        """All fields use their defaults."""
        resp = TGIInfoResponse()
        assert resp.model_id == "distributed-llm"
        assert resp.model_dtype == "float16"
        assert resp.sha == ""
        assert resp.max_input_length == 131072
        assert resp.max_total_tokens == 139264
        assert resp.version == "2.0.0"

    def test_info_response_custom_model_id(self):
        """Model ID can be overridden."""
        resp = TGIInfoResponse(model_id="my-custom-model")
        assert resp.model_id == "my-custom-model"

    def test_info_response_custom_sha(self):
        """SHA can be overridden."""
        resp = TGIInfoResponse(sha="abc123def456")
        assert resp.sha == "abc123def456"

    def test_info_response_custom_version(self):
        """Version can be overridden."""
        resp = TGIInfoResponse(version="3.1.0")
        assert resp.version == "3.1.0"

    def test_info_response_custom_limits(self):
        """Input length and total tokens can be overridden."""
        resp = TGIInfoResponse(max_input_length=4096, max_total_tokens=8192)
        assert resp.max_input_length == 4096
        assert resp.max_total_tokens == 8192

    def test_info_response_custom_dtype(self):
        """Model dtype can be overridden."""
        resp = TGIInfoResponse(model_dtype="float32")
        assert resp.model_dtype == "float32"


# ── Helper Functions ─────────────────────────────────────────────────────


class TestDefaultParams:
    """_default_params helper function."""

    def test_returns_dict(self):
        """Returns a dict."""
        params = _default_params()
        assert isinstance(params, dict)

    def test_has_all_expected_keys(self):
        """Dict contains all expected parameter keys."""
        params = _default_params()
        assert "max_new_tokens" in params
        assert "temperature" in params
        assert "top_p" in params
        assert "top_k" in params
        assert "repetition_penalty" in params
        assert "do_sample" in params
        assert "seed" in params
        assert "stop" in params

    def test_default_values(self):
        """Default values match expected constants."""
        params = _default_params()
        assert params["max_new_tokens"] == 256
        assert params["temperature"] == 0.7
        assert params["top_p"] == 0.9
        assert params["top_k"] == 0
        assert params["repetition_penalty"] == 1.0
        assert params["do_sample"] is True
        assert params["seed"] is None
        assert params["stop"] == []

    def test_independent_copies(self):
        """Each call returns a fresh copy, not a shared dict."""
        p1 = _default_params()
        p2 = _default_params()
        p1["max_new_tokens"] = 999
        assert p2["max_new_tokens"] == 256


class TestToOpenAIParams:
    """_to_openai_params conversion helper."""

    def test_none_returns_defaults(self):
        """None input produces default parameters (still TGI-keyed)."""
        params = _to_openai_params(None)
        assert params["max_new_tokens"] == 256
        assert params["temperature"] == 0.7
        assert params["top_p"] == 0.9

    def test_empty_dict_returns_defaults(self):
        """Empty dict input produces default parameters (still TGI-keyed)."""
        params = _to_openai_params({})
        # When no TGI params are provided, the mapping loop yields no
        # OpenAI keys; the dict retains the raw default keys.
        assert params["max_new_tokens"] == 256
        assert params["temperature"] == 0.7

    def test_parameter_mapping(self):
        """All mapped parameters are converted correctly."""
        tgi_params = {
            "max_new_tokens": 512,
            "temperature": 0.1,
            "top_p": 0.5,
            "top_k": 10,
            "repetition_penalty": 1.2,
            "seed": 42,
        }
        params = _to_openai_params(tgi_params)
        assert params["max_tokens"] == 512
        assert params["temperature"] == 0.1
        assert params["top_p"] == 0.5
        assert params["top_k"] == 10
        assert params["frequency_penalty"] == 1.2
        assert params["seed"] == 42

    def test_partial_override(self):
        """Only provided parameters override defaults; others stay."""
        params = _to_openai_params({"temperature": 0.9})
        assert params["temperature"] == 0.9
        # max_new_tokens was not in the input, so only the TGI default is present
        assert params["max_new_tokens"] == 256
        assert params["top_p"] == 0.9  # default preserved

    def test_stop_as_string_converted_to_list(self):
        """Single string stop is wrapped in a list."""
        params = _to_openai_params({"stop": "<|end|>"})
        assert params["stop"] == ["<|end|>"]

    def test_stop_as_list_preserved(self):
        """List stop is kept as-is."""
        params = _to_openai_params({"stop": ["<|end|>", "<|eot|>"]})
        assert params["stop"] == ["<|end|>", "<|eot|>"]

    def test_stop_empty_list(self):
        """Empty list stop is preserved."""
        params = _to_openai_params({"stop": []})
        assert params["stop"] == []

    def test_stop_empty_string_wrapped(self):
        """Empty string stop is wrapped in a list (str branch)."""
        params = _to_openai_params({"stop": ""})
        assert params["stop"] == [""]

    def test_unmapped_params_ignored(self):
        """Unknown TGI parameters are not present in output."""
        params = _to_openai_params({"unknown_param": 123})
        assert "unknown_param" not in params

    def test_zero_temperature(self):
        """Zero temperature is preserved."""
        params = _to_openai_params({"temperature": 0.0})
        assert params["temperature"] == 0.0

    def test_max_new_tokens_zero(self):
        """Zero max_new_tokens is preserved."""
        params = _to_openai_params({"max_new_tokens": 0})
        assert params["max_tokens"] == 0

    def test_large_max_new_tokens(self):
        """Boundary-large value is preserved."""
        params = _to_openai_params({"max_new_tokens": 1048576})
        assert params["max_tokens"] == 1048576

    def test_do_sample_not_in_output(self):
        """do_sample is a default-only key and not mapped from TGI params."""
        params = _to_openai_params({})
        # do_sample comes from the defaults() base
        assert params.get("do_sample") is True

    def test_output_contains_both_key_sets(self):
        """Output has both internal and original TGI keys present."""
        params = _to_openai_params({"max_new_tokens": 100})
        assert "max_tokens" in params  # internal name added
        assert "max_new_tokens" in params  # original TGI name retained

    def test_frequency_penalty_mapping(self):
        """repetition_penalty maps to frequency_penalty (both present)."""
        params = _to_openai_params({"repetition_penalty": 1.5})
        assert params["frequency_penalty"] == 1.5
        assert params["repetition_penalty"] == 1.0  # original default retained


# ── Standalone App Factory ───────────────────────────────────────────────


class TestCreateStandaloneApp:
    """create_standalone_app factory function."""

    def test_returns_fastapi_app(self):
        """Returns a FastAPI app with correct title."""
        app = create_standalone_app()
        assert app.title == "DistLLM TGI-compatible API"

    def test_all_routes_registered(self):
        """All expected endpoint paths are registered."""
        app = create_standalone_app()
        paths = {r.path for r in app.routes}
        assert "/generate" in paths
        assert "/generate_stream" in paths
        assert "/health" in paths
        assert "/info" in paths

    def test_generate_route_uses_post(self):
        """/generate is a POST endpoint."""
        app = create_standalone_app()
        methods = set()
        for r in app.routes:
            if r.path == "/generate":
                methods = r.methods
                break
        assert "POST" in methods
        assert "GET" not in methods

    def test_generate_stream_route_uses_post(self):
        """/generate_stream is a POST endpoint."""
        app = create_standalone_app()
        methods = set()
        for r in app.routes:
            if r.path == "/generate_stream":
                methods = r.methods
                break
        assert "POST" in methods

    def test_health_route_uses_get(self):
        """/health is a GET endpoint."""
        app = create_standalone_app()
        methods = set()
        for r in app.routes:
            if r.path == "/health":
                methods = r.methods
                break
        assert "GET" in methods

    def test_info_route_uses_get(self):
        """/info is a GET endpoint."""
        app = create_standalone_app()
        methods = set()
        for r in app.routes:
            if r.path == "/info":
                methods = r.methods
                break
        assert "GET" in methods


# ── tgi_router ───────────────────────────────────────────────────────────


class TestTGIRouter:
    """The module-level tgi_router APIRouter object."""

    def test_router_tags(self):
        """Router has correct tag."""
        assert tgi_router.tags == ["tgi"]

    def test_router_has_at_least_four_routes(self):
        """Router contains the four expected endpoints."""
        assert len(tgi_router.routes) >= 4

    def test_router_route_paths(self):
        """All expected paths are registered on the router."""
        paths = {r.path for r in tgi_router.routes}
        assert "/generate" in paths
        assert "/generate_stream" in paths
        assert "/health" in paths
        assert "/info" in paths


# ── HTTP Endpoint Behavior ───────────────────────────────────────────────
# All tests run without a coordinator loaded, so generate endpoints return
# 503 ("No model loaded").  Health and info work independently.


class TestEndpoints:
    """Endpoint HTTP behavior via FastAPI TestClient (no coordinator loaded)."""

    # ── /health ──────────────────────────────────────────────────────

    def test_health_returns_200(self):
        """GET /health returns 200 with healthy status."""
        from fastapi.testclient import TestClient

        app = create_standalone_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    # ── /info ────────────────────────────────────────────────────────

    def test_info_returns_defaults(self):
        """GET /info returns default model info."""
        from fastapi.testclient import TestClient

        app = create_standalone_app()
        client = TestClient(app)
        resp = client.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == "distributed-llm"
        assert data["model_dtype"] == "float16"
        assert data["sha"] == ""
        assert data["max_input_length"] == 131072
        assert data["max_total_tokens"] == 139264
        assert data["version"] == "2.0.0"

    # ── /generate (no coordinator) ───────────────────────────────────

    def test_generate_returns_503_no_coordinator(self):
        """POST /generate returns 503 when no coordinator is loaded."""
        from fastapi.testclient import TestClient

        app = create_standalone_app()
        client = TestClient(app)
        resp = client.post("/generate", json={"inputs": "hello"})
        assert resp.status_code == 503
        assert "No model loaded" in resp.json()["detail"]

    def test_generate_with_parameters_returns_503(self):
        """POST /generate with parameters returns 503 without coordinator."""
        from fastapi.testclient import TestClient

        app = create_standalone_app()
        client = TestClient(app)
        resp = client.post(
            "/generate",
            json={
                "inputs": "hello",
                "parameters": {"max_new_tokens": 128, "temperature": 0.5},
            },
        )
        assert resp.status_code == 503
        assert "No model loaded" in resp.json()["detail"]

    def test_generate_stream_flag_delegates_to_stream_503(self):
        """POST /generate with stream=True delegates and returns 503."""
        from fastapi.testclient import TestClient

        app = create_standalone_app()
        client = TestClient(app)
        resp = client.post("/generate", json={"inputs": "hi", "stream": True})
        assert resp.status_code == 503
        assert "No model loaded" in resp.json()["detail"]

    def test_generate_empty_input_returns_503(self):
        """POST /generate with empty input returns 503 (no coordinator)."""
        from fastapi.testclient import TestClient

        app = create_standalone_app()
        client = TestClient(app)
        resp = client.post("/generate", json={"inputs": ""})
        assert resp.status_code == 503
        assert "No model loaded" in resp.json()["detail"]

    def test_generate_missing_input_returns_422(self):
        """POST /generate without 'inputs' returns 422 validation error."""
        from fastapi.testclient import TestClient

        app = create_standalone_app()
        client = TestClient(app)
        resp = client.post("/generate", json={})
        assert resp.status_code == 422

    # ── /generate_stream (no coordinator) ────────────────────────────

    def test_generate_stream_returns_503_no_coordinator(self):
        """POST /generate_stream returns 503 when no coordinator is loaded."""
        from fastapi.testclient import TestClient

        app = create_standalone_app()
        client = TestClient(app)
        resp = client.post("/generate_stream", json={"inputs": "hello"})
        assert resp.status_code == 503
        assert "No model loaded" in resp.json()["detail"]

    def test_generate_stream_with_parameters_returns_503(self):
        """POST /generate_stream with parameters returns 503 without coordinator."""
        from fastapi.testclient import TestClient

        app = create_standalone_app()
        client = TestClient(app)
        resp = client.post(
            "/generate_stream",
            json={
                "inputs": "hello",
                "parameters": {"top_p": 0.8},
            },
        )
        assert resp.status_code == 503
        assert "No model loaded" in resp.json()["detail"]

    def test_generate_stream_empty_input_returns_503(self):
        """POST /generate_stream with empty input returns 503."""
        from fastapi.testclient import TestClient

        app = create_standalone_app()
        client = TestClient(app)
        resp = client.post("/generate_stream", json={"inputs": ""})
        assert resp.status_code == 503
        assert "No model loaded" in resp.json()["detail"]

    def test_generate_stream_missing_input_returns_422(self):
        """POST /generate_stream without 'inputs' returns 422."""
        from fastapi.testclient import TestClient

        app = create_standalone_app()
        client = TestClient(app)
        resp = client.post("/generate_stream", json={})
        assert resp.status_code == 422

    # ── 404 on unknown paths ─────────────────────────────────────────

    def test_unknown_path_returns_404(self):
        """GET on an unknown path returns 404."""
        from fastapi.testclient import TestClient

        app = create_standalone_app()
        client = TestClient(app)
        resp = client.get("/nonexistent")
        assert resp.status_code == 404

    def test_post_on_health_returns_405(self):
        """POST on /health (a GET-only endpoint) returns 405."""
        from fastapi.testclient import TestClient

        app = create_standalone_app()
        client = TestClient(app)
        resp = client.post("/health")
        assert resp.status_code == 405
