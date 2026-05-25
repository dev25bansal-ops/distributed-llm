"""Embedding tests: POST /v1/embeddings."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient

from distllm.api.api_state import g
from distllm.api.server import app


def make_mock_coordinator():
    coord = MagicMock()
    coord.model_name = "test-model"
    coord.nodes = {}
    coord.node_order = []
    coord.scheduler = None
    coord.prefix_cache = None
    coord.metrics_exporter = None
    coord.tokenizer = MagicMock()

    def encode_fn(text, **kwargs):
        tokens = list(range(1, len(text.split()) + 1))
        if kwargs.get("return_tensors") == "pt":
            return torch.tensor([tokens])
        return tokens
    coord.tokenizer.encode.side_effect = encode_fn

    def decode_side_effect(tokens, **kwargs):
        if isinstance(tokens, int):
            token_list = [tokens]
        elif isinstance(tokens, list):
            token_list = tokens
        else:
            token_list = tokens.tolist()
        return " ".join(f"tok-{t}" for t in token_list)
    coord.tokenizer.decode.side_effect = decode_side_effect
    coord.tokenizer.eos_token_id = 0
    coord.generate.return_value = "test response"

    mock_model = MagicMock()
    mock_model.parameters.side_effect = lambda: iter([torch.randn(10, 10)])
    mock_output = MagicMock()
    mock_output.logits = torch.randn(1, 5, 1000)
    mock_output.hidden_states = (torch.randn(1, 1, 10),)
    mock_model.return_value = mock_output
    coord.local_partitioner = MagicMock()
    coord.local_partitioner.full_model = mock_model
    coord._shutting_down = False
    coord._embedding_loader = None
    coord._vlm_pipeline = None
    coord._spec_decoder = None
    return coord


class TestEmbeddingsBasic:
    """Basic embedding generation via /v1/embeddings."""

    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ.pop("API_KEY", None)
        os.environ.pop("API_KEY_WAS_SET", None)
        os.environ["DISABLE_AUTH"] = "1"
        os.environ["DISTLLM_DEV_MODE"] = "1"
        coord = make_mock_coordinator()
        original = g.coordinator
        g.coordinator = coord
        yield
        g.coordinator = original
        os.environ.pop("DISABLE_AUTH", None)
        os.environ.pop("DISTLLM_DEV_MODE", None)

    def test_single_text_returns_200(self):
        resp = TestClient(app).post(
            "/v1/embeddings",
            json={"model": "distributed-llm", "input": ["Hello world"]},
        )
        assert resp.status_code == 200

    def test_multiple_texts_returns_200(self):
        resp = TestClient(app).post(
            "/v1/embeddings",
            json={"model": "distributed-llm", "input": ["Hello", "world", "test"]},
        )
        assert resp.status_code == 200

    def test_response_has_data(self):
        resp = TestClient(app).post(
            "/v1/embeddings",
            json={"model": "distributed-llm", "input": ["Hello world"]},
        )
        data = resp.json()
        assert "data" in data
        assert len(data["data"]) == 1

    def test_embedding_is_float_list(self):
        resp = TestClient(app).post(
            "/v1/embeddings",
            json={"model": "distributed-llm", "input": ["Hello world"]},
        )
        emb = resp.json()["data"][0]["embedding"]
        assert isinstance(emb, list)
        assert all(isinstance(v, float) for v in emb)

    def test_response_object_type(self):
        resp = TestClient(app).post(
            "/v1/embeddings",
            json={"model": "distributed-llm", "input": ["Hello world"]},
        )
        assert resp.json()["object"] == "list"

    def test_response_has_id(self):
        resp = TestClient(app).post(
            "/v1/embeddings",
            json={"model": "distributed-llm", "input": ["Hello world"]},
        )
        assert "id" in resp.json()
        assert resp.json()["id"].startswith("embed-")

    def test_embedding_index_order(self):
        resp = TestClient(app).post(
            "/v1/embeddings",
            json={"model": "distributed-llm", "input": ["first", "second", "third"]},
        )
        data = resp.json()["data"]
        assert data[0]["index"] == 0
        assert data[1]["index"] == 1
        assert data[2]["index"] == 2

    def test_batch_embedding_count(self):
        resp = TestClient(app).post(
            "/v1/embeddings",
            json={"model": "distributed-llm", "input": ["one", "two", "three"]},
        )
        data = resp.json()
        assert len(data["data"]) == 3

    def test_encoding_format_float_explicit(self):
        resp = TestClient(app).post(
            "/v1/embeddings",
            json={"model": "distributed-llm", "input": ["Hello world"], "encoding_format": "float"},
        )
        assert resp.status_code == 200
        emb = resp.json()["data"][0]["embedding"]
        assert isinstance(emb, list)
        assert len(emb) > 0
        assert all(isinstance(v, float) for v in emb)

    def test_base64_encoding(self):
        import base64, struct
        resp = TestClient(app).post(
            "/v1/embeddings",
            json={"model": "distributed-llm", "input": ["Hello world"], "encoding_format": "base64"},
        )
        assert resp.status_code == 200
        emb = resp.json()["data"][0]["embedding"]
        assert isinstance(emb, str)
        decoded = base64.b64decode(emb)
        floats = list(struct.unpack(f"{len(decoded)//4}f", decoded))
        assert len(floats) > 0
        assert all(isinstance(v, float) for v in floats)

    def test_dimension_truncation(self):
        resp = TestClient(app).post(
            "/v1/embeddings",
            json={"model": "distributed-llm", "input": ["Hello world"], "dimensions": 5},
        )
        assert resp.status_code == 200
        emb = resp.json()["data"][0]["embedding"]
        assert isinstance(emb, list)
        assert len(emb) == 5
        assert all(isinstance(v, float) for v in emb)

    def test_without_coordinator_returns_503(self):
        original = g.coordinator
        g.coordinator = None
        try:
            resp = TestClient(app).post(
                "/v1/embeddings",
                json={"model": "distributed-llm", "input": ["Hello world"]},
            )
            assert resp.status_code == 503
        finally:
            g.coordinator = original

    def test_normalize_returns_unit_vector(self):
        resp = TestClient(app).post(
            "/v1/embeddings",
            json={"model": "distributed-llm", "input": ["Hello world"], "normalize": True},
        )
        assert resp.status_code == 200
        emb = resp.json()["data"][0]["embedding"]
        norm = sum(v * v for v in emb) ** 0.5
        assert abs(norm - 1.0) < 1e-5


class TestEmbeddingFallback:
    """Verify fallback to generation model hidden states when no dedicated embedding model."""

    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ.pop("API_KEY", None)
        os.environ.pop("API_KEY_WAS_SET", None)
        os.environ["DISABLE_AUTH"] = "1"
        os.environ["DISTLLM_DEV_MODE"] = "1"
        coord = make_mock_coordinator()
        assert coord._embedding_loader is None
        original = g.coordinator
        g.coordinator = coord
        yield
        g.coordinator = original
        os.environ.pop("DISABLE_AUTH", None)
        os.environ.pop("DISTLLM_DEV_MODE", None)

    def test_fallback_to_generation_model(self):
        resp = TestClient(app).post(
            "/v1/embeddings",
            json={"model": "distributed-llm", "input": ["Hello world"]},
        )
        assert resp.status_code == 200
        emb = resp.json()["data"][0]["embedding"]
        assert isinstance(emb, list)
        assert len(emb) > 0

    def test_fallback_fails_without_local_partitioner(self):
        coord = g.coordinator
        coord.local_partitioner = None
        resp = TestClient(app).post(
            "/v1/embeddings",
            json={"model": "distributed-llm", "input": ["Hello world"]},
        )
        assert resp.status_code == 503


class TestDedicatedEmbeddingModel:
    """Verify behavior when a dedicated embedding model is available."""

    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ.pop("API_KEY", None)
        os.environ.pop("API_KEY_WAS_SET", None)
        os.environ["DISABLE_AUTH"] = "1"
        os.environ["DISTLLM_DEV_MODE"] = "1"
        coord = make_mock_coordinator()
        embed_loader = MagicMock()
        embed_loader.embedding_model = MagicMock()
        embed_loader.encode.return_value = torch.randn(1, 10)
        coord._embedding_loader = embed_loader
        original = g.coordinator
        g.coordinator = coord
        yield
        g.coordinator = original
        os.environ.pop("DISABLE_AUTH", None)
        os.environ.pop("DISTLLM_DEV_MODE", None)

    def test_dedicated_model_returns_200(self):
        resp = TestClient(app).post(
            "/v1/embeddings",
            json={"model": "distributed-llm", "input": ["Hello world"]},
        )
        assert resp.status_code == 200

    def test_dedicated_model_embedding_vector(self):
        resp = TestClient(app).post(
            "/v1/embeddings",
            json={"model": "distributed-llm", "input": ["Hello world"]},
        )
        emb = resp.json()["data"][0]["embedding"]
        assert isinstance(emb, list)
        assert len(emb) > 0


class TestRerank:
    """Rerank endpoint: POST /v1/rerank."""

    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ.pop("API_KEY", None)
        os.environ.pop("API_KEY_WAS_SET", None)
        os.environ["DISABLE_AUTH"] = "1"
        os.environ["DISTLLM_DEV_MODE"] = "1"
        coord = make_mock_coordinator()
        embed_loader = MagicMock()
        embed_loader.rerank_model = MagicMock()
        embed_loader.rerank.return_value = [(0, 0.95), (1, 0.42), (2, 0.13)]
        coord._embedding_loader = embed_loader
        original = g.coordinator
        g.coordinator = coord
        yield
        g.coordinator = original
        os.environ.pop("DISABLE_AUTH", None)
        os.environ.pop("DISTLLM_DEV_MODE", None)

    def test_rerank_success(self):
        resp = TestClient(app).post(
            "/v1/rerank",
            json={
                "query": "machine learning",
                "documents": ["AI is cool", "Weather report", "Sports news"],
            },
        )
        assert resp.status_code == 200

    def test_rerank_results_order(self):
        resp = TestClient(app).post(
            "/v1/rerank",
            json={
                "query": "machine learning",
                "documents": ["AI is cool", "Weather report", "Sports news"],
            },
        )
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["results"]) == 3
        assert data["results"][0]["relevance_score"] >= data["results"][1]["relevance_score"]

    def test_rerank_top_n(self):
        resp = TestClient(app).post(
            "/v1/rerank",
            json={
                "query": "machine learning",
                "documents": ["AI is cool", "Weather report", "Sports news"],
                "top_n": 2,
            },
        )
        data = resp.json()
        assert len(data["results"]) == 2

    def test_rerank_results_have_fields(self):
        resp = TestClient(app).post(
            "/v1/rerank",
            json={
                "query": "machine learning",
                "documents": ["AI is cool", "Weather report", "Sports news"],
            },
        )
        r = resp.json()["results"][0]
        assert "index" in r
        assert "document" in r
        assert "relevance_score" in r

    def test_rerank_without_coordinator_returns_503(self):
        original = g.coordinator
        g.coordinator = None
        try:
            resp = TestClient(app).post(
                "/v1/rerank",
                json={"query": "test", "documents": ["doc"]},
            )
            assert resp.status_code == 503
        finally:
            g.coordinator = original


class TestHybridRerank:
    """Hybrid rerank: POST /v1/rerank/hybrid."""

    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ.pop("API_KEY", None)
        os.environ.pop("API_KEY_WAS_SET", None)
        os.environ["DISABLE_AUTH"] = "1"
        os.environ["DISTLLM_DEV_MODE"] = "1"
        coord = make_mock_coordinator()
        embed_loader = MagicMock()
        embed_loader.embedding_model = MagicMock()
        embed_loader.encode.return_value = torch.randn(4, 8)
        embed_loader.rerank_model = MagicMock()
        embed_loader.rerank.return_value = [(0, 0.9), (1, 0.6), (2, 0.3)]
        coord._embedding_loader = embed_loader
        original = g.coordinator
        g.coordinator = coord
        yield
        g.coordinator = original
        os.environ.pop("DISABLE_AUTH", None)
        os.environ.pop("DISTLLM_DEV_MODE", None)

    def test_hybrid_rerank_success(self):
        resp = TestClient(app).post(
            "/v1/rerank/hybrid",
            json={
                "query": "machine learning",
                "documents": ["AI is cool", "Weather report", "Sports news"],
            },
        )
        assert resp.status_code == 200

    def test_hybrid_rerank_results_sorted(self):
        resp = TestClient(app).post(
            "/v1/rerank/hybrid",
            json={
                "query": "machine learning",
                "documents": ["AI is cool", "Weather report", "Sports news"],
            },
        )
        data = resp.json()
        assert data["object"] == "list"
        scores = [r["relevance_score"] for r in data["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_hybrid_rerank_top_n(self):
        resp = TestClient(app).post(
            "/v1/rerank/hybrid",
            json={
                "query": "machine learning",
                "documents": ["AI is cool", "Weather report", "Sports news"],
                "top_n": 1,
            },
        )
        assert len(resp.json()["results"]) == 1

    def test_hybrid_rerank_without_coordinator_returns_503(self):
        original = g.coordinator
        g.coordinator = None
        try:
            resp = TestClient(app).post(
                "/v1/rerank/hybrid",
                json={"query": "test", "documents": ["doc"]},
            )
            assert resp.status_code == 503
        finally:
            g.coordinator = original


class TestRRF:
    """Unit tests for _reciprocal_rank_fusion."""

    def test_rrf_always_returns_sorted_descending(self):
        from distllm.api.routes.embeddings import _reciprocal_rank_fusion
        emb_scores = [(0, 0.9), (1, 0.5)]
        rerank_scores = [(1, 0.8), (0, 0.3)]
        result = _reciprocal_rank_fusion(emb_scores, rerank_scores)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_includes_all_docs(self):
        from distllm.api.routes.embeddings import _reciprocal_rank_fusion
        emb_scores = [(0, 0.9), (1, 0.5)]
        rerank_scores = [(2, 0.7)]
        result = _reciprocal_rank_fusion(emb_scores, rerank_scores)
        assert len(result) == 3

    def test_rrf_k_parameter_affects_scores(self):
        from distllm.api.routes.embeddings import _reciprocal_rank_fusion
        emb_scores = [(0, 0.9), (1, 0.5)]
        rerank_scores = [(1, 0.8), (0, 0.3)]
        result_k1 = _reciprocal_rank_fusion(emb_scores, rerank_scores, k=1)
        result_k100 = _reciprocal_rank_fusion(emb_scores, rerank_scores, k=100)
        assert result_k1 != result_k100

    def test_rrf_score_formula(self):
        from distllm.api.routes.embeddings import _reciprocal_rank_fusion
        emb_scores = [(0, 0.9), (1, 0.5)]
        rerank_scores = [(0, 0.8), (1, 0.3)]
        result = _reciprocal_rank_fusion(emb_scores, rerank_scores, k=1)
        result_map = dict(result)
        expected_0 = 1.0 / (1 + 0 + 1) + 1.0 / (1 + 0 + 1)
        expected_1 = 1.0 / (1 + 1 + 1) + 1.0 / (1 + 1 + 1)
        assert abs(result_map[0] - expected_0) < 1e-10
        assert abs(result_map[1] - expected_1) < 1e-10


class TestEmbeddingsInputValidation:
    """Input validation for /v1/embeddings."""

    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ.pop("API_KEY", None)
        os.environ.pop("API_KEY_WAS_SET", None)
        os.environ["DISABLE_AUTH"] = "1"
        os.environ["DISTLLM_DEV_MODE"] = "1"
        coord = make_mock_coordinator()
        original = g.coordinator
        g.coordinator = coord
        yield
        g.coordinator = original
        os.environ.pop("DISABLE_AUTH", None)
        os.environ.pop("DISTLLM_DEV_MODE", None)

    def test_empty_input_list_accepted(self):
        resp = TestClient(app).post(
            "/v1/embeddings",
            json={"model": "distributed-llm", "input": []},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"] == []

    def test_missing_input_rejected(self):
        resp = TestClient(app).post(
            "/v1/embeddings",
            json={"model": "distributed-llm"},
        )
        assert resp.status_code == 422

    def test_negative_dimensions_rejected(self):
        resp = TestClient(app).post(
            "/v1/embeddings",
            json={"model": "distributed-llm", "input": ["Hello"], "dimensions": -1},
        )
        assert resp.status_code == 422
