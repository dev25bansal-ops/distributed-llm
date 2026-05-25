"""RAG API tests: ingest, retrieve, stats, save, build_rag_prompt."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from distllm.api.api_state import g
from distllm.api.server import app


@pytest.fixture(autouse=True)
def _disable_auth(monkeypatch):
    monkeypatch.setenv("DISABLE_AUTH", "1")
    monkeypatch.setenv("DISTLLM_DEV_MODE", "1")
    monkeypatch.delenv("API_KEY", raising=False)


@pytest.fixture
def pipeline():
    p = MagicMock()
    p.ingest = MagicMock(return_value=3)
    p.retrieve = MagicMock(return_value=[])
    p.stats = MagicMock(return_value={"documents": 5, "chunks": 20, "index_size": 4096})
    p.save_index = MagicMock()
    p.build_rag_prompt = MagicMock(return_value="enriched prompt")
    return p


class TestRagIngest:
    @pytest.fixture(autouse=True)
    def setup(self, pipeline):
        original = g.coordinator
        coord = MagicMock()
        coord._rag_pipeline = pipeline
        g.coordinator = coord
        yield
        g.coordinator = original

    def test_ingest_document(self):
        resp = TestClient(app).post(
            "/v1/rag/ingest",
            json={"document_id": "doc-1", "content": "Hello world", "metadata": {"source": "test"}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["document_id"] == "doc-1"
        assert data["chunks"] == 3

    def test_ingest_no_pipeline(self):
        original = g.coordinator
        coord = MagicMock()
        del coord._rag_pipeline
        g.coordinator = coord
        try:
            resp = TestClient(app).post(
                "/v1/rag/ingest",
                json={"document_id": "doc-2", "content": "test"},
            )
            assert resp.status_code == 503
        finally:
            g.coordinator = original

    def test_ingest_no_coordinator(self):
        original = g.coordinator
        g.coordinator = None
        try:
            resp = TestClient(app).post(
                "/v1/rag/ingest",
                json={"document_id": "doc-3", "content": "test"},
            )
            assert resp.status_code == 503
        finally:
            g.coordinator = original


class TestRagRetrieve:
    @pytest.fixture(autouse=True)
    def setup(self, pipeline):
        original = g.coordinator
        coord = MagicMock()
        coord._rag_pipeline = pipeline
        g.coordinator = coord
        yield
        g.coordinator = original

    def test_retrieve_empty_results(self):
        resp = TestClient(app).post(
            "/v1/rag/retrieve",
            json={"query": "hello", "top_k": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "hello"
        assert data["results"] == []

    def test_retrieve_with_results(self, pipeline):
        from distllm.core.rag_pipeline import DocumentChunk, RetrievalResult
        chunk = DocumentChunk(chunk_id="c1", doc_id="doc-1", content="relevant text", metadata={})
        pipeline.retrieve = MagicMock(return_value=[
            RetrievalResult(chunk=chunk, score=0.95, rank=1),
        ])
        resp = TestClient(app).post(
            "/v1/rag/retrieve",
            json={"query": "hello", "top_k": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["text"] == "relevant text"
        assert data["results"][0]["score"] == 0.95

    def test_retrieve_no_pipeline(self):
        original = g.coordinator
        coord = MagicMock()
        del coord._rag_pipeline
        g.coordinator = coord
        try:
            resp = TestClient(app).post(
                "/v1/rag/retrieve",
                json={"query": "hello"},
            )
            assert resp.status_code == 503
        finally:
            g.coordinator = original


class TestRagStats:
    @pytest.fixture(autouse=True)
    def setup(self, pipeline):
        original = g.coordinator
        coord = MagicMock()
        coord._rag_pipeline = pipeline
        g.coordinator = coord
        yield
        g.coordinator = original

    def test_stats(self):
        resp = TestClient(app).get("/v1/rag/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_documents"] == 5
        assert data["total_chunks"] == 20
        assert data["index_size"] == 4096

    def test_stats_no_pipeline(self):
        original = g.coordinator
        coord = MagicMock()
        del coord._rag_pipeline
        g.coordinator = coord
        try:
            resp = TestClient(app).get("/v1/rag/stats")
            assert resp.status_code == 503
        finally:
            g.coordinator = original


class TestRagSave:
    @pytest.fixture(autouse=True)
    def setup(self, pipeline):
        original = g.coordinator
        coord = MagicMock()
        coord._rag_pipeline = pipeline
        g.coordinator = coord
        yield
        g.coordinator = original

    def test_save(self):
        resp = TestClient(app).post("/v1/rag/save")
        assert resp.status_code == 200
        assert resp.json()["status"] == "saved"

    def test_save_no_pipeline(self):
        original = g.coordinator
        coord = MagicMock()
        del coord._rag_pipeline
        g.coordinator = coord
        try:
            resp = TestClient(app).post("/v1/rag/save")
            assert resp.status_code == 503
        finally:
            g.coordinator = original


class TestRagBuildPrompt:
    @pytest.fixture(autouse=True)
    def setup(self, pipeline):
        original = g.coordinator
        coord = MagicMock()
        coord._rag_pipeline = pipeline
        g.coordinator = coord
        yield
        g.coordinator = original

    def test_build_prompt(self):
        resp = TestClient(app).get(
            "/v1/rag/build_rag_prompt",
            params={"query": "hello", "base_prompt": "Answer:"},
        )
        assert resp.status_code == 200
        assert resp.json()["prompt"] == "enriched prompt"

    def test_build_prompt_no_pipeline(self):
        original = g.coordinator
        coord = MagicMock()
        del coord._rag_pipeline
        g.coordinator = coord
        try:
            resp = TestClient(app).get(
                "/v1/rag/build_rag_prompt",
                params={"query": "hello", "base_prompt": "Answer:"},
            )
            assert resp.status_code == 503
        finally:
            g.coordinator = original
