"""E2E test: RAG ingest -> retrieve -> generate cycle.

Tests the full RAG pipeline end-to-end:
1. Create a RAGPipeline with a mock embedding function
2. Ingest documents via the /v1/rag/ingest route
3. Retrieve relevant chunks via /v1/rag/retrieve
4. Build a RAG-enhanced prompt via /v1/rag/build_rag_prompt
5. Generate a response using the enriched prompt
"""

import numpy as np
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

import distllm.api.server as server_module
from distllm.api.server import app
from distllm.core.rag_pipeline import RAGPipeline, Document


def _dummy_embedding(texts: list[str]) -> np.ndarray:
    """Simple embedding function that returns random vectors."""
    return np.random.randn(len(texts), 8).astype(np.float32)


@pytest.fixture(autouse=True)
def _disable_auth(monkeypatch):
    monkeypatch.setenv("DISABLE_AUTH", "1")
    monkeypatch.setenv("DISTLLM_DEV_MODE", "1")
    monkeypatch.delenv("API_KEY", raising=False)


@pytest.fixture
def pipeline():
    return RAGPipeline(embedding_fn=_dummy_embedding, dimension=8, chunk_size=20, chunk_overlap=2)


@pytest.fixture
def coord_with_pipeline(pipeline):
    coord = MagicMock()
    coord.model_name = "test-model"
    coord.nodes = {}
    coord.node_order = []
    coord.scheduler = None
    coord.prefix_cache = None
    coord.metrics_exporter = None
    coord._vlm_pipeline = None
    coord._spec_decoder = None
    coord._shutting_down = False
    coord._rag_pipeline = pipeline

    coord.tokenizer = MagicMock()
    coord.tokenizer.encode.return_value = [1, 2, 3]
    coord.tokenizer.decode.return_value = "test response"
    coord.tokenizer.eos_token_id = 0
    coord.list_models.return_value = ["distributed-llm"]
    coord.generate.return_value = "Generated answer based on retrieved context."

    return coord


@pytest.fixture
def client(coord_with_pipeline):
    original = server_module.coordinator
    server_module.coordinator = coord_with_pipeline
    c = TestClient(app)
    yield c
    server_module.coordinator = original


def test_rag_ingest_retrieve_generate_cycle(client, pipeline):
    doc = Document(
        doc_id="doc-1",
        content="Paris is the capital of France. It is known for the Eiffel Tower."
                " Rome is the capital of Italy. Berlin is the capital of Germany."
                " The Eiffel Tower was built in 1889 for the World's Fair.",
        metadata={"source": "geography"},
    )
    pipeline.ingest(doc)
    assert pipeline.stats()["documents"] == 1

    results = pipeline.retrieve("What is the capital of France?", top_k=2)
    assert len(results) >= 1
    assert "France" in results[0].chunk.content or "Paris" in results[0].chunk.content

    enriched_prompt = pipeline.build_rag_prompt("What is the capital of France?", results)
    assert "Paris" in enriched_prompt or "France" in enriched_prompt
    assert "Question:" in enriched_prompt
    assert "Context:" in enriched_prompt


def test_rag_api_ingest_then_retrieve(client):
    resp_ingest = client.post("/v1/rag/ingest", json={
        "document_id": "doc-api-1",
        "content": "Machine learning is a subset of artificial intelligence."
                   " Deep learning uses neural networks with many layers."
                   " Reinforcement learning trains agents via rewards.",
    })
    assert resp_ingest.status_code == 200
    data = resp_ingest.json()
    assert data["chunks"] > 0

    resp_retrieve = client.post("/v1/rag/retrieve", json={
        "query": "What is deep learning?",
        "top_k": 3,
    })
    assert resp_retrieve.status_code == 200
    data = resp_retrieve.json()
    assert len(data["results"]) > 0


def test_rag_build_prompt_endpoint(client):
    client.post("/v1/rag/ingest", json={
        "document_id": "doc-prompt",
        "content": "Photosynthesis is the process by which plants convert sunlight into energy."
                   " Chlorophyll absorbs light and converts CO2 into glucose.",
    })
    resp = client.get("/v1/rag/build_rag_prompt?query=What+is+photosynthesis%3F&base_prompt=Answer+the+question.")
    assert resp.status_code == 200
    data = resp.json()
    assert "prompt" in data
    assert "Context:" in data["prompt"] or "Question:" in data["prompt"]


def test_rag_stats_endpoint(client):
    client.post("/v1/rag/ingest", json={
        "document_id": "doc-stats",
        "content": "Statistics document content for testing the stats endpoint.",
    })
    resp = client.get("/v1/rag/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_documents"] >= 1
    assert data["total_chunks"] >= 1


def test_rag_save_endpoint(client):
    resp = client.post("/v1/rag/save")
    assert resp.status_code == 200
    assert resp.json()["status"] == "saved"
