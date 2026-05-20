"""Comprehensive tests for all 4 new API routes: optimization, rag, agent, disagg."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import distllm.api.server as server_module
from distllm.api.server import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _cleanup():
    """Save/restore server_module.coordinator around each test."""
    original = server_module.coordinator
    server_module.coordinator = None
    yield
    server_module.coordinator = original


@pytest.fixture(autouse=True)
def _disable_auth(monkeypatch):
    """Disable auth to prevent middleware ordering issues."""
    monkeypatch.setenv("DISABLE_AUTH", "1")
    monkeypatch.setenv("DISTLLM_DEV_MODE", "1")
    monkeypatch.delenv("API_KEY", raising=False)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def coord():
    """Set up a coordinator mock on the global module."""
    c = MagicMock()
    c._shutting_down = False
    server_module.coordinator = c
    return c


# ===================================================================
# /v1/optimization routes
# ===================================================================

class TestOptimizationRoutes:
    def test_optimization_status_no_engine(self, client):
        """Without coordinator, should return 503."""
        resp = client.get("/v1/optimization/status")
        assert resp.status_code == 503

    def test_optimization_status_with_engine(self, coord, client):
        """With a mocked coordinator._self_optimizing."""
        mock_engine = MagicMock()
        mock_engine.stats.return_value = {"total_operations": 42}
        mock_engine._running = True
        coord._self_optimizing = mock_engine

        resp = client.get("/v1/optimization/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["stats"]["total_operations"] == 42

    def test_optimization_suggestions(self, coord, client):
        """Optimization suggestions should return tunable params."""
        mock_engine = MagicMock()
        mock_engine.stats.return_value = {"total_operations": 5}
        mock_engine._running = True
        mock_engine.get_suggestions.return_value = {
            "batch_size": 8,
            "kv_cache_quantization": False,
            "speculative_decoding": False,
        }
        coord._self_optimizing = mock_engine

        resp = client.get("/v1/optimization/suggestions")
        assert resp.status_code == 200
        data = resp.json()
        assert "suggestions" in data


# ===================================================================
# /v1/rag routes
# ===================================================================

class TestRagRoutes:
    def test_rag_ingest_no_pipeline(self, coord, client):
        """Without RAG pipeline, ingest should fail with 503."""
        coord._rag_pipeline = None
        resp = client.post("/v1/rag/ingest", json={
            "document_id": "doc-1",
            "content": "test content",
        })
        assert resp.status_code == 503

    def test_rag_ingest_with_pipeline(self, coord, client):
        mock_pipeline = MagicMock()
        mock_pipeline.ingest.return_value = 3
        coord._rag_pipeline = mock_pipeline

        resp = client.post("/v1/rag/ingest", json={
            "document_id": "doc-1",
            "content": "This is a test document for RAG ingestion.",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["chunks"] == 3
        assert data["status"] == "ok"

    def test_rag_retrieve(self, coord, client):
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.chunk.content = "chunk1"
        mock_result.score = 0.95
        mock_result.rank = 1
        mock_result.chunk.doc_id = "doc-1"
        mock_pipeline.retrieve.return_value = [mock_result]
        coord._rag_pipeline = mock_pipeline

        resp = client.post("/v1/rag/retrieve", json={"query": "test query", "top_k": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1

    def test_rag_stats(self, coord, client):
        mock_pipeline = MagicMock()
        mock_pipeline.stats.return_value = {"documents": 10, "chunks": 100, "index_size": 1024}
        coord._rag_pipeline = mock_pipeline

        resp = client.get("/v1/rag/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_documents"] == 10

    def test_rag_save(self, coord, client):
        mock_pipeline = MagicMock()
        coord._rag_pipeline = mock_pipeline

        resp = client.post("/v1/rag/save")
        assert resp.status_code == 200
        assert resp.json()["status"] == "saved"

    def test_rag_build_prompt(self, coord, client):
        mock_pipeline = MagicMock()
        mock_pipeline.retrieve.return_value = [{"text": "ctx1", "score": 0.9}]
        mock_pipeline.build_rag_prompt.return_value = "Retrieved context: ctx1"
        coord._rag_pipeline = mock_pipeline

        resp = client.get("/v1/rag/build_rag_prompt?query=hello&base_prompt=world")
        assert resp.status_code == 200
        data = resp.json()
        assert "prompt" in data


# ===================================================================
# /v1/agents routes
# ===================================================================

class TestAgentRoutes:
    def test_agent_run_no_loop(self, coord, client):
        coord._agent_loop = None
        resp = client.post("/v1/agents/run", json={
            "goal": "test goal",
            "tools": [],
        })
        assert resp.status_code == 503

    def test_agent_run_with_loop(self, coord, client):
        mock_loop = MagicMock()
        mock_loop.run.return_value = {
            "result": "task done",
            "iterations": 3,
            "memory": [{"step": 1}],
        }
        coord._agent_loop = mock_loop

        resp = client.post("/v1/agents/run", json={
            "goal": "Complete the task",
            "tools": [
                {"name": "search", "description": "Web search", "handler": "search_web"}
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"] == "task done"
        assert data["iterations"] == 3

    def test_agent_status(self, coord, client):
        mock_loop = MagicMock()
        mock_loop.get_state.return_value = {
            "state": "idle",
            "memory": [],
        }
        coord._agent_loop = mock_loop

        resp = client.get("/v1/agents/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True


# ===================================================================
# /v1/disagg routes
# ===================================================================

class TestDisaggRoutes:
    def test_disagg_generate(self, coord, client):
        coord._disagg_orchestrator = None
        resp = client.post("/v1/disagg/generate", json={
            "prompt_tokens": [1, 2, 3],
            "max_new_tokens": 128,
        })
        assert resp.status_code == 503

    def test_disagg_generate_with_orch(self, coord, client):
        mock_orch = MagicMock()
        mock_orch.submit = AsyncMock(return_value="disagg-1")
        coord._disagg_orchestrator = mock_orch

        resp = client.post("/v1/disagg/generate", json={
            "prompt_tokens": [10, 20, 30],
            "max_new_tokens": 64,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["request_id"] == "disagg-1"

    def test_disagg_result(self, coord, client):
        mock_orch = MagicMock()
        mock_orch.get_result = AsyncMock(return_value=[1, 2, 3])
        coord._disagg_orchestrator = mock_orch

        resp = client.get("/v1/disagg/result/test-request-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"

    def test_disagg_add_prefill_node(self, coord, client):
        mock_orch = MagicMock()
        mock_orch.router = MagicMock()
        mock_orch.router.add_prefill_node = AsyncMock()
        coord._disagg_orchestrator = mock_orch

        resp = client.post("/v1/disagg/nodes/prefill", json={
            "node_id": "prefill-1",
            "host": "10.0.0.1",
            "port": 50051,
            "capacity": 4,
        })
        assert resp.status_code == 200
        assert resp.json()["role"] == "prefill"

    def test_disagg_add_decode_node(self, coord, client):
        mock_orch = MagicMock()
        mock_orch.router = MagicMock()
        mock_orch.router.add_decode_node = AsyncMock()
        coord._disagg_orchestrator = mock_orch

        resp = client.post("/v1/disagg/nodes/decode", json={
            "node_id": "decode-1",
            "host": "10.0.0.2",
            "port": 50052,
            "capacity": 2,
        })
        assert resp.status_code == 200
        assert resp.json()["role"] == "decode"

    def test_disagg_health(self, coord, client):
        mock_orch = MagicMock()
        mock_orch.router = MagicMock()
        pref_node = MagicMock()
        pref_node.node_id = "pf1"
        pref_node.role = "prefill"
        dec_node = MagicMock()
        dec_node.node_id = "dc1"
        dec_node.role = "decode"
        mock_orch.router.prefill_pool._nodes = {"pf1": pref_node}
        mock_orch.router.decode_pool._nodes = {"dc1": dec_node}
        mock_orch.health_check.return_value = {
            "healthy": True,
            "pending_requests": 0,
            "prefill_pool": {"total_nodes": 1, "active_nodes": 1},
            "decode_pool": {"total_nodes": 1, "active_nodes": 1},
        }
        coord._disagg_orchestrator = mock_orch

        resp = client.get("/v1/disagg/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["prefill_nodes"] == 1
        assert data["decode_nodes"] == 1
