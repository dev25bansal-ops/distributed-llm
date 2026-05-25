"""Comprehensive tests for API routes: optimization, rag, agent, pipeline, disagg, versions."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from distllm.api.api_state import g as api_g
from distllm.api.server import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _cleanup():
    """Save/restore api_g.coordinator around each test."""
    original = api_g.coordinator
    api_g.coordinator = None
    yield
    api_g.coordinator = original


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
    api_g.coordinator = c
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

    def test_optimization_status_engine_none(self, coord, client):
        coord._self_optimizing = None
        resp = client.get("/v1/optimization/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is False
        assert data["status"] == "not_initialized"

    def test_optimization_status_engine_stopped(self, coord, client):
        mock_engine = MagicMock()
        mock_engine._running = False
        mock_engine.stats.return_value = {}
        coord._self_optimizing = mock_engine

        resp = client.get("/v1/optimization/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

    def test_optimization_suggestions_no_engine(self, coord, client):
        coord._self_optimizing = None
        resp = client.get("/v1/optimization/suggestions")
        assert resp.status_code == 503

    def test_optimization_suggestions_no_coordinator(self, client):
        resp = client.get("/v1/optimization/suggestions")
        assert resp.status_code == 503


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
# /v1/pipeline routes
# ===================================================================

class TestPipelineRoutes:
    def test_pipeline_execute_no_coordinator(self, client):
        api_g.coordinator = None
        resp = client.post("/v1/pipeline", json={
            "steps": [{"model": "m1", "step_type": "transform", "params": {"type": "identity"}}],
            "input": "hello",
        })
        assert resp.status_code == 503

    def test_pipeline_execute_no_composer(self, coord, client):
        coord._pipeline_composer = None
        resp = client.post("/v1/pipeline", json={
            "steps": [{"model": "m1", "step_type": "transform", "params": {"type": "identity"}}],
            "input": "hello",
        })
        assert resp.status_code == 503

    def test_pipeline_execute_success(self, coord, client):
        mock_composer = MagicMock()
        async def _iter():
            yield {"step_index": 0, "step_type": "transform", "output": "hello world", "latency_ms": 5.0, "error": None}
            yield {"step_index": 1, "step_type": "complete", "output": "hello world", "latency_ms": 10.0, "error": None}
        mock_composer.execute.return_value = _iter()
        coord._pipeline_composer = mock_composer

        resp = client.post("/v1/pipeline", json={
            "steps": [{"model": "m1", "step_type": "transform", "params": {"type": "identity"}}],
            "input": "hello",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["steps"]) == 1
        assert data["steps"][0]["step_type"] == "transform"
        assert data["steps"][0]["output"] == "hello world"

    def test_pipeline_execute_step_failure(self, coord, client):
        mock_composer = MagicMock()
        async def _iter():
            yield {"step_index": 0, "step_type": "transform", "output": None, "latency_ms": 3.0, "error": "something went wrong"}
        mock_composer.execute.return_value = _iter()
        coord._pipeline_composer = mock_composer

        resp = client.post("/v1/pipeline", json={
            "steps": [{"model": "m1", "step_type": "transform", "params": {"type": "identity"}}],
            "input": "hello",
        })
        assert resp.status_code == 200
        assert resp.json()["steps"][0]["error"] == "something went wrong"

    def test_pipeline_execute_exception(self, coord, client):
        mock_composer = MagicMock()
        async def _iter():
            raise RuntimeError("pipeline crashed")
            yield  # pragma: no cover
        mock_composer.execute.return_value = _iter()
        coord._pipeline_composer = mock_composer

        resp = client.post("/v1/pipeline", json={
            "steps": [{"model": "m1", "step_type": "transform", "params": {}}],
            "input": "hello",
        })
        assert resp.status_code == 200
        assert "pipeline crashed" in resp.json()["error"]

    def test_pipeline_register_no_coordinator(self, client):
        api_g.coordinator = None
        resp = client.post("/v1/pipeline/register", json={
            "pipeline_id": "pipe-1",
            "steps": [{"model": "m1", "step_type": "generate", "params": {}}],
        })
        assert resp.status_code == 503

    def test_pipeline_register_success(self, coord, client):
        mock_composer = MagicMock()
        coord._pipeline_composer = mock_composer

        resp = client.post("/v1/pipeline/register", json={
            "pipeline_id": "pipe-1",
            "steps": [{"model": "m1", "step_type": "generate", "params": {"temperature": 0.7}}],
            "fallback_pipeline_id": "pipe-0",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["pipeline_id"] == "pipe-1"
        assert data["status"] == "registered"
        assert data["steps_count"] == 1

    def test_pipeline_register_auto_creates_composer(self, coord, client):
        coord._pipeline_composer = None

        resp = client.post("/v1/pipeline/register", json={
            "pipeline_id": "pipe-auto",
            "steps": [{"model": "m1", "step_type": "generate", "params": {}}],
        })
        assert resp.status_code == 200
        assert coord._pipeline_composer is not None

    def test_pipeline_get_no_coordinator(self, client):
        api_g.coordinator = None
        resp = client.get("/v1/pipeline/pipe-1")
        assert resp.status_code == 503

    def test_pipeline_get_not_found(self, coord, client):
        mock_composer = MagicMock()
        mock_composer.get.return_value = None
        coord._pipeline_composer = mock_composer

        resp = client.get("/v1/pipeline/pipe-missing")
        assert resp.status_code == 404

    def test_pipeline_get_success(self, coord, client):
        from distllm.core.pipeline_composer import PipelineSpec, PipelineStep, StepType

        spec = PipelineSpec(
            pipeline_id="pipe-1",
            steps=[PipelineStep(model="m1", step_type=StepType.generate, params={"temp": 0.5})],
            fallback_pipeline_id="pipe-0",
        )
        mock_composer = MagicMock()
        mock_composer.get.return_value = spec
        coord._pipeline_composer = mock_composer

        resp = client.get("/v1/pipeline/pipe-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pipeline_id"] == "pipe-1"
        assert len(data["steps"]) == 1
        assert data["steps"][0]["step_type"] == "generate"
        assert data["fallback_pipeline_id"] == "pipe-0"


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

    def test_disagg_result_not_found(self, coord, client):
        mock_orch = MagicMock()
        mock_orch.get_result = AsyncMock(return_value=None)
        coord._disagg_orchestrator = mock_orch

        resp = client.get("/v1/disagg/result/missing-req")
        assert resp.status_code == 404

    def test_disagg_result_no_orchestrator(self, coord, client):
        coord._disagg_orchestrator = None
        resp = client.get("/v1/disagg/result/req-1")
        assert resp.status_code == 503

    def test_disagg_health_degraded(self, coord, client):
        mock_orch = MagicMock()
        mock_orch.router = MagicMock()
        mock_orch.router.prefill_pool._nodes = {}
        mock_orch.router.decode_pool._nodes = {}
        mock_orch.health_check.return_value = {
            "healthy": False,
            "pending_requests": 0,
            "prefill_pool": {"total_nodes": 0, "active_nodes": 0},
            "decode_pool": {"total_nodes": 0, "active_nodes": 0},
        }
        coord._disagg_orchestrator = mock_orch

        resp = client.get("/v1/disagg/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "degraded"
        assert resp.json()["prefill_nodes"] == 0
        assert resp.json()["decode_nodes"] == 0

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


# ===================================================================
# /v1/models/{model_id}/versions routes
# ===================================================================

def _make_version_mock(**kwargs):
    """Helper to create a mock version object with the expected attributes."""
    v = MagicMock()
    v.version_id = kwargs.get("version_id", "v1.0.0")
    v.model_id = kwargs.get("model_id", "model-1")
    v.model_path = kwargs.get("model_path", "path/to/model")
    v.status = MagicMock()
    v.status.value = kwargs.get("status", "stable")
    v.created_at = kwargs.get("created_at", 1000.0)
    v.promoted_at = kwargs.get("promoted_at", 2000.0)
    v.traffic_weight = kwargs.get("traffic_weight", 0.5)
    return v


class TestVersionRoutes:
    def test_create_version_no_coordinator(self, client):
        resp = client.post("/v1/models/m1/versions", json={
            "version_id": "v1.0.0",
            "model_path": "path/to/model",
        })
        assert resp.status_code == 503

    def test_create_version_no_version_manager(self, coord, client):
        coord._version_manager = None
        resp = client.post("/v1/models/m1/versions", json={
            "version_id": "v1.0.0",
            "model_path": "path/to/model",
        })
        assert resp.status_code == 503

    def test_create_version_success(self, coord, client):
        mock_vm = MagicMock()
        mock_vm.register_version.return_value = _make_version_mock()
        coord._version_manager = mock_vm

        resp = client.post("/v1/models/m1/versions", json={
            "version_id": "v1.0.0",
            "model_path": "path/to/model",
            "metadata": {"author": "test"},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["version_id"] == "v1.0.0"
        assert data["status"] == "stable"

    def test_list_versions(self, coord, client):
        mock_vm = MagicMock()
        mock_vm.list_versions.return_value = [
            _make_version_mock(version_id="v1.0.0", status="stable"),
            _make_version_mock(version_id="v1.1.0", status="candidate"),
        ]
        coord._version_manager = mock_vm

        resp = client.get("/v1/models/m1/versions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["versions"]) == 2
        assert data["versions"][0]["version_id"] == "v1.0.0"

    def test_list_versions_empty(self, coord, client):
        mock_vm = MagicMock()
        mock_vm.list_versions.return_value = []
        coord._version_manager = mock_vm

        resp = client.get("/v1/models/m1/versions")
        assert resp.status_code == 200
        assert resp.json()["versions"] == []

    def test_get_version_success(self, coord, client):
        mock_vm = MagicMock()
        mock_vm.get_version.return_value = _make_version_mock(version_id="v1.0.0")
        coord._version_manager = mock_vm

        resp = client.get("/v1/models/m1/versions/v1.0.0")
        assert resp.status_code == 200
        assert resp.json()["version_id"] == "v1.0.0"

    def test_get_version_not_found(self, coord, client):
        mock_vm = MagicMock()
        mock_vm.get_version.return_value = None
        coord._version_manager = mock_vm

        resp = client.get("/v1/models/m1/versions/v1.0.0")
        assert resp.status_code == 404

    def test_get_version_stats_success(self, coord, client):
        mock_vm = MagicMock()
        mock_vm.get_version_stats.return_value = {
            "version_id": "v1.0.0",
            "status": "stable",
            "traffic_weight": 1.0,
            "total_requests": 100,
            "error_rate": 0.01,
            "avg_latency_ms": 150.0,
            "p50_latency_ms": 120.0,
            "p99_latency_ms": 300.0,
            "avg_prompt_tokens": 512,
            "avg_completion_tokens": 128,
        }
        coord._version_manager = mock_vm

        resp = client.get("/v1/models/m1/versions/v1.0.0/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_requests"] == 100
        assert data["p99_latency_ms"] == 300.0

    def test_get_version_stats_not_found(self, coord, client):
        mock_vm = MagicMock()
        mock_vm.get_version_stats.return_value = None
        coord._version_manager = mock_vm

        resp = client.get("/v1/models/m1/versions/v1.0.0/stats")
        assert resp.status_code == 404

    def test_delete_version_success(self, coord, client):
        mock_vm = MagicMock()
        mock_vm.delete_version.return_value = True
        coord._version_manager = mock_vm

        resp = client.delete("/v1/models/m1/versions/v1.0.0")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_delete_version_not_found(self, coord, client):
        mock_vm = MagicMock()
        mock_vm.delete_version.return_value = False
        coord._version_manager = mock_vm

        resp = client.delete("/v1/models/m1/versions/v1.0.0")
        assert resp.status_code == 404

    def test_promote_version_success(self, coord, client):
        mock_vm = MagicMock()
        mock_vm.promote_version.return_value = True
        coord._version_manager = mock_vm

        resp = client.post("/v1/models/m1/versions/v1.0.0/promote")
        assert resp.status_code == 200
        assert resp.json()["status"] == "promoted"

    def test_promote_version_not_found(self, coord, client):
        mock_vm = MagicMock()
        mock_vm.promote_version.return_value = False
        coord._version_manager = mock_vm

        resp = client.post("/v1/models/m1/versions/v1.0.0/promote")
        assert resp.status_code == 404

    def test_compare_versions_success(self, coord, client):
        mock_vm = MagicMock()
        mock_vm.evaluate_promotion.return_value = {
            "sample_a": 50,
            "sample_b": 50,
            "sufficient_samples": True,
            "error_rate_a": 0.02,
            "error_rate_b": 0.01,
            "avg_latency_a": 200.0,
            "avg_latency_b": 150.0,
            "p50_latency_a": 180.0,
            "p50_latency_b": 140.0,
            "p99_latency_a": 400.0,
            "p99_latency_b": 300.0,
            "recommendation": "promote",
            "reason": "better performance",
            "mann_whitney_p": 0.03,
            "t_p_value": 0.02,
        }
        coord._version_manager = mock_vm

        resp = client.post("/v1/models/m1/versions/compare", json={
            "stable_version": "v1.0.0",
            "candidate_version": "v1.1.0",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["recommendation"] == "promote"
        assert data["sufficient_samples"] is True

    def test_compare_versions_missing_params(self, coord, client):
        mock_vm = MagicMock()
        coord._version_manager = mock_vm

        resp = client.post("/v1/models/m1/versions/compare", json={})
        assert resp.status_code == 400

    def test_shadow_comparisons(self, coord, client):
        mock_vm = MagicMock()
        mock_vm.get_shadow_comparisons.return_value = [
            {
                "model_id": "m1",
                "stable_version": "v1.0.0",
                "shadow_version": "v1.1.0",
                "request_id": "req-1",
                "stable_output": "output-a",
                "shadow_output": "output-b",
                "latency_stable": 100.0,
                "latency_shadow": 80.0,
                "timestamp": 3000.0,
            }
        ]
        coord._version_manager = mock_vm

        resp = client.get("/v1/models/m1/shadow")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["comparisons"]) == 1
        assert data["comparisons"][0]["shadow_version"] == "v1.1.0"

    def test_blue_green_switch(self, coord, client):
        mock_vm = MagicMock()
        mock_vm.switch_color.return_value = "green"
        coord._version_manager = mock_vm

        resp = client.post("/v1/models/m1/blue-green/switch", json={"model_id": "m1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "switched"
        assert data["active_color"] == "green"

    def test_blue_green_rollback(self, coord, client):
        mock_vm = MagicMock()
        mock_vm.rollback_color.return_value = "blue"
        coord._version_manager = mock_vm

        resp = client.post("/v1/models/m1/blue-green/rollback")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "rolled_back"
        assert data["active_color"] == "blue"


# ===================================================================
# /v1/debug routes
# ===================================================================

def _make_stored_request_mock(**kwargs):
    r = MagicMock()
    r.request_id = kwargs.get("request_id", "req-1")
    r.prompt = kwargs.get("prompt", "test prompt")
    r.timestamp = kwargs.get("timestamp", 1000.0)
    r.duration_ms = kwargs.get("duration_ms", 50.0)
    r.model = kwargs.get("model", "test-model")
    r.error = kwargs.get("error", None)
    r.replay_count = kwargs.get("replay_count", 0)
    r.params = kwargs.get("params", {"temperature": 0.7})
    r.response = kwargs.get("response", "test response")
    return r


class TestDebugRoutes:
    def test_get_recent_requests_no_coordinator(self, client):
        resp = client.get("/v1/debug/recent")
        assert resp.status_code == 503

    def test_get_recent_requests_success(self, coord, client):
        coord._replay_buffer = MagicMock()
        stored = _make_stored_request_mock()
        coord.get_recent_requests.return_value = [stored]

        resp = client.get("/v1/debug/recent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["requests"][0]["request_id"] == "req-1"

    def test_get_recent_requests_empty(self, coord, client):
        coord._replay_buffer = MagicMock()
        coord.get_recent_requests.return_value = []

        resp = client.get("/v1/debug/recent?n=5")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_get_request_detail_no_coordinator(self, client):
        resp = client.get("/v1/debug/request/req-1")
        assert resp.status_code == 503

    def test_get_request_detail_not_found(self, coord, client):
        coord._replay_buffer = MagicMock()
        coord._replay_buffer.get.return_value = None

        resp = client.get("/v1/debug/request/req-1")
        assert resp.status_code == 404

    def test_get_request_detail_success(self, coord, client):
        coord._replay_buffer = MagicMock()
        coord._replay_buffer.get.return_value = _make_stored_request_mock()

        resp = client.get("/v1/debug/request/req-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["request_id"] == "req-1"
        assert data["prompt"] == "test prompt"

    def test_replay_no_coordinator(self, client):
        resp = client.post("/v1/debug/replay", json={"request_id": "req-1"})
        assert resp.status_code == 503

    def test_replay_not_found(self, coord, client):
        coord.replay_request.return_value = None

        resp = client.post("/v1/debug/replay", json={"request_id": "req-1"})
        assert resp.status_code == 404

    def test_replay_success(self, coord, client):
        coord.replay_request.return_value = "replayed output"

        resp = client.post("/v1/debug/replay", json={"request_id": "req-1"})
        assert resp.status_code == 200
        assert resp.json()["response"] == "replayed output"

    def test_deterministic_no_coordinator(self, client):
        resp = client.post("/v1/debug/deterministic", json={"enabled": True, "seed": 42})
        assert resp.status_code == 503

    def test_deterministic_enable(self, coord, client):
        resp = client.post("/v1/debug/deterministic", json={"enabled": True, "seed": 123})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "enabled"
        assert data["seed"] == 123

    def test_deterministic_disable(self, coord, client):
        resp = client.post("/v1/debug/deterministic", json={"enabled": False})
        assert resp.status_code == 200
        assert resp.json()["status"] == "disabled"

    def test_export_buffer_no_coordinator(self, client):
        resp = client.get("/v1/debug/buffer/export")
        assert resp.status_code == 503

    def test_export_buffer_success(self, coord, client):
        coord._replay_buffer = MagicMock()
        coord._replay_buffer.export.return_value = [{"request_id": "req-1"}]

        resp = client.get("/v1/debug/buffer/export")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert len(data["entries"]) == 1


# ===================================================================
# /v1/adapters routes
# ===================================================================

class TestAdapterRoutes:
    def _enable_adapters(self, coord):
        coord.adapter_manager = MagicMock()
        coord.adapter_manager.rank_adapters.return_value = []
        coord.adapter_manager.list_adapters.return_value = []
        coord.adapter_manager.get_stats.return_value = {}
        coord.adapter_manager.active_adapter = None
        return coord.adapter_manager

    def test_list_adapters_no_coordinator(self, client):
        resp = client.get("/v1/adapters")
        assert resp.status_code == 503

    def test_list_adapters_no_manager(self, coord, client):
        coord.adapter_manager = None
        resp = client.get("/v1/adapters")
        assert resp.status_code == 503

    def test_list_adapters_success(self, coord, client):
        mgr = self._enable_adapters(coord)
        ranked = [MagicMock(adapter_id="adapter-1", rank=1, use_count=5, tenant_id="t1")]
        mgr.rank_adapters.return_value = ranked

        resp = client.get("/v1/adapters")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is None
        assert data["adapters"] == []
        assert len(data["ranking"]) == 1

    def test_adapter_load_no_coordinator(self, client):
        resp = client.post("/v1/adapters", json={
            "action": "load", "id": "lora-1", "path": "models/lora-1",
        })
        assert resp.status_code == 503

    @patch("distllm.api.routes.adapters.validate_adapter_path")
    def test_adapter_load_success(self, mock_validate, coord, client):
        mock_validate.return_value = "models/lora-1"
        mgr = self._enable_adapters(coord)
        resp = client.post("/v1/adapters", json={
            "action": "load", "id": "lora-1", "path": "models/lora-1",
            "rank": 5, "tenant_id": "tenant-1",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "loaded"
        mgr.load_adapter.assert_called_once_with("lora-1", "models/lora-1", rank=5, tenant_id="tenant-1")

    def test_adapter_load_missing_fields(self, coord, client):
        self._enable_adapters(coord)
        resp = client.post("/v1/adapters", json={"action": "load"})
        assert resp.status_code == 400

    def test_adapter_set_success(self, coord, client):
        mgr = self._enable_adapters(coord)
        resp = client.post("/v1/adapters", json={
            "action": "set", "id": "lora-1",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "active"
        mgr.set_active.assert_called_once_with("lora-1")

    def test_adapter_set_missing_id(self, coord, client):
        self._enable_adapters(coord)
        resp = client.post("/v1/adapters", json={"action": "set"})
        assert resp.status_code == 400

    def test_adapter_unload_success(self, coord, client):
        mgr = self._enable_adapters(coord)
        mgr.unload_adapter.return_value = True
        resp = client.post("/v1/adapters", json={
            "action": "unload", "id": "lora-1",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "unloaded"
        mgr.unload_adapter.assert_called_once_with("lora-1")

    def test_adapter_unload_not_found(self, coord, client):
        mgr = self._enable_adapters(coord)
        mgr.unload_adapter.return_value = False
        resp = client.post("/v1/adapters", json={
            "action": "unload", "id": "lora-1",
        })
        assert resp.status_code == 404

    def test_adapter_warmup_success(self, coord, client):
        mgr = self._enable_adapters(coord)
        mgr.warmup_adapters.return_value = 2
        resp = client.post("/v1/adapters", json={
            "action": "warmup",
            "adapters": {"lora-1": "path/1", "lora-2": "path/2"},
            "rank": 3,
            "tenant_id": "t1",
        })
        assert resp.status_code == 200
        assert resp.json()["loaded"] == 2

    def test_adapter_warmup_missing_adapters(self, coord, client):
        self._enable_adapters(coord)
        resp = client.post("/v1/adapters", json={"action": "warmup"})
        assert resp.status_code == 400

    def test_adapter_rank_success(self, coord, client):
        mgr = self._enable_adapters(coord)
        info = MagicMock()
        info.rank = 0
        mgr.get_adapter_info.return_value = info
        resp = client.post("/v1/adapters", json={
            "action": "rank", "id": "lora-1", "rank": 10,
        })
        assert resp.status_code == 200
        assert resp.json()["rank"] == 10

    def test_adapter_rank_not_found(self, coord, client):
        mgr = self._enable_adapters(coord)
        mgr.get_adapter_info.return_value = None
        resp = client.post("/v1/adapters", json={
            "action": "rank", "id": "lora-1", "rank": 10,
        })
        assert resp.status_code == 404

    def test_adapter_rank_missing_fields(self, coord, client):
        self._enable_adapters(coord)
        resp = client.post("/v1/adapters", json={"action": "rank"})
        assert resp.status_code == 400

    def test_adapter_list_action(self, coord, client):
        mgr = self._enable_adapters(coord)
        ranked = [MagicMock(adapter_id="a1", rank=1, use_count=3)]
        mgr.rank_adapters.return_value = ranked
        mgr.list_adapters.return_value = ["a1"]

        resp = client.post("/v1/adapters", json={"action": "list"})
        assert resp.status_code == 200
        assert resp.json()["adapters"] == ["a1"]

    def test_adapter_slora_register_no_manager(self, coord, client):
        self._enable_adapters(coord)
        coord._slora_manager = None
        resp = client.post("/v1/adapters", json={
            "action": "slora_register", "id": "lora-1", "path": "models/lora-1",
        })
        assert resp.status_code == 503

    @patch("distllm.api.routes.adapters.validate_adapter_path")
    def test_adapter_slora_register_success(self, mock_validate, coord, client):
        mock_validate.return_value = "models/lora-1"
        self._enable_adapters(coord)
        coord._slora_manager = MagicMock()
        resp = client.post("/v1/adapters", json={
            "action": "slora_register", "id": "lora-1", "path": "models/lora-1",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "registered"

    def test_adapter_slora_register_missing_fields(self, coord, client):
        self._enable_adapters(coord)
        coord._slora_manager = MagicMock()
        resp = client.post("/v1/adapters", json={"action": "slora_register"})
        assert resp.status_code == 400

    def test_adapter_slora_unregister_no_manager(self, coord, client):
        self._enable_adapters(coord)
        coord._slora_manager = None
        resp = client.post("/v1/adapters", json={
            "action": "slora_unregister", "id": "lora-1",
        })
        assert resp.status_code == 503

    def test_adapter_slora_unregister_success(self, coord, client):
        self._enable_adapters(coord)
        coord._slora_manager = MagicMock()
        resp = client.post("/v1/adapters", json={
            "action": "slora_unregister", "id": "lora-1",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "unregistered"

    def test_adapter_unknown_action(self, coord, client):
        self._enable_adapters(coord)
        resp = client.post("/v1/adapters", json={"action": "nonexistent"})
        assert resp.status_code == 400
