"""E2E test: Disaggregated prefill -> decode pipeline.

Tests the disaggregated serving components end-to-end:
1. Add prefill and decode nodes via the router API
2. Verify pool stats and health checks
3. Test through the HTTP API routes where possible
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient

import distllm.api.server as server_module
from distllm.api.server import app
from distllm.core.disagg_serving import DisaggOrchestrator, DisaggRouter, PrefillPool, DecodePool, PoolStatus


@pytest.fixture(autouse=True)
def _disable_auth(monkeypatch):
    monkeypatch.setenv("DISABLE_AUTH", "1")
    monkeypatch.setenv("DISTLLM_DEV_MODE", "1")
    monkeypatch.delenv("API_KEY", raising=False)


@pytest.fixture
def router():
    return DisaggRouter()


@pytest.fixture
def orch():
    router = DisaggRouter()
    return DisaggOrchestrator(router=router)


@pytest.fixture
def coord_with_orch(orch):
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
    coord._disagg_orchestrator = orch

    coord.tokenizer = MagicMock()
    coord.tokenizer.encode.return_value = [1, 2, 3]
    coord.tokenizer.decode.return_value = "generated text"
    coord.tokenizer.eos_token_id = 0
    coord.list_models.return_value = ["distributed-llm"]

    return coord


@pytest.fixture
def client(coord_with_orch):
    import distllm.api.server as server_module
    original = server_module.coordinator
    server_module.coordinator = coord_with_orch
    c = TestClient(app)
    yield c
    server_module.coordinator = original


class TestDisaggE2E:
    @pytest.mark.asyncio
    async def test_add_prefill_and_decode_nodes(self, router):
        await router.add_prefill_node(node_id="prefill-1", host="10.0.0.1", port=50051, capacity=4)
        await router.add_decode_node(node_id="decode-1", host="10.0.0.2", port=50052, capacity=2)
        stats = router.get_stats()
        assert stats["prefill"]["total_nodes"] == 1
        assert stats["decode"]["total_nodes"] == 1

    @pytest.mark.asyncio
    async def test_pool_node_lifecycle(self):
        pool = PrefillPool()
        await pool.register_node("node-1", "10.0.0.1", 50051, capacity=2)
        stats = pool.get_stats()
        assert stats["total_nodes"] == 1
        assert stats["active_nodes"] == 1

        await pool.unregister_node("node-1")
        stats = pool.get_stats()
        assert stats["total_nodes"] == 0

    @pytest.mark.asyncio
    async def test_node_selection_least_loaded(self, router):
        await router.add_prefill_node(node_id="heavy", host="10.0.0.1", port=50051, capacity=2)
        await router.add_prefill_node(node_id="light", host="10.0.0.2", port=50051, capacity=2)
        router.prefill_pool._nodes["heavy"].current_load = 2
        router.prefill_pool._nodes["light"].current_load = 1
        selected = await router.prefill_pool.select_node()
        assert selected.node_id == "light"  # least-loaded

    @pytest.mark.asyncio
    async def test_decode_pool_assign_and_release(self):
        pool = DecodePool()
        await pool.register_node("dc-1", "10.0.0.1", 50052, capacity=2)
        assigned = await pool.assign_request("req-1")
        assert assigned == "dc-1"
        await pool.release_request("req-1")
        assert pool.get_node_for_request("req-1") is None

    @pytest.mark.asyncio
    async def test_decode_pool_rejects_when_full(self):
        pool = DecodePool()
        await pool.register_node("dc-1", "10.0.0.1", 50052, capacity=1)
        await pool.assign_request("req-1")
        assigned = await pool.assign_request("req-2")
        assert assigned is None
        assert pool.get_stats()["assigned_requests"] == 1

    def test_api_add_prefill_node(self, client):
        resp = client.post("/v1/disagg/nodes/prefill", json={
            "node_id": "pf-api-1",
            "host": "10.0.0.1",
            "port": 50051,
            "capacity": 4,
        })
        assert resp.status_code == 200
        assert resp.json()["role"] == "prefill"

    def test_api_add_decode_node(self, client):
        resp = client.post("/v1/disagg/nodes/decode", json={
            "node_id": "dc-api-1",
            "host": "10.0.0.2",
            "port": 50052,
            "capacity": 2,
        })
        assert resp.status_code == 200
        assert resp.json()["role"] == "decode"
