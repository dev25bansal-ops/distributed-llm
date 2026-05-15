"""Tests for distributed MoE inference."""

import threading
import torch
from unittest.mock import MagicMock
import pytest

from distllm.core.expert_registry import ExpertRegistry
from distllm.core.moe_orchestrator import (
    MoEForwardRequest,
    MoEForwardResponse,
    MoEOrchestrator,
)


class TestExpertRegistry:
    """Test ExpertRegistry basic operations."""

    def test_register_and_lookup(self):
        reg = ExpertRegistry()
        reg.register_expert(0, "node-1", 0)
        assert reg.get_expert_nodes(0) == ["node-1"]

    def test_register_same_expert_multiple_nodes(self):
        reg = ExpertRegistry()
        reg.register_expert(0, "node-1", 0)
        reg.register_expert(0, "node-2", 0)
        nodes = reg.get_expert_nodes(0)
        assert set(nodes) == {"node-1", "node-2"}

    def test_get_node_experts(self):
        reg = ExpertRegistry()
        reg.register_expert(0, "node-1", 0)
        reg.register_expert(1, "node-1", 1)
        experts = reg.get_node_experts("node-1")
        assert set(experts) == {0, 1}

    def test_list_all(self):
        reg = ExpertRegistry()
        reg.register_expert(0, "node-1", 0)
        reg.register_expert(1, "node-2", 1)
        mapping = reg.list_all()
        assert 0 in mapping
        assert 1 in mapping
        assert "node-1" in mapping[0]
        assert "node-2" in mapping[1]

    def test_select_best_node_least_loaded(self):
        reg = ExpertRegistry()
        reg.register_expert(0, "node-1", 0)
        reg.register_expert(0, "node-2", 0)
        # node-1 has more load
        reg.record_request("node-1")
        reg.record_request("node-1")
        reg.record_request("node-2")

        best = reg.select_best_node(0)
        assert best == "node-2"

    def test_select_best_node_not_found(self):
        reg = ExpertRegistry()
        assert reg.select_best_node(999) is None

    def test_unregister_node(self):
        reg = ExpertRegistry()
        reg.register_expert(0, "node-1", 0)
        reg.register_expert(1, "node-1", 1)
        reg.register_expert(2, "node-2", 2)

        reg.unregister_node("node-1")
        assert reg.get_expert_nodes(0) == []
        assert reg.get_expert_nodes(2) == ["node-2"]
        assert reg.get_node_experts("node-1") == []

    def test_stats(self):
        reg = ExpertRegistry()
        reg.register_expert(0, "node-1", 0)
        reg.register_expert(1, "node-2", 1)
        stats = reg.stats()
        assert stats["total_experts"] == 2
        assert stats["total_nodes"] == 2

    def test_thread_safety(self):
        reg = ExpertRegistry()
        errors = []

        def register_many(prefix, node):
            try:
                for i in range(100):
                    reg.register_expert(prefix * 100 + i, node, 0)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=register_many, args=(1, "node-1"))
        t2 = threading.Thread(target=register_many, args=(2, "node-2"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert not errors


class TestMoEForwardRequestResponse:
    """Test request/response containers."""

    def test_request_creation(self):
        req = MoEForwardRequest(
            node_id="node-1",
            hidden_states=torch.randn(2, 64),
            expert_ids=[0, 1],
            routing_weights=[0.6, 0.4],
            request_id="moe-1",
        )
        assert req.node_id == "node-1"
        assert req.expert_ids == [0, 1]

    def test_response_creation(self):
        resp = MoEForwardResponse(
            node_id="node-1",
            output=torch.randn(2, 64),
            success=True,
            processing_time_ms=10.5,
        )
        assert resp.success is True
        assert resp.processing_time_ms == 10.5


class TestMoEOrchestrator:
    """Test MoE orchestrator."""

    def test_route_no_registry(self):
        orch = MoEOrchestrator(expert_registry=None)
        requests = orch.route(torch.randn(2, 64), MagicMock())
        assert requests == {}

    def test_dispatch_no_client(self):
        orch = MoEOrchestrator()
        req = MoEForwardRequest(
            node_id="node-1",
            hidden_states=torch.randn(2, 64),
            expert_ids=[0],
            routing_weights=[1.0],
            request_id="moe-1",
        )
        responses = orch.dispatch({"node-1": req}, {})
        assert responses["node-1"].success is False
        assert "No client" in responses["node-1"].error_message

    def test_aggregate_single_response(self):
        orch = MoEOrchestrator()
        resp = MoEForwardResponse(
            node_id="node-1",
            output=torch.tensor([1.0, 2.0]),
            success=True,
        )
        result = orch.aggregate({"node-1": resp})
        assert torch.equal(result, torch.tensor([1.0, 2.0]))

    def test_aggregate_multiple_responses(self):
        orch = MoEOrchestrator()
        resp1 = MoEForwardResponse(
            node_id="node-1",
            output=torch.tensor([1.0, 2.0]),
            success=True,
        )
        resp2 = MoEForwardResponse(
            node_id="node-2",
            output=torch.tensor([3.0, 4.0]),
            success=True,
        )
        result = orch.aggregate({"node-1": resp1, "node-2": resp2})
        assert torch.equal(result, torch.tensor([4.0, 6.0]))

    def test_aggregate_all_fail(self):
        orch = MoEOrchestrator()
        resp = MoEForwardResponse(
            node_id="node-1",
            output=torch.tensor([1.0]),
            success=False,
            error_message="timeout",
        )
        with pytest.raises(RuntimeError, match="All expert requests failed"):
            orch.aggregate({"node-1": resp})

    def test_forward_returns_input_when_no_requests(self):
        orch = MoEOrchestrator(expert_registry=None)
        hidden = torch.randn(2, 64)
        result = orch.forward(hidden, MagicMock(), {})
        assert torch.equal(result, hidden)
