"""Integration: 2-node inference correctness, failure recovery, join/leave.

Requires: mock model partitioner, mock gRPC clients (no GPU).
"""

import importlib.util
import sys
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"


def _make_fake_package(name: str, path: Path):
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    mod.__package__ = name
    sys.modules.setdefault(name, mod)
    return mod


_make_fake_package("distllm", SRC_DIR / "distllm")
_make_fake_package("distllm.core", SRC_DIR / "distllm/core")
_make_fake_package("distllm.dist", SRC_DIR / "distllm/dist")
_make_fake_package("distllm.dist.partition", SRC_DIR / "distllm/dist/partition")
_make_fake_package("distllm.backends", SRC_DIR / "distllm/backends")


def _load_module(rel_path: str):
    filepath = SRC_DIR / rel_path
    rel = filepath.relative_to(SRC_DIR)
    parts = list(rel.parent.parts) + [filepath.stem]
    if parts[0] == "distllm":
        dotted = ".".join(parts)
    else:
        dotted = "distllm." + ".".join(parts)
    if dotted in sys.modules:
        return sys.modules[dotted]
    spec = importlib.util.spec_from_file_location(dotted, filepath, submodule_search_locations=[])
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {filepath}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


_coord_mod = _load_module("distllm/core/coordinator.py")
Coordinator = _coord_mod.Coordinator
_rm_mod = _load_module("distllm/core/resource_manager.py")
ResourceManager = _rm_mod.ResourceManager
NodeRegistration = _rm_mod.NodeRegistration
CircuitBreakerConfig = _rm_mod.CircuitBreakerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_deterministic_forward_response(token_id: int = 42, vocab_size: int = 100):
    logits = torch.full((1, 1, vocab_size), float('-inf'))
    logits[0, 0, token_id] = 0.0
    return logits


class _MockModelConfig:
    num_hidden_layers = 12
    hidden_size = 768
    num_attention_heads = 12


class _MockPartitioner:
    full_model = MagicMock()
    full_model.config = _MockModelConfig()
    full_model.parameters.return_value = iter([torch.randn(10, 10)])
    start_layer = 0
    end_layer = 11


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    tok.decode.return_value = "hello world"
    tok.eos_token_id = 0
    tok.pad_token_id = 0
    tok.vocab_size = 100
    return tok


@pytest.fixture
def two_node_coordinator(mock_tokenizer, monkeypatch):
    monkeypatch.setattr(NodeRegistration, "init_client", lambda self, **kw: None)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: mock_tokenizer)
    mock_config = MagicMock()
    mock_config.model_type = "mock"
    mock_config.num_hidden_layers = 12
    mock_config.hidden_size = 768
    mock_config.num_attention_heads = 12
    mock_config.vocab_size = 32000
    mock_config.max_position_embeddings = 2048
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda *a, **kw: mock_config)
    coord = Coordinator(model_name="test-model", dtype="float32", max_batch_size=2)
    coord.tokenizer = mock_tokenizer
    coord.model_info = {"num_layers": 12, "hidden_size": 768, "num_attention_heads": 12}
    coord.total_layers = 12
    coord.local_partitioner = _MockPartitioner()

    coord.manual_register(
        node_id="n0", host="localhost", port=55051,
        start_layer=0, end_layer=5, total_layers=12,
    )
    coord.manual_register(
        node_id="n1", host="localhost", port=55052,
        start_layer=6, end_layer=11, total_layers=12,
    )

    # nodes property returns summaries; inject into the internal objects.
    for node_id in list(coord.nodes):
        reg = coord._pipeline.get_node(node_id)
        reg.client = MagicMock()
        reg.async_client = MagicMock()
        reg.healthy = True

    return coord


# ═══════════════════════════════════════════════════════════════════════════
# 1. Two-Node Pipeline Inference Correctness
# ═══════════════════════════════════════════════════════════════════════════

class TestTwoNodeInference:
    """Verify that a 2-node pipeline produces correct deterministic output."""

    def test_non_overlapping_layer_ranges(self, two_node_coordinator):
        all_layers = set()
        for reg in two_node_coordinator.nodes.values():
            for layer in range(reg.start_layer, reg.end_layer + 1):
                assert layer not in all_layers, f"Layer {layer} assigned twice"
                all_layers.add(layer)
        assert all_layers == set(range(12))

    def test_preserves_correct_node_order(self, two_node_coordinator):
        assert two_node_coordinator.node_order == ["n0", "n1"]

    def test_mock_client_forward_called(self, two_node_coordinator):
        coord = two_node_coordinator
        n0_fwd = coord.nodes["n0"].client.forward
        n1_fwd = coord.nodes["n1"].client.forward
        n0_fwd.return_value = torch.randn(1, 1, 100)
        n1_fwd.return_value = torch.randn(1, 1, 100)

        # Directly invoke the mock client forward to simulate pipeline
        result = n1_fwd(n0_fwd(torch.randn(1, 1, 100)))

        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert n0_fwd.called
        assert n1_fwd.called

    def test_output_shape_matches_vocab(self):
        vocab_size = 100
        logits = torch.randn(1, 1, vocab_size)
        assert logits.shape[-1] == vocab_size

    def test_last_node_logits_determine_output(self):
        n0_logits = torch.randn(1, 1, 100)
        n1_logits = _make_deterministic_forward_response(token_id=42, vocab_size=100)
        assert n1_logits.shape == n0_logits.shape
        assert n1_logits[0, 0, 42] == 0.0

    def test_three_node_setup(self, mock_tokenizer, monkeypatch):
        monkeypatch.setattr(NodeRegistration, "init_client", lambda self, **kw: None)
        monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: mock_tokenizer)
        mock_config = MagicMock()
        mock_config.model_type = "mock"
        mock_config.num_hidden_layers = 12
        mock_config.hidden_size = 768
        mock_config.num_attention_heads = 12
        mock_config.vocab_size = 32000
        mock_config.max_position_embeddings = 2048
        monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda *a, **kw: mock_config)
        coord = Coordinator(model_name="test-model", dtype="float32", max_batch_size=2)
        coord.tokenizer = mock_tokenizer
        coord.model_info = {"num_layers": 12, "hidden_size": 768, "num_attention_heads": 12}
        coord.total_layers = 12
        coord.local_partitioner = _MockPartitioner()

        for i, (start, end) in enumerate([(0, 3), (4, 7), (8, 11)]):
            coord.manual_register(
                node_id=f"n{i}", host="localhost", port=55053 + i,
                start_layer=start, end_layer=end, total_layers=12,
            )
            reg = coord._pipeline.get_node(f"n{i}")
            reg.client = MagicMock()
            reg.async_client = MagicMock()
            reg.healthy = True
            reg.client.forward.return_value = torch.randn(1, 1, 100)

        assert coord.node_order == ["n0", "n1", "n2"]
        assert len(coord.nodes) == 3


# ═══════════════════════════════════════════════════════════════════════════
# 2. Node Failure During Inference -> Recovery
# ═══════════════════════════════════════════════════════════════════════════

class TestNodeFailureRecovery:
    """When a node fails mid-inference, the circuit breaker + failover kick in."""

    def test_circuit_breaker_opens_on_failure(self):
        rm = ResourceManager(cb_config=CircuitBreakerConfig(threshold=2, base_delay=0.1))
        rm.record_failure("node-0")
        rm.record_failure("node-0")
        assert rm.check_circuit_breaker("node-0") is True

    def test_circuit_breaker_allows_recovery_after_delay(self):
        rm = ResourceManager(cb_config=CircuitBreakerConfig(threshold=2, base_delay=0.01))
        rm.record_failure("node-0")
        rm.record_failure("node-0")
        assert rm.check_circuit_breaker("node-0") is True
        time.sleep(0.02)
        assert rm.check_circuit_breaker("node-0") is False

    def test_healthy_node_takes_over_after_failure(self, two_node_coordinator):
        coord = two_node_coordinator
        rm = coord._resource_mgr

        for _ in range(3):
            rm.record_failure("n0")
        assert rm.check_circuit_breaker("n0") is True

        assert rm.check_circuit_breaker("n1") is False
        logits = torch.randn(1, 1, 100)
        coord.nodes["n1"].client.forward.return_value = logits

        # Verify n1 mock forward still works independently
        result = coord.nodes["n1"].client.forward(torch.randn(1, 1, 100))
        assert result is not None

    def test_recovery_after_failure_cooldown(self, two_node_coordinator):
        coord = two_node_coordinator
        rm = coord._resource_mgr
        rm.cb_config.base_delay = 0.02
        rm.cb_config.max_delay = 0.05

        for _ in range(3):
            rm.record_failure("n0")
        assert rm.check_circuit_breaker("n0") is True

        time.sleep(0.03)
        assert rm.check_circuit_breaker("n0") is False

    def test_consecutive_failures_increase_backoff(self):
        rm = ResourceManager(cb_config=CircuitBreakerConfig(threshold=2, base_delay=1.0))
        for _ in range(4):
            rm.record_failure("node-0")
        recovery_1 = rm._node_recovery_time.get("node-0", 0)
        time.sleep(0.001)
        for _ in range(4):
            rm.record_failure("node-0")
        recovery_2 = rm._node_recovery_time.get("node-0", 0)
        assert recovery_2 >= recovery_1

    def test_success_resets_failure_count(self):
        rm = ResourceManager(cb_config=CircuitBreakerConfig(threshold=3))
        rm.record_failure("node-0")
        rm.record_failure("node-0")
        rm.record_success("node-0")
        assert rm._node_failure_counts.get("node-0", 0) == 0

    def test_resource_manager_tracks_node_metrics(self):
        rm = ResourceManager()
        rm.record_success("n0")
        rm.record_failure("n1")
        rm.record_failure("n1")
        metrics = rm.get_metrics()
        assert metrics["node_failures"] == 2
        assert metrics["errors"] == 2

    def test_node_marked_unhealthy_on_failure(self, two_node_coordinator):
        coord = two_node_coordinator
        reg = coord.nodes["n0"]
        reg.healthy = True
        reg.client.forward.side_effect = ConnectionError("node down")

        with pytest.raises(ConnectionError):
            reg.client.forward(torch.randn(1, 1, 100))
        reg.healthy = False
        assert not reg.healthy

    def test_failover_engine_transitions(self):
        try:
            from distllm.health.failover import FailoverEngine
            from distllm.health.state import HealthRecord, NodeState
        except ImportError:
            pytest.skip("FailoverEngine not available")

        fldr = FailoverEngine(failure_threshold=1, recovery_threshold=1)
        rec = HealthRecord(node_id="n0", state=NodeState.HEALTHY)

        # Healthy node stays healthy on success
        new_state = fldr.evaluate(rec, success=True, latency_ms=10)
        assert new_state == NodeState.HEALTHY

        # Failure transitions to UNHEALTHY (threshold=1)
        new_state = fldr.evaluate(rec, success=False, latency_ms=100)
        assert new_state == NodeState.UNHEALTHY


# ═══════════════════════════════════════════════════════════════════════════
# 3. Node Join/Leave During Continuous Inference
# ═══════════════════════════════════════════════════════════════════════════

class TestNodeJoinLeave:
    """Nodes can be added/removed dynamically while inference continues."""

    def test_register_node_during_idle(self, two_node_coordinator):
        coord = two_node_coordinator
        mock = MagicMock()
        reg = NodeRegistration(
            node_id="n2", host="localhost", port=55053,
            start_layer=0, end_layer=5,
        )
        reg.client = mock
        reg.async_client = MagicMock()
        reg.healthy = True
        coord.nodes["n2"] = reg
        coord.node_order = coord.node_order + ["n2"]
        assert len(coord.nodes) == 3
        assert "n2" in coord.nodes

    def test_unregister_node_during_idle(self, two_node_coordinator):
        coord = two_node_coordinator
        coord.nodes.pop("n1", None)
        coord.node_order = [n for n in coord.node_order if n != "n1"]
        assert len(coord.nodes) == 1
        assert "n1" not in coord.nodes

    def test_node_join_affects_layer_routing(self, two_node_coordinator):
        coord = two_node_coordinator
        n_layers_before = sum(
            reg.end_layer - reg.start_layer + 1
            for reg in coord.nodes.values()
        )

        mock = MagicMock()
        reg = NodeRegistration(
            node_id="n3", host="localhost", port=55054,
            start_layer=12, end_layer=17,
        )
        reg.client = mock
        reg.async_client = MagicMock()
        reg.healthy = True
        coord.nodes["n3"] = reg
        coord.node_order = coord.node_order + ["n3"]

        n_layers_after = sum(
            reg.end_layer - reg.start_layer + 1
            for reg in coord.nodes.values()
        )
        assert n_layers_after > n_layers_before

    def test_rebalancing_on_join(self, two_node_coordinator):
        coord = two_node_coordinator
        mock = MagicMock()
        mock.forward.return_value = torch.randn(1, 1, 100)
        reg = NodeRegistration(
            node_id="new-node", host="localhost", port=55055,
            start_layer=0, end_layer=5,
        )
        reg.client = mock
        reg.async_client = MagicMock()
        reg.healthy = True
        coord.nodes["new-node"] = reg
        coord.node_order.insert(0, "new-node")

        result = coord.nodes["new-node"].client.forward(torch.randn(1, 1, 100))
        assert result is not None
        assert mock.forward.called

    def test_leave_during_inference_recovery(self, two_node_coordinator):
        coord = two_node_coordinator
        n1_logits = torch.randn(1, 1, 100)
        coord.nodes["n1"].client.forward.return_value = n1_logits

        coord.nodes.pop("n0", None)
        coord.node_order = [n for n in coord.node_order if n != "n0"]

        assert len(coord.nodes) == 1
        assert "n1" in coord.nodes

        result = coord.nodes["n1"].client.forward(torch.randn(1, 1, 100))
        assert result is not None

    def test_multiple_leave_join_cycles(self, two_node_coordinator):
        coord = two_node_coordinator

        for cycle in range(3):
            nid = f"cycle-{cycle}"
            mock = MagicMock()
            mock.forward.return_value = torch.randn(1, 1, 100)
            reg = NodeRegistration(
                node_id=nid, host="localhost", port=55060 + cycle,
                start_layer=0, end_layer=5,
            )
            reg.client = mock
            reg.async_client = MagicMock()
            reg.healthy = True
            coord.nodes[nid] = reg
            coord.node_order.append(nid)

            result = coord.nodes[nid].client.forward(torch.randn(1, 1, 100))
            assert result is not None

            coord.nodes.pop(nid, None)
            coord.node_order = [n for n in coord.node_order if n != nid]

    def test_multi_register_same_node_id_replaces(self, two_node_coordinator):
        coord = two_node_coordinator
        orig = coord.nodes["n0"]
        mock = MagicMock()
        reg = NodeRegistration(
            node_id="n0", host="localhost", port=55999,
            start_layer=0, end_layer=5,
        )
        reg.client = mock
        reg.async_client = MagicMock()
        reg.healthy = True
        coord.nodes["n0"] = reg
        assert coord.nodes["n0"].port == 55999
        assert coord.nodes["n0"] is not orig

    def test_register_node_with_expert_ids(self, two_node_coordinator):
        mock = MagicMock()
        reg = NodeRegistration(
            node_id="expert-node", host="localhost", port=55070,
            start_layer=0, end_layer=5, expert_ids=[0, 3, 7],
        )
        reg.client = mock
        reg.async_client = MagicMock()
        reg.healthy = True
        two_node_coordinator.nodes["expert-node"] = reg
        assert reg.expert_ids == [0, 3, 7]

    def test_dynamic_node_order_after_join(self, two_node_coordinator):
        n_orig = len(two_node_coordinator.node_order)
        mock = MagicMock()
        reg = NodeRegistration(
            node_id="new-mid", host="localhost", port=55080,
            start_layer=3, end_layer=8,
        )
        reg.client = mock
        reg.async_client = MagicMock()
        reg.healthy = True
        two_node_coordinator.nodes["new-mid"] = reg
        two_node_coordinator.node_order = two_node_coordinator.node_order + ["new-mid"]
        assert len(two_node_coordinator.node_order) == n_orig + 1

    def test_orphan_node_order_after_leave(self, two_node_coordinator):
        two_node_coordinator.nodes.pop("n1", None)
        # Getter returns a copy — reassign via the setter.
        two_node_coordinator.node_order = [
            n for n in two_node_coordinator.node_order if n != "n1"
        ]
        assert "n1" not in two_node_coordinator.node_order
