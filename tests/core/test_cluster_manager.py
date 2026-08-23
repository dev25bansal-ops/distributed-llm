"""Tests for ClusterManager -- node lifecycle and cluster management.

Covers:
    construction/defaults
    properties (nodes, node_order, node_count)
    auto_setup
    manual_register (basic, total_layers, weight distribution)
    register_nodes_batch
    scale_pipeline_capacity
    get_node_gpu_summary
    internal weight source registry (_register_weight_source, _get_weight_source)

Every test is deterministic (no network, no GPU, no time.sleep).
No MagicMock -- lightweight plain-Python stubs only.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_cluster_mod = load_module("distllm/core/cluster_manager.py")
ClusterManager = _cluster_mod.ClusterManager


# ===================================================================
# Stubs
# ===================================================================


class _StubPipeline:
    """Minimal PipelineOrchestrator substitute.

    Provides nodes, node_order, total_layers properties and setters,
    plus scale_concurrent_requests() for capacity scaling.
    """

    def __init__(self) -> None:
        self._nodes: dict = {}
        self._node_order: list[str] = []
        self._total_layers: int = 0

    @property
    def nodes(self) -> dict:
        return self._nodes

    @nodes.setter
    def nodes(self, value: dict) -> None:
        self._nodes = value

    @property
    def node_order(self) -> list[str]:
        return self._node_order

    @node_order.setter
    def node_order(self, value: list[str]) -> None:
        self._node_order = value

    @property
    def total_layers(self) -> int:
        return self._total_layers

    @total_layers.setter
    def total_layers(self, value: int) -> None:
        self._total_layers = value

    def scale_concurrent_requests(self, per_node_limit: int = 16) -> int:
        """Return stub capacity based on node count and limit."""
        return len(self._nodes) * per_node_limit


class _StubNodeRegistrar:
    """Minimal NodeRegistrar substitute without HuggingFace dependency."""

    def __init__(
        self,
        pipeline=None,
        model_name: str = "",
        trust_remote_code: bool | None = None,
    ):
        self.pipeline = pipeline
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.registrations: list[dict] = []
        self.auto_setup_called = False
        self.last_batch_call: dict | None = None

    def auto_setup(self, nodes_config: list[dict]) -> tuple[dict, int]:
        self.auto_setup_called = True
        self.registrations = list(nodes_config)
        return {"model_type": "test", "num_layers": 24, "hidden_size": 1024}, 24

    def manual_register(self, node_id: str, host: str, port: int,
                        start_layer: int, end_layer: int,
                        **kwargs) -> None:
        self.registrations.append({
            "node_id": node_id,
            "host": host,
            "port": port,
            "start_layer": start_layer,
            "end_layer": end_layer,
            **kwargs,
        })

    def register_nodes_batch(
        self,
        nodes_config: list[dict],
        cluster_key: str | None = None,
        max_workers: int = 8,
    ) -> dict[str, dict]:
        self.last_batch_call = {
            "nodes_config": nodes_config,
            "cluster_key": cluster_key,
            "max_workers": max_workers,
        }
        results: dict[str, dict] = {}
        for i, nc in enumerate(nodes_config):
            nid = nc.get("node_id", f"node_{i}")
            results[nid] = {
                "success": True,
                "gpu_name": "TestGPU",
                "gpu_memory_gb": 16.0,
                "error": None,
            }
        return results


# Replace NodeRegistrar in the loaded module so ClusterManager.__init__
# constructs our stub instead of the real one (which would trigger HF).
_cluster_mod.NodeRegistrar = _StubNodeRegistrar


# ===================================================================
# CONSTRUCTION / DEFAULTS
# ===================================================================


class TestClusterManagerConstruction:
    """ClusterManager.__init__ -- defaults and attribute setup."""

    def test_default_construction(self) -> None:
        """Minimal construction should set sensible defaults."""
        pipeline = _StubPipeline()
        mgr = ClusterManager(pipeline=pipeline, model_name="test-model")

        assert mgr._model_name == "test-model"
        assert mgr._trust_remote_code is None
        assert mgr._cluster_key is None
        assert mgr._distribute_weights is True
        assert mgr._model_registry == {}
        assert mgr.tokenizer is None
        assert mgr.model_info is None
        assert mgr.total_layers == 0
        assert mgr.model_revision == "main"
        assert mgr._pipeline is pipeline
        # NodeRegistrar was created with pipeline and model_name
        reg = mgr._node_registrar
        assert reg.pipeline is pipeline
        assert reg.model_name == "test-model"
        assert reg.trust_remote_code is None

    def test_custom_construction(self) -> None:
        """Custom trust_remote_code and cluster_key should be stored."""
        pipeline = _StubPipeline()
        mgr = ClusterManager(
            pipeline=pipeline,
            model_name="custom-model",
            trust_remote_code=True,
            cluster_key="secret-key",
        )
        assert mgr._model_name == "custom-model"
        assert mgr._trust_remote_code is True
        assert mgr._cluster_key == "secret-key"

    def test_node_registrar_is_stub(self) -> None:
        """Verify our stub is used (no HF call on construction)."""
        pipeline = _StubPipeline()
        mgr = ClusterManager(pipeline=pipeline, model_name="test-model")
        assert isinstance(mgr._node_registrar, _StubNodeRegistrar)


# ===================================================================
# PROPERTIES
# ===================================================================


class TestNodesProperties:
    """nodes, node_order, node_count -- delegation to pipeline."""

    def test_nodes_property_delegates_to_pipeline(self) -> None:
        pipeline = _StubPipeline()
        pipeline._nodes = {"n1": {"host": "10.0.0.1", "port": 50051}}
        mgr = ClusterManager(pipeline=pipeline, model_name="m")
        assert mgr.nodes == {"n1": {"host": "10.0.0.1", "port": 50051}}

    def test_nodes_setter(self) -> None:
        pipeline = _StubPipeline()
        mgr = ClusterManager(pipeline=pipeline, model_name="m")
        mgr.nodes = {"n1": {"host": "10.0.0.1"}}
        assert pipeline._nodes == {"n1": {"host": "10.0.0.1"}}

    def test_node_order_property_delegates(self) -> None:
        pipeline = _StubPipeline()
        pipeline._node_order = ["n1", "n2"]
        mgr = ClusterManager(pipeline=pipeline, model_name="m")
        assert mgr.node_order == ["n1", "n2"]

    def test_node_order_setter(self) -> None:
        pipeline = _StubPipeline()
        mgr = ClusterManager(pipeline=pipeline, model_name="m")
        mgr.node_order = ["n2", "n1"]
        assert pipeline._node_order == ["n2", "n1"]

    def test_node_count_empty(self) -> None:
        mgr = ClusterManager(pipeline=_StubPipeline(), model_name="m")
        assert mgr.node_count == 0

    def test_node_count_with_nodes(self) -> None:
        pipeline = _StubPipeline()
        pipeline._nodes = {"n1": {}, "n2": {}}
        mgr = ClusterManager(pipeline=pipeline, model_name="m")
        assert mgr.node_count == 2

    def test_node_count_reflects_setter(self) -> None:
        pipeline = _StubPipeline()
        mgr = ClusterManager(pipeline=pipeline, model_name="m")
        mgr.nodes = {"n1": {}, "n2": {}, "n3": {}}
        assert mgr.node_count == 3

    def test_node_count_drops_to_zero(self) -> None:
        pipeline = _StubPipeline()
        mgr = ClusterManager(pipeline=pipeline, model_name="m")
        mgr.nodes = {"n1": {}}
        assert mgr.node_count == 1
        mgr.nodes = {}
        assert mgr.node_count == 0


# ===================================================================
# AUTO SETUP
# ===================================================================


class TestAutoSetup:
    """ClusterManager.auto_setup -- delegates to registrar, sets metadata."""

    def test_auto_setup_calls_registrar(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained",
            lambda *a, **kw: None,
        )
        pipeline = _StubPipeline()
        mgr = ClusterManager(pipeline=pipeline, model_name="test-model")
        configs = [{"host": "10.0.0.1", "port": 50051, "node_id": "n1"}]
        mgr.auto_setup(configs)
        assert mgr._node_registrar.auto_setup_called is True

    def test_auto_setup_sets_model_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained",
            lambda *a, **kw: None,
        )
        pipeline = _StubPipeline()
        mgr = ClusterManager(pipeline=pipeline, model_name="test-model")
        mgr.auto_setup([{"host": "10.0.0.1"}])
        assert mgr.model_info == {
            "model_type": "test",
            "num_layers": 24,
            "hidden_size": 1024,
        }
        assert mgr.total_layers == 24

    def test_auto_setup_updates_pipeline_total_layers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained",
            lambda *a, **kw: None,
        )
        pipeline = _StubPipeline()
        mgr = ClusterManager(pipeline=pipeline, model_name="test-model")
        mgr.auto_setup([{"host": "10.0.0.1"}])
        assert pipeline.total_layers == 24

    def test_auto_setup_empty_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained",
            lambda *a, **kw: None,
        )
        pipeline = _StubPipeline()
        mgr = ClusterManager(pipeline=pipeline, model_name="test-model")
        mgr.auto_setup([])
        assert mgr._node_registrar.registrations == []


# ===================================================================
# MANUAL REGISTER
# ===================================================================


class TestManualRegister:
    """ClusterManager.manual_register -- registrations, total_layers, weight tracking."""

    def test_manual_register_basic(self) -> None:
        pipeline = _StubPipeline()
        mgr = ClusterManager(pipeline=pipeline, model_name="test-model")
        mgr.tokenizer = object()  # prevent HF AutoTokenizer call
        mgr.model_info = {"num_layers": 24}

        mgr.manual_register("n1", "10.0.0.1", 50051, 0, 7, total_layers=24)

        regs = mgr._node_registrar.registrations
        assert len(regs) == 1
        assert regs[0]["node_id"] == "n1"
        assert regs[0]["host"] == "10.0.0.1"
        assert regs[0]["port"] == 50051
        assert regs[0]["start_layer"] == 0
        assert regs[0]["end_layer"] == 7

    def test_manual_register_sets_pipeline_total_layers(self) -> None:
        pipeline = _StubPipeline()
        mgr = ClusterManager(pipeline=pipeline, model_name="test-model")
        mgr.tokenizer = object()
        mgr.model_info = {"num_layers": 24}

        mgr.manual_register("n1", "10.0.0.1", 50051, 0, 7, total_layers=24)
        assert pipeline.total_layers == 24

    def test_manual_register_skips_total_layers_when_none(self) -> None:
        pipeline = _StubPipeline()
        mgr = ClusterManager(pipeline=pipeline, model_name="test-model")
        mgr.tokenizer = object()
        mgr.model_info = {"num_layers": 24}
        pipeline.total_layers = 10

        mgr.manual_register("n1", "10.0.0.1", 50051, 0, 7, total_layers=None)
        assert pipeline.total_layers == 10  # unchanged

    def test_manual_register_infers_layers_from_model_info(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When total_layers is None and model_info is None, call get_model_info."""
        monkeypatch.setattr(
            _cluster_mod,
            "get_model_info",
            lambda *a, **kw: {"num_layers": 32},
        )
        pipeline = _StubPipeline()
        mgr = ClusterManager(pipeline=pipeline, model_name="test-model")
        mgr.tokenizer = object()
        # model_info is None (default) -- triggers the inference path
        assert mgr.model_info is None

        mgr.manual_register("n1", "10.0.0.1", 50051, 0, 7, total_layers=None)
        assert mgr.total_layers == 32
        assert pipeline.total_layers == 32

    def test_manual_register_presets_tokenizer_skips_load(self) -> None:
        """If tokenizer is already set, from_pretrained is not called."""
        pipeline = _StubPipeline()
        mgr = ClusterManager(pipeline=pipeline, model_name="test-model")
        sentinel = object()
        mgr.tokenizer = sentinel
        mgr.model_info = {"num_layers": 24}

        mgr.manual_register("n1", "10.0.0.1", 50051, 0, 7)
        assert mgr.tokenizer is sentinel  # unchanged

    def test_manual_register_passes_cluster_key(self) -> None:
        pipeline = _StubPipeline()
        mgr = ClusterManager(
            pipeline=pipeline,
            model_name="test-model",
            cluster_key="my-key",
        )
        mgr.tokenizer = object()
        mgr.model_info = {"num_layers": 24}

        mgr.manual_register("n1", "10.0.0.1", 50051, 0, 7)
        regs = mgr._node_registrar.registrations
        assert regs[0].get("cluster_key") == "my-key"

    def test_manual_register_with_expert_ids(self) -> None:
        pipeline = _StubPipeline()
        mgr = ClusterManager(pipeline=pipeline, model_name="test-model")
        mgr.tokenizer = object()
        mgr.model_info = {"num_layers": 24}

        mgr.manual_register("n1", "10.0.0.1", 50051, 0, 7, expert_ids=[0, 1, 2])
        regs = mgr._node_registrar.registrations
        assert regs[0].get("expert_ids") == [0, 1, 2]

    def test_manual_register_with_role(self) -> None:
        pipeline = _StubPipeline()
        mgr = ClusterManager(pipeline=pipeline, model_name="test-model")
        mgr.tokenizer = object()
        mgr.model_info = {"num_layers": 24}

        mgr.manual_register("n1", "10.0.0.1", 50051, 0, 7, role="prefill")
        regs = mgr._node_registrar.registrations
        assert regs[0].get("role") == "prefill"

    def test_manual_register_with_cluster_id(self) -> None:
        pipeline = _StubPipeline()
        mgr = ClusterManager(pipeline=pipeline, model_name="test-model")
        mgr.tokenizer = object()
        mgr.model_info = {"num_layers": 24}

        mgr.manual_register("n1", "10.0.0.1", 50051, 0, 7, cluster_id="my-cluster")
        regs = mgr._node_registrar.registrations
        assert regs[0].get("cluster_id") == "my-cluster"

    def test_manual_register_weight_source_second_call(self) -> None:
        """Second call with same (model, start_layer, end_layer) auto-assigns weight_source."""
        pipeline = _StubPipeline()
        pipeline._nodes = {"n1": {"host": "10.0.0.1", "port": 50051}}
        mgr = ClusterManager(pipeline=pipeline, model_name="test-model")
        mgr.tokenizer = object()
        mgr.model_info = {"num_layers": 24}

        # First registration populates weight source registry
        mgr.manual_register("n1", "10.0.0.1", 50051, 0, 7)
        assert mgr._node_registrar.registrations[0].get("weight_source") is None

        # Second registration with same layers should find the first as source
        mgr._node_registrar.registrations.clear()
        pipeline._nodes["n2"] = {"host": "10.0.0.2", "port": 50052}
        mgr.manual_register("n2", "10.0.0.2", 50052, 0, 7)
        regs = mgr._node_registrar.registrations
        assert regs[0].get("weight_source") == "10.0.0.1:50051"


# ===================================================================
# BATCH REGISTRATION
# ===================================================================


class TestRegisterNodesBatch:
    """ClusterManager.register_nodes_batch -- delegation to registrar."""

    def test_batch_empty(self) -> None:
        mgr = ClusterManager(pipeline=_StubPipeline(), model_name="m")
        result = mgr.register_nodes_batch([])
        assert result == {}

    def test_batch_single(self) -> None:
        mgr = ClusterManager(pipeline=_StubPipeline(), model_name="m")
        result = mgr.register_nodes_batch([{"node_id": "n1"}])
        assert result["n1"]["success"] is True
        assert result["n1"]["gpu_name"] == "TestGPU"

    def test_batch_multiple(self) -> None:
        mgr = ClusterManager(pipeline=_StubPipeline(), model_name="m")
        configs = [
            {"node_id": "n1", "host": "10.0.0.1"},
            {"node_id": "n2", "host": "10.0.0.2"},
        ]
        result = mgr.register_nodes_batch(configs)
        assert len(result) == 2
        assert result["n1"]["success"] is True
        assert result["n2"]["success"] is True
        assert result["n1"]["error"] is None

    def test_batch_passes_cluster_key(self) -> None:
        mgr = ClusterManager(
            pipeline=_StubPipeline(),
            model_name="m",
            cluster_key="my-key",
        )
        mgr.register_nodes_batch([{"node_id": "n1"}])
        assert mgr._node_registrar.last_batch_call is not None
        assert mgr._node_registrar.last_batch_call["cluster_key"] == "my-key"
        assert mgr._node_registrar.last_batch_call["max_workers"] == 8

    def test_batch_passes_override_cluster_key(self) -> None:
        """Explicit cluster_key argument overrides the one from construction."""
        mgr = ClusterManager(
            pipeline=_StubPipeline(),
            model_name="m",
            cluster_key="default-key",
        )
        mgr.register_nodes_batch([{"node_id": "n1"}], cluster_key="override-key")
        assert mgr._node_registrar.last_batch_call["cluster_key"] == "override-key"


# ===================================================================
# SCALE PIPELINE CAPACITY
# ===================================================================


class TestScalePipelineCapacity:
    """ClusterManager.scale_pipeline_capacity -- delegation to pipeline."""

    def test_scale_no_nodes(self) -> None:
        mgr = ClusterManager(pipeline=_StubPipeline(), model_name="m")
        assert mgr.scale_pipeline_capacity(16) == 0

    def test_scale_with_nodes(self) -> None:
        pipeline = _StubPipeline()
        pipeline._nodes = {"n1": {}, "n2": {}}
        mgr = ClusterManager(pipeline=pipeline, model_name="m")
        assert mgr.scale_pipeline_capacity(16) == 32

    def test_scale_custom_limit(self) -> None:
        pipeline = _StubPipeline()
        pipeline._nodes = {"n1": {}}
        mgr = ClusterManager(pipeline=pipeline, model_name="m")
        assert mgr.scale_pipeline_capacity(per_node_limit=8) == 8


# ===================================================================
# GPU SUMMARY
# ===================================================================


class TestGetNodeGpuSummary:
    """ClusterManager.get_node_gpu_summary -- per-node GPU info."""

    def test_empty_nodes_returns_empty_dict(self) -> None:
        mgr = ClusterManager(pipeline=_StubPipeline(), model_name="m")
        assert mgr.get_node_gpu_summary() == {}

    def test_dict_nodes_use_getattr_defaults(self) -> None:
        """With dict-valued nodes, getattr returns default values."""
        pipeline = _StubPipeline()
        pipeline._nodes = {"n1": {"host": "10.0.0.1"}}
        mgr = ClusterManager(pipeline=pipeline, model_name="m")

        summary = mgr.get_node_gpu_summary()
        assert summary["n1"]["gpu_name"] == ""
        assert summary["n1"]["memory_total_gb"] == 0.0
        assert summary["n1"]["memory_free_gb"] == 0.0

    def test_object_nodes_with_gpu_attrs(self) -> None:
        """Object-valued nodes with gpu_name/memory should report correctly."""
        pipeline = _StubPipeline()
        node = SimpleNamespace(
            gpu_name="NVIDIA A100",
            gpu_memory_total=80 * 1024 ** 3,   # 80 GB in bytes
            gpu_memory_free=40 * 1024 ** 3,    # 40 GB in bytes
        )
        pipeline._nodes = {"n1": node}
        mgr = ClusterManager(pipeline=pipeline, model_name="m")

        summary = mgr.get_node_gpu_summary()
        assert summary["n1"]["gpu_name"] == "NVIDIA A100"
        assert summary["n1"]["memory_total_gb"] == pytest.approx(80.0, rel=0.01)
        assert summary["n1"]["memory_free_gb"] == pytest.approx(40.0, rel=0.01)

    def test_partial_gpu_attrs_default_to_zero(self) -> None:
        """Nodes with only some GPU attributes get defaults for the rest."""
        pipeline = _StubPipeline()
        node = SimpleNamespace(
            gpu_name="NVIDIA T4",
            # gpu_memory_total missing
            gpu_memory_free=16 * 1024 ** 3,   # 16 GB free
        )
        pipeline._nodes = {"n1": node}
        mgr = ClusterManager(pipeline=pipeline, model_name="m")

        summary = mgr.get_node_gpu_summary()
        assert summary["n1"]["gpu_name"] == "NVIDIA T4"
        assert summary["n1"]["memory_total_gb"] == 0.0  # default
        assert summary["n1"]["memory_free_gb"] == pytest.approx(16.0, rel=0.01)


# ===================================================================
# WEIGHT SOURCE REGISTRY
# ===================================================================


class TestWeightSourceRegistry:
    """Internal _register_weight_source / _get_weight_source."""

    def test_weight_source_round_trip(self) -> None:
        pipeline = _StubPipeline()
        pipeline._nodes = {"n1": {"host": "10.0.0.1", "port": 50051}}
        mgr = ClusterManager(pipeline=pipeline, model_name="test-model")

        mgr._register_weight_source("n1", "test-model", 0, 7)
        result = mgr._get_weight_source("test-model", 0, 7)
        assert result == ("10.0.0.1", 50051)

    def test_weight_source_not_found(self) -> None:
        mgr = ClusterManager(pipeline=_StubPipeline(), model_name="test-model")
        result = mgr._get_weight_source("nonexistent", 0, 7)
        assert result is None

    def test_weight_source_overwrite(self) -> None:
        pipeline = _StubPipeline()
        pipeline._nodes = {"n1": {"host": "10.0.0.1", "port": 50051}}
        mgr = ClusterManager(pipeline=pipeline, model_name="test-model")

        mgr._register_weight_source("n1", "test-model", 0, 7)
        assert mgr._get_weight_source("test-model", 0, 7) == ("10.0.0.1", 50051)

        # Change pipeline node data and register again with same key
        pipeline._nodes["n1"] = {"host": "10.0.0.2", "port": 50052}
        mgr._register_weight_source("n1", "test-model", 0, 7)
        # Overwritten with new host/port
        assert mgr._get_weight_source("test-model", 0, 7) == ("10.0.0.2", 50052)

    def test_weight_source_node_as_object(self) -> None:
        """_register_weight_source also handles object-style nodes."""
        pipeline = _StubPipeline()
        node = SimpleNamespace(host="10.0.0.1", port=50051)
        pipeline._nodes = {"n1": node}
        mgr = ClusterManager(pipeline=pipeline, model_name="test-model")

        mgr._register_weight_source("n1", "test-model", 0, 7)
        result = mgr._get_weight_source("test-model", 0, 7)
        assert result == ("10.0.0.1", 50051)

    def test_weight_source_node_none(self) -> None:
        """When pipeline node is missing, uses defaults."""
        pipeline = _StubPipeline()
        # _nodes is empty, so node lookup returns None
        mgr = ClusterManager(pipeline=pipeline, model_name="test-model")

        mgr._register_weight_source("missing-node", "test-model", 0, 7)
        result = mgr._get_weight_source("test-model", 0, 7)
        # Host defaults to "unknown", port to 0
        assert result == ("unknown", 0)

    def test_get_weight_source_different_model(self) -> None:
        """Different model name should not find the entry."""
        pipeline = _StubPipeline()
        pipeline._nodes = {"n1": {"host": "10.0.0.1", "port": 50051}}
        mgr = ClusterManager(pipeline=pipeline, model_name="test-model")

        mgr._register_weight_source("n1", "model-a", 0, 7)
        # Different model name
        result = mgr._get_weight_source("model-b", 0, 7)
        assert result is None

    def test_get_weight_source_different_layers(self) -> None:
        """Different layer range should not find the entry."""
        pipeline = _StubPipeline()
        pipeline._nodes = {"n1": {"host": "10.0.0.1", "port": 50051}}
        mgr = ClusterManager(pipeline=pipeline, model_name="test-model")

        mgr._register_weight_source("n1", "test-model", 0, 7)
        # Same model, different layers
        result = mgr._get_weight_source("test-model", 8, 15)
        assert result is None

    def test_distribute_weights_disabled(self) -> None:
        """When _distribute_weights is False, weight_source is never set."""
        pipeline = _StubPipeline()
        pipeline._nodes = {"n1": {"host": "10.0.0.1", "port": 50051}}
        mgr = ClusterManager(pipeline=pipeline, model_name="test-model")
        mgr.tokenizer = object()
        mgr.model_info = {"num_layers": 24}
        mgr._distribute_weights = False

        # Even with matching layers, no weight_source
        mgr.manual_register("n1", "10.0.0.1", 50051, 0, 7)
        regs = mgr._node_registrar.registrations
        assert regs[0].get("weight_source") is None
        # _model_registry should still be empty
        assert mgr._model_registry == {}
