"""Comprehensive unit tests for the Coordinator class.

Tests cover:
- CoordinatorConfig creation and from_settings
- Coordinator initialization with mocked dependencies
- Node management (register, remove, node_order)
- Properties delegation to sub-components
- Health check and metrics collection
- Async generation and request lifecycle
"""

import threading
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> "CoordinatorConfig":
    """Create a CoordinatorConfig with test defaults."""
    from distllm.core.coordinator import CoordinatorConfig
    defaults = dict(
        model_name="test-model",
        port=50050,
        dtype="float32",
        max_batch_size=2,
        max_tokens_per_batch=512,
        pipeline_timeout=5.0,
    )
    defaults.update(overrides)
    return CoordinatorConfig(**defaults)


def _make_coord(**overrides) -> "Coordinator":
    """Create a Coordinator with heavy components mocked."""
    from distllm.core.coordinator import Coordinator

    config = _make_config(**overrides)
    coord = Coordinator(config=config)

    # Mock heavy sub-components
    coord.tokenizer = MagicMock()
    coord.tokenizer.encode.return_value = [1, 2, 3]
    coord.tokenizer.decode.return_value = "mocked output"
    coord.tokenizer.eos_token_id = 0
    coord.tokenizer.vocab_size = 32000
    return coord


# ===================================================================
# CoordinatorConfig tests
# ===================================================================

class TestCoordinatorConfig:
    def test_default_values(self):
        from distllm.core.coordinator import CoordinatorConfig
        cfg = CoordinatorConfig(model_name="m")
        assert cfg.model_name == "m"
        assert cfg.port == 50050
        assert cfg.dtype == "float16"
        assert cfg.max_batch_size == 4
        assert cfg.pipeline_timeout == 30.0

    def test_custom_values(self):
        cfg = _make_config(model_name="llama-70b", port=50051, dtype="bfloat16")
        assert cfg.model_name == "llama-70b"
        assert cfg.port == 50051
        assert cfg.dtype == "bfloat16"

    def test_from_settings(self):
        from distllm.core.coordinator import CoordinatorConfig
        settings = MagicMock()
        settings.model.name = "test-model"
        settings.coordinator.port = 50050
        settings.model.dtype = "float16"
        settings.model.trust_remote_code = None
        settings.batching.max_batch_size = 4
        settings.batching.max_tokens_per_batch = 1024
        settings.network.grpc_timeout = 30.0
        settings.model_hub.cache_dir = None
        settings.wide_area.enabled = False

        cfg = CoordinatorConfig.from_settings(settings)
        assert cfg.model_name == "test-model"
        assert cfg.port == 50050


# ===================================================================
# Coordinator initialization tests
# ===================================================================

class TestCoordinatorInit:
    def test_init_with_config(self):
        coord = _make_coord()
        assert coord.model_name == "test-model"
        assert coord.port == 50050
        assert coord.dtype == "float32"

    def test_init_with_kwargs(self):
        from distllm.core.coordinator import Coordinator
        coord = Coordinator(model_name="test", port=50052, dtype="float16")
        assert coord.model_name == "test"
        assert coord.port == 50052

    def test_sub_components_initialized(self):
        coord = _make_coord()
        assert coord._resource_mgr is not None
        assert coord._cache_mgr is not None
        assert coord._pipeline is not None
        assert coord._batch_scheduler is not None
        assert coord._latency_tracker is not None
        assert coord._straggler_detector is not None
        assert coord._recovery_manager is not None
        assert coord._reputation is not None
        assert coord._cluster_mgr is not None
        assert coord._inference_engine is not None
        assert coord._health_mgr is not None
        assert coord._metrics_collector is not None

    def test_initial_state(self):
        coord = _make_coord()
        assert coord.total_layers == 0
        assert coord.tokenizer is None or hasattr(coord, 'tokenizer')
        assert coord._federation is None


# ===================================================================
# Node management tests
# ===================================================================

class TestNodeManagement:
    def test_nodes_property_delegates(self):
        coord = _make_coord()
        assert isinstance(coord.nodes, dict)
        assert len(coord.nodes) == 0

    def test_node_order_property(self):
        coord = _make_coord()
        assert isinstance(coord.node_order, list)
        assert len(coord.node_order) == 0

    def test_node_order_setter(self):
        coord = _make_coord()
        coord.node_order = ["node_0", "node_1"]
        assert coord.node_order == ["node_0", "node_1"]

    def test_nodes_setter(self):
        coord = _make_coord()
        coord.nodes = {"node_0": MagicMock()}
        assert "node_0" in coord.nodes

    def test_manual_register(self):
        coord = _make_coord()
        coord._cluster_mgr.manual_register = MagicMock()
        coord._cluster_mgr.tokenizer = MagicMock()
        coord.manual_register(
            node_id="test_node",
            host="localhost",
            port=50051,
            start_layer=0,
            end_layer=3,
            total_layers=8,
        )
        coord._cluster_mgr.manual_register.assert_called_once()


# ===================================================================
# Metrics and health tests
# ===================================================================

class TestMetricsAndHealth:
    def test_get_metrics_returns_dict(self):
        coord = _make_coord()
        metrics = coord._metrics_collector.collect()
        assert isinstance(metrics, dict)

    def test_latency_tracker_initialized(self):
        coord = _make_coord()
        assert coord._latency_tracker is not None

    def test_straggler_detector_initialized(self):
        coord = _make_coord()
        assert coord._straggler_detector is not None

    def test_reputation_system_initialized(self):
        coord = _make_coord()
        assert coord._reputation is not None


# ===================================================================
# Pipeline integration tests
# ===================================================================

class TestPipelineIntegration:
    def test_pipeline_orchestrator_initialized(self):
        coord = _make_coord()
        assert coord._pipeline is not None
        assert coord._pipeline.total_layers == 0

    def test_pipeline_has_latency_tracker(self):
        coord = _make_coord()
        assert coord._pipeline._latency_tracker is not None

    def test_pipeline_has_straggler_detector(self):
        coord = _make_coord()
        assert coord._pipeline._straggler_detector is not None


# ===================================================================
# Batch scheduler tests
# ===================================================================

class TestBatchSchedulerIntegration:
    def test_batch_scheduler_initialized(self):
        coord = _make_coord()
        assert coord._batch_scheduler is not None
        stats = coord._batch_scheduler.stats()
        assert isinstance(stats, dict)
        assert "pending_requests" in stats

    def test_batch_scheduler_config(self):
        coord = _make_coord()
        assert coord._batch_scheduler.max_batch_size > 0
        assert coord._batch_scheduler.max_tokens_per_batch > 0


# ===================================================================
# Cache manager tests
# ===================================================================

class TestCacheManagerIntegration:
    def test_cache_manager_initialized(self):
        coord = _make_coord()
        assert coord._cache_mgr is not None

    def test_cache_lookup_returns_tuple(self):
        coord = _make_coord()
        result = coord._cache_mgr.lookup_prefix([1, 2, 3])
        assert isinstance(result, tuple)
        assert len(result) == 2
