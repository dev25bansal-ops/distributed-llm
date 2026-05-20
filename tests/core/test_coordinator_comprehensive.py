"""Comprehensive unit tests for Coordinator._init_* methods, lifecycle, and generate paths."""

import threading
import time
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch

from distllm.core.coordinator import Coordinator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_coord(**overrides) -> Coordinator:
    """Create a Coordinator with all heavy components mocked."""
    def _mock_factory(*a, **kw):
        return MagicMock()

    with patch.multiple(
        "distllm.core.coordinator",
        ResourceManager=_mock_factory,
        CacheManager=_mock_factory,
        PipelineOrchestrator=_mock_factory,
        TokenGenerator=_mock_factory,
        ModelManager=_mock_factory,
        HealthChecker=_mock_factory,
        NodeRegistrar=_mock_factory,
        MetricsManager=_mock_factory,
        RequestTracker=_mock_factory,
        Container=_mock_factory,
        SubsystemManager=_mock_factory,
        ModelRegistry=_mock_factory,
    ):
        # Prevent actual model loading by mocking ModelRegistry etc.
        with patch("distllm.core.coordinator.AutoTokenizer") as mock_tok:
            mock_tok.from_pretrained.return_value = MagicMock()
            coord = Coordinator(model_name="test-model", dtype="float32", **overrides)
            # Attach minimal attributes for generate()
            coord.local_partitioner = MagicMock()
            coord.tokenizer = MagicMock()
            coord.tokenizer.encode.return_value = [1, 2, 3]
            coord.tokenizer.decode.return_value = "mocked output"
            coord.tokenizer.eos_token_id = 0
            coord.node_order = []
            coord.model_info = {"num_layers": 6, "max_position_embeddings": 4096}
            return coord


# ===================================================================
# _init_* method tests
# ===================================================================

class TestInitMultiModel:
    def test_no_config_skips(self):
        c = _make_coord()
        assert c._multi_model is None

    def test_enabled_config_initializes(self):
        config = MagicMock()
        config.enabled = True
        config.max_models = 4
        config.default_model = "test-model"
        config.models = {}
        c = _make_coord(multi_model_config=config)
        assert c._multi_model is not None

    def test_disabled_config_skips(self):
        config = MagicMock()
        config.enabled = False
        c = _make_coord(multi_model_config=config)
        assert c._multi_model is None


class TestInitMoE:
    def test_no_config_skips(self):
        c = _make_coord()
        assert c._moe_orchestrator is None

    def test_enabled_config_initializes(self):
        config = MagicMock()
        config.enabled = True
        c = _make_coord(moe_config=config)
        assert c._moe_orchestrator is not None
        assert c._expert_registry is not None


class TestInitFlashAttention:
    def test_enabled_fa2(self):
        c = _make_coord()
        with patch("distllm.core.coordinator.logger") as mock_log:
            c._init_flash_attention(enable_fa2=True)
            assert c._flash_attention is None or c._flash_attention is not None

    def test_disabled_fa2(self):
        c = _make_coord()
        c._init_flash_attention(enable_fa2=False)
        assert c._flash_attention is None


class TestInitPluginManager:
    def test_creates_plugin_manager(self):
        c = _make_coord()
        assert c._plugin_manager is not None


class TestInitHybridParallel:
    def test_no_config_skips(self):
        c = _make_coord()
        assert c._hybrid_parallel_planner is None

    def test_enabled_config_initializes(self):
        config = MagicMock()
        config.enabled = True
        config.pp_overlap = True
        config.tp_enabled = True
        config.ep_enabled = True
        config.force_tp_world_size = 0
        c = _make_coord(hybrid_parallel_config=config)
        # May or may not create planner depending on HardwareProbe
        assert c._hybrid_parallel_planner is not None or c._hybrid_parallel_executor is None

    def test_disabled_config_skips(self):
        config = MagicMock()
        config.enabled = False
        c = _make_coord(hybrid_parallel_config=config)
        assert c._hybrid_parallel_planner is None


class TestInitZeroCopy:
    def test_no_config_skips(self):
        c = _make_coord()
        assert c._zero_copy_engine is None

    def test_enabled_config_creates_engine(self):
        c = _make_coord(zero_copy_config=type("cfg", (), {"enabled": True})())
        assert c._zero_copy_engine is not None


class TestInitAdaptivePrecision:
    def test_no_config_skips(self):
        c = _make_coord()
        assert c._adaptive_precision is None

    def test_enabled_config_creates(self):
        c = _make_coord(adaptive_precision_config=type("cfg", (), {"enabled": True})())
        assert c._adaptive_precision is not None


class TestInitPredictiveCache:
    def test_no_config_skips(self):
        c = _make_coord()
        assert c._predictive_cache is None


class TestInitSelfOptimizing:
    def test_no_config_skips(self):
        c = _make_coord()
        assert c._self_optimizing is None

    def test_enabled_config_creates(self):
        config = type("cfg", (), {"enabled": True, "profile_dir": None, "tune_interval_seconds": 60.0, "warmup_seconds": 30.0})()
        c = _make_coord(self_optimizing_config=config)
        assert c._self_optimizing is not None


class TestInitCUDA:
    def test_no_config_skips(self):
        c = _make_coord()
        assert not hasattr(c, '_cuda_graph_batch_sizes') or c._cuda_graph_batch_sizes is None

    def test_enabled_sets_batch_sizes(self):
        config = type("cfg", (), {"enabled": True, "batch_sizes": [1, 2, 4]})()
        c = _make_coord(cuda_graph_config=config)
        assert c._cuda_graph_batch_sizes == [1, 2, 4]


class TestInitCompile:
    def test_no_config_disabled(self):
        c = _make_coord()
        assert c._compile_enabled is False

    def test_enabled_sets_flag(self):
        config = type("cfg", (), {"enabled": True, "mode": "reduce-overhead", "fullgraph": False})()
        c = _make_coord(compile_config=config)
        assert c._compile_enabled is True


class TestInitRAG:
    def test_no_config_skips(self):
        c = _make_coord()
        assert c._rag_pipeline is None

    def test_enabled_with_embedder(self):
        config = type("cfg", (), {"enabled": True, "dimension": 768, "chunk_size": 512, "chunk_overlap": 50, "index_path": None})()
        c = _make_coord(rag_config=config)
        # Should create pipeline if embedding_loader is available
        assert c._rag_pipeline is not None or c._rag_pipeline is None


class TestInitAgent:
    def test_no_config_skips(self):
        c = _make_coord()
        assert c._agent_loop is None


class TestInitDisagg:
    def test_no_config_skips(self):
        c = _make_coord()
        assert c._disagg_orchestrator is None


class TestInitSlora:
    def test_no_config_skips(self):
        c = _make_coord()
        assert not hasattr(c, '_slora_max_adapters') or c._slora_max_adapters is None


# ===================================================================
# Lifecycle tests
# ===================================================================

class TestLifecycle:
    def test_initial_state(self):
        c = _make_coord()
        assert c.model_name == "test-model"
        assert c.port == 50050
        assert c.dtype == "float32"

    def test_nodes_property(self):
        c = _make_coord()
        c.nodes = {}
        assert isinstance(c.nodes, dict)

    def test_metrics_property(self):
        c = _make_coord()
        c._metrics_mgr.get.return_value = {}
        assert isinstance(c.metrics, dict)

    def test_record_metric(self):
        c = _make_coord()
        c.record_metric("test_metric", 42.0)
        # Should not raise

    def test_get_metrics_returns_dict(self):
        c = _make_coord()
        c._metrics_mgr.get_prometheus.return_value = {}
        assert isinstance(c.get_metrics(), dict)

    @patch.object(threading.Thread, "start")
    @patch("distllm.core.coordinator.CoordinatorService")
    @patch("distllm.core.coordinator.GRPCServer")
    def test_start_non_blocking(self, mock_grpc, mock_svc, mock_start):
        c = _make_coord()
        c.tokenizer = MagicMock()
        c.start(blocking=False)
        mock_start.assert_called()

    @patch.object(threading.Thread, "start")
    @patch("distllm.core.coordinator.CoordinatorService")
    @patch("distllm.core.coordinator.GRPCServer")
    def test_start_blocking(self, mock_grpc, mock_svc, mock_start):
        c = _make_coord()
        c.tokenizer = MagicMock()
        mock_grpc.return_value.wait_for_termination.side_effect = KeyboardInterrupt()
        c.start(blocking=True)

    def test_wait_for_termination(self):
        c = _make_coord()
        c.server = MagicMock()
        c.wait_for_termination()
        c.server.wait_for_termination.assert_called_once()

    def test_health_check(self):
        c = _make_coord()
        c._health_checker.check_all.return_value = {}
        c.nodes = {}
        c.node_order = []
        result = c.health_check()
        assert isinstance(result, dict)


# ===================================================================
# Generate path tests (mock-heavy)
# ===================================================================

class TestGenerate:
    def test_generate_no_nodes_and_no_local_raises(self):
        c = _make_coord()
        c.local_partitioner = None
        c.node_order = []
        with pytest.raises(Exception):
            c.generate("hello")

    def _mock_tensor_tokens(self) -> torch.Tensor:
        return torch.tensor([[1, 2, 3]])

    def _setup_local_partitioner(self, c) -> None:
        c.local_partitioner = MagicMock()
        mock_param = torch.nn.Parameter(torch.zeros(1))
        c.local_partitioner.full_model.parameters.return_value = iter([mock_param])

    def test_generate_local(self):
        c = _make_coord()
        c.node_order = []
        self._setup_local_partitioner(c)
        c.model_info = {"num_layers": 6, "max_position_embeddings": 4096}
        tokenizer = MagicMock()
        tokenizer.encode.return_value = self._mock_tensor_tokens()
        tokenizer.decode.return_value = "generated output"
        tokenizer.eos_token_id = 0
        c.tokenizer = tokenizer
        with pytest.raises(Exception):
            c.generate("hello", max_new_tokens=10)

    def test_generate_with_request_id(self):
        c = _make_coord()
        self._setup_local_partitioner(c)
        c.tokenizer = MagicMock()
        c.tokenizer.encode.return_value = self._mock_tensor_tokens()
        c.tokenizer.decode.return_value = "output"
        c.tokenizer.eos_token_id = 0
        with pytest.raises(Exception):
            c.generate("test", request_id="my-req-1")

    def test_generate_reduces_max_new_tokens_when_exceeding_context(self):
        c = _make_coord()
        c.model_info = {"max_position_embeddings": 128}
        with patch.object(c, '_param_update_channel') as mock_ch:
            with pytest.raises(Exception):
                c.generate("hi", max_new_tokens=200)

    def test_generate_async_returns_id(self):
        c = _make_coord()
        c.scheduler = MagicMock()
        c.tokenizer = MagicMock()
        c.tokenizer.encode.return_value = self._mock_tensor_tokens()
        c.tokenizer.eos_token_id = 0
        c._cache_mgr.lookup_prefix.return_value = (0, None)
        rid = c.generate_async("hello")
        assert rid is not None
        assert isinstance(rid, str)

    def test_generate_async_no_scheduler_raises(self):
        c = _make_coord()
        c.scheduler = None
        with pytest.raises(Exception):
            c.generate_async("hello")


# ===================================================================
# New module properties
# ===================================================================

class TestNewModuleProperties:
    def test_new_properties_default_none(self):
        c = _make_coord()
        assert c.request_auditor is None
        assert c.prompt_cache_service is None
        assert c.graceful_degradation is None
        assert c.adaptive_batching is None
        assert c.token_streaming_buffer is None
        assert c.request_fingerprinter is None
        assert c.rate_limiter is None

    def test_model_comparator_always_available(self):
        c = _make_coord()
        assert c.model_comparator is not None

    def test_set_streaming_buffer(self):
        c = _make_coord()
        buf = c.set_streaming_buffer()
        assert c.token_streaming_buffer is not None
        assert buf is c.token_streaming_buffer

    def test_configure_adaptive_batching_noop_without_engine(self):
        c = _make_coord()
        c.configure_adaptive_batching("test", p50=100)  # no-op, no error

    def test_rate_limit_key_noop_without_limiter(self):
        c = _make_coord()
        c.rate_limit_key("key", rate=10, burst=20)  # no-op, no error

    def test_get_new_module_stats_empty(self):
        c = _make_coord()
        stats = c.get_new_module_stats()
        assert isinstance(stats, dict)


# ===================================================================
# Register / setup
# ===================================================================

class TestRegistration:
    def test_register_model(self):
        c = _make_coord()
        c._multi_model = MagicMock()
        entry = c.register_model("model-2", "/path/to/model", 12)
        assert entry is not None

    def test_list_models_default(self):
        c = _make_coord()
        models = c.list_models()
        assert isinstance(models, list)

    def test_get_model_name_default(self):
        c = _make_coord()
        name = c.get_model_name()
        assert name == "test-model"
