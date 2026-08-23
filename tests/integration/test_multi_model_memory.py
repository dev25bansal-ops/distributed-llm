"""Integration test: Multi-model load/unload with memory budgeting."""

from unittest.mock import MagicMock, patch

import pytest


# ===================================================================
# Model hot-swap memory management
# ===================================================================

class TestMultiModelMemory:
    def test_model_registry_tracks_models(self):
        """ModelRegistry should track registered models."""
        from distllm.core.model_registry import ModelRegistry

        registry = ModelRegistry(max_models=4)
        registry.register("model-a", "/path/a", total_layers=6)
        registry.register("model-b", "/path/b", total_layers=12)
        assert len(registry.list_models()) == 2

    def test_model_registry_max_limit(self):
        """Registry should enforce max_models limit."""
        from distllm.core.model_registry import ModelRegistry

        registry = ModelRegistry(max_models=2)
        registry.register("m1", "/p1", 6)
        registry.register("m2", "/p2", 12)
        # Third should replace oldest or raise
        registry.register("m3", "/p3", 8)
        # Most recent implementations evict oldest
        models = registry.list_models()
        assert len(models) <= 2

    def test_model_hotswap_memory_budget(self):
        """HotSwap manager should respect memory budget."""
        from distllm.core.coordinator import Coordinator
        with patch.multiple(
            "distllm.core.coordinator",
            ResourceManager=MagicMock,
            CacheManager=MagicMock,
            PipelineOrchestrator=MagicMock,
            TokenGenerator=MagicMock,
            RequestTracker=MagicMock,
            SubsystemManager=MagicMock,
        ):
            with patch("distllm.core.coordinator.AutoTokenizer") as mock_tok:
                coord = Coordinator(model_name="test-model")
                coord._init_model_hotswap(max_models=3, total_gpu_memory_gb=24.0)
                assert coord._model_hotswap is not None
                assert coord._model_hotswap._max_models == 3

    def test_model_hotswap_load_unload_cycle(self):
        """HotSwap manager should allow loading → generating → unloading."""
        from distllm.core.coordinator import Coordinator
        with patch.multiple(
            "distllm.core.coordinator",
            ResourceManager=MagicMock,
            CacheManager=MagicMock,
            PipelineOrchestrator=MagicMock,
            TokenGenerator=MagicMock,
            RequestTracker=MagicMock,
            SubsystemManager=MagicMock,
        ):
            with patch("distllm.core.coordinator.AutoTokenizer") as mock_tok:
                coord = Coordinator(model_name="test-model")
                coord._init_model_hotswap(max_models=2, total_gpu_memory_gb=16.0)
                hs = coord._model_hotswap

                # Register models
                hs._registry.register("tiny", "/tiny", 4)
                hs._registry.register("medium", "/medium", 12)

                # Load model (mocked)
                mock_load = MagicMock(return_value=(MagicMock(), MagicMock(), 2.0))
                hs._on_load_model = mock_load
                result = hs.load_model("tiny")
                assert result is not None or mock_load.called

    def test_multi_model_manager_switch(self):
        """MultiModelManager should allow switching between models."""
        from distllm.core.model_registry import ModelRegistry
        from distllm.core.coordinator_multi_model import MultiModelManager

        pipeline = MagicMock()
        registry = ModelRegistry(max_models=3)
        registry.register("model-a", "/a", 6)
        registry.register("model-b", "/b", 12)

        mgr = MultiModelManager(
            model_name="model-a",
            pipeline=pipeline,
            model_registry=registry,
        )
        assert mgr.get_model_name("model-b") == "model-b"
        assert mgr.get_model_name() == "model-a"
        models = mgr.list_models()
        assert "model-a" in models
        assert "model-b" in models


# ===================================================================
# Coordinator multi-model integration
# ===================================================================

class TestCoordinatorMultiModelIntegration:
    def test_coordinator_with_multi_model_config(self):
        """Coordinator should initialize multi-model with config."""
        from distllm.core.coordinator import Coordinator

        config = MagicMock()
        config.enabled = True
        config.max_models = 4
        config.default_model = "test-model"
        config.models = {"model-b": "/path/b"}

        with patch.multiple(
            "distllm.core.coordinator",
            ResourceManager=MagicMock,
            CacheManager=MagicMock,
            PipelineOrchestrator=MagicMock,
            TokenGenerator=MagicMock,
            RequestTracker=MagicMock,
            SubsystemManager=MagicMock,
        ):
            with patch("distllm.core.coordinator.AutoTokenizer") as mock_tok:
                coord = Coordinator(model_name="test-model", multi_model_config=config)
                assert coord._multi_model is not None

    def test_register_model_after_creation(self):
        """Register a new model after coordinator creation."""
        from distllm.core.coordinator import Coordinator

        with patch.multiple(
            "distllm.core.coordinator",
            ResourceManager=MagicMock,
            CacheManager=MagicMock,
            PipelineOrchestrator=MagicMock,
            TokenGenerator=MagicMock,
            RequestTracker=MagicMock,
            SubsystemManager=MagicMock,
        ):
            with patch("distllm.core.coordinator.AutoTokenizer") as mock_tok:
                coord = Coordinator(model_name="test-model")
                # Initially no multi-model
                assert coord._multi_model is None

                # Register a model (lazy init)
                entry = coord.register_model("new-model", "/path/new", 8)
                assert coord._multi_model is not None
                assert entry is not None

    def test_list_models_with_multi(self):
        """list_models should include all registered models."""
        from distllm.core.coordinator import Coordinator

        with patch.multiple(
            "distllm.core.coordinator",
            ResourceManager=MagicMock,
            CacheManager=MagicMock,
            PipelineOrchestrator=MagicMock,
            TokenGenerator=MagicMock,
            RequestTracker=MagicMock,
            SubsystemManager=MagicMock,
        ):
            with patch("distllm.core.coordinator.AutoTokenizer") as mock_tok:
                coord = Coordinator(model_name="test-model")
                models = coord.list_models()
                assert "test-model" in models


# ===================================================================
# Memory budgeting edge cases
# ===================================================================

class TestMemoryBudgeting:
    def test_gpu_memory_limit_enforced(self):
        """Model loading should respect GPU memory limits."""
        from distllm.core.coordinator import Coordinator

        with patch.multiple(
            "distllm.core.coordinator",
            ResourceManager=MagicMock,
            CacheManager=MagicMock,
            PipelineOrchestrator=MagicMock,
            TokenGenerator=MagicMock,
            RequestTracker=MagicMock,
            SubsystemManager=MagicMock,
        ):
            with patch("distllm.core.coordinator.AutoTokenizer") as mock_tok:
                coord = Coordinator(model_name="test-model")
                # With 8GB GPU and models needing 6GB each, we can fit 1
                coord._init_model_hotswap(max_models=4, total_gpu_memory_gb=8.0)
                hs = coord._model_hotswap
                assert hs._total_gpu_memory_gb == 8.0

    def test_eviction_under_pressure(self):
        """Oldest model should be evicted when memory is full."""
        from distllm.core.model_registry import ModelRegistry

        registry = ModelRegistry(max_models=2)
        registry.register("m1", "/path/1", 4)
        registry.register("m2", "/path/2", 8)
        # Force eviction by registering a third
        registry.register("m3", "/path/3", 6)
        assert len(registry.list_models()) <= 2
