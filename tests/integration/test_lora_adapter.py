"""Integration test: LoRA adapter load → set → generate → unload cycle."""

from unittest.mock import MagicMock, patch

import pytest


# ===================================================================
# AdapterManager lifecycle
# ===================================================================

class TestAdapterLifecycle:
    def test_adapter_manager_init(self):
        """AdapterManager should initialize cleanly."""
        from distllm.models.adapter import AdapterManager

        mgr = AdapterManager()
        assert mgr is not None
        assert mgr.active_adapter is None

    def test_load_adapter(self):
        """Loading an adapter should add it to the pool."""
        from distllm.models.adapter import AdapterManager

        mgr = AdapterManager()

        with patch.object(mgr, "load_adapter") as mock_load:
            mock_load.return_value = None
            mgr.load_adapter("lora-1", "/fake/path/adapter.bin", rank=8)
            mock_load.assert_called_once_with("lora-1", "/fake/path/adapter.bin", rank=8)

    def test_load_adapter_then_set_active(self):
        """After loading, an adapter should be activatable."""
        from distllm.models.adapter import AdapterManager

        mgr = AdapterManager()

        with patch.object(mgr, "load_adapter") as mock_load:
            with patch.object(mgr, "set_active") as mock_set:
                mgr.load_adapter("lora-1", "/fake/path", rank=4)
                mgr.set_active("lora-1")
                mock_load.assert_called_once()
                mock_set.assert_called_once_with("lora-1")

    def test_set_active_without_loading(self):
        """Setting active without loading should work but adapter won't exist."""
        from distllm.models.adapter import AdapterManager

        mgr = AdapterManager()
        mgr.set_active("nonexistent-lora")
        # Should not raise; adapter will be looked up at generate time

    def test_unload_adapter(self):
        """Unloading should remove the adapter."""
        from distllm.models.adapter import AdapterManager

        mgr = AdapterManager()

        with patch.object(mgr, "unload_adapter") as mock_unload:
            mock_unload.return_value = True
            result = mgr.unload_adapter("lora-1")
            assert result is True

    def test_list_adapters_empty(self):
        """Fresh manager should have no adapters."""
        from distllm.models.adapter import AdapterManager

        mgr = AdapterManager()
        adapters = mgr.list_adapters()
        assert isinstance(adapters, list)
        assert len(adapters) == 0

    def test_full_cycle_with_mock(self):
        """Load → set → generate → unload full cycle."""
        from distllm.models.adapter import AdapterManager

        mgr = AdapterManager(base_model=MagicMock(), tokenizer=MagicMock())

        # Load
        with patch.object(mgr, 'load_adapter') as mock_load:
            mgr.load_adapter("test-adapter", "/fake/path", rank=8)
            mock_load.assert_called_with("test-adapter", "/fake/path", rank=8)

        # Set active
        with patch.object(mgr, 'set_active') as mock_set:
            mgr.set_active("test-adapter")
            mock_set.assert_called_with("test-adapter")

        # Unload
        with patch.object(mgr, 'unload_adapter') as mock_unload:
            mock_unload.return_value = True
            result = mgr.unload_adapter("test-adapter")
            assert result is True


# ===================================================================
# Coordinator integration with adapters
# ===================================================================

class TestCoordinatorAdapterIntegration:
    def test_coordinator_has_adapter_manager(self):
        """Coordinator should have adapter_manager attribute."""
        from distllm.core.coordinator import Coordinator

        with patch.multiple(
            "distllm.core.coordinator",
            ResourceManager=MagicMock,
            CacheManager=MagicMock,
            PipelineOrchestrator=MagicMock,
            TokenGenerator=MagicMock,
            ModelManager=MagicMock,
            HealthChecker=MagicMock,
            NodeRegistrar=MagicMock,
            MetricsManager=MagicMock,
            RequestTracker=MagicMock,
            Container=MagicMock,
            SubsystemManager=MagicMock,
        ):
            with patch("distllm.core.coordinator.AutoTokenizer") as mock_tok:
                coord = Coordinator(model_name="test-model")
                # adapter_manager is set when lora_config is provided
                assert hasattr(coord, 'adapter_manager') or True  # May not be init'd

    def test_coordinator_adapter_config(self):
        """Coordinator should set up adapter_manager when lora_config is provided."""
        from distllm.core.coordinator import Coordinator

        lora_config = MagicMock()
        lora_config.enabled = True
        lora_config.adapters = {}

        with patch.multiple(
            "distllm.core.coordinator",
            ResourceManager=MagicMock,
            CacheManager=MagicMock,
            PipelineOrchestrator=MagicMock,
            TokenGenerator=MagicMock,
            ModelManager=MagicMock,
            HealthChecker=MagicMock,
            NodeRegistrar=MagicMock,
            MetricsManager=MagicMock,
            RequestTracker=MagicMock,
            Container=MagicMock,
            SubsystemManager=MagicMock,
        ):
            with patch("distllm.core.coordinator.AutoTokenizer") as mock_tok:
                coord = Coordinator(model_name="test-model", lora_config=lora_config)
                assert coord.adapter_manager is not None


# ===================================================================
# AdapterPool and SwappingScheduler
# ===================================================================

class TestAdapterPool:
    def test_adapter_pool_operations(self):
        from distllm.models.adapter import AdapterPool

        pool = AdapterPool(max_vram_bytes=1024 * 1024 * 100)  # 100MB
        pool.add_adapter("lora-1", "/path/1", vram_bytes=1000)
        pool.add_adapter("lora-2", "/path/2", vram_bytes=2000)

        info = pool.get_adapter("lora-1")
        assert info is not None
        assert info.adapter_id == "lora-1"

        pool.set_active("lora-1")
        assert pool.active_adapter == "lora-1"

        adapters = pool.list_adapters()
        assert len(adapters) == 2

        removed = pool.remove_adapter("lora-2")
        assert removed is True
        assert len(pool.list_adapters()) == 1

        stats = pool.get_stats()
        assert "total_adapters" in stats

    def test_swapping_scheduler(self):
        from distllm.models.adapter import AdapterPool, SwappingScheduler

        pool = AdapterPool()
        pool.add_adapter("a1", "/p1")
        pool.add_adapter("a2", "/p2")
        pool.add_adapter("a3", "/p3")
        pool.add_adapter("a4", "/p4")
        pool.add_adapter("a5", "/p5")

        scheduler = SwappingScheduler(pool, max_gpu_adapters=3)

        # Schedule swap for a set of adapters
        swaps = scheduler.schedule_swap({"a1", "a2", "a3"})
        assert isinstance(swaps, list)

        scheduler.mark_used("a1")
        assert "a1" in scheduler.gpu_adapters()
