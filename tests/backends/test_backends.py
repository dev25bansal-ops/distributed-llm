"""Tests for backends/ modules."""
from __future__ import annotations

from unittest.mock import MagicMock, patch, AsyncMock
import pytest


class TestBackendProtocol:
    """Tests for backends/protocol.py."""

    def test_protocol_has_required_methods(self):
        from distllm.backends.protocol import BackendAdapter

        methods = ["load_model", "forward", "shutdown", "generate", "health_check"]
        for m in methods:
            assert hasattr(BackendAdapter, m), f"Missing method: {m}"

    def test_protocol_type_annotations(self):
        from distllm.backends.protocol import BackendAdapter

        import inspect
        sig = inspect.signature(BackendAdapter.forward)
        assert len(sig.parameters) > 0


class TestBackendRegistry:
    """Tests for backends/registry.py."""

    def test_registry_init(self):
        from distllm.backends.registry import BackendRegistry

        reg = BackendRegistry()
        assert reg is not None

    def test_register_and_list(self):
        from distllm.backends.registry import BackendRegistry
        from distllm.backends.protocol import BackendAdapter

        class TestBackend(BackendAdapter):
            def load_model(self):
                pass
            def forward(self, hidden_states=None, attention_mask=None,
                       position_ids=None, past_key_values=None, input_ids=None):
                return None, []
            def shutdown(self):
                pass
            @classmethod
            def display_name(cls):
                return "Test"
            @classmethod
            def is_available(cls):
                return True
            @classmethod
            def priority_for(cls, device_type):
                return 5

        reg = BackendRegistry()
        reg.register(TestBackend, name="test_backend")
        plugins = reg.list_backends()
        assert "test_backend" in [p.name for p in plugins]

    def test_get_backend(self):
        from distllm.backends.registry import BackendRegistry
        from distllm.backends.protocol import BackendAdapter

        class TestBackend(BackendAdapter):
            def load_model(self):
                pass
            def forward(self, hidden_states=None, attention_mask=None,
                       position_ids=None, past_key_values=None, input_ids=None):
                return None, []
            def shutdown(self):
                pass
            @classmethod
            def display_name(cls):
                return "Test"
            @classmethod
            def is_available(cls):
                return True
            @classmethod
            def priority_for(cls, device_type):
                return 5

        reg = BackendRegistry()
        reg.register(TestBackend, name="test")
        result = reg.get("test")
        assert result is TestBackend

    def test_get_nonexistent_backend(self):
        from distllm.backends.registry import BackendRegistry

        reg = BackendRegistry()
        if hasattr(reg, "get"):
            result = reg.get("nonexistent")
            assert result is None


class TestBackendConfig:
    """Tests for backends/config.py."""

    def test_config_init(self):
        from distllm.backends.config import BackendConfig

        cfg = BackendConfig()
        assert cfg is not None

    def test_config_defaults(self):
        from distllm.backends.config import BackendConfig

        cfg = BackendConfig()
        if hasattr(cfg, "backend_type"):
            assert cfg.backend_type in (None, "", "pytorch", "vllm", "llamacpp", "exllama", "onnx")


class TestPyTorchBackend:
    """Tests for backends/pytorch_backend.py."""

    def test_backend_class_exists(self):
        from distllm.backends.pytorch_backend import PyTorchNodeAdapter
        assert PyTorchNodeAdapter is not None

    @patch("distllm.backends.pytorch_backend.ModelPartitioner")
    def test_load_model(self, mock_partitioner):
        from distllm.backends.pytorch_backend import PyTorchNodeAdapter

        backend = PyTorchNodeAdapter(
            model_name="test-model",
            layer_start=0,
            layer_end=2,
            total_layers=4,
        )
        assert backend.load_model() is None  # succeeds without raising
        mock_partitioner.assert_called_once()
        assert backend.partitioner is mock_partitioner.return_value

    def test_unload(self):
        from distllm.backends.pytorch_backend import PyTorchNodeAdapter

        backend = PyTorchNodeAdapter(
            model_name="test-model",
            layer_start=0,
            layer_end=2,
            total_layers=4,
        )
        backend.shutdown()
        assert backend.partitioner is None


class TestVLLMBackend:
    """Tests for backends/vllm_backend.py."""

    def test_backend_class_exists(self):
        from distllm.backends.vllm_backend import VLLMNodeAdapter
        assert VLLMNodeAdapter is not None

    def test_backend_init(self):
        from distllm.backends.vllm_backend import VLLMNodeAdapter
        backend = VLLMNodeAdapter(model_name="test-org/test-model")
        assert backend is not None
        assert backend.state.value == "uninitialized"


class TestLlamaCppBackend:
    """Tests for backends/llamacpp_backend.py."""

    def test_backend_class_exists(self):
        from distllm.backends.llamacpp_backend import LlamacppNodeAdapter
        assert LlamacppNodeAdapter is not None

    def test_backend_init(self):
        from distllm.backends.llamacpp_backend import LlamacppNodeAdapter
        backend = LlamacppNodeAdapter(model_path="test-model")
        assert backend is not None


class TestExLlamaBackend:
    """Tests for backends/exllama_backend.py."""

    def test_backend_class_exists(self):
        from distllm.backends.exllama_backend import ExLlamaV2NodeAdapter
        assert ExLlamaV2NodeAdapter is not None

    def test_backend_init(self):
        from distllm.backends.exllama_backend import ExLlamaV2NodeAdapter
        backend = ExLlamaV2NodeAdapter(model_name="test-model")
        assert backend is not None


class TestOnnxBackend:
    """Tests for backends/onnx_backend.py."""

    def test_backend_class_exists(self):
        from distllm.backends.onnx_backend import ONNXNodeAdapter
        assert ONNXNodeAdapter is not None

    def test_backend_init(self):
        from distllm.backends.onnx_backend import ONNXNodeAdapter
        backend = ONNXNodeAdapter(model_name="test-model")
        assert backend is not None


class TestPagedAttention:
    """Tests for backends/paged_attention.py."""

    def test_paged_attention_class_exists(self):
        from distllm.backends.paged_attention import PagedAttentionManager
        assert PagedAttentionManager is not None

    def test_block_pool_init(self):
        from distllm.backends.paged_attention import PagedAttentionManager

        pam = PagedAttentionManager(
            num_blocks=100,
            block_size=16,
            num_layers=32,
            num_heads=32,
            head_dim=128,
        )
        assert pam is not None

    def test_allocate_and_free(self):
        from distllm.backends.paged_attention import PagedAttentionManager

        pam = PagedAttentionManager(
            num_blocks=10,
            block_size=16,
            num_layers=2,
            num_heads=2,
            head_dim=64,
        )
        block_ids = pam.allocate_sequence("seq-1", num_tokens=32)
        assert len(block_ids) == 2  # 32 tokens / 16 per block = 2 blocks
        assert pam.num_used_blocks == 2
        pam.free_sequence("seq-1")
        assert pam.num_used_blocks == 0

    def test_allocate_exhausts_pool(self):
        from distllm.backends.paged_attention import PagedAttentionManager

        pam = PagedAttentionManager(
            num_blocks=4,
            block_size=16,
            num_layers=2,
            num_heads=2,
            head_dim=64,
        )
        pam.allocate_sequence("seq-1", num_tokens=48)  # 3 blocks
        with pytest.raises(RuntimeError, match="Not enough"):
            pam.allocate_sequence("seq-2", num_tokens=32)  # 2 blocks, only 1 free

    def test_copy_on_write(self):
        from distllm.backends.paged_attention import PagedAttentionManager

        pam = PagedAttentionManager(
            num_blocks=10,
            block_size=16,
            num_layers=2,
            num_heads=2,
            head_dim=64,
        )
        pam.allocate_sequence("src", num_tokens=32)
        pam.copy_on_write("src", "dst")
        src_blocks = pam.get_block_table("src")
        dst_blocks = pam.get_block_table("dst")
        assert src_blocks == dst_blocks
        # Ref count should be 2 on shared blocks
        for bid in src_blocks:
            assert pam._blocks[bid].ref_count == 2

    def test_cpu_swap(self):
        from distllm.backends.paged_attention import PagedAttentionManager

        pam = PagedAttentionManager(
            num_blocks=10,
            block_size=16,
            num_layers=2,
            num_heads=2,
            head_dim=64,
        )
        pam.allocate_sequence("seq-1", num_tokens=32)
        swapped = pam.swap_blocks_to_cpu("seq-1")
        assert swapped == 2
        restored = pam.swap_blocks_to_gpu("seq-1")
        assert restored == 2
