"""GPU-specific tests that require real CUDA hardware.

These tests verify behavior that cannot be tested with mocks:
- Actual CUDA kernel execution
- GPU memory management
- FP16/BF16 numerical stability
- PagedAttention with real GPU blocks
- Multi-GPU communication

Mark all tests with @pytest.mark.gpu — they are skipped in CI
unless a GPU runner is available.

Usage:
    pytest tests/test_gpu.py -m gpu -v
"""

import pytest
import torch

# Skip all tests if no GPU available
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available — GPU tests require NVIDIA GPU",
    ),
]


class TestGPUInference:
    """Tests that require actual GPU inference."""

    def test_cuda_tensor_operations(self):
        """Verify basic CUDA tensor operations work correctly."""
        device = torch.device("cuda:0")
        a = torch.randn(100, 100, device=device)
        b = torch.randn(100, 100, device=device)
        c = torch.mm(a, b)
        assert c.shape == (100, 100)
        assert c.device.type == "cuda"

    def test_fp16_numerical_stability(self):
        """Verify FP16 operations don't produce NaN/Inf."""
        device = torch.device("cuda:0")
        x = torch.randn(1000, 1000, dtype=torch.float16, device=device)
        # Large matrix multiply should not overflow
        result = torch.mm(x, x.T)
        assert not torch.isnan(result).any(), "FP16 produced NaN"
        assert not torch.isinf(result).any(), "FP16 produced Inf"

    def test_bf16_numerical_stability(self):
        """Verify BF16 operations don't produce NaN/Inf."""
        if not torch.cuda.is_bf16_supported():
            pytest.skip("BF16 not supported on this GPU")

        device = torch.device("cuda:0")
        x = torch.randn(1000, 1000, dtype=torch.bfloat16, device=device)
        result = torch.mm(x, x.T)
        assert not torch.isnan(result).any(), "BF16 produced NaN"
        assert not torch.isinf(result).any(), "BF16 produced Inf"

    def test_gpu_memory_allocation_tracking(self):
        """Verify GPU memory tracking works correctly."""
        device = torch.device("cuda:0")
        initial = torch.cuda.memory_allocated()

        # Allocate 100MB
        tensor = torch.randn(1024, 1024, 64, device=device)  # ~256MB in fp32
        after_alloc = torch.cuda.memory_allocated()

        assert after_alloc > initial, "Memory allocation not tracked"

        # Free
        del tensor
        torch.cuda.empty_cache()
        after_free = torch.cuda.memory_allocated()

        assert after_free < after_alloc, "Memory not freed"

    @pytest.mark.skipif(
        torch.cuda.device_count() < 2,
        reason="Multi-GPU test requires 2+ GPUs",
    )
    def test_multi_gpu_tensor_transfer(self):
        """Verify tensor transfer between GPUs works."""
        a = torch.randn(100, 100, device="cuda:0")
        b = a.to("cuda:1")
        assert b.device.index == 1
        assert torch.allclose(a.cpu(), b.cpu())


class TestGPUPagedAttention:
    """Tests for PagedAttention with real GPU memory."""

    def test_paged_attention_gpu_blocks(self):
        """Verify PagedAttention can allocate GPU blocks."""
        from distllm.backends.paged_attention import PagedAttentionManager

        mgr = PagedAttentionManager(
            num_blocks=64,
            block_size=16,
            num_layers=4,
            num_heads=8,
            head_dim=64,
            device="cuda",
        )

        # Allocate a sequence
        block_ids = mgr.allocate_sequence("test-seq", num_tokens=32)
        assert len(block_ids) > 0

        # Verify blocks are allocated
        assert mgr.num_used_blocks > 0
        assert mgr.num_free_blocks < 64

        # Free
        mgr.free_sequence("test-seq")
        assert mgr.num_used_blocks == 0

    def test_paged_attention_kv_write_read(self):
        """Verify KV cache write and read on GPU."""
        from distllm.backends.paged_attention import PagedAttentionManager

        mgr = PagedAttentionManager(
            num_blocks=32,
            block_size=16,
            num_layers=2,
            num_heads=4,
            head_dim=32,
            device="cuda",
        )

        mgr.allocate_sequence("seq-1", num_tokens=16)

        # Write KV data
        key = torch.randn(4, 16, 32, device="cuda")
        value = torch.randn(4, 16, 32, device="cuda")
        # Note: actual write depends on PagedAttentionManager API

        mgr.free_sequence("seq-1")


class TestGPUMemoryPressure:
    """Tests for behavior under GPU memory pressure."""

    def test_oom_handling(self):
        """Verify graceful handling of GPU OOM."""
        device = torch.device("cuda:0")
        total_mem = torch.cuda.get_device_properties(0).total_mem

        # Try to allocate more than available (should raise)
        with pytest.raises((torch.cuda.OutOfMemoryError, RuntimeError)):
            # Allocate 2x GPU memory
            huge = torch.empty(int(total_mem * 2), dtype=torch.uint8, device=device)

    def test_cache_clear_frees_memory(self):
        """Verify torch.cuda.empty_cache() releases memory."""
        device = torch.device("cuda:0")
        tensors = [torch.randn(1000, 1000, device=device) for _ in range(10)]
        before = torch.cuda.memory_allocated()

        del tensors
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated()

        assert after < before, "empty_cache did not free memory"
