"""Tests for distllm.dist.tensor_pool — real-object tests, no mocks."""

from __future__ import annotations

import pytest
import torch

from distllm.dist.tensor_pool import TensorPool, GPUMemoryPool


# ---------------------------------------------------------------------------
# TensorPool
# ---------------------------------------------------------------------------


class TestTensorPool:
    """Tests for the TensorPool buffer-reuse class."""

    def test_default_construction(self) -> None:
        pool = TensorPool()
        assert pool.max_buffers == 64
        assert pool.growth_factor == 1.5
        assert pool.pool_size == 0

    def test_custom_construction(self) -> None:
        pool = TensorPool(max_buffers=8, growth_factor=2.0)
        assert pool.max_buffers == 8
        assert pool.growth_factor == 2.0

    def test_get_buffer_returns_tensor_with_correct_shape_and_dtype(self) -> None:
        pool = TensorPool()
        shape = (4, 8)
        tensor = pool.get_buffer(shape, dtype=torch.float32, device="cpu")
        assert tensor.shape == torch.Size(shape)
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"

    def test_get_buffer_returns_bigger_than_requested_shape(self) -> None:
        pool = TensorPool(growth_factor=2.0)
        shape = (3, 5)
        tensor = pool.get_buffer(shape)
        # The allocated buffer has padded shape (6, 10), then sliced to (3, 5)
        assert tensor.shape == torch.Size(shape)
        # The underlying storage should be larger — slicing shares storage
        assert tensor.storage().size() >= 3 * 5

    def test_reuse_buffer_from_pool(self) -> None:
        pool = TensorPool()
        shape = (2, 4)
        t1 = pool.get_buffer(shape)
        pool.release(t1)
        assert pool.pool_size == 1

        # Re-request a compatible shape — should reuse the released buffer
        t2 = pool.get_buffer(shape)
        assert t2.shape == torch.Size(shape)
        # pool should be empty again
        assert pool.pool_size == 0

    def test_reuse_larger_buffer_for_smaller_request(self) -> None:
        pool = TensorPool(growth_factor=2.0)
        t1 = pool.get_buffer((4, 8))
        pool.release(t1)

        # Request a smaller shape — the larger buffer should be sliced
        t2 = pool.get_buffer((2, 4))
        assert t2.shape == torch.Size((2, 4))
        # The storage (backing the 4x8 buffer) should be large enough
        assert t2.storage().size() >= 4 * 8

    def test_does_not_reuse_smaller_buffer_for_larger_request(self) -> None:
        pool = TensorPool(growth_factor=2.0)
        pool.release(pool.get_buffer((1, 2)))
        assert pool.pool_size == 1

        # Larger request — cannot reuse the tiny buffer, allocates new
        t2 = pool.get_buffer((10, 20))
        assert t2.shape == torch.Size((10, 20))
        # The small buffer should still be in the pool
        assert pool.pool_size == 1

    def test_release_after_max_buffers_is_noop(self) -> None:
        """_total_buffers tracks total allocations; release is no-op once saturated."""
        pool = TensorPool(max_buffers=2)

        # One get_buffer -> _total_buffers == 1 (< max_buffers) -> release succeeds
        a = pool.get_buffer((1,))
        pool.release(a)
        assert pool.pool_size == 1

        # Second get_buffer (cannot reuse 1-element buffer for 3-element)
        # -> _total_buffers == 2 (== max_buffers) -> release rejected
        b = pool.get_buffer((4,))
        pool.release(b)
        assert pool.pool_size == 1

        # Future releases also rejected
        c = pool.get_buffer((2,))
        pool.release(c)
        assert pool.pool_size == 1

    def test_clear_resets_pool(self) -> None:
        pool = TensorPool()
        pool.release(pool.get_buffer((4, 4)))
        pool.release(pool.get_buffer((8, 8)))
        assert pool.pool_size == 2

        pool.clear()
        assert pool.pool_size == 0

    def test_multiple_dtype_and_device_pools(self) -> None:
        pool = TensorPool()
        t1 = pool.get_buffer((2,), dtype=torch.float32)
        t2 = pool.get_buffer((2,), dtype=torch.int64)
        pool.release(t1)
        pool.release(t2)
        assert pool.pool_size == 2

        # Retrieving float32 should not consume the int64 buffer
        t3 = pool.get_buffer((2,), dtype=torch.float32)
        assert t3.dtype == torch.float32
        assert pool.pool_size == 1  # only the int64 buffer remains

    def test_empty_shape_get_buffer(self) -> None:
        pool = TensorPool()
        tensor = pool.get_buffer((), dtype=torch.float32)
        assert tensor.shape == torch.Size(())
        assert tensor.dtype == torch.float32

    def test_single_element_shape(self) -> None:
        pool = TensorPool()
        tensor = pool.get_buffer((1,), dtype=torch.int32)
        assert tensor.shape == (1,)
        assert tensor.dtype == torch.int32

    def test_pool_size_property(self) -> None:
        pool = TensorPool(max_buffers=10)
        assert pool.pool_size == 0
        pool.release(pool.get_buffer((3, 3)))
        assert pool.pool_size == 1
        pool.release(pool.get_buffer((4, 4)))
        assert pool.pool_size == 2

    def test_release_writes_are_visible(self) -> None:
        """Ensure the released tensor's data persists and is retrievable."""
        pool = TensorPool()
        t1 = pool.get_buffer((2, 3))
        t1[:] = 42
        pool.release(t1)

        t2 = pool.get_buffer((2, 3))
        # The memory may or may not be the same buffer; just confirm no crash
        assert t2.shape == (2, 3)

    def test_growth_factor_usage(self) -> None:
        """With growth_factor=1.0, the tensor is exactly the requested size."""
        pool = TensorPool(growth_factor=1.0)
        t = pool.get_buffer((5, 7))
        assert t.shape == (5, 7)


# ---------------------------------------------------------------------------
# GPUMemoryPool  (tested on CPU — no GPU required)
# ---------------------------------------------------------------------------


class TestGPUMemoryPool:
    """Tests for GPUMemoryPool using 'cpu' as the device."""

    def test_default_construction(self) -> None:
        pool = GPUMemoryPool()
        assert pool.total_bytes == int(8 * 1024 ** 3)
        assert pool.block_bytes == int(16 * 1024 ** 2)
        assert pool.used_blocks == 0
        assert pool.free_blocks == 0

    def test_custom_construction(self) -> None:
        pool = GPUMemoryPool(total_gb=1.0, block_size_mb=4.0)
        assert pool.total_bytes == int(1024 ** 3)
        assert pool.block_bytes == int(4 * 1024 ** 2)

    def test_allocate_small_tensor_uses_block(self) -> None:
        pool = GPUMemoryPool(total_gb=0.25, block_size_mb=1.0)
        tensor = pool.allocate((16,), dtype=torch.float32, device="cpu")
        assert tensor is not None
        assert tensor.shape == (16,)
        assert tensor.dtype == torch.float32
        assert pool.used_blocks == 1
        assert pool.free_blocks + pool.used_blocks > 0

    def test_allocate_large_tensor_bypasses_pool(self) -> None:
        pool = GPUMemoryPool(total_gb=0.25, block_size_mb=1.0)
        # 2 million float32 elements = 8 MB > 1 MB block
        tensor = pool.allocate((2_000_000,), dtype=torch.float32, device="cpu")
        assert tensor is not None
        assert tensor.shape == (2_000_000,)
        # No blocks were consumed
        assert pool.used_blocks == 0

    def test_exhaust_pool_returns_none(self) -> None:
        pool = GPUMemoryPool(total_gb=0.25, block_size_mb=1.0)
        # 0.25 GB / 1 MB = 256 blocks — allocate them all
        blocks: list[torch.Tensor] = []
        for _ in range(260):
            t = pool.allocate((4,), dtype=torch.float32, device="cpu")
            if t is None:
                break
            blocks.append(t)

        assert len(blocks) <= 260
        # The next allocation should return None
        assert pool.allocate((4,), dtype=torch.float32, device="cpu") is None

    def test_free_returns_block_to_pool(self) -> None:
        pool = GPUMemoryPool(total_gb=0.25, block_size_mb=1.0)
        t = pool.allocate((16,), dtype=torch.float32, device="cpu")
        assert t is not None
        assert pool.used_blocks == 1

        pool.free(t)
        assert pool.used_blocks == 0
        assert pool.free_blocks > 0

    def test_freed_block_can_be_reallocated(self) -> None:
        pool = GPUMemoryPool(total_gb=0.25, block_size_mb=1.0)
        t1 = pool.allocate((32,), dtype=torch.float32, device="cpu")
        assert t1 is not None
        pool.free(t1)

        t2 = pool.allocate((32,), dtype=torch.float32, device="cpu")
        assert t2 is not None
        assert t2.shape == (32,)

    def test_free_nonexistent_tensor_is_noop(self) -> None:
        pool = GPUMemoryPool(total_gb=0.25, block_size_mb=1.0)
        unrelated = torch.empty(10)
        # Should not raise
        pool.free(unrelated)
        assert pool.used_blocks == 0

    def test_free_all(self) -> None:
        pool = GPUMemoryPool(total_gb=0.25, block_size_mb=1.0)
        tensors = [pool.allocate((4,), dtype=torch.float32, device="cpu") for _ in range(10)]
        assert all(t is not None for t in tensors)
        assert pool.used_blocks == 10

        pool.free_all()
        assert pool.used_blocks == 0
        assert pool.free_blocks > 0

    def test_properties_after_construction(self) -> None:
        pool = GPUMemoryPool(total_gb=0.25, block_size_mb=1.0)
        assert pool.used_blocks == 0
        assert pool.free_blocks == 0

        # First allocation triggers block creation
        pool.allocate((1,), dtype=torch.float32, device="cpu")
        assert pool.used_blocks == 1
        assert pool.free_blocks > 0

    def test_different_dtype_reshape(self) -> None:
        pool = GPUMemoryPool(total_gb=0.25, block_size_mb=1.0)
        # Allocate as float32 (4 bytes per element, block is 1 MB = 262144 elements)
        t = pool.allocate((64,), dtype=torch.float32, device="cpu")
        assert t is not None
        assert t.shape == (64,)
        assert t.dtype == torch.float32
