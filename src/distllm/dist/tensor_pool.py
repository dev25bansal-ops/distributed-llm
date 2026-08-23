"""Pre-allocated tensor buffer pool for batch processing.

Avoids repeated torch.tensor() allocations by maintaining a pool
of reusable buffers sized for common batch shapes.
"""

from __future__ import annotations

import torch
from collections import defaultdict


class TensorPool:
    def __init__(self, max_buffers: int = 64, growth_factor: float = 1.5):
        self.max_buffers = max_buffers
        self.growth_factor = growth_factor
        self._pools: dict[tuple[str, torch.dtype], list[tuple[tuple[int, ...], torch.Tensor]]] = defaultdict(list)
        self._total_buffers = 0

    def get_buffer(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.long,
        device: str = "cpu",
    ) -> torch.Tensor:
        pool_key = (device, dtype)
        pool = self._pools[pool_key]

        for i, (buf_shape, buf) in enumerate(pool):
            if len(buf_shape) == len(shape) and all(b >= s for b, s in zip(buf_shape, shape, strict=False)):
                pool.pop(i)
                return buf[tuple(slice(0, s) for s in shape)]

        padded_shape = tuple(max(1, int(s * self.growth_factor)) for s in shape)
        buf = torch.empty(padded_shape, dtype=dtype, device=device)
        self._total_buffers += 1
        return buf[tuple(slice(0, s) for s in shape)]

    def release(self, tensor: torch.Tensor) -> None:
        if self._total_buffers >= self.max_buffers:
            return

        device = str(tensor.device)
        dtype = tensor.dtype
        pool_key = (device, dtype)
        original_shape = tuple(tensor.shape)

        self._pools[pool_key].append((original_shape, tensor))

    def clear(self) -> None:
        self._pools.clear()
        self._total_buffers = 0

    @property
    def pool_size(self) -> int:
        return sum(len(p) for p in self._pools.values())


class GPUMemoryPool:
    def __init__(self, total_gb: float = 8.0, block_size_mb: float = 16.0):
        self.total_bytes = int(total_gb * 1024 * 1024 * 1024)
        self.block_bytes = int(block_size_mb * 1024 * 1024)
        self._blocks: list[torch.Tensor] = []
        self._free_blocks: list[torch.Tensor] = []
        self._allocated: dict[int, torch.Tensor] = {}
        self._device: str | None = None

    def _init_device(self, device: str) -> None:
        if self._device == device:
            return

        self._device = device
        num_blocks = max(1, self.total_bytes // self.block_bytes)

        for _ in range(num_blocks):
            block = torch.empty(self.block_bytes // 4, dtype=torch.float32, device=device)
            self._blocks.append(block)
            self._free_blocks.append(block)

    def allocate(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> torch.Tensor | None:
        required_bytes = 1
        for s in shape:
            required_bytes *= s
        required_bytes *= torch.tensor([], dtype=dtype).element_size()

        if required_bytes > self.block_bytes:
            return torch.empty(shape, dtype=dtype, device=device)

        self._init_device(device)

        if not self._free_blocks:
            return None

        block = self._free_blocks.pop()
        tensor = block[:required_bytes // block.element_size()].view(dtype).reshape(shape)
        self._allocated[id(tensor)] = block
        return tensor

    def free(self, tensor: torch.Tensor) -> None:
        block_id = id(tensor)
        if block_id in self._allocated:
            block = self._allocated.pop(block_id)
            self._free_blocks.append(block)

    def free_all(self) -> None:
        self._free_blocks.extend(self._allocated.values())
        self._allocated.clear()

    @property
    def used_blocks(self) -> int:
        return len(self._allocated)

    @property
    def free_blocks(self) -> int:
        return len(self._free_blocks)
