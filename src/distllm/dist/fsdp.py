"""FSDP-style weight sharding across nodes.

Each node holds 1/N of each model parameter. Before the forward pass,
parameters are all-gathered to reconstruct the full weights, then
non-local shards are freed after the forward completes.

This trades off extra communication for reduced per-node memory,
enabling larger models to fit across multiple nodes.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger


@dataclass
class FSDPConfig:
    """Configuration for FSDP-style weight sharding.

    Attributes:
        world_size: Total number of sharding nodes/ranks.
        rank: Rank of this node (0-based).
        min_param_size: Minimum number of elements to shard
            (parameters below this threshold are kept full on every rank).
        cpu_offload: If True, offload non-local shards to CPU instead of
            freeing memory after the forward pass.
        mixed_precision: If set, cast gathered parameters to this dtype
            before the forward pass.
    """
    world_size: int = 1
    rank: int = 0
    min_param_size: int = 1024
    cpu_offload: bool = False
    mixed_precision: torch.dtype | None = None


class FSDPShard:
    """FSDP-style weight sharding for distributed model inference.

    Wraps an ``nn.Module`` and shards its trainable parameters across
    *world_size* ranks. Only the local shard of each parameter is
    retained in GPU memory. Before the forward pass, ``gather()``
    all-gathers all shards to reconstruct the full weights. After
    the forward pass, ``free()`` (called automatically by ``forward()``)
    discards non-local shards and restores only the local chunk.

    Usage::

        fsdp = FSDPShard(model, world_size=4, rank=0)
        fsdp.shard()                  # split params, keep local chunk
        out = fsdp.forward(x)         # gather -> forward -> free
        # --- or manual control ---
        fsdp.gather()
        out = model(x)
        fsdp.free()

    Args:
        module: The ``nn.Module`` to shard.
        world_size: Total number of sharding ranks.
        rank: Local rank (0-based).
        config: Optional ``FSDPConfig`` for advanced options.
    """

    def __init__(
        self,
        module: nn.Module,
        world_size: int | None = None,
        rank: int | None = None,
        config: FSDPConfig | None = None,
    ):
        self._module = module

        # Resolve config
        if config is None:
            config = FSDPConfig()
        if world_size is not None:
            config.world_size = world_size
        if rank is not None:
            config.rank = rank
        self._config = config

        # Internal state: maps parameter name -> (local_shard, orig_shape, orig_dtype, orig_device)
        self._sharded_params: dict[str, tuple[torch.Tensor, torch.Size, torch.dtype, torch.device]] = {}
        self._gathered = False
        self._lock = threading.Lock()

        if config.world_size < 1:
            raise ValueError(f"FSDP world_size must be >= 1, got {config.world_size}")
        if config.rank < 0 or config.rank >= config.world_size:
            raise ValueError(f"FSDP rank {config.rank} out of range [0, {config.world_size})")

    # -- Public API -----------------------------------------------------------

    def shard(self) -> None:
        """Split each parameter into ``world_size`` shards and keep the local one.

        After this call, every parameter with ``numel >= min_param_size``
        is replaced in-place by its local shard. The metadata needed to
        reconstruct the original parameter is stored internally.
        """
        if self._config.world_size <= 1:
            return

        logger.info(
            f"FSDP sharding rank {self._config.rank}/{self._config.world_size}"
        )

        count = 0
        for name, param in list(self._module.named_parameters()):
            if self._shard_one(name, param):
                count += 1

        logger.info(
            f"FSDP shard complete: {count} parameter shards on "
            f"rank {self._config.rank}"
        )

    def gather(self) -> None:
        """All-gather sharded parameters to reconstruct full weights in-place.

        Uses ``torch.distributed.all_gather``. After this call, all
        module parameters have their original shape and values.
        """
        if self._config.world_size <= 1 or self._gathered or not self._sharded_params:
            return

        ws = self._config.world_size
        group = self._get_group()

        for name, (local_shard, orig_shape, orig_dtype, orig_device) in self._sharded_params.items():
            param = _get_param_by_name(self._module, name)
            if param is None:
                continue

            # Pad the local shard so all ranks contribute same-sized tensors
            # (last rank may have fewer elements).
            numel = orig_shape.numel()
            max_chunk_size = _ceil_div(numel, ws)
            padded = local_shard.flatten()
            if padded.numel() < max_chunk_size:
                pad_len = max_chunk_size - padded.numel()
                padded = torch.cat([padded, padded.new_zeros(pad_len)])

            gather_list = [torch.empty(max_chunk_size, dtype=padded.dtype, device=padded.device) for _ in range(ws)]
            dist.all_gather(gather_list, padded, group=group)

            # Concatenate and trim back to original size
            full_flat = torch.cat(gather_list, dim=0)[:numel]
            full_tensor = full_flat.view(orig_shape).to(
                dtype=orig_dtype, device=orig_device
            )

            if self._config.mixed_precision is not None:
                full_tensor = full_tensor.to(dtype=self._config.mixed_precision)

            with torch.no_grad():
                param.data.copy_(full_tensor)

        self._gathered = True

    def free(self) -> None:
        """Discard non-local shards, keeping only the local chunk.

        After this call, each parameter is replaced with its local shard,
        freeing memory that was temporarily used for the full weights.
        """
        if self._config.world_size <= 1 or not self._gathered or not self._sharded_params:
            return

        rank = self._config.rank
        device = _resolve_device(self._module)

        for name, (local_shard, orig_shape, orig_dtype, orig_device) in self._sharded_params.items():
            param = _get_param_by_name(self._module, name)
            if param is None:
                continue

            shard = local_shard.to(device=device, dtype=orig_dtype)
            with torch.no_grad():
                param.data = shard

        self._gathered = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Gather full weights, run forward, then free non-local shards.

        This is the primary entry point for FSDP-wrapped inference::

            output = fsdp.forward(input_ids=input_ids)

        Args:
            *args: Positional arguments forwarded to the underlying module.
            **kwargs: Keyword arguments forwarded to the underlying module.

        Returns:
            The output tensor from the module forward pass.
        """
        if self._config.world_size <= 1:
            return self._module(*args, **kwargs)

        self.gather()
        try:
            return self._module(*args, **kwargs)
        finally:
            self.free()

    # -- Internal helpers -----------------------------------------------------

    def _shard_one(self, name: str, param: nn.Parameter) -> bool:
        """Shard a single parameter. Returns True if sharded, False if skipped."""
        ws = self._config.world_size
        rank = self._config.rank
        min_size = self._config.min_param_size

        numel = param.data.numel()
        if numel < min_size:
            return False

        # Must have at least one element per rank so every rank participates
        # in the all-gather with a non-empty tensor.
        if numel < ws:
            return False

        chunk_size = _ceil_div(numel, ws)
        start = rank * chunk_size
        end = min(start + chunk_size, numel)

        flat = param.data.flatten()
        local_chunk = flat[start:end].clone()

        self._sharded_params[name] = (
            local_chunk,
            param.data.shape,
            param.data.dtype,
            param.data.device,
        )

        with torch.no_grad():
            param.data = local_chunk

        return True

    @staticmethod
    def _get_group() -> dist.ProcessGroup | None:
        """Return the default FSDP process group."""
        if dist.is_initialized():
            return dist.group.WORLD
        return None


# -- Module-level helpers ---------------------------------------------------


def _ceil_div(a: int, b: int) -> int:
    """Ceiling division: ``ceil(a / b)``."""
    return -(-a // b)


def _resolve_device(module: nn.Module) -> torch.device:
    """Get the device of the first parameter in the module."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")


def _get_param_by_name(module: nn.Module, name: str) -> nn.Parameter | None:
    """Resolve a dotted parameter name on a module."""
    parts = name.split(".")
    obj: Any = module
    for part in parts:
        if isinstance(obj, (nn.ModuleList, nn.ParameterList)):
            try:
                obj = obj[int(part)]
            except (ValueError, IndexError, TypeError):
                return None
        elif isinstance(obj, nn.Module):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        else:
            return None
    return obj if isinstance(obj, nn.Parameter) else None
