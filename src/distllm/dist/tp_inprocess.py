"""In-process tensor parallelism for multi-GPU inference without Ray.

Splits linear layers across GPUs within a single process using
``torch.distributed`` with NCCL backend.  Supports row-parallel and
column-parallel slicing for transformer MLP and attention projections.

Usage::

    engine = InProcessTP(model, world_size=2)
    engine.initialize()
    output = engine.forward(input_ids)
"""

from __future__ import annotations

import os
import threading
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger


class TPSlicer:
    """Helper to split linear layer weights for tensor parallelism."""

    @staticmethod
    def split_column(weight: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
        """Split output dimension (column) across devices.

        For ``nn.Linear(in, out)``: each rank gets ``out // world_size``
        columns.  Used for column-parallel (fused QKV, O-proj).
        """
        dim = weight.shape[0]
        chunk = dim // world_size
        return weight[rank * chunk:(rank + 1) * chunk].contiguous()

    @staticmethod
    def split_row(weight: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
        """Split input dimension (row) across devices.

        For ``nn.Linear(in, out)``: each rank gets ``in // world_size``
        rows.  Used for row-parallel (MLP gate/up/down).
        """
        dim = weight.shape[1]
        chunk = dim // world_size
        return weight[:, rank * chunk:(rank + 1) * chunk].contiguous()


class TPLinear(nn.Module):
    """Linear layer sliced for tensor parallelism.

    - Column-parallel: forward is a standard linear, output is all-reduced.
    - Row-parallel: forward is a standard linear on a sliced weight,
      then all-reduced.
    """

    def __init__(self, in_features: int, out_features: int,
                 world_size: int, rank: int, split: str = "column",
                 bias: bool = True):
        super().__init__()
        self.world_size = world_size
        self.rank = rank
        self.split = split

        if split == "column":
            local_out = out_features // world_size
            self.weight = nn.Parameter(torch.empty(local_out, in_features))
        elif split == "row":
            local_in = in_features // world_size
            self.weight = nn.Parameter(torch.empty(out_features, local_in))
        else:
            raise ValueError(f"Unknown split: {split}")

        self.bias = nn.Parameter(torch.zeros(local_out if split == "column" else out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = nn.functional.linear(x, self.weight, self.bias)
        if self.split == "column":
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output


class InProcessTP:
    """In-process tensor parallelism engine.

    Replaces linear layers in a HuggingFace model with TP-sliced variants
    on separate GPUs.  Works within a single Python process using NCCL.

    Args:
        model: Loaded HuggingFace model (on CPU or meta device).
        world_size: Number of GPUs to use.
        rank: This process's rank (0-based).
        master_port: Port for NCCL init.

    Usage:
        model = AutoModelForCausalLM.from_pretrained(...)
        tp = InProcessTP(model, world_size=2, rank=0)
        tp.initialize()
        # model is now TP-sliced across GPUs 0 and 1
    """

    def __init__(
        self,
        model: nn.Module,
        world_size: int = 1,
        rank: int = 0,
        master_port: int = 29501,
    ):
        self._model = model
        self._world_size = world_size
        self._rank = rank
        self._master_port = master_port
        self._initialized = False

    def initialize(self) -> None:
        """Initialize NCCL process group and slice model layers onto GPUs."""
        if self._world_size <= 1:
            self._initialized = True
            return

        device = torch.device(f"cuda:{self._rank}")
        target_dtype = next(self._model.parameters()).dtype

        # Init NCCL
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = str(self._master_port)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                rank=self._rank,
                world_size=self._world_size,
            )

        # Move model to target device and slice layers
        self._model = self._model.to(device)
        self._model.eval()

        # Replace Attention and MLP linear layers with TP variants
        self._slice_transformer_layers()

        self._initialized = True
        logger.info(
            f"InProcessTP initialized: rank={self._rank}, "
            f"world={self._world_size}, device={device}"
        )

    def _slice_transformer_layers(self) -> None:
        """Replace all linear layers with TP-sliced variants.

        Visits every named submodule; replaces ``nn.Linear`` instances
        whose dimensions are divisible by ``world_size``.
        """
        for name, module in list(self._model.named_modules()):
            for child_name, child in list(module.named_children()):
                if isinstance(child, nn.Linear):
                    sliced = self._slice_linear(child, f"{name}.{child_name}")
                    if sliced is not None:
                        setattr(module, child_name, sliced.to(child.weight.device))

    def _slice_linear(self, linear: nn.Linear, parent_name: str) -> TPLinear | None:
        """Decide whether to slice a linear layer and return TPLinear or None."""
        in_f, out_f = linear.in_features, linear.out_features

        # Heuristic: column-parallel for attention projections (large out_dim),
        # row-parallel for MLP (large in_dim matching hidden_size)
        if in_f == out_f:
            split = "row"
        else:
            split = "column"

        if split == "column" and out_f % self._world_size != 0:
            return None
        if split == "row" and in_f % self._world_size != 0:
            return None

        tp_linear = TPLinear(
            in_features=in_f,
            out_features=out_f,
            world_size=self._world_size,
            rank=self._rank,
            split=split,
            bias=linear.bias is not None,
        )

        with torch.no_grad():
            if split == "column":
                tp_linear.weight.data.copy_(
                    TPSlicer.split_column(linear.weight.data, self._world_size, self._rank)
                )
                if linear.bias is not None:
                    tp_linear.bias.data.copy_(
                        TPSlicer.split_column(linear.bias.data.unsqueeze(1),
                                               self._world_size, self._rank).squeeze(1)
                    )
            else:
                tp_linear.weight.data.copy_(
                    TPSlicer.split_row(linear.weight.data, self._world_size, self._rank)
                )
                if linear.bias is not None:
                    tp_linear.bias.data.copy_(linear.bias.data)

        return tp_linear.to(linear.weight.device)

    @property
    def model(self) -> nn.Module:
        return self._model

    def destroy(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
        self._initialized = False
