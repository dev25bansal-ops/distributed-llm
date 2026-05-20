"""Async Tensor Parallelism: overlap communication with computation.

Uses CUDA streams to overlap NCCL all-reduce of layer outputs with
the forward pass of the next layer. This hides communication latency
behind computation, improving end-to-end throughput for tensor-parallel
distributed inference.

Design:
- Separate CUDA stream for communication (all-reduce)
- Main stream continues with next layer's computation
- Synchronize at the end to ensure all communications complete
"""

from typing import Any

import torch


class AsyncTensorParallel:
    """Overlap NCCL all-reduce with layer computation using CUDA streams.

    While layer N's output is being all-reduced on the comm stream,
    layer N+1's forward pass runs on the main stream.

    Usage:
        async_tp = AsyncTensorParallel(tp_group=group)
        for layer in layers:
            hidden_states = async_tp.forward_overlap(
                layer, hidden_states, prev_output
            )
        hidden_states = async_tp.synchronize()
    """

    def __init__(
        self,
        tp_group: Any = None,
        overlap_layers: int = 1,
        async_op: bool = True,
    ):
        """Initialize async tensor parallel.

        Args:
            tp_group: torch.distributed process group for TP.
            overlap_layers: Number of layers to overlap (1 = max overlap).
            async_op: Whether to use async all-reduce operations.
        """
        self.tp_group = tp_group
        self._overlap_layers = overlap_layers
        self._async_op = async_op

        # Separate CUDA streams when available; CPU/GLOO paths run synchronously.
        self._use_cuda_streams = torch.cuda.is_available()
        self._comm_stream = torch.cuda.Stream() if self._use_cuda_streams else None
        self._main_stream = torch.cuda.current_stream() if self._use_cuda_streams else None

        # Pending all-reduce operations
        self._pending_outputs: list[torch.Tensor] = []
        self._pending_ops: list[Any] = []

    def forward_overlap(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        prev_output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run layer forward with overlapped all-reduce of previous output.

        Args:
            layer: The layer module to run forward on.
            hidden_states: Input tensor [batch, seq, hidden].
            prev_output: Previous layer's output to all-reduce (from prior call).

        Returns:
            Current layer's output (pending all-reduce).
        """
        # Start all-reduce of previous layer's output on comm stream
        if prev_output is not None and self.tp_group is not None:
            if not self._use_cuda_streams:
                import torch.distributed as dist
                op = dist.all_reduce(
                    prev_output,
                    op=dist.ReduceOp.SUM,
                    group=self.tp_group,
                    async_op=self._async_op,
                )
                if self._async_op and op is not None:
                    self._pending_ops.append(op)
                self._pending_outputs.append(prev_output)
                return layer(hidden_states)

            with torch.cuda.stream(self._comm_stream):
                # Ensure main stream is done with prev_output before reducing
                self._comm_stream.wait_stream(self._main_stream)
                prev_output.record_stream(self._comm_stream)

                import torch.distributed as dist
                op = dist.all_reduce(
                    prev_output,
                    op=dist.ReduceOp.SUM,
                    group=self.tp_group,
                    async_op=self._async_op,
                )
                if self._async_op and op is not None:
                    self._pending_ops.append(op)
                self._pending_outputs.append(prev_output)

        # Compute current layer on main stream
        if not self._use_cuda_streams:
            return layer(hidden_states)

        with torch.cuda.stream(self._main_stream):
            # If we have pending all-reduces, ensure comm stream is done
            if self._pending_outputs:
                self._main_stream.wait_stream(self._comm_stream)

            output = layer(hidden_states)
            return output

    def forward_overlap_multiple(
        self,
        layers: list[torch.nn.Module],
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Run multiple layers with maximum comm-compute overlap.

        Uses a pipeline approach: while layer N is being all-reduced,
        layer N+1 computes.

        Args:
            layers: List of layer modules.
            hidden_states: Input tensor.

        Returns:
            Final output tensor.
        """
        prev_output = None
        for i, layer in enumerate(layers):
            hidden_states = self.forward_overlap(layer, hidden_states, prev_output)
            prev_output = hidden_states

        return hidden_states

    def synchronize(self) -> None:
        """Wait for all pending all-reduce operations to complete.

        Call this after the last layer's forward_overlap to ensure
        all communications have finished.
        """
        # Wait for all async ops
        for op in self._pending_ops:
            if op is not None:
                op.wait()

        # Synchronize streams
        if self._use_cuda_streams:
            self._comm_stream.synchronize()
            self._main_stream.wait_stream(self._comm_stream)

        # Clear pending state
        self._pending_outputs.clear()
        self._pending_ops.clear()

    def reset(self) -> None:
        """Reset pending state without synchronizing."""
        self._pending_outputs.clear()
        self._pending_ops.clear()

    def stats(self) -> dict:
        return {
            "overlap_layers": self._overlap_layers,
            "async_op": self._async_op,
            "pending_outputs": len(self._pending_outputs),
            "comm_stream_idle": self._comm_stream.query() if self._use_cuda_streams else True,
        }


class PipelineAsyncTP:
    """Higher-level wrapper for pipeline-parallel async TP.

    Manages the full pipeline forward pass with overlapped communication.

    Usage:
        pipeline_tp = PipelineAsyncTP(layers, tp_group)
        output = pipeline_tp.forward(hidden_states)
    """

    def __init__(
        self,
        layers: list[torch.nn.Module],
        tp_group: Any = None,
        overlap_count: int = 2,
    ):
        self.layers = layers
        self._async_tp = AsyncTensorParallel(
            tp_group=tp_group,
            overlap_layers=overlap_count,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run full pipeline forward with async TP overlap."""
        output = self._async_tp.forward_overlap_multiple(
            self.layers, hidden_states
        )
        self._async_tp.synchronize()
        return output

    def stats(self) -> dict:
        return {
            "num_layers": len(self.layers),
            **self._async_tp.stats(),
        }
