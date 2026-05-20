"""CUDA Graph Capture for eliminating Python overhead in the hot inference path.

Captures CUDA graphs for common batch sizes after model load. During generation,
replays the graph instead of executing Python-level forward passes.

Inspired by vLLM's CUDA graph capture for decode-phase acceleration.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from loguru import logger


@dataclass
class GraphBuffers:
    """Static input/output buffers for a captured CUDA graph."""
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    output: torch.Tensor
    # KV cache buffers (mutable, updated between replays)
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)


class CUDAGraphPool:
    """Manages CUDA graphs for common batch sizes.

    Captures graphs after model load for batch sizes [1, 2, 4, 8, 16, 32].
    Each graph captures the forward pass pattern; KV cache tensors are
    updated between replays via buffer copy.

    Usage:
        pool = CUDAGraphPool(model, batch_sizes=[1, 2, 4, 8])
        # During generation:
        if pool.has_graph(batch_size):
            output = pool.replay(batch_size, input_ids, past_kv)
        else:
            output = model(input_ids, past_key_values=past_kv)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        batch_sizes: list[int] | None = None,
        max_seq_len: int = 4096,
        num_layers: int = 0,
        num_heads: int = 0,
        head_dim: int = 0,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.model = model
        self.max_seq_len = max_seq_len
        self.batch_sizes = batch_sizes or [1, 2, 4, 8, 16, 32]
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        self._graphs: dict[int, tuple[torch.cuda.CUDAGraph, GraphBuffers]] = {}
        self._captured = False

    @property
    def available(self) -> bool:
        """Check if CUDA graphs are captured and ready."""
        return self._captured and len(self._graphs) > 0

    def has_graph(self, batch_size: int) -> bool:
        """Check if a graph exists for the given batch size."""
        return batch_size in self._graphs

    def capture_all(self) -> None:
        """Capture CUDA graphs for all configured batch sizes."""
        if self.device != "cuda" or not torch.cuda.is_available():
            logger.warning("CUDA Graph: device is not CUDA, skipping capture")
            return

        # Warmup
        torch.cuda.synchronize()

        captured = 0
        for bs in self.batch_sizes:
            try:
                self._capture_graph(bs)
                captured += 1
            except Exception as e:
                logger.warning(f"CUDA Graph: failed to capture batch_size={bs}: {e}")

        self._captured = captured > 0
        if captured:
            logger.info(f"CUDA Graph: captured {captured}/{len(self.batch_sizes)} graphs")

    def _capture_graph(self, batch_size: int) -> None:
        """Capture a single CUDA graph for the given batch size."""
        # Create static buffers
        input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
        position_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
        attention_mask = torch.ones((batch_size, 1), dtype=torch.long, device=self.device)
        output_buf = torch.zeros(
            (batch_size, 1, self.model.config.vocab_size if hasattr(self.model, 'config') else 32000),
            dtype=self.dtype,
            device=self.device,
        )

        # Create KV cache buffers
        past_kv = []
        for _ in range(self.num_layers):
            k = torch.zeros(
                (batch_size, self.num_heads, 1, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            v = torch.zeros(
                (batch_size, self.num_heads, 1, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            past_kv.append((k, v))

        buffers = GraphBuffers(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output=output_buf,
            past_key_values=past_kv,
        )

        # Warmup forward
        with torch.no_grad():
            _ = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_kv if past_kv else None,
                use_cache=True,
            )

        # Capture graph
        graph = torch.cuda.CUDAGraph()

        # Set up the capture function - it must write to static buffers
        def capture_fn():
            outputs = self.model(
                input_ids=buffers.input_ids,
                attention_mask=buffers.attention_mask,
                past_key_values=buffers.past_key_values if buffers.past_key_values else None,
                use_cache=True,
            )
            # Copy logits to output buffer
            buffers.output.copy_(outputs.logits[:, -1:, :])
            # Update KV cache buffers
            if outputs.past_key_values:
                for i, (k, v) in enumerate(outputs.past_key_values):
                    if i < len(buffers.past_key_values):
                        buffers.past_key_values[i][0].copy_(k[:, :, -1:, :].contiguous())
                        buffers.past_key_values[i][1].copy_(v[:, :, -1:, :].contiguous())

        with torch.cuda.graph(graph):
            capture_fn()

        self._graphs[batch_size] = (graph, buffers)

    def replay(
        self,
        batch_size: int,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
    ) -> torch.Tensor:
        """Replay a captured CUDA graph.

        Args:
            batch_size: Batch size (must have a captured graph).
            input_ids: Current input token IDs [batch, seq_len].
            position_ids: Optional position IDs.
            attention_mask: Optional attention mask.
            past_key_values: Optional KV cache from previous steps.

        Returns:
            Logits tensor [batch, 1, vocab_size].
        """
        if batch_size not in self._graphs:
            raise ValueError(f"No CUDA graph captured for batch_size={batch_size}")

        graph, buffers = self._graphs[batch_size]

        # Copy inputs to static buffers
        if input_ids.dim() == 2:
            # Take last token if seq_len > 1
            buffers.input_ids[:, :1].copy_(input_ids[:, -1:])
        else:
            buffers.input_ids[:, :1].copy_(input_ids.unsqueeze(-1))

        if position_ids is not None:
            if position_ids.dim() == 2:
                buffers.position_ids[:, :1].copy_(position_ids[:, -1:])
            else:
                buffers.position_ids[:, :1].copy_(position_ids.unsqueeze(-1))

        if attention_mask is not None and attention_mask.dim() == 2:
            buffers.attention_mask[:, :1].copy_(attention_mask[:, -1:])

        # Copy KV cache to static buffers
        if past_key_values and buffers.past_key_values:
            for i, (k, v) in enumerate(past_key_values):
                if i < len(buffers.past_key_values):
                    if k.dim() == 4:
                        buffers.past_key_values[i][0].copy_(k[:, :, -1:, :].contiguous())
                        buffers.past_key_values[i][1].copy_(v[:, :, -1:, :].contiguous())

        # Replay graph
        graph.replay()

        # Return output (clone to detach from static buffer)
        return buffers.output.clone()

    def get_stats(self) -> dict:
        """Get capture statistics."""
        return {
            "captured": self._captured,
            "batch_sizes": list(self._graphs.keys()),
            "graph_count": len(self._graphs),
        }
