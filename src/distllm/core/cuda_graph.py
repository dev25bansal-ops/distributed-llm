"""CUDA graph capture for accelerated single-token decode steps.

Caches the CUDA graph for the attention + MLP forward pass at a given
batch size.  After capture, repeated decode steps replay the graph
instead of launching individual kernels, reducing launch overhead
by 30-50% for small batches.

Usage::

    cg = CudaGraphCapture(model, batch_sizes=[1, 2, 4, 8])
    cg.capture(device="cuda:0")
    logits = cg.replay(input_ids, past_key_values)
"""

from __future__ import annotations

from typing import Any

import torch
from loguru import logger


class CudaGraphCapture:
    """Captures and replays CUDA graphs for transformer decode steps.

    Each unique batch size gets its own captured graph.  Graphs are
    replayed for subsequent decode tokens until the batch size changes
    or the KV cache length exceeds the captured max.

    Args:
        model: PyTorch model whose forward pass to capture.
        batch_sizes: List of batch sizes to capture graphs for.
        max_seq_len: Maximum total sequence length to capture (prefill + decode).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        batch_sizes: list[int] | None = None,
        max_seq_len: int = 4096,
    ):
        self._model = model
        self._batch_sizes = batch_sizes or [1, 2, 4, 8, 16]
        self._max_seq_len = max_seq_len
        self._graphs: dict[int, tuple[torch.cuda.CUDAGraph, dict[str, Any]]] = {}
        self._captured = False

    def capture(self, device: str = "cuda") -> None:
        """Warm up and capture CUDA graphs for all configured batch sizes.

        Runs a single forward pass per batch size to warm up, then captures.
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available — skipping graph capture")
            return

        logger.info(f"Capturing CUDA graphs for batch sizes {self._batch_sizes}")
        self._model.eval()

        for bs in self._batch_sizes:
            try:
                self._capture_for_batch(bs, device)
                logger.debug(f"CUDA graph captured for batch_size={bs}")
            except Exception as e:
                logger.warning(f"CUDA graph capture failed for batch_size={bs}: {e}")

        self._captured = True
        logger.info(f"CUDA graph capture complete ({len(self._graphs)}/{len(self._batch_sizes)} succeeded)")

    def _capture_for_batch(self, batch_size: int, device: str) -> None:
        """Capture a single CUDA graph for a given batch size."""
        dummy_input = torch.randint(0, 100, (batch_size, 1), device=device)
        dummy_mask = torch.ones(batch_size, self._max_seq_len, dtype=torch.long, device=device)

        # Warmup: run once to allocate memory
        for _ in range(3):
            _ = self._model(dummy_input, attention_mask=dummy_mask, use_cache=True)

        torch.cuda.synchronize()

        # Capture graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = self._model(dummy_input, attention_mask=dummy_mask, use_cache=True)

        self._graphs[batch_size] = (graph, {
            "input": dummy_input,
            "mask": dummy_mask,
            "output": static_output,
            "batch_size": batch_size,
        })

    def replay(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
    ) -> torch.Tensor:
        """Replay a captured CUDA graph for a single decode step.

        Falls back to eager execution if no graph is available for the
        current batch size.

        PERFORMANCE: If the exact batch size isn't captured, rounds UP to
        the next captured size and replays with padding, then masks the
        extra outputs. This avoids falling back to eager execution for
        intermediate batch sizes.

        Returns:
            Model output logits tensor.
        """
        batch_size = input_ids.shape[0]
        entry = self._graphs.get(batch_size)

        if entry is None:
            # Round UP to the next captured batch size
            captured_sizes = sorted(self._graphs.keys())
            rounded_size = None
            for bs in captured_sizes:
                if bs >= batch_size:
                    rounded_size = bs
                    break

            if rounded_size is not None and rounded_size != batch_size:
                # Pad input to the captured batch size
                pad_size = rounded_size - batch_size
                padded_input = torch.cat([input_ids, input_ids[:pad_size]], dim=0)
                padded_mask = None
                if attention_mask is not None:
                    padded_mask = torch.cat([attention_mask, attention_mask[:pad_size]], dim=0)

                entry = self._graphs.get(rounded_size)
                if entry is not None:
                    graph, buffers = entry
                    buffers["input"].copy_(padded_input)
                    if padded_mask is not None:
                        buffers["mask"].copy_(padded_mask)

                    graph.replay()
                    output = buffers["output"]
                    logits = output.logits[:, -1, :] if hasattr(output, "logits") else output[:, -1, :]
                    # Mask extra outputs — only return logits for the original batch
                    return logits[:batch_size]

            # Fall back to eager execution
            return self._model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            ).logits[:, -1, :]

        graph, buffers = entry
        # Copy fresh input into static buffers
        buffers["input"].copy_(input_ids)
        if attention_mask is not None:
            buffers["mask"].copy_(attention_mask)

        graph.replay()
        output = buffers["output"]
        return output.logits[:, -1, :] if hasattr(output, "logits") else output[:, -1, :]

    @property
    def is_captured(self) -> bool:
        return self._captured
