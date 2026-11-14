"""Regression: constructing the decoder on a CPU-only machine must not crash.

Defect: ``PipelinedSpeculativeDecoder.__init__`` defaults to ``device="cuda"``
and decides whether to allocate CUDA streams from ``device.startswith("cuda")``.
That string test does not consult ``torch.cuda.is_available()``, so on a
CPU-only host (or a CUDA build with no GPU visible) construction raised
``RuntimeError: torch.cuda.Stream requires CUDA support`` before any work
could happen.  The guard must fall back to the non-stream path instead.
"""

from __future__ import annotations

import pytest
import torch

from distllm.core.async_pipelined_speculative import PipelinedSpeculativeDecoder


def _noop_target(x, **kwargs):
    return torch.zeros(1, 1, 8)


def test_construct_with_default_device_on_cpu_only_host():
    """No explicit device -> must not raise when CUDA is unavailable."""
    if torch.cuda.is_available():
        pytest.skip("host has CUDA; the CPU-only path cannot be exercised")

    decoder = PipelinedSpeculativeDecoder(target_forward=_noop_target)
    # Falls back to the non-stream path rather than crashing.
    assert decoder._main_stream is None
    assert decoder._verify_stream is None
    assert decoder._transfer_stream is None


def test_explicit_cuda_device_without_cuda_downgrades():
    """Asking for 'cuda' explicitly on a CPU-only host downgrades, not crash."""
    if torch.cuda.is_available():
        pytest.skip("host has CUDA")

    decoder = PipelinedSpeculativeDecoder(
        target_forward=_noop_target, device="cuda", use_cuda_streams=True,
    )
    assert decoder._main_stream is None


def test_cpu_device_never_allocates_streams():
    """device='cpu' has always skipped streams; keep that behaviour pinned."""
    decoder = PipelinedSpeculativeDecoder(
        target_forward=_noop_target, device="cpu",
    )
    assert decoder._main_stream is None