"""Inference backends for distributed-llm worker nodes.

Each backend adapter implements the ``BackendAdapter`` protocol, providing
a consistent ``forward()`` interface that the gRPC ``NodeService`` can
call regardless of which inference engine is used.

Usage::

    from distllm.backends import VLLMNodeAdapter

    adapter = VLLMNodeAdapter(model_name="HuggingFaceTB/SmolLM-135M")
    adapter.load_model()
    logits, kv = adapter.forward(input_ids=input_ids)
"""

from __future__ import annotations

from .vllm_backend import VLLMNodeAdapter
from .pytorch_backend import PyTorchNodeAdapter
from .llamacpp_backend import LlamacppNodeAdapter

__all__ = [
    "VLLMNodeAdapter",
    "PyTorchNodeAdapter",
    "LlamacppNodeAdapter",
]
