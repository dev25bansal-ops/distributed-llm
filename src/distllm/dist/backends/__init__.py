"""Backend adapters for distributed inference.

Each backend adapter wraps a specific inference runtime
(vLLM, llama.cpp, Ray) so that the distributed pipeline
layer can drive any backend uniformly.
"""

from __future__ import annotations
from distllm.dist.backends.llamacpp import LlamacppPipelineEngine
try:
    from distllm.dist.backends.ray import RayPipelineEngine
except ImportError:
    RayPipelineEngine = None  # type: ignore[assignment]
from distllm.dist.backends.vllm import VLLMPipelineEngine

__all__ = [
    "LlamacppPipelineEngine",
    "RayPipelineEngine",
    "VLLMPipelineEngine",
]
