"""Backend adapters for distributed inference.

Each backend adapter wraps a specific inference runtime
(vLLM, llama.cpp, Ray) so that the distributed pipeline
layer can drive any backend uniformly.

Health monitoring, optimization profiles, and graceful
degradation are provided by the sibling modules.
"""

from __future__ import annotations
from distllm.dist.backends.llamacpp import LlamacppPipelineEngine
from distllm.dist.backends.backend_profiles import BackendProfileManager
from distllm.dist.backends.graceful_degradation import GracefulDegradationHandler
from distllm.dist.backends.health_monitor import BackendHealthMonitor
try:
    from distllm.dist.backends.ray import RayPipelineEngine
except ImportError:
    RayPipelineEngine = None  # type: ignore[assignment]
from distllm.dist.backends.vllm import VLLMPipelineEngine

__all__ = [
    "BackendHealthMonitor",
    "BackendProfileManager",
    "GracefulDegradationHandler",
    "LlamacppPipelineEngine",
    "RayPipelineEngine",
    "VLLMPipelineEngine",
]
