"""DistLLM FastAPI middleware — drop-in LLM proxy for FastAPI apps."""

__version__ = "0.1.0"

from distllm_fastapi.middleware import DistLLMMiddleware
from distllm_fastapi.router import create_distllm_router

__all__ = ["__version__", "DistLLMMiddleware", "create_distllm_router"]
