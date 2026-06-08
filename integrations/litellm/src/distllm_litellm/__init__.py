"""DistLLM LiteLLM provider — register DistLLM as a LiteLLM backend."""

__version__ = "0.1.0"

from distllm_litellm.provider import get_distllm_custom_llm

__all__ = ["__version__", "get_distllm_custom_llm"]
