"""DistLLM Semantic Kernel integration — chat and embedding services."""

__version__ = "0.1.0"

from distllm_sk.chat_completion import DistLLMChatCompletion
from distllm_sk.embeddings import DistLLMEmbeddingService

__all__ = ["__version__", "DistLLMChatCompletion", "DistLLMEmbeddingService"]
