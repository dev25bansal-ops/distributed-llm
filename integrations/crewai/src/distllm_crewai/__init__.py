"""DistLLM CrewAI integration — Tools, LLM, Embeddings, and Knowledge Source."""

__version__ = "0.1.0"

from distllm_crewai.tools import DistLLMToolProvider
from distllm_crewai.llm import DistLLMCrewLLM
from distllm_crewai.embedder import DistLLMCrewEmbedder
from distllm_crewai.knowledge_source import DistLLMKnowledgeSource

__all__ = [
    "__version__",
    "DistLLMToolProvider",
    "DistLLMCrewLLM",
    "DistLLMCrewEmbedder",
    "DistLLMKnowledgeSource",
]
