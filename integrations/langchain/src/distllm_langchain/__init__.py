"""DistLLM LangChain integration — ChatModel, LLM, Embeddings, and Tool provider."""

__version__ = "0.1.0"

try:
    from distllm.integrations._common.cost_tracker import CostTracker
except ImportError:
    from _common.cost_tracker import CostTracker

try:
    from distllm.integrations._common.model_router import DistLLMModelRouter
except ImportError:
    from _common.model_router import DistLLMModelRouter
from distllm_langchain.chat_models import DistLLMChat
from distllm_langchain.llms import DistLLM
from distllm_langchain.embeddings import DistLLMEmbeddings
from distllm_langchain.tools import DistLLMToolProvider

__all__ = [
    "__version__",
    "CostTracker",
    "DistLLMModelRouter",
    "DistLLMChat",
    "DistLLM",
    "DistLLMEmbeddings",
    "DistLLMToolProvider",
]
