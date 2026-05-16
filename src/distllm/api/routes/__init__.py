"""API route routers."""

from .chat import router as chat_router
from .completion import router as completion_router
from .embeddings import router as embeddings_router
from .adapters import router as adapters_router
from .health import router as health_router

__all__ = [
    "chat_router",
    "completion_router",
    "embeddings_router",
    "adapters_router",
    "health_router",
]
