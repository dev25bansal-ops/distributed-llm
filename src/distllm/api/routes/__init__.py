"""API route routers."""

from .chat import router as chat_router
from .completion import router as completion_router
from .embeddings import router as embeddings_router
from .health import router as health_router
from .gossip import router as gossip_router
__all__ = [
    "chat_router",
    "completion_router",
    "embeddings_router",
    "health_router",
    "gossip_router",
]
