"""API route routers."""

from .chat import router as chat_router
from .chat import v2_router as chat_v2_router
from .completion import router as completion_router
from .embeddings import router as embeddings_router
from .health import router as health_router
from .gossip import router as gossip_router
from .admin import router as admin_router
from .marketplace import router as marketplace_router
from .federated import router as federated_router
from .webrtc import router as webrtc_router
from .prompts import router as prompts_router
from .model_registry import router as model_registry_router
from .leaderboard import router as leaderboard_router
from .scheduler import router as scheduler_router
from .router_admin import router as router_admin_router
from .defrag import router as defrag_router
__all__ = [
    "chat_router",
    "chat_v2_router",
    "completion_router",
    "embeddings_router",
    "health_router",
    "gossip_router",
    "admin_router",
    "marketplace_router",
    "federated_router",
    "webrtc_router",
    "prompts_router",
    "model_registry_router",
    "leaderboard_router",
    "scheduler_router",
    "router_admin_router",
    "defrag_router",
]
