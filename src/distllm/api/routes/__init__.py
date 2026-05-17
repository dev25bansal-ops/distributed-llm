"""API route routers."""

from .chat import router as chat_router
from .completion import router as completion_router
from .embeddings import router as embeddings_router
from .adapters import router as adapters_router
from .health import router as health_router
from .versions import router as versions_router
from .multi_model import router as multi_model_router
from .batch import router as batch_router
from .audio import router as audio_router
from .images import router as images_router
from .moderations import router as moderations_router
from .files import router as files_router
from .fine_tuning import router as fine_tuning_router

__all__ = [
    "chat_router",
    "completion_router",
    "embeddings_router",
    "adapters_router",
    "health_router",
    "versions_router",
    "multi_model_router",
    "batch_router",
    "audio_router",
    "images_router",
    "moderations_router",
    "files_router",
    "fine_tuning_router",
]
