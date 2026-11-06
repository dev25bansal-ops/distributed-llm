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
from .batch import router as batch_router
from .eval import router as eval_router
from .metrics_history import router as metrics_history_router
from .api_keys import router as api_keys_router

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
    "batch_router",
    "eval_router",
    "metrics_history_router",
    "api_keys_router",
]

# ── Router mount policy (2026-11-05) ────────────────────────────────────────
# The modules below define APIRouters that are deliberately NOT re-exported
# here.  They are imported directly by server.py when mounted; the rest are
# intentionally left unmounted:
#
#   INTENTIONAL_UNMOUNT — scaffolding kept for a documented reason:
#     * files.py      — blocked on the F-066 audit fix: upload_file joins the
#                       raw client filename (`upload_dir / file.filename`),
#                       an arbitrary-file-write risk.  Do not mount until
#                       the filename is sanitized.
#     * debug.py      — replay/debug surface is self-gated behind
#                       DISTLLM_ENABLE_DEBUG_ROUTES=1 + DISTLLM_ADMIN_KEY and
#                       is a development aid, not part of the public API.
#     * moderation.py — moderation is enforced inline by
#                       ContentModerationMiddleware (DISTLLM_MODERATION=1);
#                       the /v1/moderations router is an offline classifier
#                       helper, not a mounted product surface.
#
#   DEAD — unreferenced scaffolding (no test, no doc, no mount rationale):
#     * assistants.py  — OpenAI Assistants scaffolding, nothing references it.
#     * experiments.py — A/B testing scaffolding, nothing references it.
#
#   SHOULD_MOUNT — mounted in server.py (2026-11-05), not exported here to
#   keep this module's export list stable:
#     * exchange.py, plugins.py, fine_tuning.py, tokenize.py, webhooks.py

