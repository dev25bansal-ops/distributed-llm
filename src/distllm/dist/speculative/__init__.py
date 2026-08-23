"""Speculative decoding subpackage for distributed inference.

Provides multi-draft speculation, remote draft model orchestration,
and verification strategies for accelerating distributed LLM inference.
"""

from __future__ import annotations

from distllm.dist.speculative.adaptive_spec import (
    AcceptanceStats as AcceptanceStats,
    AdaptiveSpecConfig as AdaptiveSpecConfig,
    AdaptiveSpeculator as AdaptiveSpeculator,
)
from distllm.dist.speculative.draft_cache import (
    CachedDraftOutput as CachedDraftOutput,
    DraftCache as DraftCache,
)
from distllm.dist.speculative.draft_registry import (
    DraftCapabilities as DraftCapabilities,
    DraftModelInfo as DraftModelInfo,
    DraftRequest as DraftRequest,
    RemoteDraftRegistry as RemoteDraftRegistry,
)
from distllm.dist.speculative.multi_draft import (
    DraftResult as DraftResult,
    MultiDraftConfig as MultiDraftConfig,
    MultiDraftVerifier as MultiDraftVerifier,
)

__all__ = [
    # draft_registry
    "DraftCapabilities",
    "DraftModelInfo",
    "DraftRequest",
    "RemoteDraftRegistry",
    # multi_draft
    "DraftResult",
    "MultiDraftConfig",
    "MultiDraftVerifier",
    # adaptive_spec
    "AcceptanceStats",
    "AdaptiveSpecConfig",
    "AdaptiveSpeculator",
    # draft_cache
    "CachedDraftOutput",
    "DraftCache",
]
