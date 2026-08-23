"""Regression test for P0 bug in CrossModelPrefixSharing.lookup.

Ensures that two different prompts sharing the same base model return
DIFFERENT cache entries (the sibling path must key on prefix_hash).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from distllm.core.cluster_predictive_prefetcher import (
    ClusterPredictivePrefetcher,
    WarmStartSource,
)
from distllm.core.cross_model_prefix_sharing import CrossModelPrefixSharing


def _make_sharing() -> CrossModelPrefixSharing:
    sharing = CrossModelPrefixSharing(default_ttl=3600.0)
    sharing.register_model("llama-base", shared_layers=0, total_layers=70)
    sharing.register_model(
        "llama-instruct", base_model="llama-base", shared_layers=70, total_layers=70
    )
    sharing.register_model(
        "llama-chat", base_model="llama-base", shared_layers=70, total_layers=70
    )
    return sharing


def test_sibling_returns_different_entry_for_different_prompt():
    sharing = _make_sharing()

    prompt_a = [1, 2, 3, 4]
    prompt_b = [9, 8, 7, 6]

    # llama-chat stores KV for prompt A
    kv_a = object()
    sharing.store("llama-chat", prompt_a, kv_a)

    # llama-instruct asks for a DIFFERENT prompt B with the same base model
    entry_b = sharing.lookup("llama-instruct", prompt_b)

    assert entry_b is None, (
        f"Sibling lookup for a different prompt must return None, "
        f"but got entry from source_model={entry_b.source_model!r}"
    )


def test_sibling_returns_entry_for_same_prompt():
    sharing = _make_sharing()

    prompt_a = [1, 2, 3, 4]

    kv_a = object()
    sharing.store("llama-chat", prompt_a, kv_a)

    # Same prompt, different sibling variant -> should hit
    entry = sharing.lookup("llama-instruct", prompt_a)

    assert entry is not None
    assert entry.kv_data is kv_a
    assert entry.source_model == "llama-chat"


def test_sibling_returns_the_matching_entry_among_many():
    sharing = _make_sharing()

    prompt_a = [1, 2, 3, 4]
    prompt_b = [9, 8, 7, 6]

    kv_a = object()
    kv_b = object()
    sharing.store("llama-chat", prompt_a, kv_a)
    sharing.store("llama-chat", prompt_b, kv_b)

    # Querying B must return B's KV — never A's (wrong-token injection).
    entry = sharing.lookup("llama-instruct", prompt_b)

    assert entry is not None
    assert entry.kv_data is kv_b
    assert entry.token_ids == prompt_b


def test_sibling_no_match_when_only_other_prompts_cached():
    sharing = _make_sharing()

    sharing.store("llama-chat", [1, 2, 3, 4], object())

    # A third, uncached prompt must miss even though the sibling family
    # has entries (they are for a different prefix).
    assert sharing.lookup("llama-instruct", [5, 6, 7, 8]) is None


def test_prefetcher_cross_model_only_on_matching_prompt():
    """Consumer-level: ClusterPredictivePrefetcher.warm_start must only
    report CROSS_MODEL when the cached sibling prefix equals the query."""
    sharing = _make_sharing()
    prefetcher = ClusterPredictivePrefetcher(
        node_id="node-1",
        cross_model_sharing=sharing,
    )

    sharing.store("llama-chat", [1, 2, 3, 4], object())

    # Different prompt, same family -> must MISS (not a cross-model hit).
    miss = prefetcher.warm_start([9, 8, 7, 6], model_id="llama-instruct")
    assert miss.source == WarmStartSource.MISS

    # Same prompt, different variant -> CROSS_MODEL warm start.
    hit = prefetcher.warm_start([1, 2, 3, 4], model_id="llama-instruct")
    assert hit.source == WarmStartSource.CROSS_MODEL
    assert hit.model_id == "llama-chat"
