"""Regression test for P0 cache key mismatch bug.

store_prefix must hash the FULL token list so that lookup_prefix (which
hashes the full token list) can hit the multi-tier cache. Previously
store_prefix hashed only tokens[:32], so any prefix longer than 32 tokens
was stored under a key lookup_prefix never produced.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from distllm.core.cache_manager import CacheManager


def test_store_lookup_hits_for_prefix_over_32_tokens():
    cm = CacheManager()  # prefix_cache disabled, predictive disabled
    tokens = list(range(50))  # > 32 tokens
    kv_data = object()

    cm.store_prefix(tokens, kv_data)

    match_len, result = cm.lookup_prefix(tokens)

    assert match_len > 0, "expected a cache hit (match_len > 0)"
    assert match_len == len(tokens)
    # lookup returns the stored KV data itself (not the internal blob).
    assert result is kv_data


def test_store_lookup_hits_for_short_prefix():
    cm = CacheManager()
    tokens = list(range(10))
    kv_data = object()

    cm.store_prefix(tokens, kv_data)

    match_len, result = cm.lookup_prefix(tokens)

    assert match_len == len(tokens)
    assert result is kv_data
