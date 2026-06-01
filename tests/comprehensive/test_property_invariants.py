"""Property-based testing for critical invariants.

Hypothesis-based invariant testing for layer assignment, validation bounds,
temperature safety, KV cache shapes, GPU spec structure, and proportional
scheduling.
"""

import asyncio
import socket
import struct
import threading
import time
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import numpy as np

try:
    from hypothesis import given, strategies as st, settings as hp_settings
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


from tests.comprehensive.conftest import _load_module

# Load clean modules
_kv_cache = _load_module("distllm/core/kv_cache.py")
_token_gen = _load_module("distllm/core/token_generator.py")
_profiles = _load_module("distllm/dist/partition/profiles.py")


# ── Shared helpers (duplicated from test_pipeline_layer_assignment) ──

class _MockNode:
    """Minimal stand-in for a NodeRegistration used by validation logic."""
    def __init__(self, start_layer: int, end_layer: int):
        self.start_layer = start_layer
        self.end_layer = end_layer


def _validate_layers(total_layers, existing_nodes, node_id, start_layer, end_layer):
    """Replicates PipelineOrchestrator._validate_layer_assignment_locked.

    Tested here as a spec/contract — if the real implementation changes,
    this function (and these tests) must be updated to match.
    """
    if total_layers <= 0:
        return
    if start_layer < 0 or end_layer >= total_layers:
        raise ValueError(
            f"Node {node_id}: layers {start_layer}-{end_layer} out of bounds "
            f"(model has {total_layers} layers, "
            f"valid range: 0-{total_layers - 1})"
        )
    if start_layer > end_layer:
        raise ValueError(
            f"Node {node_id}: start_layer ({start_layer}) > "
            f"end_layer ({end_layer})"
        )
    for eid, e in existing_nodes.items():
        if max(start_layer, e.start_layer) <= min(end_layer, e.end_layer):
            raise ValueError(
                f"Node {node_id}: layers {start_layer}-{end_layer} overlap "
                f"with {eid} (layers {e.start_layer}-{e.end_layer})"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 9. Property-Based Testing for Critical Invariants
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestPropertyBasedInvariants:
    """Hypothesis-based invariant testing for core logic."""

    @hp_settings(max_examples=200)
    @given(
        n_nodes=st.integers(min_value=1, max_value=10),
        total_layers=st.integers(min_value=1, max_value=128),
    )
    def test_layer_assignment_contiguous_invariant(self, n_nodes, total_layers):
        """Partition layers contiguously: each layer assigned exactly once.

        When n_nodes > total_layers, extra nodes get 0 layers and are skipped.
        """
        nodes = {}
        actual_nodes = min(n_nodes, total_layers)
        layers_per_node = [total_layers // actual_nodes] * actual_nodes
        for i in range(total_layers % actual_nodes):
            layers_per_node[i] += 1
        start = 0
        nid_map = {}
        for i, count in enumerate(layers_per_node):
            end = start + count - 1
            nid = f"n{i}"
            nodes[nid] = _MockNode(start, end)
            nid_map[i] = nid
            start = end + 1

        assert nodes[nid_map[actual_nodes - 1]].end_layer == total_layers - 1
        for i in range(actual_nodes):
            n = nodes[nid_map[i]]
            assert 0 <= n.start_layer < total_layers
            assert 0 <= n.end_layer < total_layers
            assert n.start_layer <= n.end_layer
        for i in range(actual_nodes - 1):
            cur = nodes[nid_map[i]]
            nxt = nodes[nid_map[i + 1]]
            assert cur.end_layer + 1 == nxt.start_layer

    @hp_settings(max_examples=100)
    @given(
        start_layer=st.integers(min_value=0, max_value=100),
        end_layer=st.integers(min_value=0, max_value=100),
    )
    def test_validation_invariant_start_le_end(self, start_layer, end_layer):
        """If start > end, validation rejects."""
        if start_layer <= end_layer:
            _validate_layers(101, {}, "prop", start_layer, end_layer)
        else:
            with pytest.raises(ValueError, match="start_layer"):
                _validate_layers(101, {}, "prop", start_layer, end_layer)

    @hp_settings(max_examples=100)
    @given(
        total=st.integers(min_value=1, max_value=64),
        start=st.integers(min_value=0, max_value=128),
        end=st.integers(min_value=0, max_value=128),
    )
    def test_bounds_check_invariant(self, total, start, end):
        """Layer range must be within [0, total-1]."""
        if start < 0 or end >= total:
            with pytest.raises(ValueError, match="out of bounds"):
                _validate_layers(total, {}, "prop", start, end)
        elif start <= end:
            _validate_layers(total, {}, "prop", start, end)

    @hp_settings(max_examples=100)
    @given(
        temp=st.floats(min_value=1e-6, max_value=10.0, allow_nan=False),
    )
    def test_temperature_never_produces_nan(self, temp):
        gen = _token_gen.TokenGenerator()
        logits = torch.randn(1, 50) * 3
        tokens, _ = gen.sample(logits, temperature=temp)
        assert not torch.isnan(tokens).any()

    @hp_settings(max_examples=50)
    @given(
        b=st.integers(min_value=1, max_value=2),
        h=st.integers(min_value=1, max_value=4),
        s=st.integers(min_value=1, max_value=6),
        d=st.integers(min_value=4, max_value=16),
    )
    def test_kv_cache_shape_invariant(self, b, h, s, d):
        """KV cache update preserves tensor shapes."""
        c = _kv_cache.KVCache(max_seq_len=64)
        c.init_cache(1, b, h, d, "cpu")
        k = torch.randn(b, h, s, d)
        v = torch.randn(b, h, s, d)
        out_k, out_v = c.update(0, k, v)
        assert out_k.shape == (b, h, s, d)
        assert out_v.shape == (b, h, s, d)

    @hp_settings(max_examples=100)
    @given(
        n_slots=st.integers(min_value=0, max_value=8),
    )
    def test_known_gpu_spec_structure_invariant(self, n_slots):
        """All GPU spec entries have exactly 6 elements."""
        for name, spec in list(_profiles._KNOWN_GPU_SPECS.items())[:n_slots]:
            assert len(spec) == 6, f"{name} has {len(spec)} elements"
            assert isinstance(spec[0], (int, float)), f"{name}: fp16 not numeric"
            assert isinstance(spec[5], str), f"{name}: platform not string"

    @hp_settings(max_examples=100)
    @given(
        total_layers=st.integers(min_value=1, max_value=1000),
        n_nodes=st.integers(min_value=1, max_value=100),
    )
    def test_proportional_scheduling_invariant(self, total_layers, n_nodes):
        """Test that proportional scheduling invariants hold through HeterogeneousScheduler."""
        mod = _load_module("distllm/core/heterogeneous_scheduler.py")
        mems = [24 * 1024**3 + i * 8 * 1024**3 for i in range(n_nodes)]
        configs = [
            {"node_id": f"n{i}", "host": "host", "port": 50051,
             "device_type": "cuda", "total_memory": mems[i], "gpu_name": "Generic"}
            for i in range(n_nodes)
        ]
        cluster = mod.build_heterogeneous_cluster(configs, total_layers)
        cluster = mod.assign_layers_proportional(cluster)
        # Invariant: all layers are covered exactly once
        covered = set()
        for node in cluster.nodes:
            for layer in range(node.start_layer, node.end_layer + 1):
                covered.add(layer)
        assert len(covered) == total_layers
        # Invariant: no overlap between nodes
        ranges = [(n.start_layer, n.end_layer) for n in cluster.nodes]
        for i, (s1, e1) in enumerate(ranges):
            for j, (s2, e2) in enumerate(ranges):
                if i < j:
                    assert e1 < s2, f"Overlap: n{i} [{s1},{e1}] vs n{j} [{s2},{e2}]"
        # Invariant: last node ends at total_layers - 1
        assert cluster.nodes[-1].end_layer == total_layers - 1
