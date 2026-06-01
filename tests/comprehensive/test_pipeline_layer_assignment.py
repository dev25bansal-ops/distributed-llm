"""Pipeline layer assignment validation tests.

Validates layer boundary rules that PipelineOrchestrator enforces,
including boundary checks, overlap detection, and property-based invariants.
"""

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


# ═══════════════════════════════════════════════════════════════════════════
# 1. Pipeline Layer Assignment Validation
# ═══════════════════════════════════════════════════════════════════════════

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


class TestPipelineLayerAssignment:
    """Validates layer boundary rules that PipelineOrchestrator enforces."""

    def test_valid_first_node(self):
        _validate_layers(32, {}, "n1", 0, 15)

    def test_valid_second_node_contiguous(self):
        n = self._make_nodes("n1", 0, 15)
        _validate_layers(32, n, "n2", 16, 31)

    def test_valid_single_layer(self):
        _validate_layers(1, {}, "single", 0, 0)

    def test_valid_last_layer(self):
        n = self._make_nodes("n1", 0, 30)
        _validate_layers(32, n, "n2", 31, 31)

    def test_total_layers_zero_skips_validation(self):
        _validate_layers(0, {}, "any", -5, -1)

    def test_negative_start_fails(self):
        with pytest.raises(ValueError, match="out of bounds"):
            _validate_layers(32, {}, "n", -1, 10)

    def test_end_layer_equal_total_fails(self):
        with pytest.raises(ValueError, match="out of bounds"):
            _validate_layers(32, {}, "n", 0, 32)

    def test_end_layer_exceeds_total_fails(self):
        with pytest.raises(ValueError, match="out of bounds"):
            _validate_layers(32, {}, "n", 0, 40)

    def test_start_gt_end_fails(self):
        with pytest.raises(ValueError, match="start_layer.*>.*end_layer"):
            _validate_layers(32, {}, "n", 20, 10)

    def test_exact_overlap_detected(self):
        n = self._make_nodes("existing", 8, 15)
        with pytest.raises(ValueError, match="overlap"):
            _validate_layers(32, n, "new", 8, 15)

    def test_partial_overlap_start(self):
        n = self._make_nodes("existing", 8, 15)
        with pytest.raises(ValueError, match="overlap"):
            _validate_layers(32, n, "new", 4, 10)

    def test_partial_overlap_end(self):
        n = self._make_nodes("existing", 8, 15)
        with pytest.raises(ValueError, match="overlap"):
            _validate_layers(32, n, "new", 12, 20)

    def test_adjacent_layers_no_overlap(self):
        n = self._make_nodes("existing", 0, 15)
        _validate_layers(32, n, "new", 16, 31)

    def test_gap_between_layers_allowed(self):
        n = self._make_nodes("existing", 0, 7)
        _validate_layers(32, n, "new", 16, 31)

    def test_multi_node_no_overlap(self):
        n = {}
        n.update(self._make_nodes("a", 0, 7))
        n.update(self._make_nodes("b", 8, 15))
        n.update(self._make_nodes("c", 16, 23))
        _validate_layers(32, n, "d", 24, 31)

    def test_multi_node_overlap_any_fails(self):
        n = {}
        n.update(self._make_nodes("a", 0, 7))
        n.update(self._make_nodes("b", 8, 15))
        n.update(self._make_nodes("c", 16, 23))
        with pytest.raises(ValueError, match="overlap"):
            _validate_layers(32, n, "d", 20, 31)

    def test_40_layers_model(self):
        _validate_layers(40, {}, "n0", 0, 39)

    def test_80_layers_model_split(self):
        n = self._make_nodes("n0", 0, 39)
        _validate_layers(80, n, "n1", 40, 79)

    def test_layer_boundary_edge_0(self):
        _validate_layers(32, {}, "n", 0, 0)

    def test_layer_boundary_edge_31(self):
        _validate_layers(32, {}, "n", 31, 31)

    # ── Property-based validation ──

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @hp_settings(max_examples=200)
    @given(
        total=st.integers(min_value=1, max_value=128),
        s=st.integers(min_value=0, max_value=127),
        e=st.integers(min_value=0, max_value=127),
    )
    def test_property_valid_ranges_no_error(self, total, s, e):
        if s <= e and s >= 0 and e < total:
            _validate_layers(total, {}, "prop", s, e)

    @staticmethod
    def _make_nodes(nid="n1", start=0, end=31):
        return {nid: _MockNode(start, end)}
