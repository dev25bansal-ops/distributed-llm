"""M-06: TopologyRegistry — consolidated layer-assignment math.

The orchestrator (dist/) and cluster manager (core/) previously each held
part of the layer-assignment logic. These tests pin the shared, pure
functions both layers now delegate to.
"""

from __future__ import annotations

import pytest

from distllm.core.topology_registry import (
    NodeRange,
    compute_even_split,
    compute_vram_weighted_split,
    coverage_gaps,
    validate_assignment,
)


class TestValidateAssignment:
    def test_no_overlap_passes(self):
        existing = [NodeRange("a", 0, 7), NodeRange("b", 8, 15)]
        validate_assignment("c", 16, 23, existing)  # no raise

    def test_overlap_raises(self):
        existing = [NodeRange("a", 0, 7)]
        with pytest.raises(ValueError, match="Layer overlap"):
            validate_assignment("b", 5, 12, existing)

    def test_same_node_id_ignored(self):
        # Re-registering the same node must not self-collide.
        existing = [NodeRange("a", 0, 7)]
        validate_assignment("a", 0, 7, existing)  # no raise

    def test_exact_adjacent_boundary_ok(self):
        existing = [NodeRange("a", 0, 7)]
        validate_assignment("b", 8, 15, existing)  # no raise

    def test_error_message_format_preserved(self):
        existing = [NodeRange("a", 0, 7)]
        with pytest.raises(ValueError) as exc:
            validate_assignment("b", 3, 9, existing)
        assert "b[3-9]" in str(exc.value)
        assert "a[0-7]" in str(exc.value)


class TestNodeRange:
    def test_size_inclusive(self):
        assert NodeRange("n", 0, 7).size == 8
        assert NodeRange("n", 3, 3).size == 1

    def test_overlaps_symmetric(self):
        a = NodeRange("a", 0, 7)
        b = NodeRange("b", 7, 15)
        assert a.overlaps(b) and b.overlaps(a)


class TestEvenSplit:
    def test_even_division(self):
        ranges = compute_even_split(32, ["n1", "n2"])
        assert [(r.start_layer, r.end_layer) for r in ranges] == [(0, 15), (16, 31)]

    def test_remainder_spread_to_first_nodes(self):
        ranges = compute_even_split(10, ["a", "b", "c"])
        sizes = [r.size for r in ranges]
        assert sizes == [4, 3, 3]
        # contiguous, no gaps
        assert ranges[0].end_layer + 1 == ranges[1].start_layer
        assert ranges[1].end_layer + 1 == ranges[2].start_layer

    def test_more_nodes_than_layers(self):
        ranges = compute_even_split(2, ["a", "b", "c"])
        # base=0, rem=2 -> first two nodes get 1 layer each, last gets 0
        sizes = [r.size for r in ranges]
        assert sum(sizes) == 2

    def test_empty_nodes(self):
        assert compute_even_split(10, []) == []


class TestVramWeightedSplit:
    def test_proportional_allocation(self):
        ranges = compute_vram_weighted_split(24, {"big": 24.0, "small": 8.0})
        by_id = {r.node_id: r.size for r in ranges}
        assert by_id["big"] == 18
        assert by_id["small"] == 6

    def test_fallback_on_zero_capacity(self):
        ranges = compute_vram_weighted_split(8, {"a": 0.0, "b": 0.0})
        sizes = sorted(r.size for r in ranges)
        assert sizes == [4, 4]

    def test_total_preserved(self):
        ranges = compute_vram_weighted_split(37, {"a": 3.0, "b": 5.0, "c": 2.0})
        assert sum(r.size for r in ranges) == 37


class TestCoverageGaps:
    def test_full_coverage_no_gaps(self):
        ranges = [NodeRange("a", 0, 7), NodeRange("b", 8, 15)]
        assert coverage_gaps(ranges, 16) == []

    def test_middle_gap_detected(self):
        ranges = [NodeRange("a", 0, 3), NodeRange("b", 8, 15)]
        assert coverage_gaps(ranges, 16) == [(4, 7)]

    def test_tail_gap_detected(self):
        ranges = [NodeRange("a", 0, 7)]
        assert coverage_gaps(ranges, 16) == [(8, 15)]

    def test_empty_coverage(self):
        assert coverage_gaps([], 8) == [(0, 7)]

    def test_zero_layers(self):
        assert coverage_gaps([], 0) == []


class TestOrchestratorDelegation:
    def test_orchestrator_validate_delegates_to_registry(self):
        """The orchestrator's validate_layer_assignment now routes through
        the shared registry (M-06)."""
        from distllm.dist.pipeline.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator()
        orch._nodes["a"] = type("N", (), {"start_layer": 0, "end_layer": 7})()
        # Adjacent assignment passes
        orch.validate_layer_assignment("b", 8, 15)
        # Overlapping raises with the historical message
        with pytest.raises(ValueError, match="Layer overlap"):
            orch.validate_layer_assignment("c", 3, 9)
