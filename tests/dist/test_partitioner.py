"""Tests for HardwareAwarePartitioner.

Tests use only real objects from the module (no mocks) and do not require
GPU hardware or network connectivity.  All hardware-dependent paths fall
back gracefully to CPU defaults.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from distllm.dist.partition import (
    HardwareAwarePartitioner,
    PartitionSolution,
)


# ---------------------------------------------------------------------------
# Constructor and configuration
# ---------------------------------------------------------------------------


class TestHardwareAwarePartitionerInit:
    """Constructor defaults and configuration."""

    def test_defaults(self) -> None:
        p = HardwareAwarePartitioner()
        assert p._batch_size == 1
        assert p._seq_len == 4096
        assert not p._allow_oom
        assert not p._enable_quant_tuning
        assert p._max_quality_loss == 0.05
        assert p._prefer_speed is False
        assert p._solution is None
        assert p._last_model_name is None
        assert p._gpu_profiles == []
        assert p._topology is None
        assert p._layer_weights == []

    def test_custom_parameters(self) -> None:
        p = HardwareAwarePartitioner(
            batch_size=8,
            seq_len=2048,
            allow_oom=True,
            enable_quant_tuning=True,
            max_quality_loss=0.1,
            prefer_speed=True,
        )
        assert p._batch_size == 8
        assert p._seq_len == 2048
        assert p._allow_oom
        assert p._enable_quant_tuning
        assert p._max_quality_loss == 0.1
        assert p._prefer_speed

    def test_profile_dir_created(self, tmp_path: Path) -> None:
        d = tmp_path / "part_profiles"
        p = HardwareAwarePartitioner(profile_dir=str(d))
        assert Path(p._profile_dir).is_dir()
        assert p._profile_dir.resolve() == d.resolve()

    def test_profile_dir_default(self) -> None:
        p = HardwareAwarePartitioner()
        assert ".distllm" in str(p._profile_dir)
        assert "partitions" in str(p._profile_dir)
        assert p._profile_dir.is_dir()  # created by __init__


# ---------------------------------------------------------------------------
# Config hash (cache invalidation)
# ---------------------------------------------------------------------------


class TestConfigHash:
    """_compute_config_hash behaviour."""

    def test_identical_configs(self) -> None:
        p = HardwareAwarePartitioner()
        h1 = p._compute_config_hash("m", ["a", "b"], 4096, 32, 32, 11008, 32000)
        h2 = p._compute_config_hash("m", ["a", "b"], 4096, 32, 32, 11008, 32000)
        assert h1 == h2

    def test_different_model_name(self) -> None:
        p = HardwareAwarePartitioner()
        h1 = p._compute_config_hash("a", ["x"], 4096, 32, 32, 11008, 32000)
        h2 = p._compute_config_hash("b", ["x"], 4096, 32, 32, 11008, 32000)
        assert h1 != h2

    def test_node_ids_order_independent(self) -> None:
        p = HardwareAwarePartitioner()
        h1 = p._compute_config_hash("m", ["b", "a"], 4096, 32, 32, 11008, 32000)
        h2 = p._compute_config_hash("m", ["a", "b"], 4096, 32, 32, 11008, 32000)
        assert h1 == h2

    def test_none_values(self) -> None:
        p = HardwareAwarePartitioner()
        h = p._compute_config_hash(None, None, 4096, 32, 32, 11008, 32000)
        assert isinstance(h, str)
        assert len(h) == 32  # MD5 hex digest

    def test_hash_changes_with_batch_size(self) -> None:
        p1 = HardwareAwarePartitioner(batch_size=1)
        p2 = HardwareAwarePartitioner(batch_size=8)
        h1 = p1._compute_config_hash("m", None, 4096, 32, 32, 11008, 32000)
        h2 = p2._compute_config_hash("m", None, 4096, 32, 32, 11008, 32000)
        assert h1 != h2

    def test_hash_changes_with_allow_oom(self) -> None:
        p1 = HardwareAwarePartitioner(allow_oom=False)
        p2 = HardwareAwarePartitioner(allow_oom=True)
        h1 = p1._compute_config_hash("m", None, 4096, 32, 32, 11008, 32000)
        h2 = p2._compute_config_hash("m", None, 4096, 32, 32, 11008, 32000)
        assert h1 != h2

    def test_edge_minimal_values(self) -> None:
        p = HardwareAwarePartitioner()
        h = p._compute_config_hash("", [], 1, 0, 1, 1, 1)
        assert isinstance(h, str) and len(h) == 32


# ---------------------------------------------------------------------------
# partition() — the main async entry point
# ---------------------------------------------------------------------------


class TestPartition:
    """End-to-end partitioning workflow."""

    @pytest.mark.asyncio
    async def test_defaults_produce_solution(self) -> None:
        p = HardwareAwarePartitioner()
        sol = await p.partition()
        assert isinstance(sol, PartitionSolution)
        assert sol.num_nodes >= 1

    @pytest.mark.asyncio
    async def test_small_model(self) -> None:
        p = HardwareAwarePartitioner()
        sol = await p.partition(
            hidden_size=256,
            intermediate_size=1024,
            num_layers=4,
            num_heads=4,
            head_dim=64,
            vocab_size=1000,
        )
        assert isinstance(sol, PartitionSolution)
        assert sol.num_nodes >= 1

    @pytest.mark.asyncio
    async def test_single_node(self) -> None:
        p = HardwareAwarePartitioner()
        sol = await p.partition(
            node_ids=["n0"],
            num_layers=6,
            hidden_size=512,
            intermediate_size=2048,
            num_heads=8,
            head_dim=64,
        )
        assert isinstance(sol, PartitionSolution)
        assert sol.num_nodes == 1
        # estimate_layer_weights adds embed + num_layers + lm_head = 8 total
        assert sol.coverage[0] == 0

    @pytest.mark.asyncio
    async def test_two_nodes(self) -> None:
        p = HardwareAwarePartitioner()
        sol = await p.partition(
            node_ids=["n0", "n1"],
            num_layers=12,
            hidden_size=512,
            intermediate_size=2048,
            num_heads=8,
            head_dim=64,
        )
        assert isinstance(sol, PartitionSolution)
        assert sol.num_nodes >= 1
        # estimate_layer_weights: embed + 12 transformer + lm_head = 14 total
        assert sol.coverage[0] == 0
        assert sol.coverage[1] == 14

    @pytest.mark.asyncio
    async def test_model_name_passed_through(self) -> None:
        p = HardwareAwarePartitioner()
        sol = await p.partition(model_name="my-model")
        assert isinstance(sol, PartitionSolution)
        assert p._last_model_name == "my-model"

    @pytest.mark.asyncio
    async def test_cache_reuses_same_object(self) -> None:
        p = HardwareAwarePartitioner()
        sol1 = await p.partition(model_name="cached")
        sol2 = await p.partition(model_name="cached")
        assert sol1 is sol2

    @pytest.mark.asyncio
    async def test_cache_invalidated_by_different_config(self) -> None:
        p = HardwareAwarePartitioner()
        sol1 = await p.partition(model_name="test", hidden_size=4096)
        sol2 = await p.partition(model_name="test", hidden_size=2048)
        assert sol1 is not sol2

    @pytest.mark.asyncio
    async def test_custom_batch_and_seq_len(self) -> None:
        p = HardwareAwarePartitioner(batch_size=4, seq_len=1024)
        sol = await p.partition(
            num_layers=6,
            hidden_size=512,
            intermediate_size=2048,
        )
        assert isinstance(sol, PartitionSolution)

    @pytest.mark.asyncio
    async def test_allow_oom_flag(self) -> None:
        p = HardwareAwarePartitioner(allow_oom=True)
        sol = await p.partition(num_layers=6)
        assert isinstance(sol, PartitionSolution)

    @pytest.mark.asyncio
    async def test_with_gpu_counts(self) -> None:
        p = HardwareAwarePartitioner()
        sol = await p.partition(
            node_ids=["n0", "n1"],
            gpu_counts={"n0": 2, "n1": 1},
            num_layers=12,
            hidden_size=512,
            intermediate_size=2048,
            num_heads=8,
            head_dim=64,
        )
        assert isinstance(sol, PartitionSolution)

    @pytest.mark.asyncio
    async def test_with_hostnames(self) -> None:
        """Same-hostname nodes are treated as intra-node (fast path)."""
        p = HardwareAwarePartitioner()
        sol = await p.partition(
            node_ids=["n0", "n1"],
            hostnames={"n0": "localhost", "n1": "localhost"},
            num_layers=8,
            hidden_size=256,
            intermediate_size=1024,
            num_heads=4,
            head_dim=64,
        )
        assert isinstance(sol, PartitionSolution)

    @pytest.mark.asyncio
    async def test_solution_fields(self) -> None:
        p = HardwareAwarePartitioner()
        sol = await p.partition(
            node_ids=["n0", "n1"],
            num_layers=12,
            hidden_size=512,
            intermediate_size=2048,
            num_heads=8,
            head_dim=64,
        )
        assert sol.max_node_time_ms >= 0
        assert sol.estimated_throughput_tok_s >= 0
        assert sol.pipeline_latency_ms >= 0
        assert isinstance(sol.explanation, str)
        assert sol.num_oom_nodes >= 0

    @pytest.mark.asyncio
    async def test_empty_node_ids_uses_default(self) -> None:
        """Empty node_ids list should fall back to default naming."""
        p = HardwareAwarePartitioner()
        sol = await p.partition(node_ids=[])
        assert isinstance(sol, PartitionSolution)
        assert sol.num_nodes >= 1


# ---------------------------------------------------------------------------
# solution() / get_layer_assignments() / get_node_summaries()
# ---------------------------------------------------------------------------


class TestSolutionAccessors:
    """Methods that depend on an existing PartitionSolution."""

    def test_solution_is_none_before_partition(self) -> None:
        p = HardwareAwarePartitioner()
        assert p.solution() is None

    @pytest.mark.asyncio
    async def test_solution_after_partition(self) -> None:
        p = HardwareAwarePartitioner()
        sol = await p.partition()
        assert p.solution() is sol

    def test_layer_assignments_none_before(self) -> None:
        p = HardwareAwarePartitioner()
        assert p.get_layer_assignments() is None

    @pytest.mark.asyncio
    async def test_layer_assignments_after_partition(self) -> None:
        p = HardwareAwarePartitioner()
        await p.partition(
            num_layers=8,
            hidden_size=512,
            intermediate_size=2048,
            num_heads=8,
            head_dim=64,
        )
        assigns = p.get_layer_assignments()
        assert assigns is not None
        assert len(assigns) > 0
        # First layer should be index 0
        assert assigns[0]["layer_id"] == 0
        for a in assigns:
            assert "layer_id" in a
            assert "node_id" in a
            assert "layer_type" in a

    def test_node_summaries_none_before(self) -> None:
        p = HardwareAwarePartitioner()
        assert p.get_node_summaries() is None

    @pytest.mark.asyncio
    async def test_node_summaries_after_partition(self) -> None:
        p = HardwareAwarePartitioner()
        await p.partition(
            num_layers=8,
            hidden_size=512,
            intermediate_size=2048,
            num_heads=8,
            head_dim=64,
        )
        summaries = p.get_node_summaries()
        assert summaries is not None
        assert len(summaries) > 0
        required_keys = {
            "node_id", "layers", "num_layers",
            "compute_time_ms", "fits_in_memory",
        }
        for s in summaries:
            assert required_keys.issubset(s.keys())


# ---------------------------------------------------------------------------
# compare_to_baselines()
# ---------------------------------------------------------------------------


class TestCompareBaselines:
    """Strategy comparison utility."""

    def test_none_before_partition(self) -> None:
        p = HardwareAwarePartitioner()
        assert p.compare_to_baselines() is None

    @pytest.mark.asyncio
    async def test_returns_comparison(self) -> None:
        p = HardwareAwarePartitioner()
        await p.partition(
            node_ids=["n0", "n1"],
            num_layers=12,
            hidden_size=512,
            intermediate_size=2048,
            num_heads=8,
            head_dim=64,
        )
        result = p.compare_to_baselines()
        assert result is not None
        assert "dp_minimax" in result
        assert "equal_split" in result
        assert "proportional_split" in result
        assert "improvement_over_equal" in result

    @pytest.mark.asyncio
    async def test_baseline_has_latency_and_throughput(self) -> None:
        p = HardwareAwarePartitioner()
        await p.partition(
            node_ids=["n0", "n1"],
            num_layers=12,
            hidden_size=512,
            intermediate_size=2048,
            num_heads=8,
            head_dim=64,
        )
        result = p.compare_to_baselines()
        for key in ("dp_minimax", "equal_split", "proportional_split"):
            entry = result[key]
            assert "max_latency_ms" in entry
            assert entry["max_latency_ms"] >= 0


# ---------------------------------------------------------------------------
# summary() string output
# ---------------------------------------------------------------------------


class TestSummary:
    """Human-readable summary."""

    def test_before_partition(self) -> None:
        p = HardwareAwarePartitioner()
        text = p.summary()
        assert "HardwareAwarePartitioner" in text
        assert "No partition computed" in text

    @pytest.mark.asyncio
    async def test_after_partition(self) -> None:
        p = HardwareAwarePartitioner()
        await p.partition(num_layers=6)
        text = p.summary()
        assert "HardwareAwarePartitioner" in text
        assert "Partition" in text


# ---------------------------------------------------------------------------
# Persistence: save / load partition plan JSON
# ---------------------------------------------------------------------------


class TestPersistence:
    """Save (via _save_plan) and load_plan."""

    def test_load_nonexistent_model(self, tmp_path: Path) -> None:
        p = HardwareAwarePartitioner(profile_dir=str(tmp_path))
        assert p.load_plan("no_such_model") is None

    @pytest.mark.asyncio
    async def test_save_and_reload(self, tmp_path: Path) -> None:
        p = HardwareAwarePartitioner(profile_dir=str(tmp_path))
        await p.partition(
            model_name="saved_model",
            num_layers=8,
            hidden_size=512,
            intermediate_size=2048,
            num_heads=8,
            head_dim=64,
        )
        loaded = p.load_plan("saved_model")
        assert loaded is not None
        assert loaded["model"] == "saved_model"
        assert "solution" in loaded
        assert "assignments" in loaded["solution"]
        assert len(loaded["solution"]["assignments"]) > 0

    @pytest.mark.asyncio
    async def test_save_with_model_name_slash(self, tmp_path: Path) -> None:
        """Model names with '/' get sanitized to '_' in the filename."""
        p = HardwareAwarePartitioner(profile_dir=str(tmp_path))
        await p.partition(
            model_name="org/model",
            num_layers=4,
            hidden_size=256,
            intermediate_size=1024,
            num_heads=4,
            head_dim=64,
        )
        loaded = p.load_plan("org/model")
        assert loaded is not None
        assert loaded["model"] == "org/model"


# ---------------------------------------------------------------------------
# Edge cases and boundary values
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary and unusual inputs."""

    @pytest.mark.asyncio
    async def test_zero_layers(self) -> None:
        p = HardwareAwarePartitioner()
        sol = await p.partition(
            num_layers=0,
            hidden_size=512,
            intermediate_size=2048,
        )
        assert isinstance(sol, PartitionSolution)

    @pytest.mark.asyncio
    async def test_minimal_vocab(self) -> None:
        p = HardwareAwarePartitioner()
        sol = await p.partition(
            num_layers=2,
            hidden_size=64,
            intermediate_size=256,
            num_heads=2,
            head_dim=32,
            vocab_size=1,
        )
        assert isinstance(sol, PartitionSolution)

    @pytest.mark.asyncio
    async def test_many_nodes_versus_few_layers(self) -> None:
        """More nodes than layers should be handled gracefully."""
        p = HardwareAwarePartitioner()
        sol = await p.partition(
            node_ids=[f"n{i}" for i in range(10)],
            num_layers=3,
            hidden_size=256,
            intermediate_size=512,
            num_heads=2,
            head_dim=32,
        )
        assert isinstance(sol, PartitionSolution)

    @pytest.mark.asyncio
    async def test_large_vocab_size(self) -> None:
        p = HardwareAwarePartitioner()
        sol = await p.partition(
            num_layers=4,
            hidden_size=512,
            intermediate_size=2048,
            num_heads=8,
            vocab_size=1_000_000,
        )
        assert isinstance(sol, PartitionSolution)

    @pytest.mark.asyncio
    async def test_extreme_head_dim(self) -> None:
        p = HardwareAwarePartitioner()
        sol = await p.partition(
            num_layers=4,
            hidden_size=1024,
            intermediate_size=4096,
            num_heads=1,
            head_dim=256,
        )
        assert isinstance(sol, PartitionSolution)

    @pytest.mark.asyncio
    async def test_no_model_name_saves_as_unknown(self, tmp_path: Path) -> None:
        p = HardwareAwarePartitioner(profile_dir=str(tmp_path))
        await p.partition(
            model_name=None,
            num_layers=4,
            hidden_size=256,
            intermediate_size=1024,
            num_heads=4,
            head_dim=64,
        )
        loaded = p.load_plan("unknown")
        assert loaded is not None
        assert loaded["model"] is None

    @pytest.mark.asyncio
    async def test_quant_tuning_defaults_off(self) -> None:
        """Without enable_quant_tuning, quant plan should be None."""
        p = HardwareAwarePartitioner()
        sol = await p.partition(
            num_layers=6,
            hidden_size=256,
            intermediate_size=1024,
        )
        assert sol.quant_plan is None

    @pytest.mark.asyncio
    async def test_no_topology_links_single_node(self) -> None:
        """Single-node partition with no links should still produce a solution."""
        p = HardwareAwarePartitioner()
        sol = await p.partition(
            node_ids=["single"],
            num_layers=6,
            hidden_size=256,
            intermediate_size=1024,
            num_heads=4,
            head_dim=32,
        )
        assert isinstance(sol, PartitionSolution)
        assert sol.num_nodes == 1
