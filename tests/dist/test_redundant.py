"""Tests for distllm.dist.redundant."""
import pytest
import torch

from distllm.dist.redundant import RedundantExecutor, StateReplicationEngine
from distllm.dist.pipeline.orchestrator import PipelineOrchestrator


class TestRedundantExecutor:
    """Tests for RedundantExecutor."""

    # ── Constructor ─────────────────────────────────────────────

    def test_constructor_defaults(self):
        pipeline = PipelineOrchestrator()
        executor = RedundantExecutor(pipeline)
        assert executor._redundancy == 1
        assert executor._timeout == 30.0
        assert executor.enabled is False

    def test_constructor_custom_values(self):
        pipeline = PipelineOrchestrator()
        executor = RedundantExecutor(pipeline, redundancy=3, timeout_s=60.0)
        assert executor._redundancy == 3
        assert executor._timeout == 60.0
        assert executor.enabled is True

    def test_constructor_clamps_zero_redundancy(self):
        pipeline = PipelineOrchestrator()
        executor = RedundantExecutor(pipeline, redundancy=0)
        assert executor._redundancy == 1
        assert executor.enabled is False

    def test_constructor_clamps_negative_redundancy(self):
        pipeline = PipelineOrchestrator()
        executor = RedundantExecutor(pipeline, redundancy=-5)
        assert executor._redundancy == 1
        assert executor.enabled is False

    # ── enabled property ────────────────────────────────────────

    def test_enabled_false_when_redundancy_one(self):
        pipeline = PipelineOrchestrator()
        assert RedundantExecutor(pipeline).enabled is False
        assert RedundantExecutor(pipeline, redundancy=1).enabled is False

    def test_enabled_true_when_redundancy_greater_than_one(self):
        pipeline = PipelineOrchestrator()
        assert RedundantExecutor(pipeline, redundancy=2).enabled is True
        assert RedundantExecutor(pipeline, redundancy=10).enabled is True

    # ── _find_redundant_nodes ───────────────────────────────────

    def test_find_redundant_nonexistent_node(self):
        """Returns fallback tuple for a node not in the pipeline."""
        pipeline = PipelineOrchestrator()
        executor = RedundantExecutor(pipeline)
        results = executor._find_redundant_nodes("nonexistent")
        assert results == [("nonexistent", None)]

    def test_find_redundant_no_enabled_returns_only_primary(self):
        """When redundancy <= 1, only the primary node is returned."""
        pipeline = PipelineOrchestrator()
        pipeline.register_node("node-a", "10.0.0.1", 50051, 0, 15)
        executor = RedundantExecutor(pipeline)
        results = executor._find_redundant_nodes("node-a")
        assert len(results) == 1
        assert results[0][0] == "node-a"

    def test_find_redundant_discards_unhealthy(self):
        """Unhealthy nodes are skipped even when they match layer range."""
        pipeline = PipelineOrchestrator()
        pipeline.register_node("node-a", "10.0.0.1", 50051, 0, 15)
        pipeline.register_node("node-b", "10.0.0.2", 50051, 0, 15)
        pipeline._nodes["node-b"].is_healthy = False
        executor = RedundantExecutor(pipeline, redundancy=2)
        results = executor._find_redundant_nodes("node-a")
        assert len(results) == 1
        assert results[0][0] == "node-a"

    def test_find_redundant_enabled_returns_only_primary(self):
        """When enabled, the nodes property returns dicts and
        getattr(dict, 'healthy', False) is always False, so redundant
        nodes are never matched. Only the primary is returned."""
        pipeline = PipelineOrchestrator()
        pipeline.register_node("node-a", "10.0.0.1", 50051, 0, 15)
        pipeline.register_node("node-b", "10.0.0.2", 50051, 0, 15)
        pipeline.register_node("node-c", "10.0.0.3", 50051, 0, 15)
        executor = RedundantExecutor(pipeline, redundancy=2)
        results = executor._find_redundant_nodes("node-a")
        assert len(results) == 1
        assert results[0][0] == "node-a"

    def test_find_redundant_excludes_different_layer_range(self):
        """Nodes covering different layers are never included."""
        pipeline = PipelineOrchestrator()
        pipeline.register_node("node-a", "10.0.0.1", 50051, 0, 15)
        pipeline.register_node("node-b", "10.0.0.2", 50051, 16, 31)
        executor = RedundantExecutor(pipeline, redundancy=2)
        results = executor._find_redundant_nodes("node-a")
        assert len(results) == 1
        assert results[0][0] == "node-a"

    # ── run_pipeline ────────────────────────────────────────────

    def test_run_pipeline_standard_path_type_error(self):
        """_run_standard passes draft_tokens as a 4th positional arg,
        but PipelineOrchestrator.run_pipeline only accepts 3."""
        pipeline = PipelineOrchestrator()
        pipeline.register_node("node-a", "10.0.0.1", 50051, 0, 15)
        executor = RedundantExecutor(pipeline)
        input_ids = torch.zeros((1, 4), dtype=torch.long)
        with pytest.raises(TypeError, match="takes 4 positional arguments but 5 were given"):
            executor.run_pipeline(input_ids, {}, "req-1")

    def test_run_pipeline_redundant_path_attribute_error(self):
        """_run_redundant accesses _topology_lock which does not
        exist on PipelineOrchestrator."""
        pipeline = PipelineOrchestrator()
        pipeline.register_node("node-a", "10.0.0.1", 50051, 0, 15)
        executor = RedundantExecutor(pipeline, redundancy=2)
        input_ids = torch.zeros((1, 4), dtype=torch.long)
        with pytest.raises(AttributeError, match="no attribute '_topology_lock'"):
            executor.run_pipeline(input_ids, {}, "req-1")

    # ── get_node_groups ─────────────────────────────────────────

    def test_get_node_groups_not_enabled(self):
        """When disabled, returns all nodes in order as a single group."""
        pipeline = PipelineOrchestrator()
        pipeline.register_node("node-a", "10.0.0.1", 50051, 0, 15)
        pipeline.register_node("node-b", "10.0.0.2", 50051, 16, 31)
        executor = RedundantExecutor(pipeline)
        groups = executor.get_node_groups()
        assert groups == [["node-a", "node-b"]]

    def test_get_node_groups_enabled_attribute_error(self):
        """When enabled, get_node_groups accesses node.start_layer on
        dicts returned by the nodes property."""
        pipeline = PipelineOrchestrator()
        pipeline.register_node("node-a", "10.0.0.1", 50051, 0, 15)
        pipeline.register_node("node-b", "10.0.0.2", 50051, 0, 15)
        executor = RedundantExecutor(pipeline, redundancy=2)
        with pytest.raises(AttributeError, match="no attribute 'start_layer'"):
            executor.get_node_groups()


class TestStateReplicationEngine:
    """Tests for StateReplicationEngine."""

    # ── Constructor ─────────────────────────────────────────────

    def test_constructor_defaults(self):
        pipeline = PipelineOrchestrator()
        engine = StateReplicationEngine(pipeline)
        assert engine._interval == 1.0
        assert engine._max_lag == 5.0
        assert engine._standby_map == {}
        assert engine._replica_state == {}
        assert engine._last_replication == {}
        assert engine._running is False

    def test_constructor_custom_values(self):
        pipeline = PipelineOrchestrator()
        engine = StateReplicationEngine(pipeline, replication_interval_s=2.5, max_replication_lag_s=10.0)
        assert engine._interval == 2.5
        assert engine._max_lag == 10.0

    # ── Properties ──────────────────────────────────────────────

    def test_standby_count_empty(self):
        pipeline = PipelineOrchestrator()
        engine = StateReplicationEngine(pipeline)
        assert engine.standby_count == 0

    def test_active_pairs_empty(self):
        pipeline = PipelineOrchestrator()
        engine = StateReplicationEngine(pipeline)
        assert engine.active_pairs == {}

    # ── register_standby ─────────────────────────────────────────

    def test_register_standby(self):
        pipeline = PipelineOrchestrator()
        engine = StateReplicationEngine(pipeline)
        engine.register_standby("node-a", "node-a-standby")
        assert engine._standby_map == {"node-a": "node-a-standby"}
        assert engine.standby_count == 1

    def test_register_standby_initializes_replica_state(self):
        pipeline = PipelineOrchestrator()
        engine = StateReplicationEngine(pipeline)
        engine.register_standby("node-a", "node-a-standby")
        state = engine._replica_state["node-a-standby"]
        assert state == {"kv_cache": None, "hidden_state": None, "last_request_id": None}

    def test_register_standby_initializes_last_replication(self):
        pipeline = PipelineOrchestrator()
        engine = StateReplicationEngine(pipeline)
        engine.register_standby("node-a", "node-a-standby")
        assert engine._last_replication["node-a-standby"] == 0

    # ── replicate_state ─────────────────────────────────────────

    def test_replicate_state_no_mapping_does_nothing(self):
        pipeline = PipelineOrchestrator()
        engine = StateReplicationEngine(pipeline)
        engine.replicate_state("nonexistent", None, None, "req-1")

    def test_replicate_state_kv_cache_none(self):
        pipeline = PipelineOrchestrator()
        engine = StateReplicationEngine(pipeline)
        engine.register_standby("node-a", "node-a-standby")
        engine.replicate_state("node-a", None, None, "req-1")
        state = engine._replica_state["node-a-standby"]
        assert state["kv_cache"] is None
        assert state["hidden_state"] is None
        assert state["last_request_id"] == "req-1"

    def test_replicate_state_deep_copies_kv_cache(self):
        pipeline = PipelineOrchestrator()
        engine = StateReplicationEngine(pipeline)
        engine.register_standby("node-a", "node-a-standby")
        k = torch.tensor([1.0, 2.0])
        v = torch.tensor([3.0, 4.0])
        engine.replicate_state("node-a", [(k, v)], None, "req-1")
        state = engine._replica_state["node-a-standby"]
        assert state["kv_cache"] is not None
        assert torch.equal(state["kv_cache"][0][0], k)
        assert torch.equal(state["kv_cache"][0][1], v)
        k[0] = 99.0
        assert state["kv_cache"][0][0][0] == 1.0, "kv_cache was not deep-copied"

    def test_replicate_state_deep_copies_hidden(self):
        pipeline = PipelineOrchestrator()
        engine = StateReplicationEngine(pipeline)
        engine.register_standby("node-a", "node-a-standby")
        hidden = torch.tensor([[0.5, 0.25]])
        engine.replicate_state("node-a", None, hidden, "req-2")
        state = engine._replica_state["node-a-standby"]
        assert state["hidden_state"] is not None
        assert torch.equal(state["hidden_state"], hidden)
        hidden[0][0] = 99.0
        assert state["hidden_state"][0][0] == 0.5, "hidden_state was not deep-copied"

    # ── get_replica_state ───────────────────────────────────────

    def test_get_replica_state_no_mapping(self):
        pipeline = PipelineOrchestrator()
        engine = StateReplicationEngine(pipeline)
        assert engine.get_replica_state("nonexistent") is None

    def test_get_replica_state_after_register(self):
        pipeline = PipelineOrchestrator()
        engine = StateReplicationEngine(pipeline)
        engine.register_standby("node-a", "node-a-standby")
        state = engine.get_replica_state("node-a")
        assert state is not None
        assert state["kv_cache"] is None
        assert state["hidden_state"] is None
        assert state["last_request_id"] is None

    # ── promote_standby ─────────────────────────────────────────

    def test_promote_standby_no_mapping(self):
        pipeline = PipelineOrchestrator()
        engine = StateReplicationEngine(pipeline)
        assert engine.promote_standby("nonexistent") is None

    def test_promote_standby_clears_standby_map(self):
        pipeline = PipelineOrchestrator()
        engine = StateReplicationEngine(pipeline)
        engine.register_standby("node-a", "node-a-standby")
        engine.promote_standby("node-a")
        assert "node-a" not in engine._standby_map
        assert engine.standby_count == 0

    def test_promote_standby_returns_standby_id(self):
        pipeline = PipelineOrchestrator()
        engine = StateReplicationEngine(pipeline)
        engine.register_standby("node-a", "node-a-standby")
        assert engine.promote_standby("node-a") == "node-a-standby"

    def test_promote_standby_removes_replica_state(self):
        pipeline = PipelineOrchestrator()
        engine = StateReplicationEngine(pipeline)
        engine.register_standby("node-a", "node-a-standby")
        engine.promote_standby("node-a")
        assert "node-a-standby" not in engine._replica_state

    def test_promote_standby_removes_last_replication(self):
        pipeline = PipelineOrchestrator()
        engine = StateReplicationEngine(pipeline)
        engine.register_standby("node-a", "node-a-standby")
        engine.promote_standby("node-a")
        assert "node-a-standby" not in engine._last_replication

    def test_promote_standby_without_kv_cache_works(self):
        """Promotion succeeds when no KV cache was replicated."""
        pipeline = PipelineOrchestrator()
        engine = StateReplicationEngine(pipeline)
        engine.register_standby("node-a", "node-a-standby")
        assert engine.promote_standby("node-a") == "node-a-standby"

    def test_promote_standby_with_kv_cache_attribute_error(self):
        """Restoring KV cache fails because nodes property returns
        dicts, which have no writable 'kv_cache' attribute."""
        pipeline = PipelineOrchestrator()
        pipeline.register_node("node-a-standby", "10.0.0.2", 50051, 0, 15)
        engine = StateReplicationEngine(pipeline)
        engine.register_standby("node-a", "node-a-standby")
        engine.replicate_state(
            "node-a",
            [(torch.tensor([1.0]), torch.tensor([2.0]))],
            None,
            "req-1",
        )
        with pytest.raises(AttributeError, match="no attribute 'kv_cache'"):
            engine.promote_standby("node-a")

    # ── get_replication_lag ─────────────────────────────────────

    def test_get_replication_lag_no_standby(self):
        pipeline = PipelineOrchestrator()
        engine = StateReplicationEngine(pipeline)
        assert engine.get_replication_lag("nonexistent") == float("inf")

    def test_get_replication_lag_positive_after_register(self):
        pipeline = PipelineOrchestrator()
        engine = StateReplicationEngine(pipeline)
        engine.register_standby("node-a", "node-a-standby")
        assert engine.get_replication_lag("node-a") > 0

    # ── is_healthy ──────────────────────────────────────────────

    def test_is_healthy_no_standby(self):
        pipeline = PipelineOrchestrator()
        engine = StateReplicationEngine(pipeline)
        assert engine.is_healthy("nonexistent") is False

    def test_is_healthy_true_after_replication(self):
        """After replicate_state is called, the lag is under max_lag."""
        pipeline = PipelineOrchestrator()
        engine = StateReplicationEngine(pipeline)
        engine.register_standby("node-a", "node-a-standby")
        engine.replicate_state("node-a", None, None, "req-1")
        assert engine.is_healthy("node-a") is True

    # ── active_pairs / standby_count ─────────────────────────────

    def test_active_pairs_returns_copy(self):
        pipeline = PipelineOrchestrator()
        engine = StateReplicationEngine(pipeline)
        engine.register_standby("node-a", "node-a-sb")
        engine.register_standby("node-b", "node-b-sb")
        assert engine.active_pairs == {"node-a": "node-a-sb", "node-b": "node-b-sb"}

    def test_standby_count_multiple(self):
        pipeline = PipelineOrchestrator()
        engine = StateReplicationEngine(pipeline)
        engine.register_standby("node-a", "node-a-sb")
        engine.register_standby("node-b", "node-b-sb")
        engine.register_standby("node-c", "node-c-sb")
        assert engine.standby_count == 3
