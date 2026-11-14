"""Tests for distllm.dist.redundant."""
import time

import pytest
import torch

from distllm.dist.pipeline.orchestrator import PipelineOrchestrator
from distllm.dist.redundant import RedundantExecutor, StateReplicationEngine
from distllm.errors.types import NodeUnreachableError


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    """Poll *predicate* until true or *timeout* — losing fan-out replicas
    are discarded (not joined), so their calls may land just after the
    winner returns."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


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

    def test_find_redundant_enabled_finds_peers(self):
        """When enabled, redundant peers covering the same layer range ARE
        found (P0 fix: previously the nodes property returned dicts so peers
        were never matched and redundancy was a no-op)."""
        pipeline = PipelineOrchestrator()
        pipeline.register_node("node-a", "10.0.0.1", 50051, 0, 15)
        pipeline.register_node("node-b", "10.0.0.2", 50051, 0, 15)
        pipeline.register_node("node-c", "10.0.0.3", 50051, 16, 31)
        executor = RedundantExecutor(pipeline, redundancy=2)
        results = executor._find_redundant_nodes("node-a")
        assert len(results) == 2
        assert {r[0] for r in results} == {"node-a", "node-b"}

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

    def test_run_pipeline_standard_path_delegates(self, monkeypatch):
        """P0 fix: standard path now delegates correctly (previously passed an
        unsupported draft_tokens arg and raised TypeError)."""
        pipeline = PipelineOrchestrator()
        pipeline.register_node("node-a", "10.0.0.1", 50051, 0, 15)
        executor = RedundantExecutor(pipeline)
        input_ids = torch.zeros((1, 4), dtype=torch.long)

        def fake_forward(host, port, hidden_states, kv_cache, request_id, **kw):
            return hidden_states + 1

        monkeypatch.setattr("distllm.dist.node_client.forward_request", fake_forward)
        out = executor.run_pipeline(input_ids, {}, "req-1")
        assert torch.equal(out, input_ids + 1)

    def test_run_pipeline_redundant_path_fans_out(self, monkeypatch):
        """P0 fix: redundant path fans out to peers and returns the first
        response (previously raised NotImplementedError via stub helpers)."""
        pipeline = PipelineOrchestrator()
        pipeline.register_node("node-a", "10.0.0.1", 50051, 0, 15)
        pipeline.register_node("node-b", "10.0.0.2", 50051, 0, 15)
        executor = RedundantExecutor(pipeline, redundancy=2)
        input_ids = torch.zeros((1, 4), dtype=torch.long)
        calls: list[tuple[str, int]] = []

        def fake_forward(host, port, hidden_states, kv_cache, request_id, **kw):
            calls.append((host, port))
            return hidden_states + 1

        monkeypatch.setattr("distllm.dist.node_client.forward_request", fake_forward)
        out = executor.run_pipeline(input_ids, {}, "req-1")
        assert out is not None
        # Each stage fanned out to both redundant peers.
        assert len(calls) >= 2
        assert {c[0] for c in calls} == {"10.0.0.1", "10.0.0.2"}

    # ── Same-range replicas are one logical stage ───────────────

    def test_run_pipeline_same_range_replicas_apply_layers_once(self, monkeypatch):
        """Regression: replicas covering the same layer range used to run as
        separate sequential stages, applying the same layers twice and
        corrupting the output (input 0 -> 2.0 instead of 1.0)."""
        pipeline = PipelineOrchestrator()
        pipeline.register_node("node-a", "10.0.0.1", 50051, 0, 15)
        pipeline.register_node("node-b", "10.0.0.2", 50051, 0, 15)
        executor = RedundantExecutor(pipeline, redundancy=2)
        input_ids = torch.zeros((1, 4), dtype=torch.long)
        calls: list[str] = []

        def fake_forward(host, port, hidden_states, kv_cache, request_id, **kw):
            calls.append(host)
            return hidden_states + 1

        monkeypatch.setattr("distllm.dist.node_client.forward_request", fake_forward)
        out = executor.run_pipeline(input_ids, {}, "req-1")
        # Exactly one logical stage -> the layers were applied once, even
        # though both replicas were asked.
        assert out[0, 0].item() == 1.0
        assert _wait_for(lambda: sorted(calls) == ["10.0.0.1", "10.0.0.2"])

    def test_run_pipeline_two_ranges_each_redundant(self, monkeypatch):
        """Two layer ranges x two replicas = two stages, each fanned out."""
        pipeline = PipelineOrchestrator()
        pipeline.register_node("a1", "10.0.0.1", 50051, 0, 15)
        pipeline.register_node("a2", "10.0.0.2", 50051, 0, 15)
        pipeline.register_node("b1", "10.0.0.3", 50051, 16, 31)
        pipeline.register_node("b2", "10.0.0.4", 50051, 16, 31)
        executor = RedundantExecutor(pipeline, redundancy=2)
        input_ids = torch.zeros((1, 4), dtype=torch.long)
        hosts: list[str] = []

        def fake_forward(host, port, hidden_states, kv_cache, request_id, **kw):
            hosts.append(host)
            return hidden_states + 1

        monkeypatch.setattr("distllm.dist.node_client.forward_request", fake_forward)
        out = executor.run_pipeline(input_ids, {}, "req-1")

        # Each range applied exactly once -> 1.0 + 1.0 = 2.0, never a
        # duplicated range (3.0/4.0).  All four replicas were asked.
        assert out[0, 0].item() == 2.0
        assert _wait_for(
            lambda: sorted(hosts) == ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"]
        )

    def test_run_pipeline_uses_injected_client(self):
        """Fan-out prefers injected clients (same as the standard path)."""
        import types as _types

        pipeline = PipelineOrchestrator()
        pipeline.register_node("node-a", "10.0.0.1", 50051, 0, 15)
        calls: list[str] = []

        def forward(hidden_states, kv_cache, request_id):
            calls.append("injected")
            return hidden_states + 1

        pipeline._nodes["node-a"].client = _types.SimpleNamespace(forward=forward)
        out = RedundantExecutor(pipeline, redundancy=2).run_pipeline(
            torch.zeros((1, 4), dtype=torch.long), {}, "req-1",
        )
        assert calls == ["injected"]
        assert out[0, 0].item() == 1.0

    def test_run_pipeline_failover_to_healthy_replica(self):
        """A failed replica must not fail the stage while a peer is healthy."""
        import types as _types

        pipeline = PipelineOrchestrator()
        pipeline.register_node("node-a", "10.0.0.1", 50051, 0, 15)
        pipeline.register_node("node-b", "10.0.0.2", 50051, 0, 15)

        def dead(hidden_states, kv_cache, request_id):
            raise ConnectionRefusedError("replica down")

        def alive(hidden_states, kv_cache, request_id):
            return hidden_states + 1

        pipeline._nodes["node-a"].client = _types.SimpleNamespace(forward=dead)
        pipeline._nodes["node-b"].client = _types.SimpleNamespace(forward=alive)
        out = RedundantExecutor(pipeline, redundancy=2, timeout_s=5.0).run_pipeline(
            torch.zeros((1, 4), dtype=torch.long), {}, "req-1",
        )
        assert out[0, 0].item() == 1.0

    def test_run_pipeline_fast_replica_beats_slow_primary(self):
        """The first response wins: a slow primary does not block a fast peer."""
        import types as _types

        pipeline = PipelineOrchestrator()
        pipeline.register_node("node-a", "10.0.0.1", 50051, 0, 15)
        pipeline.register_node("node-b", "10.0.0.2", 50051, 0, 15)

        def slow(hidden_states, kv_cache, request_id):
            time.sleep(0.25)
            return hidden_states + 1

        def fast(hidden_states, kv_cache, request_id):
            return hidden_states + 1

        pipeline._nodes["node-a"].client = _types.SimpleNamespace(forward=slow)
        pipeline._nodes["node-b"].client = _types.SimpleNamespace(forward=fast)
        t0 = time.monotonic()
        out = RedundantExecutor(pipeline, redundancy=2, timeout_s=5.0).run_pipeline(
            torch.zeros((1, 4), dtype=torch.long), {}, "req-1",
        )
        assert time.monotonic() - t0 < 0.2, "slow primary blocked the fast replica"
        assert out[0, 0].item() == 1.0

    def test_run_pipeline_enforces_timeout(self):
        """timeout_s is enforced: all replicas hanging raises, not hangs."""
        import types as _types

        pipeline = PipelineOrchestrator()
        pipeline.register_node("node-a", "10.0.0.1", 50051, 0, 15)
        pipeline.register_node("node-b", "10.0.0.2", 50051, 0, 15)

        def hang(hidden_states, kv_cache, request_id):
            time.sleep(1.0)
            return hidden_states + 1

        pipeline._nodes["node-a"].client = _types.SimpleNamespace(forward=hang)
        pipeline._nodes["node-b"].client = _types.SimpleNamespace(forward=hang)
        t0 = time.monotonic()
        with pytest.raises(NodeUnreachableError):
            RedundantExecutor(pipeline, redundancy=2, timeout_s=0.1).run_pipeline(
                torch.zeros((1, 4), dtype=torch.long), {}, "req-1",
            )
        assert time.monotonic() - t0 < 0.5

    def test_run_pipeline_skips_draining_replica(self):
        """admin drain parity: a draining replica gets no new fan-out work."""
        import types as _types

        class _RM:
            def get_draining_nodes(self):
                return ["node-b"]

        pipeline = PipelineOrchestrator(resource_mgr=_RM())
        pipeline.register_node("node-a", "10.0.0.1", 50051, 0, 15)
        pipeline.register_node("node-b", "10.0.0.2", 50051, 0, 15)
        calls: list[str] = []

        def mk(tag):
            def forward(hidden_states, kv_cache, request_id):
                calls.append(tag)
                return hidden_states + 1
            return forward

        pipeline._nodes["node-a"].client = _types.SimpleNamespace(forward=mk("a"))
        pipeline._nodes["node-b"].client = _types.SimpleNamespace(forward=mk("b"))
        out = RedundantExecutor(pipeline, redundancy=2, timeout_s=2.0).run_pipeline(
            torch.zeros((1, 4), dtype=torch.long), {}, "req-1",
        )
        assert calls == ["a"], "drained replica must not be fanned out to"
        assert out[0, 0].item() == 1.0

    # ── get_node_groups ─────────────────────────────────────────

    def test_get_node_groups_not_enabled(self):
        """When disabled, returns all nodes in order as a single group."""
        pipeline = PipelineOrchestrator()
        pipeline.register_node("node-a", "10.0.0.1", 50051, 0, 15)
        pipeline.register_node("node-b", "10.0.0.2", 50051, 16, 31)
        executor = RedundantExecutor(pipeline)
        groups = executor.get_node_groups()
        assert groups == [["node-a", "node-b"]]

    def test_get_node_groups_enabled_groups_peers(self):
        """P0 fix: enabled mode groups redundant peers covering the same range."""
        pipeline = PipelineOrchestrator()
        pipeline.register_node("node-a", "10.0.0.1", 50051, 0, 15)
        pipeline.register_node("node-b", "10.0.0.2", 50051, 0, 15)
        executor = RedundantExecutor(pipeline, redundancy=2)
        groups = executor.get_node_groups()
        assert len(groups) == 1
        assert set(groups[0]) == {"node-a", "node-b"}


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

    def test_promote_standby_restores_kv_cache(self):
        """P0 fix: promotion restores the replicated KV cache onto the standby
        node (previously raised AttributeError on the nodes-property dicts)."""
        pipeline = PipelineOrchestrator()
        pipeline.register_node("node-a-standby", "10.0.0.2", 50051, 0, 15)
        engine = StateReplicationEngine(pipeline)
        engine.register_standby("node-a", "node-a-standby")
        kv = [(torch.tensor([1.0]), torch.tensor([2.0]))]
        engine.replicate_state("node-a", kv, None, "req-1")
        promoted = engine.promote_standby("node-a")
        assert promoted == "node-a-standby"
        assert pipeline._nodes["node-a-standby"].kv_cache is not None

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
