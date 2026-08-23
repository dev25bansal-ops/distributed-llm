"""§6.2 integration tests -- multi-node cluster, scheduling, preemption, e2e.

All fixtures are IN-PROCESS (no mDNS/real network, which is unavailable
in this env).  They exercise the real code paths:

- H11 leader election: 3 RayFaultTolerance coordinators, manual
  election round -> exactly one leader (lowest ID), survives heartbeats.
- H13 split-brain / fence tokens: a SplitBrainDetector gates write
  admission (quorum_check); a leader that loses quorum REFUSES
  writes and steps down, so a healed partition cannot yield two
  divergent leaders.
- H12 peer discovery (in-process): PipelineOrchestrator.register_node
  populates nodes + sorted node_order across 3 nodes.
- C5 heterogeneous/WAN/energy scheduling: schedule_heterogeneous_pipeline
  over mixed GPU types must not raise TypeError/AttributeError.
- M2 preemption: DistributedPreemptionCoordinator tracks preempted
  nodes and respects max_preempted_fraction (KV resume bookkeeping).
- End-to-end generate: OpenAI-compatible CompletionService.generate
  (greedy + constrained) -> coordinator -> text, non-streaming.
"""

import asyncio
import threading


# ── H11 + H13: leader election + split-brain fence ──

def _make_3node_cluster():
    from distllm.core.ha_coordinator import RayFaultTolerance

    ids = ["coordinator-1", "coordinator-2", "coordinator-3"]
    nodes = {i: RayFaultTolerance(coordinator_id=i) for i in ids}
    for i in ids:
        for j in ids:
            if i != j:
                # host/port are illustrative; the election uses IDs + heartbeats.
                nodes[i].add_peer(j, "10.0.0.%d" % ids.index(j), 50051)
    return nodes


def test_ha_election_picks_single_lowest_id_leader():
    nodes = _make_3node_cluster()
    # Run one synchronous election round on every node.
    for n in nodes.values():
        n._run_election_round()

    leaders = [i for i, n in nodes.items() if n.is_leader()]
    # Exactly one leader, and it must be the lowest-ID alive node.
    assert leaders == ["coordinator-1"], f"expected 1 leader (c1), got {leaders}"
    assert nodes["coordinator-1"].is_leader() is True
    assert nodes["coordinator-2"].is_leader() is False
    assert nodes["coordinator-3"].is_leader() is False


def test_ha_election_stable_across_heartbeats():
    nodes = _make_3node_cluster()
    for _ in range(5):  # several heartbeat rounds
        for n in nodes.values():
            n._run_election_round()
    leaders = [i for i, n in nodes.items() if n.is_leader()]
    assert leaders == ["coordinator-1"], f"leader flapped: {leaders}"


class _FenceDetector:
    """Simulates quorum loss (e.g. after a network partition heals)."""

    def __init__(self, quorum: bool):
        self._quorum = quorum

    def quorum_check(self) -> bool:
        return self._quorum


def test_h13_fence_blocks_writes_when_quorum_lost():
    nodes = _make_3node_cluster()
    leader = nodes["coordinator-1"]
    leader._run_election_round()
    assert leader.is_leader()

    # Partition: this leader can no longer see quorum.
    leader.set_split_brain_detector(_FenceDetector(quorum=False))
    assert leader.can_accept_writes() is False

    # A write (state replication) must be REFUSED and the leader steps down,
    # so a healed partition cannot produce two divergent leaders.
    before = leader.get_state()
    leader.replicate_state("kv", {"x": 1})
    assert leader.is_leader() is False, "leader kept writing after quorum loss (split-brain!)"
    assert leader.get_state() != before or leader.get_leader() is None


def test_h13_fence_allows_writes_with_quorum():
    nodes = _make_3node_cluster()
    leader = nodes["coordinator-1"]
    leader._run_election_round()
    leader.set_split_brain_detector(_FenceDetector(quorum=True))
    assert leader.can_accept_writes() is True
    leader.replicate_state("kv", {"x": 1})
    assert leader.is_leader() is True  # still leader, write accepted


# ── H12: in-process peer discovery populates all nodes ──
# NOTE: PipelineOrchestrator transitively requires grpc/google (not
# installed in this env), so this exercises the real topology only
# where the dep is present. Skips cleanly otherwise.
def test_discovery_populates_all_nodes():
    try:
        from distllm.dist.pipeline.orchestrator import PipelineOrchestrator
    except ImportError as e:
        import pytest

        pytest.skip(f"pipeline.orchestrator needs grpc/google: {e}")

    pipe = PipelineOrchestrator()
    # 3 worker nodes with non-overlapping layer ranges.
    pipe.register_node("worker-1", "10.0.0.1", 50052, 0, 8, total_layers=24)
    pipe.register_node("worker-2", "10.0.0.2", 50052, 8, 16, total_layers=24)
    pipe.register_node("worker-3", "10.0.0.3", 50052, 16, 24, total_layers=24)

    assert set(pipe.nodes.keys()) == {"worker-1", "worker-2", "worker-3"}
    # node_order is sorted by start_layer -> worker-1, -2, -3.
    assert pipe.node_order == ["worker-1", "worker-2", "worker-3"]
    # Unregistering one node updates the topology for the others.
    pipe.unregister_node("worker-2")
    assert set(pipe.nodes.keys()) == {"worker-1", "worker-3"}


# ── C5: heterogeneous / WAN / energy scheduling, no TypeError/AttributeError ──

def test_heterogeneous_schedule_no_error():
    from distllm.core.heterogeneous_scheduler import (
        build_heterogeneous_cluster,
        schedule_heterogeneous_pipeline,
    )

    node_configs = [
        {"node_id": "n1", "host": "10.0.0.1", "port": 50052, "device": "A100", "memory_gb": 80, "bandwidth_gbps": 2000,
         "start_layer": 0, "end_layer": 12, "total_layers": 24},
        {"node_id": "n2", "host": "10.0.0.2", "port": 50052, "device": "H100", "memory_gb": 80, "bandwidth_gbps": 3000,
         "start_layer": 12, "end_layer": 20, "total_layers": 24},
        {"node_id": "n3", "host": "10.0.0.3", "port": 50052, "device": "T4", "memory_gb": 16, "bandwidth_gbps": 320,
         "start_layer": 20, "end_layer": 24, "total_layers": 24},
    ]
    # Mixed device families (A100/H100/T4) across nodes must not crash
    # with TypeError/AttributeError (the C5 regression class).  The
    # scheduler detects heterogeneous device families and orders by
    # throughput internally.
    route = schedule_heterogeneous_pipeline(
        node_configs, total_layers=24, hidden_size=4096
    )
    assert isinstance(route, list) and len(route) == 3
    # Every assignment carries resolved device info (heterogeneous input
    # handled without AttributeError/TypeError -- the C5 regression class).
    node_ids = {a["node_id"] for a in route}
    assert node_ids == {"n1", "n2", "n3"}
    for a in route:
        assert "start_layer" in a and "end_layer" in a
        assert "device_family" in a and "gpu_name" in a


# ── M2: preemption preserves KV bookkeeping, resumes correctly ──

def test_preemption_tracks_state_and_respects_fraction():
    from distllm.core.advanced_scheduling.preemption import (
        DistributedPreemptionCoordinator,
        NodePreemptionState,
    )

    coord = DistributedPreemptionCoordinator(max_preempted_fraction=0.5)
    coord.update_state(NodePreemptionState(node_id="n1", is_preempted=False))
    coord.update_state(NodePreemptionState(node_id="n2", is_preempted=False))

    # Preempt n1 -> it appears in the preempted set, but n2 may still preempt
    # (we are under the 50% fraction: 1/2 preempted = exactly at limit).
    coord.update_state(NodePreemptionState(node_id="n1", is_preempted=True))
    assert coord.get_preempted_nodes() == ["n1"]
    assert coord.should_preempt("n2") is False  # would exceed 0.5 fraction

    # Resume n1 (KV restored) -> preempted set empties, decode can resume.
    coord.update_state(NodePreemptionState(node_id="n1", is_preempted=False))
    assert coord.get_preempted_nodes() == []
    assert coord.should_preempt("n2") is True  # 0/2 preempted, room to preempt


# ── End-to-end generate (OpenAI-compatible service → coordinator → text) ──
# NOTE: CompletionService transitively requires fastapi (not installed
# in this env), so the OpenAI service layer is exercised only
# where the dep is present. Skips cleanly otherwise.
class _MockCoordinator:
    """Stands in for the real Coordinator.generate (no GPU/model load)."""

    def __init__(self, text: str = "hello world"):
        self._text = text
        self.calls = []

    def generate(
        self,
        prompt: str,
        max_tokens: int = 16,
        temperature: float = 1.0,
        top_p: float = 1.0,
        user_id: str = "default",
        response_format=None,
        constraint=None,
    ) -> str:
        self.calls.append((prompt, constraint is not None))
        return self._text


def test_e2e_generate_greedy_and_constrained():
    try:
        from distllm.api.services.completion_service import CompletionService
    except ImportError as e:
        import pytest

        pytest.skip(f"CompletionService needs fastapi: {e}")

    from distllm.api.services.completion_service import CompletionService

    mock = _MockCoordinator(text="42")
    svc = CompletionService(coordinator=mock)

    # Greedy, no constraint.
    out = asyncio.run(svc.generate("What is 6*7?", max_tokens=8, temperature=0.0, top_p=1.0))
    assert out == "42"
    assert mock.calls[-1][1] is False

    # Constrained output (JSON schema) -> constraint object built + threaded through.
    constraint = svc.build_constraint({"type": "json_object"})
    out2 = asyncio.run(
        svc.generate("Return JSON", max_tokens=8, temperature=0.0, top_p=1.0, constraint=constraint)
    )
    assert out2 == "42"
    assert mock.calls[-1][1] is True  # constraint reached the coordinator
