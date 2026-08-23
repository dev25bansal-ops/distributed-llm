"""Regression tests for Medium-severity findings M9, M10, M11, M12, M15.

These exercise the orchestrator batching, autoscaler hysteresis, healer
drain-window + node-targeted reset, KV replication/recovery, cache-coherence
GC + broadcast, and the carbon formula.
"""

import threading
import time

import pytest
import torch

from distllm.core.intelligent_autoscaler import IntelligentAutoscaler, ScalingMetrics
from distllm.core.autonomous_healer import (
    AutonomousHealer,
    GPUResetManager,
    GPUHealthState,
    GPUHeartbeat,
)
from distllm.core.kv_cache_replication import ReplicatedKVCache
from distllm.core.cache_coherence import CacheCoherenceProtocol
from distllm.core.tenant_cost_attribution import (
    TenantCostAttribution,
    RoutingAttribution,
    TenantCostRecord,
)


# ---------------------------------------------------------------------------
# M9: MoE orchestrator must batch tokens per node into ONE RPC, not one RPC
# per (token, expert).
# ---------------------------------------------------------------------------
def test_moe_orchestrator_batches_rpc_per_node():
    import sys
    import types

    from distllm.core.moe_orchestrator import MoERouter, ExpertRegistry

    # The router does ``from distllm.dist.grpc_client import GrpcClientPool``
    # lazily; that module does not exist in this tree, so stub it for the test.
    fake_mod = types.ModuleType("distllm.dist.grpc_client")

    class GrpcClientPool:
        @staticmethod
        def get_client(node_id):
            return clients.get(node_id)

    fake_mod.GrpcClientPool = GrpcClientPool
    sys.modules["distllm.dist.grpc_client"] = fake_mod

    registry = ExpertRegistry()
    router = MoERouter(registry, num_experts=4, top_k=2, hidden_size=8)
    registry.register_expert("node-a", 0, 0)
    registry.register_expert("node-a", 0, 1)
    registry.register_expert("node-b", 0, 2)
    registry.register_expert("node-b", 0, 3)

    class FakeClient:
        def __init__(self):
            self.batch_calls = 0

        def call_expert_forward_batch(self, expert_ids, hidden_states, layer_idx):
            self.batch_calls += 1
            # Return a zero tensor shaped like the stacked input.
            return torch.zeros_like(hidden_states)

    clients = {"node-a": FakeClient(), "node-b": FakeClient()}

    try:
        batch, seq, hidden = 2, 4, 8
        hidden_states = torch.randn(batch, seq, hidden)
        gate = torch.randn(batch, seq, 4)
        out = router.route(hidden_states, gate, layer_idx=0)
    finally:
        sys.modules.pop("distllm.dist.grpc_client", None)

    assert out.shape == (batch, seq, hidden)
    # 2 nodes -> at most 2 batch RPCs (one per node), NOT tokens*top_k (16).
    total_rpc = clients["node-a"].batch_calls + clients["node-b"].batch_calls
    assert total_rpc <= 2, f"expected <=2 batched RPCs per node, got {total_rpc}"
    assert clients["node-a"].batch_calls >= 1
    assert clients["node-b"].batch_calls >= 1


# ---------------------------------------------------------------------------
# M10: autoscaler must not flap +/-1 — hysteresis dead-band + sustained cycles.
# ---------------------------------------------------------------------------
def _metrics(current, util=50.0, pending=0):
    return ScalingMetrics(
        gpu_utilization=util, pending_requests=pending, current_nodes=current
    )


def test_autoscaler_hysteresis_no_flap():
    # tiny band so a 1-node difference is inside the dead-band -> no scale
    asc = IntelligentAutoscaler(
        min_nodes=1, max_nodes=20, scale_up_threshold=0.85,
        scale_down_threshold=0.3, cooldown_seconds=0.0,
        hysteresis_band=2, stable_cycles=3,
    )
    # current=5, reactive wants 6 (util>0.85? no; use pending to force +1)
    m = _metrics(5, util=95.0, pending=30)  # reactive_target -> 6, band 2 -> hold
    d1 = asc.evaluate(m)
    d2 = asc.evaluate(m)
    d3 = asc.evaluate(m)
    # Within dead-band and/or not yet stable -> no scale emitted.
    assert not d1.should_scale and not d2.should_scale and not d3.should_scale, (
        f"autoscaler flapped: {d1.reason} {d2.reason} {d3.reason}"
    )


def test_autoscaler_sustained_direction_scales():
    asc = IntelligentAutoscaler(
        min_nodes=1, max_nodes=20, scale_up_threshold=0.85,
        scale_down_threshold=0.3, cooldown_seconds=0.0,
        hysteresis_band=1, stable_cycles=3,
    )
    # Force sustained scale-up pressure (util 99% every eval).
    m = _metrics(5, util=99.0, pending=40)
    decisions = [asc.evaluate(m) for _ in range(3)]
    # After 3 consecutive same-direction evals, it should commit to a scale.
    assert any(d.should_scale for d in decisions), "expected scale after sustained pressure"


# ---------------------------------------------------------------------------
# M11: healer must keep DRAINING through the drain window, and reset the
# TARGET node (via node_executor), not the local host.
# ---------------------------------------------------------------------------
def test_healer_draining_persists_through_window():
    healer = AutonomousHealer(
        failure_threshold=0.1, recovery_threshold=0.05,
        drain_duration_s=0.05, dry_run=True,
    )
    hb = GPUHeartbeat(node_id="gpu-0", ecc_corrected_rate=200.0)
    healer.record_heartbeat(hb)
    # Force into DRAINING.
    healer._states["gpu-0"] = GPUHealthState.DRAINING
    healer._drain_start.pop("gpu-0", None)

    # First check: stays DRAINING (window not elapsed).
    healer.check_all()
    assert healer._states["gpu-0"] == GPUHealthState.DRAINING, "DRAINING did not persist"

    # Wait out the window, then it should recover (dry_run -> SHADOW).
    time.sleep(0.08)
    healer.check_all()
    assert healer._states["gpu-0"] in (
        GPUHealthState.SHADOW, GPUHealthState.RECOVERING,
    ), f"did not advance past DRAINING: {healer._states['gpu-0']}"


def test_reset_gpu_targets_node_via_executor():
    calls = []

    def fake_executor(node_id, cmd):
        calls.append((node_id, cmd))
        return 0, "name,index\n0, GeForce\n", ""

    mgr = GPUResetManager(dry_run=False, node_executor=fake_executor)
    ok = mgr.reset_gpu("gpu-node-7", device_id=0)
    assert ok
    assert calls, "node_executor was never called"
    assert calls[0][0] == "gpu-node-7", "reset ran on wrong (local) node"


# ---------------------------------------------------------------------------
# M12: real replication to peers + recovery from peers; coherence GC + broadcast.
# ---------------------------------------------------------------------------
def test_kv_replication_copies_to_peers_and_recovers():
    sent = []

    def transport(node_id, req_id, layer_idx, key, value):
        sent.append(node_id)
        return True

    cache = {"stored": None}

    class FakeLocal:
        def update(self, layer_idx, key, value):
            cache["stored"] = (layer_idx, key, value)

    repl = ReplicatedKVCache(
        FakeLocal(), node_id="node-1", replication_factor=2,
        transport=transport, peer_nodes=["node-1", "node-2", "node-3"],
    )
    k = torch.randn(1, 4)
    v = torch.randn(1, 4)
    repl.store("r1", 0, k, v)
    # Replication pushed to at least one peer.
    assert "node-2" in sent or "node-3" in sent, "no peer replication occurred"
    assert "node-1" not in sent, "replicated to self"

    # Now simulate node-1 lost the entry and recovers from a peer.
    peer_store = {"node-2": (k, v)}
    recovered = repl.recover_from_peers(
        "r1", 0,
        fetcher=lambda nid, rid, li: peer_store.get(nid),
    )
    assert recovered, "recover_from_peers failed"
    assert cache["stored"] is not None, "local cache not restored"


def test_coherence_gc_unbounded_and_broadcasts():
    broadcast = []
    proto = CacheCoherenceProtocol(
        "node-x", broadcast_invalidator=lambda h: broadcast.append(h)
    )
    # Many prefixes stored but never invalidated -> would grow unbounded.
    for i in range(50):
        proto.on_store(f"p{i}")
    # Invalidate one and confirm broadcast fires.
    proto.invalidate("p0")
    assert "p0" in broadcast, "invalidation not broadcast to peers"

    # Force old activity to age out by directly setting last_activity in the past.
    old = time.time() - 7200
    for i in range(50):
        proto._last_activity[f"p{i}"] = old
    removed = proto.cleanup_old_entries(max_age_seconds=3600)
    assert removed >= 50, f"GC did not reclaim entries: removed={removed}"
    assert len(proto._vector_clocks) == 0, "vector clocks not GC'd"


# ---------------------------------------------------------------------------
# M15: carbon formula uses intensity * energy_kwh / 1000, not intensity as energy.
# ---------------------------------------------------------------------------
def test_carbon_formula_uses_energy_term():
    attr = TenantCostAttribution()
    attr.record(
        "t1", "q1",
        actual_cost_usd=0.01,
        latency_ms=1000,
        routing=RoutingAttribution(
            compute_source="cloud", carbon_intensity=400.0, energy_kwh=0.5,
        ),
    )
    summary = attr.get_summary("t1")
    # 400 gCO2/kWh * 0.5 kWh / 1000 = 0.2 kg
    assert abs(summary.total_carbon_kg - 0.2) < 1e-9, (
        f"carbon formula wrong: {summary.total_carbon_kg}"
    )

    # Without energy_kwh, fallback uses nominal 0.4 kW * (1000ms/3.6e6 h) = 1.11e-4 kWh
    attr2 = TenantCostAttribution()
    attr2.record(
        "t2", "q2",
        actual_cost_usd=0.01,
        latency_ms=1000,
        routing=RoutingAttribution(compute_source="cloud", carbon_intensity=400.0),
    )
    s2 = attr2.get_summary("t2")
    expected = 400.0 * (0.4 * (1000.0 / 3600000.0)) / 1000.0
    assert abs(s2.total_carbon_kg - expected) < 1e-12
    # The buggy old formula would have been 400*0.001*(1000/3.6e6)=1.1e-4; new is
    # different (uses real energy), so ensure it isn't the buggy value.
    assert abs(s2.total_carbon_kg - 1.1e-4) > 1e-6
