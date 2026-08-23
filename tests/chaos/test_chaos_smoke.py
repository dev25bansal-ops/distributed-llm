"""Fast chaos smoke test — gating subset for push:main.

This is the *fast* chaos smoke that CI runs as a required gate on
``push:main`` (see ``.github/workflows/ci.yml`` → ``chaos-gate``). It
exercises the two highest-signal crash-recovery scenarios without the
~60s time budget of the full chaos suite:

1. **single-node-kill** — a worker drops out of the cluster (crash), the
   coordinator must mark it unhealthy; after the node recovers it must be
   marked healthy again.
2. **network-partition** — the coordinator is partitioned from a worker
   (cannot communicate); after the partition heals, communication must be
   restored.

The scenarios use the same in-process stub clients as
``tests/chaos/test_node_failure.py`` (no real gRPC / Docker), so the
whole file runs in well under a minute and cannot hang a push gate.

Run directly::

    pytest -v -m chaos tests/chaos/test_chaos_smoke.py
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from distllm.core.coordinator import Coordinator
from distllm.core.resource_manager import NodeRegistration
from tests.chaos.chaos_infrastructure import NetworkPartitionSimulator

pytestmark = [
    pytest.mark.chaos,
    pytest.mark.timeout(60),
]


# ---------------------------------------------------------------------------
# In-process stubs (mirrors tests/chaos/test_node_failure.py — no real network)
# ---------------------------------------------------------------------------
class _StubTokenizer:
    def __init__(self):
        self.vocab_size = 1000
        self.eos_token_id = 0
        self.pad_token_id = 0

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return [1, 2, 3]

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        return "test output"


class _StubForwardResponse:
    success = True
    error_message = ""


class _StubHealthResult:
    healthy = True
    memory_used = 1024
    memory_total = 8192


class _StubNodeClient:
    """Node client that simulates a crash via a toggle."""

    def __init__(self, host: str, port: int, **kwargs: Any):
        self.host = host
        self.port = port
        self.call_count = 0
        self._crashing = False
        self._stub = _StubStub()

    def trigger_crash(self) -> None:
        self._crashing = True

    def recover(self) -> None:
        self._crashing = False
        self.call_count = 0

    def forward(self, request: Any, timeout: Any = None) -> Any:
        self.call_count += 1
        if self._crashing:
            raise ConnectionError(f"Node {self.host}:{self.port} is unreachable")
        return _StubForwardResponse()

    def health_check(self, timeout: Any = None) -> Any:
        self.call_count += 1
        if self._crashing:
            raise ConnectionError(f"Node {self.host}:{self.port} is unreachable")
        return _StubHealthResult()


class _StubStub:
    def __init__(self):
        self.HealthCheck = _StubHealthCheck()


class _StubHealthCheck:
    def __init__(self, side_effect: Any = None):
        self.side_effect = side_effect

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.side_effect:
            raise self.side_effect
        return _StubHealthResult()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_coordinator_with_fake_nodes(num_nodes: int = 2) -> Coordinator:
    """Create a coordinator wired to in-process stub node clients."""
    coord = Coordinator(
        model_name="test-model",
        dtype="float32",
    )
    # Bypass model loading.
    coord.tokenizer = _StubTokenizer()

    for i in range(num_nodes):
        client = _StubNodeClient("localhost", 50051 + i)
        reg = NodeRegistration(
            node_id=f"node-{i}",
            host="localhost",
            port=50051 + i,
            start_layer=i * 6,
            end_layer=(i + 1) * 6 - 1,
        )
        reg.client = client
        coord.nodes[f"node-{i}"] = reg
        coord.node_order.append(f"node-{i}")
    return coord


# ---------------------------------------------------------------------------
# Scenario 1 — single-node-kill
# ---------------------------------------------------------------------------
class TestSingleNodeKillSmoke:
    """Verify crash detection + recovery for a single worker node."""

    def test_node_kill_marks_unhealthy(self):
        coord = _make_coordinator_with_fake_nodes(2)
        # Kill node-1.
        coord.nodes["node-1"].client.trigger_crash()

        health = coord.health_check()

        assert health["node-0"]["healthy"] is True
        assert health["node-1"]["healthy"] is False

    def test_node_recovery_after_kill(self):
        coord = _make_coordinator_with_fake_nodes(2)
        coord.nodes["node-1"].client.trigger_crash()
        assert coord.health_check()["node-1"]["healthy"] is False

        # Node comes back online.
        coord.nodes["node-1"].client.recover()
        assert coord.health_check()["node-1"]["healthy"] is True

    def test_forward_fails_during_kill_then_recovers(self):
        coord = _make_coordinator_with_fake_nodes(2)
        coord.nodes["node-1"].client.trigger_crash()
        with pytest.raises(ConnectionError):
            coord.nodes["node-1"].client.forward("dummy", timeout=5)

        coord.nodes["node-1"].client.recover()
        assert coord.nodes["node-1"].client.forward("dummy").success is True


# ---------------------------------------------------------------------------
# Scenario 2 — network-partition
# ---------------------------------------------------------------------------
class TestNetworkPartitionSmoke:
    """Verify partition blocks comms and healing restores it."""

    def test_partition_blocks_communication(self):
        simulator = NetworkPartitionSimulator()
        simulator.create_partition(["coordinator", "node-0"], ["node-1"])

        assert simulator.can_communicate("coordinator", "node-0") is True
        assert simulator.can_communicate("coordinator", "node-1") is False
        assert simulator.is_active is True

    def test_heal_restores_communication(self):
        simulator = NetworkPartitionSimulator()
        simulator.create_partition(["coordinator", "node-0"], ["node-1"])
        assert simulator.can_communicate("coordinator", "node-1") is False

        simulator.heal()
        assert simulator.is_active is False
        assert simulator.can_communicate("coordinator", "node-1") is True

    def test_partition_then_heal_is_fast(self):
        """Guard against a hung partition heal (must not stall the gate)."""
        simulator = NetworkPartitionSimulator()
        simulator.create_partition(["coordinator", "node-0"], ["node-1"])
        start = time.time()
        simulator.heal()
        elapsed = time.time() - start
        assert elapsed < 1.0, f"Heal took too long: {elapsed:.2f}s"
        assert simulator.can_communicate("coordinator", "node-1") is True
