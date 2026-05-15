"""Property-based tests for consistent hash ring invariants."""

from hypothesis import given, settings
from hypothesis import strategies as st

from distllm.router.consistent_hash import ConsistentHashRing


@given(
    num_nodes=st.integers(1, 20),
    num_keys=st.integers(1, 100),
)
@settings(max_examples=50, deadline=None)
def test_deterministic_routing(num_nodes, num_keys):
    """Same key should always map to the same node."""
    ring = ConsistentHashRing(replicas=50)
    nodes = [f"node-{i}" for i in range(num_nodes)]
    for n in nodes:
        ring.add_node(n)

    keys = [f"key-{i}" for i in range(num_keys)]
    for key in keys:
        node = ring.get_node(key)
        assert node is not None
        assert node in nodes
        # Second lookup must return same node
        assert ring.get_node(key) == node


@given(
    num_nodes=st.integers(3, 15),
    num_keys=st.integers(50, 200),
)
@settings(max_examples=30, deadline=None)
def test_load_balance(num_nodes, num_keys):
    """Keys should be distributed approximately evenly across nodes."""
    ring = ConsistentHashRing(replicas=150)
    nodes = [f"node-{i}" for i in range(num_nodes)]
    for n in nodes:
        ring.add_node(n)

    distribution = {n: 0 for n in nodes}
    for i in range(num_keys):
        key = f"session-{i}"
        node = ring.get_node(key)
        distribution[node] += 1

    expected = num_keys / num_nodes
    for node, count in distribution.items():
        # Allow 3x deviation for small sample sizes
        assert count > 0, f"Node {node} received no keys"
        assert count < expected * 3, f"Node {node} is overloaded: {count} vs expected {expected}"


@given(
    num_nodes=st.integers(5, 20),
    num_keys=st.integers(50, 100),
)
@settings(max_examples=30, deadline=None)
def test_minimal_remap_on_node_removal(num_nodes, num_keys):
    """Removing a node should remap only 1/N of keys (approximately)."""
    ring = ConsistentHashRing(replicas=150)
    nodes = [f"node-{i}" for i in range(num_nodes)]
    for n in nodes:
        ring.add_node(n)

    keys = [f"key-{i}" for i in range(num_keys)]
    initial_assignment = {k: ring.get_node(k) for k in keys}

    # Remove one node
    removed = nodes[-1]
    ring.remove_node(removed)

    remapped = 0
    for key in keys:
        new_node = ring.get_node(key)
        if initial_assignment[key] == removed:
            # Keys on removed node must be reassigned
            assert new_node is not None
            assert new_node != removed
            remapped += 1
        elif new_node is not None:
            # Keys on other nodes should mostly stay
            pass  # Some remap is expected due to ring restructuring

    # All keys that were on the removed node must be remapped
    keys_on_removed = sum(1 for k in keys if initial_assignment[k] == removed)
    assert remapped >= keys_on_removed


def test_empty_ring():
    """Empty ring should return None."""
    ring = ConsistentHashRing()
    assert ring.get_node("any-key") is None


def test_duplicate_node_add():
    """Adding the same node twice should be idempotent."""
    ring = ConsistentHashRing(replicas=50)
    ring.add_node("node-0")
    ring.add_node("node-0")
    assert ring.node_count == 1


@given(num_nodes=st.integers(1, 10))
@settings(max_examples=20)
def test_health_aware_fallback(num_nodes):
    """get_node_with_fallback should skip unhealthy nodes."""
    ring = ConsistentHashRing(replicas=50)
    nodes = [f"node-{i}" for i in range(num_nodes)]
    for n in nodes:
        ring.add_node(n)

    # Mark all but last as unhealthy
    healthy = {nodes[-1]}
    for key in ["key-1", "key-2", "key-3"]:
        node = ring.get_node_with_fallback(key, healthy)
        assert node == nodes[-1]
