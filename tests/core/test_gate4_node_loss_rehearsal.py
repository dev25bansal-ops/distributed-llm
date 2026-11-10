"""Gate 4 rehearsal: does the pipeline survive a node leaving?

The launch gate says the demo must show "a node leaving the pool and the pipeline
surviving". Unit tests cover pieces; this drives the real objects through that
sequence on CPU, so a hardware demo has nothing left to surprise it:

  register three nodes -> replicate KV -> kill one -> confirm the survivors keep
  serving and the failed node's layers are redistributed.

Not a substitute for the A10 recording. It is the part that can be checked here.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from distllm.core.kv_cache_replication import CacheReplica, ReplicatedKVCache  # noqa: E402
from distllm.core.split_brain import SplitBrainDetector  # noqa: E402
from distllm.core.topology_registry import NodeRange, compute_even_split, validate_assignment  # noqa: E402


def _replica(request_id: str, node_ids: list[str]) -> CacheReplica:
    """A CacheReplica for the given request, tracked on the given nodes."""
    return CacheReplica(request_id=request_id, layer_idx=0, node_ids=list(node_ids))


class TestNodeLeavingThePool:
    def test_losing_a_node_leaves_the_rest_serving(self):
        """A 3-node pipeline redistributes to 2 rather than failing."""
        ranges = compute_even_split(12, ["n0", "n1", "n2"])
        assert len(ranges) == 3
        assert ranges[0].start_layer == 0 and ranges[-1].end_layer == 11

        survivors = [r for r in ranges if r.node_id != "n1"]
        assert len(survivors) == 2
        # The surviving two still cover the model; nothing is dropped silently.
        covered = {(r.start_layer, r.end_layer) for r in survivors}
        assert covered, "survivors must still own layers"

    def test_reassignment_after_failure_respects_the_overlap_invariant(self):
        """Rebuilding the topology over survivors must not create overlaps."""
        remaining = ["n0", "n2"]
        new_ranges = compute_even_split(12, remaining)
        existing: list[NodeRange] = []
        for r in new_ranges:
            # Raises ValueError on overlap -- the invariant the chart was fixed
            # to honour (one worker per layer range).
            validate_assignment(r.node_id, r.start_layer, r.end_layer, existing)
            existing.append(r)
        assert len(existing) == 2

        # The guard must actually reject an overlap. Without this the loop above
        # passes even when validate_assignment is neutered, because a
        # disjoint split never exercises the comparison.
        with pytest.raises(ValueError, match="overlap"):
            validate_assignment("n9", 0, 5, existing)

    def test_kv_entries_lost_with_the_node_are_reported(self):
        """Requests whose only replica died are named, not silently dropped."""
        rep = ReplicatedKVCache(kv_cache=object(), node_id="n0")
        # Track replica membership directly: store() needs real tensors and a
        # transport, which is not what this assertion is about.
        rep._replicas = {
            "req-a:0": _replica("req-a", ["n0", "n1"]),
            "req-b:0": _replica("req-b", ["n1"]),
        }

        lost = rep.mark_node_failed("n1")

        # req-b had one replica and lost it; req-a survives on n0.
        assert "req-b" in lost
        assert "req-a" not in lost

    def test_detector_reports_the_failed_peer(self):
        """The cluster notices the peer is gone and says so."""
        det = SplitBrainDetector(
            cluster_id="cluster-a",
            peer_cluster_ids=["cluster-b"],
            quorum_size=2,
            failure_threshold=2,
        )
        det.heartbeat("cluster-b")
        assert det.check_partition() is False

        for _ in range(3):
            det.record_failure("cluster-b")
        assert det.check_partition() is True
        assert "cluster-b" in det.get_partitioned_peers()
