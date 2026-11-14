"""Tests for distllm.dist.redundant_executor.

Covers the redundant-execution guarantees an operator relies on:

* ``run_redundant`` returns the first *successful* replica — a fast
  failure must never beat a slow success — and a timeout raises instead
  of returning a success-shaped empty result.
* ``IncrementalRecovery`` records real deltas in its documented no-base
  mode and computes exact deltas in mixed base/no-base mode.
* ``recover_node`` never reports success while silently skipping a
  requested replication phase.
"""
import threading
import time

import pytest
import torch

from distllm.dist.redundant_executor import (
    IncrementalRecovery,
    RedundantExecutor,
)

# ---------------------------------------------------------------------------
# run_redundant — failover (finding: fast failure beat slow success)
# ---------------------------------------------------------------------------


class TestRunRedundantFailover:
    def test_fast_failure_does_not_beat_slow_success(self):
        """Replica-0 fails immediately; replica-1 succeeds after 150ms.

        Old behaviour: FIRST_COMPLETED inspected only replica-0 and raised
        'All 2 redundant replicas failed'.  The healthy backup must win.
        """
        def task(replica_id, **_kw):
            if replica_id == "replica-0":
                raise RuntimeError("replica down")
            time.sleep(0.15)
            return "healthy-result"

        result = RedundantExecutor().run_redundant(task, n_replicas=2, timeout_s=5)
        assert result.result == "healthy-result"
        assert result.replica_id == "replica-1"
        assert result.failures == 1
        assert result.total_replicas == 2

    def test_all_replicas_failing_raises(self):
        def task(replica_id, **_kw):
            raise RuntimeError(f"boom-{replica_id}")

        with pytest.raises(RuntimeError, match="All 2 redundant replicas failed"):
            RedundantExecutor().run_redundant(task, n_replicas=2, timeout_s=5)

    def test_fastest_success_wins(self):
        def task(replica_id, **_kw):
            if replica_id == "replica-0":
                time.sleep(0.2)
                return "slow"
            return "fast"

        result = RedundantExecutor().run_redundant(task, n_replicas=2, timeout_s=5)
        assert result.result == "fast"
        assert result.replica_id == "replica-1"

    def test_single_replica_success(self):
        result = RedundantExecutor().run_redundant(lambda rid, **_kw: 42, n_replicas=1)
        assert result.result == 42
        assert result.replica_id == "replica-0"


# ---------------------------------------------------------------------------
# run_redundant — timeout semantics
# ---------------------------------------------------------------------------


class TestRunRedundantTimeout:
    def test_timeout_raises_instead_of_success_shaped_none(self):
        """A timeout with zero completions must raise, not return
        RedundantResult(result=None, failures=0)."""
        t0 = time.monotonic()
        with pytest.raises(RuntimeError, match="timed out"):
            RedundantExecutor().run_redundant(
                lambda rid, **_kw: time.sleep(0.5),
                n_replicas=2,
                timeout_s=0.05,
            )
        assert time.monotonic() - t0 < 0.4

    def test_timeout_after_one_failure_reports_timeout(self):
        def task(replica_id, **_kw):
            if replica_id == "replica-0":
                raise RuntimeError("down")
            time.sleep(0.5)
            return "late"

        with pytest.raises(RuntimeError, match="timed out"):
            RedundantExecutor().run_redundant(task, n_replicas=2, timeout_s=0.1)

    def test_success_before_deadline_wins(self):
        def task(replica_id, **_kw):
            time.sleep(0.02)
            return "ok"

        result = RedundantExecutor().run_redundant(task, n_replicas=2, timeout_s=2)
        assert result.result == "ok"


# ---------------------------------------------------------------------------
# IncrementalRecovery — documented no-base mode (finding: silent no-op)
# ---------------------------------------------------------------------------


class TestIncrementalRecoveryNoBaseMode:
    def test_save_delta_without_base_records_deltas(self):
        """The documented no-base mode must record deltas; it used to
        record zero deltas forever (delta=0 skipped, base=current)."""
        rec = IncrementalRecovery()
        rec.save_delta(1, {"w": torch.full((4,), 5.0)})
        rec.save_delta(2, {"w": torch.full((4,), 7.0)})
        assert rec.delta_count == 2
        assert rec.latest_step == 2

    def test_recover_to_after_no_base_saves(self):
        rec = IncrementalRecovery()
        rec.save_delta(1, {"w": torch.full((4,), 5.0)})
        rec.save_delta(2, {"w": torch.full((4,), 7.0)})
        out = rec.recover_to(2, {"w": torch.full((4,), 5.0)})
        torch.testing.assert_close(out["w"], torch.full((4,), 7.0))

    def test_no_base_chain_with_two_parameters(self):
        rec = IncrementalRecovery()
        rec.save_delta(1, {"a": torch.tensor([1.0]), "b": torch.tensor([10.0])})
        rec.save_delta(2, {"a": torch.tensor([3.0]), "b": torch.tensor([12.0])})
        rec.save_delta(3, {"a": torch.tensor([6.0]), "b": torch.tensor([15.0])})
        checkpoint = {"a": torch.tensor([0.0]), "b": torch.tensor([9.0])}
        out = rec.recover_to(3, {k: v.clone() for k, v in checkpoint.items()})
        torch.testing.assert_close(out["a"], torch.tensor([6.0]))
        torch.testing.assert_close(out["b"], torch.tensor([15.0]))

    def test_recover_to_intermediate_step(self):
        rec = IncrementalRecovery()
        rec.save_delta(1, {"w": torch.full((2,), 5.0)})
        rec.save_delta(2, {"w": torch.full((2,), 7.0)})
        rec.save_delta(3, {"w": torch.full((2,), 20.0)})
        out = rec.recover_to(2, {"w": torch.full((2,), 5.0)})
        torch.testing.assert_close(out["w"], torch.full((2,), 7.0))


# ---------------------------------------------------------------------------
# IncrementalRecovery — mixed base/no-base mode (finding: sum-of-deltas bug)
# ---------------------------------------------------------------------------


class TestIncrementalRecoveryMixedMode:
    def test_base_then_no_base_is_exact(self):
        """save_delta(1, w=15, base=ck{w:10}) then save_delta(2, w=17)
        used to record delta=17-5=12 (sum of deltas as the 'base') and
        recover to 27.0 instead of 17.0."""
        rec = IncrementalRecovery()
        ck = {"w": torch.full((4,), 10.0)}
        rec.save_delta(1, {"w": torch.full((4,), 15.0)}, base_params=ck)
        rec.save_delta(2, {"w": torch.full((4,), 17.0)})
        out = rec.recover_to(2, dict(ck))
        torch.testing.assert_close(out["w"], torch.full((4,), 17.0))

    def test_no_base_then_base_is_exact(self):
        """A base given after an anchor-backed save must chain exactly."""
        rec = IncrementalRecovery()
        rec.save_delta(1, {"w": torch.full((4,), 5.0)})
        # base_params is the reference state *just before* step 2, i.e.
        # the value observed at step 1.
        base2 = {"w": torch.full((4,), 5.0)}
        rec.save_delta(2, {"w": torch.full((4,), 11.0)}, base_params=base2)
        out = rec.recover_to(2, {"w": torch.full((4,), 5.0)})
        torch.testing.assert_close(out["w"], torch.full((4,), 11.0))

    def test_three_step_mixed_chain(self):
        rec = IncrementalRecovery()
        ck = {"w": torch.full((2,), 10.0)}
        rec.save_delta(1, {"w": torch.full((2,), 15.0)}, base_params=ck)   # +5
        rec.save_delta(2, {"w": torch.full((2,), 17.0)})                   # +2
        rec.save_delta(3, {"w": torch.full((2,), 30.0)}, base_params={"w": torch.full((2,), 17.0)})  # +13
        out = rec.recover_to(3, dict(ck))
        torch.testing.assert_close(out["w"], torch.full((2,), 30.0))

    def test_zero_delta_still_skipped_with_base(self):
        rec = IncrementalRecovery()
        ck = {"w": torch.full((4,), 10.0)}
        rec.save_delta(1, {"w": torch.full((4,), 10.0)}, base_params=ck)
        assert rec.delta_count == 0


# ---------------------------------------------------------------------------
# recover_node — requested NCCL phase must not silently no-op
# ---------------------------------------------------------------------------


class _UninitializedReplicator:
    """Duck-typed replicator that can never initialize (simulates a
    deployment with no usable process group)."""

    is_initialized = False

    def initialize(self):
        pass

    def replicate(self, tensor, peers=None):
        raise AssertionError("replicate must not be called for a skipped phase")

    def stats(self):
        return {}


class _AlwaysReadyReplicator:
    """Duck-typed replicator that always works; records replicated tensors."""

    is_initialized = True

    def __init__(self):
        self.replicated = []

    def initialize(self):
        pass

    def replicate(self, tensor, peers=None):
        self.replicated.append(tensor)
        return [tensor]

    def stats(self):
        return {}


class TestRecoverNodeStateReplication:
    def test_requested_state_replication_that_cannot_run_fails_loudly(self):
        """The default replicator is auto_init=False; a state_tensors
        request used to return success=True with the phase skipped."""
        report = RedundantExecutor(
            replicator=_UninitializedReplicator(),
        ).recover_node("node-3", state_tensors={"kv": torch.zeros(8)})
        assert report.success is False
        assert report.error != ""
        assert "state replication" in report.error

    def test_requested_state_replication_runs_when_available(self):
        """When the replicator is ready, phase 3 must actually execute and
        report the replicated bytes."""
        replicator = _AlwaysReadyReplicator()
        report = RedundantExecutor(replicator=replicator).recover_node(
            "node-3", state_tensors={"kv": torch.zeros(8)},
        )
        assert report.success is True, report.error
        assert report.error == ""
        assert len(replicator.replicated) == 1
        assert report.replication_size_bytes == 8 * 4  # 8 float32 elements x 1 peer

    def test_recover_node_without_state_tensors_still_succeeds(self):
        """Bare recovery (docstring example) is a valid no-op report."""
        report = RedundantExecutor().recover_node("node-3")
        assert report.success is True
        assert report.error == ""

    def test_incremental_recovery_phase(self):
        """recover_node phase 1 replays recorded deltas onto the checkpoint."""
        executor = RedundantExecutor()
        ck = {"w": torch.zeros(2)}
        executor.save_delta(1, {"w": torch.tensor([5.0, 5.0])})
        report = executor.recover_node(
            "node-3", checkpoint_params=ck, target_step=1,
        )
        assert report.success is True
        assert report.steps_replayed == 1
        torch.testing.assert_close(ck["w"], torch.tensor([5.0, 5.0]))


# ---------------------------------------------------------------------------
# recover_node — phase 3 runs when a process group is available
# ---------------------------------------------------------------------------


class TestRecoverNodeWithReplicator:
    def test_single_rank_process_group_is_used_for_replication(self):
        """With torch.distributed up (single rank), phase 3 must actually
        run and report the replicated bytes."""
        import torch.distributed as dist

        created_here = False
        if not dist.is_initialized():
            import os

            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29619")
            try:
                dist.init_process_group("gloo", rank=0, world_size=1)
            except Exception as exc:  # environment cannot host a group
                pytest.skip(f"cannot create a local gloo process group: {exc}")
            created_here = True

        try:
            report = RedundantExecutor().recover_node(
                "node-3", state_tensors={"kv": torch.zeros(8)},
            )
            assert report.success is True, report.error
            assert report.replication_size_bytes > 0
        finally:
            if created_here:
                dist.destroy_process_group()
