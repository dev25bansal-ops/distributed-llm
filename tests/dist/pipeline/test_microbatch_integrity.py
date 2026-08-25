"""Micro-batched pipeline result-integrity tests (audit finding C5).

Guarantees pinned here:

  1. Happy path: output rows == input rows, no failure markers attached.
  2. Single micro-batch failure raises ``PipelineError`` naming the failed
     sequences (default ``on_partial_failure="raise"``) — the caller can
     never receive a silently-shrunk response batch.
  3. First-stage failures cascade downstream without stalling until the
     pipeline timeout, and never invoke workers on a broken batch.
  4. Multiple failures aggregate every failed micro-batch in one error.
  5. ``on_partial_failure="drop"`` preserves the legacy shape behaviour but
     attaches explicit ``failed_sequences`` metadata to the returned tensor
     and to ``stats()["last_failed_sequences"]``.
  6. A subsequent healthy run recovers cleanly (no stale failure state).

Pure unit tests: all gRPC traffic is mocked at the
``distllm.dist.node_client.forward_request_async`` boundary, mirroring the
existing orchestrator test suites.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from distllm.dist.pipeline.orchestrator import (
    PipelineError,
    PipelineOrchestrator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def register_two_nodes(orch: PipelineOrchestrator) -> None:
    """Register two non-overlapping stages (node-0: layers 0-15)."""
    orch.register_node("node-0", "10.0.0.1", 50051, 0, 15)
    orch.register_node("node-1", "10.0.0.2", 50051, 16, 31)


KV: dict[str, None] = {"node-0": None, "node-1": None}


async def _echo(**kwargs: object) -> torch.Tensor:
    """Echo the hidden states back (stands in for a healthy worker)."""
    hs = kwargs.get("hidden_states")
    assert hs is not None
    return hs.clone()


def _make_flaky(
    calls: list[str],
    fail_suffixes: tuple[str, ...],
    boom: str = "worker exploded",
):
    """Build an async forward mock failing on selected request-id suffixes.

    Request ids follow the orchestrator convention ``<rid>-s<stage>b<batch>``,
    so ``"-s1b0"`` targets stage 1 / micro-batch 0 deterministically.
    """

    async def flaky(**kwargs: object) -> torch.Tensor:
        rid = str(kwargs.get("request_id", ""))
        calls.append(rid)
        for suffix in fail_suffixes:
            if rid.endswith(suffix):
                raise RuntimeError(boom)
        return await _echo(**kwargs)

    return flaky


# ---------------------------------------------------------------------------
# 1. Happy path unchanged
# ---------------------------------------------------------------------------


class TestHappyPathUnchanged:
    @pytest.mark.asyncio
    async def test_full_batch_returned_no_failure_markers(self) -> None:
        """Batch of 4, all workers healthy -> 4 output rows, no markers."""
        orch = PipelineOrchestrator(resource_mgr=MagicMock())
        register_two_nodes(orch)
        inp = torch.randn(4, 32)

        calls: list[str] = []
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_make_flaky(calls, ()),
        ):
            result = await orch.run_pipeline_microbatched(
                inp, KV, "req-ok", micro_batch_size=2,
            )

        assert result.shape == (4, 32)
        assert getattr(result, "failed_sequences", None) is None
        stats = orch.stats()
        assert stats["errors"] == 0
        assert stats["micro_batch_count_total"] == 2
        assert stats["last_failed_sequences"] == []

    @pytest.mark.asyncio
    async def test_pipeline_error_is_runtime_error(self) -> None:
        """Callers catching broad RuntimeError keep working."""
        assert issubclass(PipelineError, RuntimeError)


# ---------------------------------------------------------------------------
# 2. Single failure must not silently shrink the batch (the C5 repro)
# ---------------------------------------------------------------------------


class TestSingleFailureRaises:
    @pytest.mark.asyncio
    async def test_last_stage_failure_raises_naming_failed_rows(self) -> None:
        """Batch of 4 where one worker call fails -> PipelineError naming
        the failed micro-batch and its input rows.  Before the fix the
        caller silently received 2 rows as if nothing happened."""
        orch = PipelineOrchestrator(resource_mgr=MagicMock())
        register_two_nodes(orch)
        inp = torch.randn(4, 32)  # mb0 = rows 0-1, mb1 = rows 2-3

        calls: list[str] = []
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_make_flaky(calls, ("-s1b0",)),
        ):
            with pytest.raises(PipelineError) as exc_info:
                await orch.run_pipeline_microbatched(
                    inp, KV, "req-a", micro_batch_size=2,
                )

        exc = exc_info.value
        assert exc.failed_micro_batches == [0]
        assert exc.failed_sequences == [0, 1]
        assert 0 in exc.errors
        assert "worker exploded" in exc.errors[0]
        # Message names the failed rows so on-call can act without unpacking.
        assert "[0, 1]" in str(exc)
        # The healthy micro-batch ran to completion; only the failed one
        # aborted (cascade must not double-count errors).
        assert orch.stats()["errors"] == 1

    @pytest.mark.asyncio
    async def test_resource_manager_sees_the_failure(self) -> None:
        rm = MagicMock()
        orch = PipelineOrchestrator(resource_mgr=rm)
        register_two_nodes(orch)

        calls: list[str] = []
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_make_flaky(calls, ("-s1b1",)),
        ):
            with pytest.raises(PipelineError):
                await orch.run_pipeline_microbatched(
                    torch.randn(4, 32), KV, "req-b", micro_batch_size=2,
                )

        rm.record_failure.assert_called_once_with("node-1")


# ---------------------------------------------------------------------------
# 3. First-stage failure: cascade, no timeout stall, no garbage RPCs
# ---------------------------------------------------------------------------


class TestFirstStageFailureCascades:
    @pytest.mark.asyncio
    async def test_upstream_failure_does_not_stall_until_timeout(self) -> None:
        """A stage-0 failure must fail the batch promptly (downstream is
        signalled, not left blocking on the dependency event until the
        pipeline timeout)."""
        orch = PipelineOrchestrator(
            resource_mgr=MagicMock(), pipeline_timeout=30.0,
        )
        register_two_nodes(orch)
        inp = torch.randn(4, 32)

        calls: list[str] = []
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_make_flaky(calls, ("-s0b1",)),
        ):
            t0 = time.monotonic()
            with pytest.raises(PipelineError) as exc_info:
                await orch.run_pipeline_microbatched(
                    inp, KV, "req-c", micro_batch_size=2,
                )
            elapsed = time.monotonic() - t0

        assert elapsed < 10.0  # far below the 30 s pipeline timeout
        exc = exc_info.value
        assert exc.failed_micro_batches == [1]
        assert exc.failed_sequences == [2, 3]

    @pytest.mark.asyncio
    async def test_cascade_skips_worker_calls_on_failed_batch(self) -> None:
        """Downstream stages must not be invoked with a missing input."""
        orch = PipelineOrchestrator(resource_mgr=MagicMock())
        register_two_nodes(orch)

        calls: list[str] = []
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_make_flaky(calls, ("-s0b0",)),
        ):
            with pytest.raises(PipelineError):
                await orch.run_pipeline_microbatched(
                    torch.randn(4, 32), KV, "req-d", micro_batch_size=2,
                )

        assert not any(rid.endswith("-s1b0") for rid in calls)


# ---------------------------------------------------------------------------
# 4. Multi-failure aggregation
# ---------------------------------------------------------------------------


class TestMultiFailure:
    @pytest.mark.asyncio
    async def test_both_batches_last_stage_failure_lists_all_rows(self) -> None:
        orch = PipelineOrchestrator(resource_mgr=MagicMock())
        register_two_nodes(orch)

        calls: list[str] = []
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_make_flaky(calls, ("-s1b0", "-s1b1")),
        ):
            with pytest.raises(PipelineError) as exc_info:
                await orch.run_pipeline_microbatched(
                    torch.randn(4, 32), KV, "req-e", micro_batch_size=2,
                )

        exc = exc_info.value
        assert exc.failed_micro_batches == [0, 1]
        assert exc.failed_sequences == [0, 1, 2, 3]
        # Exactly the two real RPC failures counted (cascades excluded —
        # here both failures happen on the last stage, so no cascades).
        assert orch.stats()["errors"] == 2

    @pytest.mark.asyncio
    async def test_everything_down_crash_reports_all_failed(self) -> None:
        """Both stages crash on every batch: no dependency deadlock, one
        structured error covering all rows."""
        orch = PipelineOrchestrator(
            resource_mgr=MagicMock(), pipeline_timeout=5.0,
        )
        register_two_nodes(orch)

        calls: list[str] = []
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_make_flaky(calls, ("-s0b0", "-s0b1")),
        ):
            with pytest.raises(PipelineError) as exc_info:
                await orch.run_pipeline_microbatched(
                    torch.randn(4, 32), KV, "req-f", micro_batch_size=2,
                )

        exc = exc_info.value
        assert exc.failed_micro_batches == [0, 1]
        assert exc.failed_sequences == [0, 1, 2, 3]
        assert "All micro-batches failed" in str(exc)
        # Stage-1 steps cascade instead of timing out: only stage-0's two
        # RPC failures are counted.
        assert orch.stats()["errors"] == 2


# ---------------------------------------------------------------------------
# 5. Explicit "drop" policy: legacy shapes, acknowledged metadata
# ---------------------------------------------------------------------------


class TestDropPolicyMarksFailures:
    @pytest.mark.asyncio
    async def test_drop_mode_attaches_failed_sequences_metadata(self) -> None:
        orch = PipelineOrchestrator(
            resource_mgr=MagicMock(), on_partial_failure="drop",
        )
        register_two_nodes(orch)
        inp = torch.randn(4, 32)

        calls: list[str] = []
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_make_flaky(calls, ("-s1b0",)),
        ):
            result = await orch.run_pipeline_microbatched(
                inp, KV, "req-g", micro_batch_size=2,
            )

        # Legacy shape behaviour preserved...
        assert result.shape == (2, 32)
        # ...but the shrinkage is explicit and machine-readable.
        assert result.failed_sequences == (0, 1)
        assert orch.stats()["last_failed_sequences"] == [0, 1]

    def test_unknown_policy_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="on_partial_failure"):
            PipelineOrchestrator(on_partial_failure="pad")


# ---------------------------------------------------------------------------
# 6. Recovery on the next batch
# ---------------------------------------------------------------------------


class TestRecoveryOnNextRun:
    @pytest.mark.asyncio
    async def test_healthy_run_after_failure_returns_full_batch(self) -> None:
        orch = PipelineOrchestrator(resource_mgr=MagicMock())
        register_two_nodes(orch)
        inp = torch.randn(4, 32)

        calls: list[str] = []
        # Failures scoped to the FIRST run's request id only ("req-h1"),
        # so the follow-up run exercises genuine recovery.
        first = _make_flaky(calls, ("req-h1-s0b1",))
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=first,
        ):
            with pytest.raises(PipelineError):
                await orch.run_pipeline_microbatched(
                    inp, KV, "req-h1", micro_batch_size=2,
                )

            result = await orch.run_pipeline_microbatched(
                inp, KV, "req-h2", micro_batch_size=2,
            )

        assert result.shape == (4, 32)
        assert getattr(result, "failed_sequences", None) is None
        # No stale failure metadata leaks from the failed run.
        assert orch.stats()["last_failed_sequences"] == []

    @pytest.mark.asyncio
    async def test_drop_mode_state_resets_after_healthy_run(self) -> None:
        orch = PipelineOrchestrator(
            resource_mgr=MagicMock(), on_partial_failure="drop",
        )
        register_two_nodes(orch)
        inp = torch.randn(4, 32)

        calls: list[str] = []
        failing = _make_flaky(calls, ("-s1b0",))
        healthy = _make_flaky(calls, ())
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=failing,
        ):
            failed_run = await orch.run_pipeline_microbatched(
                inp, KV, "req-i1", micro_batch_size=2,
            )
            assert failed_run.shape == (2, 32)

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=healthy,
        ):
            healthy_run = await orch.run_pipeline_microbatched(
                inp, KV, "req-i2", micro_batch_size=2,
            )

        assert healthy_run.shape == (4, 32)
        assert getattr(healthy_run, "failed_sequences", None) is None
        assert orch.stats()["last_failed_sequences"] == []
