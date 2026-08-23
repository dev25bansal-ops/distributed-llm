"""Regression tests for F-018 — stale peer gradients leak across rounds.

Bug: ``FederatedFineTuner._received_grads`` was keyed by peer_id and never
cleared or round-filtered.  ``train_round()`` broadcast
``{"gradients": ..., "round": r}`` but the receive/store path ignored the
``round`` field, so a peer that went silent after round N kept its round-N
gradients in the slot and they were averaged at equal weight with fresh
local gradients in every subsequent round — silently diverging FedAvg/FedProx.

Fix: contributions are tagged with their federated round; entries whose
round does not match the round in progress are pruned at round start,
dropped on receipt, and skipped before averaging.
"""

from __future__ import annotations

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_fft = load_module("distllm/core/federated_finetuner.py")
FederatedFineTuner = _fft.FederatedFineTuner

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not available")


def _make_tuner(responses):
    """Build a tuner whose gossip_receive pops one queued response per round.

    ``responses`` is a list of messages (or None) consumed in order; once
    exhausted, receive returns None (peer went silent).
    """
    queue = list(responses)

    def receive(timeout=30.0):
        if queue:
            return queue.pop(0)
        return None

    merged_history = []
    return FederatedFineTuner(
        node_id="node-a",
        local_steps=1,
        num_rounds=len(responses) + 2,
        dp_epsilon=float("inf"),  # disable clip/noise for deterministic math
        gossip_broadcast=None,
        gossip_receive=receive,
        apply_gradients=lambda grads, lr: merged_history.append(grads[0].clone()),
    ), merged_history


class TestStaleGradientIsolation:
    def test_straggler_grads_not_averaged_into_later_rounds(self):
        """The finding's exact scenario: peer contributes in round 1, then
        goes silent.  Round 2's average must use ONLY fresh local grads."""
        tuner, merged = _make_tuner(
            responses=[
                {"peer_id": "p1", "gradients": [torch.tensor([0.0])], "round": 1},
                None,  # p1 silent in round 2
            ]
        )

        # Round 1: avg(2.0 local, 0.0 peer) == 1.0
        r1 = tuner.train_round(lambda steps: [torch.tensor([2.0])])
        assert r1["gradients_received"] == 1
        assert torch.allclose(merged[-1], torch.tensor([1.0]))

        # Round 2: buggy code averaged the retained round-1 grads giving
        # (4 + 0) / 2 == 2.0; fixed code must return the local grad alone.
        r2 = tuner.train_round(lambda steps: [torch.tensor([4.0])])
        assert r2["gradients_received"] == 0
        assert torch.allclose(merged[-1], torch.tensor([4.0]))

    def test_silent_peer_slot_pruned(self):
        """A peer not heard from this round must be removed from
        _received_grads, not left carrying its previous submission."""
        tuner, _ = _make_tuner(
            responses=[
                {"peer_id": "p1", "gradients": [torch.tensor([9.0])], "round": 1},
                None,
            ]
        )
        tuner.train_round(lambda steps: [torch.tensor([1.0])])
        assert "p1" in tuner._received_grads

        tuner.train_round(lambda steps: [torch.tensor([1.0])])
        assert "p1" not in tuner._received_grads
        assert "p1" not in tuner._received_rounds


class TestRoundVersionGuard:
    def test_out_of_order_stale_message_dropped(self):
        """A late/out-of-order gossip message tagged with an earlier round
        must never be averaged into the current round."""
        tuner, merged = _make_tuner(
            responses=[
                None,  # round 1: no messages
                # Round 2 consumes a delayed round-1 message:
                {"peer_id": "late-p", "gradients": [torch.tensor([100.0])], "round": 1},
            ]
        )

        tuner.train_round(lambda steps: [torch.tensor([3.0])])

        result = tuner.train_round(lambda steps: [torch.tensor([3.0])])
        assert result["gradients_received"] == 0
        assert torch.allclose(merged[-1], torch.tensor([3.0]))

    def test_future_round_message_dropped(self):
        """Messages tagged ahead of the current round are equally invalid."""
        tuner, merged = _make_tuner(
            responses=[
                {"peer_id": "ahead-p", "gradients": [torch.tensor([100.0])], "round": 99},
            ]
        )

        result = tuner.train_round(lambda steps: [torch.tensor([5.0])])
        assert result["gradients_received"] == 0
        assert torch.allclose(merged[-1], torch.tensor([5.0]))

    def test_current_round_message_accepted_and_averaged(self):
        """Happy path unchanged: a contribution tagged with the round in
        progress is stored and averaged."""
        tuner, merged = _make_tuner(
            responses=[
                {"peer_id": "p1", "gradients": [torch.tensor([6.0])], "round": 1},
            ]
        )

        result = tuner.train_round(lambda steps: [torch.tensor([2.0])])
        assert result["gradients_received"] == 1
        assert torch.allclose(merged[-1], torch.tensor([4.0]))  # avg(2, 6)

    def test_missing_round_field_treated_as_current(self):
        """Legacy senders that omit 'round' keep working (accepted as
        current-round), preserving backward compatibility."""
        tuner, merged = _make_tuner(
            responses=[
                {"peer_id": "legacy", "gradients": [torch.tensor([6.0])]},
            ]
        )

        result = tuner.train_round(lambda steps: [torch.tensor([2.0])])
        assert result["gradients_received"] == 1
        assert torch.allclose(merged[-1], torch.tensor([4.0]))  # avg(2, 6)


class TestAverageDefenses:
    def test_average_skips_mismatched_round_entries(self):
        """Defense in depth: _average_gradients itself skips entries tagged
        with a non-current round even if one somehow lingers."""
        fft = FederatedFineTuner(node_id="n1")
        fft._received_grads["stale-peer"] = [torch.tensor([100.0])]
        fft._received_grads["fresh-peer"] = [torch.tensor([6.0])]
        fft._received_rounds["stale-peer"] = 1
        fft._received_rounds["fresh-peer"] = fft._round = 7

        result = fft._average_gradients([torch.tensor([2.0])])
        assert torch.allclose(result[0], torch.tensor([4.0]))  # avg(2, 6)

    def test_layer_count_guard_still_applies(self):
        """Pre-existing guard: mismatched layer counts are never averaged."""
        fft = FederatedFineTuner(node_id="n1")
        fft._received_grads["bad-shape"] = [torch.tensor([1.0]), torch.tensor([2.0])]
        fft._received_rounds["bad-shape"] = fft._round

        result = fft._average_gradients([torch.tensor([2.0])])
        assert torch.allclose(result[0], torch.tensor([2.0]))

    def test_tensor_shape_mismatch_not_broadcast_averaged(self):
        """A peer with the right layer count but wrong tensor shapes must be
        skipped, not broadcast-corrupted (avg would be [2.0, 2.5] otherwise)."""
        fft = FederatedFineTuner(node_id="n1")
        fft._received_grads["wrong-shape"] = [torch.tensor([3.0])]
        fft._received_rounds["wrong-shape"] = fft._round

        result = fft._average_gradients([torch.tensor([1.0, 2.0])])
        assert torch.allclose(result[0], torch.tensor([1.0, 2.0]))
