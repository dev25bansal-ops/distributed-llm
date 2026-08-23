"""M14 regression tests — federated credit + reputation primitive.

These tests assert that the federated layer has a *real* incentive primitive
(CreditLedger) wired into federated rounds:

1. CreditLedger (stdlib-only) awards credit on record_contribution and
   reflects it in get_balance.
2. A failed/abandoned round lowers a node's reputation; success raises it;
   get_reputation is monotonic-ish.
3. The FederatedMergeCoordinator credits a node when it submits a valid
   adapter and exposes balances read-only via get_ledger_balances().
4. The FederatedFineTuner calls the ledger on a successful train_round
   (injected ledger is exercised).

The tests import the ledger module directly (it is stdlib-only by design) and
exercise the real coordinator / finetuner (torch is available under .venv311).
"""

from __future__ import annotations

import os
import tempfile

import pytest

from distllm.core.federated_incentives import CreditLedger


# ── 1. CreditLedger core primitive ──────────────────────────────────────

def test_record_contribution_adds_credits():
    ledger = CreditLedger()
    assert ledger.get_balance("node-a") == 0.0

    credits = ledger.record_contribution("r1", "node-a", weight_metric=50.0)
    assert credits == 50.0
    assert ledger.get_balance("node-a") == 50.0

    # A second contribution accumulates.
    ledger.record_contribution("r2", "node-a", weight_metric=10.0)
    assert ledger.get_balance("node-a") == 60.0


def test_get_balance_reflects_contribution_only_for_node():
    ledger = CreditLedger(credit_per_unit=2.0)
    ledger.record_contribution("r1", "node-a", weight_metric=5.0)
    ledger.record_contribution("r1", "node-b", weight_metric=3.0)
    # credit_per_unit scales the award.
    assert ledger.get_balance("node-a") == 10.0
    assert ledger.get_balance("node-b") == 6.0
    assert ledger.get_balance("node-unknown") == 0.0


def test_negative_or_zero_weight_yields_no_negative_credit():
    ledger = CreditLedger()
    ledger.record_contribution("r1", "node-a", weight_metric=-5.0)
    assert ledger.get_balance("node-a") == 0.0


# ── 2. Reputation monotonic-ish behavior ───────────────────────────────

def test_submit_failure_lowers_reputation():
    ledger = CreditLedger()
    base = ledger.get_reputation("node-a")
    new = ledger.apply_reputation("node-a", success=False)
    assert new < base
    assert ledger.get_reputation("node-a") == new


def test_reputation_rises_on_success_falls_on_failure():
    ledger = CreditLedger()
    r0 = ledger.get_reputation("node-a")
    r1 = ledger.apply_reputation("node-a", success=True)
    assert r1 > r0  # success raises
    r2 = ledger.apply_reputation("node-a", success=False)
    assert r2 < r1  # failure lowers
    # And it never goes below the floor or above the ceiling.
    for _ in range(100):
        ledger.apply_reputation("node-a", success=False)
    assert ledger.get_reputation("node-a") >= 0.0


def test_reputation_bounded():
    ledger = CreditLedger()
    for _ in range(200):
        ledger.apply_reputation("node-a", success=True)
    assert ledger.get_reputation("node-a") <= 5.0


# ── 3. Coordinator wiring (real, importable under .venv311) ────────────

def _make_coordinator(ledger=None):
    from distllm.dist.federated_merge import FederatedMergeCoordinator

    coord = FederatedMergeCoordinator(min_nodes_per_round=2, credit_ledger=ledger)
    return coord


def test_coordinator_credits_on_adapter_submit():
    ledger = CreditLedger()
    coord = _make_coordinator(ledger=ledger)

    coord.register_node("node-a", dataset_size=100)
    coord.register_node("node-b", dataset_size=100)
    round_info = coord.start_round()
    assert round_info is not None

    assert coord.submit_node_adapter("node-a", "path/a.pt", loss=0.5, dataset_size=100)
    assert ledger.get_balance("node-a") == 100.0
    assert ledger.get_reputation("node-a") > 1.0
    assert ledger.get_contribution_count("node-a") == 1


def test_coordinator_exposes_read_only_ledger():
    ledger = CreditLedger()
    coord = _make_coordinator(ledger=ledger)
    coord.register_node("node-a", dataset_size=40)
    coord.register_node("node-b", dataset_size=60)
    coord.start_round()
    coord.submit_node_adapter("node-a", "path/a.pt", loss=0.4, dataset_size=40)

    snap = coord.get_ledger_balances()
    assert "node-a" in snap
    assert snap["node-a"]["balance"] == 40.0
    assert snap["node-a"]["reputation"] > 1.0
    assert snap["node-a"]["contributions"] == 1


def test_coordinator_penalizes_reputation_on_merge_failure():
    ledger = CreditLedger()
    coord = _make_coordinator(ledger=ledger)

    coord.register_node("node-a", dataset_size=100)
    coord.register_node("node-b", dataset_size=100)
    coord.start_round()
    coord.submit_node_adapter("node-a", "path/a.pt", loss=0.3, dataset_size=100)
    coord.submit_node_adapter("node-b", "path/b.pt", loss=0.3, dataset_size=100)

    # Both submissions succeeded -> reputation was raised for each node.
    rep_after_submit = {
        "node-a": ledger.get_reputation("node-a"),
        "node-b": ledger.get_reputation("node-b"),
    }
    assert rep_after_submit["node-a"] > 1.0
    assert rep_after_submit["node-b"] > 1.0

    # The submitted adapter paths do not exist, so merge_adapters hits the
    # failure branch (torch.load raises) which lowers every participant's
    # reputation via the ledger.
    merged = coord.merge_adapters()
    assert merged is None
    assert ledger.get_reputation("node-a") < rep_after_submit["node-a"]
    assert ledger.get_reputation("node-b") < rep_after_submit["node-b"]


# ── 4. FineTuner wiring ────────────────────────────────────────────────

def _fake_grad(n):
    import torch

    return [torch.zeros(2, 2) for _ in range(n)]


def test_finetuner_calls_ledger_on_successful_round():
    ledger = CreditLedger()
    from distllm.core.federated_finetuner import FederatedFineTuner

    finetuner = FederatedFineTuner(
        node_id="node-a",
        local_steps=25,
        num_rounds=1,
        credit_ledger=ledger,
    )

    # Verify the injected ledger is the one being used.
    assert finetuner._ledger is ledger

    def local_train_fn(steps):
        return _fake_grad(1)

    metrics = finetuner.train_round(local_train_fn)
    assert metrics["round"] == 1

    # A successful round should have credited the node by its local_steps.
    assert ledger.get_balance("node-a") == 25.0
    assert ledger.get_reputation("node-a") > 1.0
    assert ledger.get_contribution_count("node-a") == 1


def test_finetuner_mark_round_failure_lowers_reputation():
    ledger = CreditLedger()
    from distllm.core.federated_finetuner import FederatedFineTuner

    finetuner = FederatedFineTuner(
        node_id="node-a", credit_ledger=ledger
    )
    rep_before = ledger.get_reputation("node-a")
    finetuner.mark_round_failure("node-b")
    assert ledger.get_reputation("node-b") < 1.0
    # node-a unchanged when a different node failed.
    assert ledger.get_reputation("node-a") == rep_before


# ── 5. Optional sqlite persistence sanity ──────────────────────────────

def test_ledger_persists_across_instances():
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "ledger.sqlite")
    l1 = CreditLedger(db_path=db)
    l1.record_contribution("r1", "node-a", weight_metric=10.0)
    l1.apply_reputation("node-a", success=True)

    l2 = CreditLedger(db_path=db)
    assert l2.get_balance("node-a") == 10.0
    assert l2.get_reputation("node-a") > 1.0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
