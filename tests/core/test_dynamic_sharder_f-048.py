"""Regression tests for audit finding F-048.

F-048: dynamic_sharder installs a new partition after migrations that never
transferred data. With the default ``on_layer_transfer=None`` the transfer
step was skipped yet the layer was still marked COMPLETE and
``_execute_reshard`` unconditionally installed ``plan.new_partition`` —
routing layers to nodes that never received them.

Fixed contract (fail closed):
- Without a transfer callback, every migration fails and the OLD partition
  is kept (no silent success).
- If any transfer fails, the OLD partition is kept for the whole round.
- Only when every migration succeeds is the new partition installed.
"""

from __future__ import annotations

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/dynamic_sharder.py")
DynamicSharder = _mod.DynamicSharder
MigrationState = _mod.MigrationState


def _make_sharder(on_transfer=None) -> DynamicSharder:
    sharder = DynamicSharder(on_layer_transfer=on_transfer)
    sharder.set_initial_partition({"node-a": [0, 1, 2, 3]})
    return sharder


class TestNoTransferCallbackFailsClosed:
    def test_default_sharder_does_not_install_untransferred_partition(self):
        """DynamicSharder() with no on_layer_transfer must NOT report success."""
        sharder = _make_sharder()
        plan = sharder.on_node_join("node-b")
        assert plan is not None  # a plan was generated...

        # ...but no data moved, so the old partition must be kept.
        part = sharder.get_current_partition()
        assert "node-b" not in part
        assert part == {"node-a": [0, 1, 2, 3]}

        s = sharder.stats()
        assert s["migrations_failed"] == len(plan.migrations)
        assert s["migrations_completed"] == 0
        assert s["reshards"] == 0

    def test_failed_migrations_marked_failed(self):
        sharder = _make_sharder()
        sharder.on_node_join("node-b")
        states = {m.state for m in sharder._active_migrations}
        # active list is cleared after the round; check via a fresh run below
        assert states == set()

        sharder2 = _make_sharder()
        plan = sharder2.on_node_join("node-b")
        assert all(m.state == MigrationState.FAILED for m in plan.migrations)
        assert all(m.error for m in plan.migrations)


class TestTransferFailureKeepsOldPartition:
    def test_partition_kept_when_callback_returns_false(self):
        sharder = _make_sharder(on_transfer=lambda layer, src, dst: False)
        plan = sharder.on_node_join("node-b")
        assert plan is not None

        part = sharder.get_current_partition()
        assert part == {"node-a": [0, 1, 2, 3]}
        s = sharder.stats()
        assert s["migrations_failed"] == len(plan.migrations)
        assert s["reshards"] == 0


class TestSuccessfulTransfersInstallPartition:
    def test_partition_installed_only_after_all_transfers_succeed(self):
        calls: list[tuple[int, str, str]] = []

        def transfer(layer_id: int, source: str, target: str) -> bool:
            calls.append((layer_id, source, target))
            return True

        sharder = _make_sharder(on_transfer=transfer)
        plan = sharder.on_node_join("node-b")

        assert plan is not None
        assert len(calls) == len(plan.migrations) > 0
        # Every planned migration actually invoked the real transfer callback.
        expected = {(m.layer_id, m.source_node, m.target_node) for m in plan.migrations}
        assert set(calls) == expected

        part = sharder.get_current_partition()
        assert "node-b" in part
        total = sum(len(v) for v in part.values())
        assert total == 4

        s = sharder.stats()
        assert s["reshards"] == 1
        assert s["migrations_completed"] == len(plan.migrations)
        assert s["migrations_failed"] == 0
