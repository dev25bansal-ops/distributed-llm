"""Tests for coordinator crash recovery.

Verifies that the system can recover from coordinator crashes
mid-request, including checkpoint persistence and state restoration.
"""

import json
import os
import tempfile
import threading
import time

import pytest
from unittest.mock import MagicMock, patch

from distllm.dist.recovery import NodeRecoveryManager, SequenceCheckpoint


class TestCoordinatorCrashRecovery:
    """Test recovery from coordinator crashes."""

    def test_checkpoint_persists_across_restart(self, tmp_path):
        """Checkpoints saved to disk survive coordinator restart."""
        persist_path = str(tmp_path / "checkpoints.json")
        mgr = NodeRecoveryManager(persist_path=persist_path)

        # Save checkpoints
        mgr.save_checkpoint(
            request_id="req-1",
            kv_cache=None,
            prompt_tokens=[1, 2, 3],
            generated_tokens=[4, 5],
            node_id="node-1",
        )
        mgr.save_to_disk()

        # Simulate restart — create new manager
        mgr2 = NodeRecoveryManager(persist_path=persist_path)
        loaded = mgr2.load_from_disk()

        assert loaded is True
        ckpt = mgr2.get_checkpoint("req-1")
        assert ckpt is not None
        assert ckpt.prompt_tokens == [1, 2, 3]
        assert ckpt.generated_tokens == [4, 5]

    def test_kv_cache_persisted_with_flag(self, tmp_path):
        """KV cache is persisted when include_kv_cache=True."""
        persist_path = str(tmp_path / "checkpoints.json")
        mgr = NodeRecoveryManager(persist_path=persist_path)

        import torch
        kv_cache = {"layer_0": [torch.randn(1, 32, 10, 64), torch.randn(1, 32, 10, 64)]}

        mgr.save_checkpoint(
            request_id="req-1",
            kv_cache=kv_cache,
            prompt_tokens=[1, 2, 3],
            generated_tokens=[4, 5],
            node_id="node-1",
        )
        mgr.save_to_disk(include_kv_cache=True)

        # Verify companion .kv.pt file exists
        kv_path = persist_path + ".kv.pt"
        assert os.path.exists(kv_path)

        # Load and verify
        mgr2 = NodeRecoveryManager(persist_path=persist_path)
        mgr2.load_from_disk()
        ckpt = mgr2.get_checkpoint("req-1")
        assert ckpt is not None
        assert ckpt.kv_cache is not None

    def test_recovery_after_node_failure(self):
        """Recovery manager handles node failure and redistributes layers."""
        mgr = NodeRecoveryManager()

        # Save checkpoints for a node
        for i in range(5):
            mgr.save_checkpoint(
                request_id=f"req-{i}",
                kv_cache=None,
                prompt_tokens=[i],
                generated_tokens=[i + 100],
                node_id="node-1",
            )

        # Track callbacks
        drain_called = []
        redistribute_called = []
        recover_called = []

        mgr.set_drain_callback(lambda nid: drain_called.append(nid))
        mgr.set_redistribute_layers_callback(lambda nid, plan: redistribute_called.append(nid))
        mgr.set_recover_sequences_callback(lambda nid, seqs: recover_called.extend(seqs))

        # Trigger recovery
        plan = mgr.on_node_failure("node-1")

        assert "node-1" in drain_called
        assert plan.failed_node_id == "node-1"
        assert len(plan.recovered_sequences) > 0

    def test_async_recovery(self):
        """Async recovery delegates to sync version."""
        import asyncio

        mgr = NodeRecoveryManager()
        mgr.save_checkpoint(
            request_id="req-1",
            kv_cache=None,
            prompt_tokens=[1],
            generated_tokens=[2],
            node_id="node-1",
        )

        plan = asyncio.run(mgr.on_node_failure_async("node-1"))
        assert plan.failed_node_id == "node-1"

    def test_checkpoint_ttl_eviction(self):
        """Checkpoints older than TTL are evicted."""
        mgr = NodeRecoveryManager(checkpoint_ttl_s=0.1)

        mgr.save_checkpoint(
            request_id="req-1",
            kv_cache=None,
            prompt_tokens=[1],
            generated_tokens=[2],
            node_id="node-1",
        )

        # Wait for TTL
        time.sleep(0.15)

        evicted = mgr.evict_stale_checkpoints()
        assert evicted > 0
        assert mgr.get_checkpoint("req-1") is None

    def test_multiple_nodes_recovery(self):
        """Recovery handles multiple nodes failing independently."""
        mgr = NodeRecoveryManager()

        for i in range(3):
            mgr.save_checkpoint(
                request_id=f"req-{i}",
                kv_cache=None,
                prompt_tokens=[i],
                generated_tokens=[i + 100],
                node_id=f"node-{i % 2}",  # 2 nodes
            )

        # Fail node-0
        plan0 = mgr.on_node_failure("node-0")
        assert plan0.failed_node_id == "node-0"

        # Fail node-1
        plan1 = mgr.on_node_failure("node-1")
        assert plan1.failed_node_id == "node-1"

        # Both should have recovery attempts
        metrics = mgr.get_metrics()
        assert metrics["failed_nodes"] == 2

    def test_recovery_history_recorded(self):
        """Recovery events are recorded in history."""
        mgr = NodeRecoveryManager()

        mgr.save_checkpoint(
            request_id="req-1",
            kv_cache=None,
            prompt_tokens=[1],
            generated_tokens=[2],
            node_id="node-1",
        )

        mgr.on_node_failure("node-1")

        history = mgr.get_recovery_history()
        assert len(history) == 1
        assert history[0]["node_id"] == "node-1"
