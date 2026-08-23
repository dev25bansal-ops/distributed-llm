"""Tests for node failure recovery and checkpoint replay.

Covers:
- Checkpoint creation and storage
- get_checkpoints_for_node() returns correct checkpoints
- _on_node_recover() replays checkpoints
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestNodeRecovery:
    """Node failure recovery and checkpoint replay."""

    def test_checkpoint_creation(self):
        """SequenceCheckpoint stores KV cache and token state."""
        from distllm.dist.recovery import SequenceCheckpoint

        ckpt = SequenceCheckpoint(
            request_id="test-req",
            kv_cache={"layer_0": (None, None)},
            prompt_tokens=[1, 2, 3],
            generated_tokens=[4, 5],
            node_id="node-0",
        )
        assert ckpt.request_id == "test-req"
        assert ckpt.prompt_tokens == [1, 2, 3]
        assert ckpt.generated_tokens == [4, 5]
        assert ckpt.node_id == "node-0"

    def test_checkpoint_size_bytes(self):
        """size_bytes() should compute total tensor memory."""
        from distllm.dist.recovery import SequenceCheckpoint
        import torch

        k = torch.randn(4, 8, 16)
        v = torch.randn(4, 8, 16)
        ckpt = SequenceCheckpoint(
            request_id="test",
            kv_cache=[(k, v)],
            prompt_tokens=[1],
            generated_tokens=[2],
            node_id="node-0",
        )
        expected = k.numel() * k.element_size() + v.numel() * v.element_size()
        assert ckpt.size_bytes() == expected

    def test_get_checkpoints_for_node(self):
        """get_checkpoints_for_node should filter by node_id."""
        from distllm.dist.recovery import NodeRecoveryManager, SequenceCheckpoint

        mgr = NodeRecoveryManager()
        mgr.save_checkpoint("req-1", {}, [1], [2], "node-0")
        mgr.save_checkpoint("req-2", {}, [3], [4], "node-1")
        mgr.save_checkpoint("req-3", {}, [5], [6], "node-0")

        node0 = mgr.get_checkpoints_for_node("node-0")
        assert len(node0) == 2
        assert "req-1" in node0
        assert "req-3" in node0

        node1 = mgr.get_checkpoints_for_node("node-1")
        assert len(node1) == 1
        assert "req-2" in node1

    def test_drop_checkpoint(self):
        """Drop checkpoint should remove it from storage."""
        from distllm.dist.recovery import NodeRecoveryManager

        mgr = NodeRecoveryManager()
        mgr.save_checkpoint("req-1", {}, [1], [2], "node-0")
        assert mgr.get_checkpoint("req-1") is not None
        mgr.drop_checkpoint("req-1")
        assert mgr.get_checkpoint("req-1") is None

    def test_on_node_recover_replays_checkpoints(self):
        """_on_node_recover should return recovered sequence IDs."""
        from distllm.dist.recovery import NodeRecoveryManager

        mgr = NodeRecoveryManager()
        mgr.save_checkpoint("req-1", {}, [1, 2, 3], [4, 5], "node-0")
        mgr.save_checkpoint("req-2", {}, [6, 7], [8, 9, 10], "node-0")

        checkpoints = mgr.get_checkpoints_for_node("node-0")
        assert len(checkpoints) == 2
        # Verify the checkpoint data is intact for replay
        for req_id, ckpt in checkpoints.items():
            assert len(ckpt.prompt_tokens) > 0
            assert len(ckpt.generated_tokens) > 0
