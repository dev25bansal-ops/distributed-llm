"""Real tests for block transfer service -- BlockTransferServer, BlockTransferClient, DTOs.

Zero mocks -- all tests use real instances and deterministic logic.
No GPU, no network, no timing-dependent assertions.
"""

from __future__ import annotations

import torch
import pytest

from distllm.dist.block_transfer_service import (
    BlockData,
    BlockTransferClient,
    BlockTransferRequest,
    BlockTransferResponse,
    BlockTransferServer,
    create_fetch_fn,
)


# ── Test helpers (real classes, not mocks) ──────────────────────────


class _FakeKVSlice:
    """Deterministic KV slice provider -- returns fixed tensors per (block, layer)."""

    def __init__(self, num_layers: int = 4):
        self.num_layers = num_layers

    def get_kv_slice(self, block_id: int, layer_idx: int):
        k = torch.full((2, 4, 16, 64), float(block_id), dtype=torch.float16)
        v = torch.full((2, 4, 16, 64), float(layer_idx), dtype=torch.float16)
        return k, v


class _FakeManager:
    """Manager with a valid pool."""

    def __init__(self, num_layers: int = 4):
        self.pool = _FakeKVSlice(num_layers=num_layers)


class _BrokenPool:
    """Pool whose get_kv_slice always raises."""

    num_layers = 2

    def get_kv_slice(self, block_id: int, layer_idx: int):
        raise RuntimeError("internal pool error")


class _BrokenManager:
    """Manager whose pool.get_kv_slice raises."""

    def __init__(self):
        self.pool = _BrokenPool()


class _EmptyManager:
    """Manager with no pool attribute at all."""

    pass


# ── BlockData ───────────────────────────────────────────────────────


class TestBlockData:
    """BlockData dataclass -- construction, attributes, round-trip."""

    def test_attributes(self):
        k = b"\x00\x01" * 8
        v = b"\x02\x03" * 8
        bd = BlockData(
            block_id=42,
            layer_idx=1,
            key_data=k,
            value_data=v,
            key_shape=[2, 4, 16, 64],
            value_shape=[2, 4, 16, 64],
            dtype="torch.float16",
        )
        assert bd.block_id == 42
        assert bd.layer_idx == 1
        assert bd.key_data == k
        assert bd.value_data == v
        assert bd.key_shape == [2, 4, 16, 64]
        assert bd.value_shape == [2, 4, 16, 64]
        assert bd.dtype == "torch.float16"

    def test_roundtrip_tensor(self):
        """Serialising a tensor and reconstructing it yields identical values."""
        k = torch.full((2, 4, 16, 64), 3.0, dtype=torch.float16)
        v = torch.full((2, 4, 16, 64), 7.0, dtype=torch.float16)
        bd = BlockData(
            block_id=1,
            layer_idx=0,
            key_data=k.numpy().tobytes(),
            value_data=v.numpy().tobytes(),
            key_shape=list(k.shape),
            value_shape=list(v.shape),
            dtype=str(k.dtype),
        )
        k_recon = (
            torch.frombuffer(bytearray(bd.key_data), dtype=torch.float16)
            .reshape(bd.key_shape)
            .clone()
        )
        v_recon = (
            torch.frombuffer(bytearray(bd.value_data), dtype=torch.float16)
            .reshape(bd.value_shape)
            .clone()
        )
        assert torch.equal(k_recon, k)
        assert torch.equal(v_recon, v)

    def test_minimal_values(self):
        """BlockData accepts empty bytes and single-element shapes."""
        bd = BlockData(
            block_id=0,
            layer_idx=0,
            key_data=b"",
            value_data=b"",
            key_shape=[],
            value_shape=[],
            dtype="torch.float16",
        )
        assert bd.key_data == b""
        assert bd.key_shape == []


# ── BlockTransferRequest ────────────────────────────────────────────


class TestBlockTransferRequest:
    """BlockTransferRequest dataclass -- defaults and explicit values."""

    def test_defaults(self):
        req = BlockTransferRequest(block_ids=[1, 2, 3])
        assert req.block_ids == [1, 2, 3]
        assert req.layer_indices is None
        assert req.requester_node_id == ""

    def test_explicit_values(self):
        req = BlockTransferRequest(
            block_ids=[10, 20],
            layer_indices=[0, 1],
            requester_node_id="node-42",
        )
        assert req.block_ids == [10, 20]
        assert req.layer_indices == [0, 1]
        assert req.requester_node_id == "node-42"

    def test_empty_block_ids(self):
        req = BlockTransferRequest(block_ids=[])
        assert req.block_ids == []

    def test_single_block_id(self):
        req = BlockTransferRequest(block_ids=[99])
        assert req.block_ids == [99]


# ── BlockTransferResponse ───────────────────────────────────────────


class TestBlockTransferResponse:
    """BlockTransferResponse dataclass -- defaults and composition."""

    def test_defaults(self):
        resp = BlockTransferResponse(blocks=[])
        assert resp.blocks == []
        assert resp.success is True
        assert resp.error == ""

    def test_explicit_failure(self):
        resp = BlockTransferResponse(blocks=[], success=False, error="something broke")
        assert resp.success is False
        assert resp.error == "something broke"

    def test_with_blocks(self):
        bd = BlockData(
            block_id=1,
            layer_idx=0,
            key_data=b"k",
            value_data=b"v",
            key_shape=[1],
            value_shape=[1],
            dtype="torch.float16",
        )
        resp = BlockTransferResponse(blocks=[bd])
        assert len(resp.blocks) == 1
        assert resp.blocks[0] is bd


# ── BlockTransferServer ─────────────────────────────────────────────


class TestBlockTransferServer:
    """Server-side request handling, lifecycle and stats."""

    # -- Error paths --------------------------------------------------

    def test_no_pool_returns_error(self):
        server = BlockTransferServer(_EmptyManager(), port=0)
        req = BlockTransferRequest(block_ids=[1])
        resp = server.handle_request(req)
        assert resp.success is False
        assert resp.error == "No pool available"
        assert resp.blocks == []

    def test_none_manager_returns_error(self):
        server = BlockTransferServer(None, port=0)
        req = BlockTransferRequest(block_ids=[1])
        resp = server.handle_request(req)
        assert resp.success is False
        assert "No pool" in resp.error

    def test_broken_pool_skips_blocks(self):
        """When get_kv_slice raises, inner try/except skips that block/layer."""
        server = BlockTransferServer(_BrokenManager(), port=0)
        req = BlockTransferRequest(block_ids=[1])
        resp = server.handle_request(req)
        # All calls fail -> blocks is empty, but request still "succeeds"
        assert resp.success is True
        assert resp.blocks == []

    # -- Happy path ---------------------------------------------------

    def test_handle_request_success(self):
        server = BlockTransferServer(_FakeManager(num_layers=4), port=0)
        req = BlockTransferRequest(block_ids=[0, 1])
        resp = server.handle_request(req)
        assert resp.success is True
        assert resp.error == ""
        # 2 blocks * 4 layers = 8 BlockData entries
        assert len(resp.blocks) == 8

        # Verify block order: block_id outer loop, layer_idx inner
        assert resp.blocks[0].block_id == 0
        assert resp.blocks[0].layer_idx == 0
        assert resp.blocks[3].block_id == 0
        assert resp.blocks[3].layer_idx == 3
        assert resp.blocks[4].block_id == 1
        assert resp.blocks[4].layer_idx == 0
        assert resp.blocks[7].block_id == 1
        assert resp.blocks[7].layer_idx == 3

        # Verify serialized metadata
        bd = resp.blocks[0]
        assert bd.key_shape == [2, 4, 16, 64]
        assert bd.value_shape == [2, 4, 16, 64]
        assert bd.dtype == "torch.float16"
        assert len(bd.key_data) > 0
        assert len(bd.value_data) > 0

    def test_handle_request_filtered_layers(self):
        server = BlockTransferServer(_FakeManager(num_layers=4), port=0)
        req = BlockTransferRequest(block_ids=[5], layer_indices=[0, 2])
        resp = server.handle_request(req)
        assert resp.success is True
        assert len(resp.blocks) == 2
        assert resp.blocks[0].block_id == 5
        assert resp.blocks[0].layer_idx == 0
        assert resp.blocks[1].block_id == 5
        assert resp.blocks[1].layer_idx == 2

    def test_handle_request_empty_block_ids(self):
        server = BlockTransferServer(_FakeManager(num_layers=2), port=0)
        req = BlockTransferRequest(block_ids=[])
        resp = server.handle_request(req)
        assert resp.success is True
        assert resp.blocks == []

    def test_handle_request_empty_layer_indices_falls_back_to_all(self):
        """An empty list for layer_indices is falsy, so all layers are used."""
        server = BlockTransferServer(_FakeManager(num_layers=3), port=0)
        req = BlockTransferRequest(block_ids=[1], layer_indices=[])
        resp = server.handle_request(req)
        # [] or list(range(3)) -> [0, 1, 2]
        assert len(resp.blocks) == 3
        assert resp.blocks[0].layer_idx == 0
        assert resp.blocks[1].layer_idx == 1
        assert resp.blocks[2].layer_idx == 2

    # -- Serialization round-trip via server --------------------------

    def test_handle_request_serialization_roundtrip(self):
        """Tensors serialised by the server can be fully reconstructed."""
        pool = _FakeKVSlice(num_layers=1)
        k_orig, v_orig = pool.get_kv_slice(7, 0)
        server = BlockTransferServer(_FakeManager(num_layers=1), port=0)
        req = BlockTransferRequest(block_ids=[7], layer_indices=[0])
        resp = server.handle_request(req)
        assert len(resp.blocks) == 1
        bd = resp.blocks[0]
        k_recon = (
            torch.frombuffer(bytearray(bd.key_data), dtype=torch.float16)
            .reshape(bd.key_shape)
            .clone()
        )
        v_recon = (
            torch.frombuffer(bytearray(bd.value_data), dtype=torch.float16)
            .reshape(bd.value_shape)
            .clone()
        )
        assert torch.equal(k_recon, k_orig)
        assert torch.equal(v_recon, v_orig)

    # -- Lifecycle ----------------------------------------------------

    def test_start_stop(self):
        server = BlockTransferServer(_EmptyManager(), port=50051)
        assert server.stats()["running"] is False
        server.start()
        assert server.stats()["running"] is True
        server.stop()
        assert server.stats()["running"] is False

    # -- Stats --------------------------------------------------------

    def test_stats_initial(self):
        server = BlockTransferServer(_EmptyManager(), port=50051)
        s = server.stats()
        assert s["requests_served"] == 0
        assert s["blocks_transferred"] == 0
        assert s["bytes_sent"] == 0
        assert s["errors"] == 0
        assert s["running"] is False
        assert s["port"] == 50051

    def test_stats_after_request(self):
        server = BlockTransferServer(_FakeManager(num_layers=2), port=0)
        req = BlockTransferRequest(block_ids=[0, 1])
        server.handle_request(req)
        s = server.stats()
        assert s["requests_served"] == 1
        assert s["blocks_transferred"] == 4  # 2 blocks * 2 layers
        assert s["bytes_sent"] > 0

    def test_handler_none_pool_does_not_increment_served(self):
        """When pool is None, handle_request returns early without stats."""
        server = BlockTransferServer(_EmptyManager(), port=0)
        req = BlockTransferRequest(block_ids=[1, 2, 3])
        server.handle_request(req)
        s = server.stats()
        # The early return happens before requests_served increments
        assert s["requests_served"] == 0
        # This is not a thrown exception, so errors stays 0
        assert s["errors"] == 0

    # -- repr ---------------------------------------------------------

    def test_repr(self):
        server = BlockTransferServer(_EmptyManager(), port=50051)
        r = repr(server)
        assert "BlockTransferServer" in r
        assert "port=50051" in r
        assert "served=0" in r
        assert "blocks=0" in r


# ── BlockTransferClient ─────────────────────────────────────────────


class TestBlockTransferClient:
    """Client-side fetch operations and stats (no gRPC available in tests)."""

    def test_init(self):
        client = BlockTransferClient("node-1:50051")
        assert client._address == "node-1:50051"
        assert client._timeout == 10.0

        client2 = BlockTransferClient("node-2:50051", timeout_s=5.0)
        assert client2._timeout == 5.0

    def test_fetch_blocks_no_grpc(self):
        client = BlockTransferClient("localhost:9999")
        result = client.fetch_blocks([1, 2, 3])
        assert result is None

    def test_fetch_block_no_grpc(self):
        client = BlockTransferClient("localhost:9999")
        result = client.fetch_block(1)
        assert result is None

    def test_fetch_block_with_layer_idx_no_grpc(self):
        client = BlockTransferClient("localhost:9999")
        result = client.fetch_block(1, layer_idx=2)
        assert result is None

    def test_fetch_blocks_empty_list_no_grpc(self):
        client = BlockTransferClient("localhost:9999")
        result = client.fetch_blocks([])
        assert result is None

    def test_fetch_blocks_none_layer_indices_no_grpc(self):
        client = BlockTransferClient("localhost:9999")
        result = client.fetch_blocks([1], layer_indices=None)
        assert result is None

    # -- Stats --------------------------------------------------------

    def test_stats_initial(self):
        client = BlockTransferClient("remote:50051")
        s = client.stats()
        assert s["requests_made"] == 0
        assert s["blocks_fetched"] == 0
        assert s["bytes_received"] == 0
        assert s["errors"] == 0
        assert s["peer_address"] == "remote:50051"

    def test_stats_after_fetch_blocks(self):
        client = BlockTransferClient("localhost:9999")
        client.fetch_blocks([1])
        s = client.stats()
        assert s["requests_made"] == 1

    def test_stats_after_fetch_block(self):
        client = BlockTransferClient("localhost:9999")
        client.fetch_block(42)
        s = client.stats()
        assert s["requests_made"] == 1

    # -- repr ---------------------------------------------------------

    def test_repr(self):
        client = BlockTransferClient("peer:50051")
        r = repr(client)
        assert "BlockTransferClient" in r
        assert "peer=peer:50051" in r
        assert "requests=0" in r


# ── create_fetch_fn ─────────────────────────────────────────────────


class TestCreateFetchFn:
    """Factory for DistributedBlockFetcher-compatible callables."""

    def test_returns_callable(self):
        client = BlockTransferClient("localhost:9999")
        fn = create_fetch_fn(client)
        assert callable(fn)

    def test_fetch_fn_delegates_to_client(self):
        client = BlockTransferClient("localhost:9999")
        fn = create_fetch_fn(client)
        result = fn(42, "peer-node")
        assert result is None  # no gRPC available
        assert client.stats()["requests_made"] == 1

    def test_fetch_fn_ignores_peer_node_id(self):
        """Second argument is accepted but unused by the closure."""
        client = BlockTransferClient("localhost:9999")
        fn = create_fetch_fn(client)
        r1 = fn(1, "any-node")
        r2 = fn(1, "different-node")
        # Both calls behave identically regardless of peer_node_id
        assert r1 is None
        assert r2 is None


# ── Test count verification ─────────────────────────────────────────


def test_test_count():
    """Verify the file contains at least 20 test functions."""
    import re
    from pathlib import Path

    content = Path(__file__).read_text()
    tests = re.findall(r"def test_", content)
    assert len(tests) >= 28, f"Found {len(tests)} tests, need >= 28"
