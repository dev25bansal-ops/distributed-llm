"""Tests for WideAreaPipeline: WAN-aware P2P forwarding, accumulation, fallback."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
import torch

from distllm.core.wide_area_config import WideAreaConfig
from distllm.core.wide_area_pipeline import WideAreaPipeline


class TestWideAreaConfig:
    """Tests for WideAreaConfig dataclass."""

    def test_defaults(self):
        cfg = WideAreaConfig()
        assert cfg.enabled is False
        assert cfg.p2p_forwarding is True
        assert cfg.token_accumulation is True
        assert cfg.accumulation_window == 3
        assert cfg.wan_timeout_seconds == 120.0
        assert cfg.fallback_to_local is True
        assert cfg.adaptive_batching is True

    def test_custom_values(self):
        cfg = WideAreaConfig(
            enabled=True,
            accumulation_window=5,
            wan_timeout_seconds=60.0,
            fallback_to_local=False,
        )
        assert cfg.enabled is True
        assert cfg.accumulation_window == 5
        assert cfg.wan_timeout_seconds == 60.0
        assert cfg.fallback_to_local is False

    def test_range_of_values(self):
        cfg = WideAreaConfig(
            enabled=True, p2p_forwarding=False, token_accumulation=False,
            accumulation_window=1, wan_timeout_seconds=300.0,
            adaptive_batching=False, latency_sample_interval=5.0,
            fallback_to_local=False, heartbeat_interval_seconds=1.0,
        )
        assert not cfg.p2p_forwarding
        assert not cfg.token_accumulation
        assert cfg.accumulation_window == 1
        assert cfg.wan_timeout_seconds == 300.0
        assert not cfg.adaptive_batching
        assert cfg.latency_sample_interval == 5.0
        assert cfg.heartbeat_interval_seconds == 1.0


def _make_mock_node(node_id="n1", host="10.0.0.1", port=50051, healthy=True):
    node = MagicMock()
    node.node_id = node_id
    node.host = host
    node.port = port
    node.healthy = healthy
    node.use_tls = False
    node.ca_cert = None
    node.client = MagicMock()
    node.async_client = None
    return node


@pytest.fixture
def pipeline():
    wp = WideAreaPipeline(total_layers=32)
    wp._executor = MagicMock()  # parent shutdown expects this
    return wp


class TestWideAreaPipeline:
    """Tests for WideAreaPipeline core operations."""

    def test_init(self, pipeline):
        assert pipeline.wan is not None
        assert pipeline.wan.accumulation_window == 3
        assert pipeline._p2p_clients == {}
        assert pipeline._current_window == 3

    def test_init_custom_config(self):
        cfg = WideAreaConfig(enabled=True, accumulation_window=5)
        wp = WideAreaPipeline(total_layers=32, wan_config=cfg)
        assert wp.wan.accumulation_window == 5

    def test_register_node_with_p2p(self, pipeline):
        with patch.object(pipeline, '_setup_p2p_client') as mock_setup:
            pipeline.register_node(
                "n1", "10.0.0.1", 50051, 0, 8,
                next_node_uri="10.0.0.2:50052",
            )
            assert "n1" in pipeline.nodes
            mock_setup.assert_called_once_with(
                "n1", "10.0.0.2:50052", False, None
            )

    def test_register_node_without_p2p(self, pipeline):
        with patch.object(pipeline, '_setup_p2p_client') as mock_setup:
            pipeline.wan.p2p_forwarding = False
            pipeline.register_node("n1", "10.0.0.1", 50051, 0, 8)
            mock_setup.assert_not_called()

    def test_setup_p2p_client_success(self, pipeline):
        with patch(
            "distllm.core.wide_area_pipeline.AsyncNodeClient"
        ) as mock_client_cls:
            pipeline._setup_p2p_client("n1", "10.0.0.2:50052", False, None)
            mock_client_cls.assert_called_once_with(
                host="10.0.0.2", port=50052, use_tls=False, ca_cert=None
            )
            assert "n1" in pipeline._p2p_clients

    def test_setup_p2p_client_failure(self, pipeline):
        with patch(
            "distllm.core.wide_area_pipeline.AsyncNodeClient",
            side_effect=RuntimeError("Connection failed"),
        ):
            pipeline._setup_p2p_client("n1", "10.0.0.2:50052", False, None)
            assert "n1" not in pipeline._p2p_clients

    def test_setup_p2p_links_from_topology(self, pipeline):
        pipeline.register_node("n1", "10.0.0.1", 50051, 0, 7)
        pipeline.register_node("n2", "10.0.0.2", 50052, 8, 15)
        pipeline.register_node("n3", "10.0.0.3", 50053, 16, 23)
        pipeline.node_order = ["n1", "n2", "n3"]

        with (
            patch.object(pipeline, '_setup_p2p_client') as mock_setup,
            patch.object(pipeline.nodes.get('n1'), 'use_tls', False, create=True),
            patch.object(pipeline.nodes.get('n1'), 'ca_cert', None, create=True),
            patch.object(pipeline.nodes.get('n2'), 'use_tls', False, create=True),
            patch.object(pipeline.nodes.get('n2'), 'ca_cert', None, create=True),
        ):
            count = pipeline.setup_p2p_links_from_topology()
            assert count == 2  # n1->n2, n2->n3
            assert mock_setup.call_count == 2

    def test_setup_p2p_links_empty(self, pipeline):
        pipeline.node_order = []
        count = pipeline.setup_p2p_links_from_topology()
        assert count == 0


class TestLatencyMeasurement:
    """Tests for WAN link latency measurement."""

    def test_measure_link_latency_no_client(self, pipeline):
        lat = pipeline.measure_link_latency("n1", "n2")
        assert lat == -1.0

    def test_measure_link_latency_success(self, pipeline):
        mock_client = MagicMock()
        mock_client.stub.Ping.return_value = MagicMock()
        pipeline._p2p_clients["n1"] = mock_client
        with patch(
            "distllm.core.wide_area_pipeline.time.perf_counter",
            side_effect=[1.0, 1.05],
        ):
            lat = pipeline.measure_link_latency("n1", "n2")
            assert lat == pytest.approx(50.0, rel=0.1)

    def test_measure_link_latency_failure(self, pipeline):
        mock_client = MagicMock()
        mock_client.stub.Ping.side_effect = RuntimeError("Ping failed")
        pipeline._p2p_clients["n1"] = mock_client
        lat = pipeline.measure_link_latency("n1", "n2")
        assert lat == -1.0

    def test_get_estimated_link_rtt_no_samples(self, pipeline):
        rtt = pipeline.get_estimated_link_rtt_ms("n1", "n2")
        assert rtt == -1.0

    def test_get_estimated_link_rtt_with_samples(self, pipeline):
        pipeline._link_latencies[("n1", "n2")] = [10.0, 20.0, 30.0]
        rtt = pipeline.get_estimated_link_rtt_ms("n1", "n2")
        assert rtt == 20.0  # median

    def test_measure_keeps_last_10(self, pipeline):
        mock_client = MagicMock()
        mock_client.stub.Ping.return_value = MagicMock()
        pipeline._p2p_clients["n1"] = mock_client
        with patch(
            "distllm.core.wide_area_pipeline.time.perf_counter",
            side_effect=[float(i) for i in range(22)],
        ):
            for _ in range(15):
                pipeline.measure_link_latency("n1", "n2")
            samples = pipeline._link_latencies[("n1", "n2")]
            assert len(samples) == 10


class TestAdaptiveAccumulation:
    """Tests for adaptive accumulation window adjustment."""

    def test_adjust_no_adaptive(self, pipeline):
        pipeline.wan.adaptive_batching = False
        result = pipeline._adjust_accumulation_window()
        assert result == pipeline._current_window

    def test_adjust_within_interval(self, pipeline):
        pipeline._last_adjustment = time.time()
        result = pipeline._adjust_accumulation_window()
        assert result == pipeline._current_window

    def test_adjust_with_rtt_data(self, pipeline):
        pipeline.register_node("n1", "10.0.0.1", 50051, 0, 7)
        pipeline.register_node("n2", "10.0.0.2", 50052, 8, 15)
        pipeline.node_order = ["n1", "n2"]
        pipeline._link_latencies[("n1", "n2")] = [200.0]
        pipeline._last_adjustment = time.time() - 60
        pipeline.wan.latency_sample_interval = 1.0
        pipeline.wan.accumulation_window = 10

        result = pipeline._adjust_accumulation_window()
        # RTT=200ms, decode=50ms/token -> window = ceil(200/50) = 4
        assert result == 4

    def test_adjust_clamps_to_max(self, pipeline):
        pipeline.register_node("n1", "10.0.0.1", 50051, 0, 7)
        pipeline.register_node("n2", "10.0.0.2", 50052, 8, 15)
        pipeline.node_order = ["n1", "n2"]
        pipeline._link_latencies[("n1", "n2")] = [5000.0]
        pipeline._last_adjustment = time.time() - 60
        pipeline.wan.latency_sample_interval = 1.0
        pipeline.wan.accumulation_window = 10

        result = pipeline._adjust_accumulation_window()
        assert result == 10

    def test_adjust_zero_rtt(self, pipeline):
        pipeline._last_adjustment = time.time() - 60
        pipeline.wan.latency_sample_interval = 1.0
        result = pipeline._adjust_accumulation_window()
        assert result == pipeline._current_window


class TestRunPipelineAsyncP2P:
    """Tests for asynchronous P2P pipeline execution."""

    @pytest.mark.asyncio
    async def test_no_nodes_raises(self, pipeline):
        with pytest.raises(RuntimeError, match="No nodes registered"):
            input_ids = torch.zeros((1, 10), dtype=torch.long)
            await pipeline.run_pipeline_async_p2p(
                input_ids, {}, "req-1"
            )

    @pytest.mark.asyncio
    async def test_single_node_execution(self, pipeline):
        pipeline.nodes["n1"] = _make_mock_node()
        pipeline.node_order = ["n1"]

        with patch.object(pipeline, '_async_execute_node', new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = torch.zeros((1, 10, 100))
            input_ids = torch.zeros((1, 10), dtype=torch.long)
            result = await pipeline.run_pipeline_async_p2p(
                input_ids, {"n1": None}, "req-1"
            )
            assert result.shape == (1, 10, 100)

    @pytest.mark.asyncio
    async def test_three_node_p2p_chain(self, pipeline):
        """With 3+ nodes, middle node uses P2P forwarding."""
        pipeline.nodes["n1"] = _make_mock_node("n1")
        pipeline.nodes["n2"] = _make_mock_node("n2")
        pipeline.nodes["n3"] = _make_mock_node("n3")
        pipeline.node_order = ["n1", "n2", "n3"]
        pipeline.wan.p2p_forwarding = True

        with (
            patch.object(pipeline, '_async_execute_node', new_callable=AsyncMock) as mock_exec,
            patch.object(pipeline, '_execute_with_p2p_forward', new_callable=AsyncMock) as mock_p2p,
            patch.object(pipeline, '_process_forward_response') as mock_proc,
        ):
            mock_exec.return_value = torch.zeros((1, 10, 100))
            mock_p2p.return_value = MagicMock()
            mock_proc.return_value = torch.zeros((1, 10, 100))
            input_ids = torch.zeros((1, 10), dtype=torch.long)
            result = await pipeline.run_pipeline_async_p2p(
                input_ids, {"n1": None, "n2": None, "n3": None}, "req-1"
            )
            assert result.shape == (1, 10, 100)
            # n1 = first node (exec), n2 = middle (P2P), n3 = last (exec)
            assert mock_exec.call_count == 2
            assert mock_p2p.call_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_triggers_fallback(self, pipeline):
        pipeline.nodes["n1"] = _make_mock_node("n1", healthy=False)
        pipeline.node_order = ["n1"]
        pipeline.wan.fallback_to_local = True

        mock_rm = MagicMock()
        mock_rm.check_circuit_breaker.return_value = True
        pipeline.resource_mgr = mock_rm

        with patch.object(pipeline, '_run_local_fallback', new_callable=AsyncMock) as mock_fallback:
            mock_fallback.return_value = torch.zeros((1, 10, 100))
            input_ids = torch.zeros((1, 10), dtype=torch.long)
            result = await pipeline.run_pipeline_async_p2p(
                input_ids, {"n1": None}, "req-1"
            )
            assert result.shape == (1, 10, 100)
            mock_fallback.assert_called_once()

    @pytest.mark.asyncio
    async def test_circuit_breaker_raises_without_fallback(self, pipeline):
        pipeline.nodes["n1"] = _make_mock_node("n1", healthy=False)
        pipeline.node_order = ["n1"]
        pipeline.wan.fallback_to_local = False

        mock_rm = MagicMock()
        mock_rm.check_circuit_breaker.return_value = True
        pipeline.resource_mgr = mock_rm

        input_ids = torch.zeros((1, 10), dtype=torch.long)
        from distllm.errors.types import NodeUnreachableError
        with pytest.raises(NodeUnreachableError):
            await pipeline.run_pipeline_async_p2p(
                input_ids, {"n1": None}, "req-1"
            )

    @pytest.mark.asyncio
    async def test_find_fallback_node(self, pipeline):
        pipeline.nodes["n1"] = _make_mock_node("n1")
        fallback = pipeline._find_fallback_node("n1", pipeline.nodes["n1"])
        assert fallback is None


class TestTokenAccumulation:
    """Tests for token accumulation pipeline."""

    @pytest.mark.asyncio
    async def test_no_draft_model(self, pipeline):
        input_ids = torch.zeros((1, 10), dtype=torch.long)
        with patch.object(pipeline, 'run_pipeline_async_p2p', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = torch.zeros((1, 10, 100))
            result = await pipeline.run_pipeline_accumulated(
                input_ids, {}, "req-1",
            )
            mock_run.assert_called_once()
            assert result.shape == (1, 10, 100)

    @pytest.mark.asyncio
    async def test_no_token_accumulation(self, pipeline):
        pipeline.wan.token_accumulation = False
        input_ids = torch.zeros((1, 10), dtype=torch.long)
        draft_fn = MagicMock(return_value=[1, 2, 3])
        with patch.object(pipeline, 'run_pipeline_async_p2p', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = torch.zeros((1, 10, 100))
            result = await pipeline.run_pipeline_accumulated(
                input_ids, {}, "req-1", draft_model_fn=draft_fn,
            )
            mock_run.assert_called_once()
            assert result.shape == (1, 10, 100)

    @pytest.mark.asyncio
    async def test_with_draft_model(self, pipeline):
        pipeline._current_window = 3
        pipeline.wan.token_accumulation = True
        input_ids = torch.zeros((1, 10), dtype=torch.long)
        prefill_logits = torch.zeros((1, 1, 100))
        draft_fn = MagicMock(return_value=[7, 8, 9])

        with patch.object(pipeline, 'run_pipeline_async_p2p', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = torch.zeros((1, 13, 100))
            result = await pipeline.run_pipeline_accumulated(
                input_ids, {}, "req-1", draft_model_fn=draft_fn,
                prefill_logits=prefill_logits,
            )
            mock_run.assert_called_once()
            draft_fn.assert_called_once_with(prefill_logits, 3)

    @pytest.mark.asyncio
    async def test_empty_draft_falls_back(self, pipeline):
        pipeline.wan.token_accumulation = True
        input_ids = torch.zeros((1, 10), dtype=torch.long)
        draft_fn = MagicMock(return_value=[])

        with patch.object(pipeline, 'run_pipeline_async_p2p', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = torch.zeros((1, 10, 100))
            result = await pipeline.run_pipeline_accumulated(
                input_ids, {}, "req-1", draft_model_fn=draft_fn,
                prefill_logits=torch.zeros((1, 1, 100)),
            )
            mock_run.assert_called_once()
            assert result.shape[1] == 10


class TestLocalFallback:
    """Tests for local fallback on WAN failure."""

    @pytest.mark.asyncio
    async def test_run_local_fallback_no_model(self, pipeline):
        input_ids = torch.zeros((1, 10), dtype=torch.long)
        with pytest.raises(RuntimeError, match="no local model"):
            await pipeline._run_local_fallback(input_ids, "req-1")

    @pytest.mark.asyncio
    async def test_run_local_fallback_with_model(self, pipeline):
        mock_model = MagicMock()
        mock_model.return_value.logits = torch.zeros((1, 10, 100))
        mock_tokenizer = MagicMock()
        pipeline.set_local_fallback_model(mock_model, mock_tokenizer)

        input_ids = torch.zeros((1, 10), dtype=torch.long)
        result = await pipeline._run_local_fallback(input_ids, "req-1")
        assert result.shape == (1, 10, 100)


class TestShutdown:
    """Tests for shutdown behavior."""

    def test_shutdown_clears_p2p(self, pipeline):
        mock_client = MagicMock()
        pipeline._p2p_clients["n1"] = mock_client
        pipeline._p2p_clients["n2"] = mock_client
        pipeline.shutdown()
        assert pipeline._p2p_clients == {}
        assert mock_client.close.call_count == 2

    def test_shutdown_empty(self, pipeline):
        pipeline.shutdown()
        assert pipeline._p2p_clients == {}

    def test_shutdown_handles_close_error(self, pipeline):
        bad_client = MagicMock()
        bad_client.close.side_effect = RuntimeError("close failed")
        pipeline._p2p_clients["n1"] = bad_client
        pipeline.shutdown()  # Should not raise
