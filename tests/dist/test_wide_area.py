"""Tests for WideAreaPipeline — WAN-aware distributed inference."""

from __future__ import annotations

import asyncio
import statistics

import pytest
import torch

from distllm.dist.wide_area import WideAreaPipeline
from distllm.dist.config import WideAreaConfig

VOCAB = 100


# ── Real (non-mock) helper callables used by draft/target forwards ──


def _draft_always_token_zero(prefix: torch.Tensor, **kwargs) -> torch.Tensor:
    """Draft model that always assigns highest probability to token 0."""
    logits = torch.full((1, 1, VOCAB), -10.0)
    logits[:, :, 0] = 10.0
    return logits


class _FakeLatencyTracker:
    """Stand-in for a real LatencyTracker; not a mock."""

    def get_recent_latencies(self, limit: int = 20) -> list[float]:
        return [45.0, 48.0, 52.0, 47.0, 49.0, 51.0]


class _FakeEmptyLatencyTracker:
    """Tracker that returns too few samples to calibrate."""

    def get_recent_latencies(self, limit: int = 20) -> list[float]:
        return []


class _FakeFailingLatencyTracker:
    """Tracker that raises on access (simulates transient error)."""

    def get_recent_latencies(self, limit: int = 20) -> list[float]:
        msg = "transient tracker error"
        raise RuntimeError(msg)


# ── WideAreaConfig ──


class TestWideAreaConfig:
    """WideAreaConfig — pydantic settings for WAN pipeline."""

    def test_default_values(self) -> None:
        config = WideAreaConfig()
        assert config.enabled is False
        assert config.p2p_forwarding is True
        assert config.token_accumulation is True
        assert config.accumulation_window == 3
        assert config.wan_timeout_seconds == 120.0
        assert config.max_accumulation_retries == 3
        assert config.adaptive_batching is True
        assert config.latency_sample_interval == 10.0
        assert config.fallback_to_local is True
        assert config.compression_level == 2
        assert config.heartbeat_interval_seconds == 5.0
        assert config.transport == "auto"

    def test_custom_values(self) -> None:
        config = WideAreaConfig(
            enabled=True,
            accumulation_window=8,
            wan_timeout_seconds=60.0,
            adaptive_batching=False,
            transport="grpc",
            compression_level=0,
        )
        assert config.enabled is True
        assert config.accumulation_window == 8
        assert config.wan_timeout_seconds == 60.0
        assert config.adaptive_batching is False
        assert config.transport == "grpc"
        assert config.compression_level == 0

    def test_accumulation_window_minimum_one(self) -> None:
        config = WideAreaConfig(accumulation_window=1)
        assert config.accumulation_window == 1

    def test_compression_level_zero_one_two(self) -> None:
        for level in (0, 1, 2):
            config = WideAreaConfig(compression_level=level)
            assert config.compression_level == level

    def test_compression_level_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="compression_level"):
            WideAreaConfig(compression_level=3)
        with pytest.raises(ValueError, match="compression_level"):
            WideAreaConfig(compression_level=-1)

    def test_enabled_edge_case(self) -> None:
        """Explicit False stays False, True stays True."""
        assert WideAreaConfig(enabled=False).enabled is False
        assert WideAreaConfig(enabled=True).enabled is True

    def test_float_fields_accept_ints(self) -> None:
        """wan_timeout_seconds accepts int (pydantic coercion)."""
        config = WideAreaConfig(wan_timeout_seconds=30)
        assert isinstance(config.wan_timeout_seconds, float)
        assert config.wan_timeout_seconds == 30.0


# ── WideAreaPipeline ──

class TestWideAreaPipelineConstruction:
    """Construction with various argument combinations."""

    def test_default_init(self) -> None:
        pipeline = WideAreaPipeline()
        assert pipeline.total_layers == 0
        assert isinstance(pipeline.wan, WideAreaConfig)
        assert pipeline.wan.enabled is False
        assert pipeline.quic_transport is None

    def test_init_with_wan_config(self) -> None:
        config = WideAreaConfig(enabled=True, transport="grpc")
        pipeline = WideAreaPipeline(wan_config=config)
        assert pipeline.wan is config
        assert pipeline.wan.enabled is True
        assert pipeline.wan.transport == "grpc"

    def test_init_with_quic_transport_object(self) -> None:
        pipeline = WideAreaPipeline(quic_transport="dummy-transport")
        assert pipeline.quic_transport == "dummy-transport"

    def test_init_with_latency_tracker(self) -> None:
        tracker = _FakeLatencyTracker()
        pipeline = WideAreaPipeline(latency_tracker=tracker)
        assert pipeline._latency_tracker is tracker  # noqa: SLF001

    def test_init_with_resource_mgr_none(self) -> None:
        """Passing None for resource_mgr should not crash."""
        pipeline = WideAreaPipeline(resource_mgr=None)
        assert pipeline is not None

    def test_init_wan_disabled_does_not_auto_init_quic(self) -> None:
        """When wan.enabled is False (default), _auto_init_quic is skipped."""
        pipeline = WideAreaPipeline()
        assert pipeline.quic_transport is None

    def test_init_wan_enabled_auto_init_grpc_path(self) -> None:
        """enabled=True + transport='grpc' skips QUIC init."""
        config = WideAreaConfig(enabled=True, transport="grpc")
        pipeline = WideAreaPipeline(wan_config=config)
        assert pipeline.quic_transport is None

    def test_init_wan_enabled_auto_path_no_aioquic(self) -> None:
        """enabled=True + transport='auto' (default) with aioquic absent."""
        config = WideAreaConfig(enabled=True)
        pipeline = WideAreaPipeline(wan_config=config)
        # quic_transport may be None (aioquic not installed) or a client
        # (if aioquic is available). Either way, no exception.
        assert pipeline.wan.enabled is True

    def test_init_wan_enabled_quic_forced_no_aioquic(self) -> None:
        """enabled=True + transport='quic' should raise if aioquic missing."""
        config = WideAreaConfig(enabled=True, transport="quic")
        with pytest.raises(RuntimeError, match="QUIC transport requested"):
            WideAreaPipeline(wan_config=config)

    def test_multiple_instances_independent(self) -> None:
        p1 = WideAreaPipeline()
        p2 = WideAreaPipeline()
        assert p1 is not p2
        assert p1.wan is not p2.wan
        assert p1._current_window == p2._current_window  # noqa: SLF001


class TestWideAreaPipelineProperties:
    """quic_transport property getter/setter."""

    def test_quic_transport_default_none(self) -> None:
        pipeline = WideAreaPipeline()
        assert pipeline.quic_transport is None

    def test_quic_transport_setter(self) -> None:
        pipeline = WideAreaPipeline()
        pipeline.quic_transport = "custom-value"
        assert pipeline.quic_transport == "custom-value"
        assert pipeline._quic_transport == "custom-value"  # noqa: SLF001

    def test_quic_transport_set_none(self) -> None:
        pipeline = WideAreaPipeline(quic_transport="old")
        pipeline.quic_transport = None
        assert pipeline.quic_transport is None

    def test_quic_transport_set_multiple_times(self) -> None:
        pipeline = WideAreaPipeline()
        for val in ("a", "b", None, "c"):
            pipeline.quic_transport = val
            assert pipeline.quic_transport == val


class TestWideAreaPipelineLatency:
    """Latency measurement methods (get_estimated_link_rtt_ms,
    get_measured_latency, get_latency_stats)."""

    # -- get_estimated_link_rtt_ms ---------------------------------------

    def test_estimated_rtt_no_samples(self) -> None:
        pipeline = WideAreaPipeline()
        rtt = pipeline.get_estimated_link_rtt_ms("node-a", "node-b")
        assert rtt == -1.0

    def test_estimated_rtt_with_samples(self) -> None:
        pipeline = WideAreaPipeline()
        pipeline._link_latencies[("a", "b")] = [10.0, 20.0, 30.0]  # noqa: SLF001
        rtt = pipeline.get_estimated_link_rtt_ms("a", "b")
        assert rtt == 20.0  # median

    def test_estimated_rtt_reverse_key(self) -> None:
        pipeline = WideAreaPipeline()
        pipeline._link_latencies[("a", "b")] = [100.0]  # noqa: SLF001
        rtt = pipeline.get_estimated_link_rtt_ms("b", "a")
        assert rtt == -1.0  # reverse lookup NOT performed

    def test_estimated_rtt_single_sample(self) -> None:
        pipeline = WideAreaPipeline()
        pipeline._link_latencies[("x", "y")] = [42.0]  # noqa: SLF001
        assert pipeline.get_estimated_link_rtt_ms("x", "y") == 42.0

    def test_estimated_rtt_large_sample_set(self) -> None:
        pipeline = WideAreaPipeline()
        samples = [float(i) for i in range(1, 101)]  # 1..100
        pipeline._link_latencies[("a", "b")] = samples  # noqa: SLF001
        assert pipeline.get_estimated_link_rtt_ms("a", "b") == 50.5

    # -- get_measured_latency --------------------------------------------

    def test_measured_latency_no_samples(self) -> None:
        pipeline = WideAreaPipeline()
        lat = pipeline.get_measured_latency("a", "b")
        assert lat is None

    def test_measured_latency_with_samples(self) -> None:
        pipeline = WideAreaPipeline()
        pipeline._link_latencies[("a", "b")] = [10.0, 20.0, 30.0, 99.0]  # noqa: SLF001
        # median of last 10 (all 4): 25.0
        assert pipeline.get_measured_latency("a", "b") == 25.0

    def test_measured_latency_reverse_key(self) -> None:
        pipeline = WideAreaPipeline()
        pipeline._link_latencies[("b", "a")] = [15.0, 25.0, 35.0]  # noqa: SLF001
        lat = pipeline.get_measured_latency("a", "b")
        assert lat == 25.0  # reverse lookup

    def test_measured_latency_no_match_either_direction(self) -> None:
        pipeline = WideAreaPipeline()
        pipeline._link_latencies[("c", "d")] = [50.0]  # noqa: SLF001
        assert pipeline.get_measured_latency("a", "b") is None

    # -- get_latency_stats -----------------------------------------------

    def test_latency_stats_empty(self) -> None:
        pipeline = WideAreaPipeline()
        stats = pipeline.get_latency_stats()
        assert stats == {}

    def test_latency_stats_with_data(self) -> None:
        pipeline = WideAreaPipeline()
        pipeline._link_latencies[("a", "b")] = [10.0, 20.0]  # noqa: SLF001
        pipeline._link_latencies[("c", "d")] = [5.0, 15.0]  # noqa: SLF001
        stats = pipeline.get_latency_stats()
        assert "a→b" in stats
        assert "c→d" in stats
        assert stats["a→b"]["median_ms"] == 15.0
        assert stats["a→b"]["min_ms"] == 10.0
        assert stats["a→b"]["max_ms"] == 20.0
        assert stats["a→b"]["samples"] == 2
        assert stats["c→d"]["median_ms"] == 10.0

    def test_latency_stats_unicode_arrow_present(self) -> None:
        pipeline = WideAreaPipeline()
        pipeline._link_latencies[("x", "y")] = [30.0]  # noqa: SLF001
        stats = pipeline.get_latency_stats()
        key = "x→y"
        assert key in stats

    def test_latency_stats_many_samples(self) -> None:
        pipeline = WideAreaPipeline()
        samples = [float(i) for i in range(50)]
        pipeline._link_latencies[("a", "b")] = samples  # noqa: SLF001
        stats = pipeline.get_latency_stats()
        assert stats["a→b"]["median_ms"] == statistics.median(samples[-20:])
        assert stats["a→b"]["min_ms"] == min(samples[-20:])
        assert stats["a→b"]["max_ms"] == max(samples[-20:])
        assert stats["a→b"]["samples"] == 50


class TestWideAreaPipelineCalibrate:
    """_calibrate_decode_ms — internal decode-time calibration."""

    def test_default_no_tracker(self) -> None:
        pipeline = WideAreaPipeline()
        assert pipeline._calibrate_decode_ms() == 50.0  # noqa: SLF001

    def test_tracker_explicitly_none(self) -> None:
        pipeline = WideAreaPipeline(latency_tracker=None)
        assert pipeline._calibrate_decode_ms() == 50.0  # noqa: SLF001

    def test_with_real_tracker_data(self) -> None:
        pipeline = WideAreaPipeline(latency_tracker=_FakeLatencyTracker())
        # median of [45, 48, 52, 47, 49, 51] = 48.5... wait, let me calculate
        # sorted: [45, 47, 48, 49, 51, 52]
        # median of 6 items = (48 + 49) / 2 = 48.5
        assert pipeline._calibrate_decode_ms() == 48.5  # noqa: SLF001

    def test_insufficient_data_returns_default(self) -> None:
        pipeline = WideAreaPipeline(latency_tracker=_FakeEmptyLatencyTracker())
        assert pipeline._calibrate_decode_ms() == 50.0  # noqa: SLF001

    def test_tracker_raises_returns_default(self) -> None:
        pipeline = WideAreaPipeline(latency_tracker=_FakeFailingLatencyTracker())
        assert pipeline._calibrate_decode_ms() == 50.0  # noqa: SLF001

    def test_result_is_at_least_one(self) -> None:
        """Calibrate should floor at 1.0 ms."""
        class _TinyTracker:  # noqa: N801
            @staticmethod
            def get_recent_latencies(limit: int = 20) -> list[float]:
                return [0.01, 0.02, 0.03, 0.04, 0.05]

        pipeline = WideAreaPipeline(latency_tracker=_TinyTracker())
        assert pipeline._calibrate_decode_ms() >= 1.0  # noqa: SLF001


class TestWideAreaPipelineAdjustWindow:
    """_adjust_accumulation_window — adaptive batching window."""

    def test_adaptive_disabled(self) -> None:
        config = WideAreaConfig(adaptive_batching=False, accumulation_window=7)
        pipeline = WideAreaPipeline(wan_config=config)
        assert pipeline._adjust_accumulation_window() == 7  # noqa: SLF001

    def test_adaptive_enabled_no_nodes(self) -> None:
        """With no registered nodes, max_rtt stays 0 -> current window."""
        config = WideAreaConfig(adaptive_batching=True, accumulation_window=5)
        pipeline = WideAreaPipeline(wan_config=config)
        result = pipeline._adjust_accumulation_window()  # noqa: SLF001
        assert result == 5

    def test_within_sample_interval_returns_cached(self) -> None:
        """When _last_adjustment is recent, returns _current_window."""
        config = WideAreaConfig(adaptive_batching=True, latency_sample_interval=9999.0)
        pipeline = WideAreaPipeline(wan_config=config)
        r1 = pipeline._adjust_accumulation_window()  # noqa: SLF001
        r2 = pipeline._adjust_accumulation_window()  # noqa: SLF001
        assert r2 == r1

    def test_current_window_persists(self) -> None:
        pipeline = WideAreaPipeline()
        pipeline._current_window = 42  # noqa: SLF001
        assert pipeline._adjust_accumulation_window() == 42  # noqa: SLF001

    def test_current_window_min_one(self) -> None:
        """Adaptive logic floors to 1."""
        config = WideAreaConfig(adaptive_batching=True, accumulation_window=1)
        pipeline = WideAreaPipeline(wan_config=config)
        assert pipeline._adjust_accumulation_window() == 1  # noqa: SLF001


class TestWideAreaPipelineShutdown:
    """shutdown — cleanup."""

    def test_shutdown_clears_nodes(self) -> None:
        pipeline = WideAreaPipeline()
        pipeline.register_node("n0", "host0", 5000, 0, 5)
        assert len(pipeline.nodes) > 0
        pipeline.shutdown()
        assert len(pipeline.nodes) == 0

    def test_shutdown_empty_no_error(self) -> None:
        pipeline = WideAreaPipeline()
        pipeline.shutdown()  # should not raise

    def test_shutdown_idempotent(self) -> None:
        pipeline = WideAreaPipeline()
        pipeline.shutdown()
        pipeline.shutdown()  # second call should not raise

    def test_shutdown_after_node_registration(self) -> None:
        pipeline = WideAreaPipeline()
        pipeline.register_node("n0", "h0", 5001, 0, 2)
        pipeline.register_node("n1", "h1", 5002, 3, 5)
        pipeline.shutdown()
        assert pipeline.node_order == []


class TestWideAreaPipelineFallback:
    """Local fallback model setup and error paths."""

    def test_run_local_fallback_no_model_raises(self) -> None:
        pipeline = WideAreaPipeline()
        with pytest.raises(RuntimeError, match="no local model available"):
            asyncio.run(
                pipeline._run_local_fallback(  # noqa: SLF001
                    torch.tensor([[1, 2, 3]]), "test-req"
                )
            )

    def test_set_local_fallback_model_and_tokenizer(self) -> None:
        pipeline = WideAreaPipeline()
        model = object()
        tokenizer = object()
        pipeline.set_local_fallback_model(model, tokenizer)
        assert pipeline._local_model is model  # noqa: SLF001
        assert pipeline._local_tokenizer is tokenizer  # noqa: SLF001

    def test_set_local_fallback_model_none(self) -> None:
        pipeline = WideAreaPipeline()
        pipeline.set_local_fallback_model(None, None)
        assert pipeline._local_model is None  # noqa: SLF001
        assert pipeline._local_tokenizer is None  # noqa: SLF001
        # Still raises because model is None
        with pytest.raises(RuntimeError, match="no local model available"):
            asyncio.run(
                pipeline._run_local_fallback(  # noqa: SLF001
                    torch.tensor([[1, 2, 3]]), "test-req"
                )
            )

    def test_set_local_fallback_model_twice_overwrites(self) -> None:
        pipeline = WideAreaPipeline()
        pipeline.set_local_fallback_model("first", "tok-first")
        pipeline.set_local_fallback_model("second", "tok-second")
        assert pipeline._local_model == "second"  # noqa: SLF001


class TestWideAreaPipelineSpeculative:
    """run_pipeline_speculative — WAN speculative decoding integration.

    Tests use max_new_tokens=0 to bypass the generation loop (no WAN call),
    verifying wiring and parameter forwarding only.
    """

    @staticmethod
    def _run(coro):
        return asyncio.run(coro)

    def test_zero_new_tokens_returns_prompt(self) -> None:
        pipeline = WideAreaPipeline()
        input_ids = torch.tensor([[1, 2, 3]])
        result = self._run(
            pipeline.run_pipeline_speculative(
                input_ids,
                {},
                "test-0",
                draft_model_fn=_draft_always_token_zero,
                max_new_tokens=0,
            )
        )
        assert result.shape == (1, 3)
        assert (result == input_ids).all()

    def test_input_not_mutated(self) -> None:
        pipeline = WideAreaPipeline()
        input_ids = torch.tensor([[1, 2, 3]])
        original = input_ids.clone()
        self._run(
            pipeline.run_pipeline_speculative(
                input_ids,
                {},
                "test-1",
                draft_model_fn=_draft_always_token_zero,
                max_new_tokens=0,
            )
        )
        assert (input_ids == original).all()

    def test_with_custom_kwargs(self) -> None:
        pipeline = WideAreaPipeline()
        input_ids = torch.tensor([[5]])
        result = self._run(
            pipeline.run_pipeline_speculative(
                input_ids,
                {},
                "test-2",
                draft_model_fn=_draft_always_token_zero,
                max_new_tokens=0,
                temperature=0.5,
                num_candidates=4,
            )
        )
        assert (result == input_ids).all()

    def test_single_token_prompt(self) -> None:
        pipeline = WideAreaPipeline()
        input_ids = torch.tensor([[42]])
        result = self._run(
            pipeline.run_pipeline_speculative(
                input_ids,
                {},
                "test-3",
                draft_model_fn=_draft_always_token_zero,
                max_new_tokens=0,
            )
        )
        assert (result == input_ids).all()

    def test_empty_node_kv_caches(self) -> None:
        pipeline = WideAreaPipeline()
        input_ids = torch.tensor([[1, 2, 3]])
        result = self._run(
            pipeline.run_pipeline_speculative(
                input_ids,
                {},
                "test-4",
                draft_model_fn=_draft_always_token_zero,
                max_new_tokens=0,
            )
        )
        assert (result == input_ids).all()


class TestWideAreaPipelineAutoDiscovery:
    """Auto-discovery start and static discovery method."""

    def test_start_auto_discovery_sets_flag(self) -> None:
        pipeline = WideAreaPipeline()
        assert pipeline._auto_discovery_running is False  # noqa: SLF001
        pipeline.start_auto_discovery()
        assert pipeline._auto_discovery_running is True  # noqa: SLF001
        pipeline._auto_discovery_running = False  # noqa: SLF001; cleanup

    def test_start_auto_discovery_twice_no_error(self) -> None:
        pipeline = WideAreaPipeline()
        pipeline.start_auto_discovery()
        # Second call should return immediately (already running)
        pipeline.start_auto_discovery()
        assert pipeline._auto_discovery_running is True  # noqa: SLF001
        pipeline._auto_discovery_running = False  # noqa: SLF001; cleanup

    def test_discover_nodes_static_no_zeroconf(self) -> None:
        """Static _discover_nodes returns [] when zeroconf unavailable."""
        result = WideAreaPipeline._discover_nodes("_distllm._tcp", 5353)
        assert isinstance(result, list)
        # Either empty list (ImportError caught) or potentially discovered nodes
        # In CI without zeroconf it should always be empty.
        assert len(result) == 0


class TestWideAreaPipelineErrorPaths:
    """Methods that fail due to missing setup (test error type and message)."""

    def test_run_async_p2p_missing_topology_lock(self) -> None:
        """run_pipeline_async_p2p accesses self._topology_lock which does not
        exist on the parent PipelineOrchestrator, raising AttributeError."""
        pipeline = WideAreaPipeline()
        input_ids = torch.tensor([[1, 2, 3]])
        with pytest.raises(AttributeError):
            asyncio.run(
                pipeline.run_pipeline_async_p2p(input_ids, {}, "test-fail")
            )

    def test_run_accumulated_missing_topology_lock(self) -> None:
        """run_pipeline_accumulated calls run_pipeline_async_p2p which also
        fails on missing _topology_lock."""
        pipeline = WideAreaPipeline()
        input_ids = torch.tensor([[1, 2, 3]])
        with pytest.raises(AttributeError):
            asyncio.run(
                pipeline.run_pipeline_accumulated(input_ids, {}, "test-fail")
            )


class TestWideAreaPipelineNodeManagement:
    """Node registration and lookup (inherited from PipelineOrchestrator)."""

    def test_register_node(self) -> None:
        pipeline = WideAreaPipeline()
        pipeline.register_node("n0", "10.0.0.1", 5001, 0, 5)
        assert "n0" in pipeline.nodes
        node_info = pipeline.nodes["n0"]
        # PipelineNode dataclass — attribute access.
        assert node_info.host == "10.0.0.1"
        assert node_info.port == 5001

    def test_register_multiple_nodes(self) -> None:
        pipeline = WideAreaPipeline()
        pipeline.register_node("n0", "h0", 5000, 0, 3)
        pipeline.register_node("n1", "h1", 5001, 4, 7)
        assert len(pipeline.nodes) == 2
        assert pipeline.node_order == ["n0", "n1"]

    def test_node_order_by_layer(self) -> None:
        pipeline = WideAreaPipeline()
        pipeline.register_node("n1", "h1", 5001, 4, 7)
        pipeline.register_node("n0", "h0", 5000, 0, 3)
        assert pipeline.node_order == ["n0", "n1"]

    def test_unregister_node(self) -> None:
        pipeline = WideAreaPipeline()
        pipeline.register_node("n0", "h0", 5000, 0, 3)
        pipeline.register_node("n1", "h1", 5001, 4, 7)
        pipeline.unregister_node("n0")
        assert "n0" not in pipeline.nodes
        assert pipeline.node_order == ["n1"]

    def test_unregister_nonexistent(self) -> None:
        pipeline = WideAreaPipeline()
        pipeline.unregister_node("nonexistent")  # should not raise
