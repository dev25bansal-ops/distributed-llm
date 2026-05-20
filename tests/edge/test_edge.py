"""Tests for the edge deployment module."""
import pytest
import tempfile
from pathlib import Path

from distllm.edge.models import (
    EdgeConfig,
    QuantizationType,
    ModelShard,
    EdgeHealth,
    EdgeNodeInfo,
)
from distllm.edge.quantized import QuantizationBackend, QuantizedModel
from distllm.edge.sharding import ModelShardManager
from distllm.edge.routing import EdgeRouter, EdgeRouteDecision


class TestEdgeModels:
    def test_quantization_type_values(self):
        assert QuantizationType.INT4.value == "int4"
        assert QuantizationType.INT8.value == "int8"
        assert QuantizationType.NF4.value == "nf4"

    def test_model_shard_defaults(self):
        shard = ModelShard(
            shard_id="s1", model_name="m1", shard_index=0,
            total_shards=2, bytes_size=1024,
        )
        assert shard.quantization == QuantizationType.INT4
        assert shard.device == "cpu"

    def test_edge_health_defaults(self):
        h = EdgeHealth(node_id="e1")
        assert h.healthy is True
        assert h.active_requests == 0

    def test_edge_node_info_defaults(self):
        info = EdgeNodeInfo(node_id="e1")
        assert info.port == 9100
        assert info.device == "cpu"

    def test_edge_config_defaults(self):
        cfg = EdgeConfig()
        assert cfg.enabled is True
        assert cfg.port == 9100
        assert cfg.quantization == QuantizationType.INT4
        assert "llama" in cfg.models[0]


class TestQuantizationBackend:
    def test_int8_roundtrip(self):
        orig = [0.5, -0.5, 1.0, -1.0, 0.0]
        packed = QuantizationBackend.quantize_weights(orig, QuantizationType.INT8)
        deq = QuantizationBackend.dequantize_weights(packed, QuantizationType.INT8, len(orig))
        assert len(deq) == len(orig)
        for a, b in zip(orig, deq):
            assert abs(a - b) < 0.02

    def test_int4_roundtrip(self):
        orig = [0.5, -0.5, 1.0, -1.0, 0.0]
        packed = QuantizationBackend.quantize_weights(orig, QuantizationType.INT4)
        deq = QuantizationBackend.dequantize_weights(packed, QuantizationType.INT4, len(orig))
        assert len(deq) == len(orig)
        for a, b in zip(orig, deq):
            assert abs(a - b) < 0.15

    def test_fp16_roundtrip(self):
        orig = [0.5, -0.5, 1.0, -1.0]
        packed = QuantizationBackend.quantize_weights(orig, QuantizationType.FP16)
        deq = QuantizationBackend.dequantize_weights(packed, QuantizationType.FP16, len(orig))
        assert len(deq) == len(orig)
        for a, b in zip(orig, deq):
            assert abs(a - b) < 0.001

    def test_nf4_roundtrip(self):
        orig = [0.5, -0.5, 0.0, 1.0, -1.0]
        packed = QuantizationBackend.quantize_weights(orig, QuantizationType.NF4)
        deq = QuantizationBackend.dequantize_weights(packed, QuantizationType.NF4, len(orig))
        assert len(deq) == len(orig)


@pytest.mark.asyncio
class TestQuantizedModel:
    async def test_generate_requires_load(self):
        model = QuantizedModel("test-model")
        with pytest.raises(RuntimeError, match="not loaded"):
            await model.generate([])

    async def test_generate_after_load(self):
        model = QuantizedModel("test-model")
        model.load()
        result = await model.generate([{"role": "user", "content": "hello"}])
        assert "choices" in result
        assert result["object"] == "chat.completion"

    async def test_generate_stream(self):
        model = QuantizedModel("test-model")
        model.load()
        chunks = []
        async for chunk in model.generate_stream([{"role": "user", "content": "hello"}]):
            chunks.append(chunk)
        assert len(chunks) > 1
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    async def test_loaded_property(self):
        model = QuantizedModel("test-model")
        assert model.loaded is False
        model.load()
        assert model.loaded is True

    async def test_load_from_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "quant_config.json"
            p.write_text('{"quant_type": "int8", "compression_ratio": 2.0}')
            model = QuantizedModel("test-model")
            model.load(str(Path(tmp)))
            assert model.loaded is True


class TestModelShardManager:
    def test_shard_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ModelShardManager(tmp)
            shards = mgr.shard_model("test", 5000, 1000)
            assert len(shards) == 5
            data = mgr.load_shards("test")
            assert len(data) == 5000

    def test_get_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ModelShardManager(tmp)
            mgr.shard_model("test", 2000, 1000)
            shards = mgr.get_shards("test")
            assert len(shards) == 2

    def test_memory_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ModelShardManager(tmp)
            mgr.shard_model("test", 2000, 1000)
            assert mgr.memory_usage("test") == 2000

    def test_get_shards_missing_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ModelShardManager(tmp)
            assert mgr.get_shards("nonexistent") == []

    def test_remove_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ModelShardManager(tmp)
            mgr.shard_model("test", 1000, 500)
            mgr.remove_model("test")
            assert mgr.get_shards("test") == []


class TestEdgeRouter:
    def test_route_edge_when_idle(self):
        cfg = EdgeConfig(max_concurrent_requests=8)
        router = EdgeRouter(cfg)
        decision = router.decide({"model": "llama-3.2-1b"}, active_requests=1)
        assert decision == EdgeRouteDecision.EDGE

    def test_route_cloud_when_overloaded(self):
        cfg = EdgeConfig(max_concurrent_requests=2)
        router = EdgeRouter(cfg)
        decision = router.decide({"model": "llama-3.2-1b"}, active_requests=5)
        assert decision == EdgeRouteDecision.CLOUD

    def test_route_cloud_when_model_not_deployed(self):
        cfg = EdgeConfig(models=["llama"])
        router = EdgeRouter(cfg)
        decision = router.decide({"model": "unknown-model"})
        assert decision == EdgeRouteDecision.CLOUD

    def test_force_cloud_after_consecutive_overloads(self):
        cfg = EdgeConfig(max_concurrent_requests=2)
        router = EdgeRouter(cfg)
        # 3 consecutive overloads should trigger force cloud
        for _ in range(3):
            router.decide({}, active_requests=5)
        decision = router.decide({}, active_requests=0)
        # Should still be cloud because force_cloud is active
        assert decision == EdgeRouteDecision.CLOUD

    def test_reset(self):
        cfg = EdgeConfig(max_concurrent_requests=1)
        router = EdgeRouter(cfg)
        router.decide({}, active_requests=5)
        router.reset()
        decision = router.decide({}, active_requests=0)
        assert decision == EdgeRouteDecision.EDGE
