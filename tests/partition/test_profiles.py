"""Tests for GPUProfiler."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from distllm.dist.partition.profiles import GPUProfile, GPUProfiler, LayerWeights


class TestGPUProfilerEstimateWeights:
    def test_layer_count(self):
        profiler = GPUProfiler()
        weights = profiler.estimate_layer_weights(
            hidden_size=1024, intermediate_size=4096,
            num_layers=4, num_heads=8, head_dim=64, vocab_size=10000,
        )
        assert len(weights) == 6  # embed + 4 transformer + lm_head

    def test_layer_types(self):
        profiler = GPUProfiler()
        weights = profiler.estimate_layer_weights(num_layers=4)
        assert weights[0].layer_type == "embed"
        assert weights[1].layer_type == "transformer"
        assert weights[-1].layer_type == "lm_head"

    def test_sequential_ids(self):
        profiler = GPUProfiler()
        weights = profiler.estimate_layer_weights(num_layers=3)
        for i, w in enumerate(weights):
            assert w.layer_id == i

    def test_transformer_flops_positive(self):
        profiler = GPUProfiler()
        weights = profiler.estimate_layer_weights()
        for w in weights:
            if w.layer_type == "transformer":
                assert w.flops_per_token > 0

    def test_kv_cache_only_transformer(self):
        profiler = GPUProfiler()
        weights = profiler.estimate_layer_weights(hidden_size=1024, num_heads=8, head_dim=64)
        for w in weights:
            if w.layer_type == "transformer":
                assert w.kv_cache_bytes_per_token > 0
            else:
                assert w.kv_cache_bytes_per_token == 0

    def test_weight_scales_with_hidden(self):
        profiler = GPUProfiler()
        small = profiler.estimate_layer_weights(hidden_size=1024, num_layers=2)
        large = profiler.estimate_layer_weights(hidden_size=4096, num_layers=2)
        assert large[1].weight_memory_bytes > small[1].weight_memory_bytes * 4

    def test_embed_scales_with_vocab(self):
        profiler = GPUProfiler()
        small = profiler.estimate_layer_weights(vocab_size=1000)
        large = profiler.estimate_layer_weights(vocab_size=32000)
        assert large[0].weight_memory_bytes > small[0].weight_memory_bytes


class TestGPUProfilerProfileGPUs:
    @patch.object(GPUProfiler, "_device_count", return_value=2)
    @patch.object(GPUProfiler, "_get_device_name", side_effect=["H100", "A100"])
    @patch.object(GPUProfiler, "_get_memory_info", return_value=(80 * 1024**3, 40 * 1024**3))
    @patch.object(GPUProfiler, "_bench_matmul_fp16", return_value=0.0)
    @patch.object(GPUProfiler, "_bench_memory_bandwidth", return_value=0.0)
    @patch.object(GPUProfiler, "_bench_strided_bandwidth", return_value=0.0)
    @patch.object(GPUProfiler, "_bench_p2p_bandwidth", return_value=0.0)
    @patch.object(GPUProfiler, "_bench_cpu_flops", return_value=0.0)
    def test_profile_known_gpus(self, *_):
        profiler = GPUProfiler()
        profiles = profiler.profile_all_gpus()
        assert len(profiles) == 2
        assert profiles[0].name == "H100"
        assert profiles[1].name == "A100"

    @patch.object(GPUProfiler, "_device_count", return_value=0)
    def test_empty_fallback(self, _):
        profiler = GPUProfiler()
        profiles = profiler.profile_all_gpus()
        assert len(profiles) == 1
        assert profiles[0].name == "cpu_fallback"


class TestGPUProfileSpecs:
    def test_known_gpu_specs_present(self):
        from distllm.dist.partition.profiles import _KNOWN_GPU_SPECS
        assert "H100" in _KNOWN_GPU_SPECS
        assert "A100" in _KNOWN_GPU_SPECS
        assert "RTX 4090" in _KNOWN_GPU_SPECS
        assert "MI300X" in _KNOWN_GPU_SPECS

    def test_v100_tflops_correct(self):
        from distllm.dist.partition.profiles import _KNOWN_GPU_SPECS
        assert _KNOWN_GPU_SPECS["V100"][0] == 125.0

    def test_a40_bw_correct(self):
        from distllm.dist.partition.profiles import _KNOWN_GPU_SPECS
        assert _KNOWN_GPU_SPECS["A40"][2] == 696.0

    def test_a30_bw_correct(self):
        from distllm.dist.partition.profiles import _KNOWN_GPU_SPECS
        assert _KNOWN_GPU_SPECS["A30"][2] == 933.0


class TestLayerWeights:
    def test_total_memory(self):
        lw = LayerWeights(layer_id=0, weight_memory_bytes=100, activation_memory_bytes=50)
        assert lw.total_memory_bytes == 150

    def test_defaults(self):
        lw = LayerWeights(layer_id=5)
        assert lw.layer_type == "transformer"


class TestGPUProfile:
    def test_defaults(self):
        p = GPUProfile(gpu_id=0, name="Test")
        assert p.total_memory_bytes == 0
        assert p.compute_tflops == 0.0

    def test_custom(self):
        p = GPUProfile(gpu_id=1, name="H100", total_memory_bytes=80 * 1024**3, compute_tflops=989.0)
        assert p.compute_tflops == 989.0
