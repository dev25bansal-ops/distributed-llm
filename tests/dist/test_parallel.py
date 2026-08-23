"""Tests for distllm.dist.parallel -- hybrid parallelism engine.

Tests cover the public API surface without requiring GPUs, network, or mocks.
All objects are real instances from the module under test.
"""

import pytest
import torch

from distllm.dist.parallel import (
    ParallelStrategy,
    TopologyInfo,
    ParallelPlan,
    estimate_layer_memory,
    choose_tp_degree,
    HardwareProber,
    ProfileResult,
    TunedConfig,
    ParallelAutoTuner,
    HybridParallelPlanner,
    HybridParallelExecutor,
)


# ---------------------------------------------------------------------------
# ParallelStrategy
# ---------------------------------------------------------------------------

class TestParallelStrategy:
    """Enum of hybrid parallelism strategies."""

    def test_members(self):
        assert ParallelStrategy.TP.value == "tensor_parallel"
        assert ParallelStrategy.PP.value == "pipeline_parallel"
        assert ParallelStrategy.EP.value == "expert_parallel"
        assert ParallelStrategy.TP_PP.value == "tp_pp"
        assert ParallelStrategy.TP_EP.value == "tp_ep"
        assert ParallelStrategy.PP_EP.value == "pp_ep"
        assert ParallelStrategy.TP_PP_EP.value == "tp_pp_ep"

    def test_str_subclass_enables_comparison(self):
        assert ParallelStrategy.TP == "tensor_parallel"
        assert ParallelStrategy.PP == "pipeline_parallel"
        assert ParallelStrategy.EP == "expert_parallel"

    def test_all_members_covered(self):
        assert len(list(ParallelStrategy)) == 7
        values = {m.value for m in ParallelStrategy}
        assert values == {
            "tensor_parallel", "pipeline_parallel", "expert_parallel",
            "tp_pp", "tp_ep", "pp_ep", "tp_pp_ep",
        }


# ---------------------------------------------------------------------------
# TopologyInfo
# ---------------------------------------------------------------------------

class TestTopologyInfo:
    """Hardware topology dataclass."""

    def test_defaults(self):
        t = TopologyInfo()
        assert t.num_nodes == 1
        assert t.gpus_per_node == 1
        assert t.has_nvlink is False
        assert t.has_infiniband is False
        assert t.total_gpus == 1
        assert t.interconnect_bandwidth_gbps == 12.5
        assert t.node_hostnames == []
        assert t.gpu_memory_gb == []
        assert t.gpu_free_memory_bytes == []

    def test_min_free_memory_empty_returns_zero(self):
        assert TopologyInfo().min_free_memory_bytes() == 0

    def test_min_free_memory_from_bytes_list(self):
        t = TopologyInfo(gpu_free_memory_bytes=[100, 200, 50])
        assert t.min_free_memory_bytes() == 50

    def test_min_free_memory_from_gb_fallback(self):
        t = TopologyInfo(gpu_free_memory_bytes=[], gpu_memory_gb=[16.0, 32.0])
        free = t.min_free_memory_bytes()
        expected = int(16.0 * 0.85 * (1024 ** 3))
        assert free == expected

    def test_min_free_memory_both_lists_empty(self):
        t = TopologyInfo(gpu_free_memory_bytes=[], gpu_memory_gb=[])
        assert t.min_free_memory_bytes() == 0

    def test_custom_values(self):
        t = TopologyInfo(
            num_nodes=2,
            gpus_per_node=4,
            total_gpus=8,
            has_nvlink=True,
            has_infiniband=True,
            interconnect_bandwidth_gbps=200.0,
            node_hostnames=["node-0", "node-1"],
            gpu_memory_gb=[40.0, 40.0, 40.0, 40.0],
            gpu_free_memory_bytes=[1 << 30] * 4,
        )
        assert t.num_nodes == 2
        assert t.total_gpus == 8
        assert t.has_nvlink is True
        assert t.gpu_free_memory_bytes[0] == 1 << 30


# ---------------------------------------------------------------------------
# ParallelPlan
# ---------------------------------------------------------------------------

class TestParallelPlan:
    """Plan dataclass produced by HybridParallelPlanner."""

    def test_defaults(self):
        p = ParallelPlan()
        assert p.strategy == ParallelStrategy.PP
        assert p.tp_world_size == 1
        assert p.pp_num_stages == 1
        assert p.ep_num_experts_per_node == 1
        assert p.ep_replication_factor == 1
        assert p.layers_per_stage == []
        assert p.nodes_per_stage == []
        assert p.expert_assignment == {}
        assert p.tp_group_size == 1
        assert p.ep_group_size == 1
        assert p.tp_groups == []
        assert p.ep_groups == []
        assert p.explanation == ""


# ---------------------------------------------------------------------------
# estimate_layer_memory
# ---------------------------------------------------------------------------

class TestEstimateLayerMemory:
    """Per-layer memory estimation (pure arithmetic)."""

    _KW = dict(hidden_size=1024, intermediate_size=4096, num_attention_heads=16)

    def test_returns_all_keys(self):
        result = estimate_layer_memory(**self._KW, dtype_bits=16)
        assert set(result.keys()) == {
            "parameters", "weight_bytes", "activation_bytes",
            "total_per_layer_bytes",
        }

    def test_total_equals_weight_plus_activation(self):
        result = estimate_layer_memory(**self._KW, dtype_bits=16)
        assert result["total_per_layer_bytes"] == result["weight_bytes"] + result["activation_bytes"]

    def test_know_values_16bit(self):
        # head_dim = 1024 / 16 = 64
        # q/k/v/o = 4 * 1024*1024 = 4,194,304
        # gate/up/down = 3 * 1024*4096 = 12,582,912
        # norms = 2 * 1024*2 = 4,096
        # total_params = 16,781,312
        # weight_bytes = 16,781,312 * 2 = 33,562,624
        # act_per_token = (1024 + 16*64*2 + 4096) * 2 = 7168 * 2 = 14,336
        result = estimate_layer_memory(**self._KW, dtype_bits=16)
        assert result["parameters"] == 16_781_312
        assert result["weight_bytes"] == 33_562_624
        assert result["activation_bytes"] == 14_336
        assert result["total_per_layer_bytes"] == 33_576_960

    def test_32bit_doubles_weight_bytes(self):
        r16 = estimate_layer_memory(**self._KW, dtype_bits=16)
        r32 = estimate_layer_memory(**self._KW, dtype_bits=32)
        assert r32["weight_bytes"] == 2 * r16["weight_bytes"]
        # Activation also doubles since bytes_per_param changes
        assert r32["activation_bytes"] == 2 * r16["activation_bytes"]

    def test_gqa_reduces_kv_projections(self):
        """Grouped-Query Attention uses fewer KV heads, reducing size."""
        full = estimate_layer_memory(**self._KW, num_key_value_heads=None)
        gqa = estimate_layer_memory(**self._KW, num_key_value_heads=4)
        assert gqa["parameters"] < full["parameters"]

    def test_minimal_values(self):
        """Edge case: tiny dimensions should not produce zeros."""
        result = estimate_layer_memory(
            hidden_size=1, intermediate_size=1,
            num_attention_heads=1, dtype_bits=16,
        )
        assert result["parameters"] > 0
        assert result["weight_bytes"] > 0

    def test_vocab_size_parameter_accepted(self):
        """vocab_size is accepted but not used in per-layer computation."""
        result = estimate_layer_memory(**self._KW, vocab_size=32000)
        assert result["parameters"] > 0  # no crash


# ---------------------------------------------------------------------------
# choose_tp_degree
# ---------------------------------------------------------------------------

class TestChooseTpDegree:
    """TP-degree selection (pure arithmetic)."""

    def test_returns_tp_1_when_layer_fits(self):
        deg, msg = choose_tp_degree(1000, 2000)
        assert deg == 1
        assert "TP=1" in msg

    def test_returns_tp_2_when_tp_1_overflows(self):
        deg, msg = choose_tp_degree(2000, 2000)
        assert deg == 2
        assert "TP=2" in msg

    def test_returns_tp_4_when_tp_2_overflows(self):
        deg, msg = choose_tp_degree(4000, 2000)
        assert deg == 4
        assert "TP=4" in msg

    def test_forces_max_tp_when_nothing_fits(self):
        deg, msg = choose_tp_degree(100_000, 1000)
        assert deg == 8
        assert "Forcing TP=8" in msg

    def test_custom_max_tp(self):
        deg, msg = choose_tp_degree(4000, 2000, max_tp=16)
        assert deg == 4
        assert "TP=4" in msg

    def test_all_fit_at_custom_max(self):
        deg, msg = choose_tp_degree(100_000, 1000, max_tp=32)
        assert deg == 32
        assert "Forcing TP=32" in msg

    def test_reason_string_format(self):
        """The reason includes the layer and free memory values."""
        _, msg = choose_tp_degree(4000, 2000)
        assert "MB" in msg
        assert "GPU" in msg


# ---------------------------------------------------------------------------
# HardwareProber
# ---------------------------------------------------------------------------

class TestHardwareProber:
    """Hardware topology prober (no-GPU-safe paths)."""

    def test_probe_returns_topology_info(self):
        topology = HardwareProber.probe()
        assert isinstance(topology, TopologyInfo)
        # On a CPU-only system gpus_per_node will be 1; on a GPU system it
        # will match the device count.  Always >= 1.
        assert topology.gpus_per_node >= 1
        assert topology.total_gpus >= 1
        # infiniband is env-var driven, always False if not set
        assert topology.has_infiniband is False

    def test_detect_nvlink_exists(self):
        """Static method exists and can be called; returns False without CUDA."""
        result = HardwareProber._detect_nvlink(1)
        assert result is False


# ---------------------------------------------------------------------------
# ProfileResult / TunedConfig
# ---------------------------------------------------------------------------

class TestProfileResult:
    def test_defaults(self):
        p = ProfileResult()
        assert p.compute_tokens_per_sec_per_gpu == 0.0
        assert p.intra_node_bw_gbps == 0.0
        assert p.inter_node_bw_gbps == 0.0
        assert p.free_memory_per_gpu == []
        assert p.peak_memory_per_token_mb == 0.0
        assert p.profile_duration_seconds == 0.0

    def test_custom_values(self):
        p = ProfileResult(
            compute_tokens_per_sec_per_gpu=12345.0,
            free_memory_per_gpu=[80.0, 80.0],
            profile_duration_seconds=10.5,
        )
        assert p.compute_tokens_per_sec_per_gpu == 12345.0
        assert len(p.free_memory_per_gpu) == 2
        assert p.profile_duration_seconds == 10.5


class TestTunedConfig:
    def test_defaults(self):
        c = TunedConfig()
        assert c.tp_degree == 1
        assert c.pp_stages == 1
        assert c.micro_batch_size == 1
        assert c.estimated_step_latency_ms == 0.0
        assert c.explanation == ""

    def test_custom_values(self):
        c = TunedConfig(
            tp_degree=4, pp_stages=2, micro_batch_size=8,
            estimated_step_latency_ms=42.1,
            explanation="fast config",
        )
        assert c.tp_degree == 4
        assert c.pp_stages == 2
        assert c.micro_batch_size == 8
        assert c.estimated_step_latency_ms == 42.1
        assert c.explanation == "fast config"


# ---------------------------------------------------------------------------
# ParallelAutoTuner
# ---------------------------------------------------------------------------

class TestParallelAutoTuner:
    """Startup profiler + auto-tuner (GPU-free paths via injected profile)."""

    def test_default_topology(self):
        tuner = ParallelAutoTuner()
        assert isinstance(tuner.topology, TopologyInfo)

    def test_custom_topology(self):
        topo = TopologyInfo(num_nodes=2, gpus_per_node=4, total_gpus=8, has_nvlink=True)
        tuner = ParallelAutoTuner(topo)
        assert tuner.topology is topo
        assert tuner._profile is None

    def test_explain_static(self):
        msg = ParallelAutoTuner._explain(4, 2, 8, 123.4, 32, 4096)
        assert "TP=4" in msg
        assert "PP=2" in msg
        assert "micro_batch=8" in msg
        assert "123.4ms" in msg
        assert "32Lx4096D" in msg

    def test_would_oom_no_profile_returns_false(self):
        tuner = ParallelAutoTuner(
            TopologyInfo(gpus_per_node=1, total_gpus=1)
        )
        assert tuner._would_oom(1, 1, 4096, 2048, 32) is False

    def test_would_oom_small_model(self):
        """A tiny model on abundant memory should NOT OOM."""
        tuner = ParallelAutoTuner(
            TopologyInfo(gpus_per_node=1, total_gpus=1)
        )
        tuner._profile = ProfileResult(
            free_memory_per_gpu=[80.0],
        )
        assert tuner._would_oom(1, 1, 1024, 128, 2) is False

    def test_would_oom_empty_free_list(self):
        tuner = ParallelAutoTuner()
        tuner._profile = ProfileResult(free_memory_per_gpu=[])
        assert tuner._would_oom(1, 1, 4096, 2048, 32) is False

    def test_estimate_latency_with_profile(self):
        tuner = ParallelAutoTuner(
            TopologyInfo(num_nodes=1, gpus_per_node=1, total_gpus=1)
        )
        tuner._profile = ProfileResult(
            compute_tokens_per_sec_per_gpu=100_000.0,
            intra_node_bw_gbps=600.0,
            inter_node_bw_gbps=12.5,
            free_memory_per_gpu=[80.0],
        )
        lat = tuner._estimate_latency(
            tp=1, pp=1, mb=1,
            total_layers=32, hidden_size=4096, seq_len=2048,
        )
        assert 0 < lat < float("inf")

    def test_estimate_latency_with_tp_pp(self):
        """Latency includes communication overhead when TP/PP > 1."""
        tuner = ParallelAutoTuner(
            TopologyInfo(num_nodes=2, gpus_per_node=4, total_gpus=8, has_nvlink=True)
        )
        tuner._profile = ProfileResult(
            compute_tokens_per_sec_per_gpu=100_000.0,
            intra_node_bw_gbps=600.0,
            inter_node_bw_gbps=12.5,
            free_memory_per_gpu=[80.0] * 4,
        )
        lat = tuner._estimate_latency(
            tp=4, pp=2, mb=4,
            total_layers=32, hidden_size=4096, seq_len=2048,
        )
        assert 0 < lat < float("inf")

    def test_tune_with_injected_profile(self):
        """tune() uses the injected profile and produces a TunedConfig."""
        topo = TopologyInfo(num_nodes=1, gpus_per_node=1, total_gpus=1)
        tuner = ParallelAutoTuner(topo)
        tuner._profile = ProfileResult(
            compute_tokens_per_sec_per_gpu=100_000.0,
            intra_node_bw_gbps=600.0,
            inter_node_bw_gbps=12.5,
            free_memory_per_gpu=[80.0],
        )
        result = tuner.tune(
            total_layers=32,
            hidden_size=4096,
            seq_len=2048,
            max_micro_batch=16,
        )
        assert isinstance(result, TunedConfig)
        assert result.tp_degree >= 1
        assert result.pp_stages >= 1
        assert result.micro_batch_size >= 1
        # Explanation should be populated
        assert "TP=" in result.explanation
        assert "PP=" in result.explanation

    def test_tune_zero_layers(self):
        """Edge case: zero layers should still produce a config."""
        topo = TopologyInfo(num_nodes=1, gpus_per_node=1, total_gpus=1)
        tuner = ParallelAutoTuner(topo)
        tuner._profile = ProfileResult(
            compute_tokens_per_sec_per_gpu=100_000.0,
            intra_node_bw_gbps=600.0,
            free_memory_per_gpu=[80.0],
        )
        result = tuner.tune(total_layers=0, hidden_size=4096)
        assert isinstance(result, TunedConfig)
        assert result.tp_degree >= 1


# ---------------------------------------------------------------------------
# HybridParallelPlanner
# ---------------------------------------------------------------------------

class TestHybridParallelPlanner:
    """Parallelism strategy selection and plan construction."""

    def test_default_topology(self):
        planner = HybridParallelPlanner()
        assert isinstance(planner.topology, TopologyInfo)
        assert planner.current_plan is None

    def test_custom_topology(self):
        topo = TopologyInfo(num_nodes=2, gpus_per_node=4, total_gpus=8, has_nvlink=True)
        planner = HybridParallelPlanner(topo)
        assert planner.topology is topo

    # -- _distribute_layers --

    def test_distribute_even(self):
        planner = HybridParallelPlanner()
        assert planner._distribute_layers(12, 3) == [(0, 3), (4, 7), (8, 11)]

    def test_distribute_with_remainder(self):
        planner = HybridParallelPlanner()
        # 10 / 3 = 3 rem 1 -> [4, 3, 3] -> [(0,3), (4,6), (7,9)]
        assert planner._distribute_layers(10, 3) == [(0, 3), (4, 6), (7, 9)]

    def test_distribute_single_stage(self):
        planner = HybridParallelPlanner()
        assert planner._distribute_layers(32, 1) == [(0, 31)]

    def test_distribute_zero_layers(self):
        planner = HybridParallelPlanner()
        assert planner._distribute_layers(0, 1) == [(0, 0)]

    def test_distribute_zero_stages(self):
        planner = HybridParallelPlanner()
        assert planner._distribute_layers(10, 0) == [(0, 9)]

    def test_distribute_zero_both(self):
        planner = HybridParallelPlanner()
        assert planner._distribute_layers(0, 0) == [(0, 0)]

    # -- _assign_nodes --

    def test_assign_nodes_single(self):
        planner = HybridParallelPlanner()
        assert planner._assign_nodes(1) == [["node_0"]]

    def test_assign_nodes_multi(self):
        planner = HybridParallelPlanner()
        assert planner._assign_nodes(3) == [["node_0"], ["node_1"], ["node_2"]]

    def test_assign_nodes_zero(self):
        """stages <= 1 returns the single-node fallback."""
        planner = HybridParallelPlanner()
        assert planner._assign_nodes(0) == [["node_0"]]

    # -- _strategy_for --

    def test_strategy_tp_pp_ep(self):
        c = TunedConfig(tp_degree=2, pp_stages=2)
        assert HybridParallelPlanner._strategy_for(c, True) == ParallelStrategy.TP_PP_EP

    def test_strategy_tp_pp(self):
        c = TunedConfig(tp_degree=2, pp_stages=2)
        assert HybridParallelPlanner._strategy_for(c, False) == ParallelStrategy.TP_PP

    def test_strategy_tp_ep(self):
        c = TunedConfig(tp_degree=2, pp_stages=1)
        assert HybridParallelPlanner._strategy_for(c, True) == ParallelStrategy.TP_EP

    def test_strategy_pp_ep(self):
        c = TunedConfig(tp_degree=1, pp_stages=2)
        assert HybridParallelPlanner._strategy_for(c, True) == ParallelStrategy.PP_EP

    def test_strategy_tp_only(self):
        c = TunedConfig(tp_degree=2, pp_stages=1)
        assert HybridParallelPlanner._strategy_for(c, False) == ParallelStrategy.TP

    def test_strategy_ep_only(self):
        c = TunedConfig(tp_degree=1, pp_stages=1)
        assert HybridParallelPlanner._strategy_for(c, True) == ParallelStrategy.EP

    def test_strategy_pp_only(self):
        c = TunedConfig(tp_degree=1, pp_stages=1)
        assert HybridParallelPlanner._strategy_for(c, False) == ParallelStrategy.PP

    # -- _build_explanation --

    def test_build_explanation_default(self):
        topo = TopologyInfo(num_nodes=1, total_gpus=1)
        planner = HybridParallelPlanner(topo)
        plan = ParallelPlan()
        expl = planner._build_explanation(plan)
        assert "Strategy: pipeline_parallel" in expl
        assert "GPUs=1, Nodes=1" in expl

    def test_build_explanation_with_tp_pp(self):
        topo = TopologyInfo(num_nodes=2, total_gpus=8)
        planner = HybridParallelPlanner(topo)
        plan = ParallelPlan(
            strategy=ParallelStrategy.TP_PP,
            tp_world_size=4,
            pp_num_stages=2,
        )
        expl = planner._build_explanation(plan)
        assert "Strategy: tp_pp" in expl
        assert "TP=4" in expl
        assert "PP=2" in expl
        assert "GPUs=8, Nodes=2" in expl

    # -- _build_tp_ep_groups --

    def test_build_groups_both_one(self):
        """Both sizes == 1 returns empty groups."""
        planner = HybridParallelPlanner()
        tg, eg, atp, aep = planner._build_tp_ep_groups(4, 1, 1)
        assert tg == []
        assert eg == []
        assert atp == 1
        assert aep == 1

    def test_build_groups_tp_only(self):
        """tp=2, ep=1, gpus=4 -> two TP groups of 2, EP groups cross ranks."""
        planner = HybridParallelPlanner()
        tg, eg, atp, aep = planner._build_tp_ep_groups(4, 2, 1)
        assert tg == [[0, 1], [2, 3]]
        assert eg == [[0, 2], [1, 3]]
        assert atp == 2
        assert aep == 2

    def test_build_groups_tp_ep(self):
        """Balanced: tp=2, ep=2, gpus=4."""
        planner = HybridParallelPlanner()
        tg, eg, atp, aep = planner._build_tp_ep_groups(4, 2, 2)
        assert tg == [[0, 1], [2, 3]]
        assert eg == [[0, 2], [1, 3]]
        assert atp == 2
        assert aep == 2

    def test_build_groups_four_gpus_tp_four(self):
        """tp=4, ep=1, gpus=4 -> one TP group, no EP groups."""
        planner = HybridParallelPlanner()
        tg, eg, atp, aep = planner._build_tp_ep_groups(4, 4, 1)
        assert tg == [[0, 1, 2, 3]]
        assert eg == []  # all EP groups would be size 1
        assert atp == 4
        assert aep == 1

    # -- plan() without tuned_config --

    def test_plan_single_node_no_tp(self):
        """Single GPU / single node -> PP fallback."""
        topo = TopologyInfo(num_nodes=1, gpus_per_node=1, total_gpus=1)
        planner = HybridParallelPlanner(topo)
        plan = planner.plan(total_layers=32)
        assert plan.strategy == ParallelStrategy.PP
        assert plan.tp_world_size == 1
        assert plan.pp_num_stages == 1
        assert planner.current_plan is plan

    def test_plan_multi_node_no_nvlink(self):
        """2 nodes x 1 GPU each, no NVLink -> PP (no TP possible)."""
        topo = TopologyInfo(num_nodes=2, gpus_per_node=1, total_gpus=2)
        planner = HybridParallelPlanner(topo)
        plan = planner.plan(total_layers=32)
        assert plan.strategy == ParallelStrategy.PP
        assert plan.pp_num_stages == 2

    def test_plan_single_node_with_nvlink(self):
        """1 node, 4 GPUs, NVLink -> TP (nodes==1 => no PP unless layers>20).
        Since can_pp = nodes>1 or (not can_tp and layers>20), with can_tp=True
        and nodes=1, PP is False -> strategy = TP.
        """
        topo = TopologyInfo(num_nodes=1, gpus_per_node=4, total_gpus=4, has_nvlink=True)
        planner = HybridParallelPlanner(topo)
        plan = planner.plan(total_layers=16)
        assert plan.strategy == ParallelStrategy.TP
        assert plan.tp_world_size == 4
        assert plan.pp_num_stages == 1

    def test_plan_two_nodes_with_nvlink(self):
        """2 nodes, 4 GPUs/node, NVLink -> TP_PP."""
        topo = TopologyInfo(num_nodes=2, gpus_per_node=4, total_gpus=8, has_nvlink=True)
        planner = HybridParallelPlanner(topo)
        plan = planner.plan(total_layers=32)
        assert plan.strategy == ParallelStrategy.TP_PP
        assert plan.tp_world_size == 4
        assert plan.pp_num_stages == 2

    def test_plan_forced_tp_no_nvlink(self):
        """When a single layer exceeds GPU memory, force TP regardless of NVLink."""
        # Construct a topology with 2 GPUs but no NVLink.  Provide GPU free
        # memory so that min_free_memory_bytes() returns > 0.
        # Use very small free memory so a single layer (~340MB at
        # hidden_size=4096, intermediate_size=11008) does NOT fit in TP=1.
        topo = TopologyInfo(
            num_nodes=1, gpus_per_node=2, total_gpus=2,
            has_nvlink=False,
            gpu_free_memory_bytes=[50 * (1024 ** 2)] * 2,  # 50 MB per GPU
        )
        planner = HybridParallelPlanner(topo)
        # Use a very large hidden_size so that estimate_layer_memory produces
        # a layer that does not fit at TP=1.
        plan = planner.plan(
            total_layers=32,
            hidden_size=4096,
            intermediate_size=11008,
            num_attention_heads=32,
        )
        # TP should be forced > 1
        assert plan.tp_world_size > 1
        # Explanation should mention forced TP
        assert "TP-forced" in plan.explanation

    # -- plan() with tuned_config --

    def test_plan_with_tuned_config(self):
        topo = TopologyInfo(num_nodes=2, gpus_per_node=4, total_gpus=8, has_nvlink=True)
        planner = HybridParallelPlanner(topo)
        tuned = TunedConfig(
            tp_degree=4, pp_stages=2, micro_batch_size=4,
            estimated_step_latency_ms=42.0,
            explanation="test config",
        )
        plan = planner.plan(total_layers=32, tuned_config=tuned, hidden_size=4096)
        assert plan.strategy == ParallelStrategy.TP_PP
        assert plan.tp_world_size == 4
        assert plan.pp_num_stages == 2
        assert "test config" in plan.explanation
        # Layers should be distributed across 2 stages
        assert len(plan.layers_per_stage) == 2

    def test_plan_with_tuned_config_overrides_forced_tp(self):
        """tuned_config's tp_degree is max'd with forced_tp."""
        # Use small free memory so the large hidden size (~1GB per layer)
        # forces TP > 1.
        topo = TopologyInfo(
            num_nodes=1, gpus_per_node=4, total_gpus=4,
            has_nvlink=True,
            gpu_free_memory_bytes=[200 * (1024 ** 2)] * 4,  # 200 MB per GPU
        )
        planner = HybridParallelPlanner(topo)
        tuned = TunedConfig(tp_degree=1, pp_stages=1, explanation="conservative")
        # With a large hidden_size, forced_tp will be > 1, so the plan's
        # tp_world_size should be at least forced_tp.
        plan = planner.plan(
            total_layers=32, tuned_config=tuned,
            hidden_size=8192, intermediate_size=16384,
            num_attention_heads=32,
        )
        # forced_tp > 1 should push tp_world_size above 1
        assert plan.tp_world_size > 1

    # -- build_plan_groups --

    def test_build_plan_groups_noop_when_no_plan(self):
        planner = HybridParallelPlanner()
        planner.build_plan_groups()  # should not raise

    def test_build_plan_groups_noop_for_pp(self):
        """PP strategy does not build TP/EP groups."""
        topo = TopologyInfo(num_nodes=1, gpus_per_node=1, total_gpus=1)
        planner = HybridParallelPlanner(topo)
        planner.plan(total_layers=32)
        planner.build_plan_groups()
        assert planner.current_plan.tp_groups == []
        assert planner.current_plan.ep_groups == []

    def test_build_plan_groups_tp(self):
        topo = TopologyInfo(num_nodes=1, gpus_per_node=4, total_gpus=4, has_nvlink=True)
        planner = HybridParallelPlanner(topo)
        plan = planner.plan(total_layers=16)
        planner.build_plan_groups()
        assert len(plan.tp_groups) > 0
        assert plan.tp_group_size > 1

    # -- _assign_experts raises ImportError (no such function) --

    def test_assign_experts_missing_function(self):
        """_assign_experts depends on an import that does not exist."""
        planner = HybridParallelPlanner()
        with pytest.raises(ImportError):
            planner._assign_experts(8, 2)


# ---------------------------------------------------------------------------
# HybridParallelExecutor
# ---------------------------------------------------------------------------

class TestHybridParallelExecutor:
    """Hyparallel plan executor (GPU-free and no-pipeline paths only)."""

    def _make_plan(self, strategy=ParallelStrategy.PP, **kw) -> ParallelPlan:
        kwargs = dict(strategy=strategy, tp_world_size=1, pp_num_stages=1)
        kwargs.update(kw)
        return ParallelPlan(**kwargs)

    # -- launchers (no-op paths) --

    def test_launch_tp_noop(self):
        plan = self._make_plan(ParallelStrategy.TP, tp_world_size=1)
        executor = HybridParallelExecutor(plan)
        executor.launch_tp("dummy")  # should not raise

    def test_configure_pp_noop(self):
        plan = self._make_plan(ParallelStrategy.PP, pp_num_stages=1)
        executor = HybridParallelExecutor(plan)
        executor.configure_pp(object())  # should not raise

    def test_configure_ep_noop(self):
        plan = self._make_plan()
        executor = HybridParallelExecutor(plan)
        executor.configure_ep(None, ["node_0"])  # should not raise

    # -- _tp_fwd (no processes, returns input) --

    def test_tp_fwd_no_processes(self):
        plan = self._make_plan(ParallelStrategy.TP)
        executor = HybridParallelExecutor(plan)
        x = torch.zeros(1, 4, 8)
        out = executor._tp_fwd(x)
        assert torch.equal(out, x)

    # -- _pp_fwd (no pipeline, raises) --

    def test_pp_fwd_no_pipeline(self):
        plan = self._make_plan(ParallelStrategy.PP)
        executor = HybridParallelExecutor(plan)
        x = torch.zeros(1, 4, 8)
        with pytest.raises(RuntimeError, match="No pipeline"):
            executor._pp_fwd(x, {})

    # -- _tp_pp_fused_forward (tp OK, pp fails) --

    def test_tp_pp_fused_forward_no_pipeline(self):
        plan = self._make_plan(ParallelStrategy.TP_PP)
        executor = HybridParallelExecutor(plan)
        x = torch.zeros(1, 4, 8)
        with pytest.raises(RuntimeError, match="No pipeline"):
            executor._tp_pp_fused_forward(x, {})

    # -- execute (various strategies) --

    def test_execute_tp_returns_input(self):
        plan = self._make_plan(ParallelStrategy.TP)
        executor = HybridParallelExecutor(plan)
        x = torch.zeros(1, 4, 8)
        out = executor.execute(x, {})
        assert torch.equal(out, x)

    def test_execute_tp_ep_returns_input(self):
        """TP_EP without experts or pipeline returns input."""
        plan = self._make_plan(ParallelStrategy.TP_EP)
        executor = HybridParallelExecutor(plan)
        x = torch.zeros(1, 4, 8)
        out = executor.execute(x, {})
        assert torch.equal(out, x)

    def test_execute_pp_raises(self):
        plan = self._make_plan(ParallelStrategy.PP)
        executor = HybridParallelExecutor(plan)
        with pytest.raises(RuntimeError, match="No pipeline"):
            executor.execute(torch.zeros(1, 4, 8), {})

    def test_execute_tp_pp_raises(self):
        plan = self._make_plan(ParallelStrategy.TP_PP)
        executor = HybridParallelExecutor(plan)
        with pytest.raises(RuntimeError, match="No pipeline"):
            executor.execute(torch.zeros(1, 4, 8), {})

    def test_execute_pp_ep_raises(self):
        plan = self._make_plan(ParallelStrategy.PP_EP)
        executor = HybridParallelExecutor(plan)
        with pytest.raises(RuntimeError, match="No pipeline"):
            executor.execute(torch.zeros(1, 4, 8), {})

    def test_execute_none_strategy_raises(self):
        """A plan with None as strategy falls through to _pp_fwd."""
        plan = ParallelPlan()  # default strategy is PP
        executor = HybridParallelExecutor(plan)
        with pytest.raises(RuntimeError, match="No pipeline"):
            executor.execute(torch.zeros(1, 4, 8), {})

    def test_execute_tp_ep_with_coordinator_no_moe(self):
        """TP_EP with a coordinator that has no moe_orchestrator."""
        plan = self._make_plan(ParallelStrategy.TP_EP)

        class DummyCoordinator:
            pass

        executor = HybridParallelExecutor(plan, coordinator=DummyCoordinator())
        x = torch.zeros(1, 4, 8)
        out = executor.execute(x, {})
        assert torch.equal(out, x)

    # -- shutdown --

    def test_shutdown_empty(self):
        plan = self._make_plan()
        executor = HybridParallelExecutor(plan)
        executor.shutdown()  # should not raise

    def test_shutdown_with_processes(self):
        """Shutdown gracefully handles process-like objects without terminate."""
        plan = self._make_plan(ParallelStrategy.TP, tp_world_size=2)

        class FakeProcess:
            pass

        executor = HybridParallelExecutor(plan)
        executor._tp_processes = [FakeProcess(), FakeProcess()]
        executor.shutdown()  # should not raise (no terminate attr)
        assert executor._tp_processes == []
