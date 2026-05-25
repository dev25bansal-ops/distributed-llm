"""Tests for expert fusion."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from distllm.core.moe_fusion import (
    FusionConfig,
    FusedExpertMLP,
    GroupedExpertMLP,
    fuse_expert_weights,
)


@pytest.fixture
def expert_mlps():
    """Create simple per-expert MLPs for testing."""
    hidden = 64
    intermediate = 256
    num_experts = 8

    experts = []
    for _ in range(num_experts):
        expert = nn.ModuleDict({
            "gate_proj": nn.Linear(hidden, intermediate, bias=False),
            "up_proj": nn.Linear(hidden, intermediate, bias=False),
            "down_proj": nn.Linear(intermediate, hidden, bias=False),
        })
        experts.append(expert)
    return experts


class TestFusionConfig:
    def test_defaults(self):
        config = FusionConfig()
        assert config.fused_experts_per_group == 0
        assert config.min_experts_for_fusion == 4
        assert config.use_grouped_gemm is True
        assert config.precision == "bf16"

    def test_custom(self):
        config = FusionConfig(
            fused_experts_per_group=4,
            min_experts_for_fusion=8,
            use_weight_merging=False,
        )
        assert config.fused_experts_per_group == 4
        assert config.min_experts_for_fusion == 8
        assert config.use_weight_merging is False


class TestFusedExpertMLP:
    @pytest.fixture
    def fused_mlp(self):
        return FusedExpertMLP(
            num_experts=4,
            hidden_size=32,
            intermediate_size=128,
            activation="silu",
        )

    def test_init(self, fused_mlp):
        assert fused_mlp.num_experts == 4
        assert fused_mlp.hidden_size == 32
        assert fused_mlp.intermediate_size == 128
        assert fused_mlp.gate_proj.shape == (4, 32, 128)
        assert fused_mlp.up_proj.shape == (4, 32, 128)
        assert fused_mlp.down_proj.shape == (4, 128, 32)

    def test_no_fusion_forward(self, fused_mlp):
        fused_mlp.config.fused_experts_per_group = 1
        hidden = torch.randn(8, 32)
        expert_indices = torch.randint(0, 4, (8, 2))
        routing_weights = torch.rand(8, 2)
        out = fused_mlp(hidden, expert_indices, routing_weights)
        assert out.shape == (8, 32)

    def test_fused_forward_shape(self, fused_mlp):
        fused_mlp.config.fused_experts_per_group = 2
        hidden = torch.randn(8, 32)
        expert_indices = torch.randint(0, 4, (8, 2))
        routing_weights = torch.rand(8, 2)
        out = fused_mlp(hidden, expert_indices, routing_weights)
        assert out.shape == (8, 32)

    def test_output_deterministic(self, fused_mlp):
        fused_mlp.config.fused_experts_per_group = 2
        hidden = torch.randn(4, 32)
        expert_indices = torch.randint(0, 4, (4, 2))
        routing_weights = torch.rand(4, 2)

        out1 = fused_mlp(hidden, expert_indices, routing_weights)
        out2 = fused_mlp(hidden, expert_indices, routing_weights)
        assert torch.allclose(out1, out2)

    def test_zero_routing_weight(self, fused_mlp):
        fused_mlp.config.fused_experts_per_group = 1
        hidden = torch.randn(4, 32)
        expert_indices = torch.randint(0, 4, (4, 2))
        routing_weights = torch.zeros(4, 2)
        out = fused_mlp(hidden, expert_indices, routing_weights)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_single_expert(self, fused_mlp):
        fused_mlp.config.fused_experts_per_group = 1
        hidden = torch.randn(4, 32)
        expert_indices = torch.zeros(4, 2, dtype=torch.long)
        routing_weights = torch.ones(4, 2)
        # All tokens go to expert 0
        out = fused_mlp(hidden, expert_indices, routing_weights)
        assert out.shape == (4, 32)
        assert not torch.isnan(out).any()

    def test_large_number_experts(self):
        mlp = FusedExpertMLP(
            num_experts=32,
            hidden_size=16,
            intermediate_size=64,
        )
        hidden = torch.randn(4, 16)
        expert_indices = torch.randint(0, 32, (4, 2))
        routing_weights = torch.rand(4, 2)
        out = mlp(hidden, expert_indices, routing_weights)
        assert out.shape == (4, 16)

    def test_precision_fp16(self):
        mlp = FusedExpertMLP(
            num_experts=4,
            hidden_size=32,
            intermediate_size=128,
            config=FusionConfig(precision="fp16"),
        )
        hidden = torch.randn(4, 32, dtype=torch.float16)
        expert_indices = torch.randint(0, 4, (4, 2))
        routing_weights = torch.rand(4, 2)
        out = mlp(hidden, expert_indices, routing_weights)
        assert out.shape == (4, 32)

    def test_get_fusion_map(self, fused_mlp):
        fm = fused_mlp.get_fusion_map()
        assert fm["num_experts"] == 4
        assert fm["fused_groups"] > 0
        assert fm["total_params"] > 0

    def test_gelu_activation(self):
        mlp = FusedExpertMLP(
            num_experts=4,
            hidden_size=32,
            intermediate_size=128,
            activation="gelu",
        )
        hidden = torch.randn(4, 32)
        expert_indices = torch.randint(0, 4, (4, 2))
        routing_weights = torch.rand(4, 2)
        out = mlp(hidden, expert_indices, routing_weights)
        assert out.shape == (4, 32)


class TestGroupedExpertMLP:
    @pytest.fixture
    def grouped_mlp(self):
        return GroupedExpertMLP(
            num_experts=4,
            hidden_size=32,
            intermediate_size=128,
            group_size=2,
        )

    def test_init(self, grouped_mlp):
        assert grouped_mlp.num_experts == 4
        assert grouped_mlp.group_size == 2
        assert grouped_mlp.gate_proj.shape == (4, 32, 128)

    def test_forward_shape(self, grouped_mlp):
        hidden = torch.randn(8, 32)
        expert_indices = torch.randint(0, 4, (8, 2))
        routing_weights = torch.rand(8, 2)
        out = grouped_mlp(hidden, expert_indices, routing_weights)
        assert out.shape == (8, 32)

    def test_all_same_expert(self, grouped_mlp):
        hidden = torch.randn(4, 32)
        expert_indices = torch.zeros(4, 2, dtype=torch.long)
        routing_weights = torch.ones(4, 2)
        out = grouped_mlp(hidden, expert_indices, routing_weights)
        assert out.shape == (4, 32)

    def test_different_group_sizes(self):
        for gs in [1, 2, 4]:
            mlp = GroupedExpertMLP(
                num_experts=4,
                hidden_size=16,
                intermediate_size=64,
                group_size=gs,
            )
            hidden = torch.randn(4, 16)
            expert_indices = torch.randint(0, 4, (4, 2))
            routing_weights = torch.rand(4, 2)
            out = mlp(hidden, expert_indices, routing_weights)
            assert out.shape == (4, 16)

    def test_gelu_activation(self):
        mlp = GroupedExpertMLP(
            num_experts=4,
            hidden_size=32,
            intermediate_size=128,
            group_size=2,
            activation="gelu",
        )
        hidden = torch.randn(4, 32)
        expert_indices = torch.randint(0, 4, (4, 2))
        routing_weights = torch.rand(4, 2)
        out = mlp(hidden, expert_indices, routing_weights)
        assert out.shape == (4, 32)


class TestFuseExpertWeights:
    def test_fuse_from_list(self, expert_mlps):
        config = FusionConfig(fused_experts_per_group=1)
        fused = fuse_expert_weights(expert_mlps[:4], config)
        assert fused.num_experts == 4
        assert fused.hidden_size == 64
        assert fused.intermediate_size == 256

    def test_fuse_weight_values(self, expert_mlps):
        config = FusionConfig(fused_experts_per_group=1)
        fused = fuse_expert_weights(expert_mlps[:2], config)
        # Check weights match the original experts
        for i, expert in enumerate(expert_mlps[:2]):
            assert torch.allclose(
                fused.gate_proj[i],
                expert["gate_proj"].weight.T,
                atol=1e-6,
            )

    def test_fused_forward_matches_per_expert(self, expert_mlps):
        """Verify fused forward output matches per-expert computation."""
        hidden = torch.randn(4, 64)
        top_k = 1
        expert_indices = torch.randint(0, 4, (4, top_k))
        routing_weights = torch.ones(4, top_k)

        # Per-expert reference
        ref_output = torch.zeros(4, 64)
        for i in range(4):
            mask = expert_indices[:, 0] == i
            if not mask.any():
                continue
            e = expert_mlps[i]
            h = hidden[mask]
            gate = e["gate_proj"](h)
            up = e["up_proj"](h)
            act = F.silu(gate)
            intermediate = act * up
            ref_output[mask] = e["down_proj"](intermediate)

        # Fused forward (disable precision cast for exact match test)
        config = FusionConfig(fused_experts_per_group=1, precision="fp32")
        fused = fuse_expert_weights(expert_mlps[:4], config)
        fused_out = fused(hidden, expert_indices, routing_weights)

        assert torch.allclose(ref_output, fused_out, atol=1e-5)
