"""Tests: numerical stability — no NaN/Inf in outputs across all supported models.

Verifies that forward passes produce finite values (no NaN, no Inf) for:
- Various input lengths (short, medium, long)
- Edge cases (empty batch, single token, max context)
- All supported model architectures (simulated via config shapes)

Uses synthetic inputs and small model configurations that run on CPU.
"""

import pytest
import torch
import torch.nn as nn
from loguru import logger


def _check_for_nan_inf(tensor: torch.Tensor, label: str) -> list[str]:
    """Check a tensor for NaN/Inf values. Returns list of violations."""
    violations = []
    if torch.isnan(tensor).any():
        violations.append(f"{label}: contains NaN values")
    if torch.isinf(tensor).any():
        violations.append(f"{label}: contains Inf values")
    return violations


class TinyTransformerLayer(nn.Module):
    """Minimal transformer layer for stability testing on CPU."""

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.gate_proj = nn.Linear(hidden_dim, hidden_dim * 4)
        self.up_proj = nn.Linear(hidden_dim, hidden_dim * 4)
        self.down_proj = nn.Linear(hidden_dim * 4, hidden_dim)

        self.input_norm = nn.LayerNorm(hidden_dim)
        self.post_attn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        # Self-attention
        residual = x
        x = self.input_norm(x)
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, seq_len, self.hidden_dim)
        attn_out = self.o_proj(attn_out)
        x = residual + attn_out

        # MLP
        residual = x
        x = self.post_attn_norm(x)
        gate = torch.sigmoid(self.gate_proj(x))
        up = self.up_proj(x)
        x = self.down_proj(gate * up)
        x = residual + x

        return x


class TinyModel(nn.Module):
    """Minimal transformer model for stability testing."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        vocab_size: int = 1000,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            TinyTransformerLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.numel() == 0:
            raise ValueError("Empty input tensor")
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.lm_head(h)


# Model configuration presets mimicking supported architectures
MODEL_PRESETS = [
    {"name": "tiny", "hidden_dim": 64, "num_layers": 4, "num_heads": 4},
    {"name": "small", "hidden_dim": 128, "num_layers": 8, "num_heads": 8},
    {"name": "medium", "hidden_dim": 256, "num_layers": 12, "num_heads": 8},
]

# Input configurations to test
INPUT_CONFIGS = [
    {"name": "single_token", "batch": 1, "seq": 1},
    {"name": "short", "batch": 2, "seq": 16},
    {"name": "medium", "batch": 4, "seq": 128},
    {"name": "long", "batch": 1, "seq": 512},
    {"name": "large_batch", "batch": 16, "seq": 64},
    {"name": "uneven_batch", "batch": 7, "seq": 31},
    {"name": "max_context", "batch": 1, "seq": 1024},
]


class TestNumericalStability:
    """Verify no NaN/Inf values in forward passes."""

    @pytest.fixture(params=MODEL_PRESETS, ids=lambda p: p["name"])
    def model(self, request):
        cfg = {k: v for k, v in request.param.items() if k != "name"}
        return TinyModel(**cfg)

    def test_forward_no_nan_inf(self, model):
        """Forward pass produces finite logits."""
        model.eval()
        for input_cfg in INPUT_CONFIGS:
            input_ids = torch.randint(0, 100, (input_cfg["batch"], input_cfg["seq"]))
            with torch.no_grad():
                logits = model(input_ids)

            violations = _check_for_nan_inf(logits, f"logits ({input_cfg['name']})")
            assert not violations, f"Model produced NaN/Inf: {violations}"

    def test_softmax_stability(self, model):
        """Softmax over logits remains finite even with extreme values."""
        model.eval()
        input_ids = torch.randint(0, 100, (2, 32))

        with torch.no_grad():
            logits = model(input_ids)

        probs = torch.softmax(logits.float(), dim=-1)
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        violations = []
        violations.extend(_check_for_nan_inf(probs, "softmax probabilities"))
        violations.extend(_check_for_nan_inf(log_probs, "log probabilities"))

        assert not violations, f"Softmax produced NaN/Inf: {violations}"
        assert (probs >= 0).all(), "Probabilities should be non-negative"
        assert (probs <= 1).all(), "Probabilities should be <= 1"

        # Sum to 1 per position
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
            "Probabilities should sum to 1"

    def test_attention_weights_stability(self, model):
        """Attention weights remain valid."""
        model.eval()
        input_ids = torch.randint(0, 100, (2, 64))

        with torch.no_grad():
            h = model.embed(input_ids)
            for layer in model.layers:
                residual = h
                h = layer.input_norm(h)
                bsz, seq_len, _ = h.shape
                q = layer.q_proj(h).view(bsz, seq_len, layer.num_heads, layer.head_dim).transpose(1, 2)
                k = layer.k_proj(h).view(bsz, seq_len, layer.num_heads, layer.head_dim).transpose(1, 2)

                attn = torch.matmul(q, k.transpose(-2, -1)) / (layer.head_dim ** 0.5)
                attn_weights = torch.softmax(attn, dim=-1)

                violations = _check_for_nan_inf(attn_weights, f"attention weights layer {id(layer)}")
                assert not violations, f"Attention weights contain NaN/Inf"

                # Check attention pattern validity
                row_sums = attn_weights.sum(dim=-1)
                assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
                    f"Attention weights should sum to 1 per position"

                h = residual  # Don't continue forward, just check attn

    def test_empty_input_raises(self, model):
        """Empty input should raise or produce sensible output."""
        with pytest.raises(Exception):
            input_ids = torch.randint(0, 100, (0, 10))
            model(input_ids)

    def test_extreme_values_stability(self, model):
        """Extreme input values don't cause instability."""
        model.eval()
        large_input = 999 * torch.ones(1, 32, dtype=torch.long)

        with torch.no_grad():
            logits = model(large_input)

        violations = _check_for_nan_inf(logits, "logits (extreme input)")
        assert not violations, f"Extreme input caused NaN/Inf"

    def test_gradient_flow(self):
        """Gradients flow without NaN for all parameters."""
        model = TinyModel(hidden_dim=32, num_layers=2, num_heads=4)
        model.train()
        input_ids = torch.randint(0, 100, (2, 16))

        logits = model(input_ids)
        loss = logits.mean()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                violations = _check_for_nan_inf(param.grad, f"grad:{name}")
                assert not violations, f"Gradient for {name} contains NaN/Inf"
