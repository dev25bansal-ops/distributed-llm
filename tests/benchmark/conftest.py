"""Shared fixtures for benchmark tests.

Fixtures here are auto-discovered by pytest. Helper functions and
pre-loaded modules live in helpers.py to be explicitly imported.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from unittest.mock import MagicMock

import pytest
import torch

from helpers import (  # noqa: E402
    Coordinator,
    NodeRegistration,
    DeviceInfo,
    HeterogeneousCluster,
    HeterogeneousNode,
    assign_layers_proportional,
    estimate_heterogeneous_throughput,
    get_device_compatibility_map,
)


# ---------------------------------------------------------------------------
# Benchmark fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    tok.decode.return_value = "hello world"
    tok.eos_token_id = 0
    tok.pad_token_id = 0
    tok.vocab_size = 32000
    return tok


@pytest.fixture
def perf_model_config():
    """Standard model configs used across benchmarks."""
    return {
        "1B":  {"layers": 12,  "hidden": 768,  "heads": 12,  "vocab": 32000, "params_b": 1.0},
        "3B":  {"layers": 24,  "hidden": 1536, "heads": 16,  "vocab": 32000, "params_b": 3.0},
        "7B":  {"layers": 32,  "hidden": 4096, "heads": 32,  "vocab": 32000, "params_b": 7.0},
        "13B": {"layers": 40,  "hidden": 5120, "heads": 40,  "vocab": 32000, "params_b": 13.0},
        "34B": {"layers": 56,  "hidden": 7168, "heads": 64,  "vocab": 32000, "params_b": 34.0},
        "70B": {"layers": 80,  "hidden": 8192, "heads": 64,  "vocab": 32000, "params_b": 70.0},
    }


@pytest.fixture
def mock_forward_fn():
    """A mock forward function with configurable hidden dim."""
    def _make_forward(hidden_dim: int = 768, delay_s: float = 0.001):
        def forward(x):
            if delay_s > 0:
                import time
                time.sleep(delay_s)
            return torch.randn(x.shape[0], x.shape[1], 32000)
        return forward
    return _make_forward


@pytest.fixture
def coordinator_with_mock_nodes(mock_tokenizer, monkeypatch):
    """Coordinator with 2 registered mock nodes."""
    monkeypatch.setattr(NodeRegistration, "init_client", lambda self, **kw: None)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: mock_tokenizer)
    mock_config = MagicMock()
    mock_config.model_type = "mock"
    mock_config.num_hidden_layers = 12
    mock_config.hidden_size = 768
    mock_config.num_attention_heads = 12
    mock_config.vocab_size = 32000
    mock_config.max_position_embeddings = 2048
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda *a, **kw: mock_config)

    coord = Coordinator(model_name="test-model", dtype="float32", max_batch_size=4)
    coord.tokenizer = mock_tokenizer
    coord.model_info = {"num_layers": 12, "hidden_size": 768, "num_attention_heads": 12}
    coord.total_layers = 12

    for i in range(2):
        coord.manual_register(
            node_id=f"n{i}", host="localhost", port=55060 + i,
            start_layer=i * 6, end_layer=(i + 1) * 6 - 1, total_layers=12,
        )
        reg = coord.nodes[f"n{i}"]
        reg.client = MagicMock()
        reg.async_client = MagicMock()
        reg.healthy = True
        reg.client.forward.return_value = torch.randn(1, 1, 32000)

    # Monkeypatch generate to avoid real inference engine calls
    coordinator_mock_cls = type(coord)

    def _mock_generate(self, prompt, max_new_tokens=32, temperature=0.7, top_p=0.9, top_k=50):
        return f"mock response for: {prompt[:40]}"

    monkeypatch.setattr(coordinator_mock_cls, "generate", _mock_generate)
    return coord


@pytest.fixture
def mock_cluster():
    """Standard homogeneous cluster fixture."""
    nv_device = DeviceInfo(
        device_type="cuda", device_family="NVIDIA", device_id=0,
        name="A100-SXM-80GB", total_memory_bytes=80 * 1024**3,
        tflops_fp16=312.0, memory_bandwidth_gbps=2039.0,
    )
    return HeterogeneousCluster(
        nodes=[
            HeterogeneousNode(
                node_id="a100-1", host="localhost", port=55060,
                device_info=nv_device, throughput_score=1000.0,
            ),
            HeterogeneousNode(
                node_id="a100-2", host="localhost", port=55061,
                device_info=nv_device, throughput_score=1000.0,
            ),
        ],
        total_layers=32, hidden_size=4096,
    )
