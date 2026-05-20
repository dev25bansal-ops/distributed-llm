"""Shared pytest fixtures for distributed-llm tests.

Provides reusable fixtures for:
- Mock coordinator (no GPU/model required)
- Mock tokenizer
- Mock gRPC stubs
- FastAPI test client
- TLS test certificates
"""

import os
import secrets
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# --- Tokenizer Fixtures ---


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer that behaves like a HuggingFace tokenizer."""
    tokenizer = MagicMock()

    def encode_fn(text, **kwargs):
        tokens = list(range(1, len(text.split()) + 1))
        if kwargs.get("return_tensors") == "pt":
            return torch.tensor([tokens])
        return tokens

    tokenizer.encode.side_effect = encode_fn
    tokenizer.decode.side_effect = lambda tokens, **kwargs: " ".join(f"tok-{t}" for t in tokens)
    tokenizer.eos_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.pad_token_id = 0
    tokenizer.vocab_size = 1000
    return tokenizer


@pytest.fixture
def mock_tokenizer_with_eos(mock_tokenizer):
    """Tokenizer that returns EOS token on specific input."""
    original_encode = mock_tokenizer.encode.side_effect

    def encode_with_eos(text, **kwargs):
        tokens = original_encode(text, **kwargs)
        if text == "short":
            tokens = [1, 2, 0]  # Ends with EOS
        return tokens

    mock_tokenizer.encode.side_effect = encode_with_eos
    return mock_tokenizer


# --- Coordinator Fixtures ---


@pytest.fixture
def mock_coordinator(mock_tokenizer):
    """Create a Coordinator with mocked model and nodes (no GPU required)."""
    from distllm.core.coordinator import Coordinator

    coord = Coordinator(
        model_name="test-model",
        dtype="float32",
        max_batch_size=1,
        max_tokens_per_batch=4096,
    )
    coord.tokenizer = mock_tokenizer
    coord.model_info = {"num_layers": 12, "hidden_size": 768, "num_attention_heads": 12}
    coord.total_layers = 12

    # Mock local partitioner
    coord.local_partitioner = MagicMock()
    mock_model = MagicMock()
    mock_model.parameters.side_effect = lambda: iter([torch.randn(10, 10)])
    coord.local_partitioner.full_model = mock_model

    return coord


@pytest.fixture
def mock_coordinator_with_scheduler(mock_tokenizer):
    """Coordinator with batch scheduler enabled."""
    from distllm.core.coordinator import Coordinator

    coord = Coordinator(
        model_name="test-model",
        dtype="float32",
        max_batch_size=4,
        max_tokens_per_batch=512,
    )
    coord.tokenizer = mock_tokenizer
    coord.model_info = {"num_layers": 12, "hidden_size": 768, "num_attention_heads": 12}
    coord.total_layers = 12

    coord.local_partitioner = MagicMock()
    mock_model = MagicMock()
    mock_model.parameters.side_effect = lambda: iter([torch.randn(10, 10)])
    coord.local_partitioner.full_model = mock_model

    return coord


@pytest.fixture
def mock_coordinator_with_nodes(mock_tokenizer):
    """Coordinator with mock node registrations."""
    from distllm.core.coordinator import Coordinator
    from distllm.core.resource_manager import NodeRegistration

    coord = Coordinator(
        model_name="test-model",
        dtype="float32",
    )
    coord.tokenizer = mock_tokenizer
    coord.model_info = {"num_layers": 12, "hidden_size": 768, "num_attention_heads": 12}
    coord.total_layers = 12

    # Register mock nodes
    for i in range(2):
        mock_client = MagicMock()
        mock_client.health_check.return_value = MagicMock(
            healthy=True,
            memory_used=1024,
            memory_total=8192,
        )
        mock_client.stub = MagicMock()

        reg = NodeRegistration(
            node_id=f"node-{i}",
            host="localhost",
            port=50051 + i,
            start_layer=i * 6,
            end_layer=(i + 1) * 6 - 1,
        )
        reg.client = mock_client
        reg.healthy = True
        coord.nodes[f"node-{i}"] = reg
        coord.node_order.append(f"node-{i}")

    return coord


# --- API Server Fixtures ---


@pytest.fixture
def api_client(mock_coordinator, monkeypatch):
    """FastAPI TestClient with mock coordinator injected."""
    from fastapi.testclient import TestClient

    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_KEY_WAS_SET", raising=False)
    monkeypatch.setenv("DISABLE_AUTH", "1")
    monkeypatch.setenv("DISTLLM_DEV_MODE", "1")

    # Inject mock coordinator
    import distllm.api.server as server_module
    from distllm.api.server import app

    original_coordinator = server_module.coordinator
    server_module.coordinator = mock_coordinator

    client = TestClient(app)
    yield client

    # Restore original
    server_module.coordinator = original_coordinator


@pytest.fixture
def api_client_no_coordinator(monkeypatch):
    """FastAPI TestClient without any coordinator (unhealthy state)."""
    from fastapi.testclient import TestClient

    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_KEY_WAS_SET", raising=False)
    monkeypatch.setenv("DISABLE_AUTH", "1")
    monkeypatch.setenv("DISTLLM_DEV_MODE", "1")

    import distllm.api.server as server_module
    from distllm.api.server import app

    original_coordinator = server_module.coordinator
    server_module.coordinator = None

    client = TestClient(app)
    yield client

    server_module.coordinator = original_coordinator


# --- Auth Fixtures ---


@pytest.fixture
def api_client_with_auth(mock_coordinator, monkeypatch):
    """FastAPI TestClient with API_KEY auth enabled."""
    from fastapi.testclient import TestClient

    from distllm.api.server import app

    test_api_key = secrets.token_urlsafe(32)
    monkeypatch.delenv("DISABLE_AUTH", raising=False)
    monkeypatch.delenv("DISTLLM_DEV_MODE", raising=False)
    monkeypatch.delenv("API_KEY_WAS_SET", raising=False)
    monkeypatch.setenv("API_KEY", test_api_key)

    import distllm.api.server as server_module

    original_coordinator = server_module.coordinator
    server_module.coordinator = mock_coordinator

    client = TestClient(app)
    client.test_api_key = test_api_key
    yield client

    server_module.coordinator = original_coordinator
    monkeypatch.delenv("API_KEY", raising=False)


# --- TLS Fixtures ---


@pytest.fixture
def tls_cert_dir():
    """Create a temporary directory for TLS certificates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def tls_certificates(tls_cert_dir):
    """Generate self-signed TLS certificates for testing.

    Uses the project TLS generator so the fixture works on Windows without openssl.
    """
    from distllm.core.tls import generate_self_signed_certs

    cert_file, key_file, ca_cert_file = generate_self_signed_certs(tls_cert_dir)

    client_cert_file = os.path.join(tls_cert_dir, "client.crt")
    client_key_file = os.path.join(tls_cert_dir, "client.key")

    return {
        "cert_file": cert_file,
        "key_file": key_file,
        "ca_cert_file": ca_cert_file,
        "client_cert_file": client_cert_file,
        "client_key_file": client_key_file,
        "cert_dir": tls_cert_dir,
    }


# --- gRPC Fixtures ---


@pytest.fixture
def mock_grpc_stub():
    """Create a mock gRPC stub."""
    stub = MagicMock()
    stub.ForwardPass.return_value = MagicMock(
        success=True,
        error_message="",
    )
    stub.HealthCheck.return_value = MagicMock(
        healthy=True,
        memory_used=1024,
        memory_total=8192,
        gpu_utilization=0.5,
    )
    return stub


@pytest.fixture
def mock_forward_pass_response():
    """Create a mock ForwardPassResponse proto."""
    from distllm.communication.node_pb2 import ForwardPassResponse, Tensor

    response = ForwardPassResponse(
        request_id="test-request",
        success=True,
        error_message="",
    )

    # Add a minimal tensor for output
    output_tensor = Tensor()
    output_tensor.shape.extend([1, 10])
    output_tensor.dtype = "torch.float32"
    # 10 float32 values = 40 bytes
    import struct

    raw_data = struct.pack("<10f", *[0.1 * i for i in range(10)])
    output_tensor.raw_data = raw_data
    response.output.CopyFrom(output_tensor)

    return response


# --- KV Cache Fixtures ---


@pytest.fixture
def sample_kv_cache():
    """Create a KVCache with sample data."""
    from distllm.core.kv_cache import KVCache

    cache = KVCache()
    cache.init_cache(
        num_layers=2,
        batch_size=1,
        num_heads=4,
        head_dim=8,
        device="cpu",
    )
    return cache


@pytest.fixture
def kv_cache_manager():
    """Create a KVCacheManager with sample caches."""
    from distllm.core.kv_cache import KVCacheManager

    manager = KVCacheManager()
    for i in range(3):
        manager.create(
            f"req-{i}",
            num_layers=2,
            batch_size=1,
            num_heads=4,
            head_dim=8,
            device="cpu",
        )
    return manager


# --- Health Service Fixtures ---


@pytest.fixture
def health_record():
    """Create a sample HealthRecord."""
    from distllm.health.state import HealthRecord, NodeState

    record = HealthRecord(
        node_id="node-0",
        state=NodeState.HEALTHY,
        layer_range="0-5",
    )
    return record


@pytest.fixture
def failover_engine():
    """Create a FailoverEngine with low thresholds for fast testing."""
    from distllm.health.failover import FailoverEngine

    engine = FailoverEngine(
        failure_threshold=2,
        degraded_latency_ms=100.0,
        recovery_threshold=1,
    )
    return engine


# --- Speculative Decoding Fixtures ---


@pytest.fixture
def coordinator_with_speculative(mock_tokenizer):
    """Coordinator with speculative decoding config."""
    from distllm.config.loader import SpeculativeConfig
    from distllm.core.coordinator import Coordinator

    spec_config = SpeculativeConfig(
        draft_model="test-draft-model",
        num_assistant_tokens=3,
    )

    coord = Coordinator(
        model_name="test-model",
        dtype="float32",
        speculative_config=spec_config,
    )
    coord.tokenizer = mock_tokenizer
    coord.num_assistant_tokens = 3

    coord.local_partitioner = MagicMock()
    mock_model = MagicMock()
    mock_model.parameters.side_effect = lambda: iter([torch.randn(10, 10)])
    coord.local_partitioner.full_model = mock_model

    # Mock draft model
    coord.draft_model = MagicMock()

    return coord


# --- Quantization Fixtures ---


@pytest.fixture
def quantization_config_bnb_8bit():
    """QuantizationConfig for 8-bit."""
    from distllm.config.loader import QuantizationConfig

    return QuantizationConfig(method="bnb_8bit", llm_int8_threshold=6.0)


@pytest.fixture
def quantization_config_bnb_4bit():
    """QuantizationConfig for 4-bit."""
    from distllm.config.loader import QuantizationConfig

    return QuantizationConfig(
        method="bnb_4bit",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


@pytest.fixture
def quantization_config_unsupported():
    """QuantizationConfig for unsupported method (gptq)."""
    from distllm.config.loader import QuantizationConfig

    return QuantizationConfig(method="gptq")


# --- Model Hub Fixtures ---


@pytest.fixture
def mock_hf_hub(tmp_path, monkeypatch):
    """Mock huggingface_hub API for testing ModelHub without network calls.

    Creates a fake cache directory with simulated downloaded models.
    """
    import json
    import time

    cache_dir = tmp_path / "hf_cache"
    cache_dir.mkdir()

    # Mock snapshot_download to create a fake model directory
    def mock_snapshot_download(repo_id, revision=None, token=None, cache_dir=None, allow_patterns=None, resume_download=True):
        model_dir = Path(cache_dir) / repo_id / (revision or "main")
        model_dir.mkdir(parents=True, exist_ok=True)
        # Create fake model files
        (model_dir / "config.json").write_text('{"model_type": "gpt2"}')
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 1000)
        # Write manifest
        manifest = {
            "model_id": repo_id,
            "revision": revision or "main",
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "size_bytes": 1000,
            "files": ["config.json", "model.safetensors"],
        }
        with open(model_dir / ".manifest", "w") as f:
            json.dump(manifest, f)
        return str(model_dir)

    monkeypatch.setattr("distllm.models.model_hub.snapshot_download", mock_snapshot_download)

    # Mock HfApi.model_info
    mock_model_info = MagicMock()
    mock_model_info.siblings_total_size = 1000
    mock_model_info.tags = ["text-generation", "en"]
    mock_model_info.pipeline_tag = "text-generation"
    mock_model_info.downloads = 100
    mock_model_info.likes = 5
    mock_model_info.last_modified = "2024-01-01"

    def mock_model_info_fn(model_name, token=None):
        return mock_model_info

    mock_api = MagicMock()
    mock_api.model_info.side_effect = mock_model_info_fn
    monkeypatch.setattr("distllm.models.model_hub.HfApi", lambda: mock_api)

    return {"cache_dir": str(cache_dir)}
