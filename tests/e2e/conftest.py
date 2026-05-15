"""E2E test fixtures.

Provides fixtures that spin up a fully mocked API server:
- Mock coordinator with working generate() method
- FastAPI test client hitting the real API server with middleware
"""

import pytest
import torch
from unittest.mock import MagicMock


@pytest.fixture
def e2e_coordinator():
    """Create a fully mocked coordinator for E2E API tests.

    Unlike the integration fixtures, this uses a pure MagicMock
    so that API requests complete immediately without real model inference.
    """
    coord = MagicMock()
    coord.model_name = "test-model"
    coord.nodes = {}
    coord.node_order = []
    coord.scheduler = None
    coord.prefix_cache = None
    coord.metrics_exporter = None

    def encode_fn(text, **kwargs):
        tokens = list(range(1, len(text.split()) + 1))
        if kwargs.get("return_tensors") == "pt":
            return torch.tensor([tokens])
        return tokens

    coord.tokenizer = MagicMock()
    coord.tokenizer.encode.side_effect = encode_fn
    coord.tokenizer.decode.side_effect = lambda tokens, **kwargs: " ".join(
        f"tok-{t}" for t in (tokens if isinstance(tokens, list) else tokens.tolist())
    )
    coord.tokenizer.eos_token_id = 0
    coord.tokenizer.bos_token_id = 1
    coord.generate.return_value = "Hello! This is a test response."

    # Mock local_partitioner.full_model for streaming support
    mock_model = MagicMock()
    mock_model.parameters.side_effect = lambda: iter([torch.randn(10, 10)])
    mock_output = MagicMock()
    mock_output.logits = torch.randn(1, 5, 1000)
    mock_output.past_key_values = MagicMock()
    mock_model.return_value = mock_output
    coord.local_partitioner = MagicMock()
    coord.local_partitioner.full_model = mock_model

    coord.list_models.return_value = ["distributed-llm"]

    return coord


@pytest.fixture
def e2e_api_client(e2e_coordinator):
    """FastAPI TestClient with E2E coordinator injected.

    Hits the real API server middleware stack (auth, rate limiting, etc.).
    """
    from fastapi.testclient import TestClient

    import distllm.api.server as server_module
    from distllm.api.server import app

    original_coordinator = server_module.coordinator
    server_module.coordinator = e2e_coordinator

    client = TestClient(app, raise_server_exceptions=True)
    yield client

    server_module.coordinator = original_coordinator
