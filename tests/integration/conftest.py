"""Shared fixtures for integration tests — uses direct file imports.

Injects fake package entries into sys.modules to prevent
distllm/__init__.py from executing (circular import chain).
"""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

import pytest
import torch

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"


def _make_fake_package(name: str, path: Path):
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    mod.__package__ = name
    sys.modules.setdefault(name, mod)
    return mod


# Inject fake packages to prevent real distllm/__init__.py from loading
_make_fake_package("distllm", SRC_DIR / "distllm")
_make_fake_package("distllm.core", SRC_DIR / "distllm/core")
_make_fake_package("distllm.dist", SRC_DIR / "distllm/dist")
_make_fake_package("distllm.dist.partition", SRC_DIR / "distllm/dist/partition")
_make_fake_package("distllm.backends", SRC_DIR / "distllm/backends")


def _load_module(rel_path: str):
    filepath = SRC_DIR / rel_path
    rel = filepath.relative_to(SRC_DIR)
    parts = list(rel.parent.parts) + [filepath.stem]
    if parts[0] == "distllm":
        dotted = ".".join(parts)
    else:
        dotted = "distllm." + ".".join(parts)
    if dotted in sys.modules:
        return sys.modules[dotted]
    spec = importlib.util.spec_from_file_location(dotted, filepath, submodule_search_locations=[])
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {filepath}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-load modules needed by fixtures
_coord_mod = _load_module("distllm/core/coordinator.py")
Coordinator = _coord_mod.Coordinator
_rm_mod = _load_module("distllm/core/resource_manager.py")
NodeRegistration = _rm_mod.NodeRegistration


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    tok.decode.return_value = "hello world"
    tok.eos_token_id = 0
    tok.pad_token_id = 0
    tok.vocab_size = 100
    return tok


@pytest.fixture
def mock_model_partitioner():
    partitioner = MagicMock()
    mock_model = MagicMock()
    mock_param = torch.randn(10, 10)
    mock_model.parameters.return_value = iter([mock_param])
    mock_model.config = MagicMock()
    mock_model.config.num_hidden_layers = 12
    mock_model.config.hidden_size = 768
    mock_model.config.num_attention_heads = 12
    partitioner.full_model = mock_model
    partitioner.start_layer = 0
    partitioner.end_layer = 11
    return partitioner


@pytest.fixture
def integration_coordinator(mock_tokenizer, mock_model_partitioner):
    coord = Coordinator(
        model_name="test-model",
        dtype="float32",
        max_batch_size=4,
        max_tokens_per_batch=512,
    )
    coord.tokenizer = mock_tokenizer
    coord.model_info = {"num_layers": 12, "hidden_size": 768, "num_attention_heads": 12}
    coord.total_layers = 12
    coord.local_partitioner = mock_model_partitioner
    return coord


@pytest.fixture
def integration_coordinator_with_nodes(mock_tokenizer, mock_model_partitioner):
    coord = Coordinator(
        model_name="test-model",
        dtype="float32",
        max_batch_size=4,
        max_tokens_per_batch=512,
    )
    coord.tokenizer = mock_tokenizer
    coord.model_info = {"num_layers": 12, "hidden_size": 768, "num_attention_heads": 12}
    coord.total_layers = 12
    coord.local_partitioner = mock_model_partitioner

    for i in range(2):
        mock_client = MagicMock()
        mock_health = MagicMock()
        mock_health.healthy = True
        mock_health.memory_used = 1024
        mock_health.memory_total = 8192
        mock_health.gpu_utilization = 0.5
        mock_client.health_check.return_value = mock_health

        mock_forward = MagicMock()
        mock_forward.success = True
        mock_forward.error_message = ""
        mock_forward.request_id = "test-request"
        mock_client.forward_pass.return_value = mock_forward

        mock_async_client = AsyncMock()
        mock_async_client.health_check.return_value = mock_health
        mock_async_client.forward_pass.return_value = mock_forward

        reg = NodeRegistration(
            node_id=f"node-{i}",
            host="localhost",
            port=50051 + i,
            start_layer=i * 6,
            end_layer=(i + 1) * 6 - 1,
        )
        reg.client = mock_client
        reg.async_client = mock_async_client
        reg.healthy = True
        coord.nodes[f"node-{i}"] = reg
        coord.node_order.append(f"node-{i}")

    return coord
