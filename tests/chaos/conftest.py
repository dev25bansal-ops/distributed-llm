"""Shared fixtures for chaos engineering tests.

Uses direct file imports (fake package injection) to avoid the
circular import chain in distllm/__init__.py.
"""

import importlib.util
import sys
import time as _time
import types
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"


def _make_fake_package(name: str, path: Path):
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    mod.__package__ = name
    sys.modules.setdefault(name, mod)
    return mod


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


_coord_mod = _load_module("distllm/core/coordinator.py")
Coordinator = _coord_mod.Coordinator
_rm_mod = _load_module("distllm/core/resource_manager.py")
ResourceManager = _rm_mod.ResourceManager
NodeRegistration = _rm_mod.NodeRegistration
CircuitBreakerConfig = _rm_mod.CircuitBreakerConfig



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
def chaos_coordinator(request, mock_tokenizer, monkeypatch):
    """Coordinator with multiple nodes for chaos testing.

    Supports parametrization with:
    - ``num_nodes`` (default 3): number of mock nodes to register
    - ``node_kwargs`` (default {}): extra kwargs for each NodeRegistration
    """
    num_nodes = getattr(request, "param", {}).get("num_nodes", 3)
    node_kwargs = getattr(request, "param", {}).get("node_kwargs", {})

    monkeypatch.setattr(NodeRegistration, "init_client", lambda self, **kw: None)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: mock_tokenizer)
    mock_config = MagicMock()
    mock_config.model_type = "mock"
    mock_config.num_hidden_layers = num_nodes * 6
    mock_config.hidden_size = 768
    mock_config.num_attention_heads = 12
    mock_config.vocab_size = 32000
    mock_config.max_position_embeddings = 2048
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda *a, **kw: mock_config)

    coord = Coordinator(model_name="chaos-test", dtype="float32", max_batch_size=4)
    coord.tokenizer = mock_tokenizer
    total = num_nodes * 6
    coord.model_info = {"num_layers": total, "hidden_size": 768, "num_attention_heads": 12}
    coord.total_layers = total

    layers_per = max(1, total // num_nodes)

    for i in range(num_nodes):
        coord.manual_register(
            node_id=f"n{i}", host="localhost", port=57000 + i,
            start_layer=i * layers_per,
            end_layer=(i + 1) * layers_per - 1,
            total_layers=total,
        )
        reg = coord.nodes[f"n{i}"]
        reg.client = MagicMock()
        reg.async_client = MagicMock()
        reg.healthy = True

        def _make_forward(healthy_flag=[True]):
            def forward(req, timeout=None):
                if not healthy_flag[0]:
                    import grpc
                    raise grpc.RpcError(grpc.StatusCode.UNAVAILABLE)
                resp = MagicMock()
                resp.success = True
                resp.error_message = ""
                resp.request_id = "test"
                resp.output = MagicMock()
                return resp
            return forward

        reg.client.stub = MagicMock()
        reg.client.stub.ForwardPass = MagicMock(side_effect=_make_forward())
        reg.client.stub.HealthCheck = MagicMock(return_value=MagicMock(
            healthy=True, memory_used_bytes=1024, memory_total_bytes=8192,
        ))
        reg.client.forward = MagicMock(return_value=MagicMock())
        reg.client.forward_pass = MagicMock(return_value=MagicMock(success=True, error_message=""))
        reg.client.health_check = MagicMock(return_value=MagicMock(healthy=True))

    return coord


@pytest.fixture
def rm_with_cb():
    """ResourceManager with a low-threshold circuit breaker for faster tests."""
    cb = CircuitBreakerConfig(threshold=2, base_delay=0.1, max_delay=5.0)
    rm = ResourceManager(cb_config=cb)
    for i in range(3):
        rm._node_failure_counts[f"n{i}"] = 0
    return rm


@pytest.fixture
def mock_grpc_error():
    """Factory for gRPC error responses."""
    import grpc

    def _make(code=grpc.StatusCode.UNAVAILABLE, details="chaos injected"):
        error = grpc.RpcError(details)
        error.code = lambda: code
        error.details = lambda: details
        return error

    return _make
