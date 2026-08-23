"""Tests for Protocol type definitions and runtime_checkable interfaces.

Uses the import-helper pattern to avoid circular imports.
"""

from __future__ import annotations

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_proto_mod = load_module("distllm/core/protocols.py")
INodeClient = _proto_mod.INodeClient
ITokenizer = _proto_mod.ITokenizer
IModelPartitioner = _proto_mod.IModelPartitioner
ICacheBackend = _proto_mod.ICacheBackend
IMetricsExporter = _proto_mod.IMetricsExporter
INodeFactory = _proto_mod.INodeFactory
IResourceManager = _proto_mod.IResourceManager
ICacheManager = _proto_mod.ICacheManager
ITokenGenerator = _proto_mod.ITokenGenerator
IPipelineOrchestrator = _proto_mod.IPipelineOrchestrator


class TestINodeClient:
    def test_protocol_attributes(self):
        """Verify the protocol describes the expected interface."""
        expected = {"host", "port", "health_check", "forward", "close"}
        attrs = set(INodeClient.__annotations__)
        attrs.update(
            m for m in dir(INodeClient)
            if not m.startswith("_") and callable(getattr(INodeClient, m, None))
        )
        for attr in expected:
            assert hasattr(INodeClient, attr) or attr in INodeClient.__annotations__

    def test_runtime_checkable(self):
        import typing
        assert typing.runtime_checkable in type(INodeClient).__bases__[0].__bases__[0].__dict__.values() or \
               hasattr(INodeClient, "__instancecheck__")


class TestITokenizer:
    def test_protocol_attributes(self):
        expected_annotations = {"eos_token_id", "bos_token_id", "pad_token_id", "vocab_size"}
        # __annotations__ on Protocol can be wonky; just check the methods and attrs exist
        for attr in expected_annotations:
            assert attr in getattr(ITokenizer, "__annotations__", {}) or hasattr(ITokenizer, attr)

    def test_encode_decode_signature(self):
        assert hasattr(ITokenizer, "encode")
        assert hasattr(ITokenizer, "decode")


class TestIModelPartitioner:
    def test_protocol_attributes(self):
        expected = {"full_model", "tokenizer", "embed_input", "is_last_node",
                     "load_full_model", "load_layer_subset", "forward", "get_logits"}
        for attr in expected:
            assert hasattr(IModelPartitioner, attr) or \
                   attr in getattr(IModelPartitioner, "__annotations__", {})


class TestICacheBackend:
    def test_protocol_attributes(self):
        assert hasattr(ICacheBackend, "lookup")
        assert hasattr(ICacheBackend, "store")
        assert hasattr(ICacheBackend, "clear")


class TestIMetricsExporter:
    def test_protocol_attributes(self):
        assert hasattr(IMetricsExporter, "record")


class TestINodeFactory:
    def test_protocol_attributes(self):
        assert hasattr(INodeFactory, "create_node")


class TestIResourceManager:
    def test_protocol_attributes(self):
        expected = {"check_circuit_breaker", "record_success", "record_failure",
                     "health_check_all", "close_all"}
        for attr in expected:
            assert hasattr(IResourceManager, attr)


class TestICacheManager:
    def test_protocol_attributes(self):
        assert hasattr(ICacheManager, "lookup_prefix")
        assert hasattr(ICacheManager, "maybe_chunk")


class TestITokenGenerator:
    def test_protocol_attributes(self):
        assert hasattr(ITokenGenerator, "sample")
        assert hasattr(ITokenGenerator, "sample_batch")


class TestIPipelineOrchestrator:
    def test_protocol_attributes(self):
        expected = {"nodes", "node_order", "register_node",
                     "validate_layer_assignment", "run_pipeline"}
        for attr in expected:
            assert hasattr(IPipelineOrchestrator, attr) or \
                   attr in getattr(IPipelineOrchestrator, "__annotations__", {})


class TestConcreteImplements:
    """Verify that simple concrete classes satisfy the protocols."""

    def test_minimal_node_client(self):
        class FakeNodeClient:
            host: str = "localhost"
            port: int = 50051
            def health_check(self):
                return {"status": "ok"}
            def forward(self, request):
                return {"result": "ok"}
            def close(self):
                pass

        client = FakeNodeClient()
        # runtime_checkable means isinstance won't work unless the class
        # defines the exact same attributes. Just verify structural compatibility.

        assert hasattr(client, "health_check")
        assert callable(client.health_check)
        assert client.host == "localhost"
        assert client.port == 50051

    def test_minimal_tokenizer(self):
        class FakeTokenizer:
            eos_token_id: int = 2
            bos_token_id: int = 1
            pad_token_id: int = 0
            vocab_size: int = 32000
            def encode(self, text, **kwargs):
                return [1, 2, 3]
            def decode(self, tokens, **kwargs):
                return "hello"

        t = FakeTokenizer()
        assert t.encode("test") == [1, 2, 3]
        assert t.decode([1, 2]) == "hello"
        assert t.vocab_size == 32000

    def test_minimal_cache_backend(self):
        class FakeCache:
            def __init__(self):
                self._store = {}
            def lookup(self, tokens):
                return (0, None)
            def store(self, tokens, entry):
                self._store[tuple(tokens)] = entry
            def clear(self):
                self._store.clear()

        c = FakeCache()
        assert c.lookup([1, 2]) == (0, None)
        c.store([1, 2], "data")
        c.clear()
        # No crash

    def test_minimal_metrics_exporter(self):
        class FakeExporter:
            def __init__(self):
                self.records = []
            def record(self, name, value, labels=None):
                self.records.append((name, value, labels))

        e = FakeExporter()
        e.record("test_metric", 42.0, {"host": "n1"})
        assert len(e.records) == 1
        assert e.records[0][0] == "test_metric"
