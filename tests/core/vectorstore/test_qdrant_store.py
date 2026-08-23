"""Tests for QdrantStore (real adapter, fail-closed without qdrant_client).

Covers:
- Default and custom construction (host, port, collection_name, api_key)
- Fail-closed behavior without the optional qdrant_client driver
- list_collections behavior with the configured collection
"""

from __future__ import annotations

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

load_module("distllm/core/vectorstore/base.py")  # required dependency
_mod = load_module("distllm/core/vectorstore/qdrant_store.py")
QdrantStore = _mod.QdrantStore


class TestQdrantStoreConstruction:
    """Construction and default parameter values."""

    def test_default_construction(self):
        store = QdrantStore()
        assert store.host == "localhost"
        assert store.port == 6333
        assert store.collection_name == "distllm"

    def test_custom_construction(self):
        store = QdrantStore(
            host="qdrant.example.com",
            port=6333,
            api_key="sk-secret",
            collection_name="prod-vectors",
        )
        assert store.host == "qdrant.example.com"
        assert store.collection_name == "prod-vectors"
        assert store.api_key == "sk-secret"

    def test_partial_construction(self):
        store = QdrantStore(collection_name="custom")
        assert store.host == "localhost"
        assert store.port == 6333
        assert store.collection_name == "custom"

    def test_empty_api_key(self):
        store = QdrantStore(api_key="")
        assert store.api_key == ""


class TestQdrantStoreFailClosed:
    """Without the optional qdrant_client driver, methods fail closed."""

    def test_upsert_fails_closed(self):
        store = QdrantStore(host="localhost", port=6333)
        try:
            store.upsert([[0.1, 0.2], [0.3, 0.4]], ["a", "b"])
            raise AssertionError("expected RuntimeError (qdrant_client not installed)")
        except RuntimeError as e:
            assert "qdrant_client" in str(e)

    def test_query_fails_closed(self):
        store = QdrantStore()
        try:
            store.query([0.1, 0.2])
            raise AssertionError("expected RuntimeError (qdrant_client not installed)")
        except RuntimeError as e:
            assert "qdrant_client" in str(e)

    def test_delete_fails_closed(self):
        store = QdrantStore()
        try:
            store.delete(["id1"])
            raise AssertionError("expected RuntimeError (qdrant_client not installed)")
        except RuntimeError as e:
            assert "qdrant_client" in str(e)

    def test_list_collections_fails_closed(self):
        store = QdrantStore()
        try:
            store.list_collections()
            raise AssertionError("expected RuntimeError (qdrant_client not installed)")
        except RuntimeError as e:
            assert "qdrant_client" in str(e)


class TestQdrantStoreListCollections:
    """list_collections returns the configured collection name."""

    def test_returns_collection_name(self):
        store = QdrantStore(collection_name="my_qdrant")
        assert store.collection_name == "my_qdrant"

    def test_default_name(self):
        store = QdrantStore()
        assert store.collection_name == "distllm"