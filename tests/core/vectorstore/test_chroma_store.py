"""Tests for ChromaStore (real adapter, fail-closed without chromadb).

Covers:
- Default and custom construction (host, port, collection_name)
- Fail-closed behavior without the optional chromadb driver
- list_collections behavior with the configured collection
"""

from __future__ import annotations

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

load_module("distllm/core/vectorstore/base.py")  # required dependency
_mod = load_module("distllm/core/vectorstore/chroma_store.py")
ChromaStore = _mod.ChromaStore


class TestChromaStoreConstruction:
    """Construction and default parameter values."""

    def test_default_construction(self):
        store = ChromaStore()
        assert store.host == "localhost"
        assert store.port == 8000
        assert store.collection_name == "distllm"

    def test_custom_construction(self):
        store = ChromaStore(host="10.0.0.99", port=9000, collection_name="my-vectors")
        assert store.host == "10.0.0.99"
        assert store.port == 9000
        assert store.collection_name == "my-vectors"

    def test_partial_construction(self):
        store = ChromaStore(port=7000)
        assert store.host == "localhost"
        assert store.port == 7000
        assert store.collection_name == "distllm"


class TestChromaStoreFailClosed:
    """Without the optional chromadb driver, methods fail closed."""

    def test_upsert_fails_closed(self):
        store = ChromaStore(host="localhost", port=8000)
        try:
            store.upsert([[0.1, 0.2]], ["x"])
            raise AssertionError("expected RuntimeError (chromadb not installed)")
        except RuntimeError as e:
            assert "chromadb" in str(e)

    def test_query_fails_closed(self):
        store = ChromaStore()
        try:
            store.query([0.1, 0.2])
            raise AssertionError("expected RuntimeError (chromadb not installed)")
        except RuntimeError as e:
            assert "chromadb" in str(e)

    def test_delete_fails_closed(self):
        store = ChromaStore()
        try:
            store.delete(["id1"])
            raise AssertionError("expected RuntimeError (chromadb not installed)")
        except RuntimeError as e:
            assert "chromadb" in str(e)

    def test_list_collections_fails_closed(self):
        store = ChromaStore()
        try:
            store.list_collections()
            raise AssertionError("expected RuntimeError (chromadb not installed)")
        except RuntimeError as e:
            assert "chromadb" in str(e)


class TestChromaStoreListCollections:
    """list_collections returns the configured collection name."""

    def test_returns_collection_name(self):
        store = ChromaStore(collection_name="my_coll")
        assert store.collection_name == "my_coll"

    def test_default_name(self):
        store = ChromaStore()
        assert store.collection_name == "distllm"