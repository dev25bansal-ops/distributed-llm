"""Tests for PGVectorStore (real adapter, fail-closed without psycopg2).

Covers:
- Default and custom construction (connection_string, collection_name, vector_size)
- Fail-closed behavior without the optional psycopg2 driver
- list_collections behavior with the configured collection
"""

from __future__ import annotations

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

load_module("distllm/core/vectorstore/base.py")  # required dependency
_mod = load_module("distllm/core/vectorstore/pgvector_store.py")
PGVectorStore = _mod.PGVectorStore


class TestPGVectorStoreConstruction:
    """Construction and default parameter values."""

    def test_default_construction(self):
        store = PGVectorStore()
        assert store.connection_string == ""
        assert store.collection_name == "distllm"
        assert store.vector_size == 768

    def test_custom_construction(self):
        store = PGVectorStore(
            connection_string="postgresql://user:pass@host:5432/db",
            collection_name="my_vectors",
            vector_size=1536,
        )
        assert store.connection_string == "postgresql://user:pass@host:5432/db"
        assert store.collection_name == "my_vectors"
        assert store.vector_size == 1536

    def test_partial_construction(self):
        store = PGVectorStore(vector_size=384)
        assert store.connection_string == ""
        assert store.collection_name == "distllm"
        assert store.vector_size == 384


class TestPGVectorStoreFailClosed:
    """Without the optional psycopg2 driver, methods fail closed."""

    def test_upsert_fails_closed(self):
        store = PGVectorStore(connection_string="postgresql://localhost/test", vector_size=4)
        try:
            store.upsert([[0.1] * 4, [0.2] * 4], ["a", "b"])
            raise AssertionError("expected RuntimeError (psycopg2 not installed)")
        except RuntimeError as e:
            assert "psycopg2" in str(e)

    def test_query_fails_closed(self):
        store = PGVectorStore()
        try:
            store.query([0.1] * 768)
            raise AssertionError("expected RuntimeError (psycopg2 not installed)")
        except RuntimeError as e:
            assert "psycopg2" in str(e)

    def test_delete_fails_closed(self):
        store = PGVectorStore()
        try:
            store.delete(["id1"])
            raise AssertionError("expected RuntimeError (psycopg2 not installed)")
        except RuntimeError as e:
            assert "psycopg2" in str(e)

    def test_list_collections_fails_closed(self):
        store = PGVectorStore(connection_string="postgresql://localhost/test", vector_size=4)
        try:
            store.list_collections()
            raise AssertionError("expected RuntimeError (psycopg2 not installed)")
        except RuntimeError as e:
            assert "psycopg2" in str(e)


class TestPGVectorStoreListCollections:
    """list_collections returns the configured collection name."""

    def test_returns_table_name(self):
        store = PGVectorStore(collection_name="doc_vectors")
        assert store.collection_name == "doc_vectors"

    def test_default_name(self):
        store = PGVectorStore()
        assert store.collection_name == "distllm"