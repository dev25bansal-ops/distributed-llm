"""Tests for the VectorDBInterface contract and VectorDBFactory."""

from __future__ import annotations

from typing import Any

import pytest

from distllm.core.vectorstore import VectorDBFactory, VectorDBInterface


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _MockStore(VectorDBInterface):
    """Minimal in-memory implementation for contract testing."""

    def __init__(self) -> None:
        self._closed = False
        self._data: dict[str, dict[str, Any]] = {}

    def upsert(
        self,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
        *,
        namespace: str = "",
        batch_size: int | None = None,
    ) -> int:
        self._closed = False
        for vec, meta in zip(vectors, metadata):
            rid = meta.get("id", str(len(self._data)))
            self._data[rid] = {"vector": vec, "metadata": meta, "namespace": namespace}
        return len(vectors)

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        *,
        metadata_filter: dict[str, Any] | None = None,
        namespace: str = "",
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        results = [
            {
                "id": rid,
                "score": 0.5,
                "metadata": info["metadata"],
            }
            for rid, info in self._data.items()
            if info["namespace"] == namespace
        ]
        return results[:top_k]

    def delete(
        self,
        *,
        ids: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        namespace: str = "",
    ) -> int:
        count = 0
        if ids:
            for rid in ids:
                if rid in self._data:
                    del self._data[rid]
                    count += 1
        return count

    def close(self) -> None:
        self._closed = True


@pytest.fixture
def mock_store() -> _MockStore:
    return _MockStore()


@pytest.fixture
def sample_vectors() -> list[list[float]]:
    return [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


@pytest.fixture
def sample_metadata() -> list[dict[str, Any]]:
    return [{"id": "doc1", "source": "wiki"}, {"id": "doc2", "source": "news"}]


# ---------------------------------------------------------------------------
# ABC contract tests
# ---------------------------------------------------------------------------


class TestVectorDBInterface:
    """Verify that VectorDBInterface cannot be instantiated directly."""

    def test_abstract_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            VectorDBInterface()  # type: ignore[abstract]

    def test_mock_store_is_instance(self, mock_store: _MockStore) -> None:
        assert isinstance(mock_store, VectorDBInterface)


class TestUpsert:
    def test_upsert_returns_count(
        self,
        mock_store: _MockStore,
        sample_vectors: list[list[float]],
        sample_metadata: list[dict[str, Any]],
    ) -> None:
        count = mock_store.upsert(sample_vectors, sample_metadata)
        assert count == 2

    def test_upsert_with_namespace(
        self,
        mock_store: _MockStore,
    ) -> None:
        vectors = [[0.1, 0.2]]
        metadata = [{"id": "ns_doc"}]
        count = mock_store.upsert(vectors, metadata, namespace="test_ns")
        assert count == 1

    def test_upsert_with_batch_size(
        self,
        mock_store: _MockStore,
    ) -> None:
        vectors = [[0.1], [0.2], [0.3], [0.4]]
        metadata = [{"id": f"d{i}"} for i in range(4)]
        count = mock_store.upsert(vectors, metadata, batch_size=2)
        assert count == 4

    def test_upsert_batch_size_none(
        self,
        mock_store: _MockStore,
    ) -> None:
        vectors = [[0.1], [0.2]]
        metadata = [{"id": "a"}, {"id": "b"}]
        count = mock_store.upsert(vectors, metadata, batch_size=None)
        assert count == 2

    def test_upsert_empty_vectors(
        self,
        mock_store: _MockStore,
    ) -> None:
        count = mock_store.upsert([], [])
        assert count == 0


class TestQuery:
    def test_query_returns_hits(
        self,
        mock_store: _MockStore,
    ) -> None:
        mock_store.upsert([[0.1, 0.2]], [{"id": "d1"}])
        results = mock_store.query([0.1, 0.2], top_k=5)
        assert len(results) == 1
        assert results[0]["id"] == "d1"
        assert "score" in results[0]

    def test_query_respects_top_k(
        self,
        mock_store: _MockStore,
    ) -> None:
        vectors = [[0.1], [0.2], [0.3]]
        metadata = [{"id": f"d{i}"} for i in range(3)]
        mock_store.upsert(vectors, metadata)
        results = mock_store.query([0.1], top_k=2)
        assert len(results) == 2

    def test_query_returns_empty_when_no_data(self, mock_store: _MockStore) -> None:
        results = mock_store.query([0.1, 0.2])
        assert results == []

    def test_query_with_metadata_filter(
        self,
        mock_store: _MockStore,
    ) -> None:
        mock_store.upsert(
            [[0.1], [0.2]],
            [{"id": "d1", "source": "wiki"}, {"id": "d2", "source": "news"}],
        )
        # The mock doesn't filter by metadata_filter, but the parameter
        # should be accepted without error
        results = mock_store.query([0.1], metadata_filter={"source": "wiki"})
        assert isinstance(results, list)

    def test_query_with_namespace(
        self,
        mock_store: _MockStore,
    ) -> None:
        mock_store.upsert([[0.1]], [{"id": "d1"}], namespace="ns1")
        results = mock_store.query([0.1], namespace="ns1")
        assert len(results) == 1
        results = mock_store.query([0.1], namespace="other")
        assert len(results) == 0

    def test_query_include_metadata(
        self,
        mock_store: _MockStore,
    ) -> None:
        mock_store.upsert([[0.1]], [{"id": "d1", "source": "wiki"}])
        results = mock_store.query([0.1], include_metadata=True)
        assert "metadata" in results[0]
        results_no_meta = mock_store.query([0.1], include_metadata=False)
        assert "metadata" in results_no_meta[0]  # mock always returns it


class TestDelete:
    def test_delete_by_id(
        self,
        mock_store: _MockStore,
    ) -> None:
        mock_store.upsert([[0.1]], [{"id": "d1"}])
        count = mock_store.delete(ids=["d1"])
        assert count == 1
        assert mock_store.query([0.1]) == []

    def test_delete_nonexistent_id(
        self,
        mock_store: _MockStore,
    ) -> None:
        count = mock_store.delete(ids=["ghost"])
        assert count == 0

    def test_delete_with_metadata_filter(
        self,
        mock_store: _MockStore,
    ) -> None:
        mock_store.upsert(
            [[0.1]], [{"id": "d1", "source": "wiki"}],
        )
        count = mock_store.delete(metadata_filter={"source": "wiki"})
        # Mock doesn't filter — just verifies no error
        assert isinstance(count, int)

    def test_delete_no_args(self, mock_store: _MockStore) -> None:
        count = mock_store.delete()
        assert count == 0


class TestClose:
    def test_close_method(self, mock_store: _MockStore) -> None:
        assert not mock_store._closed
        mock_store.close()
        assert mock_store._closed

    def test_context_manager_calls_close(self, mock_store: _MockStore) -> None:
        with mock_store as store:
            assert store is mock_store
            assert not mock_store._closed
        assert mock_store._closed

    def test_double_close_ok(self, mock_store: _MockStore) -> None:
        mock_store.close()
        mock_store.close()  # should not raise
        assert mock_store._closed


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestVectorDBFactory:
    """Verify factory creates the right types and raises on unknowns."""

    def test_list_providers_returns_sorted(self) -> None:
        providers = VectorDBFactory.list_providers()
        assert providers == sorted(providers)
        assert "pinecone" in providers
        assert "qdrant" in providers
        assert "weaviate" in providers
        assert "milvus" in providers

    def test_create_pinecone_requires_api_key(self) -> None:
        # No config and no env var → should raise ValueError
        with pytest.raises(ValueError, match="api_key"):
            VectorDBFactory.create("pinecone")

    def test_create_qdrant_defaults(self) -> None:
        store = VectorDBFactory.create("qdrant", {"url": "http://localhost:6333"})
        # The Qdrant client is lazy-initialized; just verify we get an instance
        from distllm.core.vectorstore.providers.qdrant import _QdrantStore
        assert isinstance(store, _QdrantStore)

    def test_create_unknown_provider(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            VectorDBFactory.create("nonexistent")

    def test_create_case_insensitive(self) -> None:
        store = VectorDBFactory.create("QDRANT", {"url": "http://localhost:6333"})
        from distllm.core.vectorstore.providers.qdrant import _QdrantStore
        assert isinstance(store, _QdrantStore)

    def test_all_providers_return_interface(self) -> None:
        from distllm.core.vectorstore.providers.pinecone import _PineconeStore
        from distllm.core.vectorstore.providers.qdrant import _QdrantStore
        from distllm.core.vectorstore.providers.weaviate import _WeaviateStore
        from distllm.core.vectorstore.providers.milvus import _MilvusStore

        assert issubclass(_PineconeStore, VectorDBInterface)
        assert issubclass(_QdrantStore, VectorDBInterface)
        assert issubclass(_WeaviateStore, VectorDBInterface)
        assert issubclass(_MilvusStore, VectorDBInterface)


# ---------------------------------------------------------------------------
# Parameter name tests — verify ``filter`` → ``metadata_filter`` rename
# ---------------------------------------------------------------------------


class TestParameterRename:
    """Confirm that ``filter`` is no longer used as a parameter name."""

    def test_upsert_has_batch_size_param(self) -> None:
        import inspect
        sig = inspect.signature(VectorDBInterface.upsert)
        assert "batch_size" in sig.parameters
        assert sig.parameters["batch_size"].default is None

    def test_query_has_metadata_filter_not_filter(self) -> None:
        import inspect
        sig = inspect.signature(VectorDBInterface.query)
        assert "metadata_filter" in sig.parameters
        assert "filter" not in sig.parameters

    def test_delete_has_metadata_filter_not_filter(self) -> None:
        import inspect
        sig = inspect.signature(VectorDBInterface.delete)
        assert "metadata_filter" in sig.parameters
        assert "filter" not in sig.parameters

    def test_rag_pipeline_has_metadata_filter(self) -> None:
        from distllm.core.vectorstore.rag import RAGPipeline
        import inspect
        sig = inspect.signature(RAGPipeline.retrieve)
        assert "metadata_filter" in sig.parameters
        assert "filter" not in sig.parameters
