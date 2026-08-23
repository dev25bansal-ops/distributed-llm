"""Tests for the RAGPipeline: embed, store, retrieve, delete."""

from __future__ import annotations

from typing import Any

import pytest

from distllm.core.vectorstore import RAGPipeline, VectorDBInterface


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class _MockStore(VectorDBInterface):
    """In-memory vector store for testing the pipeline."""

    def __init__(self) -> None:
        self._data: list[dict[str, Any]] = []
        self.last_upsert_batch_size: int | None = None
        self.last_upsert_namespace: str = ""
        self.last_query_namespace: str = ""
        self.last_query_filter: dict[str, Any] | None = None
        self.last_delete_ids: list[str] | None = None
        self.last_delete_filter: dict[str, Any] | None = None
        self.last_delete_namespace: str = ""

    def upsert(
        self,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
        *,
        namespace: str = "",
        batch_size: int | None = None,
    ) -> int:
        self.last_upsert_batch_size = batch_size
        self.last_upsert_namespace = namespace
        for vec, meta in zip(vectors, metadata):
            self._data.append(
                {
                    "id": meta.get("id", str(len(self._data))),
                    "vector": vec,
                    "metadata": {k: v for k, v in meta.items() if k != "id"},
                    "namespace": namespace,
                }
            )
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
        self.last_query_filter = metadata_filter
        self.last_query_namespace = namespace
        hits = [
            {
                "id": d["id"],
                "score": 0.5,
                "metadata": d["metadata"],
            }
            for d in self._data
            if d["namespace"] == namespace
        ]
        return hits[:top_k]

    def delete(
        self,
        *,
        ids: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        namespace: str = "",
    ) -> int:
        self.last_delete_ids = ids
        self.last_delete_filter = metadata_filter
        self.last_delete_namespace = namespace
        count = 0
        if ids:
            remaining: list[dict[str, Any]] = []
            for d in self._data:
                if d["id"] in ids:
                    count += 1
                else:
                    remaining.append(d)
            self._data = remaining
        return count

    def close(self) -> None:
        self._data.clear()


def _dummy_embedder(texts: list[str]) -> list[list[float]]:
    """Deterministic embedder for testing."""
    return [[float(ord(c)) for c in text[:4].ljust(4, "\x00")] for text in texts]


@pytest.fixture
def mock_store() -> _MockStore:
    return _MockStore()


@pytest.fixture
def pipeline(mock_store: _MockStore) -> RAGPipeline:
    return RAGPipeline(embedder=_dummy_embedder, vector_store=mock_store)


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------


class TestEmbed:
    def test_embed_returns_vectors(self, pipeline: RAGPipeline) -> None:
        chunks = ["hello", "world"]
        vectors = pipeline.embed(chunks)
        assert len(vectors) == 2
        assert all(isinstance(v, list) for v in vectors)
        assert all(isinstance(x, float) for v in vectors for x in v)

    def test_embed_empty_list(self, pipeline: RAGPipeline) -> None:
        vectors = pipeline.embed([])
        assert vectors == []

    def test_embed_single_chunk(self, pipeline: RAGPipeline) -> None:
        vectors = pipeline.embed(["test"])
        assert len(vectors) == 1


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class TestStore:
    def test_store_returns_count(
        self,
        pipeline: RAGPipeline,
        mock_store: _MockStore,
    ) -> None:
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        metadata = [{"id": "a"}, {"id": "b"}]
        count = pipeline.store(embeddings, metadata)
        assert count == 2

    def test_store_with_namespace(
        self,
        pipeline: RAGPipeline,
        mock_store: _MockStore,
    ) -> None:
        embeddings = [[0.1]]
        metadata = [{"id": "doc1"}]
        pipeline.store(embeddings, metadata, namespace="ns1")
        assert mock_store.last_upsert_namespace == "ns1"

    def test_store_with_batch_size(
        self,
        pipeline: RAGPipeline,
        mock_store: _MockStore,
    ) -> None:
        embeddings = [[0.1], [0.2]]
        metadata = [{"id": "a"}, {"id": "b"}]
        pipeline.store(embeddings, metadata, batch_size=1)
        assert mock_store.last_upsert_batch_size == 1

    def test_store_raises_without_store(self) -> None:
        pipe = RAGPipeline(embedder=_dummy_embedder, vector_store=None)
        with pytest.raises(RuntimeError, match="store is not set"):
            pipe.store([[0.1]], [{"id": "doc1"}])


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------


class TestRetrieve:
    def test_retrieve_returns_results(
        self,
        pipeline: RAGPipeline,
    ) -> None:
        pipeline.store([[0.1, 0.2]], [{"id": "d1"}])
        results = pipeline.retrieve("hello")
        assert len(results) > 0
        assert results[0]["id"] == "d1"

    def test_retrieve_with_metadata_filter(
        self,
        pipeline: RAGPipeline,
        mock_store: _MockStore,
    ) -> None:
        pipeline.store([[0.1]], [{"id": "d1"}])
        pipeline.retrieve("hello", metadata_filter={"source": "wiki"})
        assert mock_store.last_query_filter == {"source": "wiki"}

    def test_retrieve_with_namespace(
        self,
        pipeline: RAGPipeline,
        mock_store: _MockStore,
    ) -> None:
        pipeline.store([[0.1]], [{"id": "d1"}], namespace="ns1")
        pipeline.retrieve("hello", namespace="ns1")
        assert mock_store.last_query_namespace == "ns1"

    def test_retrieve_top_k(
        self,
        pipeline: RAGPipeline,
    ) -> None:
        embeddings = [[0.1], [0.2], [0.3], [0.4], [0.5]]
        metadata = [{"id": f"d{i}"} for i in range(5)]
        pipeline.store(embeddings, metadata)
        results = pipeline.retrieve("hello", top_k=3)
        assert len(results) == 3

    def test_retrieve_raises_without_store(self) -> None:
        pipe = RAGPipeline(embedder=_dummy_embedder, vector_store=None)
        with pytest.raises(RuntimeError, match="store is not set"):
            pipe.retrieve("hello")


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_by_ids(
        self,
        pipeline: RAGPipeline,
        mock_store: _MockStore,
    ) -> None:
        pipeline.store([[0.1]], [{"id": "d1"}])
        count = pipeline.delete(ids=["d1"])
        assert count == 1

    def test_delete_passes_ids(
        self,
        pipeline: RAGPipeline,
        mock_store: _MockStore,
    ) -> None:
        pipeline.delete(ids=["a", "b"])
        assert mock_store.last_delete_ids == ["a", "b"]

    def test_delete_passes_metadata_filter(
        self,
        pipeline: RAGPipeline,
        mock_store: _MockStore,
    ) -> None:
        pipeline.delete(metadata_filter={"source": "wiki"})
        assert mock_store.last_delete_filter == {"source": "wiki"}

    def test_delete_passes_namespace(
        self,
        pipeline: RAGPipeline,
        mock_store: _MockStore,
    ) -> None:
        pipeline.delete(ids=["d1"], namespace="ns1")
        assert mock_store.last_delete_namespace == "ns1"

    def test_delete_raises_without_store(self) -> None:
        pipe = RAGPipeline(embedder=_dummy_embedder, vector_store=None)
        with pytest.raises(RuntimeError, match="store is not set"):
            pipe.delete(ids=["d1"])


# ---------------------------------------------------------------------------
# Store setter / getter
# ---------------------------------------------------------------------------


class TestStoreAccessors:
    def test_vector_store_property_initially_none(self) -> None:
        pipe = RAGPipeline(embedder=_dummy_embedder, vector_store=None)
        assert pipe.vector_store is None

    def test_vector_store_setter(
        self,
        mock_store: _MockStore,
    ) -> None:
        pipe = RAGPipeline(embedder=_dummy_embedder, vector_store=None)
        assert pipe.vector_store is None
        pipe.vector_store = mock_store
        assert pipe.vector_store is mock_store

    def test_vector_store_property_via_constructor(
        self,
        mock_store: _MockStore,
        pipeline: RAGPipeline,
    ) -> None:
        assert pipeline.vector_store is mock_store


# ---------------------------------------------------------------------------
# Full flow (integration-style with mocks)
# ---------------------------------------------------------------------------


class TestFullFlow:
    def test_embed_store_retrieve_cycle(self) -> None:
        store = _MockStore()
        pipe = RAGPipeline(embedder=_dummy_embedder, vector_store=store)

        # Embed
        chunks = ["apple banana", "cherry date"]
        vectors = pipe.embed(chunks)
        assert len(vectors) == 2

        # Store
        metadata = [{"id": "doc1"}, {"id": "doc2"}]
        count = pipe.store(vectors, metadata)
        assert count == 2

        # Retrieve
        results = pipe.retrieve("apple", top_k=1)
        assert len(results) >= 1

        # Delete
        deleted = pipe.delete(ids=["doc1"])
        assert deleted == 1

    def test_embed_store_with_batch_size(self) -> None:
        store = _MockStore()
        pipe = RAGPipeline(embedder=_dummy_embedder, vector_store=store)
        vectors = pipe.embed(["a", "b", "c", "d"])
        metadata = [{"id": f"d{i}"} for i in range(4)]
        pipe.store(vectors, metadata, batch_size=2)
        assert store.last_upsert_batch_size == 2

    def test_delete_no_args_returns_zero(self) -> None:
        store = _MockStore()
        pipe = RAGPipeline(embedder=_dummy_embedder, vector_store=store)
        count = pipe.delete()
        assert count == 0
