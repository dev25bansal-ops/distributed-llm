"""High-level RAG pipeline: embed, store, retrieve."""

from __future__ import annotations

from typing import Any

from distllm.core.vectorstore.base import VectorDBInterface


class RAGPipeline:
    """High-level pipeline: embed chunks, store embeddings, retrieve by query.

    The ``embedder`` callable maps a list of text strings to a list of
    embedding vectors — any SDK (OpenAI, Cohere, DistLLM, Sentence
    Transformers, etc.) can be plugged in as long as it matches the
    signature ``(texts: list[str]) -> list[list[float]]``.

    Usage::

        pipe = RAGPipeline(
            embedder=openai_embedder,
            vector_store=VectorDBFactory.create("qdrant"),
        )
        vectors = pipe.embed(["chunk1", "chunk2"])
        pipe.store(vectors, [{"id": "1"}, {"id": "2"}])
        results = pipe.retrieve("my question", top_k=3)
    """

    def __init__(
        self,
        embedder: Any,
        vector_store: VectorDBInterface | None = None,
    ) -> None:
        self._embedder = embedder
        self._store: VectorDBInterface | None = vector_store

    @property
    def vector_store(self) -> VectorDBInterface | None:
        """The backing vector store (may be ``None`` if not yet set)."""
        return self._store

    @vector_store.setter
    def vector_store(self, value: VectorDBInterface) -> None:
        self._store = value

    # --- public API -------------------------------------------------------

    def embed(self, chunks: list[str]) -> list[list[float]]:
        """Convert text chunks into embedding vectors.

        Returns a list of float vectors, one per input chunk.  The caller
        should pair these with matching metadata and pass both to
        :meth:`store`.
        """
        return self._embedder(chunks)

    def store(
        self,
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]],
        *,
        namespace: str = "",
        batch_size: int | None = None,
    ) -> int:
        """Persist embeddings and their metadata into the vector store.

        Args:
            embeddings: Output from :meth:`embed`.
            metadata: Parallel list of metadata dicts (each must contain
                ``"id"``).
            namespace: Optional partition.
            batch_size: Optional batch size for the upsert.

        Returns:
            Number of records upserted.

        Raises:
            RuntimeError: No vector store configured.
        """
        if self._store is None:
            raise RuntimeError(
                "RAGPipeline.store is not set — assign a VectorDBInterface "
                "instance or pass one to the constructor"
            )
        return self._store.upsert(
            embeddings, metadata, namespace=namespace, batch_size=batch_size
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        metadata_filter: dict[str, Any] | None = None,
        namespace: str = "",
    ) -> list[dict[str, Any]]:
        """Embed ``query`` and search the vector store.

        Args:
            query: Natural-language query string.
            top_k: Number of results.
            metadata_filter: Metadata filter (provider-specific).
            namespace: Optional partition.

        Returns:
            List of hit dicts with ``"id"``, ``"score"``, ``"metadata"``.

        Raises:
            RuntimeError: No vector store configured.
        """
        if self._store is None:
            raise RuntimeError(
                "RAGPipeline.store is not set — assign a VectorDBInterface "
                "instance or pass one to the constructor"
            )
        query_vector = self._embedder([query])[0]
        return self._store.query(
            query_vector,
            top_k=top_k,
            metadata_filter=metadata_filter,
            namespace=namespace,
        )

    def delete(
        self,
        *,
        ids: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        namespace: str = "",
    ) -> int:
        """Remove records from the vector store.

        Args:
            ids: Record IDs to remove.
            metadata_filter: Metadata filter (provider-specific).
            namespace: Optional partition.

        Returns:
            Number of records deleted.

        Raises:
            RuntimeError: No vector store configured.
        """
        if self._store is None:
            raise RuntimeError(
                "RAGPipeline.store is not set — assign a VectorDBInterface "
                "instance or pass one to the constructor"
            )
        return self._store.delete(
            ids=ids, metadata_filter=metadata_filter, namespace=namespace
        )
