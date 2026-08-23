"""Abstract interface for vector database providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VectorDBInterface(ABC):
    """Abstract vector store with upsert, query, delete, and close operations.

    Implementations wrap a concrete vector-database SDK (Pinecone, Qdrant,
    Weaviate, Milvus, …) behind a uniform interface so that callers can
    switch providers without changing business logic.
    """

    # ------------------------------------------------------------------
    # Required operations
    # ------------------------------------------------------------------

    @abstractmethod
    def upsert(
        self,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
        *,
        namespace: str = "",
        batch_size: int | None = None,
    ) -> int:
        """Insert or update vectors with associated metadata.

        Args:
            vectors: Dense vector embeddings, one per record.
            metadata: Parallel list of metadata dicts (must include an
                ``"id"`` key for upsert identity).
            namespace: Optional index partition / namespace.
            batch_size: If set, split the records into batches of this
                size before sending to the provider.  ``None`` means
                use the provider's default (usually a single request).

        Returns:
            Number of records successfully upserted.
        """

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        *,
        metadata_filter: dict[str, Any] | None = None,
        namespace: str = "",
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        """Search for the ``top_k`` nearest neighbours of ``vector``.

        Args:
            vector: Query embedding.
            top_k: Number of neighbours to return.
            metadata_filter: Metadata filter expression (provider-specific
                syntax such as a dict of key/value pairs or a structured
                filter object).
            namespace: Optional index partition.
            include_metadata: Whether to attach stored metadata to hits.

        Returns:
            Each hit is a dict with keys ``"id"``, ``"score"``, and
            (if ``include_metadata``) ``"metadata"``.
        """

    @abstractmethod
    def delete(
        self,
        *,
        ids: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        namespace: str = "",
    ) -> int:
        """Delete vectors by ``ids``, ``metadata_filter``, or both (union).

        Args:
            ids: Specific record IDs to delete.
            metadata_filter: Metadata filter expression.
            namespace: Optional index partition.

        Returns:
            Number of records deleted.
        """

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    @abstractmethod
    def close(self) -> None:
        """Close the underlying client connection and release resources.

        After calling this method the instance should not be used for
        further operations until a new connection is established.
        """

    def __enter__(self) -> VectorDBInterface:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
