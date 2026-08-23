"""Qdrant vector store provider.

Requires the ``qdrant-client`` SDK (``pip install qdrant-client``).
"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger

from distllm.core.vectorstore.base import VectorDBInterface


class _QdrantStore(VectorDBInterface):
    """Qdrant_ vector store wrapper.

    .. _Qdrant: https://qdrant.tech
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._url: str = (
            config.get("url")
            or os.environ.get("QDRANT_URL", "http://localhost:6333")
        )
        self._api_key: str | None = config.get("api_key") or os.environ.get(
            "QDRANT_API_KEY"
        )
        self._collection: str = config.get(
            "collection", config.get("index", "default")
        )
        self._dimension: int = config.get("dimension", 1536)
        self._client: Any = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import (
                Distance,
                PointStruct,
                VectorParams,
            )
        except ImportError as exc:
            raise ImportError(
                "Qdrant SDK not installed — run `pip install qdrant-client`"
            ) from exc

        self._client = QdrantClient(
            url=self._url, api_key=self._api_key, prefer_grpc=True
        )
        # Keep model references so pickle-based concurrency works
        self._PointStruct = PointStruct
        self._VectorParams = VectorParams
        self._Distance = Distance

        collections = self._client.get_collections().collections
        if not any(c.name == self._collection for c in collections):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._dimension, distance=Distance.COSINE
                ),
            )
        return self._client

    # ------------------------------------------------------------------
    # VectorDBInterface
    # ------------------------------------------------------------------

    def upsert(
        self,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
        *,
        namespace: str = "",
        batch_size: int | None = None,
    ) -> int:
        if namespace:
            raise ValueError("Qdrant provider does not support namespaces")
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata must have equal length")
        client = self._ensure_client()
        points = [
            self._PointStruct(
                id=meta.get("id", i),
                vector=vec,
                payload={k: v for k, v in meta.items() if k != "id"},
            )
            for i, (vec, meta) in enumerate(zip(vectors, metadata))
        ]
        if batch_size is not None:
            total = 0
            for start in range(0, len(points), batch_size):
                batch = points[start : start + batch_size]
                client.upsert(
                    collection_name=self._collection, points=batch
                )
                total += len(batch)
            return total
        client.upsert(
            collection_name=self._collection, points=points
        )
        return len(points)

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        *,
        metadata_filter: dict[str, Any] | None = None,
        namespace: str = "",
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        client = self._ensure_client()
        from qdrant_client.http.models import Filter as QdrantFilter

        q_filter: QdrantFilter | None = None
        if metadata_filter is not None:
            q_filter = QdrantFilter(**metadata_filter)  # type: ignore[arg-type]
        hits = client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=q_filter,
            limit=top_k,
            with_payload=include_metadata,
        ).points
        return [
            {
                "id": str(p.id),
                "score": p.score,
                "metadata": p.payload if include_metadata else {},
            }
            for p in hits
        ]

    def delete(
        self,
        *,
        ids: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        namespace: str = "",
    ) -> int:
        client = self._ensure_client()
        from qdrant_client.http.models import Filter as QdrantFilter
        from qdrant_client.http.models import FilterSelector

        if ids:
            op_info = client.delete(
                collection_name=self._collection,
                points_selector=ids,
            )
            return op_info.status  # type: ignore[return-value]
        if metadata_filter:
            q_filter = QdrantFilter(**metadata_filter)  # type: ignore[arg-type]
            op_info = client.delete(
                collection_name=self._collection,
                points_selector=FilterSelector(filter=q_filter),
            )
            return op_info.status  # type: ignore[return-value]
        return 0

    def close(self) -> None:
        """Close the Qdrant client connection."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                logger.trace("Qdrant client close ignored")
            self._client = None
