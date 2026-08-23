"""Pinecone vector store provider.

Requires the ``pinecone`` SDK (``pip install pinecone``).
"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger

from distllm.core.vectorstore.base import VectorDBInterface


class _PineconeStore(VectorDBInterface):
    """Pinecone_ vector store wrapper.

    .. _Pinecone: https://www.pinecone.io
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._api_key: str = config.get("api_key") or os.environ.get(
            "PINECONE_API_KEY", ""
        )
        if not self._api_key:
            raise ValueError(
                "Pinecone API key required — set PINECONE_API_KEY env or "
                "pass api_key in config"
            )
        self._index_name: str = config.get(
            "index", config.get("collection", "default")
        )
        self._environment: str = config.get(
            "environment"
        ) or os.environ.get("PINECONE_ENVIRONMENT", "us-east-1-aws")
        self._host: str | None = config.get("host")
        self._dimension: int = config.get("dimension", 1536)
        self._metric: str = config.get("metric", "cosine")
        self._idx: Any = None  # lazy
        self._pc: Any = None  # Pinecone client instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_index(self) -> Any:
        if self._idx is not None:
            return self._idx
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:
            raise ImportError(
                "Pinecone SDK not installed — run `pip install pinecone`"
            ) from exc

        self._pc = Pinecone(api_key=self._api_key)
        if self._index_name not in self._pc.list_indexes().names():
            self._pc.create_index(
                name=self._index_name,
                dimension=self._dimension,
                metric=self._metric,
                spec=ServerlessSpec(cloud="aws", region=self._environment),
            )
        self._idx = self._pc.Index(self._index_name, host=self._host)
        return self._idx

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
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata must have equal length")
        idx = self._ensure_index()
        records = [
            (meta.get("id", str(i)), vec, {k: v for k, v in meta.items() if k != "id"})
            for i, (vec, meta) in enumerate(zip(vectors, metadata))
        ]
        if batch_size is not None:
            total = 0
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                resp = idx.upsert(vectors=batch, namespace=namespace)
                total += resp.get("upserted_count", len(batch))
            return total
        resp = idx.upsert(vectors=records, namespace=namespace)
        return resp.get("upserted_count", len(records))

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        *,
        metadata_filter: dict[str, Any] | None = None,
        namespace: str = "",
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        idx = self._ensure_index()
        resp = idx.query(
            vector=vector,
            top_k=top_k,
            filter=metadata_filter,
            namespace=namespace,
            include_metadata=include_metadata,
        )
        return [
            {
                "id": m.get("id", ""),
                "score": m.get("score", 0.0),
                "metadata": m.get("metadata", {}),
            }
            for m in resp.get("matches", [])
        ]

    def delete(
        self,
        *,
        ids: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        namespace: str = "",
    ) -> int:
        idx = self._ensure_index()
        resp = idx.delete(ids=ids, filter=metadata_filter, namespace=namespace)
        return resp or 0

    def close(self) -> None:
        """Close the Pinecone client and release resources."""
        if self._pc is not None:
            try:
                self._pc.close()
            except Exception:
                logger.trace("Pinecone client close ignored")
            self._pc = None
            self._idx = None
