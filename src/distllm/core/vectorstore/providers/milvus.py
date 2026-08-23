"""Milvus / Zilliz Cloud vector store provider.

Requires the ``pymilvus`` SDK (``pip install pymilvus``).
"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger

from distllm.core.vectorstore.base import VectorDBInterface


class _MilvusStore(VectorDBInterface):
    """Milvus_ / Zilliz Cloud vector store wrapper.

    .. _Milvus: https://milvus.io
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._uri: str = (
            config.get("uri")
            or os.environ.get("MILVUS_URI", "http://localhost:19530")
        )
        self._token: str | None = config.get("token") or os.environ.get(
            "MILVUS_TOKEN"
        )
        self._collection: str = config.get(
            "collection", config.get("index", "default")
        )
        self._dimension: int = config.get("dimension", 1536)
        self._collection_obj: Any = None
        self._connections: list[Any] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> Any:
        if self._collection_obj is not None:
            return self._collection_obj
        try:
            from pymilvus import (
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
                connections,
                utility,
            )
        except ImportError as exc:
            raise ImportError(
                "Milvus SDK (pymilvus) not installed — "
                "run `pip install pymilvus`"
            ) from exc

        conn = connections.connect(
            alias="distllm_default",
            uri=self._uri,
            token=self._token,
        )
        self._connections.append(conn)

        if not utility.has_collection(self._collection):
            schema = CollectionSchema(
                [
                    FieldSchema(
                        name="id",
                        dtype=DataType.VARCHAR,
                        is_primary=True,
                        max_length=128,
                    ),
                    FieldSchema(
                        name="vector",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=self._dimension,
                    ),
                    FieldSchema(name="metadata", dtype=DataType.JSON),
                ]
            )
            col = Collection(name=self._collection, schema=schema)
            col.create_index(
                field_name="vector",
                index_params={
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128},
                },
            )
            col.load()
        else:
            col = Collection(name=self._collection)
            col.load()

        self._collection_obj = col
        return col

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
        col = self._ensure_collection()
        ids = [meta.get("id", str(i)) for i, meta in enumerate(metadata)]
        clean_meta = [
            {k: v for k, v in meta.items() if k != "id"} for meta in metadata
        ]

        if batch_size is not None:
            total = 0
            for start in range(0, len(vectors), batch_size):
                end = start + batch_size
                batch_data = [
                    ids[start:end],
                    vectors[start:end],
                    clean_meta[start:end],
                ]
                mr = col.insert(batch_data)
                total += len(mr.primary_keys)
        else:
            data = [ids, vectors, clean_meta]
            mr = col.insert(data)
            total = len(mr.primary_keys)

        col.flush()
        return total

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        *,
        metadata_filter: dict[str, Any] | None = None,
        namespace: str = "",
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        col = self._ensure_collection()
        expr: str | None = None
        if metadata_filter:
            # Milvus uses string expressions; pass filter["expr"] or
            # fallback to a simple key==value conjunction
            expr = metadata_filter.get("expr")
            if expr is None:
                parts = [
                    f'{k} == "{v}"' if isinstance(v, str) else f"{k} == {v}"
                    for k, v in metadata_filter.items()
                    if k != "expr"
                ]
                expr = " and ".join(parts)

        results = col.search(
            data=[vector],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            expr=expr,
            output_fields=["metadata"] if include_metadata else [],
        )
        hits: list[dict[str, Any]] = []
        for result_group in results:
            for hit in result_group:
                entry: dict[str, Any] = {
                    "id": hit.id,
                    "score": hit.score,
                }
                if include_metadata:
                    entry["metadata"] = hit.entity.get("metadata", {})
                hits.append(entry)
        return hits

    def delete(
        self,
        *,
        ids: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        namespace: str = "",
    ) -> int:
        col = self._ensure_collection()
        expr: str | None = None
        if ids:
            formatted = ", ".join(f'"{i}"' for i in ids)
            expr = f"id in [{formatted}]"
        elif metadata_filter:
            expr = metadata_filter.get("expr")
            if expr is None:
                parts = [
                    f'{k} == "{v}"' if isinstance(v, str) else f"{k} == {v}"
                    for k, v in metadata_filter.items()
                ]
                expr = " and ".join(parts)
        if expr:
            result = col.delete(expr)
            return result.delete_count if hasattr(result, "delete_count") else 0
        return 0

    def close(self) -> None:
        """Disconnect all Milvus connections."""
        try:
            from pymilvus import connections
        except ImportError:
            return
        # The provider connects with alias "distllm_default"; disconnect that
        # exact alias (previously a per-instance _i alias was used that never
        # matched, so clients were never released).
        try:
            connections.disconnect(alias="distllm_default")
        except Exception:
            logger.trace("Milvus disconnect ignored")
        self._connections.clear()
        self._collection_obj = None
