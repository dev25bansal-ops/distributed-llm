"""Unified vector store integration layer for Pinecone, Qdrant, Weaviate, and Milvus.

Provides an abstract interface, optional-provider implementations (SDKs
loaded lazily so only installed providers are required), a factory to
instantiate providers from config/env-vars, and a high-level RAG pipeline
that chains embedding, storage, and retrieval.

Usage::

    # Minimal — env vars must be set
    store = VectorDBFactory.create("pinecone")
    store.upsert(vectors=[[0.1, 0.2, ...]], metadata=[{"id": "doc1"}])

    # Explicit config
    store = VectorDBFactory.create(
        "qdrant",
        {"url": os.environ["QDRANT_URL"], "collection": "my_docs"},
    )

    # RAG pipeline with a custom embedder
    def my_embed(texts: list[str]) -> list[list[float]]: ...
    pipe = RAGPipeline(embedder=my_embed, vector_store=store)
    chunks = pipe.embed(["doc text..."])
    pipe.store(chunks)
    results = pipe.retrieve("user query", top_k=5)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class VectorDBInterface(ABC):
    """Abstract vector store with upsert, query, and delete operations."""

    @abstractmethod
    def upsert(
        self,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
        *,
        namespace: str = "",
    ) -> int:
        """Insert or update vectors with associated metadata.

        Args:
            vectors: Dense vector embeddings, one per record.
            metadata: Parallel list of metadata dicts (must include an
                ``"id"`` key for upsert identity).
            namespace: Optional index partition / namespace.

        Returns:
            Number of records successfully upserted.
        """

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        *,
        filter: dict[str, Any] | None = None,
        namespace: str = "",
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        """Search for the ``top_k`` nearest neighbours of ``vector``.

        Args:
            vector: Query embedding.
            top_k: Number of neighbours to return.
            filter: Metadata filter expression (provider-specific syntax).
            namespace: Optional index partition.
            include_metadata: Whether to attach the stored metadata to hits.

        Returns:
            Each hit is a dict with keys ``"id"``, ``"score"``, and
            (if ``include_metadata``) ``"metadata"``.
        """

    @abstractmethod
    def delete(
        self,
        *,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,
        namespace: str = "",
    ) -> int:
        """Delete vectors by ``ids``, ``filter``, or both (union).

        Args:
            ids: Specific record IDs to delete.
            filter: Metadata filter expression.
            namespace: Optional index partition.

        Returns:
            Number of records deleted.
        """


# ---------------------------------------------------------------------------
# Provider implementations  (SDKs are lazy-imported per method call)
# ---------------------------------------------------------------------------


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

    def _ensure_index(self) -> Any:
        if self._idx is not None:
            return self._idx
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:
            raise ImportError(
                "Pinecone SDK not installed — run `pip install pinecone`"
            ) from exc

        pc = Pinecone(api_key=self._api_key, environment=self._environment)
        if self._index_name not in pc.list_indexes().names():
            pc.create_index(
                name=self._index_name,
                dimension=self._dimension,
                metric=self._metric,
                spec=ServerlessSpec(cloud="aws", region=self._environment),
            )
        self._idx = pc.Index(self._index_name, host=self._host)
        return self._idx

    def upsert(
        self,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
        *,
        namespace: str = "",
    ) -> int:
        idx = self._ensure_index()
        records = [
            (meta.get("id", str(i)), vec, {k: v for k, v in meta.items() if k != "id"})
            for i, (vec, meta) in enumerate(zip(vectors, metadata))
        ]
        resp = idx.upsert(vectors=records, namespace=namespace)
        return resp.get("upserted_count", len(records))

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        *,
        filter: dict[str, Any] | None = None,
        namespace: str = "",
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        idx = self._ensure_index()
        resp = idx.query(
            vector=vector,
            top_k=top_k,
            filter=filter,
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
        filter: dict[str, Any] | None = None,
        namespace: str = "",
    ) -> int:
        idx = self._ensure_index()
        resp = idx.delete(ids=ids, filter=filter, namespace=namespace)
        return resp or 0


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

    def upsert(
        self,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
        *,
        namespace: str = "",
    ) -> int:
        client = self._ensure_client()
        points = [
            self._PointStruct(
                id=meta.get("id", i),
                vector=vec,
                payload={k: v for k, v in meta.items() if k != "id"},
            )
            for i, (vec, meta) in enumerate(zip(vectors, metadata))
        ]
        op_info = client.upsert(
            collection_name=self._collection, points=points
        )
        return op_info.status  # type: ignore[return-value]

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        *,
        filter: dict[str, Any] | None = None,
        namespace: str = "",
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        client = self._ensure_client()
        from qdrant_client.http.models import Filter as QdrantFilter

        q_filter: QdrantFilter | None = None
        if filter is not None:
            q_filter = QdrantFilter(**filter)  # type: ignore[arg-type]
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
        filter: dict[str, Any] | None = None,
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
        if filter:
            q_filter = QdrantFilter(**filter)  # type: ignore[arg-type]
            op_info = client.delete(
                collection_name=self._collection,
                points_selector=FilterSelector(filter=q_filter),
            )
            return op_info.status  # type: ignore[return-value]
        return 0


class _WeaviateStore(VectorDBInterface):
    """Weaviate_ vector store wrapper.

    .. _Weaviate: https://weaviate.io
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._url: str = (
            config.get("url")
            or os.environ.get("WEAVIATE_URL", "http://localhost:8080")
        )
        self._api_key: str | None = config.get("api_key") or os.environ.get(
            "WEAVIATE_API_KEY"
        )
        self._class_name: str = config.get(
            "class", config.get("collection", "Document")
        )
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import weaviate
            from weaviate.classes.config import Configure, Property, DataType
            from weaviate.classes.data import DataObject
        except ImportError as exc:
            raise ImportError(
                "Weaviate SDK not installed — run `pip install weaviate-client`"
            ) from exc

        auth = (
            weaviate.auth.AuthApiKey(api_key=self._api_key)
            if self._api_key
            else None
        )
        self._client = weaviate.connect_to_local(
            host=self._url, auth_credentials=auth
        )
        self._Property = Property
        self._DataType = DataType
        self._Configure = Configure
        self._DataObject = DataObject

        if not self._client.collections.exists(self._class_name):
            self._client.collections.create(
                name=self._class_name,
                vectorizer_config=Configure.Vectorizer.none(),
            )
        return self._client

    def upsert(
        self,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
        *,
        namespace: str = "",
    ) -> int:
        client = self._ensure_client()
        coll = client.collections.get(self._class_name)
        with coll.batch.fixed_size() as batch:
            for vec, meta in zip(vectors, metadata):
                props = {k: v for k, v in meta.items() if k != "id"}
                batch.add_object(
                    properties=props,
                    vector=vec,
                    uuid=meta.get("id"),
                )
        return len(vectors)

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        *,
        filter: dict[str, Any] | None = None,
        namespace: str = "",
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        client = self._ensure_client()
        coll = client.collections.get(self._class_name)

        kwargs: dict[str, Any] = {
            "query_vector": vector,
            "limit": top_k,
            "return_metadata": ["score"],
        }
        if include_metadata:
            kwargs["return_properties"] = True

        resp = coll.query.near_vector(**kwargs)
        return [
            {
                "id": str(o.uuid),
                "score": o.metadata.score if o.metadata else 0.0,
                "metadata": o.properties if include_metadata else {},
            }
            for o in resp.objects
        ]

    def delete(
        self,
        *,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,
        namespace: str = "",
    ) -> int:
        client = self._ensure_client()
        coll = client.collections.get(self._class_name)
        count = 0
        if ids:
            for uuid_str in ids:
                try:
                    coll.data.delete_by_id(uuid_str)
                    count += 1
                except Exception:
                    logger.warning("Weaviate delete miss for id %s", uuid_str)
        if filter:
            # Weaviate BM25 / where filters are complex; log unsupported
            logger.warning(
                "Weaviate filter-based delete not yet implemented via "
                "generic interface — use ids instead"
            )
        return count


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

    def _ensure_collection(self) -> Any:
        if self._collection_obj is not None:
            return self._collection_obj
        try:
            from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
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

    def upsert(
        self,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
        *,
        namespace: str = "",
    ) -> int:
        col = self._ensure_collection()
        ids = [meta.get("id", str(i)) for i, meta in enumerate(metadata)]
        data = [
            ids,
            vectors,
            [
                {k: v for k, v in meta.items() if k != "id"}
                for meta in metadata
            ],
        ]
        mr = col.insert(data)
        col.flush()
        return len(mr.primary_keys)

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        *,
        filter: dict[str, Any] | None = None,
        namespace: str = "",
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        col = self._ensure_collection()
        expr: str | None = None
        if filter:
            # Milvus uses string expressions; pass filter["expr"] or
            # fallback to a simple key==value conjunction
            expr = filter.get("expr")
            if expr is None:
                parts = [
                    f'{k} == "{v}"' if isinstance(v, str) else f"{k} == {v}"
                    for k, v in filter.items()
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
        filter: dict[str, Any] | None = None,
        namespace: str = "",
    ) -> int:
        col = self._ensure_collection()
        expr: str | None = None
        if ids:
            formatted = ", ".join(f'"{i}"' for i in ids)
            expr = f"id in [{formatted}]"
        elif filter:
            expr = filter.get("expr")
            if expr is None:
                parts = [
                    f'{k} == "{v}"' if isinstance(v, str) else f"{k} == {v}"
                    for k, v in filter.items()
                ]
                expr = " and ".join(parts)
        if expr:
            result = col.delete(expr)
            return result.delete_count if hasattr(result, "delete_count") else 0
        return 0


# ---------------------------------------------------------------------------
# Registry — maps provider names (case-insensitive) to implementation classes
# ---------------------------------------------------------------------------

_PROVIDER_REGISTRY: dict[str, type[VectorDBInterface]] = {
    "pinecone": _PineconeStore,
    "qdrant": _QdrantStore,
    "weaviate": _WeaviateStore,
    "milvus": _MilvusStore,
}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class VectorDBFactory:
    """Factory to instantiate a ``VectorDBInterface`` by provider name.

    Configuration is resolved from ``config`` dict with env-var fallbacks.
    """

    @staticmethod
    def create(
        provider: str,
        config: dict[str, Any] | None = None,
    ) -> VectorDBInterface:
        """Create a vector-store client.

        Args:
            provider: One of ``"pinecone"``, ``"qdrant"``, ``"weaviate"``,
                ``"milvus"`` (case-insensitive).
            config: Provider-specific overrides.  Falls back to the
                environment variables documented per-provider.

        Returns:
            A ready-to-use vector-store instance.

        Raises:
            ValueError: Unknown provider name.
        """
        key = provider.lower().strip()
        cls = _PROVIDER_REGISTRY.get(key)
        if cls is None:
            known = ", ".join(sorted(_PROVIDER_REGISTRY))
            raise ValueError(
                f"Unknown vector-db provider {provider!r}. "
                f"Known: {known}"
            )
        return cls(config or {})

    @staticmethod
    def list_providers() -> list[str]:
        """Return sorted list of registered provider names."""
        return sorted(_PROVIDER_REGISTRY)


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------


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
    def store(self) -> VectorDBInterface | None:
        """The backing vector store (may be ``None`` if not yet set)."""
        return self._store

    @store.setter
    def store(self, value: VectorDBInterface) -> None:
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
    ) -> int:
        """Persist embeddings and their metadata into the vector store.

        Args:
            embeddings: Output from :meth:`embed`.
            metadata: Parallel list of metadata dicts (each must contain
                ``"id"``).
            namespace: Optional partition.

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
        return self._store.upsert(embeddings, metadata, namespace=namespace)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        filter: dict[str, Any] | None = None,
        namespace: str = "",
    ) -> list[dict[str, Any]]:
        """Embed ``query`` and search the vector store.

        Args:
            query: Natural-language query string.
            top_k: Number of results.
            filter: Metadata filter (provider-specific).
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
            filter=filter,
            namespace=namespace,
        )

    def delete(
        self,
        *,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,
        namespace: str = "",
    ) -> int:
        """Remove records from the vector store.

        Args:
            ids: Record IDs to remove.
            filter: Metadata filter (provider-specific).
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
        return self._store.delete(ids=ids, filter=filter, namespace=namespace)
