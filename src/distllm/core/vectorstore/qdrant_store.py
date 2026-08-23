"""Qdrant vector store adapter (RAG is external / bring-your-own vector DB).

Real adapter implementing :class:`~distllm.core.vectorstore.base.VectorStore`
against the documented ``qdrant_client`` API.  ``qdrant_client`` is an
optional dependency — the module imports without it but methods fail closed
with install guidance.  We never fabricate a connection.
"""

from __future__ import annotations

import uuid
from typing import Any

# Legacy adapter (old interface, not registered in VectorDBFactory, kept for
# back-compat).  The current ABC is VectorDBInterface; this alias keeps the
# module importable.  Deletion/repointing is tracked as audit finding B21.
from distllm.core.vectorstore.base import VectorDBInterface as VectorStore

# Stable namespace for deriving collision-free UUIDs from arbitrary string ids.
_ID_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")


class QdrantStore(VectorStore):
    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "distllm", vector_size: int = 768, api_key: str | None = None):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.api_key = api_key
        self._client = None

    def _require_client(self):
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import QdrantClient  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "qdrant_client is not installed. Install it to use QdrantStore: "
                "pip install qdrant-client"
            ) from exc
        self._client = QdrantClient(host=self.host, port=self.port, api_key=self.api_key)
        return self._client

    def _ensure_collection(self) -> None:
        try:
            from qdrant_client.models import Distance, VectorParams  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "qdrant_client is not installed. Install it to use QdrantStore: "
                "pip install qdrant-client"
            ) from exc

        client = self._require_client()
        if not client.collection_exists(self.collection_name):
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def upsert(self, embeddings: list[list[float]], ids: list[str], metadata: list[dict] | None = None) -> None:
        """Insert/replace ``embeddings`` (``ids`` + optional ``metadata``)."""
        try:
            from qdrant_client.models import PointStruct  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "qdrant_client is not installed. Install it to use QdrantStore: "
                "pip install qdrant-client"
            ) from exc

        client = self._require_client()
        self._ensure_collection()
        points = [
            PointStruct(
                id=_to_point_id(ids[i]),
                vector=embeddings[i],
                payload=metadata[i] if metadata else None,
            )
            for i in range(len(ids))
        ]
        client.upsert(collection_name=self.collection_name, points=points)

    def query(self, embedding: list[float], top_k: int = 10) -> list[dict]:
        """Return the ``top_k`` nearest neighbours as ``{id, score, metadata}``."""
        client = self._require_client()
        res = client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=top_k,
        )
        return [
            {"id": str(h.id), "score": h.score, "metadata": h.payload}
            for h in res
        ]

    def delete(self, ids: list[str]) -> None:
        """Delete ``ids`` from the collection."""
        try:
            from qdrant_client.models import PointIdsList  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "qdrant_client is not installed. Install it to use QdrantStore: "
                "pip install qdrant-client"
            ) from exc

        client = self._require_client()
        client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=[_to_point_id(i) for i in ids]),
        )

    def list_collections(self) -> list[str]:
        """List available collection names."""
        client = self._require_client()
        return [c.name for c in client.get_collections().collections]

    def close(self) -> None:
        """Release the cached Qdrant client (no-op when never connected)."""
        self._client = None


def new_point_id() -> str:
    """Generate a fresh, globally-unique Qdrant point id (native UUID str)."""
    return str(uuid.uuid4())


def _to_point_id(id_value: str | int) -> str:
    """Map an arbitrary id to a native Qdrant UUID string.

    Qdrant accepts either unsigned-int or UUID-string point ids. We use full
    128-bit UUID strings so distinct source ids never collide:

    * an id that is already a valid UUID string is used verbatim;
    * any other string is mapped deterministically via UUID5 (SHA-1 over a
      fixed namespace), preserving idempotent upsert/delete without the
      64-bit truncation that caused collisions;
    * an int id is rendered as a UUID5 of its decimal form for consistency.

    No hashing to 64 bits — the previous ``_to_u64`` truncated SHA-256 to 8
    bytes, shrinking the id space to 2**64 and inviting birthday collisions.
    """
    if isinstance(id_value, int):
        return str(uuid.uuid5(_ID_NAMESPACE, str(id_value)))
    try:
        # Already a valid UUID string -> use as-is (native Qdrant id).
        return str(uuid.UUID(str(id_value)))
    except (ValueError, AttributeError, TypeError):
        return str(uuid.uuid5(_ID_NAMESPACE, str(id_value)))
