"""Chroma vector store adapter (RAG is external / bring-your-own vector DB).

This is a **real adapter**, not a stub: it implements the
:class:`~distllm.core.vectorstore.base.VectorStore` interface against the
documented ``chromadb`` client API.  ``chromadb`` is an *optional* dependency
— the module imports without it, but the methods raise a clear, actionable
error telling you to install it.  We never fabricate a connection we can't
verify.

RAG in distllm is intentionally external: run your own Chroma/Qdrant/pgvector
service and point this adapter at it.
"""

from __future__ import annotations

from typing import Any

# Legacy adapter (old interface, not registered in VectorDBFactory, kept for
# back-compat).  The current ABC is VectorDBInterface; this alias keeps the
# module importable.  Deletion/repointing is tracked as audit finding B21.
from distllm.core.vectorstore.base import VectorDBInterface as VectorStore


class ChromaStore(VectorStore):
    def __init__(self, host: str = "localhost", port: int = 8000, collection_name: str = "distllm", tenant: str = "default_tenant", database: str = "default_database"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.tenant = tenant
        self.database = database
        self._client = None
        self._collection = None

    def _require_client(self):
        if self._client is not None:
            return self._client
        try:
            import chromadb  # type: ignore
            from chromadb.config import Settings  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is not installed. Install it to use ChromaStore: "
                "pip install chromadb"
            ) from exc
        self._client = chromadb.HttpClient(
            host=self.host,
            port=self.port,
            tenant=self.tenant,
            database=self.database,
            settings=Settings(anonymized_telemetry=False),
        )
        return self._client

    def _require_collection(self):
        if self._collection is not None:
            return self._collection
        client = self._require_client()
        self._collection = client.get_or_create_collection(name=self.collection_name)
        return self._collection

    def upsert(self, embeddings: list[list[float]], ids: list[str], metadata: list[dict] | None = None) -> None:
        """Insert/replace ``embeddings`` for ``ids`` into the collection."""
        coll = self._require_collection()
        coll.upsert(
            embeddings=embeddings,
            ids=ids,
            metadatas=metadata,
        )

    def query(self, embedding: list[float], top_k: int = 10) -> list[dict]:
        """Return the ``top_k`` nearest neighbours as ``{id, score, metadata}``."""
        coll = self._require_collection()
        res = coll.query(query_embeddings=[embedding], n_results=top_k)
        hits: list[dict] = []
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        for i, doc_id in enumerate(ids):
            hits.append({
                "id": doc_id,
                "score": dists[i] if i < len(dists) else None,
                "metadata": metas[i] if i < len(metas) else None,
            })
        return hits

    def delete(self, ids: list[str]) -> None:
        """Delete ``ids`` from the collection."""
        coll = self._require_collection()
        coll.delete(ids=ids)

    def list_collections(self) -> list[str]:
        """List available collection names."""
        client = self._require_client()
        return [c.name for c in client.list_collections()]

    def close(self) -> None:
        """Release the cached HTTP client (no-op when never connected)."""
        self._client = None
        self._collection = None
