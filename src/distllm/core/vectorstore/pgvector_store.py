"""pgvector (PostgreSQL) vector store adapter.

RAG is external: point this at your own Postgres+pgvector instance.  This is
a **real adapter** implementing :class:`~distllm.core.vectorstore.base.VectorStore`
against ``psycopg2`` (optional dependency).  The module imports without the
driver; methods fail closed with install guidance and never fabricate a
connection.

Requires the ``vector`` extension on the target database::

    CREATE EXTENSION IF NOT EXISTS vector;
"""

from __future__ import annotations

from typing import Any

# Legacy adapter (old interface, not registered in VectorDBFactory, kept for
# back-compat).  The current ABC is VectorDBInterface; this alias keeps the
# module importable.  Deletion/repointing is tracked as audit finding B21.
from distllm.core.vectorstore.base import VectorDBInterface as VectorStore


class PGVectorStore(VectorStore):
    def __init__(self, connection_string: str = "", collection_name: str = "distllm", vector_size: int = 768):
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._conn = None

    def _require_conn(self):
        if self._conn is not None:
            return self._conn
        try:
            import psycopg2  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "psycopg2 is not installed. Install it to use PGVectorStore: "
                "pip install psycopg2-binary  (and enable the 'vector' extension)"
            ) from exc
        if not self.connection_string:
            raise RuntimeError("PGVectorStore requires a connection_string.")
        self._conn = psycopg2.connect(self.connection_string)
        return self._conn

    def _ensure_table(self) -> None:
        conn = self._require_conn()
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self._table()} ("
                "  id TEXT PRIMARY KEY,"
                f"  embedding vector({self.vector_size}),"
                "  metadata JSONB"
                ");"
            )
        conn.commit()

    def _table(self) -> str:
        # Quote the collection name to avoid SQL injection via the identifier.
        safe = "".join(ch for ch in self.collection_name if ch.isalnum() or ch == "_")
        return f'"{safe}"'

    def upsert(self, embeddings: list[list[float]], ids: list[str], metadata: list[dict] | None = None) -> None:
        """Insert/replace ``embeddings`` for ``ids`` (+ optional ``metadata``)."""
        import json

        conn = self._require_conn()
        self._ensure_table()
        with conn.cursor() as cur:
            for i, doc_id in enumerate(ids):
                vec = "[" + ",".join(str(float(x)) for x in embeddings[i]) + "]"
                meta = json.dumps(metadata[i]) if metadata else None
                cur.execute(
                    f"INSERT INTO {self._table()} (id, embedding, metadata) "
                    "VALUES (%s, %s::vector, %s) "
                    "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata;",
                    (doc_id, vec, meta),
                )
        conn.commit()

    def query(self, embedding: list[float], top_k: int = 10) -> list[dict]:
        """Return ``top_k`` nearest neighbours by cosine distance."""
        conn = self._require_conn()
        vec = "[" + ",".join(str(float(x)) for x in embedding) + "]"
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT id, 1 - (embedding <=> %s::vector) AS score, metadata "
                f"FROM {self._table()} ORDER BY embedding <=> %s::vector LIMIT %s;",
                (vec, vec, top_k),
            )
            rows = cur.fetchall()
        return [
            {"id": r[0], "score": float(r[1]), "metadata": r[2]}
            for r in rows
        ]

    def delete(self, ids: list[str]) -> None:
        """Delete ``ids`` from the collection."""
        conn = self._require_conn()
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self._table()} WHERE id = ANY(%s);", (list(ids),)
            )
        conn.commit()

    def list_collections(self) -> list[str]:
        """List tables that look like pgvector collections in this database."""
        conn = self._require_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public';"
            )
            return [r[0] for r in cur.fetchall()]

    def close(self) -> None:
        """Close the underlying DB connection if open."""
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None
