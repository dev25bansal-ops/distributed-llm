"""Tests for vector store backends.

The three top-level adapters (PGVectorStore, QdrantStore, ChromaStore) are
real fail-closed adapters over the ``VectorDBInterface`` ABC; without their
optional drivers they raise RuntimeError with install guidance.  These tests
verify construction defaults and that the adapters subclass the interface.
"""

from distllm.core.vectorstore.base import VectorDBInterface
from distllm.core.vectorstore.pgvector_store import PGVectorStore
from distllm.core.vectorstore.qdrant_store import QdrantStore
from distllm.core.vectorstore.chroma_store import ChromaStore


class TestVectorStore:
    def test_pgvector_creation(self):
        store = PGVectorStore(
            connection_string="postgresql://localhost/test",
            vector_size=384,
        )
        assert store.vector_size == 384
        assert store.collection_name == "distllm"

    def test_qdrant_creation(self):
        store = QdrantStore(host="localhost", port=6333)
        assert store.host == "localhost"
        assert store.port == 6333
        assert store.collection_name == "distllm"

    def test_chroma_creation(self):
        store = ChromaStore(host="chroma-host", port=8000)
        assert store.host == "chroma-host"
        assert store.port == 8000
        assert store.collection_name == "distllm"

    def test_adapters_subclass_interface(self):
        assert issubclass(PGVectorStore, VectorDBInterface)
        assert issubclass(QdrantStore, VectorDBInterface)
        assert issubclass(ChromaStore, VectorDBInterface)