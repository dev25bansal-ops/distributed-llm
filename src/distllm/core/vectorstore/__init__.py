"""Vector store abstraction layer.

Provides an abstract :class:`VectorDBInterface`, concrete implementations
for Pinecone, Qdrant, Weaviate, and Milvus, a :class:`VectorDBFactory` for
provider-agnostic instantiation, and a :class:`RAGPipeline` for
embed/store/retrieve workflows.

Usage::

    from distllm.core.vectorstore import VectorDBFactory, RAGPipeline

    store = VectorDBFactory.create("qdrant", {"url": "http://localhost:6333"})
    pipe = RAGPipeline(embedder=my_embed_fn, vector_store=store)
    vectors = pipe.embed(["hello world"])
    pipe.store(vectors, [{"id": "doc1"}])
    results = pipe.retrieve("hello")
    store.close()
"""

from __future__ import annotations

from distllm.core.vectorstore.base import VectorDBInterface
from distllm.core.vectorstore.factory import VectorDBFactory
from distllm.core.vectorstore.rag import RAGPipeline

__all__ = [
    "VectorDBInterface",
    "VectorDBFactory",
    "RAGPipeline",
]
