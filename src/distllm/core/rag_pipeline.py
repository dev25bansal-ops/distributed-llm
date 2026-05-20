"""RAG (Retrieval-Augmented Generation) pipeline with vector DB integration.

Supports FAISS for local vector search, document ingestion, chunking,
retrieval, and RAG prompt templating.
"""
import json
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


@dataclass
class Document:
    """A document for RAG ingestion."""
    doc_id: str
    content: str
    metadata: dict = field(default_factory=dict)
    chunks: list["DocumentChunk"] = field(default_factory=list)


@dataclass
class DocumentChunk:
    """A chunk of a document."""
    chunk_id: str
    doc_id: str
    content: str
    embedding: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """A single retrieval result."""
    chunk: DocumentChunk
    score: float
    rank: int


class TextChunker:
    """Splits documents into chunks for embedding."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i:i + self.chunk_size]
            chunks.append(" ".join(chunk_words))
            i += self.chunk_size - self.overlap
        return chunks


class RAGPipeline:
    """RAG pipeline with FAISS vector store and embedding integration.
    
    Usage:
        pipeline = RAGPipeline(embedding_fn=my_embedding_fn)
        pipeline.ingest(Document(doc_id="1", content="..."))
        results = pipeline.retrieve("query text", top_k=5)
        prompt = pipeline.build_rag_prompt("original query", results)
    """
    
    def __init__(
        self,
        embedding_fn: callable,
        dimension: int = 768,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        index_path: str | None = None,
    ):
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, using in-memory search")
        
        self._embedding_fn = embedding_fn
        self._dimension = dimension
        self._chunker = TextChunker(chunk_size, chunk_overlap)
        self._index_path = index_path
        
        # Document storage
        self._documents: dict[str, Document] = {}
        self._chunks: dict[str, DocumentChunk] = {}
        
        # FAISS index
        self._index = None
        if FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            logger.info(f"FAISS index created: {dimension}d")
        
        # Chunk embeddings
        self._chunk_embeddings: dict[str, np.ndarray] = {}
    
    def ingest(self, document: Document) -> int:
        """Ingest a document into the RAG pipeline.
        
        Returns:
            Number of chunks created.
        """
        # Chunk the document
        chunk_texts = self._chunker.chunk(document.content)
        chunks = []
        
        for i, chunk_text in enumerate(chunk_texts):
            chunk_id = f"{document.doc_id}:chunk:{i}"
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                content=chunk_text,
                metadata={"chunk_index": i, "total_chunks": len(chunk_texts)},
            )
            chunks.append(chunk)
            self._chunks[chunk_id] = chunk
        
        document.chunks = chunks
        self._documents[document.doc_id] = document
        
        # Compute embeddings
        if chunk_texts:
            embeddings = self._embedding_fn(chunk_texts)
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings, dtype=np.float32)
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
                self._chunk_embeddings[chunk.chunk_id] = embedding
            
            # Add to FAISS index
            if self._index is not None:
                self._index.add(embeddings)
            
            logger.info(f"Ingested document {document.doc_id}: {len(chunks)} chunks")
        
        return len(chunks)
    
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve relevant chunks for a query.
        
        Args:
            query: Query text.
            top_k: Number of results to return.
            
        Returns:
            List of RetrievalResult sorted by relevance.
        """
        # Embed query
        query_embedding = self._embedding_fn([query])
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        results = []
        
        if self._index is not None and FAISS_AVAILABLE:
            # FAISS search
            scores, indices = self._index.search(query_embedding, min(top_k, len(self._chunks)))
            chunk_ids = list(self._chunks.keys())
            
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < 0 or idx >= len(chunk_ids):
                    continue
                chunk_id = chunk_ids[idx]
                chunk = self._chunks.get(chunk_id)
                if chunk:
                    results.append(RetrievalResult(chunk=chunk, score=float(score), rank=i + 1))
        else:
            # In-memory cosine similarity search
            query_vec = query_embedding[0]
            scores = []
            for chunk_id, embedding in self._chunk_embeddings.items():
                norm_q = np.linalg.norm(query_vec)
                norm_e = np.linalg.norm(embedding)
                if norm_q > 0 and norm_e > 0:
                    sim = np.dot(query_vec, embedding) / (norm_q * norm_e)
                    scores.append((chunk_id, sim))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            for i, (chunk_id, score) in enumerate(scores[:top_k]):
                chunk = self._chunks.get(chunk_id)
                if chunk:
                    results.append(RetrievalResult(chunk=chunk, score=float(score), rank=i + 1))
        
        return results
    
    def build_rag_prompt(self, query: str, results: list[RetrievalResult], max_context_tokens: int = 4096) -> str:
        """Build a RAG-enhanced prompt.
        
        Args:
            query: Original query.
            results: Retrieved chunks.
            max_context_tokens: Maximum context length.
            
        Returns:
            RAG-enhanced prompt string.
        """
        context_parts = []
        total_length = 0
        
        for result in results:
            chunk_text = result.chunk.content
            if total_length + len(chunk_text) > max_context_tokens:
                break
            context_parts.append(f"[Document: {result.chunk.doc_id}]\n{chunk_text}")
            total_length += len(chunk_text)
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def save_index(self, path: str | None = None) -> None:
        """Save the FAISS index and chunk data."""
        save_path = path or self._index_path
        if save_path is None:
            return
        
        if self._index is not None and FAISS_AVAILABLE:
            faiss.write_index(self._index, f"{save_path}.faiss")
        
        # Save chunk data
        chunk_data = {
            chunk_id: {
                "doc_id": chunk.doc_id,
                "content": chunk.content,
                "metadata": chunk.metadata,
            }
            for chunk_id, chunk in self._chunks.items()
        }
        with open(f"{save_path}.json", "w") as f:
            json.dump(chunk_data, f)
        
        logger.info(f"RAG index saved to {save_path}")
    
    def stats(self) -> dict:
        """Get pipeline statistics."""
        return {
            "documents": len(self._documents),
            "chunks": len(self._chunks),
            "index_size": self._index.ntotal if self._index else 0,
            "faiss_available": FAISS_AVAILABLE,
        }
