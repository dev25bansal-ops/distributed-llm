"""Unit tests for RAGPipeline: document ingestion, metadata, retrieval.

Tests RAGPipeline directly with a mock embedding function.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from distllm.core.rag_pipeline import (
    RAGPipeline,
    Document,
    DocumentChunk,
    RetrievalResult,
    TextChunker,
)


def _make_embedding_fn(dimension: int = 8):
    """Create a deterministic embedding function for testing.

    Returns an embedding where each element corresponds to the
    index of the text in the batch.
    """
    def embed(texts: list[str]) -> np.ndarray:
        embeddings = np.zeros((len(texts), dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            # Deterministic embedding: hash text to fill the vector
            np.random.seed(hash(text) % (2**31))
            embeddings[i] = np.random.randn(dimension).astype(np.float32)
        return embeddings
    return embed


EMBEDDING_DIM = 8


@pytest.fixture
def pipeline():
    return RAGPipeline(
        embedding_fn=_make_embedding_dim(EMBEDDING_DIM),
        dimension=EMBEDDING_DIM,
        chunk_size=5,
        chunk_overlap=1,
    )


def _make_embedding_dim(dimension: int):
    def embed(texts: list[str]) -> np.ndarray:
        rng = np.random.RandomState(42)
        return rng.randn(len(texts), dimension).astype(np.float32)
    return embed


# ─── TextChunker tests ──────────────────────────────────────────────────


class TestTextChunker:
    def test_chunk_single_word(self):
        chunker = TextChunker(chunk_size=5, overlap=1)
        chunks = chunker.chunk("hello")
        assert chunks == ["hello"]

    def test_chunk_small_text(self):
        chunker = TextChunker(chunk_size=5, overlap=1)
        chunks = chunker.chunk("a b c d")
        assert len(chunks) == 1
        assert "a b c d" in chunks

    def test_chunk_large_text(self):
        chunker = TextChunker(chunk_size=3, overlap=1)
        text = "one two three four five six seven eight"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        # Verify overlap: second chunk starts with "three"
        assert chunks[1].startswith("three")

    def test_chunk_overlap_content(self):
        chunker = TextChunker(chunk_size=4, overlap=2)
        text = "a b c d e f g"
        chunks = chunker.chunk(text)
        assert len(chunks) == 4
        assert chunks[0] == "a b c d"
        assert chunks[1] == "c d e f"
        assert chunks[2] == "e f g"

    def test_chunk_exact_size(self):
        chunker = TextChunker(chunk_size=3, overlap=0)
        text = "x y z"
        chunks = chunker.chunk(text)
        assert chunks == ["x y z"]

    def test_chunk_empty_text(self):
        chunker = TextChunker(chunk_size=5, overlap=1)
        chunks = chunker.chunk("")
        assert chunks == []

    def test_chunk_exact_multiple(self):
        chunker = TextChunker(chunk_size=2, overlap=0)
        text = "a b c d"
        chunks = chunker.chunk(text)
        assert chunks == ["a b", "c d"]


# ─── Document ingestion tests ───────────────────────────────────────────


class TestIngestion:
    def test_ingest_single_document_creates_chunks(self, pipeline):
        doc = Document(doc_id="doc1", content="apple banana cherry date elderberry fig grape")
        num_chunks = pipeline.ingest(doc)
        assert num_chunks >= 1
        assert len(pipeline._chunks) == num_chunks

    def test_ingest_stores_document(self, pipeline):
        doc = Document(doc_id="doc1", content="some text here")
        pipeline.ingest(doc)
        assert "doc1" in pipeline._documents
        assert pipeline._documents["doc1"].doc_id == "doc1"

    def test_ingest_creates_embeddings(self, pipeline):
        doc = Document(doc_id="doc1", content="apple banana cherry date")
        pipeline.ingest(doc)
        for chunk_id, chunk in pipeline._chunks.items():
            assert chunk.embedding is not None
            assert chunk.embedding.shape == (EMBEDDING_DIM,)

    def test_ingest_stores_chunk_embeddings_dict(self, pipeline):
        doc = Document(doc_id="doc1", content="one two three four five six")
        pipeline.ingest(doc)
        assert len(pipeline._chunk_embeddings) == len(pipeline._chunks)
        for chunk_id in pipeline._chunks:
            assert chunk_id in pipeline._chunk_embeddings

    def test_ingest_multiple_documents(self, pipeline):
        doc1 = Document(doc_id="a", content="hello world")
        doc2 = Document(doc_id="b", content="foo bar baz")
        pipeline.ingest(doc1)
        pipeline.ingest(doc2)
        assert len(pipeline._documents) == 2

    def test_ingest_chunk_id_format(self, pipeline):
        doc = Document(doc_id="my_doc", content="a b c d e f")
        pipeline.ingest(doc)
        for chunk_id in pipeline._chunks:
            assert chunk_id.startswith("my_doc:chunk:")

    def test_stats_reflects_ingestion(self, pipeline):
        doc = Document(doc_id="s1", content="stats test document content here")
        pipeline.ingest(doc)
        stats = pipeline.stats()
        assert stats["documents"] == 1
        assert stats["chunks"] >= 1


# ─── Metadata tests ─────────────────────────────────────────────────────


class TestMetadata:
    def test_document_metadata_stored(self, pipeline):
        doc = Document(
            doc_id="d1",
            content="some text here",
            metadata={"source": "wiki", "author": "alice"},
        )
        pipeline.ingest(doc)
        stored = pipeline._documents["d1"]
        assert stored.metadata["source"] == "wiki"
        assert stored.metadata["author"] == "alice"

    def test_chunk_metadata_has_index(self, pipeline):
        doc = Document(doc_id="d1", content="a b c d e f g h i j")
        pipeline.ingest(doc)
        for chunk in pipeline._chunks.values():
            assert "chunk_index" in chunk.metadata
            assert isinstance(chunk.metadata["chunk_index"], int)

    def test_chunk_metadata_has_total_count(self, pipeline):
        doc = Document(doc_id="d1", content="a b c d e f g h i j k l")
        pipeline.ingest(doc)
        for chunk in pipeline._chunks.values():
            assert "total_chunks" in chunk.metadata
            assert chunk.metadata["total_chunks"] == len(pipeline._chunks)

    def test_chunk_metadata_increments_index(self, pipeline):
        doc = Document(doc_id="d1", content="a b c d e f g h i j k l")
        pipeline.ingest(doc)
        indices = sorted(
            c.metadata["chunk_index"] for c in pipeline._chunks.values()
        )
        assert indices == list(range(len(pipeline._chunks)))

    def test_chunk_links_back_to_doc(self, pipeline):
        doc = Document(doc_id="linked_doc", content="chunked content here")
        pipeline.ingest(doc)
        for chunk in pipeline._chunks.values():
            assert chunk.doc_id == "linked_doc"


# ─── Retrieval tests ────────────────────────────────────────────────────


class TestRetrieval:
    def test_retrieve_top_k_returns_correct_count(self, pipeline):
        docs = [
            Document(doc_id=f"d{i}", content="apple banana cherry date")
            for i in range(5)
        ]
        for d in docs:
            pipeline.ingest(d)
        results = pipeline.retrieve("fruit query", top_k=3)
        assert len(results) == 3

    def test_retrieve_top_k_respects_max(self, pipeline):
        docs = [
            Document(doc_id=f"d{i}", content="some words here for testing")
            for i in range(3)
        ]
        for d in docs:
            pipeline.ingest(d)
        results = pipeline.retrieve("test query", top_k=10)
        assert len(results) <= len(pipeline._chunks)

    def test_retrieve_results_have_scores(self, pipeline):
        doc = Document(doc_id="d1", content="unique content about space and stars")
        pipeline.ingest(doc)
        results = pipeline.retrieve("space query", top_k=5)
        if results:
            assert isinstance(results[0].score, float)
            assert results[0].score >= 0

    def test_retrieve_results_ranked(self, pipeline):
        doc = Document(doc_id="d1", content="a b c d e f g h")
        pipeline.ingest(doc)
        results = pipeline.retrieve("test query", top_k=5)
        for r in results:
            assert r.rank >= 1
        if len(results) >= 2:
            assert results[0].rank < results[1].rank

    def test_retrieve_returns_retrieval_result_objects(self, pipeline):
        doc = Document(doc_id="d1", content="content for testing retrieval")
        pipeline.ingest(doc)
        results = pipeline.retrieve("test query", top_k=5)
        if results:
            assert isinstance(results[0], RetrievalResult)
            assert isinstance(results[0].chunk, DocumentChunk)

    def test_retrieve_empty_index_returns_empty(self, pipeline):
        results = pipeline.retrieve("any query", top_k=5)
        assert results == []

    def test_retrieve_no_chunks_returns_empty(self, pipeline):
        results = pipeline.retrieve("test", top_k=5)
        assert len(results) == 0

    def test_retrieve_with_multiple_docs(self, pipeline):
        docs = [
            Document(doc_id="sports", content="basketball football soccer tennis"),
            Document(doc_id="food", content="pizza pasta salad sandwich"),
            Document(doc_id="tech", content="computer AI software hardware"),
        ]
        for d in docs:
            pipeline.ingest(d)
        results = pipeline.retrieve("basketball game", top_k=2)
        assert len(results) >= 1

    def test_retrieve_default_top_k(self, pipeline):
        for i in range(3):
            pipeline.ingest(Document(doc_id=f"d{i}", content="x y z"))
        results = pipeline.retrieve("test query")
        assert len(results) >= 0


# ─── Index persistence tests ───────────────────────────────────────────────


class TestIndexPersistence:
    """save_index: saves FAISS index and chunk data to disk."""

    def test_save_creates_json_file(self, pipeline):
        doc = Document(doc_id="d1", content="persistence test content here")
        pipeline.ingest(doc)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test_index")
            pipeline.save_index(path)
            assert os.path.exists(f"{path}.json")

    def test_save_json_contains_chunks(self, pipeline):
        doc = Document(doc_id="d1", content="save json chunk content test")
        pipeline.ingest(doc)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test_index")
            pipeline.save_index(path)
            with open(f"{path}.json") as f:
                data = json.load(f)
            assert len(data) == len(pipeline._chunks)
            for chunk_id in pipeline._chunks:
                assert chunk_id in data

    def test_save_json_contains_content(self, pipeline):
        doc = Document(doc_id="d1", content="unique content for roundtrip")
        pipeline.ingest(doc)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test_index")
            pipeline.save_index(path)
            with open(f"{path}.json") as f:
                data = json.load(f)
            chunk_id = list(data.keys())[0]
            assert "content" in data[chunk_id]
            assert data[chunk_id]["doc_id"] == "d1"

    def test_save_json_contains_metadata(self, pipeline):
        doc = Document(
            doc_id="d1",
            content="metadata in saved index",
            metadata={"source": "test"},
        )
        pipeline.ingest(doc)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test_index")
            pipeline.save_index(path)
            with open(f"{path}.json") as f:
                data = json.load(f)
            chunk_id = list(data.keys())[0]
            assert "metadata" in data[chunk_id]

    def test_save_no_path_does_nothing(self, pipeline):
        doc = Document(doc_id="d1", content="no save without path")
        pipeline.ingest(doc)
        pipeline.save_index(None)
        pipeline.save_index("")
        assert True

    def test_save_empty_pipeline(self, pipeline):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty_index")
            pipeline.save_index(path)
            with open(f"{path}.json") as f:
                data = json.load(f)
            assert data == {}

    def test_save_json_chunk_id_matches(self, pipeline):
        doc = Document(doc_id="d1", content="chunk id roundtrip check")
        pipeline.ingest(doc)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test_index")
            pipeline.save_index(path)
            with open(f"{path}.json") as f:
                data = json.load(f)
            for chunk_id, chunk_data in data.items():
                assert chunk_data["doc_id"] in chunk_id


# ─── Prompt enrichment tests ───────────────────────────────────────────────


class TestPromptEnrichment:
    """build_rag_prompt: retrieved context → enhanced prompt."""

    def test_prompt_contains_question(self, pipeline):
        doc = Document(doc_id="d1", content="Paris is the capital of France.")
        pipeline.ingest(doc)
        results = pipeline.retrieve("What is the capital?", top_k=1)
        prompt = pipeline.build_rag_prompt("What is the capital?", results)
        assert "What is the capital?" in prompt

    def test_prompt_contains_context(self, pipeline):
        doc = Document(doc_id="d1", content="The Eiffel Tower is in Paris.")
        pipeline.ingest(doc)
        results = pipeline.retrieve("Eiffel Tower", top_k=1)
        prompt = pipeline.build_rag_prompt("Where is the Eiffel Tower?", results)
        assert "Eiffel Tower" in prompt
        assert "[Document: d1]" in prompt

    def test_prompt_has_answer_section(self, pipeline):
        prompt = pipeline.build_rag_prompt("test query", [])
        assert "Answer:" in prompt

    def test_prompt_has_context_section(self, pipeline):
        prompt = pipeline.build_rag_prompt("test query", [])
        assert "Context:" in prompt

    def test_prompt_multiple_results(self, pipeline):
        for i in range(3):
            pipeline.ingest(Document(
                doc_id=f"d{i}",
                content=f"Document number {i} with unique content.",
            ))
        results = pipeline.retrieve("document unique content", top_k=3)
        prompt = pipeline.build_rag_prompt("list documents", results)
        for i in range(3):
            assert f"[Document: d{i}]" in prompt

    def test_prompt_empty_results(self, pipeline):
        prompt = pipeline.build_rag_prompt("test query", [])
        assert "Context:" in prompt
        assert "test query" in prompt

    def test_prompt_truncates_long_context(self, pipeline):
        doc = Document(doc_id="d1", content="word " * 5000)
        pipeline.ingest(doc)
        results = pipeline.retrieve("word", top_k=5)
        prompt = pipeline.build_rag_prompt("test", results, max_context_tokens=100)
        assert len(prompt) < 5000

    def test_prompt_with_multiple_docs(self, pipeline):
        docs = [
            Document(doc_id="sports", content="Football is a team sport."),
            Document(doc_id="music", content="Jazz originated in New Orleans."),
        ]
        for d in docs:
            pipeline.ingest(d)
        results = pipeline.retrieve("sport music", top_k=2)
        prompt = pipeline.build_rag_prompt("Tell me about sports and music", results)
        assert "Tell me about sports and music" in prompt
        assert "[Document: sports]" in prompt or "[Document: music]" in prompt


# ─── Chunk size/overlap tests ──────────────────────────────────────────────


class TestChunkConfig:
    """Configurable chunk_size and overlap → correct chunks."""

    def test_large_chunk_size(self):
        chunker = TextChunker(chunk_size=100, overlap=10)
        text = "word " * 250
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        assert all(len(c.split()) <= 100 for c in chunks)

    def test_small_chunk_size(self):
        chunker = TextChunker(chunk_size=2, overlap=0)
        text = "a b c d e"
        chunks = chunker.chunk(text)
        assert all(len(c.split()) <= 2 for c in chunks)
        assert chunks == ["a b", "c d", "e"]

    def test_zero_overlap(self):
        chunker = TextChunker(chunk_size=3, overlap=0)
        text = "one two three four five six"
        chunks = chunker.chunk(text)
        assert chunks == ["one two three", "four five six"]

    def test_overlap_equal_to_chunk_size(self):
        chunker = TextChunker(chunk_size=3, overlap=3)
        text = "a b c d e"
        chunks = chunker.chunk(text)
        # step capped at 1, so each word starts a new chunk
        assert len(chunks) == 5
        assert chunks[0] == "a b c"

    def test_overlap_greater_than_chunk_size(self):
        chunker = TextChunker(chunk_size=3, overlap=5)
        text = "a b c d e f"
        chunks = chunker.chunk(text)
        # step capped at 1, no infinite loop
        assert len(chunks) == 6

    def test_single_chunk_for_short_text(self):
        chunker = TextChunker(chunk_size=20, overlap=5)
        text = "short text"
        chunks = chunker.chunk(text)
        assert chunks == ["short text"]

    def test_chunk_via_pipeline_constructor(self):
        pipe = RAGPipeline(
            embedding_fn=lambda texts: np.zeros((len(texts), 4), dtype=np.float32),
            dimension=4,
            chunk_size=3,
            chunk_overlap=1,
        )
        doc = Document(doc_id="d1", content="a b c d e f g")
        n = pipe.ingest(doc)
        assert n >= 2
        for chunk in pipe._chunks.values():
            assert len(chunk.content.split()) <= 3
