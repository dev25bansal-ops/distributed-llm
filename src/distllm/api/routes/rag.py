"""RAG API routes for vector search and document management."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..api_state import g
from distllm.core.rag_pipeline import Document, RetrievalResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/rag", tags=["rag"])


class IngestRequest(BaseModel):
    document_id: str
    content: str
    metadata: dict = Field(default_factory=dict)


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5


class IngestResponse(BaseModel):
    status: str
    document_id: str
    chunks: int


class RetrieveResponse(BaseModel):
    query: str
    results: list[dict]


class RagStatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    index_size: int


def _get_coordinator():
    """Get the coordinator instance from the app state."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="Coordinator not available")
    return coord


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest document",
    description="Ingest a document into the RAG pipeline. The document is chunked, embedded, and indexed for vector similarity search. Returns the number of chunks created.",
    response_description="Ingestion confirmation with chunk count",
    responses={
        503: {"description": "Coordinator not available or RAG pipeline not initialized"},
    },
)
async def rag_ingest(request: IngestRequest):
    """Ingest a document into the RAG pipeline."""
    coord = _get_coordinator()
    pipeline = getattr(coord, "_rag_pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    doc = Document(
        doc_id=request.document_id,
        content=request.content,
        metadata=request.metadata,
    )
    num_chunks = pipeline.ingest(doc)
    return IngestResponse(status="ok", document_id=request.document_id, chunks=num_chunks)


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Retrieve relevant chunks",
    description="Retrieve the most relevant document chunks for a query from the RAG index. Returns chunks ranked by relevance score with document ID, text content, and rank position.",
    response_description="Ranked list of relevant chunks with scores",
    responses={
        503: {"description": "Coordinator not available or RAG pipeline not initialized"},
    },
)
async def rag_retrieve(request: RetrieveRequest):
    """Retrieve relevant chunks for a query."""
    coord = _get_coordinator()
    pipeline = getattr(coord, "_rag_pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    results = pipeline.retrieve(request.query, top_k=request.top_k)
    return RetrieveResponse(
        query=request.query,
        results=[{"text": r.chunk.content, "score": r.score, "rank": r.rank, "doc_id": r.chunk.doc_id} for r in results],
    )


@router.get(
    "/stats",
    response_model=RagStatsResponse,
    summary="Get RAG pipeline stats",
    description="Return statistics about the RAG pipeline, including total ingested documents, total chunks in the index, and index size in bytes.",
    response_description="RAG pipeline statistics",
    responses={
        503: {"description": "Coordinator not available or RAG pipeline not initialized"},
    },
)
async def rag_stats():
    """Return RAG pipeline statistics."""
    coord = _get_coordinator()
    pipeline = getattr(coord, "_rag_pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    stats = pipeline.stats()
    return RagStatsResponse(
        total_documents=stats.get("documents", 0),
        total_chunks=stats.get("chunks", 0),
        index_size=stats.get("index_size", 0),
    )


@router.post(
    "/save",
    summary="Persist RAG index",
    description="Persist the current RAG index (embeddings and metadata) to disk for durability across restarts. Index can be reloaded on startup.",
    response_description="Save confirmation",
    responses={
        503: {"description": "Coordinator not available or RAG pipeline not initialized"},
    },
)
async def rag_save():
    """Persist the RAG index to disk."""
    coord = _get_coordinator()
    pipeline = getattr(coord, "_rag_pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    pipeline.save_index()
    return {"status": "saved"}


@router.get(
    "/build_rag_prompt",
    summary="Build RAG-enriched prompt",
    description="Build a prompt enriched with retrieved context from the RAG index. Takes a query and base prompt, retrieves the top 5 most relevant chunks, and constructs an augmented prompt with the retrieved context.",
    response_description="RAG-enriched prompt text",
    responses={
        503: {"description": "Coordinator not available or RAG pipeline not initialized"},
    },
)
async def rag_build_prompt(query: str, base_prompt: str):
    """Build a RAG-enriched prompt with retrieved context."""
    coord = _get_coordinator()
    pipeline = getattr(coord, "_rag_pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    results = pipeline.retrieve(query, top_k=5)
    enriched = pipeline.build_rag_prompt(query, results)
    return {"prompt": enriched}
