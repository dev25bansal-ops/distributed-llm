"""Embedding and reranking routes: POST /v1/embeddings, POST /v1/rerank."""

import base64
import struct
import time
import uuid

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from ..api_state import g


router = APIRouter(tags=["embedding"])


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "model": "distributed-llm",
                "input": ["Hello world", "Test sentence"],
                "encoding_format": "float",
            }]
        }
    )
    model: str = Field(default="distributed-llm", description="Model identifier")
    input: list[str] = Field(..., max_length=1024, description="Input text(s) to embed (max 1024 texts)")
    encoding_format: str = Field(default="float", description="Output format: 'float' or 'base64'")
    dimensions: int | None = Field(default=None, ge=1, description="Number of dimensions for the embedding")
    normalize: bool = Field(default=True, description="Whether to L2-normalize embeddings")
    user: str | None = Field(default=None, description="End-user identifier")


class EmbeddingObject(BaseModel):
    index: int = Field(..., description="Index of the embedding in the input list")
    object: str = "embedding"
    embedding: list[float] = Field(..., description="The embedding vector")


class EmbeddingResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"embed-{uuid.uuid4().hex[:12]}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "distributed-llm"
    data: list[EmbeddingObject]
    usage: dict = Field(default_factory=dict)


class RerankRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "model": "distributed-llm",
                "query": "What is machine learning?",
                "documents": ["ML is a subset of AI.", "The weather is nice today."],
                "top_n": 2,
            }]
        }
    )
    model: str = Field(default="distributed-llm", description="Model identifier")
    query: str = Field(..., description="Query text to rank against")
    documents: list[str] = Field(..., description="List of documents to rerank")
    top_n: int | None = Field(default=None, ge=1, description="Return top N results")


class RerankResult(BaseModel):
    index: int = Field(..., description="Original index of the document")
    document: str = Field(..., description="The document text")
    relevance_score: float = Field(..., description="Relevance score (higher = more relevant)")


class RerankResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"rerank-{uuid.uuid4().hex[:12]}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "distributed-llm"
    results: list[RerankResult]
    usage: dict = Field(default_factory=dict)


class HybridRerankRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "model": "distributed-llm",
                "query": "What is machine learning?",
                "documents": ["ML is a subset of AI.", "The weather is nice today."],
                "top_n": 2,
                "rrf_k": 60,
            }]
        }
    )
    model: str = Field(default="distributed-llm", description="Model identifier")
    query: str = Field(..., description="Query text")
    documents: list[str] = Field(..., description="List of documents to rerank")
    top_n: int | None = Field(default=None, ge=1, description="Return top N results")
    rrf_k: int = Field(default=60, ge=1, description="RRF constant k (default 60)")


def _reciprocal_rank_fusion(
    embedding_scores: list[tuple[int, float]],
    rerank_scores: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Compute Reciprocal Rank Fusion (RRF) scores from two ranking lists.

    RRF score for a document = sum(1 / (k + rank_i)) for each ranking list i.

    Args:
        embedding_scores: [(doc_index, score), ...] sorted by embedding similarity.
        rerank_scores: [(doc_index, score), ...] sorted by cross-encoder relevance.
        k: RRF constant (default 60).

    Returns:
        [(doc_index, rrf_score), ...] sorted by RRF score descending.
    """
    rrf_map: dict[int, float] = {}

    # Embedding ranking scores
    for rank, (idx, _score) in enumerate(embedding_scores):
        rrf_map[idx] = rrf_map.get(idx, 0.0) + 1.0 / (k + rank + 1)

    # Reranking scores
    for rank, (idx, _score) in enumerate(rerank_scores):
        rrf_map[idx] = rrf_map.get(idx, 0.0) + 1.0 / (k + rank + 1)

    return sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)


def _encode_base64(values: list[float]) -> bytes:
    """Encode a list of floats as base64 (32-bit float, little-endian)."""
    return base64.b64encode(struct.pack(f"{len(values)}f", *values))


@router.post(
    "/v1/embeddings",
    summary="Create embeddings",
    description="Generate vector embeddings for input text(s). Uses a dedicated embedding model when available, otherwise falls back to extracting hidden states from the generation model. Supports L2 normalization, dimension truncation, and float or base64 encoding formats.",
    response_description="List of embedding vectors with token usage",
    responses={
        503: {"description": "No model loaded, tokenizer not available, or embedding generation requires a local model"},
    },
)
async def create_embeddings(request: EmbeddingRequest):
    """Create embeddings for input text.

    Uses dedicated embedding model (AutoModel) if loaded, otherwise falls back
    to the generation model's hidden states. Supports normalization, dimension
    truncation, and float/base64 encoding formats.
    """
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    if not coord.tokenizer:
        raise HTTPException(status_code=503, detail="Tokenizer not available")

    start_time = time.time()
    embeddings = []
    total_tokens = 0

    # Check if dedicated embedding model is available
    embed_loader = getattr(coord, "_embedding_loader", None)

    if embed_loader and embed_loader.embedding_model is not None:
        # Use dedicated embedding model
        max_len = getattr(coord, "_embedding_max_length", 512)
        normalize = request.normalize and getattr(coord, "_embedding_normalize", True)
        emb_tensor = embed_loader.encode(request.input, normalize=normalize, max_length=max_len)

        # Dimension truncation
        if request.dimensions is not None:
            emb_tensor = emb_tensor[:, :request.dimensions]

        for idx in range(len(request.input)):
            vec = emb_tensor[idx].tolist()
            embeddings.append(vec)
            total_tokens += len(coord.tokenizer.encode(request.input[idx]))
    else:
        # Fallback: use generation model's hidden states
        if not hasattr(coord, "local_partitioner") or not coord.local_partitioner:
            raise HTTPException(
                status_code=503,
                detail="Embedding generation requires a loaded model. Use --local flag or connect to worker nodes.",
            )

        model = coord.local_partitioner.full_model
        device = next(model.parameters()).device
        normalize = request.normalize

        for idx, text in enumerate(request.input):
            input_ids = coord.tokenizer.encode(text, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else outputs.last_hidden_state
                attention_mask = torch.ones_like(input_ids)
                masked = last_hidden * attention_mask.unsqueeze(-1)
                embedding = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

            vec = embedding[0].tolist()

            # Dimension truncation
            if request.dimensions is not None:
                vec = vec[:request.dimensions]

            # Normalization
            if normalize:
                norm = sum(v * v for v in vec) ** 0.5
                if norm > 0:
                    vec = [v / norm for v in vec]

            embeddings.append(vec)
            total_tokens += input_ids.shape[-1]

    # Encode output
    if request.encoding_format == "base64":
        data_objects = [
            EmbeddingObject(index=i, embedding=_encode_base64(emb))
            for i, emb in enumerate(embeddings)
        ]
    else:
        data_objects = [
            EmbeddingObject(index=i, embedding=emb)
            for i, emb in enumerate(embeddings)
        ]

    elapsed = time.time() - start_time

    return EmbeddingResponse(
        model=request.model,
        data=data_objects,
        usage={
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
            "processing_time": round(elapsed, 3),
        },
    )


@router.post(
    "/v1/rerank",
    summary="Rerank documents",
    description="Score and reorder documents by relevance to a query using a cross-encoder model. Returns documents sorted by relevance score descending, with top_n filtering for efficient RAG pipelines.",
    response_description="Ranked list of documents with relevance scores",
    responses={
        503: {"description": "No model loaded or reranking model not configured"},
    },
)
async def create_rerank(request: RerankRequest):
    """Rerank documents using cross-encoder model.

    Takes a query and list of documents, scores each pair, and returns
    documents sorted by relevance score.
    """
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    # Check if rerank model is available
    embed_loader = getattr(coord, "_embedding_loader", None)
    if embed_loader is None or embed_loader.rerank_model is None:
        raise HTTPException(
            status_code=503,
            detail="Reranking requires a cross-encoder model. Configure embedding.rerank_model.",
        )

    start_time = time.time()

    # Score and rank
    scores = embed_loader.rerank(
        query=request.query,
        documents=request.documents,
        batch_size=getattr(coord, "_embedding_batch_size", 32),
        max_length=getattr(coord, "_embedding_max_length", 512),
    )

    # Apply top_n filter
    if request.top_n is not None:
        scores = scores[:request.top_n]

    results = [
        RerankResult(
            index=idx,
            document=request.documents[idx],
            relevance_score=score,
        )
        for idx, score in scores
    ]

    elapsed = time.time() - start_time

    return RerankResponse(
        model=request.model,
        results=results,
        usage={
            "total_tokens": sum(len(coord.tokenizer.encode(request.query) + coord.tokenizer.encode(doc)) for doc in request.documents),
            "processing_time": round(elapsed, 3),
        },
    )


@router.post(
    "/v1/rerank/hybrid",
    summary="Hybrid rerank with RRF",
    description="Combine embedding-based similarity scores with cross-encoder reranking scores using Reciprocal Rank Fusion (RRF). Provides improved ranking quality in RAG pipelines by leveraging both bi-encoder and cross-encoder signals.",
    response_description="Fused ranked list of documents with RRF scores",
    responses={
        503: {"description": "No model loaded or embedding/reranker models not configured"},
    },
)
async def create_hybrid_rerank(request: HybridRerankRequest):
    """Hybrid reranking using Reciprocal Rank Fusion (RRF).

    Combines embedding-based similarity scores with cross-encoder reranking
    scores using RRF for improved ranking quality in RAG pipelines.
    """
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    embed_loader = getattr(coord, "_embedding_loader", None)
    if embed_loader is None:
        raise HTTPException(status_code=503, detail="Embedding loader not available")

    has_embedding = embed_loader.embedding_model is not None
    has_reranker = embed_loader.rerank_model is not None

    if not has_embedding and not has_reranker:
        raise HTTPException(
            status_code=503,
            detail="Hybrid reranking requires at least an embedding or reranker model",
        )

    start_time = time.time()

    # Get embedding similarity scores (if available)
    embedding_scores: list[tuple[int, float]] = []
    if has_embedding:
        # Encode query and documents
        all_texts = [request.query] + request.documents
        embeddings = embed_loader.encode(all_texts, normalize=True)
        query_emb = embeddings[0:1]
        doc_embs = embeddings[1:]

        # Cosine similarity
        sims = (query_emb @ doc_embs.T).squeeze(0).cpu().tolist()
        embedding_scores = sorted(
            [(i, s) for i, s in enumerate(sims)],
            key=lambda x: x[1],
            reverse=True,
        )

    # Get cross-encoder reranking scores (if available)
    rerank_scores: list[tuple[int, float]] = []
    if has_reranker:
        scores = embed_loader.rerank(
            query=request.query,
            documents=request.documents,
            batch_size=getattr(coord, "_embedding_batch_size", 32),
            max_length=getattr(coord, "_embedding_max_length", 512),
        )
        rerank_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Fuse with RRF
    if embedding_scores and rerank_scores:
        fused = _reciprocal_rank_fusion(embedding_scores, rerank_scores, k=request.rrf_k)
    elif rerank_scores:
        fused = [(idx, score) for idx, score in rerank_scores]
    else:
        fused = [(idx, score) for idx, score in embedding_scores]

    # Apply top_n
    if request.top_n is not None:
        fused = fused[: request.top_n]

    results = [
        RerankResult(
            index=idx,
            document=request.documents[idx],
            relevance_score=round(score, 6),
        )
        for idx, score in fused
    ]

    elapsed = time.time() - start_time

    return RerankResponse(
        model=request.model,
        results=results,
        usage={
            "total_tokens": sum(len(coord.tokenizer.encode(request.query) + coord.tokenizer.encode(doc)) for doc in request.documents),
            "processing_time": round(elapsed, 3),
            "method": "hybrid_rrf",
            "rrf_k": request.rrf_k,
        },
    )
