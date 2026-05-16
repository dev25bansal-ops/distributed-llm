"""Embedding routes: POST /v1/embeddings."""

import time
import uuid
from typing import List, Optional

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
    input: List[str] = Field(..., description="Input text(s) to embed")
    encoding_format: str = Field(default="float", description="Output format: 'float' or 'base64'")
    dimensions: Optional[int] = Field(default=None, ge=1, description="Number of dimensions for the embedding")
    user: Optional[str] = Field(default=None, description="End-user identifier")


class EmbeddingObject(BaseModel):
    index: int = Field(..., description="Index of the embedding in the input list")
    object: str = "embedding"
    embedding: List[float] = Field(..., description="The embedding vector")


class EmbeddingResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"embed-{uuid.uuid4().hex[:12]}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "distributed-llm"
    data: List[EmbeddingObject]
    usage: dict = Field(default_factory=dict)


@router.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """Create embeddings for input text.

    Generates dense vector embeddings for each input string.
    If no embedding model is loaded, falls back to pooling the last hidden state.
    """
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    if not coord.tokenizer:
        raise HTTPException(status_code=503, detail="Tokenizer not available")

    start_time = time.time()
    embeddings = []
    total_tokens = 0

    for idx, text in enumerate(request.input):
        input_ids = coord.tokenizer.encode(text, return_tensors="pt")
        if hasattr(coord, "local_partitioner") and coord.local_partitioner:
            model = coord.local_partitioner.full_model
            device = next(model.parameters()).device
            input_ids = input_ids.to(device)

            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
                # Use mean pooling over last hidden state
                last_hidden = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else outputs.last_hidden_state
                attention_mask = torch.ones_like(input_ids)
                masked = last_hidden * attention_mask.unsqueeze(-1)
                embedding = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                embeddings.append(embedding[0].tolist())
        else:
            raise HTTPException(
                status_code=503,
                detail="Embedding generation requires a loaded model. Use --local flag or connect to worker nodes.",
            )

        total_tokens += input_ids.shape[-1]

    elapsed = time.time() - start_time

    return EmbeddingResponse(
        model=request.model,
        data=[
            EmbeddingObject(index=i, embedding=emb)
            for i, emb in enumerate(embeddings)
        ],
        usage={
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    )
