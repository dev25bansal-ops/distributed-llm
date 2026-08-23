"""Token counting endpoint — estimate token counts without inference.

Usage::

    POST /v1/tokenize
    {
        "model": "distributed-llm",
        "input": "Hello world"
    }

Response::

    {
        "object": "tokenize",
        "model": "distributed-llm",
        "input_tokens": 2,
        "input": "Hello world"
    }
"""

from __future__ import annotations

import tiktoken
from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

router = APIRouter(tags=["tokenize"])


class TokenizeRequest(BaseModel):
    model: str = Field(default="distributed-llm", description="Model identifier (affects tokenizer)")
    input: str = Field(..., description="Text to tokenize", max_length=131072)


class TokenizeResponse(BaseModel):
    object: str = "tokenize"
    model: str
    input_tokens: int
    input: str


# Cache encoding by model name
_ENCODING_CACHE: dict[str, tiktoken.Encoding] = {}


def _get_encoding(model: str) -> tiktoken.Encoding:
    """Return a tiktoken encoding suitable for *model*."""
    if model in _ENCODING_CACHE:
        return _ENCODING_CACHE[model]

    try:
        if model.startswith("gpt-4") or model.startswith("gpt-3.5"):
            enc = tiktoken.encoding_for_model(model)
        else:
            enc = tiktoken.get_encoding("cl100k_base")
    except (KeyError, ValueError):
        enc = tiktoken.get_encoding("cl100k_base")

    _ENCODING_CACHE[model] = enc
    return enc


@router.post("/v1/tokenize", response_model=TokenizeResponse)
async def tokenize(body: TokenizeRequest):
    """Count tokens for a piece of text without running inference."""
    try:
        enc = _get_encoding(body.model)
        tokens = enc.encode(body.input, disallowed_special=())
        return TokenizeResponse(
            model=body.model,
            input_tokens=len(tokens),
            input=body.input,
        )
    except Exception as exc:
        logger.warning(f"Tokenization failed: {exc}")
        raise HTTPException(status_code=400, detail=f"Tokenization failed: {exc}")
