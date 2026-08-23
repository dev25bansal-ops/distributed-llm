"""Embedding and reranking service -- encapsulates business logic from routes/embeddings.py.

Usage::

    from distllm.api.services.embedding_service import EmbeddingService

    service = EmbeddingService(coordinator)
    result = await service.embed(input_texts=["hello"], model="distributed-llm")
    ranked = await service.rerank(query="...", documents=["..."])
    hybrid = await service.hybrid_rerank(query="...", documents=["..."])
"""

from __future__ import annotations

import asyncio
import base64
import struct
import time
import uuid
from typing import Any


class EmbeddingService:
    """Encapsulates embedding and reranking business logic.

    The constructor takes a *coordinator* (not importing from ``api_state``).
    Each method maps to a distinct phase of the ``/v1/embeddings`` or
    ``/v1/rerank`` flow.
    """

    def __init__(self, coordinator: Any) -> None:
        self._coordinator = coordinator

    # -- tokenizer (lazy) -----------------------------------------------------------

    def _get_tokenizer(self) -> Any:
        """Lazy access to the coordinator's tokenizer.

        Returns:
            The tokenizer instance.

        Raises:
            RuntimeError: If the tokenizer has not been loaded yet.
        """
        tokenizer = getattr(self._coordinator, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Tokenizer not available")
        return tokenizer

    # -- fallback embedding from generation model ------------------------------------

    def _extract_hidden_state(self, text: str) -> list[float]:
        """Fallback: extract a mean-pooled embedding from the generation model.

        Used when no dedicated embedding model is loaded.  Performs mean
        pooling over the last hidden state of the generation model.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            RuntimeError: If no local model is loaded.
        """
        import torch

        coord = self._coordinator
        partitioner = getattr(coord, "local_partitioner", None)
        if partitioner is None or getattr(partitioner, "full_model", None) is None:
            raise RuntimeError(
                "No local model loaded for fallback embedding generation. "
                "Start with --local or connect worker nodes."
            )

        model = partitioner.full_model
        device = next(model.parameters()).device
        tokenizer = self._get_tokenizer()

        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                last_hidden = outputs.hidden_states[-1]
            else:
                last_hidden = outputs.last_hidden_state

            attention_mask = torch.ones_like(input_ids)
            masked = last_hidden * attention_mask.unsqueeze(-1)
            embedding = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

        return embedding[0].tolist()

    # -- encoding helpers -----------------------------------------------------------

    @staticmethod
    def _encode_base64(values: list[float]) -> str:
        """Encode a list of floats as a base64 string (32-bit LE float).

        Args:
            values: The float vector to encode.

        Returns:
            A base64-encoded ASCII string.
        """
        return base64.b64encode(
            struct.pack(f"{len(values)}f", *values)
        ).decode("ascii")

    # -- embed ----------------------------------------------------------------------

    async def embed(
        self,
        input_texts: list[str],
        model: str = "distributed-llm",
        encoding_format: str = "float",
        dimensions: int | None = None,
        normalize: bool = True,
        user: str | None = None,
    ) -> dict[str, Any]:
        """Generate vector embeddings for a list of input texts.

        Uses a dedicated embedding model (``coord._embedding_loader``) when
        available; otherwise falls back to extracting mean-pooled hidden
        states from the generation model.  Supports L2 normalisation,
        dimension truncation, and ``float`` or ``base64`` encoding formats.

        Args:
            input_texts: List of input strings to embed (max 1024).
            model: Model identifier to return in the response.
            encoding_format: ``"float"`` (list of floats) or
                ``"base64"`` (base64-encoded binary).
            dimensions: If set, truncate embeddings to this many dimensions.
            normalize: Whether to L2-normalise each embedding vector.
            user: Optional end-user identifier (for usage tracking).

        Returns:
            A dict matching the OpenAI ``/v1/embeddings`` response schema::

                {
                    "id": str,
                    "object": "list",
                    "created": int,
                    "model": str,
                    "data": [
                        {"index": int, "object": "embedding", "embedding": list | str},
                        ...
                    ],
                    "usage": {"prompt_tokens": int, "total_tokens": int},
                }

        Raises:
            RuntimeError: If the tokenizer or model is not available.
        """
        coord = self._coordinator
        tokenizer = self._get_tokenizer()
        embed_loader = getattr(coord, "_embedding_loader", None)
        use_dedicated = embed_loader is not None and getattr(
            embed_loader, "embedding_model", None
        ) is not None

        embeddings: list[list[float]] = []
        total_tokens = 0

        if use_dedicated:
            # -- Dedicated embedding model path -----------------------------------------
            max_len = getattr(coord, "_embedding_max_length", 512)
            normalize_emb = normalize and getattr(
                coord, "_embedding_normalize", True
            )

            def _encode_dedicated() -> list[list[float]]:

                emb_tensor = embed_loader.encode(
                    input_texts,
                    normalize=normalize_emb,
                    max_length=max_len,
                )
                if dimensions is not None:
                    emb_tensor = emb_tensor[:, :dimensions]
                return [emb_tensor[i].tolist() for i in range(len(input_texts))]

            embeddings = await asyncio.to_thread(_encode_dedicated)

            for text in input_texts:
                total_tokens += len(tokenizer.encode(text))
        else:
            # -- Fallback: generation model hidden states --------------------------------
            for text in input_texts:
                vec = await asyncio.to_thread(self._extract_hidden_state, text)
                if dimensions is not None:
                    vec = vec[:dimensions]
                if normalize:
                    norm = sum(v * v for v in vec) ** 0.5
                    if norm > 0:
                        vec = [v / norm for v in vec]
                embeddings.append(vec)
                total_tokens += len(tokenizer.encode(text))

        # -- Encode output --------------------------------------------------------------
        if encoding_format == "base64":
            data = [
                {
                    "index": i,
                    "object": "embedding",
                    "embedding": self._encode_base64(emb),
                }
                for i, emb in enumerate(embeddings)
            ]
        else:
            data = [
                {"index": i, "object": "embedding", "embedding": emb}
                for i, emb in enumerate(embeddings)
            ]

        return {
            "id": f"embed-{uuid.uuid4().hex[:12]}",
            "object": "list",
            "created": int(time.time()),
            "model": model,
            "data": data,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        }

    # -- rerank ------------------------------------------------------------------------

    async def rerank(
        self,
        query: str,
        documents: list[str],
        model: str = "distributed-llm",
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Score and reorder documents by relevance to *query*.

        Uses a cross-encoder reranker loaded through
        ``coord._embedding_loader.rerank_model``.

        Args:
            query: The query text.
            documents: List of document texts to rerank.
            model: Model identifier (for logging / response metadata).
            top_n: If set, return only the top *N* results.

        Returns:
            A list of dicts sorted by relevance score (descending)::

                [
                    {"index": int, "document": str, "relevance_score": float},
                    ...
                ]

        Raises:
            RuntimeError: If the reranker model is not available.
        """
        coord = self._coordinator
        embed_loader = getattr(coord, "_embedding_loader", None)

        if embed_loader is None or getattr(embed_loader, "rerank_model", None) is None:
            raise RuntimeError(
                "Reranking requires a cross-encoder model. "
                "Configure embedding.rerank_model."
            )

        batch_size = getattr(coord, "_embedding_batch_size", 32)
        max_length = getattr(coord, "_embedding_max_length", 512)

        def _score() -> list[tuple[int, float]]:
            return embed_loader.rerank(
                query=query,
                documents=documents,
                batch_size=batch_size,
                max_length=max_length,
            )

        scores = await asyncio.to_thread(_score)

        if top_n is not None:
            scores = scores[:top_n]

        return [
            {
                "index": idx,
                "document": documents[idx],
                "relevance_score": score,
            }
            for idx, score in scores
        ]

    # -- hybrid rerank -----------------------------------------------------------------

    @staticmethod
    def _reciprocal_rank_fusion(
        embedding_scores: list[tuple[int, float]],
        rerank_scores: list[tuple[int, float]],
        k: int = 60,
    ) -> list[tuple[int, float]]:
        """Compute Reciprocal Rank Fusion (RRF) from two ranking lists.

        RRF score for a document = sum(1 / (k + rank_i)) for each list *i*.

        Args:
            embedding_scores: ``[(doc_index, score), ...]`` sorted by
                embedding similarity (descending).
            rerank_scores: ``[(doc_index, score), ...]`` sorted by
                cross-encoder relevance (descending).
            k: RRF constant (default 60).

        Returns:
            ``[(doc_index, rrf_score), ...]`` sorted by RRF score descending.
        """
        rrf_map: dict[int, float] = {}

        for rank, (idx, _score) in enumerate(embedding_scores):
            rrf_map[idx] = rrf_map.get(idx, 0.0) + 1.0 / (k + rank + 1)

        for rank, (idx, _score) in enumerate(rerank_scores):
            rrf_map[idx] = rrf_map.get(idx, 0.0) + 1.0 / (k + rank + 1)

        return sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)

    async def hybrid_rerank(
        self,
        query: str,
        documents: list[str],
        model: str = "distributed-llm",
        top_n: int | None = None,
        rrf_k: int = 60,
    ) -> dict[str, Any]:
        """Combine embedding similarity with cross-encoder scores via RRF.

        When both embedding and reranker models are available, their
        per-document ranks are fused with Reciprocal Rank Fusion.  When
        only one is available it acts as a standard rerank or embedding-
        similarity sort.

        Args:
            query: The query text.
            documents: List of document texts to rerank.
            model: Model identifier (for response metadata).
            top_n: If set, return only the top *N* results.
            rrf_k: RRF constant (default 60).

        Returns:
            A dict with keys::

                {
                    "results": [
                        {"index": int, "document": str, "relevance_score": float},
                        ...
                    ],
                    "usage": {
                        "total_tokens": int,
                        "method": "hybrid_rrf",
                        "rrf_k": int,
                    },
                }

        Raises:
            RuntimeError: If no embedding or reranker model is available.
        """
        coord = self._coordinator
        embed_loader = getattr(coord, "_embedding_loader", None)

        if embed_loader is None:
            raise RuntimeError("Embedding loader not available")

        has_embedding = getattr(embed_loader, "embedding_model", None) is not None
        has_reranker = getattr(embed_loader, "rerank_model", None) is not None

        if not has_embedding and not has_reranker:
            raise RuntimeError(
                "Hybrid reranking requires at least an embedding or reranker model"
            )

        tokenizer = self._get_tokenizer()
        batch_size = getattr(coord, "_embedding_batch_size", 32)
        max_length = getattr(coord, "_embedding_max_length", 512)

        # Total tokens for usage metadata
        total_tokens = sum(
            len(tokenizer.encode(query) + tokenizer.encode(doc))
            for doc in documents
        )

        # -- Embedding similarity (bi-encoder) -----------------------------------------
        def _compute_embedding_scores() -> list[tuple[int, float]]:

            all_texts = [query] + documents
            embeddings = embed_loader.encode(all_texts, normalize=True)
            query_emb = embeddings[0:1]
            doc_embs = embeddings[1:]
            sims = (query_emb @ doc_embs.T).squeeze(0).cpu().tolist()
            return sorted(
                [(i, s) for i, s in enumerate(sims)],
                key=lambda x: x[1],
                reverse=True,
            )

        # -- Cross-encoder reranking scores --------------------------------------------
        def _compute_rerank_scores() -> list[tuple[int, float]]:
            return embed_loader.rerank(
                query=query,
                documents=documents,
                batch_size=batch_size,
                max_length=max_length,
            )

        embedding_scores: list[tuple[int, float]] = []
        rerank_scores: list[tuple[int, float]] = []

        if has_embedding:
            embedding_scores = await asyncio.to_thread(_compute_embedding_scores)
        if has_reranker:
            rerank_scores = await asyncio.to_thread(_compute_rerank_scores)

        # -- Fuse with RRF -------------------------------------------------------------
        if embedding_scores and rerank_scores:
            fused = self._reciprocal_rank_fusion(
                embedding_scores, rerank_scores, k=rrf_k
            )
        elif rerank_scores:
            fused = [(idx, score) for idx, _score in rerank_scores]
        else:
            fused = [(idx, score) for idx, _score in embedding_scores]

        if top_n is not None:
            fused = fused[:top_n]

        results = [
            {
                "index": idx,
                "document": documents[idx],
                "relevance_score": round(score, 6),
            }
            for idx, score in fused
        ]

        return {
            "results": results,
            "usage": {
                "total_tokens": total_tokens,
                "method": "hybrid_rrf",
                "rrf_k": rrf_k,
            },
        }
