"""Embedding and reranking model loader.

Uses AutoModel (not AutoModelForCausalLM) for proper embedding generation
with mean-pooling, normalization, and cross-encoder reranking support.
"""

from typing import List, Optional, Tuple

import torch
from loguru import logger
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification


class EmbeddingModelLoader:
    """Loads and manages embedding/reranking models.

    Supports:
    - Dense embedding models (sentence-transformers, AutoModel)
    - Cross-encoder reranking models (AutoModelForSequenceClassification)
    """

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        rerank_model: Optional[str] = None,
        device: str = "auto",
        dtype: str = "float16",
        trust_remote_code: bool = False,
    ):
        self.embedding_model_name = embedding_model
        self.rerank_model_name = rerank_model
        self.device = device
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code

        self.embedding_model = None
        self.embedding_tokenizer = None
        self.rerank_model = None
        self.rerank_tokenizer = None
        self._embedding_dimension: Optional[int] = None

    def load_embedding_model(self) -> bool:
        """Load the embedding model using AutoModel.

        Returns:
            True if model was loaded successfully.
        """
        if not self.embedding_model_name:
            logger.debug("No embedding model configured")
            return False

        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        device_map = "auto" if self.device == "auto" else self.device

        self.embedding_tokenizer = AutoTokenizer.from_pretrained(
            self.embedding_model_name,
            trust_remote_code=self.trust_remote_code,
        )
        self.embedding_model = AutoModel.from_pretrained(
            self.embedding_model_name,
            torch_dtype=self._torch_dtype(),
            device_map=device_map,
            trust_remote_code=self.trust_remote_code,
        )
        self.embedding_model.eval()

        # Detect embedding dimension from model config
        hidden_size = getattr(self.embedding_model.config, "hidden_size", None)
        if hidden_size:
            self._embedding_dimension = hidden_size

        logger.info(
            f"Embedding model loaded: {self.embedding_model_name} "
            f"(dim={self._embedding_dimension})"
        )
        return True

    def load_rerank_model(self) -> bool:
        """Load the cross-encoder reranking model.

        Returns:
            True if model was loaded successfully.
        """
        if not self.rerank_model_name:
            logger.debug("No rerank model configured")
            return False

        logger.info(f"Loading rerank model: {self.rerank_model_name}")
        device_map = "auto" if self.device == "auto" else self.device

        self.rerank_tokenizer = AutoTokenizer.from_pretrained(
            self.rerank_model_name,
            trust_remote_code=self.trust_remote_code,
        )
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
            self.rerank_model_name,
            torch_dtype=self._torch_dtype(),
            device_map=device_map,
            trust_remote_code=self.trust_remote_code,
        )
        self.rerank_model.eval()

        logger.info(f"Rerank model loaded: {self.rerank_model_name}")
        return True

    def encode(
        self,
        texts: List[str],
        normalize: bool = True,
        max_length: int = 512,
    ) -> torch.Tensor:
        """Encode texts into embedding vectors.

        Args:
            texts: List of input texts.
            normalize: Whether to L2-normalize embeddings.
            max_length: Maximum sequence length.

        Returns:
            [batch_size, hidden_size] tensor of embeddings.
        """
        if self.embedding_model is None or self.embedding_tokenizer is None:
            raise RuntimeError("Embedding model not loaded")

        device = next(self.embedding_model.parameters()).device

        inputs = self.embedding_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            # Mean pooling over last hidden state
            last_hidden = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            masked = last_hidden * attention_mask.unsqueeze(-1)
            embeddings = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)

        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def rerank(
        self,
        query: str,
        documents: List[str],
        batch_size: int = 32,
        max_length: int = 512,
    ) -> List[Tuple[int, float]]:
        """Score query-document pairs using cross-encoder.

        Args:
            query: The query text.
            documents: List of document texts to rank.
            batch_size: Batch size for scoring.
            max_length: Maximum sequence length.

        Returns:
            List of (original_index, score) sorted by score descending.
        """
        if self.rerank_model is None or self.rerank_tokenizer is None:
            raise RuntimeError("Rerank model not loaded")

        device = next(self.rerank_model.parameters()).device
        scores = []

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            pairs = [(query, doc) for doc in batch_docs]

            inputs = self.rerank_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = self.rerank_model(**inputs)
                batch_scores = outputs.logits.squeeze(-1).tolist()

            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]

            for j, score in enumerate(batch_scores):
                scores.append((i + j, float(score)))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    @property
    def embedding_dimension(self) -> Optional[int]:
        return self._embedding_dimension

    def _torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype, torch.float16)
