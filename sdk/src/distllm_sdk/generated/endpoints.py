"""Auto-generated API endpoint constants."""

ENDPOINTS: dict[str, tuple[str, str]] = {
    "createChatCompletion": ("POST", "/v1/chat/completions"),
    "createCompletion": ("POST", "/v1/completions"),
    "createEmbedding": ("POST", "/v1/embeddings"),
    "listModels": ("GET", "/v1/models"),
    "getHealth": ("GET", "/health"),
    "createBatch": ("POST", "/v1/batches"),
    "listBatches": ("GET", "/v1/batches"),
}
