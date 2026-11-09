// Auto-generated API endpoint constants.
export const ENDPOINTS = {
    "createChatCompletion": { method: "post", path: "/v1/chat/completions" },
    "createCompletion": { method: "post", path: "/v1/completions" },
    "createEmbedding": { method: "post", path: "/v1/embeddings" },
    "listModels": { method: "get", path: "/v1/models" },
    "getHealth": { method: "get", path: "/health" },
    "createBatch": { method: "post", path: "/v1/batch/submit" },
    "getBatchStatus": { method: "get", path: "/v1/batch/{batch_id}/status" },
    "streamBatchResults": { method: "get", path: "/v1/batch/{batch_id}/stream" },
    "cancelBatch": { method: "post", path: "/v1/batch/{batch_id}/cancel" },
};
