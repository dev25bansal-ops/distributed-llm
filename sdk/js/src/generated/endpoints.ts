// Auto-generated API endpoint constants.
export const ENDPOINTS = {
  "createChatCompletion": { method: "post", path: "/v1/chat/completions" },
  "createCompletion": { method: "post", path: "/v1/completions" },
  "createEmbedding": { method: "post", path: "/v1/embeddings" },
  "listModels": { method: "get", path: "/v1/models" },
  "getHealth": { method: "get", path: "/health" },
  // Batches take prompts inline and run in the background; there is no
  // /v1/batches route and no batch-list route on the server.
  "submitBatch": { method: "post", path: "/v1/batch/submit" },
} as const;
