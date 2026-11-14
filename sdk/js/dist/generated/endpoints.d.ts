export declare const ENDPOINTS: {
    readonly createChatCompletion: {
        readonly method: "post";
        readonly path: "/v1/chat/completions";
    };
    readonly createCompletion: {
        readonly method: "post";
        readonly path: "/v1/completions";
    };
    readonly createEmbedding: {
        readonly method: "post";
        readonly path: "/v1/embeddings";
    };
    readonly listModels: {
        readonly method: "get";
        readonly path: "/v1/models";
    };
    readonly getHealth: {
        readonly method: "get";
        readonly path: "/health";
    };
    readonly createBatch: {
        readonly method: "post";
        readonly path: "/v1/batch/submit";
    };
    readonly getBatchStatus: {
        readonly method: "get";
        readonly path: "/v1/batch/{batch_id}/status";
    };
    readonly streamBatchResults: {
        readonly method: "get";
        readonly path: "/v1/batch/{batch_id}/stream";
    };
    readonly cancelBatch: {
        readonly method: "post";
        readonly path: "/v1/batch/{batch_id}/cancel";
    };
};
