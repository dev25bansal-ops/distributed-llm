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
    readonly listBatches: {
        readonly method: "post";
        readonly path: "/v1/batches";
    };
};
