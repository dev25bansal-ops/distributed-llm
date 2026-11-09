export interface ChatCompletionRequest {
    model?: string;
    messages: ChatMessage[];
    temperature?: number;
    top_p?: number;
    max_tokens?: number;
    stream?: boolean;
    response_format?: Record<string, any>;
    adapter?: string;
    tools?: Tool[];
    stop?: string[];
}
export interface ChatMessage {
    role: string;
    content: string;
    name?: string;
    tool_call_id?: string;
}
export interface ChatCompletionResponse {
    id: string;
    model: string;
    created?: number;
    choices: ChatChoice[];
    usage?: UsageInfo;
    generation_time?: number;
}
export interface ChatChoice {
    index?: number;
    message?: ChatMessage;
    finish_reason?: string;
}
export interface CompletionRequest {
    model?: string;
    prompt: string;
    temperature?: number;
    max_tokens?: number;
    stream?: boolean;
}
export interface CompletionResponse {
    id: string;
    model: string;
    choices: CompletionChoice[];
    usage?: UsageInfo;
}
export interface CompletionChoice {
    index?: number;
    text?: string;
    finish_reason?: string;
}
export interface EmbeddingRequest {
    model?: string;
    input: string;
}
export interface EmbeddingResponse {
    model?: string;
    data?: EmbeddingObject[];
    usage?: UsageInfo;
}
export interface EmbeddingObject {
    index?: number;
    embedding?: number[];
}
export interface ModelList {
    data?: ModelInfo[];
}
export interface ModelInfo {
    id?: string;
    owned_by?: string;
    created?: number;
}
export interface HealthResponse {
    status?: string;
    model?: string;
    nodes?: number;
    uptime?: number;
}
export interface BatchSubmitItem {
    request_id?: string;
    prompt: string;
    max_tokens?: number;
    temperature?: number;
}
export interface BatchSubmitRequest {
    items: BatchSubmitItem[];
    webhook_url?: string;
    webhook_token?: string;
    item_timeout?: number;
}
export interface BatchSubmitResponse {
    batch_id: string;
    status: string;
    total_items?: number;
    created_at?: number;
    message?: string;
}
export interface BatchItemResult {
    request_id: string;
    prompt?: string;
    max_tokens?: number;
    temperature?: number;
    status?: 'pending' | 'running' | 'completed' | 'failed' | 'timeout' | 'cancelled';
    result?: Record<string, any>;
    error?: string;
    cost: number;
    started_at?: number;
    completed_at?: number;
}
export interface BatchStatusResponse {
    batch_id: string;
    status: 'pending' | 'running' | 'completed' | 'failed' | 'partially_completed' | 'cancelled';
    total_items: number;
    pending: number;
    running: number;
    completed: number;
    failed: number;
    timed_out: number;
    cancelled: number;
    total_cost: number;
    created_at: number;
    completed_at?: number;
    items?: BatchItemResult[];
}
export interface BatchCancelResponse {
    batch_id: string;
    status: string;
    cancelled_items: number;
    message: string;
}
export interface UsageInfo {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
    cost_usd?: number;
    tokens_per_second?: number;
}
export interface Tool {
    type?: string;
    function?: Record<string, any>;
}
export interface ErrorResponse {
    error?: Record<string, any>;
}
