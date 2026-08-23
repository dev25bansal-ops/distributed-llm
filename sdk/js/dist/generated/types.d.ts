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
export interface BatchRequest {
    input_file_id: string;
    endpoint: string;
    metadata?: Record<string, any>;
}
export interface BatchJob {
    id?: string;
    status?: string;
    input_file_id?: string;
    created_at?: number;
}
export interface BatchList {
    data?: BatchJob[];
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
