/**
 * DistLLM JavaScript/TypeScript SDK
 *
 * OpenAI-compatible client for distributed LLM inference.
 *
 * Usage:
 *   import { DistLLMClient } from 'distllm-sdk';
 *
 *   const client = new DistLLMClient({ baseUrl: 'http://localhost:8000' });
 *   const response = await client.chat.completions.create({
 *     model: 'distributed-llm',
 *     messages: [{ role: 'user', content: 'Hello!' }],
 *   });
 *   console.log(response.choices[0].message.content);
 */
export interface ChatMessage {
    role: 'system' | 'user' | 'assistant' | 'tool';
    content: string;
    name?: string;
    tool_call_id?: string;
}
export interface ChatCompletionRequest {
    model?: string;
    messages: ChatMessage[];
    temperature?: number;
    top_p?: number;
    max_tokens?: number;
    stream?: boolean;
    response_format?: {
        type: 'text' | 'json_object' | 'json_schema';
        schema?: object;
    };
    adapter?: string;
    tools?: Tool[];
    stop?: string[];
}
export interface ChatCompletionResponse {
    id: string;
    model: string;
    created: number;
    choices: ChatChoice[];
    usage?: UsageInfo;
    generation_time?: number;
}
export interface ChatChoice {
    index: number;
    message?: ChatMessage;
    delta?: Partial<ChatMessage>;
    finish_reason?: 'stop' | 'length' | 'tool_calls';
}
export interface CompletionRequest {
    model?: string;
    prompt: string;
    temperature?: number;
    top_p?: number;
    max_tokens?: number;
    stream?: boolean;
    stop?: string[];
}
export interface CompletionResponse {
    id: string;
    model: string;
    created: number;
    choices: CompletionChoice[];
    usage?: UsageInfo;
}
export interface CompletionChoice {
    index: number;
    text: string;
    finish_reason?: string;
}
export interface EmbeddingRequest {
    model?: string;
    input: string | string[];
}
export interface EmbeddingResponse {
    model: string;
    data: EmbeddingObject[];
    usage?: UsageInfo;
}
export interface EmbeddingObject {
    index: number;
    embedding: number[];
}
export interface ModelList {
    data: ModelInfo[];
}
export interface ModelInfo {
    id: string;
    owned_by: string;
    created: number;
}
export interface HealthResponse {
    status: string;
    model?: string;
    nodes?: number;
    uptime?: number;
}
export interface UsageInfo {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    cost_usd?: number;
    tokens_per_second?: number;
}
export interface Tool {
    type: 'function';
    function: {
        name: string;
        description?: string;
        parameters?: object;
    };
}
export interface ApiError {
    error: {
        message: string;
        type: string;
        code?: string;
    };
}
export interface ClientOptions {
    baseUrl?: string;
    apiKey?: string;
    timeout?: number;
    maxRetries?: number;
    headers?: Record<string, string>;
}
/**
 * DistLLM client for Node.js and browsers.
 */
export declare class DistLLMClient {
    private baseUrl;
    private apiKey;
    private timeout;
    private maxRetries;
    private defaultHeaders;
    chat: {
        completions: ChatCompletions;
    };
    completions: Completions;
    embeddings: Embeddings;
    models: Models;
    constructor(options?: ClientOptions);
    /** @internal */
    request<T>(method: string, path: string, body?: object): Promise<T>;
    /** @internal */
    stream(method: string, path: string, body: object): AsyncGenerator<string>;
}
declare class ChatCompletions {
    private client;
    constructor(client: DistLLMClient);
    create(request: ChatCompletionRequest): Promise<ChatCompletionResponse>;
    stream(request: Omit<ChatCompletionRequest, 'stream'>): AsyncGenerator<string>;
}
declare class Completions {
    private client;
    constructor(client: DistLLMClient);
    create(request: CompletionRequest): Promise<CompletionResponse>;
}
declare class Embeddings {
    private client;
    constructor(client: DistLLMClient);
    create(request: EmbeddingRequest): Promise<EmbeddingResponse>;
}
declare class Models {
    private client;
    constructor(client: DistLLMClient);
    list(): Promise<ModelList>;
}
export declare class DistLLMApiError extends Error {
    statusCode: number;
    errorType: string;
    constructor(message: string, statusCode: number, errorType?: string);
}
export {};
