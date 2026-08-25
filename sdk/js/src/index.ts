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
  response_format?: { type: 'text' | 'json_object' | 'json_schema'; schema?: object };
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
  /**
   * Timeout in milliseconds.
   *
   * - Non-streaming requests: bounds the WHOLE exchange (connection,
   *   response headers, and body read).
   * - Streaming (SSE) requests: bounds connection + response headers, then
   *   acts as an IDLE timeout that re-arms between chunks. A stream that
   *   keeps producing chunks may legally run longer than this value; a
   *   stream that goes silent for longer is aborted.
   *
   * @default 120_000
   */
  timeout?: number;
  maxRetries?: number;
  headers?: Record<string, string>;
}

/**
 * Thrown when a request exceeds `timeout`, or a stream exceeds the idle
 * gap between chunks. Distinguishable from network failures via `name`.
 */
export class DistLLMTimeoutError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'DistLLMTimeoutError';
  }
}

/**
 * DistLLM client for Node.js and browsers.
 */
export class DistLLMClient {
  private baseUrl: string;
  private apiKey: string;
  private timeout: number;
  private maxRetries: number;
  private defaultHeaders: Record<string, string>;

  public chat: { completions: ChatCompletions };
  public completions: Completions;
  public embeddings: Embeddings;
  public models: Models;

  constructor(options: ClientOptions = {}) {
    this.baseUrl = (options.baseUrl || 'http://localhost:8000').replace(/\/$/, '');
    this.apiKey = options.apiKey || '';
    this.timeout = options.timeout || 120_000;
    this.maxRetries = options.maxRetries || 3;
    this.defaultHeaders = options.headers || {};

    this.chat = { completions: new ChatCompletions(this) };
    this.completions = new Completions(this);
    this.embeddings = new Embeddings(this);
    this.models = new Models(this);
  }

  /** @internal */
  async request<T>(method: string, path: string, body?: object): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...this.defaultHeaders,
    };
    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    let lastError: Error | null = null;
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      // The timeout bounds the WHOLE exchange (connect + headers + body),
      // not just the pre-header phase.
      const controller = new AbortController();
      const timeoutId = setTimeout(
        () => controller.abort(new DistLLMTimeoutError(`Request to ${path} timed out after ${this.timeout}ms`)),
        this.timeout,
      );

      try {
        const response = await fetch(url, {
          method,
          headers,
          body: body ? JSON.stringify(body) : undefined,
          signal: controller.signal,
        });

        if (!response.ok) {
          const errorBody = await response.json().catch(() => ({}));
          const error = errorBody as ApiError;
          throw new DistLLMApiError(
            error?.error?.message || `HTTP ${response.status}`,
            response.status,
            error?.error?.type || 'api_error',
          );
        }

        return (await response.json()) as T;
      } catch (error) {
        lastError = error as Error;
        if (error instanceof DistLLMApiError && error.statusCode < 500) {
          throw error;
        }
        // Abortions we scheduled ourselves are timeouts, not transient
        // network faults: surface them instead of retrying.
        if (
          error instanceof DistLLMTimeoutError ||
          controller.signal.aborted
        ) {
          throw error instanceof DistLLMTimeoutError
            ? error
            : new DistLLMTimeoutError(`Request to ${path} timed out after ${this.timeout}ms`);
        }
        if (attempt < this.maxRetries) {
          await new Promise(r => setTimeout(r, Math.min(1000 * 2 ** attempt, 30_000)));
        }
      } finally {
        clearTimeout(timeoutId);
      }
    }
    throw lastError;
  }

  /**
   * Streaming SSE request.
   *
   * Timeout semantics differ from {@link DistLLMClient.request}: the
   * configured timeout bounds CONNECTION + RESPONSE HEADERS only. After
   * headers arrive, the deadline is replaced by an IDLE TIMER that re-arms
   * before every chunk read — long-lived generations that keep trickling
   * data are never killed mid-stream, but a silent/dead server cannot hang
   * the consumer forever.
   *
   * @internal
   */
  async *stream(method: string, path: string, body: object): AsyncGenerator<string> {
    const url = `${this.baseUrl}${path}`;
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...this.defaultHeaders,
    };
    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    const controller = new AbortController();
    // Phase 1: bound connection establishment + time-to-first-byte.
    let timerId = setTimeout(
      () => controller.abort(),
      this.timeout,
    );

    try {
      const response = await fetch(url, {
        method,
        headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new DistLLMApiError(`HTTP ${response.status}`, response.status);
      }

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      // Phase 2: swap the whole-stream deadline for an idle timer that
      // re-arms around every read. A productive stream runs indefinitely;
      // silence longer than `timeout` kills it.
      clearTimeout(timerId);
      timerId = setTimeout(
        () => controller.abort(new DistLLMTimeoutError(`Stream idle for over ${this.timeout}ms`)),
        this.timeout,
      );

      while (true) {
        let chunk: Awaited<ReturnType<typeof reader.read>>;
        try {
          chunk = await reader.read();
        } catch (error) {
          if (controller.signal.aborted) {
            throw new DistLLMTimeoutError(
              `Stream idle: no data received for ${this.timeout}ms`,
            );
          }
          throw error;
        }
        const { done, value } = chunk;
        if (done) break;

        // Got bytes — re-arm the idle window for the next gap.
        clearTimeout(timerId);
        timerId = setTimeout(
          () => controller.abort(new DistLLMTimeoutError(`Stream idle for over ${this.timeout}ms`)),
          this.timeout,
        );

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6).trim();
          if (data === '[DONE]') return;
          try {
            const parsed = JSON.parse(data);
            const content = parsed.choices?.[0]?.delta?.content;
            if (content) yield content;
          } catch {}
        }
      }
    } catch (error) {
      // The connect-phase abort carries no reason (older runtimes drop it),
      // so map any post-abort failure to the connect-timeout variant.
      if (
        !(error instanceof DistLLMApiError) &&
        !(error instanceof DistLLMTimeoutError) &&
        controller.signal.aborted
      ) {
        throw new DistLLMTimeoutError(
          `Connection timed out after ${this.timeout}ms waiting for response headers from ${path}`,
        );
      }
      throw error;
    } finally {
      clearTimeout(timerId);
    }
  }
}

class ChatCompletions {
  constructor(private client: DistLLMClient) {}

  async create(request: ChatCompletionRequest): Promise<ChatCompletionResponse> {
    return this.client.request<ChatCompletionResponse>('POST', '/v1/chat/completions', request);
  }

  async *stream(request: Omit<ChatCompletionRequest, 'stream'>): AsyncGenerator<string> {
    yield* this.client.stream('POST', '/v1/chat/completions', { ...request, stream: true });
  }
}

class Completions {
  constructor(private client: DistLLMClient) {}

  async create(request: CompletionRequest): Promise<CompletionResponse> {
    return this.client.request<CompletionResponse>('POST', '/v1/completions', request);
  }
}

class Embeddings {
  constructor(private client: DistLLMClient) {}

  async create(request: EmbeddingRequest): Promise<EmbeddingResponse> {
    return this.client.request<EmbeddingResponse>('POST', '/v1/embeddings', request);
  }
}

class Models {
  constructor(private client: DistLLMClient) {}

  async list(): Promise<ModelList> {
    return this.client.request<ModelList>('GET', '/v1/models');
  }
}

export class DistLLMApiError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public errorType: string = 'api_error',
  ) {
    super(message);
    this.name = 'DistLLMApiError';
  }
}
