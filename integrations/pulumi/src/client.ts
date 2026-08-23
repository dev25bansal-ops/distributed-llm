/**
 * HTTP client for the DistLLM API.
 *
 * Provides a thin wrapper around node-fetch with proper error handling,
 * timeout support, and optional API key authentication.
 */

import * as pulumi from "@pulumi/pulumi";

/** Configuration for the DistLLM API client. */
export interface ClientConfig {
  /** Base URL of the DistLLM API (default: http://localhost:8000). */
  readonly endpoint: string;
  /** Optional API key for authenticated endpoints. */
  readonly apiKey?: string;
  /** Request timeout in milliseconds (default: 120000). */
  readonly timeout: number;
}

/** Structured error returned by the DistLLM API. */
export interface ApiError {
  readonly status: number;
  readonly message: string;
  readonly details?: string;
}

/**
 * Low-level HTTP client for the DistLLM REST API.
 *
 * Handles request serialization, response deserialization, auth headers,
 * and non-2xx -> ApiError conversion.
 */
export class Client {
  private readonly config: ClientConfig;

  constructor(config: Partial<ClientConfig> = {}) {
    this.config = {
      endpoint: config.endpoint ?? "http://localhost:8000",
      apiKey: config.apiKey,
      timeout: config.timeout ?? 120_000,
    };
  }

  /** Normalise the endpoint URL (strip trailing slash, add /v1). */
  private baseUrl(): string {
    return `${this.config.endpoint.replace(/\/+$/, "")}/v1`;
  }

  /** Build headers common to every request. */
  private headers(): Record<string, string> {
    const h: Record<string, string> = {
      "Content-Type": "application/json",
      Accept: "application/json",
    };
    if (this.config.apiKey) {
      h["Authorization"] = `Bearer ${this.config.apiKey}`;
    }
    return h;
  }

  /** Perform an HTTP request, converting non-2xx to a rejected promise. */
  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
  ): Promise<T> {
    const url = `${this.baseUrl()}${path}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.config.timeout);

    try {
      const { default: fetch } = await import("node-fetch");
      const res = await fetch(url, {
        method,
        headers: this.headers(),
        body: body !== undefined ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      if (!res.ok) {
        let message = res.statusText;
        let details: string | undefined;
        try {
          const errBody = (await res.json()) as Record<string, unknown>;
          if (typeof errBody.message === "string") message = errBody.message;
          if (typeof errBody.detail === "string") details = errBody.detail;
        } catch {
          // ignore parse errors — fall back to status text
        }
        throw createApiError(res.status, message, details);
      }

      // 204 No Content
      if (res.status === 204) return undefined as unknown as T;

      return (await res.json()) as T;
    } catch (err) {
      if (isApiError(err)) throw err;
      // AbortError / network error
      throw createApiError(
        0,
        err instanceof Error ? err.message : "Unknown request error",
      );
    } finally {
      clearTimeout(timer);
    }
  }

  // ── Public helpers ─────────────────────────────────────────────

  /** GET request. */
  async get<T>(path: string): Promise<T> {
    return this.request<T>("GET", path);
  }

  /** POST request. */
  async post<T>(path: string, body: unknown): Promise<T> {
    return this.request<T>("POST", path, body);
  }

  /** PUT request. */
  async put<T>(path: string, body: unknown): Promise<T> {
    return this.request<T>("PUT", path, body);
  }

  /** DELETE request. */
  async delete<T = void>(path: string): Promise<T> {
    return this.request<T>("DELETE", path);
  }
}

// ── Helpers ─────────────────────────────────────────────────────

function createApiError(
  status: number,
  message: string,
  details?: string,
): Error & { readonly apiError: ApiError } {
  const err = new Error(`DistLLM API error (${status}): ${message}`) as Error &
    ApiError;
  err.status = status;
  err.message = message;
  err.details = details;
  return err as unknown as Error & { readonly apiError: ApiError };
}

function isApiError(err: unknown): err is Error & { readonly status: number } {
  return err instanceof Error && typeof (err as Record<string, unknown>).status === "number";
}
