// ---------------------------------------------------------------------------
// Models API (typed, OpenAI-compatible /v1/models)
// ---------------------------------------------------------------------------

export interface ModelInfo {
  id: string;
  object?: string;
  created?: number;
  owned_by?: string;
  // Allow extra OpenAI-compatible fields without losing type safety.
  [key: string]: unknown;
}

export interface ModelsList {
  object?: string;
  data: ModelInfo[];
}

/**
 * Fetch the list of available models from a DistLLM API server.
 * Returns an empty array when the endpoint responds with no data.
 */
export async function fetchModels(apiUrl: string): Promise<ModelInfo[]> {
  const url = `${apiUrl.replace(/\/+$/, "")}/v1/models`;
  const resp = await fetch(url, { signal: AbortSignal.timeout(10_000) });
  if (!resp.ok) {
    throw new Error(`Failed to list models: HTTP ${resp.status}`);
  }
  const data = (await resp.json()) as ModelsList;
  if (!data || !Array.isArray(data.data)) {
    throw new Error("Unexpected /v1/models response shape");
  }
  return data.data;
}
