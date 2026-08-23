"use strict";
// ---------------------------------------------------------------------------
// Models API (typed, OpenAI-compatible /v1/models)
// ---------------------------------------------------------------------------
Object.defineProperty(exports, "__esModule", { value: true });
exports.fetchModels = fetchModels;
/**
 * Fetch the list of available models from a DistLLM API server.
 * Returns an empty array when the endpoint responds with no data.
 */
async function fetchModels(apiUrl) {
    const url = `${apiUrl.replace(/\/+$/, "")}/v1/models`;
    const resp = await fetch(url, { signal: AbortSignal.timeout(10_000) });
    if (!resp.ok) {
        throw new Error(`Failed to list models: HTTP ${resp.status}`);
    }
    const data = (await resp.json());
    if (!data || !Array.isArray(data.data)) {
        throw new Error("Unexpected /v1/models response shape");
    }
    return data.data;
}
//# sourceMappingURL=modelsApi.js.map