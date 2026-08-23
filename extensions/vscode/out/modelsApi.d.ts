export interface ModelInfo {
    id: string;
    object?: string;
    created?: number;
    owned_by?: string;
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
export declare function fetchModels(apiUrl: string): Promise<ModelInfo[]>;
