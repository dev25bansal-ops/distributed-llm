/**
 * @distllm/pulumi — A Pulumi provider for DistLLM
 *
 * Manage DistLLM clusters, models, deployments, nodes, and federations
 * through Pulumi infrastructure-as-code.
 *
 * ## Quick Start
 *
 * ```typescript
 * import { DistLLMProvider, DistLLMModel, DistLLMDeployment } from "@distllm/pulumi";
 *
 * const provider = new DistLLMProvider("cluster", {
 *   endpoint: "http://localhost:8000",
 * });
 *
 * const model = new DistLLMModel("llama", {
 *   name: "meta-llama/Llama-2-7b",
 * }, { provider });
 *
 * const deployment = new DistLLMDeployment("llama-prod", {
 *   modelName: model.name,
 *   replicas: 2,
 *   gpuMemory: 16384,
 * }, { provider });
 * ```
 *
 * @module @distllm/pulumi
 */

// ── Provider ─────────────────────────────────────────────────────
export { DistLLMProvider } from "./provider";
export type { DistLLMProviderConfig } from "./provider";

// ── Resources ────────────────────────────────────────────────────
export { DistLLMModel } from "./model";
export type { ModelArgs, ModelParams, ModelState } from "./model";

export { DistLLMDeployment } from "./deployment";
export type { DeploymentArgs, DeploymentState } from "./deployment";

export { DistLLMNode } from "./node";
export type { NodeArgs, NodeResources, NodeState } from "./node";

export { DistLLMFederation } from "./federation";
export type { FederationArgs, FederationPeer, FederationConfig, FederationState } from "./federation";

// ── Data Sources ─────────────────────────────────────────────────
export { getClusterStatus } from "./clusterStatus";
export type { ClusterStatus, NodeHealth } from "./clusterStatus";

export { getModels } from "./models";
export type { ModelList } from "./models";

export { getNodes } from "./nodes";
export type { NodeList } from "./nodes";

// ── Client (advanced usage) ──────────────────────────────────────
export { Client } from "./client";
export type { ClientConfig, ApiError } from "./client";
