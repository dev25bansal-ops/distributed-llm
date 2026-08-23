/**
 * Pulumi data source for querying DistLLM cluster status.
 *
 * Returns current cluster health, node count, active models, and
 * aggregate resource utilisation.
 *
 * ## Usage
 *
 * ```typescript
 * import { getClusterStatus } from "@distllm/pulumi";
 *
 * const status = getClusterStatus();
 * export const healthy = status.healthy;
 * export const nodeCount = status.nodeCount;
 * ```
 */

import * as pulumi from "@pulumi/pulumi";
import { Client } from "./client";
import { buildClientConfig } from "./provider";

/** Cluster status information. */
export interface ClusterStatus {
  /** Whether the cluster reports as healthy. */
  readonly healthy: boolean;
  /** Number of registered nodes. */
  readonly nodeCount: number;
  /** Number of loaded models. */
  readonly modelCount: number;
  /** Number of active deployments. */
  readonly deploymentCount: number;
  /** Aggregate GPU memory used (MB). */
  readonly gpuMemoryUsed?: number;
  /** Aggregate GPU memory total (MB). */
  readonly gpuMemoryTotal?: number;
  /** Cluster uptime in seconds. */
  readonly uptimeSeconds?: number;
  /** Cluster version string. */
  readonly version?: string;
  /** Detailed node status breakdown. */
  readonly nodes?: NodeHealth[];
}

/** Health information for an individual node. */
export interface NodeHealth {
  /** Node ID. */
  readonly nodeId: string;
  /** Node status. */
  readonly status: string;
  /** GPU utilisation percentage. */
  readonly gpuUtil?: number;
  /** Memory utilisation percentage. */
  readonly memoryUtil?: number;
}

/**
 * Retrieve the current cluster status.
 *
 * @param opts Optional Pulumi invoke options for provider context.
 * @returns Cluster status data.
 */
export function getClusterStatus(
  opts?: pulumi.InvokeOptions,
): pulumi.Output<ClusterStatus> {
  return pulumi.output(
    pulumi.runtime.invoke<ClusterStatus>(
      "distllm:index:getClusterStatus",
      {},
      opts,
    ),
  );
}

/** Invoke implementation registered by the provider. */
export async function getClusterStatusHandler(
  providerConfig: Record<string, unknown>,
): Promise<ClusterStatus> {
  const client = new Client(buildClientConfig(providerConfig));

  try {
    const result = await client.get<ClusterStatus>("/cluster/status");

    return {
      healthy: result.healthy,
      nodeCount: result.nodeCount,
      modelCount: result.modelCount,
      deploymentCount: result.deploymentCount,
      gpuMemoryUsed: result.gpuMemoryUsed,
      gpuMemoryTotal: result.gpuMemoryTotal,
      uptimeSeconds: result.uptimeSeconds,
      version: result.version,
      nodes: result.nodes,
    };
  } catch {
    return {
      healthy: false,
      nodeCount: 0,
      modelCount: 0,
      deploymentCount: 0,
    };
  }
}
