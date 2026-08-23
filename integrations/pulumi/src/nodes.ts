/**
 * Pulumi data source for listing DistLLM cluster nodes.
 *
 * Returns all worker nodes registered with the cluster, including
 * their resource capacity and current status.
 *
 * ## Usage
 *
 * ```typescript
 * import { getNodes } from "@distllm/pulumi";
 *
 * const nodes = getNodes();
 * nodes.apply(ns => ns.forEach(n => {
 *   console.log(`${n.nodeId}: ${n.status} (${n.resources?.gpuMemory}MB)`);
 * }));
 * ```
 */

import * as pulumi from "@pulumi/pulumi";
import { Client } from "./client";
import { buildClientConfig } from "./provider";
import type { NodeResources, NodeState } from "./node";

/** List of nodes with their status. */
export interface NodeList {
  /** Array of node states. */
  readonly nodes: NodeState[];
}

/**
 * List all nodes in the DistLLM cluster.
 *
 * @param opts Optional Pulumi invoke options for provider context.
 * @returns Output containing the list of cluster nodes.
 */
export function getNodes(
  opts?: pulumi.InvokeOptions,
): pulumi.Output<NodeState[]> {
  return pulumi.output(
    pulumi.runtime.invoke<NodeList>(
      "distllm:index:getNodes",
      {},
      opts,
    ).apply((result) => result.nodes),
  );
}

/** Invoke implementation for the nodes data source. */
export async function getNodesHandler(
  providerConfig: Record<string, unknown>,
): Promise<NodeList> {
  const client = new Client(buildClientConfig(providerConfig));

  try {
    const nodes = await client.get<NodeState[]>("/nodes");
    return { nodes };
  } catch {
    return { nodes: [] };
  }
}
