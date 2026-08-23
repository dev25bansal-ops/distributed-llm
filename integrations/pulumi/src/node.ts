/**
 * Pulumi resource for managing DistLLM cluster nodes.
 *
 * Nodes represent worker instances in the DistLLM cluster that can
 * be provisioned, configured, and decommissioned.
 *
 * ## Usage
 *
 * ```typescript
 * import { DistLLMNode } from "@distllm/pulumi";
 *
 * const node = new DistLLMNode("gpu-node-1", {
 *   nodeId: "node-1",
 *   labels: { gpu: "a100", region: "us-east-1" },
 *   resources: { gpuMemory: 40960, numGpus: 4 },
 * });
 * ```
 */

import * as pulumi from "@pulumi/pulumi";
import { Client } from "./client";
import { buildClientConfig } from "./provider";

/** Input arguments for registering a DistLLM node. */
export interface NodeArgs {
  /** Unique node identifier. */
  readonly nodeId: pulumi.Input<string>;
  /** Key-value labels for the node. */
  readonly labels?: pulumi.Input<Record<string, string>>;
  /** Node resource capacity. */
  readonly resources?: pulumi.Input<NodeResources>;
}

/** Node resource specification. */
export interface NodeResources {
  /** Available GPU memory in MB. */
  readonly gpuMemory?: number;
  /** Number of GPUs available. */
  readonly numGpus?: number;
  /** CPU cores available. */
  readonly cpuCores?: number;
  /** System memory in MB. */
  readonly systemMemory?: number;
}

/** Output state for a DistLLM node. */
export interface NodeState {
  /** Node identifier. */
  readonly nodeId: string;
  /** Node labels. */
  readonly labels?: Record<string, string>;
  /** Resource capacity. */
  readonly resources?: NodeResources;
  /** Node status (online, offline, draining). */
  readonly status: string;
  /** Timestamp when the node joined the cluster. */
  readonly joinedAt?: string;
}

/**
 * Dynamic resource provider for DistLLM nodes.
 */
class NodeResourceProvider implements pulumi.dynamic.ResourceProvider {
  private readonly client: Client;

  constructor(providerConfig: Record<string, unknown>) {
    this.client = new Client(buildClientConfig(providerConfig));
  }

  async create(inputs: NodeArgs): Promise<pulumi.dynamic.CreateResult> {
    const body: Record<string, unknown> = {
      node_id: inputs.nodeId,
    };
    if (inputs.labels) body.labels = inputs.labels;
    if (inputs.resources) body.resources = inputs.resources;

    const result = await this.client.post<NodeState>("/nodes", body);

    return {
      id: result.nodeId,
      outs: result,
    };
  }

  async read(
    id: string,
    _props: NodeState,
  ): Promise<pulumi.dynamic.ReadResult> {
    try {
      const node = await this.client.get<NodeState>(
        `/nodes/${encodeURIComponent(id)}`,
      );
      return { id, props: node };
    } catch {
      return { id, props: { nodeId: id, status: "not_found" } };
    }
  }

  async update(
    id: string,
    _oldInputs: NodeArgs,
    newInputs: NodeArgs,
  ): Promise<pulumi.dynamic.UpdateResult> {
    const body: Record<string, unknown> = {};
    if (newInputs.labels) body.labels = newInputs.labels;
    if (newInputs.resources) body.resources = newInputs.resources;

    const result = await this.client.put<NodeState>(
      `/nodes/${encodeURIComponent(id)}`,
      body,
    );

    return { outs: result };
  }

  async delete(id: string): Promise<void> {
    try {
      await this.client.delete(`/nodes/${encodeURIComponent(id)}`);
    } catch {
      // 404 is acceptable
    }
  }
}

/**
 * A DistLLM cluster node resource.
 *
 * Nodes are worker instances that host model shards and serve
 * inference requests.
 */
export class DistLLMNode extends pulumi.dynamic.Resource {
  /** @internal */
  public readonly nodeId: pulumi.Output<string>;
  /** Node labels. */
  public readonly labels: pulumi.Output<Record<string, string> | undefined>;
  /** Node resource capacity. */
  public readonly resources: pulumi.Output<NodeResources | undefined>;
  /** Node status. */
  public readonly status: pulumi.Output<string>;
  /** Join timestamp. */
  public readonly joinedAt: pulumi.Output<string | undefined>;

  constructor(
    name: string,
    args: NodeArgs,
    opts?: pulumi.CustomResourceOptions,
  ) {
    const providerConfig = extractProviderConfig(opts);
    super(
      new NodeResourceProvider(providerConfig),
      `distllm:node:${name}`,
      {
        nodeId: args.nodeId,
        labels: args.labels,
        resources: args.resources,
        status: undefined,
        joinedAt: undefined,
      },
      opts,
    );
    this.nodeId = pulumi.output(this.get("nodeId"));
    this.labels = pulumi.output(this.get("labels"));
    this.resources = pulumi.output(this.get("resources"));
    this.status = pulumi.output(this.get("status"));
    this.joinedAt = pulumi.output(this.get("joinedAt"));
  }
}

function extractProviderConfig(
  opts?: pulumi.CustomResourceOptions,
): Record<string, unknown> {
  if (opts?.provider) {
    const prov = opts.provider as pulumi.ProviderResource & {
      endpoint?: pulumi.Output<string>;
      apiKey?: pulumi.Output<string | undefined>;
    };
    return {
      endpoint: prov.endpoint?.getSync(),
      apiKey: prov.apiKey?.getSync(),
    };
  }
  return {};
}
