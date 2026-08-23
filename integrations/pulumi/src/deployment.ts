/**
 * Pulumi resource for managing DistLLM deployments.
 *
 * Deployments represent a model running with a specific configuration
 * (replicas, GPU memory, etc.) on the DistLLM cluster.
 *
 * ## Usage
 *
 * ```typescript
 * import { DistLLMDeployment } from "@distllm/pulumi";
 *
 * const deployment = new DistLLMDeployment("my-deploy", {
 *   modelName: "meta-llama/Llama-2-7b",
 *   replicas: 2,
 *   gpuMemory: 16384,
 * });
 * ```
 */

import * as pulumi from "@pulumi/pulumi";
import { Client } from "./client";
import { buildClientConfig } from "./provider";

/** Input arguments for creating a DistLLM deployment. */
export interface DeploymentArgs {
  /** Name of the model to deploy. */
  readonly modelName: pulumi.Input<string>;
  /** Number of replicas (default: 1). */
  readonly replicas?: pulumi.Input<number>;
  /** GPU memory per replica in MB. */
  readonly gpuMemory?: pulumi.Input<number>;
  /** Resource labels for the deployment. */
  readonly labels?: pulumi.Input<Record<string, string>>;
  /** Environment variables for the deployment. */
  readonly env?: pulumi.Input<Record<string, string>>;
}

/** Outputs from a DistLLM deployment. */
export interface DeploymentState {
  /** Deployment ID. */
  readonly id: string;
  /** Deployed model name. */
  readonly modelName: string;
  /** Number of replicas. */
  readonly replicas: number;
  /** GPU memory per replica. */
  readonly gpuMemory?: number;
  /** Current status. */
  readonly status: string;
  /** Endpoint URL for inference. */
  readonly endpoint?: string;
  /** Resource labels. */
  readonly labels?: Record<string, string>;
}

/**
 * Dynamic resource provider for DistLLM deployments.
 */
class DeploymentResourceProvider implements pulumi.dynamic.ResourceProvider {
  private readonly client: Client;

  constructor(providerConfig: Record<string, unknown>) {
    this.client = new Client(buildClientConfig(providerConfig));
  }

  async create(inputs: DeploymentArgs): Promise<pulumi.dynamic.CreateResult> {
    const body: Record<string, unknown> = {
      model_name: inputs.modelName,
      replicas: inputs.replicas ?? 1,
    };
    if (inputs.gpuMemory !== undefined) body.gpu_memory = inputs.gpuMemory;
    if (inputs.labels) body.labels = inputs.labels;
    if (inputs.env) body.env = inputs.env;

    const result = await this.client.post<DeploymentState>("/deployments", body);

    return {
      id: result.id,
      outs: result,
    };
  }

  async read(
    id: string,
    _props: DeploymentState,
  ): Promise<pulumi.dynamic.ReadResult> {
    try {
      const dep = await this.client.get<DeploymentState>(
        `/deployments/${encodeURIComponent(id)}`,
      );
      return { id, props: dep };
    } catch {
      return { id, props: { id, modelName: "", replicas: 0, status: "not_found" } };
    }
  }

  async update(
    id: string,
    _oldInputs: DeploymentArgs,
    newInputs: DeploymentArgs,
  ): Promise<pulumi.dynamic.UpdateResult> {
    const body: Record<string, unknown> = {
      model_name: newInputs.modelName,
      replicas: newInputs.replicas ?? 1,
    };
    if (newInputs.gpuMemory !== undefined) body.gpu_memory = newInputs.gpuMemory;
    if (newInputs.labels) body.labels = newInputs.labels;

    const result = await this.client.put<DeploymentState>(
      `/deployments/${encodeURIComponent(id)}`,
      body,
    );

    return { outs: result };
  }

  async delete(id: string): Promise<void> {
    try {
      await this.client.delete(`/deployments/${encodeURIComponent(id)}`);
    } catch {
      // 404 is acceptable
    }
  }
}

/**
 * A DistLLM deployment resource.
 *
 * Deployments make a model available for inference with a specified
 * replica count and resource configuration.
 */
export class DistLLMDeployment extends pulumi.dynamic.Resource {
  /** @internal */
  public readonly modelName: pulumi.Output<string>;
  /** Number of replicas. */
  public readonly replicas: pulumi.Output<number>;
  /** GPU memory per replica. */
  public readonly gpuMemory: pulumi.Output<number | undefined>;
  /** Current status. */
  public readonly status: pulumi.Output<string>;
  /** Inference endpoint URL. */
  public readonly endpoint: pulumi.Output<string | undefined>;

  constructor(
    name: string,
    args: DeploymentArgs,
    opts?: pulumi.CustomResourceOptions,
  ) {
    const providerConfig = extractProviderConfig(opts);
    super(
      new DeploymentResourceProvider(providerConfig),
      `distllm:deployment:${name}`,
      {
        id: undefined,
        modelName: args.modelName,
        replicas: args.replicas ?? 1,
        gpuMemory: args.gpuMemory,
        status: undefined,
        endpoint: undefined,
        labels: args.labels,
      },
      opts,
    );
    this.modelName = pulumi.output(this.get("modelName"));
    this.replicas = pulumi.output(this.get("replicas"));
    this.gpuMemory = pulumi.output(this.get("gpuMemory"));
    this.status = pulumi.output(this.get("status"));
    this.endpoint = pulumi.output(this.get("endpoint"));
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
