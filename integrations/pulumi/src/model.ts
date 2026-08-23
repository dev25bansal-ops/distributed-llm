/**
 * Pulumi resource for managing DistLLM models.
 *
 * Models are loaded/unloaded on the DistLLM cluster. Behind the scenes
 * this calls POST/DELETE /v1/models/{name} on the DistLLM API.
 *
 * ## Usage
 *
 * ```typescript
 * import { DistLLMModel } from "@distllm/pulumi";
 *
 * const model = new DistLLMModel("llama", {
 *   name: "meta-llama/Llama-2-7b",
 *   params: {
 *     dtype: "float16",
 *     maxMemory: 16384,
 *   },
 * });
 *
 * export const modelStatus = model.status;
 * ```
 */

import * as pulumi from "@pulumi/pulumi";
import { Client } from "./client";
import { buildClientConfig } from "./provider";

/** Input arguments when creating a DistLLM model. */
export interface ModelArgs {
  /** Fully qualified model name (e.g. "meta-llama/Llama-2-7b"). */
  readonly name: pulumi.Input<string>;
  /** Optional model configuration overrides. */
  readonly params?: pulumi.Input<ModelParams>;
}

/** Model configuration parameters. */
export interface ModelParams {
  /** Data type for model weights (e.g. "float16", "bfloat16", "int8"). */
  readonly dtype?: string;
  /** Maximum GPU memory in MB. */
  readonly maxMemory?: number;
  /** Number of GPUs to use. */
  readonly numGpus?: number;
  /** Model revision/hash. */
  readonly revision?: string;
}

/** Outputs exposed by a DistLLM model resource. */
export interface ModelState {
  /** Model name. */
  readonly name: string;
  /** Current status (loading, loaded, unloading, error). */
  readonly status: string;
  /** Model configuration applied. */
  readonly params?: ModelParams;
  /** Timestamp when the model was loaded. */
  readonly loadedAt?: string;
}

/**
 * Pulumi dynamic resource provider for DistLLM models.
 *
 * Maps Pulumi resource lifecycle (create/read/update/delete) to the
 * DistLLM Models API.
 */
export class ModelResourceProvider implements pulumi.dynamic.ResourceProvider {
  private readonly client: Client;

  constructor(providerConfig: Record<string, unknown>) {
    this.client = new Client(buildClientConfig(providerConfig));
  }

  async create(inputs: ModelArgs): Promise<pulumi.dynamic.CreateResult> {
    const body: Record<string, unknown> = { name: inputs.name };
    if (inputs.params) {
      body.params = inputs.params;
    }

    await this.client.post("/models", body);

    return {
      id: inputs.name as string,
      outs: {
        name: inputs.name,
        status: "loading",
        params: inputs.params ?? {},
      },
    };
  }

  async read(
    id: string,
    _props: ModelState,
  ): Promise<pulumi.dynamic.ReadResult> {
    try {
      const model = await this.client.get<ModelState>(`/models/${encodeURIComponent(id)}`);
      return { id, props: model };
    } catch {
      // Model not found — mark as gone
      return { id, props: { name: id, status: "not_found" } };
    }
  }

  async update(
    id: string,
    _oldInputs: ModelArgs,
    newInputs: ModelArgs,
  ): Promise<pulumi.dynamic.UpdateResult> {
    const body: Record<string, unknown> = { name: newInputs.name };
    if (newInputs.params) {
      body.params = newInputs.params;
    }

    await this.client.put(`/models/${encodeURIComponent(id)}`, body);

    return {
      outs: {
        name: newInputs.name,
        status: "updated",
        params: newInputs.params ?? {},
      },
    };
  }

  async delete(id: string): Promise<void> {
    try {
      await this.client.delete(`/models/${encodeURIComponent(id)}`);
    } catch {
      // 404 is acceptable — already gone
    }
  }
}

/**
 * A DistLLM model resource.
 *
 * Creates and manages the lifecycle of a model on the DistLLM cluster.
 */
export class DistLLMModel extends pulumi.dynamic.Resource {
  /** @internal */
  public readonly name: pulumi.Output<string>;
  /** Current model status. */
  public readonly status: pulumi.Output<string>;
  /** Model configuration. */
  public readonly params: pulumi.Output<ModelParams | undefined>;
  /** Load timestamp. */
  public readonly loadedAt: pulumi.Output<string | undefined>;

  constructor(
    name: string,
    args: ModelArgs,
    opts?: pulumi.CustomResourceOptions,
  ) {
    const providerConfig = extractProviderConfig(opts);
    super(
      new ModelResourceProvider(providerConfig),
      `distllm:model:${name}`,
      {
        name: args.name,
        status: undefined,
        params: args.params,
        loadedAt: undefined,
      },
      opts,
    );
    this.name = pulumi.output(this.get("name"));
    this.status = pulumi.output(this.get("status"));
    this.params = pulumi.output(this.get("params"));
    this.loadedAt = pulumi.output(this.get("loadedAt"));
  }
}

/** Extract the provider config from resource options. */
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
