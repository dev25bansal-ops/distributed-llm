/**
 * Pulumi data source for listing DistLLM models.
 *
 * Returns all models registered with the DistLLM cluster, including
 * their current status and configuration.
 *
 * ## Usage
 *
 * ```typescript
 * import { getModels } from "@distllm/pulumi";
 *
 * const models = getModels();
 * models.apply(m => m.forEach(model => {
 *   console.log(`${model.name}: ${model.status}`);
 * }));
 * ```
 */

import * as pulumi from "@pulumi/pulumi";
import { Client } from "./client";
import { buildClientConfig } from "./provider";
import type { ModelParams, ModelState } from "./model";

/** List of models with their status. */
export interface ModelList {
  /** Array of model states. */
  readonly models: ModelState[];
}

/**
 * List all models on the DistLLM cluster.
 *
 * @param opts Optional Pulumi invoke options for provider context.
 * @returns Output containing the list of models.
 */
export function getModels(
  opts?: pulumi.InvokeOptions,
): pulumi.Output<ModelState[]> {
  return pulumi.output(
    pulumi.runtime.invoke<ModelList>(
      "distllm:index:getModels",
      {},
      opts,
    ).apply((result) => result.models),
  );
}

/** Invoke implementation for the models data source. */
export async function getModelsHandler(
  providerConfig: Record<string, unknown>,
): Promise<ModelList> {
  const client = new Client(buildClientConfig(providerConfig));

  try {
    const models = await client.get<ModelState[]>("/models");
    return { models };
  } catch {
    return { models: [] };
  }
}
