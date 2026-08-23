/**
 * Pulumi dynamic provider for DistLLM.
 *
 * This provider manages the lifecycle of DistLLM resources (models,
 * deployments, nodes, federations) through the DistLLM REST API.
 *
 * Usage:
 * ```typescript
 * import { DistLLMProvider } from "@distllm/pulumi";
 *
 * const provider = new DistLLMProvider("my-cluster", {
 *   endpoint: "http://localhost:8000",
 * });
 *
 * const model = new DistLLMModel("llama", {
 *   name: "meta-llama/Llama-2-7b",
 * }, { provider });
 * ```
 */

import * as pulumi from "@pulumi/pulumi";
import { Client, ClientConfig } from "./client";

/** Provider-level configuration passed to every resource. */
export interface DistLLMProviderConfig {
  /** DistLLM API endpoint (default: http://localhost:8000). */
  readonly endpoint?: string;
  /** Optional API key. */
  readonly apiKey?: string;
  /** Request timeout in seconds (default: 120). */
  readonly timeout?: number;
}

/** Internal shape stored in provider config. */
export interface ProviderInternal {
  readonly client: Client;
}

/**
 * Pulumi dynamic provider for DistLLM.
 *
 * Configure once with endpoint/credentials; all resources created under
 * this provider share the same HTTP client.
 */
export class DistLLMProvider extends pulumi.ProviderResource {
  public readonly endpoint: pulumi.Output<string>;
  public readonly apiKey?: pulumi.Output<string | undefined>;

  constructor(
    name: string,
    args: DistLLMProviderConfig = {},
    opts?: pulumi.ResourceOptions,
  ) {
    const inputs: pulumi.Inputs = {
      endpoint: args.endpoint ?? "http://localhost:8000",
      apiKey: args.apiKey,
      timeout: args.timeout ?? 120,
    };
    super("distllm", name, inputs, opts);
    this.endpoint = pulumi.output(inputs.endpoint);
    this.apiKey = pulumi.output(inputs.apiKey);
  }
}

/**
 * Build a ClientConfig from Pulumi provider configuration.
 * Called by the resource lifecycle handlers.
 */
export function buildClientConfig(
  providerConfig: Record<string, unknown>,
): ClientConfig {
  return {
    endpoint: (providerConfig.endpoint as string) ?? "http://localhost:8000",
    apiKey: providerConfig.apiKey as string | undefined,
    timeout: ((providerConfig.timeout as number) ?? 120) * 1000,
  };
}
