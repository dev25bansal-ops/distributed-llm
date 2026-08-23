/**
 * Pulumi resource for managing DistLLM federations.
 *
 * Federations connect multiple DistLLM clusters together for
 * cross-cluster inference routing and load balancing.
 *
 * ## Usage
 *
 * ```typescript
 * import { DistLLMFederation } from "@distllm/pulumi";
 *
 * const federation = new DistLLMFederation("global", {
 *   name: "global-federation",
 *   peers: [
 *     { endpoint: "http://cluster-a:8000", weight: 2 },
 *     { endpoint: "http://cluster-b:8000", weight: 1 },
 *   ],
 *   config: {
 *     routingStrategy: "latency-based",
 *     heartbeatInterval: 30,
 *   },
 * });
 * ```
 */

import * as pulumi from "@pulumi/pulumi";
import { Client } from "./client";
import { buildClientConfig } from "./provider";

/** A peer cluster in the federation. */
export interface FederationPeer {
  /** Peer cluster API endpoint. */
  readonly endpoint: string;
  /** Optional routing weight (higher = more traffic). */
  readonly weight?: number;
  /** Optional peer labels. */
  readonly labels?: Record<string, string>;
}

/** Federation configuration options. */
export interface FederationConfig {
  /** Routing strategy (round-robin, latency-based, capacity-based). */
  readonly routingStrategy?: string;
  /** Heartbeat interval in seconds. */
  readonly heartbeatInterval?: number;
  /** Connection timeout in seconds. */
  readonly connectionTimeout?: number;
  /** Maximum number of retries for federated requests. */
  readonly maxRetries?: number;
}

/** Input arguments for creating a federation. */
export interface FederationArgs {
  /** Federation name. */
  readonly name: pulumi.Input<string>;
  /** List of peer clusters. */
  readonly peers: pulumi.Input<pulumi.Input<FederationPeer>[]>;
  /** Federation configuration. */
  readonly config?: pulumi.Input<FederationConfig>;
}

/** Output state for a federation. */
export interface FederationState {
  /** Federation name. */
  readonly name: string;
  /** Registered peers. */
  readonly peers: FederationPeer[];
  /** Federation configuration. */
  readonly config?: FederationConfig;
  /** Federation status (active, degraded, inactive). */
  readonly status: string;
  /** Number of healthy peers. */
  readonly healthyPeers: number;
}

/**
 * Dynamic resource provider for DistLLM federations.
 */
class FederationResourceProvider implements pulumi.dynamic.ResourceProvider {
  private readonly client: Client;

  constructor(providerConfig: Record<string, unknown>) {
    this.client = new Client(buildClientConfig(providerConfig));
  }

  async create(inputs: FederationArgs): Promise<pulumi.dynamic.CreateResult> {
    const body: Record<string, unknown> = {
      name: inputs.name,
      peers: inputs.peers,
    };
    if (inputs.config) body.config = inputs.config;

    const result = await this.client.post<FederationState>("/federation", body);

    return {
      id: result.name,
      outs: result,
    };
  }

  async read(
    id: string,
    _props: FederationState,
  ): Promise<pulumi.dynamic.ReadResult> {
    try {
      const federation = await this.client.get<FederationState>(
        `/federation/${encodeURIComponent(id)}`,
      );
      return { id, props: federation };
    } catch {
      return {
        id,
        props: { name: id, peers: [], status: "not_found", healthyPeers: 0 },
      };
    }
  }

  async update(
    id: string,
    _oldInputs: FederationArgs,
    newInputs: FederationArgs,
  ): Promise<pulumi.dynamic.UpdateResult> {
    const body: Record<string, unknown> = {
      peers: newInputs.peers,
    };
    if (newInputs.config) body.config = newInputs.config;

    const result = await this.client.put<FederationState>(
      `/federation/${encodeURIComponent(id)}`,
      body,
    );

    return { outs: result };
  }

  async delete(id: string): Promise<void> {
    try {
      await this.client.delete(`/federation/${encodeURIComponent(id)}`);
    } catch {
      // 404 is acceptable
    }
  }
}

/**
 * A DistLLM federation resource.
 *
 * Federations connect multiple DistLLM clusters, enabling cross-cluster
 * inference routing and aggregate capacity management.
 */
export class DistLLMFederation extends pulumi.dynamic.Resource {
  /** @internal */
  public readonly name: pulumi.Output<string>;
  /** Registered peers. */
  public readonly peers: pulumi.Output<FederationPeer[]>;
  /** Federation configuration. */
  public readonly config: pulumi.Output<FederationConfig | undefined>;
  /** Current status. */
  public readonly status: pulumi.Output<string>;
  /** Number of healthy peers. */
  public readonly healthyPeers: pulumi.Output<number>;

  constructor(
    name: string,
    args: FederationArgs,
    opts?: pulumi.CustomResourceOptions,
  ) {
    const providerConfig = extractProviderConfig(opts);
    super(
      new FederationResourceProvider(providerConfig),
      `distllm:federation:${name}`,
      {
        name: args.name,
        peers: args.peers,
        config: args.config,
        status: undefined,
        healthyPeers: undefined,
      },
      opts,
    );
    this.name = pulumi.output(this.get("name"));
    this.peers = pulumi.output(this.get("peers"));
    this.config = pulumi.output(this.get("config"));
    this.status = pulumi.output(this.get("status"));
    this.healthyPeers = pulumi.output(this.get("healthyPeers"));
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
