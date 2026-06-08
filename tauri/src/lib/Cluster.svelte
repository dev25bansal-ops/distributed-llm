<script lang="ts">
  import { createCluster, joinCluster, leaveCluster } from "./api";
  import { clusterStore, logStore } from "./stores";
  import { Card, Button, Input, ErrorBanner, StatusDot, toastStore } from "./ui";
  import type { ClusterStatus } from "./types";
  import { onMount, onDestroy } from "svelte";

  let { initialAction, connectAddr }: { initialAction?: string | null; connectAddr?: string | null } = $props();

  let status = $state<ClusterStatus | null>(null);
  let loading = $state(false);
  let error = $state<string | null>(null);

  // Create form
  let createPort = $state(8000);
  let createModel = $state("");

  // Join form
  let joinHost = $state("127.0.0.1");
  let joinPort = $state(8000);

  // 3.4: Subscribe to shared cluster store instead of own polling
  let unsubscribe: (() => void) | undefined;

  onMount(() => {
    unsubscribe = clusterStore.subscribe((d) => {
      status = d.cluster;
    });
    // 5.8: Pre-fill join form from deep link address
    if (connectAddr) {
      const parts = connectAddr.split(":");
      if (parts.length === 2) {
        joinHost = parts[0];
        joinPort = parseInt(parts[1]) || 8000;
      }
    }
  });

  onDestroy(() => {
    unsubscribe?.();
  });

  async function handleCreate() {
    loading = true;
    error = null;
    try {
      logStore.info("cluster", `Creating cluster on port ${createPort}`);
      await createCluster(createPort, createModel || undefined);
      await clusterStore.refresh();
      toastStore.success("Cluster created on port " + createPort);
      logStore.info("cluster", `Cluster created successfully on port ${createPort}`);
    } catch (e: unknown) {
      error = String(e);
      logStore.error("cluster", `Failed to create cluster: ${e}`);
    } finally {
      loading = false;
    }
  }

  async function handleJoin() {
    loading = true;
    error = null;
    try {
      logStore.info("cluster", `Joining cluster at ${joinHost}:${joinPort}`);
      await joinCluster(joinHost, joinPort);
      await clusterStore.refresh();
      toastStore.success("Joined cluster at " + joinHost + ":" + joinPort);
      logStore.info("cluster", `Successfully joined cluster at ${joinHost}:${joinPort}`);
    } catch (e: unknown) {
      error = String(e);
      logStore.error("cluster", `Failed to join cluster: ${e}`);
    } finally {
      loading = false;
    }
  }

  async function handleLeave() {
    if (!window.confirm("Are you sure you want to leave the cluster? This will stop all distributed inference.")) {
      return;
    }
    loading = true;
    error = null;
    try {
      logStore.info("cluster", "Leaving cluster");
      await leaveCluster();
      await clusterStore.refresh();
      toastStore.info("Left the cluster");
      logStore.info("cluster", "Left the cluster successfully");
    } catch (e: unknown) {
      error = String(e);
      logStore.error("cluster", `Failed to leave cluster: ${e}`);
    } finally {
      loading = false;
    }
  }

  function fmtAddr(addr: string | null | undefined): string {
    return addr ?? "—";
  }
</script>

<div class="cluster-page">
  <h1 class="page-title">Cluster</h1>

  <ErrorBanner message={error ?? ""} ondismiss={() => (error = null)} />

  {#if status?.running}
    <Card title="Cluster Running">
      <div class="status-row">
        <StatusDot variant="green" />
        <span>Active</span>
        <span class="mono">— {fmtAddr(status.coordinator_addr)}</span>
      </div>
      <div class="cluster-info">
        <div class="info-item">
          <span class="info-label">Nodes</span>
          <span class="info-value">{status.nodes.length}</span>
        </div>
      </div>
      <div class="button-row">
        <Button variant="danger" onclick={handleLeave} disabled={loading}>
          {loading ? "Stopping..." : "Leave Cluster"}
        </Button>
        <Button variant="ghost" onclick={() => clusterStore.refresh()} disabled={loading}>
          Refresh Status
        </Button>
      </div>
    </Card>
  {:else}
    <Card title="Create Cluster" description="Start a new coordinator on this machine. Others can join via your IP address.">
      <div class="form-row">
        <label class="form-label" for="create-port">Port</label>
        <Input id="create-port" type="number" bind:value={createPort} min={1024} max={65535} />
      </div>
      <div class="form-row">
        <label class="form-label" for="create-model">Model (optional)</label>
        <Input id="create-model" placeholder="e.g. HuggingFaceTB/SmolLM-135M" bind:value={createModel} />
      </div>
      <Button onclick={handleCreate} disabled={loading}>
        {loading ? "Creating..." : "Create Cluster"}
      </Button>
    </Card>

    <Card title="Join Cluster" description="Connect to an existing coordinator.">
      <div class="form-row">
        <label class="form-label" for="join-host">Coordinator Host</label>
        <Input id="join-host" placeholder="192.168.1.100" bind:value={joinHost} />
      </div>
      <div class="form-row">
        <label class="form-label" for="join-port">Port</label>
        <Input id="join-port" type="number" bind:value={joinPort} min={1024} max={65535} />
      </div>
      <Button onclick={handleJoin} disabled={loading}>
        {loading ? "Joining..." : "Join Cluster"}
      </Button>
    </Card>
  {/if}
</div>

<style>
  .cluster-page { max-width: 600px; }
  .cluster-info { margin-bottom: 16px; }
  .info-item { display: flex; gap: 8px; font-size: 14px; }
  .info-label { color: var(--text-secondary); }
  .info-value { font-family: var(--font-mono); font-weight: 600; }
  .button-row { display: flex; gap: 8px; margin-top: 4px; }
</style>
