<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { discoveryStore, clusterStore, logStore } from "./stores";
  import { ErrorBanner, StatusDot } from "./ui";
  import type { DiscoveredService, ClusterStatus } from "./types";

  let services = $state<DiscoveredService[]>([]);
  let active = $state(false);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let cluster = $state<ClusterStatus | null>(null);

  let unsubscribe: (() => void) | undefined;
  let unsubCluster: (() => void) | undefined;

  onMount(() => {
    unsubscribe = discoveryStore.subscribe((d) => {
      services = d.services;
      active = d.active;
      loading = d.loading;
      error = d.error;
    });
    unsubCluster = clusterStore.subscribe((d) => {
      cluster = d.cluster;
    });
  });

  onDestroy(() => {
    unsubscribe?.();
    unsubCluster?.();
  });

  async function handleStart() {
    logStore.info("discovery", "Starting mDNS discovery");
    await discoveryStore.start();
  }

  async function handleStop() {
    logStore.info("discovery", "Stopping mDNS discovery");
    await discoveryStore.stop();
  }

  function timeAgo(ts: number): string {
    const secs = Math.floor((Date.now() - ts) / 1000);
    if (secs < 60) return `${secs}s ago`;
    if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
    return `${Math.floor(secs / 3600)}h ago`;
  }

  function isLocal(s: DiscoveredService): boolean {
    return s.host === "127.0.0.1" || s.host === "localhost";
  }
</script>

<div class="discovery-page">
  <h1 class="page-title">Network Discovery</h1>

  <ErrorBanner message={error ?? ""} ondismiss={() => (error = null)} />

  <div class="card">
    <h2 class="card-title">mDNS Discovery</h2>
    <p class="card-desc">
      Automatically discover Distributed LLM coordinators on your local network using mDNS.
      No manual IP:port entry needed.
    </p>

    <div class="status-row">
      <StatusDot variant={active ? "green" : "gray"} />
      <span>{active ? "Discovery active — scanning for services" : "Discovery inactive"}</span>
    </div>

    <div class="button-row">
      {#if active}
        <button class="btn btn-danger" onclick={handleStop}>Stop Discovery</button>
      {:else}
        <button class="btn btn-primary" onclick={handleStart}>Start Discovery</button>
      {/if}
    </div>
  </div>

  <div class="card">
    <h2 class="card-title">Discovered Services ({services.length})</h2>

    {#if services.length === 0}
      <div class="empty-state">
        {#if active}
          Scanning for nearby coordinators...
        {:else}
          Start discovery to find Distributed LLM instances on your network.
        {/if}
      </div>
    {:else}
      <div class="service-list">
        {#each services as svc (svc.host + svc.port)}
          <div class="service-item" class:local={isLocal(svc)}>
            <div class="service-main">
              <StatusDot variant="green" />
              <div class="service-info">
                <span class="service-addr">{svc.host}:{svc.port}</span>
                <span class="service-meta">
                  {#if isLocal(svc)}
                    Local instance
                  {:else}
                    Network instance · Seen {timeAgo(svc.discovered_at)}
                  {/if}
                </span>
              </div>
            </div>
            <div class="service-actions">
              {#if !isLocal(svc) && cluster && !cluster.running}
                <button class="btn btn-ghost btn-sm" onclick={() => {
                  // Navigate to cluster page to join
                  window.dispatchEvent(new CustomEvent("navigate", { detail: "cluster" }));
                }}>
                  Join
                </button>
              {/if}
              {#if isLocal(svc)}
                <span class="local-badge">You</span>
              {/if}
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </div>

  <div class="card">
    <h2 class="card-title">How it works</h2>
    <div class="how-list">
      <div class="how-item">
        <span class="how-num">1</span>
        <span>mDNS broadcasts announce coordinator services on the local network</span>
      </div>
      <div class="how-item">
        <span class="how-num">2</span>
        <span>This device listens for broadcasts and maintains a service list</span>
      </div>
      <div class="how-item">
        <span class="how-num">3</span>
        <span>Click "Join" to connect to a discovered coordinator as a worker node</span>
      </div>
    </div>
  </div>
</div>

<style>
  .discovery-page { max-width: 700px; }
  .status-row { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; font-size: 14px; }
  .button-row { display: flex; gap: 8px; }
  .service-list { display: flex; flex-direction: column; gap: 2px; }
  .service-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px; border-radius: 8px; transition: background 0.15s;
  }
  .service-item:hover { background: var(--bg-input); }
  .service-item.local { border-left: 3px solid var(--accent); }
  .service-main { display: flex; align-items: center; gap: 10px; }
  .service-info { display: flex; flex-direction: column; }
  .service-addr { font-family: var(--font-mono); font-size: 13px; font-weight: 600; }
  .service-meta { font-size: 11px; color: var(--text-muted); }
  .service-actions { display: flex; gap: 6px; align-items: center; }
  .btn-sm { padding: 4px 12px; font-size: 12px; }
  .btn-ghost { background: transparent; color: var(--text-secondary); border: 1px solid var(--border); border-radius: 6px; cursor: pointer; }
  .btn-ghost:hover { background: var(--bg-input); color: var(--text-primary); }
  .local-badge {
    font-size: 10px; font-weight: 600; color: var(--accent);
    background: color-mix(in srgb, var(--accent) 15%, transparent);
    padding: 2px 8px; border-radius: 4px;
  }
  .how-list { display: flex; flex-direction: column; gap: 10px; }
  .how-item { display: flex; align-items: flex-start; gap: 10px; font-size: 13px; color: var(--text-secondary); }
  .how-num {
    width: 22px; height: 22px; border-radius: 50%;
    background: var(--bg-input); color: var(--text-muted);
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 600; flex-shrink: 0;
  }
</style>
