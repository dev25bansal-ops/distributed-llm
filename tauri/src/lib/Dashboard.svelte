<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { joinCluster, checkCoordinator, updateTrayStatus } from "./api";
  import { clusterStore, logStore } from "./stores";
  import { Card, ErrorBanner, StatusDot, Skeleton } from "./ui";
  import type { ClusterStatus, GpuInfo, SystemInfo } from "./types";
  import { listen } from "@tauri-apps/api/event";
  import OllamaConfig from "./OllamaConfig.svelte";

  let { grafanaToggle = 0 }: { grafanaToggle?: number } = $props();

  let cluster = $state<ClusterStatus | null>(null);
  let gpus = $state<GpuInfo[]>([]);
  let sysInfo = $state<SystemInfo | null>(null);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let grafanaUrl = $state<string>("http://localhost:3000");
  let showGrafana = $state(false);
  let coordinatorDetected = $state(false);

  // 3.4: Subscribe to shared cluster store (single polling source)
  let unsubscribe: (() => void) | undefined;
  let unlistenCrash: (() => void) | undefined;
  let unlistenDismiss: (() => void) | undefined;

  // H3: Validate Grafana URL to prevent iframe injection
  let validGrafanaUrl = $derived.by(() => {
    try {
      const url = new URL(grafanaUrl);
      if (url.protocol === "http:" || url.protocol === "https:") {
        return grafanaUrl;
      }
    } catch {
      // invalid URL
    }
    return null;
  });

  // 4.10: Handle Grafana toggle via keyboard shortcut
  $effect(() => {
    if (grafanaToggle > 0) {
      showGrafana = !showGrafana;
    }
  });

  onMount(async () => {
    await autoDetectCoordinator();

    // Subscribe to shared store
    unsubscribe = clusterStore.subscribe((d) => {
      cluster = d.cluster;
      gpus = d.gpus;
      sysInfo = d.sysInfo;
      loading = d.loading;
      error = d.error;
    });

    // 3.5: Listen for process crash events from health monitor
    unlistenCrash = await listen<string>("process-crashed", (e) => {
      const msg = `${e.payload} process crashed unexpectedly. The cluster may be unavailable.`;
      error = msg;
      logStore.error("dashboard", msg);
      updateTrayStatus(false, 0);
    });

    // 4.10: Listen for Escape key dismiss
    function handleDismiss() {
      error = null;
    }
    window.addEventListener("dismiss-errors", handleDismiss);
    unlistenDismiss = () => window.removeEventListener("dismiss-errors", handleDismiss);
  });

  onDestroy(() => {
    unsubscribe?.();
    unlistenCrash?.();
    unlistenDismiss?.();
  });

  async function autoDetectCoordinator() {
    try {
      logStore.info("dashboard", "Auto-detecting coordinator on localhost:8000");
      const detected = await checkCoordinator("127.0.0.1", 8000);
      if (detected) {
        coordinatorDetected = true;
        logStore.info("dashboard", "Coordinator detected, auto-joining");
        try {
          await joinCluster("127.0.0.1", 8000);
          logStore.info("dashboard", "Auto-joined cluster successfully");
        } catch {
          // Already joined or failed, that's ok
        }
      } else {
        logStore.info("dashboard", "No coordinator detected on localhost:8000");
      }
    } catch {
      logStore.info("dashboard", "No coordinator running");
    }
  }

  function fmtBytes(bytes: number): string {
    const gb = bytes / 1024 / 1024 / 1024;
    return gb.toFixed(1) + " GB";
  }

  function fmtPct(v: number): string {
    return v.toFixed(1) + "%";
  }

  function statusColor(util: number): string {
    if (util > 80) return "var(--danger)";
    if (util > 50) return "var(--warning)";
    return "var(--success)";
  }
</script>

<div class="dashboard">
  <div class="dashboard-header">
    <h1 class="page-title">Dashboard</h1>
    <div class="coordinator-status">
      {#if coordinatorDetected}
        <StatusDot variant="green" />
        <span>Coordinator detected on localhost:8000</span>
      {:else}
        <StatusDot variant="red" />
        <span>No coordinator detected</span>
      {/if}
    </div>
  </div>

  <ErrorBanner message={error ?? ""} ondismiss={() => (error = null)} />

  {#if loading}
    <div class="loading-grid">
      <Card title="Cluster Status">
        <Skeleton height="60px" />
      </Card>
      <Card title="GPU Monitoring">
        <Skeleton height="80px" />
      </Card>
      <Card title="System">
        <Skeleton height="40px" />
      </Card>
    </div>
  {:else}
    <!-- Cluster Status Card -->
    <Card title="Cluster Status">
      {#if cluster?.running}
        <div class="status-row">
          <StatusDot variant="green" />
          <span>Running</span>
          {#if cluster.coordinator_addr}
            <span class="mono">— {cluster.coordinator_addr}</span>
          {/if}
        </div>
        {#if cluster.nodes.length > 0}
          <div class="node-table">
            <div class="node-header">
              <span>Node ID</span>
              <span>GPU</span>
              <span>Utilization</span>
              <span>Layers</span>
              <span>Health</span>
            </div>
            {#each cluster.nodes as node (node.node_id)}
              <div class="node-row">
                <span class="mono">{node.node_id}</span>
                <span>{node.gpu_name}</span>
                <span style="color: {statusColor(node.gpu_utilization)}">
                  {fmtPct(node.gpu_utilization)}
                </span>
                <span class="mono">{node.layers}</span>
                <StatusDot variant={node.healthy ? 'green' : 'red'} />
              </div>
            {/each}
          </div>
        {:else}
          <div class="empty-state">Coordinator starting... no nodes connected yet.</div>
        {/if}
      {:else}
        <div class="status-row">
          <StatusDot variant="gray" />
          <span>Inactive</span>
        </div>
        <div class="empty-state">
          Create or join a cluster to get started.
        </div>
      {/if}
    </Card>

    <!-- GPU Monitoring -->
    <Card title="GPU Monitoring">
      {#if gpus.length === 0}
        <div class="empty-state">
          No NVIDIA GPUs detected, or NVML driver not available.
        </div>
      {:else}
        {#each gpus as gpu (gpu.index)}
          <div class="gpu-card">
            <div class="gpu-header">
              <span class="gpu-name">GPU {gpu.index}: {gpu.name}</span>
              <span class="gpu-temp">{gpu.temperature.toFixed(0)}°C</span>
            </div>
            <div class="gpu-bars">
              <div class="bar-label">
                <span>Utilization</span>
                <span>{fmtPct(gpu.utilization)}</span>
              </div>
              <div class="bar-track">
                <div
                  class="bar-fill util"
                  style="width: {gpu.utilization}%; background: {statusColor(gpu.utilization)}"
                ></div>
              </div>
              <div class="bar-label">
                <span>Memory</span>
                <span>{fmtBytes(gpu.memory_used)} / {fmtBytes(gpu.memory_total)}</span>
              </div>
              <div class="bar-track">
                <div
                  class="bar-fill mem"
                  style="width: {gpu.memory_total > 0 ? (gpu.memory_used / gpu.memory_total * 100) : 0}%"
                ></div>
              </div>
            </div>
          </div>
        {/each}
      {/if}
    </Card>

    <!-- System Info -->
    {#if sysInfo}
      <Card title="System">
        <div class="sys-grid">
          <div class="sys-item">
            <span class="sys-label">OS</span>
            <span class="sys-value">{sysInfo.os}</span>
          </div>
          <div class="sys-item">
            <span class="sys-label">CPU</span>
            <span class="sys-value">{sysInfo.cpu}</span>
          </div>
          <div class="sys-item">
            <span class="sys-label">RAM</span>
            <span class="sys-value">{sysInfo.ram_gb} GB</span>
          </div>
          <div class="sys-item">
            <span class="sys-label">Python</span>
            <span class="sys-value mono">{sysInfo.python_version ?? "N/A"}</span>
          </div>
          <div class="sys-item">
            <span class="sys-label">distllm</span>
            <span class="sys-value mono">v{sysInfo.distllm_version}</span>
          </div>
          <div class="sys-item">
            <span class="sys-label">GPUs</span>
            <span class="sys-value">{sysInfo.gpus.length} detected</span>
          </div>
        </div>
      </Card>
    {/if}

    <!-- Ollama Compatibility -->
    <Card title="Ollama">
      <OllamaConfig />
    </Card>

    <!-- Grafana Observability Dashboard -->
    <section class="card">
      <div class="grafana-header">
        <h2 class="card-title">Observability Dashboard</h2>
        <div class="grafana-controls">
          <input
            type="text"
            class="grafana-url-input"
            placeholder="Grafana URL (e.g., http://localhost:3000)"
            bind:value={grafanaUrl}
          />
          <button class="grafana-toggle" onclick={() => showGrafana = !showGrafana}>
            {showGrafana ? 'Hide' : 'Show'} Grafana
          </button>
        </div>
      </div>
      {#if showGrafana}
        <div class="grafana-container">
          {#if validGrafanaUrl}
            <iframe
              src="{validGrafanaUrl}/d/distllm/distllm-overview?orgId=1&refresh=10s&kiosk"
              title="Grafana Dashboard"
              frameborder="0"
              sandbox="allow-scripts allow-same-origin allow-popups"
              allowfullscreen
            ></iframe>
          {:else}
            <div class="empty-state" style="color: var(--danger);">
              Invalid Grafana URL. Must start with http:// or https://
            </div>
          {/if}
        </div>
      {:else}
        <div class="empty-state">
          Click "Show Grafana" to embed the observability dashboard.
          Requires Grafana running with the DistLLM dashboard provisioned.
        </div>
      {/if}
    </section>
  {/if}
</div>

<style>
  .dashboard { max-width: 900px; }
  .dashboard-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
  .coordinator-status { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text-muted); }
  .loading-grid { display: flex; flex-direction: column; gap: 16px; }
  .node-table { width: 100%; font-size: 13px; }
  .node-header, .node-row { display: grid; grid-template-columns: 2fr 2fr 1fr 1fr 0.5fr; gap: 8px; padding: 8px 0; align-items: center; }
  .node-header { color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid var(--border); }
  .node-row { border-bottom: 1px solid color-mix(in srgb, var(--border) 50%, transparent); }
  .gpu-card { margin-bottom: 12px; }
  .gpu-card:last-child { margin-bottom: 0; }
  .gpu-header { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 13px; }
  .gpu-name { font-weight: 500; }
  .gpu-temp { color: var(--warning); }
  .bar-label { display: flex; justify-content: space-between; font-size: 12px; color: var(--text-secondary); margin-bottom: 4px; }
  .bar-track { height: 6px; background: var(--bg-input); border-radius: 3px; overflow: hidden; margin-bottom: 10px; }
  .bar-fill { height: 100%; border-radius: 3px; transition: width 0.5s ease; }
  .bar-fill.mem { background: var(--gpu-mem); }
  .sys-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .sys-item { display: flex; flex-direction: column; gap: 2px; }
  .sys-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
  .sys-value { font-size: 13px; }
  .grafana-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px; }
  .grafana-controls { display: flex; gap: 8px; align-items: center; }
  .grafana-url-input { padding: 6px 10px; border: 1px solid var(--border); border-radius: 6px; font-size: 12px; width: 280px; background: var(--bg-input); color: var(--text); }
  .grafana-toggle { padding: 6px 14px; background: var(--accent); color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 500; }
  .grafana-toggle:hover { opacity: 0.9; }
  .grafana-container { width: 100%; height: 450px; border-radius: 8px; overflow: hidden; }
  .grafana-container iframe { width: 100%; height: 100%; border: none; }
</style>
