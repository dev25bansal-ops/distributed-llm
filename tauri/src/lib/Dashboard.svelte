<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { getClusterStatus, getGpuMetrics, getSystemInfo } from "./api";
  import type { ClusterStatus, GpuInfo, SystemInfo } from "./types";

  let cluster = $state<ClusterStatus | null>(null);
  let gpus = $state<GpuInfo[]>([]);
  let sysInfo = $state<SystemInfo | null>(null);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let pollTimer: ReturnType<typeof setInterval> | undefined;

  onMount(async () => {
    await loadAll();
    pollTimer = setInterval(loadAll, 3000);
  });

  onDestroy(() => {
    if (pollTimer) clearInterval(pollTimer);
  });

  async function loadAll() {
    try {
      const [c, g, s] = await Promise.all([
        getClusterStatus(),
        getGpuMetrics(),
        getSystemInfo(),
      ]);
      cluster = c;
      gpus = g;
      sysInfo = s;
      error = null;
    } catch (e: unknown) {
      error = String(e);
    } finally {
      loading = false;
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
  <h1 class="page-title">Dashboard</h1>

  {#if error}
    <div class="error-banner">{error}</div>
  {/if}

  {#if loading}
    <div class="loading">Loading system information...</div>
  {:else}
    <!-- Cluster Status Card -->
    <section class="card">
      <h2 class="card-title">Cluster Status</h2>
      {#if cluster?.running}
        <div class="status-row">
          <span class="status-dot green"></span>
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
            {#each cluster.nodes as node}
              <div class="node-row">
                <span class="mono">{node.node_id}</span>
                <span>{node.gpu_name}</span>
                <span style="color: {statusColor(node.gpu_utilization)}">
                  {fmtPct(node.gpu_utilization)}
                </span>
                <span class="mono">{node.layers}</span>
                <span class="status-dot {node.healthy ? 'green' : 'red'}"></span>
              </div>
            {/each}
          </div>
        {:else}
          <div class="empty-state">Coordinator starting... no nodes connected yet.</div>
        {/if}
      {:else}
        <div class="status-row">
          <span class="status-dot gray"></span>
          <span>Inactive</span>
        </div>
        <div class="empty-state">
          Create or join a cluster to get started.
        </div>
      {/if}
    </section>

    <!-- GPU Monitoring -->
    <section class="card">
      <h2 class="card-title">GPU Monitoring</h2>
      {#if gpus.length === 0}
        <div class="empty-state">
          No NVIDIA GPUs detected, or NVML driver not available.
        </div>
      {:else}
        {#each gpus as gpu}
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
    </section>

    <!-- System Info -->
    {#if sysInfo}
      <section class="card">
        <h2 class="card-title">System</h2>
        <div class="sys-grid">
          <div class="sys-item">
            <span class="sys-label">OS</span>
            <span class="sys-value">{sysInfo.os}</span>
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
      </section>
    {/if}
  {/if}
</div>

<style>
  .dashboard { max-width: 900px; }
  .page-title { font-size: 22px; font-weight: 700; margin-bottom: 20px; }
  .loading { color: var(--text-secondary); padding: 40px 0; text-align: center; }
  .error-banner {
    background: color-mix(in srgb, var(--danger) 15%, transparent);
    color: var(--danger);
    padding: 10px 14px;
    border-radius: 8px;
    margin-bottom: 16px;
    font-size: 13px;
  }
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
  }
  .card-title { font-size: 15px; font-weight: 600; margin-bottom: 14px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; }
  .status-row { display: flex; align-items: center; gap: 10px; font-size: 14px; margin-bottom: 12px; }
  .status-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  .status-dot.green { background: var(--success); box-shadow: 0 0 6px var(--success); }
  .status-dot.red { background: var(--danger); }
  .status-dot.gray { background: var(--text-muted); }
  .mono { font-family: var(--font-mono); font-size: 12px; }
  .node-table { width: 100%; font-size: 13px; }
  .node-header, .node-row { display: grid; grid-template-columns: 2fr 2fr 1fr 1fr 0.5fr; gap: 8px; padding: 8px 0; align-items: center; }
  .node-header { color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid var(--border); }
  .node-row { border-bottom: 1px solid color-mix(in srgb, var(--border) 50%, transparent); }
  .empty-state { color: var(--text-muted); font-size: 13px; padding: 12px 0; }
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
</style>
