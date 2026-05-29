<script lang="ts">
  import { createCluster, joinCluster, leaveCluster, getClusterStatus } from "./api";
  import type { ClusterStatus } from "./types";

  let { initialAction }: { initialAction?: string | null } = $props();

  let status = $state<ClusterStatus | null>(null);
  let loading = $state(false);
  let error = $state<string | null>(null);

  // Create form
  let createPort = $state(8000);
  let createModel = $state("");

  // Join form
  let joinHost = $state("127.0.0.1");
  let joinPort = $state(8000);

  let pollTimer: ReturnType<typeof setInterval> | undefined;

  function startPoll() {
    stopPoll();
    pollTimer = setInterval(refreshStatus, 3000);
  }

  function stopPoll() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = undefined;
    }
  }

  async function refreshStatus() {
    try {
      status = await getClusterStatus();
    } catch {
      // ignore poll errors
    }
  }

  async function handleCreate() {
    loading = true;
    error = null;
    try {
      status = await createCluster(createPort, createModel || undefined);
      startPoll();
    } catch (e: unknown) {
      error = String(e);
    } finally {
      loading = false;
    }
  }

  async function handleJoin() {
    loading = true;
    error = null;
    try {
      status = await joinCluster(joinHost, joinPort);
      startPoll();
    } catch (e: unknown) {
      error = String(e);
    } finally {
      loading = false;
    }
  }

  async function handleLeave() {
    loading = true;
    error = null;
    try {
      await leaveCluster();
      status = null;
      stopPoll();
    } catch (e: unknown) {
      error = String(e);
    } finally {
      loading = false;
    }
  }

  async function handleRefresh() {
    await refreshStatus();
  }

  function fmtAddr(addr: string | null | undefined): string {
    return addr ?? "—";
  }
</script>

<div class="cluster-page">
  <h1 class="page-title">Cluster</h1>

  {#if error}
    <div class="error-banner">{error}</div>
  {/if}

  {#if status?.running}
    <!-- Running state -->
    <section class="card">
      <h2 class="card-title">Cluster Running</h2>
      <div class="status-row">
        <span class="status-dot green"></span>
        <span>Active</span>
        <span class="mono">— {fmtAddr(status.coordinator_addr)}</span>
      </div>
      <div class="cluster-info">
        <div class="info-item">
          <span class="info-label">Nodes</span>
          <span class="info-value">{status.nodes.length}</span>
        </div>
      </div>
      <button class="btn btn-danger" onclick={handleLeave} disabled={loading}>
        {loading ? "Stopping..." : "Leave Cluster"}
      </button>
      <button class="btn btn-ghost" onclick={handleRefresh} disabled={loading}>
        Refresh Status
      </button>
    </section>
  {:else}
    <!-- Create -->
    <section class="card">
      <h2 class="card-title">Create Cluster</h2>
      <p class="card-desc">Start a new coordinator on this machine. Others can join via your IP address.</p>
      <div class="form-row">
        <label class="form-label" for="create-port">Port</label>
        <input id="create-port" type="number" class="input" bind:value={createPort} min={1024} max={65535} />
      </div>
      <div class="form-row">
        <label class="form-label" for="create-model">Model (optional)</label>
        <input id="create-model" type="text" class="input" placeholder="e.g. HuggingFaceTB/SmolLM-135M" bind:value={createModel} />
      </div>
      <button class="btn btn-primary" onclick={handleCreate} disabled={loading}>
        {loading ? "Creating..." : "Create Cluster"}
      </button>
    </section>

    <!-- Join -->
    <section class="card">
      <h2 class="card-title">Join Cluster</h2>
      <p class="card-desc">Connect to an existing coordinator.</p>
      <div class="form-row">
        <label class="form-label" for="join-host">Coordinator Host</label>
        <input id="join-host" type="text" class="input" placeholder="192.168.1.100" bind:value={joinHost} />
      </div>
      <div class="form-row">
        <label class="form-label" for="join-port">Port</label>
        <input id="join-port" type="number" class="input" bind:value={joinPort} min={1024} max={65535} />
      </div>
      <button class="btn btn-primary" onclick={handleJoin} disabled={loading}>
        {loading ? "Joining..." : "Join Cluster"}
      </button>
    </section>
  {/if}
</div>

<style>
  .cluster-page { max-width: 600px; }
  .page-title { font-size: 22px; font-weight: 700; margin-bottom: 20px; }
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
  .card-title { font-size: 15px; font-weight: 600; margin-bottom: 6px; }
  .card-desc { font-size: 13px; color: var(--text-secondary); margin-bottom: 16px; }
  .status-row { display: flex; align-items: center; gap: 10px; font-size: 14px; margin-bottom: 14px; }
  .status-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  .status-dot.green { background: var(--success); box-shadow: 0 0 6px var(--success); }
  .mono { font-family: var(--font-mono); font-size: 12px; }
  .form-row { margin-bottom: 14px; }
  .form-label { display: block; font-size: 12px; color: var(--text-secondary); margin-bottom: 4px; }
  .input {
    width: 100%;
    padding: 10px 12px;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 14px;
    transition: border-color 0.15s;
  }
  .input:focus { border-color: var(--accent); }
  .cluster-info { margin-bottom: 16px; }
  .info-item { display: flex; gap: 8px; font-size: 14px; }
  .info-label { color: var(--text-secondary); }
  .info-value { font-family: var(--font-mono); font-weight: 600; }
  .btn {
    padding: 10px 20px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    transition: all 0.15s;
    margin-right: 8px;
    margin-top: 4px;
  }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-primary { background: var(--accent); color: #fff; }
  .btn-primary:hover:not(:disabled) { background: var(--accent-hover); }
  .btn-danger { background: var(--danger); color: #fff; }
  .btn-danger:hover:not(:disabled) { background: color-mix(in srgb, var(--danger) 80%, #fff); }
  .btn-ghost { background: transparent; color: var(--text-secondary); border: 1px solid var(--border); }
  .btn-ghost:hover:not(:disabled) { background: var(--bg-input); }
</style>
