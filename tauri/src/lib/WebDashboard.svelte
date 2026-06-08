<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { webDashboardStore, logStore } from "./stores";
  import { ErrorBanner } from "./ui";
  import type { WebDashboardConfig, WebDashboardStatus } from "./types";

  let config = $state<WebDashboardConfig>({
    enabled: false,
    port: 8080,
    auth_required: true,
    auth_token: "",
    cors_origins: ["*"],
  });
  let status = $state<WebDashboardStatus>({ running: false, url: "", connections: 0 });
  let loading = $state(true);
  let error = $state<string | null>(null);
  let showToken = $state(false);

  let unsubscribe: (() => void) | undefined;

  onMount(() => {
    unsubscribe = webDashboardStore.subscribe((d) => {
      config = d.config;
      status = d.status;
      loading = d.loading;
      error = d.error;
    });
    webDashboardStore.refresh();
  });

  onDestroy(() => unsubscribe?.());

  async function handleStart() {
    logStore.info("webdashboard", "Starting web dashboard server");
    await webDashboardStore.start();
  }

  async function handleStop() {
    logStore.info("webdashboard", "Stopping web dashboard server");
    await webDashboardStore.stop();
  }

  async function saveConfig() {
    await webDashboardStore.updateConfig(config);
  }

  function generateToken() {
    const arr = new Uint8Array(32);
    crypto.getRandomValues(arr);
    config.auth_token = Array.from(arr, (b) => b.toString(16).padStart(2, "0")).join("");
    saveConfig();
  }
</script>

<div class="webdashboard-page">
  <h1 class="page-title">Web Dashboard</h1>

  <ErrorBanner message={error ?? ""} ondismiss={() => (error = null)} />

  <div class="card">
    <h2 class="card-title">Server Status</h2>
    <div class="status-row">
      {#if status.running}
        <span class="status-badge running">Running</span>
        <span class="status-url">{status.url}</span>
        <span class="status-conn">{status.connections} active connection{status.connections !== 1 ? "s" : ""}</span>
      {:else}
        <span class="status-badge stopped">Stopped</span>
      {/if}
    </div>
    <div class="button-row">
      {#if status.running}
        <button class="btn btn-danger" onclick={handleStop}>Stop Server</button>
      {:else}
        <button class="btn btn-primary" onclick={handleStart} disabled={!config.enabled}>Start Server</button>
      {/if}
    </div>
  </div>

  <div class="card">
    <h2 class="card-title">Configuration</h2>
    <p class="card-desc">Serve the Svelte frontend from the coordinator for headless management and mobile access.</p>

    <div class="form-grid">
      <div class="form-row">
        <label class="form-label">
          <input type="checkbox" bind:checked={config.enabled} onchange={saveConfig} />
          Enable Web Dashboard
        </label>
      </div>
      <div class="form-row">
        <label class="form-label" for="wd-port">Port</label>
        <input id="wd-port" class="input" type="number" bind:value={config.port} min={1} max={65535} onchange={saveConfig} />
      </div>
      <div class="form-row">
        <label class="form-label">
          <input type="checkbox" bind:checked={config.auth_required} onchange={saveConfig} />
          Require Authentication
        </label>
      </div>
      {#if config.auth_required}
        <div class="form-row token-row">
          <label class="form-label" for="wd-token">Auth Token</label>
          <div class="token-input">
            <input
              id="wd-token"
              class="input"
              type={showToken ? "text" : "password"}
              bind:value={config.auth_token}
              onchange={saveConfig}
              placeholder="Generate or enter a token"
            />
            <button class="btn-icon" onclick={() => showToken = !showToken} title={showToken ? "Hide" : "Show"}>
              {showToken ? "🙈" : "👁"}
            </button>
            <button class="btn btn-ghost btn-sm" onclick={generateToken}>Generate</button>
          </div>
        </div>
      {/if}
      <div class="form-row">
        <label class="form-label" for="wd-cors">CORS Origins (comma-separated)</label>
        <input
          id="wd-cors"
          class="input"
          value={config.cors_origins.join(", ")}
          onchange={(e) => {
            config.cors_origins = (e.target as HTMLInputElement).value.split(",").map(s => s.trim()).filter(Boolean);
            saveConfig();
          }}
          placeholder="*"
        />
      </div>
    </div>
  </div>

  <div class="card">
    <h2 class="card-title">Access</h2>
    {#if status.running}
      <div class="access-info">
        <div class="access-item">
          <span class="access-label">Local</span>
          <code class="access-url">http://localhost:{config.port}</code>
        </div>
        <div class="access-item">
          <span class="access-label">Network</span>
          <code class="access-url">http://&lt;your-ip&gt;:{config.port}</code>
        </div>
        {#if config.auth_required}
          <div class="access-note">
            Authentication required. Pass the token as a query parameter: <code>?token=YOUR_TOKEN</code>
          </div>
        {/if}
      </div>
    {:else}
      <div class="empty-state">Start the server to access the web dashboard.</div>
    {/if}
  </div>
</div>

<style>
  .webdashboard-page { max-width: 700px; }
  .status-row { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
  .status-badge {
    padding: 4px 12px; border-radius: 6px; font-size: 12px; font-weight: 600; text-transform: uppercase;
  }
  .status-badge.running { background: color-mix(in srgb, var(--success) 20%, transparent); color: var(--success); }
  .status-badge.stopped { background: color-mix(in srgb, var(--text-muted) 20%, transparent); color: var(--text-muted); }
  .status-url { font-family: var(--font-mono); font-size: 13px; color: var(--accent); }
  .status-conn { font-size: 12px; color: var(--text-muted); }
  .button-row { display: flex; gap: 8px; }
  .form-grid { display: flex; flex-direction: column; gap: 14px; }
  .form-row { display: flex; flex-direction: column; gap: 4px; }
  .form-label { font-size: 12px; color: var(--text-secondary); display: flex; align-items: center; gap: 6px; cursor: pointer; }
  .form-label input[type="checkbox"] { accent-color: var(--accent); }
  .token-row { grid-column: 1 / -1; }
  .token-input { display: flex; gap: 6px; align-items: center; }
  .token-input .input { flex: 1; }
  .btn-icon { background: none; border: none; cursor: pointer; font-size: 16px; padding: 4px; }
  .btn-sm { padding: 6px 12px; font-size: 12px; }
  .access-info { display: flex; flex-direction: column; gap: 10px; }
  .access-item { display: flex; align-items: center; gap: 10px; }
  .access-label { font-size: 12px; color: var(--text-muted); width: 60px; }
  .access-url { font-family: var(--font-mono); font-size: 13px; background: var(--bg-input); padding: 4px 8px; border-radius: 4px; }
  .access-note { font-size: 12px; color: var(--text-muted); margin-top: 4px; }
  .access-note code { background: var(--bg-input); padding: 2px 4px; border-radius: 3px; font-size: 11px; }
</style>
