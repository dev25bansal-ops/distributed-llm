<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { settingsStore } from "./stores";
  import { applyTheme } from "./stores/settings-store";
  import { Card, Input, ErrorBanner, toastStore } from "./ui";
  import type { AppSettings } from "./stores/settings-store";
  import { checkForUpdates, installUpdate, initNotifications, notify } from "./api";
  import type { UpdateCheckResult } from "./api";

  let settings = $state<AppSettings>({
    defaultClusterPort: 8000,
    grafanaUrl: "http://localhost:3000",
    theme: "dark",
    autoJoin: false,
    downloadDir: "",
    notifications: {
      clusterEvents: true,
      modelDownloads: true,
      inferenceRequests: false,
      errors: true,
      native: true,
      updateAvailable: true,
    },
    pythonPath: "",
    apiEndpoint: "",
    updateServerUrl: "https://releases.distributed-llm.dev",
  });
  let error = $state<string | null>(null);
  let unsubscribe: (() => void) | undefined;

  // Update check state
  let updateStatus = $state<"idle" | "checking" | "available" | "uptodate" | "error">("idle");
  let updateInfo = $state<UpdateCheckResult | null>(null);
  let installing = $state(false);

  onMount(() => {
    unsubscribe = settingsStore.subscribe((s) => {
      settings = s;
    });
  });

  onDestroy(() => unsubscribe?.());

  function save() {
    settingsStore.update(settings);
    toastStore.success("Settings saved");
  }

  function resetDefaults() {
    if (!window.confirm("Reset all settings to defaults?")) return;
    settingsStore.reset();
    toastStore.info("Settings reset to defaults");
  }

  function applyThemeAndSave() {
    applyTheme(settings.theme);
  }

  $effect(() => {
    settings.theme;
    applyThemeAndSave();
  });

  async function handleCheckUpdates() {
    updateStatus = "checking";
    updateInfo = null;
    try {
      const update = await checkForUpdates();
      if (update?.available) {
        updateStatus = "available";
        updateInfo = {
          available: true,
          version: update.version,
          body: update.body,
          date: update.date,
        };
        toastStore.info(`Update v${update.version} available`);
      } else {
        updateStatus = "uptodate";
        toastStore.success("You have the latest version");
      }
    } catch (e) {
      updateStatus = "error";
      error = String(e);
    }
  }

  async function handleInstallUpdate() {
    installing = true;
    try {
      toastStore.info("Downloading update...");
      await installUpdate();
      toastStore.success("Update installed successfully");
    } catch (e) {
      toastStore.error(`Install failed: ${e}`);
    } finally {
      installing = false;
    }
  }

  async function handleTestNativeNotification() {
    const granted = await initNotifications();
    if (granted) {
      notify("Distributed LLM", "Native notifications are working!");
    } else {
      toastStore.error("Notification permission not granted");
    }
  }
</script>

<div class="settings-page">
  <h1 class="page-title">Settings</h1>

  <ErrorBanner message={error ?? ""} ondismiss={() => (error = null)} />

  <!-- Cluster Settings -->
  <Card title="Cluster">
    <div class="form-row">
      <label class="form-label" for="set-port">Default Cluster Port</label>
      <Input
        id="set-port"
        type="number"
        bind:value={settings.defaultClusterPort}
        min={1024}
        max={65535}
      />
    </div>
    <div class="form-row">
      <label class="toggle-label">
        <input
          type="checkbox"
          bind:checked={settings.autoJoin}
          class="toggle-input"
        />
        <span class="toggle-switch"></span>
        <span>Auto-join last cluster on launch</span>
      </label>
    </div>
  </Card>

  <!-- Appearance -->
  <Card title="Appearance">
    <div class="form-row">
      <label class="form-label">Theme</label>
      <div class="theme-options">
        {#each ["dark", "light", "auto"] as theme (theme)}
          <button
            class="theme-btn"
            class:selected={settings.theme === theme}
            onclick={() => settingsStore.update({ theme })}
          >
            {#if theme === "dark"}🌙{:else if theme === "light"}☀️{:else}💻{/if}
            <span>{theme.charAt(0).toUpperCase() + theme.slice(1)}</span>
          </button>
        {/each}
      </div>
    </div>
  </Card>

  <!-- API & Integrations -->
  <Card title="API & Integrations">
    <div class="form-row">
      <label class="form-label" for="set-api-endpoint">API Endpoint</label>
      <Input
        id="set-api-endpoint"
        bind:value={settings.apiEndpoint}
        placeholder="http://localhost:8000"
      />
      <span class="field-hint">Override the default coordinator API URL</span>
    </div>
    <div class="form-row">
      <label class="form-label" for="set-grafana">Grafana URL</label>
      <Input
        id="set-grafana"
        bind:value={settings.grafanaUrl}
        placeholder="http://localhost:3000"
      />
    </div>
    <div class="form-row">
      <label class="form-label" for="set-python">Python Path (optional)</label>
      <Input
        id="set-python"
        bind:value={settings.pythonPath}
        placeholder="/usr/bin/python3"
      />
    </div>
  </Card>

  <!-- Downloads -->
  <Card title="Downloads">
    <div class="form-row">
      <label class="form-label" for="set-dl-dir">Download Directory</label>
      <Input
        id="set-dl-dir"
        bind:value={settings.downloadDir}
        placeholder="Default (app data dir)"
      />
    </div>
  </Card>

  <!-- Notifications -->
  <Card title="Notifications">
    <div class="notif-grid">
      <label class="toggle-label">
        <input
          type="checkbox"
          bind:checked={settings.notifications.clusterEvents}
          class="toggle-input"
        />
        <span class="toggle-switch"></span>
        <span>Cluster events (node join/leave)</span>
      </label>
      <label class="toggle-label">
        <input
          type="checkbox"
          bind:checked={settings.notifications.modelDownloads}
          class="toggle-input"
        />
        <span class="toggle-switch"></span>
        <span>Model downloads</span>
      </label>
      <label class="toggle-label">
        <input
          type="checkbox"
          bind:checked={settings.notifications.inferenceRequests}
          class="toggle-input"
        />
        <span class="toggle-switch"></span>
        <span>Inference requests</span>
      </label>
      <label class="toggle-label">
        <input
          type="checkbox"
          bind:checked={settings.notifications.errors}
          class="toggle-input"
        />
        <span class="toggle-switch"></span>
        <span>Errors</span>
      </label>
      <label class="toggle-label">
        <input
          type="checkbox"
          bind:checked={settings.notifications.updateAvailable}
          class="toggle-input"
        />
        <span class="toggle-switch"></span>
        <span>Update available</span>
      </label>
      <label class="toggle-label">
        <input
          type="checkbox"
          bind:checked={settings.notifications.native}
          class="toggle-input"
        />
        <span class="toggle-switch"></span>
        <span>Native OS notifications</span>
      </label>
    </div>
    <div class="form-row" style="margin-top: 12px;">
      <button class="btn btn-ghost btn-small" onclick={handleTestNativeNotification}>
        Test Notification
      </button>
    </div>
  </Card>

  <!-- Software Updates -->
  <Card title="Software Updates">
    <div class="form-row">
      <label class="form-label" for="set-update-url">Update Server URL</label>
      <Input
        id="set-update-url"
        bind:value={settings.updateServerUrl}
        placeholder="https://releases.distributed-llm.dev"
      />
    </div>
    <div class="update-section">
      <button
        class="btn btn-primary"
        onclick={handleCheckUpdates}
        disabled={updateStatus === "checking"}
      >
        {#if updateStatus === "checking"}
          Checking...
        {:else}
          Check for Updates
        {/if}
      </button>

      {#if updateStatus === "uptodate"}
        <span class="update-status update-ok">You're up to date!</span>
      {:else if updateStatus === "available"}
        <div class="update-available">
          <span class="update-status update-new">
            Update v{updateInfo?.version ?? "?"} available
          </span>
          {#if updateInfo?.body}
            <p class="update-body">{updateInfo.body}</p>
          {/if}
          <button
            class="btn btn-primary"
            onclick={handleInstallUpdate}
            disabled={installing}
          >
            {installing ? "Installing..." : "Download & Install"}
          </button>
        </div>
      {:else if updateStatus === "error"}
        <span class="update-status update-error">Update check failed</span>
      {/if}
    </div>
  </Card>

  <!-- Actions -->
  <div class="actions">
    <button class="btn btn-primary" onclick={save}>Save Settings</button>
    <button class="btn btn-ghost" onclick={resetDefaults}>Reset to Defaults</button>
  </div>
</div>

<style>
  .settings-page {
    max-width: 600px;
  }
  .form-row {
    margin-bottom: 16px;
  }
  .form-row:last-child {
    margin-bottom: 0;
  }
  .form-label {
    display: block;
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 4px;
  }
  .field-hint {
    display: block;
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 2px;
  }
  .toggle-label {
    display: flex;
    align-items: center;
    gap: 10px;
    cursor: pointer;
    font-size: 14px;
  }
  .toggle-input {
    display: none;
  }
  .toggle-switch {
    width: 36px;
    height: 20px;
    background: rgba(128, 128, 128, 0.3);
    border-radius: 10px;
    position: relative;
    transition: background 0.2s;
    flex-shrink: 0;
  }
  .toggle-switch::after {
    content: "";
    position: absolute;
    top: 2px;
    left: 2px;
    width: 16px;
    height: 16px;
    background: white;
    border-radius: 50%;
    transition: transform 0.2s;
  }
  .toggle-input:checked + .toggle-switch {
    background: var(--accent);
  }
  .toggle-input:checked + .toggle-switch::after {
    transform: translateX(16px);
  }
  .theme-options {
    display: flex;
    gap: 8px;
  }
  .theme-btn {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 12px 16px;
    background: var(--bg-input);
    border: 2px solid transparent;
    border-radius: 8px;
    color: var(--text-secondary);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.15s;
  }
  .theme-btn:hover {
    border-color: var(--border);
  }
  .theme-btn.selected {
    border-color: var(--accent);
    color: var(--text);
    background: color-mix(in srgb, var(--accent) 8%, var(--bg-input));
  }
  .theme-btn span {
    font-size: 12px;
    font-weight: 500;
  }
  .notif-grid {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .update-section {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-top: 12px;
  }
  .update-status {
    font-size: 14px;
    font-weight: 500;
  }
  .update-ok {
    color: var(--success);
  }
  .update-new {
    color: var(--accent);
  }
  .update-error {
    color: var(--danger);
  }
  .update-available {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 12px;
    background: var(--bg-input);
    border-radius: 8px;
    border: 1px solid color-mix(in srgb, var(--accent) 30%, transparent);
  }
  .update-body {
    font-size: 12px;
    color: var(--text-secondary);
    line-height: 1.4;
  }
  .actions {
    display: flex;
    gap: 8px;
    margin-top: 8px;
  }
  .btn {
    padding: 10px 20px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    border: none;
    transition: opacity 0.15s;
  }
  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .btn-primary {
    background: var(--accent);
    color: white;
  }
  .btn-primary:hover:not(:disabled) {
    opacity: 0.9;
  }
  .btn-ghost {
    background: transparent;
    color: var(--text-secondary);
    border: 1px solid var(--border);
  }
  .btn-ghost:hover:not(:disabled) {
    background: var(--bg-input);
  }
  .btn-small {
    padding: 6px 14px;
    font-size: 12px;
  }
</style>
