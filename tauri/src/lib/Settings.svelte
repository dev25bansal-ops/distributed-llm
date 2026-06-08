<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { settingsStore } from "./stores";
  import { applyTheme } from "./stores/settings-store";
  import { Card, Input, ErrorBanner, toastStore } from "./ui";
  import type { AppSettings } from "./stores/settings-store";

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
    },
    pythonPath: "",
  });
  let error = $state<string | null>(null);
  let unsubscribe: (() => void) | undefined;

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
</script>

<div class="settings-page">
  <h1 class="page-title">Settings</h1>

  <ErrorBanner message={error ?? ""} ondismiss={() => (error = null)} />

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

  <Card title="Integrations">
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

  <Card title="Notifications">
    <div class="notif-grid">
      <label class="toggle-label">
        <input
          type="checkbox"
          bind:checked={settings.notifications.clusterEvents}
          class="toggle-input"
        />
        <span class="toggle-switch"></span>
        <span>Cluster events</span>
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
    </div>
  </Card>

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
  }
  .btn-primary {
    background: var(--accent);
    color: white;
  }
  .btn-primary:hover {
    opacity: 0.9;
  }
  .btn-ghost {
    background: transparent;
    color: var(--text-secondary);
    border: 1px solid var(--border);
  }
  .btn-ghost:hover {
    background: var(--bg-input);
  }
</style>
