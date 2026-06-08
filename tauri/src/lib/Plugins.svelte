<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { pluginStore, logStore } from "./stores";
  import { ErrorBanner } from "./ui";
  import type { PluginConfig, PluginKind } from "./types";

  let plugins = $state<PluginConfig[]>([]);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let editing = $state<PluginConfig | null>(null);
  let showForm = $state(false);

  let unsubscribe: (() => void) | undefined;

  onMount(() => {
    unsubscribe = pluginStore.subscribe((d) => {
      plugins = d.plugins;
      loading = d.loading;
      error = d.error;
    });
    pluginStore.refresh();
  });

  onDestroy(() => unsubscribe?.());

  function newPlugin(kind: PluginKind) {
    editing = {
      id: `plugin-${Date.now()}`,
      name: "",
      kind,
      enabled: true,
      endpoint: "",
      api_key: "",
      extra: {},
      created_at: Date.now(),
    };
    showForm = true;
  }

  function editPlugin(p: PluginConfig) {
    editing = { ...p };
    showForm = true;
  }

  async function save() {
    if (!editing || !editing.name) return;
    const isNew = !plugins.some((p) => p.id === editing!.id);
    if (isNew) {
      await pluginStore.add(editing);
      logStore.info("plugins", `Plugin added: ${editing.name} (${editing.kind})`);
    } else {
      await pluginStore.update(editing);
      logStore.info("plugins", `Plugin updated: ${editing.name}`);
    }
    showForm = false;
    editing = null;
  }

  async function remove(id: string) {
    if (window.confirm("Remove this plugin?")) {
      const p = plugins.find((p) => p.id === id);
      await pluginStore.remove(id);
      logStore.info("plugins", `Plugin removed: ${p?.name ?? id}`);
    }
  }

  async function toggle(p: PluginConfig) {
    await pluginStore.update({ ...p, enabled: !p.enabled });
    logStore.info("plugins", `Plugin ${p.enabled ? "disabled" : "enabled"}: ${p.name}`);
  }

  async function testConnection(id: string) {
    const ok = await pluginStore.test(id);
    error = ok ? null : "Connection test failed";
    logStore.info("plugins", `Plugin test ${ok ? "passed" : "failed"}: ${id}`);
  }

  function kindIcon(kind: PluginKind): string {
    switch (kind) {
      case "backend": return "⚙";
      case "auth": return "🔒";
      case "monitoring": return "📊";
    }
  }

  function kindLabel(kind: PluginKind): string {
    switch (kind) {
      case "backend": return "Model Backend";
      case "auth": return "Auth Provider";
      case "monitoring": return "Monitoring";
    }
  }
</script>

<div class="plugins-page">
  <h1 class="page-title">Plugins</h1>

  <ErrorBanner message={error ?? ""} ondismiss={() => (error = null)} />

  <div class="plugin-types">
    <button class="type-card" onclick={() => newPlugin("backend")}>
      <span class="type-icon">⚙</span>
      <span class="type-label">Add Backend</span>
      <span class="type-desc">Custom model backends (vLLM, Ollama, etc.)</span>
    </button>
    <button class="type-card" onclick={() => newPlugin("auth")}>
      <span class="type-icon">🔒</span>
      <span class="type-label">Add Auth</span>
      <span class="type-desc">Authentication providers (OAuth, API key, etc.)</span>
    </button>
    <button class="type-card" onclick={() => newPlugin("monitoring")}>
      <span class="type-icon">📊</span>
      <span class="type-label">Add Monitoring</span>
      <span class="type-desc">Monitoring integrations (Prometheus, Datadog, etc.)</span>
    </button>
  </div>

  {#if showForm && editing}
    <div class="card form-card">
      <h2 class="card-title">{plugins.some(p => p.id === editing!.id) ? "Edit" : "New"} Plugin</h2>
      <div class="form-grid">
        <div class="form-row">
          <label class="form-label" for="p-name">Name</label>
          <input id="p-name" class="input" bind:value={editing.name} placeholder="My Plugin" />
        </div>
        <div class="form-row">
          <label class="form-label" for="p-type">Type</label>
          <input id="p-type" class="input" value={kindLabel(editing.kind)} readonly />
        </div>
        <div class="form-row">
          <label class="form-label" for="p-endpoint">Endpoint URL</label>
          <input id="p-endpoint" class="input" bind:value={editing.endpoint} placeholder="http://localhost:11434" />
        </div>
        <div class="form-row">
          <label class="form-label" for="p-apikey">API Key (optional)</label>
          <input id="p-apikey" class="input" type="password" bind:value={editing.api_key} placeholder="sk-..." />
        </div>
      </div>
      <div class="form-actions">
        <button class="btn btn-primary" onclick={save} disabled={!editing.name}>Save</button>
        <button class="btn btn-ghost" onclick={() => { showForm = false; editing = null; }}>Cancel</button>
      </div>
    </div>
  {/if}

  <div class="card">
    <h2 class="card-title">Installed Plugins</h2>
    {#if plugins.length === 0}
      <div class="empty-state">No plugins configured. Add a backend, auth provider, or monitoring integration above.</div>
    {:else}
      <div class="plugin-list">
        {#each plugins as p (p.id)}
          <div class="plugin-item" class:disabled={!p.enabled}>
            <div class="plugin-main">
              <span class="plugin-icon">{kindIcon(p.kind)}</span>
              <div class="plugin-info">
                <span class="plugin-name">{p.name}</span>
                <span class="plugin-meta">{kindLabel(p.kind)} · {p.endpoint || "No endpoint"}</span>
              </div>
            </div>
            <div class="plugin-actions">
              <button class="btn-sm btn-ghost" onclick={() => testConnection(p.id)} title="Test connection">Test</button>
              <button class="btn-sm btn-ghost" onclick={() => toggle(p)} title={p.enabled ? "Disable" : "Enable"}>
                {p.enabled ? "On" : "Off"}
              </button>
              <button class="btn-sm btn-ghost" onclick={() => editPlugin(p)} title="Edit">Edit</button>
              <button class="btn-sm btn-danger" onclick={() => remove(p.id)} title="Remove">✕</button>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>

<style>
  .plugins-page { max-width: 800px; }
  .plugin-types { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px; }
  .type-card {
    display: flex; flex-direction: column; align-items: center; gap: 6px;
    padding: 20px 12px; background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 10px; cursor: pointer; transition: all 0.15s; text-align: center;
  }
  .type-card:hover { border-color: var(--accent); background: var(--bg-input); }
  .type-icon { font-size: 28px; }
  .type-label { font-weight: 600; font-size: 13px; }
  .type-desc { font-size: 11px; color: var(--text-muted); }
  .form-card { margin-bottom: 16px; }
  .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px; }
  .form-actions { display: flex; gap: 8px; }
  .plugin-list { display: flex; flex-direction: column; gap: 2px; }
  .plugin-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px; border-radius: 8px; transition: background 0.15s;
  }
  .plugin-item:hover { background: var(--bg-input); }
  .plugin-item.disabled { opacity: 0.5; }
  .plugin-main { display: flex; align-items: center; gap: 12px; }
  .plugin-icon { font-size: 20px; width: 32px; text-align: center; }
  .plugin-info { display: flex; flex-direction: column; }
  .plugin-name { font-weight: 600; font-size: 13px; }
  .plugin-meta { font-size: 11px; color: var(--text-muted); }
  .plugin-actions { display: flex; gap: 4px; align-items: center; }
  .btn-sm { padding: 4px 10px; font-size: 11px; border-radius: 6px; background: none; border: 1px solid var(--border); color: var(--text-secondary); cursor: pointer; }
  .btn-sm:hover { background: var(--bg-input); color: var(--text-primary); }
  .btn-sm.btn-danger { border-color: var(--danger); color: var(--danger); }
  .btn-sm.btn-danger:hover { background: color-mix(in srgb, var(--danger) 15%, transparent); }
</style>
