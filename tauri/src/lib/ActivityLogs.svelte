<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { logStore } from "./stores";
  import { Card, Input, ErrorBanner, toastStore } from "./ui";
  import type { LogEntry, LogLevel } from "./stores/log-store";

  let logs = $state<LogEntry[]>([]);
  let filter = $state<LogLevel | "all">("all");
  let search = $state("");
  let error = $state<string | null>(null);
  let logContainer = $state<HTMLElement | null>(null);
  let unsubscribe: (() => void) | undefined;

  onMount(() => {
    unsubscribe = logStore.subscribe((l) => {
      logs = l;
    });
  });

  onDestroy(() => unsubscribe?.());

  let filteredLogs = $derived.by(() => {
    let result = logs;
    if (filter !== "all") {
      result = result.filter((l) => l.level === filter);
    }
    if (search.trim()) {
      const q = search.toLowerCase();
      result = result.filter(
        (l) =>
          l.message.toLowerCase().includes(q) ||
          l.category.toLowerCase().includes(q),
      );
    }
    return result;
  });

  let counts = $derived({
    all: logs.length,
    info: logs.filter((l) => l.level === "info").length,
    warn: logs.filter((l) => l.level === "warn").length,
    error: logs.filter((l) => l.level === "error").length,
  });

  function fmtTime(ts: number): string {
    return new Date(ts).toLocaleTimeString();
  }

  function fmtDate(ts: number): string {
    return new Date(ts).toLocaleDateString();
  }

  function handleExport() {
    const content = logStore.export();
    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `distllm-logs-${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    toastStore.success("Logs exported");
  }

  function handleClear() {
    if (!window.confirm("Clear all logs?")) return;
    logStore.clear();
    toastStore.info("Logs cleared");
  }

  // Auto-scroll to bottom when new logs arrive
  $effect(() => {
    if (logs.length > 0 && logContainer) {
      logContainer.scrollTop = logContainer.scrollHeight;
    }
  });
</script>

<div class="logs-page">
  <div class="logs-header">
    <h1 class="page-title">Activity Logs</h1>
    <div class="header-actions">
      <button class="btn btn-ghost" onclick={handleExport}>Export</button>
      <button class="btn btn-ghost danger" onclick={handleClear}>Clear</button>
    </div>
  </div>

  <ErrorBanner message={error ?? ""} ondismiss={() => (error = null)} />

  <div class="logs-toolbar">
    <div class="filter-tabs">
      <button
        class="filter-tab"
        class:active={filter === "all"}
        onclick={() => (filter = "all")}
      >
        All ({counts.all})
      </button>
      <button
        class="filter-tab"
        class:active={filter === "info"}
        onclick={() => (filter = "info")}
      >
        Info ({counts.info})
      </button>
      <button
        class="filter-tab"
        class:active={filter === "warn"}
        onclick={() => (filter = "warn")}
      >
        Warn ({counts.warn})
      </button>
      <button
        class="filter-tab"
        class:active={filter === "error"}
        onclick={() => (filter = "error")}
      >
        Error ({counts.error})
      </button>
    </div>
    <input
      class="search-input"
      type="text"
      placeholder="Search logs..."
      bind:value={search}
    />
  </div>

  <div class="log-container" bind:this={logContainer}>
    {#if filteredLogs.length === 0}
      <div class="empty-state">
        {logs.length === 0 ? "No logs yet" : "No logs match current filter"}
      </div>
    {:else}
      {#each filteredLogs as entry (entry.id)}
        <div class="log-entry {entry.level}">
          <span class="log-time">{fmtTime(entry.timestamp)}</span>
          <span class="log-level">{entry.level.toUpperCase()}</span>
          <span class="log-category">{entry.category}</span>
          <span class="log-message">{entry.message}</span>
        </div>
      {/each}
    {/if}
  </div>
</div>

<style>
  .logs-page {
    display: flex;
    flex-direction: column;
    height: calc(100vh - 48px);
  }
  .logs-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }
  .header-actions {
    display: flex;
    gap: 8px;
  }
  .btn {
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    border: none;
  }
  .btn-ghost {
    background: transparent;
    color: var(--text-secondary);
    border: 1px solid var(--border);
  }
  .btn-ghost:hover {
    background: var(--bg-input);
  }
  .btn-ghost.danger {
    color: var(--danger);
    border-color: color-mix(in srgb, var(--danger) 30%, transparent);
  }
  .btn-ghost.danger:hover {
    background: color-mix(in srgb, var(--danger) 10%, transparent);
  }
  .logs-toolbar {
    display: flex;
    gap: 12px;
    align-items: center;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }
  .filter-tabs {
    display: flex;
    gap: 4px;
    background: var(--bg-input);
    border-radius: 8px;
    padding: 3px;
  }
  .filter-tab {
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary);
    background: transparent;
    border: none;
    cursor: pointer;
    transition: all 0.15s;
  }
  .filter-tab.active {
    background: var(--accent);
    color: white;
  }
  .filter-tab:hover:not(.active) {
    background: color-mix(in srgb, var(--accent) 15%, transparent);
  }
  .search-input {
    flex: 1;
    min-width: 200px;
    padding: 7px 12px;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-size: 13px;
  }
  .search-input:focus {
    outline: none;
    border-color: var(--accent);
  }
  .search-input::placeholder {
    color: var(--text-muted);
  }
  .log-container {
    flex: 1;
    overflow-y: auto;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-family: var(--font-mono);
    font-size: 12px;
  }
  .log-entry {
    display: flex;
    gap: 12px;
    padding: 6px 14px;
    border-bottom: 1px solid color-mix(in srgb, var(--border) 30%, transparent);
    align-items: baseline;
  }
  .log-entry:last-child {
    border-bottom: none;
  }
  .log-entry:hover {
    background: color-mix(in srgb, var(--accent) 5%, transparent);
  }
  .log-entry.error {
    background: color-mix(in srgb, var(--danger) 5%, transparent);
  }
  .log-entry.warn {
    background: color-mix(in srgb, var(--warning) 5%, transparent);
  }
  .log-time {
    color: var(--text-muted);
    flex-shrink: 0;
    width: 80px;
  }
  .log-level {
    flex-shrink: 0;
    width: 50px;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
  }
  .log-entry.info .log-level {
    color: var(--accent);
  }
  .log-entry.warn .log-level {
    color: var(--warning);
  }
  .log-entry.error .log-level {
    color: var(--danger);
  }
  .log-category {
    flex-shrink: 0;
    width: 100px;
    color: var(--text-secondary);
  }
  .log-message {
    color: var(--text);
    word-break: break-word;
    flex: 1;
  }
  .empty-state {
    color: var(--text-muted);
    font-size: 13px;
    padding: 24px;
    text-align: center;
    font-family: var(--font-sans);
  }
</style>
