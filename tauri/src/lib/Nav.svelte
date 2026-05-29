<script lang="ts">
  import type { Page } from "./types";

  let { currentPage, onNavigate }: { currentPage: Page; onNavigate: (p: Page) => void } = $props();

  const items: { id: Page; label: string; icon: string }[] = [
    { id: "dashboard", label: "Dashboard", icon: "◉" },
    { id: "cluster", label: "Cluster", icon: "⚡" },
    { id: "models", label: "Models", icon: "▤" },
    { id: "friends", label: "Friends", icon: "⊕" },
  ];
</script>

<nav class="sidebar">
  <div class="logo">
    <span class="logo-icon">◆</span>
    <span class="logo-text">Distributed<br />LLM</span>
  </div>

  <div class="nav-items">
    {#each items as item}
      <button
        class="nav-item"
        class:active={currentPage === item.id}
        onclick={() => onNavigate(item.id)}
      >
        <span class="nav-icon">{item.icon}</span>
        <span class="nav-label">{item.label}</span>
      </button>
    {/each}
  </div>

  <div class="sidebar-footer">
    <span class="version">v0.4.0</span>
  </div>
</nav>

<style>
  .sidebar {
    width: 180px;
    min-width: 180px;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    padding: 0;
  }
  .logo {
    padding: 20px 16px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .logo-icon {
    font-size: 24px;
    color: var(--accent);
  }
  .logo-text {
    font-size: 13px;
    font-weight: 700;
    line-height: 1.3;
    color: var(--text-primary);
  }
  .nav-items {
    flex: 1;
    padding: 12px 8px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    border-radius: 8px;
    background: transparent;
    color: var(--text-secondary);
    font-size: 14px;
    transition: all 0.15s;
  }
  .nav-item:hover {
    background: var(--bg-card);
    color: var(--text-primary);
  }
  .nav-item.active {
    background: var(--accent);
    color: #fff;
  }
  .nav-icon {
    font-size: 16px;
    width: 20px;
    text-align: center;
  }
  .sidebar-footer {
    padding: 12px 16px;
    border-top: 1px solid var(--border);
  }
  .version {
    font-size: 11px;
    color: var(--text-muted);
    font-family: var(--font-mono);
  }
</style>
