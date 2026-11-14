<script lang="ts">
  import type { Page } from "./types";

  let { currentPage, onNavigate }: { currentPage: Page; onNavigate: (p: Page) => void } = $props();

  // One stroke style, 16px, currentColor. Emoji and geometric glyphs rendered
  // differently on every platform (and one of them was a corrupted character),
  // so the nav had no consistent icon language.
  const items: { id: Page; label: string; d: string }[] = [
    { id: "dashboard", label: "Dashboard", d: "M3 3h6v6H3zM11 3h6v6h-6zM3 11h6v6H3zM11 11h6v6h-6z" },
    { id: "cluster", label: "Cluster", d: "M4 4h6v4H4zM14 4h6v4h-6zM4 12h6v4H4zM14 12h6v4h-6zM10 6h4M10 14h4" },
    { id: "chat", label: "Chat", d: "M3 5h18v10H9l-4 4v-4H3z" },
    { id: "models", label: "Models", d: "M4 6l8-3 8 3-8 3zM4 6v8l8 3 8-3V6M12 9v8" },
    { id: "benchmark", label: "Benchmark", d: "M4 20V10M10 20V4M16 20v-7M22 20V8" },
    { id: "topology", label: "Topology", d: "M12 3v4M6 21v-4l6-6 6 6v4M4 7h4M16 7h4" },
    { id: "multimodel", label: "Multi-Model", d: "M3 3h8v8H3zM13 13h8v8h-8zM13 3h8v8h-8z" },
    { id: "discovery", label: "Discovery", d: "M12 21a9 9 0 100-18 9 9 0 000 18zM12 8v4l3 2" },
    { id: "webdashboard", label: "Web UI", d: "M3 5h18v14H3zM3 9h18M6 7h.01M9 7h.01" },
    { id: "plugins", label: "Plugins", d: "M9 3v4M15 3v4M6 7h12v6a6 6 0 01-12 0zM12 19v2" },
    { id: "friends", label: "Friends", d: "M9 11a4 4 0 100-8 4 4 0 000 8zM2 21v-2a5 5 0 015-5h4a5 5 0 015 5v2M17 11a3 3 0 100-6" },
    { id: "logs", label: "Logs", d: "M4 6h16M4 12h16M4 18h10" },
    { id: "settings", label: "Settings", d: "M12 15a3 3 0 100-6 3 3 0 000 6zM19 12a7 7 0 00-.1-1l2-1.6-2-3.4-2.4 1a7 7 0 00-1.7-1L14.5 3h-5l-.3 2.6a7 7 0 00-1.7 1l-2.4-1-2 3.4L5.1 11a7 7 0 000 2l-2 1.6 2 3.4 2.4-1a7 7 0 001.7 1l.3 2.6h5l.3-2.6a7 7 0 001.7-1l2.4 1 2-3.4-2-1.6a7 7 0 00.1-1z" },
  ];
</script>

<nav class="sidebar" aria-label="Main">
  <a class="logo" href="#" onclick={(e) => { e.preventDefault(); onNavigate("dashboard"); }}>
    <span class="logo-mark" aria-hidden="true">
      <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><path d="M3 6l9-3 9 3-9 3zM3 12l9-3 9 3-9 3zM3 18l9-3 9 3-9 3z"/></svg>
    </span>
    <span class="logo-text">Distributed<br />LLM</span>
  </a>

  <div class="nav-items">
    {#each items as item}
      <button
        class="nav-item"
        class:active={currentPage === item.id}
        onclick={() => onNavigate(item.id)}
        aria-current={currentPage === item.id ? "page" : undefined}
      >
        <svg class="nav-icon" viewBox="0 0 24 24" width="16" height="16"
             fill="none" stroke="currentColor" stroke-width="1.6"
             stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d={item.d} />
        </svg>
        <span class="nav-label">{item.label}</span>
      </button>
    {/each}
  </div>

  <div class="sidebar-footer">
    <span class="version">v0.6.2.2</span>
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
  .logo-mark {
    display: grid;
    place-items: center;
    width: 26px;
    height: 26px;
    border-radius: 4px;
    background: var(--accent);
    color: var(--bg-primary);
    flex: none;
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
    width: 20px;
    height: 20px;
    flex: none;
    opacity: 0.75;
  }
  .nav-item.active .nav-icon { opacity: 1; }
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
