<script lang="ts">
  import "./app.css";
  import Nav from "./lib/Nav.svelte";
  import Dashboard from "./lib/Dashboard.svelte";
  import Cluster from "./lib/Cluster.svelte";
  import Models from "./lib/Models.svelte";
  import Friends from "./lib/Friends.svelte";
  import { listen } from "@tauri-apps/api/event";
  import type { Page } from "./lib/types";

  let currentPage = $state<Page>("dashboard");
  let traySource = $state<string | null>(null);

  const unlisten = listen<string>("navigate", (e) => {
    traySource = e.payload;
    if (e.payload === "create_cluster" || e.payload === "join_cluster") {
      currentPage = "cluster";
    } else if (e.payload === "show") {
      currentPage = "dashboard";
    }
  });

  function navigate(page: Page) {
    currentPage = page;
    traySource = null;
  }
</script>

<div class="app-layout">
  <Nav {currentPage} onNavigate={navigate} />
  <main class="main-content">
    {#if currentPage === "dashboard"}
      <Dashboard />
    {:else if currentPage === "cluster"}
      <Cluster initialAction={traySource} />
    {:else if currentPage === "models"}
      <Models />
    {:else if currentPage === "friends"}
      <Friends />
    {/if}
  </main>
</div>

<style>
  .app-layout {
    display: flex;
    height: 100vh;
    width: 100vw;
  }
  .main-content {
    flex: 1;
    overflow-y: auto;
    padding: 24px 32px;
  }
</style>
