<script lang="ts">
  import "./app.css";
  import Nav from "./lib/Nav.svelte";
  import Dashboard from "./lib/Dashboard.svelte";
  import Cluster from "./lib/Cluster.svelte";
  import Chat from "./lib/Chat.svelte";
  import Models from "./lib/Models.svelte";
  import Benchmark from "./lib/Benchmark.svelte";
  import Topology from "./lib/Topology.svelte";
  import MultiModel from "./lib/MultiModel.svelte";
  import Discovery from "./lib/Discovery.svelte";
  import WebDashboard from "./lib/WebDashboard.svelte";
  import Plugins from "./lib/Plugins.svelte";
  import Friends from "./lib/Friends.svelte";
  import OnboardingWizard from "./lib/OnboardingWizard.svelte";
  import Settings from "./lib/Settings.svelte";
  import ActivityLogs from "./lib/ActivityLogs.svelte";
  import { Toast, toastStore } from "./lib/ui";
  import { applyTheme } from "./lib/stores/settings-store";
  import { settingsStore } from "./lib/stores";
  import { joinCluster, updateTrayStatus, addRecentCluster, checkForUpdates, installUpdate, initNotifications, notify } from "./lib/api";
  import { clusterStore } from "./lib/stores";
  import { listen } from "@tauri-apps/api/event";
  import { onDestroy, onMount } from "svelte";
  import type { Page } from "./lib/types";

  let currentPage = $state<Page>("dashboard");
  let traySource = $state<string | null>(null);
  let grafanaToggle = $state(0);
  let connectAddr = $state<string | null>(null);
  // 5.1: Show onboarding wizard on first launch
  let showWizard = $state(!localStorage.getItem("distllm_onboarded"));

  const unlisten = listen<string>("navigate", (e) => {
    traySource = e.payload;
    if (e.payload === "create_cluster" || e.payload === "join_cluster") {
      currentPage = "cluster";
    } else if (e.payload === "show") {
      currentPage = "dashboard";
    }
  });

  // 5.8: Handle deep link connect events from tray recent clusters and protocol handler
  const unlistenDeepLink = listen<string>("deep-link-connect", (e) => {
    const url = e.payload;
    // Parse distllm://connect/<host>:<port>/<invite_code>
    const match = url.match(/^distllm:\/\/connect\/([^:/]+):(\d+)\/(.+)$/);
    if (match) {
      const [, host, port, code] = match;
      connectAddr = `${host}:${port}`;
      currentPage = "cluster";
      traySource = "join_cluster";
      // Auto-join after a short delay to let the cluster page mount
      setTimeout(async () => {
        try {
          await joinCluster(host, parseInt(port));
          await addRecentCluster(`${host}:${port}`);
          toastStore.success(`Joined cluster at ${host}:${port}`);
          updateTrayStatus(true, 1, `http://${host}:${port}`);
        } catch (err) {
          toastStore.error(`Failed to join: ${err}`);
        }
        connectAddr = null;
      }, 300);
    }
  });

  // 5.7: Update tray when cluster status changes
  let unsubCluster = clusterStore.subscribe((d) => {
    const c = d.cluster;
    if (c?.running) {
      const nodeCount = c.nodes?.length ?? 1;
      updateTrayStatus(true, nodeCount, c.coordinator_addr ?? undefined);
    }
  });

  // 4.10: Keyboard shortcuts
  const pageMap: Record<string, Page> = {
    "1": "dashboard",
    "2": "cluster",
    "3": "chat",
    "4": "models",
    "5": "benchmark",
    "6": "topology",
    "7": "multimodel",
    "8": "discovery",
    "9": "plugins",
  };

  function handleKeydown(e: KeyboardEvent) {
    const isCtrl = e.ctrlKey || e.metaKey;

    // Ctrl+1-9: Navigate to page
    if (isCtrl && !e.shiftKey && !e.altKey && !e.key.match(/[^1-9]/)) {
      const target = pageMap[e.key];
      if (target) {
        e.preventDefault();
        currentPage = target;
        traySource = null;
      }
    }

    // Ctrl+G: Toggle Grafana
    if (isCtrl && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "g") {
      e.preventDefault();
      grafanaToggle++;
    }

    // Ctrl+,: Open settings
    if (isCtrl && e.key === ",") {
      e.preventDefault();
      currentPage = "settings";
      traySource = null;
    }

    // Escape: Dismiss errors (triggered via custom event)
    if (e.key === "Escape") {
      window.dispatchEvent(new CustomEvent("dismiss-errors"));
    }
  }

  // ── Periodic update check ──────────────────────────────────────
  let updateIntervalId: number | undefined;
  const UPDATE_CHECK_INTERVAL_MS = 1000 * 60 * 60 * 24; // once per day

  async function performUpdateCheck() {
    try {
      const update = await checkForUpdates();
      if (update?.available) {
        const settings = settingsStore.getSnapshot();
        if (settings.notifications.updateAvailable) {
          // In-app toast
          toastStore.info(
            `Update v${update.version} available — go to Settings to install`,
          );
          // Native notification
          if (settings.notifications.native) {
            notify(
              "Update Available",
              `Distributed LLM v${update.version} is ready to install`,
            );
          }
        }
      }
    } catch {
      // Silent fail on startup check
    }
  }

  // ── Native OS notification helpers ────────────────────────────
  async function sendNativeNotification(title: string, body: string) {
    const settings = settingsStore.getSnapshot();
    if (settings.notifications.native) {
      notify(title, body);
    }
  }

  // ── Server-sent events for cluster notifications ──────────────
  let previousNodeCount = 0;

  function checkClusterChanges() {
    const state = clusterStore.getSnapshot();
    const cluster = state.cluster;
    if (!cluster?.running) {
      previousNodeCount = 0;
      return;
    }

    const nodeCount = cluster.nodes?.length ?? 0;
    const settings = settingsStore.getSnapshot();

    if (settings.notifications.clusterEvents && previousNodeCount > 0) {
      if (nodeCount > previousNodeCount) {
        const joined = nodeCount - previousNodeCount;
        const msg = `${joined} node${joined > 1 ? "s have" : " has"} joined the cluster`;
        toastStore.success(msg);
        sendNativeNotification("Cluster Node Joined", msg);
      } else if (nodeCount < previousNodeCount) {
        const left = previousNodeCount - nodeCount;
        const msg = `${left} node${left > 1 ? "s have" : " has"} left the cluster`;
        toastStore.warning(msg);
        sendNativeNotification("Cluster Node Left", msg);
      }
    }

    previousNodeCount = nodeCount;
  }

  // Watch cluster store for changes
  let unsubClusterChanges = clusterStore.subscribe(() => {
    checkClusterChanges();
  });

  // ── Error event handling ──────────────────────────────────────
  const unlistenProcessCrashed = listen<string>("process-crashed", async (e) => {
    const processName = e.payload;
    toastStore.error(`${processName} process crashed unexpectedly`);
    const settings = settingsStore.getSnapshot();
    if (settings.notifications.errors) {
      if (settings.notifications.native) {
        notify("Process Crashed", `The ${processName} process exited unexpectedly`);
      }
    }
  });

  const unlistenClusterStopped = listen("cluster-stopped", async () => {
    previousNodeCount = 0;
    const settings = settingsStore.getSnapshot();
    if (settings.notifications.clusterEvents && settings.notifications.native) {
      notify("Cluster Stopped", "The cluster has been shut down");
    }
  });

  onMount(() => {
    // Apply saved theme on load
    applyTheme();
    window.addEventListener("keydown", handleKeydown);

    // Listen for system theme changes (for "auto" mode)
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener("change", () => applyTheme());

    // Initialize native notifications on startup
    initNotifications().then((granted) => {
      if (!granted) {
        console.warn("Native notification permission not granted");
      }
    });

    // Periodic update check (once per day, first check delayed 30s after startup)
    setTimeout(() => {
      performUpdateCheck();
    }, 30000);

    // Clear any existing interval and start a new one
    if (updateIntervalId !== undefined) {
      clearInterval(updateIntervalId);
    }
    updateIntervalId = window.setInterval(performUpdateCheck, UPDATE_CHECK_INTERVAL_MS);
  });

  onDestroy(() => {
    unlisten.then((fn) => fn());
    unlistenDeepLink.then((fn) => fn());
    unlistenProcessCrashed.then((fn) => fn());
    unlistenClusterStopped.then((fn) => fn());
    unsubCluster();
    unsubClusterChanges();
    window.removeEventListener("keydown", handleKeydown);
    if (updateIntervalId !== undefined) {
      clearInterval(updateIntervalId);
    }
  });

  function navigate(page: Page) {
    currentPage = page;
    traySource = null;
  }

  function onWizardComplete() {
    showWizard = false;
  }
</script>

{#if showWizard}
  <OnboardingWizard oncomplete={onWizardComplete} />
{/if}

<div class="app-layout">
  <Nav {currentPage} onNavigate={navigate} />
  <main class="main-content" class:chat-active={currentPage === "chat"}>
    {#if currentPage === "dashboard"}
      <Dashboard {grafanaToggle} />
    {:else if currentPage === "cluster"}
      <Cluster initialAction={traySource} {connectAddr} />
    {:else if currentPage === "chat"}
      <Chat />
    {:else if currentPage === "models"}
      <Models />
    {:else if currentPage === "benchmark"}
      <Benchmark />
    {:else if currentPage === "topology"}
      <Topology />
    {:else if currentPage === "multimodel"}
      <MultiModel />
    {:else if currentPage === "discovery"}
      <Discovery />
    {:else if currentPage === "webdashboard"}
      <WebDashboard />
    {:else if currentPage === "plugins"}
      <Plugins />
    {:else if currentPage === "friends"}
      <Friends />
    {:else if currentPage === "logs"}
      <ActivityLogs />
    {:else if currentPage === "settings"}
      <Settings />
    {/if}
  </main>
  <Toast />
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
  .main-content.chat-active {
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
</style>
