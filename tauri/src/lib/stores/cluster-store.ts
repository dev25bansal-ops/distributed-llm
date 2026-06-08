import { getClusterStatus, getGpuMetrics, getSystemInfo } from "../api";
import type { ClusterStatus, GpuInfo, SystemInfo } from "../types";

type ClusterStoreData = {
  cluster: ClusterStatus | null;
  gpus: GpuInfo[];
  sysInfo: SystemInfo | null;
  loading: boolean;
  error: string | null;
};

type Listener = (data: ClusterStoreData) => void;

const DEFAULT_DATA: ClusterStoreData = {
  cluster: null,
  gpus: [],
  sysInfo: null,
  loading: true,
  error: null,
};

let data: ClusterStoreData = { ...DEFAULT_DATA };
let listeners: Listener[] = [];
let pollTimer: ReturnType<typeof setInterval> | undefined;
let pollInterval = 3000;
let consecutiveErrors = 0;

function notify() {
  for (const fn of listeners) {
    fn({ ...data });
  }
}

async function fetchAll() {
  try {
    const [cluster, gpus, sysInfo] = await Promise.all([
      getClusterStatus(),
      getGpuMetrics(),
      getSystemInfo(),
    ]);
    data = { ...data, cluster, gpus, sysInfo, loading: false, error: null };
    consecutiveErrors = 0;
    pollInterval = 3000; // Reset backoff
  } catch (e: unknown) {
    consecutiveErrors++;
    // Exponential backoff: 3s, 6s, 12s, max 30s
    if (consecutiveErrors > 1) {
      pollInterval = Math.min(3000 * Math.pow(2, consecutiveErrors - 1), 30000);
      restartPoll();
    }
    data = { ...data, loading: false, error: String(e) };
  }
  notify();
}

function restartPoll() {
  if (pollTimer) {
    clearInterval(pollTimer);
  }
  pollTimer = setInterval(fetchAll, pollInterval);
}

function startPoll() {
  fetchAll();
  restartPoll();
}

function stopPoll() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = undefined;
  }
}

// Pause polling when window is hidden
function handleVisibility() {
  if (document.hidden) {
    stopPoll();
  } else {
    startPoll();
  }
}

if (typeof document !== "undefined") {
  document.addEventListener("visibilitychange", handleVisibility);
}

export const clusterStore = {
  subscribe(fn: Listener) {
    listeners.push(fn);
    fn({ ...data });
    // Start polling on first subscriber
    if (listeners.length === 1) {
      startPoll();
    }
    return () => {
      listeners = listeners.filter((l) => l !== fn);
      if (listeners.length === 0) {
        stopPoll();
      }
    };
  },

  /** Force an immediate refresh */
  async refresh() {
    await fetchAll();
  },

  /** Get current snapshot without subscribing */
  getSnapshot(): ClusterStoreData {
    return { ...data };
  },
};
