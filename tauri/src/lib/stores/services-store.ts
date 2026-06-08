import type { WebDashboardConfig, WebDashboardStatus, DiscoveredService } from "../types";
import {
  getWebDashboardConfig,
  setWebDashboardConfig,
  getWebDashboardStatus,
  startWebDashboard,
  stopWebDashboard,
  getDiscoveredServices,
  startDiscovery,
  stopDiscovery,
  getDiscoveryStatus,
} from "../api";

// Web Dashboard store
type WebDashData = {
  config: WebDashboardConfig;
  status: WebDashboardStatus;
  loading: boolean;
  error: string | null;
};

type WebDashListener = (data: WebDashData) => void;

let webData: WebDashData = {
  config: { enabled: false, port: 8080, auth_required: true, auth_token: "", cors_origins: ["*"] },
  status: { running: false, url: "", connections: 0 },
  loading: true,
  error: null,
};

let webListeners: WebDashListener[] = [];

function notifyWeb() {
  for (const fn of webListeners) {
    fn({ ...webData });
  }
}

export const webDashboardStore = {
  subscribe(fn: WebDashListener) {
    webListeners.push(fn);
    fn({ ...webData });
    return () => {
      webListeners = webListeners.filter((l) => l !== fn);
    };
  },

  async refresh() {
    webData.loading = true;
    notifyWeb();
    try {
      const [config, status] = await Promise.all([
        getWebDashboardConfig().catch(() => webData.config),
        getWebDashboardStatus().catch(() => webData.status),
      ]);
      webData = { ...webData, config, status, loading: false, error: null };
    } catch (e: unknown) {
      webData = { ...webData, loading: false, error: String(e) };
    }
    notifyWeb();
  },

  async updateConfig(config: WebDashboardConfig) {
    webData.config = config;
    notifyWeb();
    try {
      await setWebDashboardConfig(config);
    } catch (e: unknown) {
      webData.error = String(e);
      notifyWeb();
    }
  },

  async start() {
    try {
      webData.status = await startWebDashboard();
      webData.error = null;
    } catch (e: unknown) {
      webData.error = String(e);
    }
    notifyWeb();
  },

  async stop() {
    try {
      await stopWebDashboard();
      webData.status = { running: false, url: "", connections: 0 };
      webData.error = null;
    } catch (e: unknown) {
      webData.error = String(e);
    }
    notifyWeb();
  },
};

// Discovery store
type DiscData = {
  services: DiscoveredService[];
  active: boolean;
  loading: boolean;
  error: string | null;
};

type DiscListener = (data: DiscData) => void;

let discData: DiscData = {
  services: [],
  active: false,
  loading: true,
  error: null,
};

let discListeners: DiscListener[] = [];
let discPollTimer: ReturnType<typeof setInterval> | undefined;

function notifyDisc() {
  for (const fn of discListeners) {
    fn({ ...discData, services: [...discData.services] });
  }
}

async function pollDiscovery() {
  try {
    const [services, status] = await Promise.all([
      getDiscoveredServices().catch(() => []),
      getDiscoveryStatus().catch(() => ({ active: false, service_count: 0 })),
    ]);
    discData = { ...discData, services, active: status.active, loading: false, error: null };
  } catch (e: unknown) {
    discData = { ...discData, loading: false, error: String(e) };
  }
  notifyDisc();
}

export const discoveryStore = {
  subscribe(fn: DiscListener) {
    discListeners.push(fn);
    fn({ ...discData, services: [...discData.services] });
    if (discListeners.length === 1) {
      pollDiscovery();
      discPollTimer = setInterval(pollDiscovery, 5000);
    }
    return () => {
      discListeners = discListeners.filter((l) => l !== fn);
      if (discListeners.length === 0 && discPollTimer) {
        clearInterval(discPollTimer);
        discPollTimer = undefined;
      }
    };
  },

  async start() {
    try {
      await startDiscovery();
      discData.active = true;
      discData.error = null;
    } catch (e: unknown) {
      discData.error = String(e);
    }
    notifyDisc();
  },

  async stop() {
    try {
      await stopDiscovery();
      discData.active = false;
      discData.services = [];
      discData.error = null;
    } catch (e: unknown) {
      discData.error = String(e);
    }
    notifyDisc();
  },
};
