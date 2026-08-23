const STORAGE_KEY = "distllm_settings";

export interface AppSettings {
  defaultClusterPort: number;
  grafanaUrl: string;
  theme: "dark" | "light" | "auto";
  autoJoin: boolean;
  downloadDir: string;
  notifications: {
    clusterEvents: boolean;
    modelDownloads: boolean;
    inferenceRequests: boolean;
    errors: boolean;
    native: boolean;
    updateAvailable: boolean;
  };
  pythonPath: string;
  apiEndpoint: string;
  updateServerUrl: string;
}

const defaults: AppSettings = {
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
};

function loadSettings(): AppSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      return { ...defaults, ...parsed, notifications: { ...defaults.notifications, ...parsed.notifications } };
    }
  } catch {
    // corrupted data, use defaults
  }
  return { ...defaults };
}

function saveSettings(settings: AppSettings) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
}

export function applyTheme(theme?: "dark" | "light" | "auto") {
  const t = theme ?? current.theme;
  const root = document.documentElement;
  const isDark = t === "dark" || (t === "auto" && window.matchMedia("(prefers-color-scheme: dark)").matches);

  if (isDark) {
    root.style.setProperty("--bg-primary", "#0f0f1a");
    root.style.setProperty("--bg-secondary", "#1a1a2e");
    root.style.setProperty("--bg-card", "#1e1e36");
    root.style.setProperty("--bg-input", "#252542");
    root.style.setProperty("--border", "#2d2d50");
    root.style.setProperty("--text-primary", "#e0e0f0");
    root.style.setProperty("--text-secondary", "#9090b0");
    root.style.setProperty("--text-muted", "#606080");
    root.style.setProperty("--success", "#22cc66");
    root.style.setProperty("--warning", "#ffaa33");
    root.style.setProperty("--danger", "#ff4466");
    root.style.colorScheme = "dark";
  } else {
    root.style.setProperty("--bg-primary", "#f5f5fa");
    root.style.setProperty("--bg-secondary", "#ebebfa");
    root.style.setProperty("--bg-card", "#ffffff");
    root.style.setProperty("--bg-input", "#e8e8f5");
    root.style.setProperty("--border", "#d0d0e0");
    root.style.setProperty("--text-primary", "#1a1a30");
    root.style.setProperty("--text-secondary", "#606080");
    root.style.setProperty("--text-muted", "#9090b0");
    root.style.setProperty("--success", "#1a9a4a");
    root.style.setProperty("--warning", "#cc7700");
    root.style.setProperty("--danger", "#cc2244");
    root.style.colorScheme = "light";
  }
}

type SettingsListener = (settings: AppSettings) => void;

let current: AppSettings = loadSettings();
let listeners: SettingsListener[] = [];

function notify() {
  for (const fn of listeners) {
    fn({ ...current });
  }
}

export const settingsStore = {
  subscribe(fn: SettingsListener) {
    listeners.push(fn);
    fn({ ...current });
    return () => {
      listeners = listeners.filter((l) => l !== fn);
    };
  },
  getSnapshot(): AppSettings {
    return { ...current };
  },
  update(partial: Partial<AppSettings>) {
    current = { ...current, ...partial };
    saveSettings(current);
    if (partial.theme !== undefined) applyTheme(partial.theme);
    notify();
  },
  updateNotifications(partial: Partial<AppSettings["notifications"]>) {
    current = { ...current, notifications: { ...current.notifications, ...partial } };
    saveSettings(current);
    notify();
  },
  reset() {
    current = { ...defaults };
    saveSettings(current);
    applyTheme(current.theme);
    notify();
  },
};
