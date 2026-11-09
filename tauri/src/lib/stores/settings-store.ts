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
  apiKey: string;
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
  apiKey: "",
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
    root.style.setProperty("--bg-primary", "#1c1c1c");
    root.style.setProperty("--bg-secondary", "#232323");
    root.style.setProperty("--bg-card", "#262626");
    root.style.setProperty("--bg-input", "#2a2a2a");
    root.style.setProperty("--border", "#383838");
    root.style.setProperty("--text-primary", "#f1f1ef");
    root.style.setProperty("--text-secondary", "#c4c4be");
    root.style.setProperty("--text-muted", "#9a9b96");
    root.style.setProperty("--success", "#8fa68e");
    root.style.setProperty("--warning", "#c9a26e");
    root.style.setProperty("--danger", "#c97b6e");
    root.style.colorScheme = "dark";
  } else {
    /* Light is the same design on a pale field, not a different palette:
       concrete ground, near-black ink, the same sage accent. */
    root.style.setProperty("--bg-primary", "#c8c5ba");
    root.style.setProperty("--bg-secondary", "#d2cfc4");
    root.style.setProperty("--bg-card", "#e0ded4");
    root.style.setProperty("--bg-input", "#dedbd1");
    root.style.setProperty("--border", "#b3b0a5");
    root.style.setProperty("--text-primary", "#1c1c1c");
    root.style.setProperty("--text-secondary", "#3a3a34");
    root.style.setProperty("--text-muted", "#5c5c56");
    root.style.setProperty("--success", "#5c7a5b");
    root.style.setProperty("--warning", "#8a6a3a");
    root.style.setProperty("--danger", "#a4523f");
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
