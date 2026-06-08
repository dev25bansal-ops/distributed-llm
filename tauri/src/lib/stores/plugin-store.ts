import type { PluginConfig } from "../types";
import { getPlugins, savePlugin, deletePlugin, testPlugin } from "../api";

type PluginStoreData = {
  plugins: PluginConfig[];
  loading: boolean;
  error: string | null;
};

type Listener = (data: PluginStoreData) => void;

const STORAGE_KEY = "distllm-plugins";

function loadLocal(): PluginConfig[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch { /* ignore */ }
  return [];
}

function saveLocal(plugins: PluginConfig[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(plugins));
  } catch { /* ignore */ }
}

let data: PluginStoreData = {
  plugins: loadLocal(),
  loading: false,
  error: null,
};

let listeners: Listener[] = [];

function notify() {
  for (const fn of listeners) {
    fn({ ...data, plugins: [...data.plugins] });
  }
}

export const pluginStore = {
  subscribe(fn: Listener) {
    listeners.push(fn);
    fn({ ...data, plugins: [...data.plugins] });
    return () => {
      listeners = listeners.filter((l) => l !== fn);
    };
  },

  getSnapshot(): PluginStoreData {
    return { ...data, plugins: [...data.plugins] };
  },

  async refresh() {
    data.loading = true;
    notify();
    try {
      const plugins = await getPlugins();
      data = { ...data, plugins, loading: false, error: null };
      saveLocal(plugins);
    } catch (e: unknown) {
      // Fall back to local
      data = { ...data, loading: false, error: null };
    }
    notify();
  },

  async add(plugin: PluginConfig) {
    data.plugins = [...data.plugins, plugin];
    saveLocal(data.plugins);
    notify();
    try {
      await savePlugin(plugin);
    } catch { /* stored locally */ }
  },

  async update(plugin: PluginConfig) {
    data.plugins = data.plugins.map((p) => (p.id === plugin.id ? plugin : p));
    saveLocal(data.plugins);
    notify();
    try {
      await savePlugin(plugin);
    } catch { /* stored locally */ }
  },

  async remove(pluginId: string) {
    data.plugins = data.plugins.filter((p) => p.id !== pluginId);
    saveLocal(data.plugins);
    notify();
    try {
      await deletePlugin(pluginId);
    } catch { /* stored locally */ }
  },

  async test(pluginId: string): Promise<boolean> {
    try {
      return await testPlugin(pluginId);
    } catch {
      return false;
    }
  },
};
