import type { ModelSlot, ModelRoutingRule } from "../types";
import { getModelSlots, loadModelSlot, unloadModelSlot, getRoutingRules, setRoutingRule, deleteRoutingRule } from "../api";

type MultiModelStoreData = {
  slots: ModelSlot[];
  rules: ModelRoutingRule[];
  loading: boolean;
  error: string | null;
};

type Listener = (data: MultiModelStoreData) => void;

let data: MultiModelStoreData = {
  slots: [],
  rules: [],
  loading: true,
  error: null,
};

let listeners: Listener[] = [];
let pollTimer: ReturnType<typeof setInterval> | undefined;

function notify() {
  for (const fn of listeners) {
    fn({ ...data, slots: [...data.slots], rules: [...data.rules] });
  }
}

async function fetchAll() {
  try {
    const [slots, rules] = await Promise.all([
      getModelSlots().catch(() => []),
      getRoutingRules().catch(() => []),
    ]);
    data = { ...data, slots, rules, loading: false, error: null };
  } catch (e: unknown) {
    data = { ...data, loading: false, error: String(e) };
  }
  notify();
}

function startPoll() {
  fetchAll();
  pollTimer = setInterval(fetchAll, 5000);
}

function stopPoll() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = undefined;
  }
}

export const multiModelStore = {
  subscribe(fn: Listener) {
    listeners.push(fn);
    fn({ ...data, slots: [...data.slots], rules: [...data.rules] });
    if (listeners.length === 1) startPoll();
    return () => {
      listeners = listeners.filter((l) => l !== fn);
      if (listeners.length === 0) stopPoll();
    };
  },

  getSnapshot(): MultiModelStoreData {
    return { ...data, slots: [...data.slots], rules: [...data.rules] };
  },

  async refresh() {
    await fetchAll();
  },

  async loadModel(slotId: string, modelId: string) {
    try {
      data.error = null;
      notify();
      await loadModelSlot(slotId, modelId);
      await fetchAll();
    } catch (e: unknown) {
      data.error = String(e);
      notify();
    }
  },

  async unloadModel(slotId: string) {
    try {
      await unloadModelSlot(slotId);
      await fetchAll();
    } catch (e: unknown) {
      data.error = String(e);
      notify();
    }
  },

  async addRule(rule: ModelRoutingRule) {
    try {
      await setRoutingRule(rule);
      await fetchAll();
    } catch (e: unknown) {
      data.error = String(e);
      notify();
    }
  },

  async removeRule(ruleId: string) {
    try {
      await deleteRoutingRule(ruleId);
      await fetchAll();
    } catch (e: unknown) {
      data.error = String(e);
      notify();
    }
  },
};
