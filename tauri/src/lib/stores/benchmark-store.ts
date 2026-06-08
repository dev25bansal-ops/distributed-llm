import type { BenchmarkRun, BenchmarkConfig } from "../types";
import { runBenchmark } from "../api";

type BenchmarkStoreData = {
  runs: BenchmarkRun[];
  running: boolean;
  progress: number;
  error: string | null;
  abortController: AbortController | null;
};

type Listener = (data: BenchmarkStoreData) => void;

const STORAGE_KEY = "distllm-benchmarks";

function loadRuns(): BenchmarkRun[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch { /* ignore */ }
  return [];
}

function saveRuns(runs: BenchmarkRun[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(runs.slice(-200)));
  } catch { /* ignore */ }
}

let data: BenchmarkStoreData = {
  runs: loadRuns(),
  running: false,
  progress: 0,
  error: null,
  abortController: null,
};

let listeners: Listener[] = [];

function notify() {
  for (const fn of listeners) {
    fn({ ...data, runs: [...data.runs] });
  }
}

export const benchmarkStore = {
  subscribe(fn: Listener) {
    listeners.push(fn);
    fn({ ...data, runs: [...data.runs] });
    return () => {
      listeners = listeners.filter((l) => l !== fn);
    };
  },

  getSnapshot(): BenchmarkStoreData {
    return { ...data, runs: [...data.runs] };
  },

  async start(config: BenchmarkConfig, baseUrl: string) {
    if (data.running) return;
    data.running = true;
    data.progress = 0;
    data.error = null;
    notify();

    const controller = new AbortController();
    data.abortController = controller;

    try {
      const results = await runBenchmark(baseUrl, config, controller.signal);
      data.runs = [...data.runs, ...results];
      saveRuns(data.runs);
      data.progress = 100;
    } catch (e: unknown) {
      if (!(e instanceof DOMException && e.name === "AbortError")) {
        data.error = String(e);
      }
    } finally {
      data.running = false;
      data.abortController = null;
      notify();
    }
  },

  stop() {
    if (data.abortController) {
      data.abortController.abort();
      data.running = false;
      data.abortController = null;
      notify();
    }
  },

  clear() {
    data.runs = [];
    data.error = null;
    saveRuns(data.runs);
    notify();
  },

  getByModel(model: string): BenchmarkRun[] {
    return data.runs.filter((r) => r.model === model);
  },
};
