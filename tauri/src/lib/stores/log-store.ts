const STORAGE_KEY = "distllm_logs";
const MAX_LOGS = 500;

export type LogLevel = "info" | "warn" | "error";

export interface LogEntry {
  id: number;
  timestamp: number;
  level: LogLevel;
  category: string;
  message: string;
}

function loadLogs(): LogEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch {
    // corrupted
  }
  return [];
}

function persistLogs(logs: LogEntry[]) {
  // Keep only last MAX_LOGS
  const trimmed = logs.slice(-MAX_LOGS);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
}

type LogListener = (logs: LogEntry[]) => void;

let logs: LogEntry[] = loadLogs();
let listeners: LogListener[] = [];
let nextId = logs.length > 0 ? Math.max(...logs.map((l) => l.id)) + 1 : 0;

function notify() {
  for (const fn of listeners) {
    fn([...logs]);
  }
}

function addLog(level: LogLevel, category: string, message: string) {
  const entry: LogEntry = {
    id: nextId++,
    timestamp: Date.now(),
    level,
    category,
    message,
  };
  logs = [...logs, entry];
  persistLogs(logs);
  notify();
}

export const logStore = {
  subscribe(fn: LogListener) {
    listeners.push(fn);
    fn([...logs]);
    return () => {
      listeners = listeners.filter((l) => l !== fn);
    };
  },
  info(category: string, message: string) {
    addLog("info", category, message);
  },
  warn(category: string, message: string) {
    addLog("warn", category, message);
  },
  error(category: string, message: string) {
    addLog("error", category, message);
  },
  clear() {
    logs = [];
    persistLogs(logs);
    notify();
  },
  export(): string {
    return logs
      .map((l) => {
        const ts = new Date(l.timestamp).toISOString();
        return `[${ts}] [${l.level.toUpperCase()}] [${l.category}] ${l.message}`;
      })
      .join("\n");
  },
  getAll(): LogEntry[] {
    return [...logs];
  },
};
