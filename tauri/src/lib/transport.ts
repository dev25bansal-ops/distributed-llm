// Transport seam between the Tauri desktop shell and a plain browser.
//
// In Tauri, `window.__TAURI_INTERNALS__` is injected by the shell and the
// `@tauri-apps/*` packages work. In a plain browser it is undefined and any
// `invoke(...)` call would crash with `undefined.invoke` / `transformCallback`
// errors, so every transport helper below branches on `isTauri()` first.
import { listen } from "@tauri-apps/api/event";
import { settingsStore } from "./stores/settings-store";

declare global {
  interface Window {
    __TAURI_INTERNALS__?: unknown;
  }
}

/** Tauri 2 injects `window.__TAURI_INTERNALS__` into the webview. */
export function isTauri(): boolean {
  return typeof window !== "undefined" && window.__TAURI_INTERNALS__ !== undefined;
}

export type UnlistenFn = () => void;

/**
 * Subscribe to a Tauri event. In browser mode this is a no-op that resolves
 * to an inert unlisten function, so call sites can keep the same shape in
 * both modes (`await listenEvent(...)` or `listenEvent(...).then(fn => fn())`).
 */
export function listenEvent<T = unknown>(
  event: string,
  handler: (payload: T) => void,
): Promise<UnlistenFn> {
  if (!isTauri()) {
    return Promise.resolve(() => {});
  }
  return listen<T>(event, (e) => handler(e.payload as T));
}

export const DEFAULT_API_BASE = "http://localhost:8000";

/**
 * Coordinator base URL for browser mode.
 *
 * - Explicit `apiEndpoint` from Settings wins (Tauri and browser alike).
 * - Otherwise, when served by Vite (`npm run dev`), use the same origin and
 *   let vite.config.ts's dev proxy forward /v1, /api, /admin, /health, ...
 *   to the backend. This sidesteps CORS (the Python server does not allow
 *   origin localhost:1420 by default) and CSRF origin rejection entirely.
 * - Otherwise (static build in a plain browser) fall back to the default port.
 *
 * Reads the settings store directly (not the `./stores` barrel) to avoid a
 * circular import with api.ts.
 */
export function apiBaseUrl(): string {
  const configured = settingsStore.getSnapshot().apiEndpoint?.trim();
  if (configured) {
    return configured.replace(/\/+$/, "");
  }
  if (typeof window !== "undefined") {
    const { protocol, port, hostname } = window.location;
    const isViteDev =
      (protocol === "http:" || protocol === "https:") &&
      (port === "1420" || port === "1421");
    if (isViteDev) {
      return ""; // same-origin relative paths → dev proxy
    }
  }
  return DEFAULT_API_BASE;
}

/** API key from settings, never taken from a module-load-time snapshot. */
export function getApiKey(): string {
  return settingsStore.getSnapshot().apiKey?.trim() ?? "";
}

/** Authorization header for the coordinator; omitted when no key is set. */
export function authHeaders(): Record<string, string> {
  const key = getApiKey();
  return key ? { Authorization: `Bearer ${key}` } : {};
}

/**
 * Thrown by API functions that have no browser equivalent (they depend on
 * Rust-side state or the desktop chrome). Every caller of these functions
 * catches — verified by grep — so this never becomes an unhandled rejection.
 */
export class BrowserUnsupportedError extends Error {
  constructor(feature: string) {
    super(`${feature} is not available in browser mode (requires the Tauri desktop shell)`);
    this.name = "BrowserUnsupportedError";
  }
}
