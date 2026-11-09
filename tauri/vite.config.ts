import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

const host = process.env.TAURI_DEV_HOST;

// Browser-mode backend. When the app runs in a plain browser (`npm run dev`),
// all coordinator HTTP calls go through this same-origin proxy, which forwards
// them to the DistLLM API server. Two server-side obstacles make this the
// correct fix rather than a convenience:
//   1. CORS: the Python server only allows origins localhost:3000/8080 by
//      default — not Vite's 1420 — so direct cross-origin fetches are blocked
//      by the browser before they reach the server.
//   2. CSRF: CSRFSameOriginMiddleware rejects POSTs whose Origin header is not
//      allow-listed. Browsers attach Origin: http://localhost:1420 to POSTs
//      even through a proxy, so the proxy strips Origin/Referer below; the
//      middleware explicitly permits requests carrying neither header
//      (its documented non-browser-client path).
//
// Override the target with DISTLLM_PROXY_TARGET when the backend runs
// elsewhere, e.g. DISTLLM_PROXY_TARGET=http://127.0.0.1:8165 npm run dev.
const proxyTarget = process.env.DISTLLM_PROXY_TARGET || "http://127.0.0.1:8000";

function stripOriginHeaders(proxy: {
  on: (event: string, cb: (proxyReq: { removeHeader: (name: string) => void }) => void) => void;
}) {
  proxy.on("proxyReq", (proxyReq) => {
    proxyReq.removeHeader("origin");
    proxyReq.removeHeader("referer");
  });
}

const proxied = {
  target: proxyTarget,
  changeOrigin: true,
  configure: stripOriginHeaders,
};

export default defineConfig({
  plugins: [svelte()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? { protocol: "ws", host, port: 1421 }
      : undefined,
    watch: {
      ignored: ["**/src-tauri/**"],
    },
    proxy: {
      // Coordinator REST + Prometheus + OpenAI-compatible surface. Paths only
      // ever used by the app's browser transport; Tauri mode uses invoke().
      "/v1": proxied,
      "/api": proxied,
      "/admin": proxied,
      "/health": proxied,
      "/metrics": proxied,
      "/ready": proxied,
      "/live": proxied,
    },
  },
});
