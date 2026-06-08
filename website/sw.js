/**
 * DistLLM Service Worker — Offline support and asset caching.
 *
 * Strategy:
 * - HTML navigations: network-first (always get fresh content)
 * - Static assets (JS/CSS/images): cache-first (fastest load)
 * - API responses: network-only (never cache dynamic data)
 *
 * Version: v1.0.0
 */

const CACHE_NAME = "distllm-v1";
const STATIC_ASSETS = [
  "/css/base.css",
  "/css/layout.css",
  "/css/components.css",
  "/css/modern.css",
  "/css/animations.css",
  "/css/themes.css",
  "/js/utils.js",
  "/js/theme.js",
  "/js/main.js",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS))
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // API calls: network only
  if (url.pathname.startsWith("/v1/") || url.pathname.startsWith("/api/")) {
    return;
  }

  // Static assets: cache-first
  if (STATIC_ASSETS.includes(url.pathname)) {
    event.respondWith(
      caches.match(event.request).then((cached) => cached || fetch(event.request))
    );
    return;
  }

  // HTML navigations: network-first
  if (url.pathname === "/" || url.pathname.endsWith(".html")) {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          return response;
        })
        .catch(() => caches.match(event.request))
    );
    return;
  }
});
