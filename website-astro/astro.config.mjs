import { defineConfig } from "astro/config";
import sitemap from "@astrojs/sitemap";
import mdx from "@astrojs/mdx";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  site: "https://distllm.dev",
  integrations: [sitemap(), mdx()],
  vite: { plugins: [tailwindcss()] },
  markdown: {
    shikiConfig: { themes: { light: "vitesse-light", dark: "vitesse-light" } },
  },
  redirects: {
    "/docs.html": "/docs/",
    "/api.html": "/playground/",
    "/playground.html": "/playground/",
    "/blog.html": "/blog/",
    "/changelog.html": "/changelog/",
    "/benchmarks.html": "/benchmarks/",
    "/glossary.html": "/glossary/",
    "/comparisons.html": "/comparisons/",
    "/security.html": "/security/",
    "/use-cases.html": "/use-cases/",
    "/integrations.html": "/integrations/",
    "/community.html": "/community/",
    "/learn.html": "/learn/",
    "/privacy.html": "/privacy/",
    "/terms.html": "/terms/",
    "/offline.html": "/",
    "/pricing-compare": "/#pricing",
    "/enterprise": "/use-cases/",
    "/showcase": "/community/",
    "/status": "/",
    "/tutorials": "/learn/",
    "/feed.xml": "/rss.xml",
  },
});
