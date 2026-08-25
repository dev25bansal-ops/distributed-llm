export interface DocsNavGroup {
  group: string;
  items: { slug: string; label: string }[];
}

export const DOCS_NAV: DocsNavGroup[] = [
  {
    group: "Getting Started",
    items: [
      { slug: "installation", label: "Installation" },
      { slug: "quick-start", label: "Quick Start" },
    ],
  },
  {
    group: "Configuration",
    items: [{ slug: "configuration", label: "YAML Config" }],
  },
  {
    group: "Architecture",
    items: [
      { slug: "architecture", label: "Architecture" },
      { slug: "pipeline-parallelism", label: "Pipeline Parallelism" },
      { slug: "auto-discovery", label: "Auto-Discovery" },
      { slug: "multi-node", label: "Multi-Node Setup" },
      { slug: "federated-training", label: "Federated Training" },
    ],
  },
  {
    group: "API",
    items: [
      { slug: "chat-completions", label: "Chat Completions" },
      { slug: "embeddings", label: "Embeddings" },
      { slug: "health-checks", label: "Health Checks" },
    ],
  },
  {
    group: "Deployment",
    items: [
      { slug: "docker", label: "Docker" },
      { slug: "kubernetes", label: "Kubernetes" },
    ],
  },
  {
    group: "Reference",
    items: [
      { slug: "model-compatibility", label: "Model Compatibility" },
      { slug: "performance-tuning", label: "Performance Tuning" },
      { slug: "security", label: "Security" },
      { slug: "observability", label: "Observability" },
      { slug: "integrations-docs", label: "Integrations" },
    ],
  },
];

export function getDocsNav(): DocsNavGroup[] {
  return DOCS_NAV;
}

export function getDocNeighbors(slug: string): {
  prev?: { slug: string; label: string };
  next?: { slug: string; label: string };
} {
  const flat = DOCS_NAV.flatMap((g) => g.items);
  const idx = flat.findIndex((i) => i.slug === slug);
  return {
    prev: idx > 0 ? flat[idx - 1] : undefined,
    next: idx >= 0 && idx < flat.length - 1 ? flat[idx + 1] : undefined,
  };
}

/**
 * Build-time guard: every docs collection entry must appear in DOCS_NAV,
 * otherwise it is unreachable from the sidebar (an "orphan" page).
 * Call this from the docs route's getStaticPaths() so `npm run build`
 * fails loudly when a new .mdx file is added without a nav entry.
 */
export function assertDocsNavCoverage(slugs: string[]): void {
  const navList = DOCS_NAV.flatMap((g) => g.items.map((i) => i.slug));
  const navSet = new Set(navList);
  const orphans = slugs.filter((s) => !navSet.has(s));
  if (orphans.length > 0) {
    throw new Error(
      `DOCS_NAV is missing entries for docs content (unreachable from sidebar): ${orphans.join(", ")}. ` +
        "Add each slug to src/config/docs-nav.ts.",
    );
  }
  const dupes = navList.filter((s, i) => navList.indexOf(s) !== i);
  if (dupes.length > 0) {
    throw new Error(`DOCS_NAV contains duplicate slugs: ${dupes.join(", ")}.`);
  }
}
