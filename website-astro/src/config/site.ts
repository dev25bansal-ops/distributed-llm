export const SITE = {
  name: "DistLLM",
  title: "DistLLM — Pool consumer GPUs to run large models",
  description:
    "Open-source distributed LLM inference. Pool GPUs across multiple devices to run models no single machine can handle — pipeline parallelism, speculative decoding, one-command deploy.",
  url: "https://distllm.dev",
  github: "https://github.com/distributed-llm/distributed-llm",
  plausibleDomain: "distllm.dev",
};

export const NAV = [
  { label: "Docs", href: "/docs/" },
  { label: "Playground", href: "/playground/" },
  { label: "Benchmarks", href: "/benchmarks/" },
  { label: "Integrations", href: "/integrations/" },
  { label: "Blog", href: "/blog/" },
  { label: "Community", href: "/community/" },
];

export const FOOTER_COLUMNS = [
  {
    heading: "Product",
    links: [
      { label: "Docs", href: "/docs/installation/" },
      { label: "Playground", href: "/playground/" },
      { label: "Benchmarks", href: "/benchmarks/" },
      { label: "Integrations", href: "/integrations/" },
    ],
  },
  {
    heading: "Resources",
    links: [
      { label: "Blog", href: "/blog/" },
      { label: "Changelog", href: "/changelog/" },
      { label: "Glossary", href: "/glossary/" },
      { label: "Learn", href: "/learn/" },
    ],
  },
  {
    heading: "Project",
    links: [
      { label: "Use Cases", href: "/use-cases/" },
      { label: "Security", href: "/security/" },
      { label: "Community", href: "/community/" },
      { label: "GitHub", href: SITE.github, external: true },
    ],
  },
];
