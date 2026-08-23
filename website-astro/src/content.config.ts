import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const docs = defineCollection({
  loader: glob({ pattern: "**/*.mdx", base: "./src/content/docs" }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    group: z.enum([
      "Getting Started",
      "Configuration",
      "Architecture",
      "API",
      "Deployment",
      "Reference",
    ]),
    order: z.number(),
    icon: z.string().optional(),
  }),
});

const blog = defineCollection({
  loader: glob({ pattern: "**/*.{md,mdx}", base: "./src/content/blog" }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    updatedDate: z.coerce.date().optional(),
    author: z.string().default("DistLLM Team"),
    tags: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
    linkOut: z.string().url().optional(),
  }),
});

const changelog = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/changelog" }),
  schema: z.object({
    version: z.string(),
    releaseDate: z.coerce.date(),
    breaking: z.boolean().default(false),
    highlights: z.array(z.string()).default([]),
  }),
});

export const collections = { docs, blog, changelog };
