import rss from "@astrojs/rss";
import { getCollection } from "astro:content";
import type { APIContext } from "astro";

export async function GET(context: APIContext) {
  const entries = (await getCollection("changelog")).sort(
    (a, b) => b.data.releaseDate.valueOf() - a.data.releaseDate.valueOf(),
  );
  return rss({
    title: "DistLLM Changelog",
    description: "Every DistLLM release.",
    site: context.site!,
    items: entries.map((entry) => ({
      title: `v${entry.data.version}`,
      description: entry.data.highlights.join(" · "),
      pubDate: entry.data.releaseDate,
      link: `/changelog/${entry.id}/`,
    })),
  });
}
