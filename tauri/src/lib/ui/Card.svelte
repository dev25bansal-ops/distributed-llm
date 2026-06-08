<script lang="ts">
  import type { Snippet } from "svelte";

  let {
    title,
    description,
    children,
  }: {
    title?: string;
    description?: string;
    children?: Snippet;
  } = $props();
</script>

<section class="card">
  {#if title}
    <h2 class="card-title">{title}</h2>
  {/if}
  {#if description}
    <p class="card-desc">{description}</p>
  {/if}
  {@render children?.()}
</section>
