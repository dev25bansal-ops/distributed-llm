<script lang="ts">
  import type { Snippet } from "svelte";

  let {
    variant = "primary",
    disabled = false,
    onclick,
    children,
  }: {
    variant?: "primary" | "danger" | "ghost";
    disabled?: boolean;
    onclick?: () => void;
    children?: Snippet;
  } = $props();
</script>

<button
  class="btn btn-{variant}"
  {disabled}
  {onclick}
>
  {@render children?.()}
</button>
