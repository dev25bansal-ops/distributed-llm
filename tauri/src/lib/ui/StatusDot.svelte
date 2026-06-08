<script lang="ts">
  let {
    variant = "gray",
  }: {
    variant?: "green" | "red" | "gray";
  } = $props();
</script>

<span class="status-dot {variant}"></span>
