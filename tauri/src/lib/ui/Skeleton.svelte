<script lang="ts">
  let {
    width = "100%",
    height = "16px",
    style = "",
  }: {
    width?: string;
    height?: string;
    style?: string;
  } = $props();
</script>

<div
  class="skeleton"
  style="width: {width}; height: {height}; {style}"
></div>
