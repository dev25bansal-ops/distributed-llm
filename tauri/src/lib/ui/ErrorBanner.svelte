<script lang="ts">
  let {
    message,
    ondismiss,
  }: {
    message: string;
    ondismiss?: () => void;
  } = $props();
</script>

{#if message}
  <div class="error-banner" role="alert">
    <span class="error-icon">⚠</span>
    <span class="error-text">{message}</span>
    {#if ondismiss}
      <button class="error-dismiss" onclick={ondismiss} aria-label="Dismiss error">
        ✕
      </button>
    {/if}
  </div>
{/if}

<style>
  .error-icon {
    flex-shrink: 0;
    font-size: 14px;
  }
  .error-text {
    flex: 1;
  }
  .error-dismiss {
    background: none;
    border: none;
    color: inherit;
    cursor: pointer;
    padding: 2px 4px;
    font-size: 12px;
    opacity: 0.7;
    border-radius: 4px;
  }
  .error-dismiss:hover {
    opacity: 1;
    background: color-mix(in srgb, currentColor 10%, transparent);
  }
</style>
