<script lang="ts">
  import { onMount } from "svelte";
  import { toastStore } from "./toast-store";

  let toasts = $state<
    Array<{
      id: number;
      message: string;
      type: "success" | "error" | "info" | "warning";
    }>
  >([]);

  onMount(() => {
    return toastStore.subscribe((t) => {
      toasts = t;
    });
  });
</script>

{#if toasts.length > 0}
  <div class="toast-container">
    {#each toasts as toast (toast.id)}
      <div class="toast {toast.type}" role="status">
        <span class="toast-icon">
          {#if toast.type === "success"}✓{:else if toast.type === "error"}✕{:else if toast.type === "warning"}⚠{:else}ℹ{/if}
        </span>
        <span class="toast-message">{toast.message}</span>
        <button
          class="toast-dismiss"
          onclick={() => toastStore.dismiss(toast.id)}
          aria-label="Dismiss notification"
        >
          ✕
        </button>
      </div>
    {/each}
  </div>
{/if}

<style>
  .toast-icon {
    flex-shrink: 0;
    font-size: 14px;
    line-height: 1;
  }
  .toast-message {
    flex: 1;
  }
  .toast-dismiss {
    flex-shrink: 0;
    background: none;
    border: none;
    color: inherit;
    cursor: pointer;
    padding: 2px 4px;
    font-size: 12px;
    opacity: 0.6;
    border-radius: 4px;
    line-height: 1;
  }
  .toast-dismiss:hover {
    opacity: 1;
    background: color-mix(in srgb, currentColor 15%, transparent);
  }
</style>
