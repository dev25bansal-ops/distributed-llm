<script lang="ts">
  let {
    value = $bindable(""),
    type = "text",
    placeholder = "",
    id,
    min,
    max,
    readonly = false,
    onfocus,
  }: {
    value?: string | number;
    type?: string;
    placeholder?: string;
    id?: string;
    min?: number;
    max?: number;
    readonly?: boolean;
    onfocus?: (e: FocusEvent) => void;
  } = $props();
</script>

<input
  {id}
  {type}
  {placeholder}
  {readonly}
  {min}
  {max}
  class="input"
  bind:value
  {onfocus}
/>
