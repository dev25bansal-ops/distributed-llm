<script lang="ts">
  import { onMount, afterUpdate } from "svelte";
  import { chatStore, clusterStore, logStore } from "./stores";
  import { ErrorBanner } from "./ui";
  import InferenceMetrics from "./InferenceMetrics.svelte";
  import type { ChatMessage, ChatOptions, InferenceMetrics as Metrics } from "./types";

  let messages = $state<ChatMessage[]>([]);
  let options = $state<ChatOptions>({
    temperature: 0.7,
    top_p: 0.9,
    max_tokens: 2048,
    system_prompt: "You are a helpful assistant.",
  });
  let streaming = $state(false);
  let metrics = $state<Metrics | null>(null);
  let error = $state<string | null>(null);

  let inputText = $state("");
  let showSettings = $state(false);
  let messagesEl = $state<HTMLDivElement | null>(null);
  let inputEl = $state<HTMLTextAreaElement | null>(null);

  let unsubscribe: (() => void) | undefined;

  onMount(() => {
    unsubscribe = chatStore.subscribe((d) => {
      messages = d.messages;
      streaming = d.streaming;
      metrics = d.metrics;
      error = d.error;
    });
    // Load saved options
    const saved = chatStore.getOptions();
    options = { ...saved };
    chatStore.updateOptions(saved);
  });

  afterUpdate(() => {
    if (messagesEl) {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
  });

  function handleSend() {
    if (!inputText.trim() || streaming) return;
    const snap = clusterStore.getSnapshot();
    const addr = snap.cluster?.coordinator_addr;
    if (!addr) {
      error = "No cluster running. Start or join a cluster first.";
      return;
    }
    logStore.info("chat", `Inference request sent (${inputText.length} chars)`);
    chatStore.send(inputText, addr);
    inputText = "";
    // Refocus input
    setTimeout(() => inputEl?.focus(), 50);
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleStop() {
    chatStore.stop();
  }

  function handleClear() {
    if (window.confirm("Clear all messages?")) {
      chatStore.clearMessages();
    }
  }

  function updateOption<K extends keyof ChatOptions>(key: K, value: ChatOptions[K]) {
    options = { ...options, [key]: value };
    chatStore.updateOptions({ [key]: value });
  }
</script>

<div class="chat-page">
  <div class="chat-header">
    <h1 class="page-title">Chat</h1>
    <div class="chat-header-actions">
      <button class="icon-btn" title="Settings" onclick={() => showSettings = !showSettings}>
        {showSettings ? "✕" : "⚙"}
      </button>
      <button class="icon-btn danger" title="Clear conversation" onclick={handleClear}>
        🗑
      </button>
    </div>
  </div>

  <ErrorBanner message={error ?? ""} ondismiss={() => error = null} />

  {#if showSettings}
    <div class="settings-panel">
      <div class="setting-row">
        <label class="setting-label" for="sys-prompt">System Prompt</label>
        <textarea
          id="sys-prompt"
          class="setting-textarea"
          rows="3"
          value={options.system_prompt}
          oninput={(e) => updateOption("system_prompt", (e.target as HTMLTextAreaElement).value)}
        ></textarea>
      </div>
      <div class="settings-grid">
        <div class="setting-row">
          <label class="setting-label" for="temp">Temperature: {options.temperature.toFixed(2)}</label>
          <input
            id="temp"
            type="range"
            class="setting-range"
            min="0"
            max="2"
            step="0.05"
            value={options.temperature}
            oninput={(e) => updateOption("temperature", parseFloat((e.target as HTMLInputElement).value))}
          />
        </div>
        <div class="setting-row">
          <label class="setting-label" for="top-p">Top P: {options.top_p.toFixed(2)}</label>
          <input
            id="top-p"
            type="range"
            class="setting-range"
            min="0"
            max="1"
            step="0.05"
            value={options.top_p}
            oninput={(e) => updateOption("top_p", parseFloat((e.target as HTMLInputElement).value))}
          />
        </div>
        <div class="setting-row">
          <label class="setting-label" for="max-tokens">Max Tokens</label>
          <input
            id="max-tokens"
            type="number"
            class="input"
            value={options.max_tokens}
            oninput={(e) => updateOption("max_tokens", parseInt((e.target as HTMLInputElement).value) || 2048)}
          />
        </div>
      </div>
    </div>
  {/if}

  <div class="messages-container" bind:this={messagesEl}>
    {#if messages.length === 0}
      <div class="empty-chat">
        <div class="empty-icon">💬</div>
        <p class="empty-title">Start a conversation</p>
        <p class="empty-hint">Send a message to begin chatting with your distributed model.</p>
      </div>
    {:else}
      {#each messages as msg (msg.id)}
        <div class="message {msg.role}">
          <div class="msg-avatar">
            {#if msg.role === "user"} You {:else if msg.role === "system"} ⚙ {:else} AI {/if}
          </div>
          <div class="msg-content">
            <div class="msg-text">{msg.content}{#if streaming && msg.role === "assistant" && msg === messages[messages.length - 1]}<span class="cursor">|</span>{/if}</div>
          </div>
        </div>
      {/each}
    {/if}
  </div>

  <InferenceMetrics {metrics} {streaming} />

  <div class="input-area">
    <textarea
      bind:this={inputEl}
      class="chat-input"
      placeholder="Type a message..."
      rows="1"
      bind:value={inputText}
      onkeydown={handleKeydown}
      disabled={streaming}
    ></textarea>
    {#if streaming}
      <button class="send-btn stop" onclick={handleStop} title="Stop generating">■</button>
    {:else}
      <button
        class="send-btn"
        onclick={handleSend}
        disabled={!inputText.trim()}
        title="Send message"
      >➤</button>
    {/if}
  </div>
</div>

<style>
  .chat-page {
    display: flex;
    flex-direction: column;
    height: 100%;
    max-width: 800px;
    margin: 0 auto;
  }

  .chat-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    flex-shrink: 0;
  }
  .chat-header-actions {
    display: flex;
    gap: 4px;
  }
  .icon-btn {
    width: 32px;
    height: 32px;
    border-radius: 6px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--text-secondary);
    font-size: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.15s;
  }
  .icon-btn:hover {
    background: var(--bg-input);
    color: var(--text-primary);
  }
  .icon-btn.danger:hover {
    color: var(--danger);
    border-color: var(--danger);
  }

  .settings-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
    flex-shrink: 0;
  }
  .setting-row {
    margin-bottom: 12px;
  }
  .setting-row:last-child {
    margin-bottom: 0;
  }
  .setting-label {
    display: block;
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 4px;
  }
  .setting-textarea {
    width: 100%;
    padding: 8px 10px;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
    font-family: inherit;
    resize: vertical;
    outline: none;
  }
  .setting-textarea:focus {
    border-color: var(--accent);
  }
  .settings-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
  }
  .setting-range {
    width: 100%;
    accent-color: var(--accent);
  }

  .messages-container {
    flex: 1;
    overflow-y: auto;
    padding: 8px 0;
    display: flex;
    flex-direction: column;
    gap: 12px;
    min-height: 0;
  }

  .empty-chat {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex: 1;
    color: var(--text-muted);
    text-align: center;
    padding: 40px 0;
  }
  .empty-icon {
    font-size: 48px;
    margin-bottom: 12px;
    opacity: 0.4;
  }
  .empty-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 4px;
  }
  .empty-hint {
    font-size: 13px;
  }

  .message {
    display: flex;
    gap: 10px;
    max-width: 100%;
  }
  .message.user {
    flex-direction: row-reverse;
  }
  .msg-avatar {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background: var(--bg-card);
    border: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 600;
    flex-shrink: 0;
    color: var(--text-secondary);
  }
  .message.user .msg-avatar {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }
  .msg-content {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 14px;
    max-width: 85%;
    min-width: 60px;
  }
  .message.user .msg-content {
    background: color-mix(in srgb, var(--accent) 20%, var(--bg-card));
    border-color: color-mix(in srgb, var(--accent) 30%, var(--border));
  }
  .msg-text {
    font-size: 14px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .cursor {
    color: var(--accent);
    animation: blink-cursor 0.8s steps(1) infinite;
  }
  @keyframes blink-cursor {
    50% { opacity: 0; }
  }

  .input-area {
    display: flex;
    align-items: flex-end;
    gap: 8px;
    padding: 12px 0 4px;
    flex-shrink: 0;
  }
  .chat-input {
    flex: 1;
    padding: 10px 14px;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 10px;
    color: var(--text-primary);
    font-size: 14px;
    font-family: inherit;
    resize: none;
    outline: none;
    min-height: 42px;
    max-height: 150px;
    line-height: 1.5;
  }
  .chat-input:focus {
    border-color: var(--accent);
  }
  .chat-input::placeholder {
    color: var(--text-muted);
  }
  .chat-input:disabled {
    opacity: 0.6;
  }
  .send-btn {
    width: 42px;
    height: 42px;
    border-radius: 10px;
    background: var(--accent);
    color: #fff;
    border: none;
    font-size: 16px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    transition: all 0.15s;
  }
  .send-btn:hover:not(:disabled) {
    background: var(--accent-hover);
  }
  .send-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  .send-btn.stop {
    background: var(--danger);
  }
  .send-btn.stop:hover {
    background: color-mix(in srgb, var(--danger) 80%, #fff);
  }
</style>
