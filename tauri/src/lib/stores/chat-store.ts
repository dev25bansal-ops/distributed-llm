import type { ChatMessage, ChatOptions, InferenceMetrics } from "../types";
import { streamChatCompletion } from "../api";

type ChatStoreData = {
  messages: ChatMessage[];
  options: ChatOptions;
  streaming: boolean;
  metrics: InferenceMetrics | null;
  error: string | null;
  abortController: AbortController | null;
};

type Listener = (data: ChatStoreData) => void;

const DEFAULT_OPTIONS: ChatOptions = {
  temperature: 0.7,
  top_p: 0.9,
  max_tokens: 2048,
  system_prompt: "You are a helpful assistant.",
};

const STORAGE_KEY = "distllm-chat-history";

function loadMessages(): ChatMessage[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      return JSON.parse(raw);
    }
  } catch {
    // ignore
  }
  return [];
}

function saveMessages(messages: ChatMessage[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(messages.slice(-100)));
  } catch {
    // ignore
  }
}

let data: ChatStoreData = {
  messages: loadMessages(),
  options: { ...DEFAULT_OPTIONS },
  streaming: false,
  metrics: null,
  error: null,
  abortController: null,
};

let listeners: Listener[] = [];

function notify() {
  for (const fn of listeners) {
    fn({ ...data, messages: [...data.messages] });
  }
}

function msgId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

export const chatStore = {
  subscribe(fn: Listener) {
    listeners.push(fn);
    fn({ ...data, messages: [...data.messages] });
    return () => {
      listeners = listeners.filter((l) => l !== fn);
    };
  },

  getSnapshot(): ChatStoreData {
    return { ...data, messages: [...data.messages] };
  },

  send(content: string, baseUrl: string) {
    if (data.streaming || !content.trim()) return;

    const userMsg: ChatMessage = {
      id: msgId(),
      role: "user",
      content: content.trim(),
      timestamp: Date.now(),
    };

    const assistantMsg: ChatMessage = {
      id: msgId(),
      role: "assistant",
      content: "",
      timestamp: Date.now(),
    };

    data.messages = [...data.messages, userMsg, assistantMsg];
    data.streaming = true;
    data.metrics = null;
    data.error = null;
    saveMessages(data.messages);
    notify();

    const controller = new AbortController();
    data.abortController = controller;

    const buildMessages = (): ChatMessage[] => {
      const systemMsg: ChatMessage = {
        id: "system",
        role: "system",
        content: data.options.system_prompt,
        timestamp: 0,
      };
      const all = [systemMsg, ...data.messages.filter((m) => m.id !== assistantMsg.id)];
      return all;
    };

    streamChatCompletion(baseUrl, buildMessages(), data.options, {
      onToken: (token) => {
        const msgs = [...data.messages];
        const last = msgs[msgs.length - 1];
        if (last && last.role === "assistant") {
          msgs[msgs.length - 1] = { ...last, content: last.content + token };
          data.messages = msgs;
          saveMessages(data.messages);
          notify();
        }
      },
      onDone: (metrics) => {
        data.streaming = false;
        data.abortController = null;
        data.metrics = {
          ttft: metrics.ttft,
          tokens_per_sec: metrics.tokens_per_sec,
          inter_token_latency: metrics.inter_token_latency,
          total_tokens: metrics.total_tokens,
          total_time: metrics.total_time,
        };
        saveMessages(data.messages);
        notify();
      },
      onError: (error) => {
        data.streaming = false;
        data.abortController = null;
        data.error = error;
        // Remove empty assistant message
        data.messages = data.messages.filter(
          (m) => !(m.role === "assistant" && m.content === ""),
        );
        notify();
      },
    }, controller.signal);
  },

  stop() {
    if (data.abortController) {
      data.abortController.abort();
      data.streaming = false;
      data.abortController = null;
      data.metrics = null;
      notify();
    }
  },

  clearMessages() {
    data.messages = [];
    data.metrics = null;
    data.error = null;
    saveMessages(data.messages);
    notify();
  },

  updateOptions(opts: Partial<ChatOptions>) {
    data.options = { ...data.options, ...opts };
    notify();
  },

  getOptions(): ChatOptions {
    return { ...data.options };
  },
};
