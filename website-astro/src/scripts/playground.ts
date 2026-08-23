// DistLLM Playground island — live requests against a running server.
// All elements are addressed via [data-pg="…"] attributes on the page.

interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

interface State {
  baseUrl: string;
  apiKey: string;
  model: string;
  message: string;
  history: ChatTurn[];
  temperature: number;
  maxTokens: number;
  stream: boolean;
  lang: "curl" | "python" | "javascript";
}

const $ = <T extends HTMLElement>(name: string) =>
  document.querySelector<T>(`[data-pg="${name}"]`);

const state: State = {
  baseUrl: localStorage.getItem("pg-base-url") ?? "http://localhost:8000",
  apiKey: "",
  model: "TinyStories-1M",
  message: "Once upon a time",
  temperature: 0.7,
  maxTokens: 128,
  stream: true,
  lang: "curl" as const,
  history: [] as ChatTurn[],
};

let controller: AbortController | null = null;

function readControls(): void {
  state.baseUrl = ($("baseUrl") as HTMLInputElement)?.value.trim().replace(/\/+$/, "") ?? state.baseUrl;
  state.apiKey = ($("apiKey") as HTMLInputElement)?.value ?? "";
  state.model = ($("model") as HTMLInputElement)?.value.trim() ?? state.model;
  state.message = ($("message") as HTMLTextAreaElement)?.value ?? state.message;
  state.temperature = parseFloat(($("temperature") as HTMLInputElement)?.value ?? "0.7");
  state.maxTokens = parseInt(($("maxTokens") as HTMLInputElement)?.value ?? "128", 10);
  state.stream = ($("stream") as HTMLInputElement)?.checked ?? true;
}

function hydrateFromUrl(): void {
  const q = new URLSearchParams(location.search);
  if (q.has("base")) { state.baseUrl = q.get("base")!; }
  if (q.has("model")) { state.model = q.get("model")!; }
  if (q.has("temp")) { state.temperature = parseFloat(q.get("temp")!); }
  if (q.has("max_tokens")) { state.maxTokens = parseInt(q.get("max_tokens")!, 10); }
  if (q.has("stream")) { state.stream = q.get("stream") === "true"; }

  const base = $("baseUrl") as HTMLInputElement | null;
  const model = $("model") as HTMLInputElement | null;
  const temp = $("temperature") as HTMLInputElement | null;
  const maxTok = $("maxTokens") as HTMLInputElement | null;
  const stream = $("stream") as HTMLInputElement | null;
  if (base) base.value = state.baseUrl;
  if (model) model.value = state.model;
  if (temp) temp.value = String(state.temperature);
  if (maxTok) maxTok.value = String(state.maxTokens);
  if (stream) stream.checked = state.stream;

  history.replaceState(null, "", location.pathname + (q.toString() ? `?${q}` : ""));
}

function body(): string {
  const multiturn = ($("multiturn") as HTMLInputElement | null)?.checked ?? false;
  const messages: ChatTurn[] = multiturn
    ? [...state.history, { role: "user", content: state.message }]
    : [{ role: "user", content: state.message }];
  return JSON.stringify({
    model: state.model,
    messages,
    temperature: state.temperature,
    max_tokens: state.maxTokens,
    stream: state.stream,
  });
}

function headers(): Record<string, string> {
  return {
    "Content-Type": "application/json",
    ...(state.apiKey ? { Authorization: `Bearer ${state.apiKey}` } : {}),
  };
}

function renderSnippet(): void {
  const el = document.querySelector<HTMLElement>('[data-pg="snippet"]');
  if (!el) return;
  const key = state.apiKey ? "sk-…" : "$API_KEY";
  const b = body();
  if (state.lang === "curl") {
    el.textContent =
      `curl ${state.baseUrl}/v1/chat/completions \\\n` +
      `  -H "Authorization: Bearer ${key}" \\\n` +
      `  -H "Content-Type: application/json" \\\n` +
      `  -d '${b.replace(/'/g, "'\\''")}'`;
  } else if (state.lang === "python") {
    el.textContent =
      `from openai import OpenAI\n\n` +
      `client = OpenAI(base_url="${state.baseUrl}/v1", api_key="${key}")\n\n` +
      `stream = client.chat.completions.create(\n` +
      `    model="${state.model}",\n` +
      `    messages=[{"role": "user", "content": ${JSON.stringify(state.message)}}],\n` +
      `    temperature=${state.temperature},\n` +
      `    max_tokens=${state.maxTokens},\n` +
      `    stream=${state.stream ? "True" : "False"},\n` +
      `)\n\nfor chunk in stream:\n    print(chunk.choices[0].delta.content or "", end="")`;
  } else {
    el.textContent =
      `import OpenAI from "openai";\n\n` +
      `const client = new OpenAI({ baseURL: "${state.baseUrl}/v1", apiKey: "${key}" });\n\n` +
      `const stream = await client.chat.completions.create({\n` +
      `  model: "${state.model}",\n` +
      `  messages: [{ role: "user", content: ${JSON.stringify(state.message)} }],\n` +
      `  temperature: ${state.temperature},\n` +
      `  max_tokens: ${state.maxTokens},\n` +
      `  stream: ${state.stream},\n` +
      `});\n\nfor await (const chunk of stream) {\n  process.stdout.write(chunk.choices[0]?.delta?.content ?? "");\n}`;
  }
}

function setStatus(text: string, kind: "idle" | "ok" | "err" | "busy"): void {
  const el = document.querySelector<HTMLElement>('[data-pg="status"]');
  if (!el) return;
  el.textContent = text;
  el.className = "rounded-full px-2.5 py-0.5 font-mono text-xs " + {
    idle: "bg-sunken text-muted",
    ok: "bg-brand-50 text-brand-700",
    err: "bg-pop-100 text-pop-500",
    busy: "bg-accent-100 text-accent-600",
  }[kind];
}

async function send(): Promise<void> {
  readControls();
  localStorage.setItem("pg-base-url", state.baseUrl);

  const output = document.querySelector<HTMLElement>('[data-pg="output"]');
  const stopBtn = $('stop') as HTMLButtonElement | null;
  const sendBtn = $('send') as HTMLButtonElement | null;
  if (!output) return;

  controller = new AbortController();
  if (sendBtn) sendBtn.disabled = true;
  if (stopBtn) stopBtn.disabled = false;
  output.textContent = "";
  setStatus("streaming…", "busy");
  const userMessage = state.message;

  try {
    const res = await fetch(`${state.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: headers(),
      body: body(),
      signal: controller.signal,
    });
    setStatus(`${res.status} ${res.statusText}`, res.ok ? "ok" : "err");
    if (!res.ok || !res.body) {
      const detail = await res.text().catch(() => "");
      output.textContent = `HTTP ${res.status}\n${detail.slice(0, 2000)}`;
      return;
    }

    const recordHistory = (assistant: string) => {
      state.history.push({ role: "user", content: userMessage });
      state.history.push({ role: "assistant", content: assistant });
    };
    const copyBtn = $("copyOut") as HTMLButtonElement | null;
    copyBtn?.removeAttribute("disabled");

    if (!state.stream) {
      const data = await res.json();
      const content = data.choices?.[0]?.message?.content ?? "";
      output.textContent = content || JSON.stringify(data, null, 2);
      recordHistory(content);
      return;
    }
    let streamed = "";

    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
    let buf = "";
    let count = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += value;
      const events = buf.split("\n\n");
      buf = events.pop() ?? "";
      for (const evt of events) {
        for (const line of evt.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (payload === "[DONE]") continue;
          try {
            const chunk = JSON.parse(payload);
            const delta: string | undefined = chunk.choices?.[0]?.delta?.content;
            if (delta) {
              count += 1;
              streamed += delta;
              output.textContent += delta;
              output.scrollTop = output.scrollHeight;
              setStatus(`streaming… ${count} chunks`, "busy");
            }
          } catch {
            /* tolerate keep-alive lines */
          }
        }
      }
    }
    recordHistory(streamed);
    setStatus(`done — ${count} chunks`, "ok");
  } catch (err) {
    if ((err as Error).name === "AbortError") {
      setStatus("aborted", "err");
      output.textContent += "\n[stopped]";
      return;
    }
    setStatus("error", "err");
    output.textContent =
      `${(err as Error).message}\n\n` +
      `Could not reach the server. Check:\n` +
      `• The DistLLM server is running at ${state.baseUrl}\n` +
      `• The server sends Access-Control-Allow-Origin for this page's origin\n` +
      `• http://localhost is exempt from mixed-content blocking in most browsers`;
  } finally {
    controller = null;
    if (sendBtn) sendBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
  }
}

function share(): void {
  readControls();
  const q = new URLSearchParams({
    base: state.baseUrl,
    model: state.model,
    temp: String(state.temperature),
    max_tokens: String(state.maxTokens),
    stream: String(state.stream),
  });
  const url = `${location.origin}${location.pathname}?${q}`;
  navigator.clipboard.writeText(url).then(() => {
    const btn = $('share');
    if (btn) {
      btn.textContent = "Copied!";
      setTimeout(() => (btn.textContent = "Share"), 1500);
    }
  });
  history.replaceState(null, "", location.pathname + `?${q}`);
}

function wireLangTabs(): void {
  const tabs = document.querySelectorAll<HTMLButtonElement>("[data-pg-lang]");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      state.lang = (tab.dataset.pgLang as State["lang"]) ?? "curl";
      tabs.forEach((t) => {
        const active = t === tab;
        t.setAttribute("aria-selected", active ? "true" : "false");
        t.classList.toggle("bg-brand-50", active);
        t.classList.toggle("text-brand-700", active);
        t.classList.toggle("text-muted", !active);
      });
      renderSnippet();
    });
  });
}

export function initPlayground(): void {
  hydrateFromUrl();

  const temp = $("temperature") as HTMLInputElement | null;
  const tempVal = document.querySelector<HTMLElement>('[data-pg="tempVal"]');
  temp?.addEventListener("input", () => {
    state.temperature = parseFloat(temp.value);
    if (tempVal) tempVal.textContent = temp.value;
    renderSnippet();
  });

  const maxTok = $("maxTokens") as HTMLInputElement | null;
  const maxTokVal = document.querySelector<HTMLElement>('[data-pg="maxTokVal"]');
  maxTok?.addEventListener("input", () => {
    state.maxTokens = parseInt(maxTok.value, 10);
    if (maxTokVal) maxTokVal.textContent = maxTok.value;
    renderSnippet();
  });

  ["baseUrl", "apiKey", "model", "message"].forEach((name) => {
    $(name)?.addEventListener("input", () => {
      readControls();
      renderSnippet();
    });
  });

  const stream = $("stream") as HTMLInputElement | null;
  stream?.addEventListener("change", () => {
    state.stream = stream.checked;
    renderSnippet();
  });

  $("send")?.addEventListener("click", send);
  $("share")?.addEventListener("click", share);
  ($('stop') as HTMLButtonElement | null)?.addEventListener("click", () => controller?.abort());
  wireLangTabs();

  // Copy response button
  const copyOut = $("copyOut") as HTMLButtonElement | null;
  copyOut?.addEventListener("click", async () => {
    const output = document.querySelector<HTMLElement>('[data-pg="output"]');
    if (!output) return;
    await navigator.clipboard.writeText(output.textContent ?? "");
    copyOut.textContent = "Copied!";
    setTimeout(() => (copyOut.textContent = "Copy"), 1500);
  });

  // Fetch available models from /v1/models into the datalist.
  const refreshModels = async () => {
    readControls();
    try {
      const res = await fetch(`${state.baseUrl}/v1/models`, { headers: headers() });
      if (!res.ok) return;
      const data = await res.json();
      const list = document.getElementById("pg-models");
      if (!list) return;
      list.innerHTML = "";
      for (const m of data.data ?? []) {
        const opt = document.createElement("option");
        opt.value = m.id;
        list.appendChild(opt);
      }
    } catch {
      /* server not reachable — leave datalist empty */
    }
  };
  $('refreshModels')?.addEventListener("click", refreshModels);
  // Also auto-populate once when the Connection section opens.
  $("baseUrl")?.closest("details")?.addEventListener("toggle", (e) => {
    if ((e.target as HTMLDetailsElement).open) void refreshModels();
  });

  renderSnippet();
}

if (document.querySelector('[data-pg="controls"]')) {
  initPlayground();
}
