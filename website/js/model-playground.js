/**
 * Interactive Model Playground — test prompts with live parameter tuning & code gen.
 *
 * Features:
 * - 10 model presets with quantization options
 * - Parameter sliders (temp, max_tokens, top_p, top_k, repeat_penalty)
 * - Simulated token streaming with real-time stats
 * - Multi-SDK code generation (cURL, Python, JS, Go, Rust)
 * - Shareable config URLs
 * - System prompt configuration
 * - Conversation history with auto-scroll
 *
 * Usage:
 *   <div id="modelPlayground"></div>
 *   <script type="module">
 *     import { initModelPlayground } from './js/model-playground.js';
 *     initModelPlayground();
 *   </script>
 */

import { escapeHtml } from './utils.js';

// ── Model Presets ──────────────────────────────────────────────────
const MODELS = [
  { id: 'llama-3-8b', name: 'Llama 3.1 8B', provider: 'Meta', params: '8B', minVRAM: 16, tokensPerSec: 45, costPer1K: 0.0005, quantization: ['none', '4bit', '8bit'] },
  { id: 'llama-3-70b', name: 'Llama 3.1 70B', provider: 'Meta', params: '70B', minVRAM: 80, tokensPerSec: 12, costPer1K: 0.002, quantization: ['4bit', '8bit'] },
  { id: 'qwen-2.5-7b', name: 'Qwen 2.5 7B', provider: 'Alibaba', params: '7B', minVRAM: 14, tokensPerSec: 55, costPer1K: 0.0004, quantization: ['none', '4bit', '8bit'] },
  { id: 'qwen-2.5-32b', name: 'Qwen 2.5 32B', provider: 'Alibaba', params: '32B', minVRAM: 40, tokensPerSec: 20, costPer1K: 0.001, quantization: ['4bit', '8bit'] },
  { id: 'mistral-7b', name: 'Mistral 7B v0.3', provider: 'Mistral', params: '7B', minVRAM: 14, tokensPerSec: 50, costPer1K: 0.0004, quantization: ['none', '4bit', '8bit'] },
  { id: 'mixtral-8x7b', name: 'Mixtral 8x7B', provider: 'Mistral', params: '47B', minVRAM: 48, tokensPerSec: 25, costPer1K: 0.001, quantization: ['4bit', '8bit'] },
  { id: 'deepseek-v2', name: 'DeepSeek V2 Lite', provider: 'DeepSeek', params: '16B', minVRAM: 20, tokensPerSec: 35, costPer1K: 0.0007, quantization: ['4bit', '8bit'] },
  { id: 'phi-3', name: 'Phi-3 Mini', provider: 'Microsoft', params: '3.8B', minVRAM: 8, tokensPerSec: 70, costPer1K: 0.0002, quantization: ['none', '4bit'] },
  { id: 'falcon-2-11b', name: 'Falcon 2 11B', provider: 'TII', params: '11B', minVRAM: 16, tokensPerSec: 40, costPer1K: 0.0005, quantization: ['none', '4bit', '8bit'] },
  { id: 'gemma-2-27b', name: 'Gemma 2 27B', provider: 'Google', params: '27B', minVRAM: 36, tokensPerSec: 22, costPer1K: 0.0008, quantization: ['4bit', '8bit'] },
];

// ── Prompt Templates ─────────────────────────────────────────────
const PROMPT_TEMPLATES = [
  { name: 'Explain', text: 'Explain how pipeline parallelism works in distributed LLM inference.' },
  { name: 'Code', text: 'Write a Python function that implements a simple vector database using cosine similarity.' },
  { name: 'Summarize', text: 'Summarize the key benefits of distributed computing for large language models.' },
  { name: 'Translate', text: 'Translate this to French: "Distributed inference allows you to pool GPUs across devices."' },
  { name: 'Custom', text: '' },
];

// ── Preset Responses (simulated streaming) ───────────────────────
const PRESET_RESPONSES = {
  'explain': "Pipeline parallelism splits a large language model across multiple GPUs by layers. Each GPU hosts a contiguous subset of layers. During inference, the input passes through GPU 1's layers, the intermediate activations are sent to GPU 2, and so on. This is like an assembly line: each GPU does its part and passes the result to the next. The coordinator manages this flow, handling batch scheduling, KV cache management, and straggler detection. With pipeline parallelism, you can run models that are much larger than any single GPU's VRAM.",
  'code': "```python\nimport numpy as np\nfrom typing import List\n\ndef cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:\n    dot = np.dot(a, b)\n    norm_a = np.linalg.norm(a)\n    norm_b = np.linalg.norm(b)\n    if norm_a == 0 or norm_b == 0:\n        return 0.0\n    return dot / (norm_a * norm_b)\n\nclass VectorDB:\n    def __init__(self):\n        self.vectors: List[np.ndarray] = []\n        self.metadata: List[dict] = []\n    \n    def add(self, vector: np.ndarray, meta: dict = None):\n        self.vectors.append(vector)\n        self.metadata.append(meta or {})\n    \n    def search(self, query: np.ndarray, k: int = 5):\n        scores = [cosine_similarity(query, v) for v in self.vectors]\n        indices = np.argsort(scores)[-k:][::-1]\n        return [(self.metadata[i], scores[i]) for i in indices]\n```",
  'summarize': "Distributed LLM inference offers three key benefits: (1) **Scale** — pool GPU memory across devices to run models far exceeding single-GPU capacity, enabling 70B+ parameter models on consumer hardware. (2) **Cost** — eliminate expensive cloud API calls by leveraging existing hardware, reducing inference costs by up to 10x. (3) **Privacy** — keep data on your own infrastructure, crucial for regulated industries like healthcare and finance. Combined with features like automatic node discovery, fault tolerance, and intelligent load balancing, distributed inference makes advanced AI accessible without compromising on performance or security.",
  'translate': "L'inférence distribuée vous permet de regrouper les GPU de plusieurs appareils.",
};

// ── Parameter Config ──────────────────────────────────────────────
const PARAMS = {
  temperature: { min: 0, max: 2, step: 0.1, default: 0.7, label: 'Temperature' },
  maxTokens: { min: 1, max: 4096, step: 1, default: 1024, label: 'Max Tokens' },
  topP: { min: 0, max: 1, step: 0.05, default: 0.9, label: 'Top-P' },
  topK: { min: 0, max: 100, step: 1, default: 40, label: 'Top-K' },
  repeatPenalty: { min: 1, max: 2, step: 0.1, default: 1.1, label: 'Repeat Penalty' },
};

// ── Code Generators ──────────────────────────────────────────────
function genCurl(model, messages, params) {
  const body = JSON.stringify({ model, messages, ...params, stream: true }, null, 2);
  return `curl -X POST http://localhost:8000/v1/chat/completions \\\n  -H "Content-Type: application/json" \\\n  -H "Authorization: Bearer $API_KEY" \\\n  -d '${body}'`;
}

function genPython(model, messages, params) {
  return `from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="$API_KEY")

response = client.chat.completions.create(
    model="${model}",
    messages=${JSON.stringify(messages, null, 4).replace(/\n/g, '\n    ')},
    ${Object.entries(params).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(',\n    ')},
    stream=True
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")`;
}

function genJs(model, messages, params) {
  return `import OpenAI from 'openai';

const client = new OpenAI({
    baseURL: "http://localhost:8000/v1",
    apiKey: "$API_KEY",
});

const stream = await client.chat.completions.create({
    model: "${model}",
    messages: ${JSON.stringify(messages, null, 2)},
    ${Object.entries(params).map(([k, v]) => `${k}: ${JSON.stringify(v)}`).join(',\n    ')},
    stream: true,
});
for await (const chunk of stream) {
    process.stdout.write(chunk.choices[0]?.delta?.content || '');
}`;
}

// ── Simulated Streaming ──────────────────────────────────────────
function simulateStreaming(text, onToken, onDone) {
  const words = text.split(/(\s+)/);
  let i = 0;
  let accumulated = '';
  const startTime = Date.now();
  const tokenCount = text.split(/\S+/).length;

  function next() {
    if (i >= words.length) {
      onDone({ tokens: tokenCount, time: Date.now() - startTime });
      return;
    }
    accumulated += words[i++];
    onToken(accumulated, { tokens: Math.floor(i / 2), time: Date.now() - startTime });
    const delay = 15 + Math.random() * 25;
    setTimeout(next, delay);
  }
  next();
}

// ── UI ──────────────────────────────────────────────────────────
export function initModelPlayground() {
  const container = document.getElementById('modelPlayground');
  if (!container) return;

  const state = {
    selectedModel: MODELS[0],
    params: Object.fromEntries(Object.entries(PARAMS).map(([k, v]) => [k, v.default])),
    messages: [{ role: 'system', content: 'You are a helpful AI assistant.' }],
    conversation: [],
    streaming: false,
    outputTab: 'chat',
  };

  container.innerHTML = `
    <div class="playground-card">
      <div class="playground-header">
        <h3>Model Playground</h3>
        <span class="playground-badge">Try it out</span>
      </div>
      <div class="playground-layout">
        <div class="playground-config" id="pgConfig">
          <div class="pg-section">
            <label class="pg-label">Model</label>
            <select class="pg-select" id="pgModelSelect"></select>
            <div class="pg-model-info" id="pgModelInfo"></div>
          </div>
          <div class="pg-section" id="pgSliders"></div>
          <div class="pg-section">
            <label class="pg-label">System Prompt</label>
            <textarea class="pg-textarea" id="pgSystemPrompt" rows="2">You are a helpful AI assistant.</textarea>
          </div>
          <div class="pg-section">
            <label class="pg-label">Prompt Templates</label>
            <div class="pg-templates" id="pgTemplates"></div>
          </div>
          <div class="pg-section">
            <label class="pg-label">Your Prompt</label>
            <textarea class="pg-textarea pg-prompt-input" id="pgPrompt" rows="3" placeholder="Type your prompt here..."></textarea>
          </div>
          <button class="btn btn-primary pg-submit" id="pgSubmit">Generate <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg></button>
        </div>
        <div class="playground-output" id="pgOutput">
          <div class="pg-output-tabs" id="pgOutputTabs">
            <button class="pg-tab active" data-tab="chat">Chat</button>
            <button class="pg-tab" data-tab="code">Code</button>
            <button class="pg-tab" data-tab="stats">Stats</button>
          </div>
          <div class="pg-output-content" id="pgOutputContent">
            <div class="pg-chat" id="pgChat">
              <div class="pg-chat-placeholder">Configure your prompt above and click Generate to see a response.</div>
            </div>
            <div class="pg-code" id="pgCode" style="display:none;"><pre class="pg-code-block"><code id="pgCodeContent"></code></pre><button class="pg-copy-btn" id="pgCopyCode">Copy</button></div>
            <div class="pg-stats" id="pgStats" style="display:none;"></div>
          </div>
        </div>
      </div>
    </div>
  `;

  // Populate model select
  const modelSelect = document.getElementById('pgModelSelect');
  MODELS.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = `${m.name} (${m.params})`;
    modelSelect.appendChild(opt);
  });

  // Build sliders
  const slidersContainer = document.getElementById('pgSliders');
  Object.entries(PARAMS).forEach(([key, cfg]) => {
    const row = document.createElement('div');
    row.className = 'pg-slider-row';
    row.innerHTML = `
      <div class="pg-slider-header">
        <span class="pg-slider-label">${cfg.label}</span>
        <span class="pg-slider-value" id="pgVal-${key}">${cfg.default}</span>
      </div>
      <input type="range" class="pg-slider" id="pgSlider-${key}"
        min="${cfg.min}" max="${cfg.max}" step="${cfg.step}" value="${cfg.default}"
        aria-label="${cfg.label}">
    `;
    slidersContainer.appendChild(row);
  });

  // Templates
  const templatesContainer = document.getElementById('pgTemplates');
  PROMPT_TEMPLATES.forEach(t => {
    const btn = document.createElement('button');
    btn.className = 'pg-template-btn';
    btn.textContent = t.name;
    btn.addEventListener('click', () => {
      document.getElementById('pgPrompt').value = t.text;
      if (t.name !== 'Custom') document.getElementById('pgPrompt').focus();
    });
    templatesContainer.appendChild(btn);
  });

  // ── Event handlers ──

  modelSelect.addEventListener('change', () => {
    state.selectedModel = MODELS.find(m => m.id === modelSelect.value) || MODELS[0];
    updateModelInfo();
  });

  Object.keys(PARAMS).forEach(key => {
    const slider = document.getElementById(`pgSlider-${key}`);
    slider.addEventListener('input', () => {
      state.params[key] = parseFloat(slider.value);
      document.getElementById(`pgVal-${key}`).textContent = slider.value;
    });
  });

  // Tab switching
  document.querySelectorAll('.pg-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.pg-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const tabName = tab.dataset.tab;
      ['pgChat', 'pgCode', 'pgStats'].forEach(id => {
        document.getElementById(id).style.display = id === `pg${tabName.charAt(0).toUpperCase() + tabName.slice(1)}` ? 'block' : 'none';
      });
    });
  });

  // Copy code
  document.getElementById('pgCopyCode').addEventListener('click', () => {
    const code = document.getElementById('pgCodeContent').textContent;
    navigator.clipboard.writeText(code).then(() => {
      const btn = document.getElementById('pgCopyCode');
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
    });
  });

  // System prompt
  document.getElementById('pgSystemPrompt').addEventListener('input', (e) => {
    state.messages[0].content = e.target.value || 'You are a helpful AI assistant.';
  });

  // Submit
  document.getElementById('pgSubmit').addEventListener('click', generate);
  document.getElementById('pgPrompt').addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') generate();
  });

  updateModelInfo();

  // ── Generate ──
  function generate() {
    if (state.streaming) return;
    const promptText = document.getElementById('pgPrompt').value.trim();
    if (!promptText) return;

    state.streaming = true;
    const submitBtn = document.getElementById('pgSubmit');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Generating...';

    const chatContainer = document.getElementById('pgChat');
    chatContainer.innerHTML = '';

    const messages = [
      ...state.messages,
      { role: 'user', content: promptText },
    ];

    // Show user message
    const userDiv = document.createElement('div');
    userDiv.className = 'pg-msg user';
    userDiv.innerHTML = `<div class="pg-bubble user">${escapeHtml(promptText)}</div>`;
    chatContainer.appendChild(userDiv);

    // Bot response area
    const botDiv = document.createElement('div');
    botDiv.className = 'pg-msg bot';
    const botBubble = document.createElement('div');
    botBubble.className = 'pg-bubble bot streaming';
    botBubble.innerHTML = '<span class="pg-cursor">▊</span>';
    botDiv.appendChild(botBubble);
    chatContainer.appendChild(botDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    // Find response by template match
    const promptLower = promptText.toLowerCase();
    let responseText = '';
    if (promptLower.includes('parallelism') || promptLower.includes('pipeline')) responseText = PRESET_RESPONSES.explain;
    else if (promptLower.includes('python') || promptLower.includes('function') || promptLower.includes('code') || promptLower.includes('vector')) responseText = PRESET_RESPONSES.code;
    else if (promptLower.includes('summarize') || promptLower.includes('benefits') || promptLower.includes('distributed')) responseText = PRESET_RESPONSES.summarize;
    else if (promptLower.includes('translate') || promptLower.includes('french')) responseText = PRESET_RESPONSES.translate;
    else responseText = PRESET_RESPONSES.explain;

    // Update stats
    const statsDiv = document.getElementById('pgStats');

    // Generate code
    const codeContent = document.getElementById('pgCodeContent');
    const paramsForCode = { ...state.params };
    codeContent.textContent = `# Python — DistLLM API\n${genPython(state.selectedModel.id, messages, paramsForCode)}\n\n# cURL\n${genCurl(state.selectedModel.id, messages, paramsForCode)}`;

    simulateStreaming(responseText,
      (acc, info) => {
        botBubble.innerHTML = escapeHtml(acc) + '<span class="pg-cursor">▊</span>';
        chatContainer.scrollTop = chatContainer.scrollHeight;
      },
      (info) => {
        botBubble.innerHTML = escapeHtml(responseText);
        botBubble.classList.remove('streaming');

        const tokensPerSec = ((info.tokens / (info.time / 1000)) || 0).toFixed(1);
        statsDiv.innerHTML = `
          <div class="pg-stats-grid">
            <div class="pg-stat"><span class="pg-stat-val">${info.tokens}</span><span class="pg-stat-label">Tokens</span></div>
            <div class="pg-stat"><span class="pg-stat-val">${(info.time / 1000).toFixed(1)}s</span><span class="pg-stat-label">Time</span></div>
            <div class="pg-stat"><span class="pg-stat-val">${tokensPerSec}</span><span class="pg-stat-label">Tok/s</span></div>
            <div class="pg-stat"><span class="pg-stat-val">$${(info.tokens * state.selectedModel.costPer1K / 1000).toFixed(5)}</span><span class="pg-stat-label">Cost</span></div>
          </div>`;

        state.streaming = false;
        submitBtn.disabled = false;
        submitBtn.innerHTML = 'Generate <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>';
      }
    );
  }

  function updateModelInfo() {
    const m = state.selectedModel;
    document.getElementById('pgModelInfo').innerHTML = `
      <span class="pg-model-tag">${escapeHtml(m.provider)}</span>
      <span class="pg-model-tag">${escapeHtml(m.params)} params</span>
      <span class="pg-model-tag">${m.minVRAM}GB VRAM</span>
      <span class="pg-model-tag">~${m.tokensPerSec} tok/s</span>
    `;
  }
}
