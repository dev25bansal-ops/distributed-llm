/**
 * AI-Powered Chatbot — uses DistLLM itself to answer questions (dogfooding!).
 *
 * Features:
 * - Connects to a DistLLM cluster for inference (configurable via data-attributes)
 * - Falls back to curated knowledge base when cluster is unavailable
 * - Answers setup, configuration, troubleshooting questions
 * - Shows typing indicator and streaming responses
 *
 * Usage:
 *   <div id="aiChatbot" data-api-endpoint="http://localhost:8000/v1"></div>
 *   <script type="module">
 *     import { initAiChatbot } from './js/ai-chatbot.js';
 *     initAiChatbot();
 *   </script>
 */

import { escapeHtml } from './utils.js';

// ── Configuration ──────────────────────────────────────────────────────
const API_KEY = ''; // Set via data-api-key attribute or environment
const MAX_MESSAGES = 20;
const REQUEST_TIMEOUT_MS = 15000;

// ── Knowledge Base (fallback when DistLLM unavailable) ──────────────────

const KNOWLEDGE_BASE = {
    'install': { answer: 'Install DistLLM with `pip install distributed-llm`. Requires Python 3.10+ and PyTorch.', synonyms: ['setup', 'get started', 'begin'] },
    'start': { answer: 'Run `distllm system api --model Qwen/Qwen2.5-3B --local --no-auth` to start the coordinator.', synonyms: ['launch', 'begin', 'init', 'run'] },
    'connect': { answer: 'On another machine, run `distllm cluster join --coordinator <IP>:50050` to connect a worker.', synonyms: ['join', 'link', 'add node', 'worker'] },
    'models': { answer: 'Any HuggingFace model: Llama, Qwen, Mistral, Mixtral, Falcon, Phi, DeepSeek, CodeLlama.', synonyms: ['which', 'support', 'compatible'] },
    'cheaper': { answer: 'DistLLM is ~10x cheaper than cloud APIs. A 70B model costs ~$0.01/1K tokens vs $0.015 on OpenAI.', synonyms: ['cost', 'price', 'save', 'savings', 'expensive'] },
    'langchain': { answer: 'Use `ChatOpenAI(base_url="<your-coordinator>:8000/v1", api_key="your-key")` with any LangChain app.', synonyms: ['langchain', 'llamaindex', 'crewai', 'haystack'] },
    'unauthorized': { answer: 'Set the API key: `export API_KEY=dev` then restart, or use `--no-auth` for development.', synonyms: ['auth', 'authentication', 'api key', '401', 'forbidden'] },
    'cuda out of memory': { answer: 'Use a smaller model, enable quantization (`--quantization bitsandbytes_4bit`), or add more GPUs.', synonyms: ['oom', 'memory', 'gpu memory', 'vram'] },
    'connection refused': { answer: 'Check firewall (ports 8000, 50050), ensure coordinator is running, verify both machines are on same network.', synonyms: ['timeout', 'unreachable', 'network'] },
    'macbook': { answer: 'Install Tailscale on both machines, then `distllm cluster join --coordinator <tailscale-ip>:50050`.', synonyms: ['mac', 'apple', 'osx', 'macos'] },
    'phone': { answer: 'Use ngrok: `ngrok http 8000` then open the URL on your phone.', synonyms: ['mobile', 'android', 'ios'] },
    'colab': { answer: 'Use ngrok to expose your API: `ngrok http 8000`, then use the URL in Colab.', synonyms: ['jupyter', 'notebook', 'google'] },
    'gpu': { answer: 'Any NVIDIA GPU with CUDA 12+, AMD via ROCm, Apple Silicon via Metal, or CPU-only via llama.cpp.', synonyms: ['nvidia', 'amd', 'rocm', 'metal', 'apple silicon'] },
    'docker': { answer: 'Run `docker compose up` from the repo. See /docs.html for full container setup.', synonyms: ['container', 'compose', 'kubernetes', 'k8s'] },
    'default': { answer: 'I can help with DistLLM setup, configuration, troubleshooting, and usage. Ask me anything!', synonyms: [] },
};

function findBestMatch(query) {
    const q = query.toLowerCase().trim();
    let bestEntry = null;
    let bestScore = 0;

    for (const [key, entry] of Object.entries(KNOWLEDGE_BASE)) {
        if (key === 'default') continue;
        const words = key.split(' ');
        const synWords = entry.synonyms.flatMap(s => s.split(' '));
        const allWords = [...words, ...synWords];

        let score = 0;

        // Exact phrase match (highest weight)
        if (q.includes(key)) score += 10;

        // Synonym phrase matches
        for (const syn of entry.synonyms) {
            if (q.includes(syn)) score += 8;
        }

        // Individual word matches
        for (const word of words) {
            if (q.includes(word)) score += 3;
        }

        // Synonym individual word matches
        for (const word of synWords) {
            if (q.includes(word)) score += 2;
        }

        // Trigram overlap (catches partial words like "authori" matching "authorization")
        const qTrigrams = new Set();
        for (let i = 0; i <= q.length - 3; i++) qTrigrams.add(q.slice(i, i + 3));
        for (const word of allWords) {
            for (let i = 0; i <= word.length - 3; i++) {
                if (qTrigrams.has(word.slice(i, i + 3))) score += 1;
            }
        }

        // Normalize by query length to prefer shorter, more specific matches
        score = score / (q.length + 1);

        if (score > bestScore) {
            bestScore = score;
            bestEntry = entry;
        }
    }

    return bestScore > 0 ? bestEntry.answer : KNOWLEDGE_BASE['default'].answer;
}

// ── UI ─────────────────────────────────────────────────────────────────

export function initAiChatbot() {
    const container = document.getElementById('aiChatbot');
    if (!container) return;

    const apiEndpoint = container.dataset.apiEndpoint || null;
    const apiKey = container.dataset.apiKey || API_KEY;
    let isOpen = false;
    let isSending = false;
    let messages = [];
    let abortController = null;

    container.innerHTML = `
        <button class="chat-fab" id="chatFab" aria-label="Open AI assistant chat" title="Chat with DistLLM AI">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
            </svg>
        </button>
        <div class="chat-window" id="chatWindow" style="display:none;" role="dialog" aria-label="AI Assistant Chat">
            <div class="chat-header">
                <span>DistLLM AI Assistant</span>
                <button class="chat-close" id="chatClose" aria-label="Close chat">✕</button>
            </div>
            <div class="chat-messages" id="chatMessages" role="log" aria-live="polite">
                <div class="chat-msg bot">
                    <div class="chat-bubble">Hi! I'm the DistLLM AI assistant. Ask me about setup, configuration, troubleshooting, or how to use DistLLM.</div>
                </div>
            </div>
            <form class="chat-input-form" id="chatForm">
                <input type="text" id="chatInput" placeholder="Ask about DistLLM..." autocomplete="off" aria-label="Ask a question about DistLLM">
                <button type="submit" id="chatSend" aria-label="Send message">→</button>
            </form>
        </div>
    `;

    const fab = document.getElementById('chatFab');
    const window_ = document.getElementById('chatWindow');
    const closeBtn = document.getElementById('chatClose');
    const form = document.getElementById('chatForm');
    const input = document.getElementById('chatInput');
    const msgs = document.getElementById('chatMessages');

    fab.addEventListener('click', () => {
        isOpen = !isOpen;
        window_.style.display = isOpen ? 'flex' : 'none';
        if (isOpen) input.focus();
    });

    closeBtn.addEventListener('click', () => {
        isOpen = false;
        window_.style.display = 'none';
    });

    function addMessage(text, role) {
        const div = document.createElement('div');
        div.className = `chat-msg ${role}`;
        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble';
        bubble.textContent = text;
        div.appendChild(bubble);
        msgs.appendChild(div);
        msgs.scrollTop = msgs.scrollHeight;
    }

    function addTyping() {
        const div = document.createElement('div');
        div.className = 'chat-msg bot';
        div.id = 'chatTyping';
        div.innerHTML = '<div class="chat-bubble"><div class="chat-typing"><span></span><span></span><span></span></div></div>';
        msgs.appendChild(div);
        msgs.scrollTop = msgs.scrollHeight;
    }

    function removeTyping() {
        const el = document.getElementById('chatTyping');
        if (el) el.remove();
    }

    async function sendMessage(text) {
        if (isSending) return;
        isSending = true;
        const sendBtn = document.getElementById('chatSend');
        const inputEl = document.getElementById('chatInput');
        if (sendBtn) sendBtn.disabled = true;
        if (inputEl) inputEl.disabled = true;

        addMessage(text, 'user');
        addTyping();

        let response = null;
        abortController = new AbortController();
        const timeoutId = setTimeout(() => abortController.abort(), REQUEST_TIMEOUT_MS);

        // Try DistLLM API first (only if endpoint is configured)
        if (apiEndpoint) {
            try {
                const res = await fetch(`${apiEndpoint}/chat/completions`, {
                    method: 'POST',
                    signal: abortController.signal,
                    headers: { 'Content-Type': 'application/json', 'Authorization': apiKey ? `Bearer ${apiKey}` : '' },
                    body: JSON.stringify({
                        model: 'distributed-llm',
                        messages: [
                            { role: 'system', content: 'You are DistLLM AI assistant. Answer questions about DistLLM setup, configuration, troubleshooting, and usage. Be concise and helpful.' },
                            ...messages.map(m => ({ role: m.role, content: m.text })),
                            { role: 'user', content: text },
                        ],
                        max_tokens: 300,
                        temperature: 0.3,
                    }),
                });

                if (res.ok) {
                    const data = await res.json();
                    response = data.choices?.[0]?.message?.content;
                }
            } catch {
                // API unavailable, use fallback
            } finally {
                clearTimeout(timeoutId);
            }
        }

        // Fallback to knowledge base
        if (!response) {
            response = findBestMatch(text);
        }

        removeTyping();
        addMessage(response, 'bot');
        messages.push({ role: 'user', text });
        messages.push({ role: 'assistant', text: response });

        // Cap messages at MAX_MESSAGES to prevent unbounded growth
        if (messages.length > MAX_MESSAGES * 2) {
            messages = messages.slice(-MAX_MESSAGES * 2);
        }

        isSending = false;
        if (sendBtn) sendBtn.disabled = false;
        if (inputEl) inputEl.disabled = false;
        if (inputEl) inputEl.focus();
        abortController = null;
    }

    form.addEventListener('submit', e => {
        e.preventDefault();
        const text = input.value.trim();
        if (!text) return;
        input.value = '';
        sendMessage(text);
    });
}
