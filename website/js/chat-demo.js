/**
 * A1. Interactive Chat Demo — Pre-filled prompt with canned streaming response.
 * No backend needed — just JS with pre-recorded responses.
 */
const DEMO_RESPONSES = {
    "What is DistLLM?": "DistLLM is an open-source distributed LLM inference engine that pools GPUs across multiple devices to run large language models. It uses pipeline parallelism to split model layers across machines, enabling you to run 70B+ parameter models on consumer hardware like RTX 4090s and 4060s working together.",
    "How does pipeline parallelism work?": "Pipeline parallelism splits a model's layers across devices. For example, a 70B model with 80 layers could be split across 4 GPUs: layers 0-19 on GPU 0, layers 20-39 on GPU 1, layers 40-59 on GPU 2, and layers 60-79 on GPU 3. Each device processes its layers and passes hidden states to the next. DistLLM handles the coordination, batching, and fault recovery automatically.",
    "Compare DistLLM vs vLLM": "vLLM excels at single-node high-throughput inference with PagedAttention. DistLLM's differentiator is multi-node pipeline parallelism — pooling heterogeneous GPUs across machines. If your model fits on one GPU, vLLM may be faster. If it doesn't, or you want to combine multiple devices, DistLLM is the right choice. DistLLM also supports vLLM as a backend.",
};

const DEFAULT_RESPONSE = "I'm a demo of DistLLM's inference capabilities. In a real setup, this response would come from your distributed GPU cluster running the actual model. Try asking 'What is DistLLM?', 'How does pipeline parallelism work?', or 'Compare DistLLM vs vLLM'.";

export function initChatDemo() {
    const container = document.getElementById('chatDemo');
    if (!container) return;

    container.innerHTML = `
        <div class="chat-demo-window">
            <div class="chat-demo-header">
                <div class="chat-demo-dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
                <span class="chat-demo-title">DistLLM Chat Demo</span>
            </div>
            <div class="chat-demo-messages" id="chatMessages"></div>
            <div class="chat-demo-input-row">
                <input type="text" class="chat-demo-input" id="chatInput" placeholder="Ask about DistLLM..." autocomplete="off">
                <button class="chat-demo-send" id="chatSend">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
                </button>
            </div>
            <div class="chat-demo-suggestions">
                <button class="chat-demo-suggestion" data-q="What is DistLLM?">What is DistLLM?</button>
                <button class="chat-demo-suggestion" data-q="How does pipeline parallelism work?">Pipeline parallelism?</button>
                <button class="chat-demo-suggestion" data-q="Compare DistLLM vs vLLM">vs vLLM?</button>
            </div>
        </div>
    `;

    const input = document.getElementById('chatInput');
    const sendBtn = document.getElementById('chatSend');
    const messages = document.getElementById('chatMessages');

    const sendMessage = () => {
        const q = input.value.trim();
        if (!q) return;
        input.value = '';
        addMessage(q, 'user');
        const response = DEMO_RESPONSES[q] || DEFAULT_RESPONSE;
        setTimeout(() => streamResponse(response), 400);
    };

    input.addEventListener('keydown', e => { if (e.key === 'Enter') sendMessage(); });
    sendBtn.addEventListener('click', sendMessage);

    container.querySelectorAll('.chat-demo-suggestion').forEach(btn => {
        btn.addEventListener('click', () => {
            input.value = btn.dataset.q;
            sendMessage();
        });
    });
}

function addMessage(text, role) {
    const messages = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = `chat-msg chat-msg-${role}`;
    div.textContent = text;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
    return div;
}

function streamResponse(text) {
    const messages = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = 'chat-msg chat-msg-assistant';
    messages.appendChild(div);

    let i = 0;
    const interval = setInterval(() => {
        div.textContent = text.slice(0, i + 1);
        messages.scrollTop = messages.scrollHeight;
        i++;
        if (i >= text.length) clearInterval(interval);
    }, 12);
}
