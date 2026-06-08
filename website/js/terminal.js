/**
 * Terminal typing animation — lines appear with staggered timing.
 */
import { escapeHtml } from './utils.js';

const LINES = [
    { text: '# Install DistLLM', cls: 't-comment', prompt: false },
    { text: 'pip install distributed-llm', cls: 't-cmd', prompt: true },
    { text: '', cls: '', prompt: false },
    { text: '# Start coordinator on your main machine', cls: 't-comment', prompt: false },
    { text: 'distllm cluster start --model meta-llama/Llama-3.2-7B', cls: 't-cmd', prompt: true },
    { text: '✓ Coordinator started on 0.0.0.0:50050', cls: 't-output', prompt: false },
    { text: '✓ Model loaded: Llama-3.2-7B (14GB, FP16)', cls: 't-output', prompt: false },
    { text: '✓ API server on http://localhost:8000', cls: 't-output', prompt: false },
    { text: '', cls: '', prompt: false },
    { text: '# Join from other devices', cls: 't-comment', prompt: false },
    { text: 'distllm cluster join', cls: 't-cmd', prompt: true },
    { text: '✓ Connected to coordinator at 192.168.1.100', cls: 't-output', prompt: false },
    { text: '✓ Assigned layers 12-17 (RTX 4060, 8GB VRAM)', cls: 't-output', prompt: false },
    { text: '', cls: '', prompt: false },
    { text: '# Query the OpenAI-compatible API', cls: 't-comment', prompt: false },
    { text: 'curl http://localhost:8000/v1/chat/completions \\', cls: 't-cmd', prompt: true },
    { text: '  -d \'{"model":"llama-3.2-7b","messages\":[...]}\'', cls: 't-str', prompt: false },
    { text: '{"choices":[{"message":{"content":"Hi! How can I help?"}}]}', cls: 't-output', prompt: false },
];

export function initTerminal() {
    const body = document.getElementById('terminalBody');
    if (!body) return;

    let animated = false;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && !animated) {
                animated = true;
                animate(body);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });

    observer.observe(body.parentElement);
}

function animate(body) {
    body.innerHTML = '';

    LINES.forEach((ld, i) => {
        const line = document.createElement('span');
        line.className = 'line';
        const esc = escapeHtml(ld.text);

        if (ld.text === '') {
            line.innerHTML = '&nbsp;';
        } else if (ld.prompt) {
            line.innerHTML = `<span class="t-prompt">$</span> <span class="${ld.cls}">${esc}</span>`;
        } else {
            line.innerHTML = `<span class="${ld.cls}">${esc}</span>`;
        }

        body.appendChild(line);
        setTimeout(() => line.classList.add('visible'), i * 120 + 200);
    });

    setTimeout(() => {
        const cursor = document.createElement('span');
        cursor.className = 'terminal-cursor';
        body.lastElementChild.appendChild(cursor);
    }, LINES.length * 120 + 400);
}
