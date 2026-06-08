/**
 * WASM-Based In-Browser Demo
 *
 * Runs a tiny 0.5B model entirely in the browser using:
 * - ONNX Runtime Web for inference
 * - WebGPU for GPU acceleration (when available)
 * - WASM fallback for CPU inference
 *
 * Features:
 * - No server required
 * - Real-time token generation
 * - WebGPU acceleration when available
 * - Graceful fallback to WASM/CPU
 * - Streaming output
 *
 * Usage:
 *   <div id="browserDemo"></div>
 *   <script type="module">
 *     import { initBrowserDemo } from './js/browser-demo.js';
 *     initBrowserDemo();
 *   </script>
 */

// ── Configuration ──────────────────────────────────────────────────────

const CONFIG = {
    // Model configuration (using a tiny model for demo)
    modelId: 'Xenova/SmolLM-135M-Instruct', // 135M params, small enough for browser
    // Alternative models:
    // 'Xenova/SmolLM-135M-Instruct' - 135M, fast, good quality
    // 'Xenova/SmolLM-360M-Instruct' - 360M, better quality
    // 'Xenova/Phi-3.5-mini-instruct' - 3.8B, requires more memory

    // Generation settings
    maxNewTokens: 100,
    temperature: 0.7,
    topP: 0.9,

    // UI settings
    typingSpeed: 30, // ms per token
};

// ── State ──────────────────────────────────────────────────────────────

let model = null;
let tokenizer = null;
let isGenerating = false;
let webgpuAvailable = false;

// ── Feature Detection ──────────────────────────────────────────────────

async function detectWebGPU() {
    try {
        if (!navigator.gpu) return false;
        const adapter = await navigator.gpu.requestAdapter();
        return !!adapter;
    } catch {
        return false;
    }
}

// ── Model Loading ──────────────────────────────────────────────────────

async function loadModel(progressCallback) {
    // Dynamic import of Transformers.js
    const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0');

    // Configure for WebGPU if available
    webgpuAvailable = await detectWebGPU();

    if (webgpuAvailable) {
        env.backends.onnx.wasm.proxy = false;
        progressCallback('Loading model with WebGPU acceleration...');
    } else {
        progressCallback('Loading model (WASM/CPU mode)...');
    }

    // Load the model as a text generation pipeline
    model = await pipeline('text-generation', CONFIG.modelId, {
        progress_callback: (progress) => {
            if (progress.status === 'downloading') {
                const percent = Math.round(progress.progress || 0);
                progressCallback(`Downloading model: ${percent}%`);
            } else if (progress.status === 'loading') {
                progressCallback('Loading model into memory...');
            }
        },
        device: webgpuAvailable ? 'webgpu' : 'wasm',
        dtype: 'q4', // Use quantized model for smaller size
    });

    progressCallback('Model loaded! Ready to chat.');
    return true;
}

// ── Text Generation ────────────────────────────────────────────────────

async function generateText(prompt, onToken, onComplete) {
    if (!model || isGenerating) return;

    isGenerating = true;

    try {
        // Format as chat
        const messages = [
            { role: 'system', content: 'You are a helpful AI assistant running in the browser.' },
            { role: 'user', content: prompt }
        ];

        // Generate with streaming
        const result = await model(messages, {
            max_new_tokens: CONFIG.maxNewTokens,
            temperature: CONFIG.temperature,
            top_p: CONFIG.topP,
            do_sample: true,
            callback_function: (tokens) => {
                // Stream tokens as they're generated
                const text = model.tokenizer.decode(tokens[0].output_token_ids, { skip_special_tokens: true });
                onToken(text);
            }
        });

        onComplete();
    } catch (e) {
        console.error('[BrowserDemo] Generation error:', e);
        onComplete('Error: ' + e.message);
    } finally {
        isGenerating = false;
    }
}

// ── UI ─────────────────────────────────────────────────────────────────

export function initBrowserDemo() {
    const container = document.getElementById('browserDemo');
    if (!container) return;

    let isLoaded = false;
    let chatHistory = [];

    function render() {
        container.innerHTML = `
            <div class="browser-demo">
                <div class="demo-header">
                    <h3>🌐 In-Browser AI Demo</h3>
                    <div class="demo-badge ${webgpuAvailable ? 'webgpu' : 'wasm'}">
                        ${webgpuAvailable ? '⚡ WebGPU' : '🔧 WASM'}
                    </div>
                </div>

                <div class="demo-info">
                    <p>Run a ${CONFIG.modelId.split('/').pop()} model <strong>entirely in your browser</strong> — no server, no API, no data leaves your device.</p>
                </div>

                <div class="demo-status" id="demoStatus">
                    ${!isLoaded ? `
                        <div class="demo-loading">
                            <button class="demo-load-btn" id="demoLoadBtn">
                                🚀 Load Model (~50MB)
                            </button>
                            <p class="demo-load-hint">First load may take 10-30 seconds</p>
                        </div>
                    ` : `
                        <div class="demo-ready">
                            <span class="demo-ready-icon">✅</span>
                            <span>Model loaded and ready</span>
                        </div>
                    `}
                </div>

                <div class="demo-chat" id="demoChat">
                    ${chatHistory.map(msg => `
                        <div class="demo-msg ${msg.role}">
                            <div class="demo-msg-avatar">${msg.role === 'user' ? '👤' : '🤖'}</div>
                            <div class="demo-msg-content">${msg.content}</div>
                        </div>
                    `).join('')}

                    ${isGenerating ? `
                        <div class="demo-msg assistant">
                            <div class="demo-msg-avatar">🤖</div>
                            <div class="demo-msg-content typing">
                                <span class="typing-dot"></span>
                                <span class="typing-dot"></span>
                                <span class="typing-dot"></span>
                            </div>
                        </div>
                    ` : ''}
                </div>

                <div class="demo-input-area">
                    <input 
                        type="text" 
                        id="demoInput" 
                        placeholder="${isLoaded ? 'Ask anything...' : 'Load the model first...'}"
                        ${!isLoaded ? 'disabled' : ''}
                        ${isGenerating ? 'disabled' : ''}
                    >
                    <button 
                        class="demo-send-btn" 
                        id="demoSendBtn"
                        ${!isLoaded || isGenerating ? 'disabled' : ''}
                    >
                        ➤
                    </button>
                </div>

                <div class="demo-footer">
                    <p>🔒 <strong>Privacy-first</strong>: Everything runs locally. No data is sent to any server.</p>
                    <p>⚡ <strong>${webgpuAvailable ? 'WebGPU' : 'WASM'}</strong>: ${webgpuAvailable ? 'GPU-accelerated inference' : 'CPU-based inference (still fast!)'}</p>
                </div>
            </div>
        `;

        setupEventListeners();
    }

    function setupEventListeners() {
        // Load button
        const loadBtn = container.querySelector('#demoLoadBtn');
        if (loadBtn) {
            loadBtn.addEventListener('click', async () => {
                loadBtn.disabled = true;
                loadBtn.textContent = '⏳ Loading...';

                try {
                    await loadModel((status) => {
                        loadBtn.textContent = status;
                    });

                    isLoaded = true;
                    render();
                } catch (e) {
                    loadBtn.textContent = '❌ Failed: ' + e.message;
                    setTimeout(() => {
                        loadBtn.disabled = false;
                        loadBtn.textContent = '🚀 Try Again';
                    }, 3000);
                }
            });
        }

        // Send button
        const sendBtn = container.querySelector('#demoSendBtn');
        const input = container.querySelector('#demoInput');

        if (sendBtn && input) {
            const sendMessage = async () => {
                const message = input.value.trim();
                if (!message || isGenerating) return;

                // Add user message
                chatHistory.push({ role: 'user', content: message });
                input.value = '';
                render();

                // Generate response
                let fullResponse = '';
                await generateText(
                    message,
                    (token) => {
                        fullResponse = token;
                        // Update the last assistant message in real-time
                        const lastMsg = chatHistory[chatHistory.length - 1];
                        if (lastMsg?.role === 'assistant') {
                            lastMsg.content = fullResponse;
                        } else {
                            chatHistory.push({ role: 'assistant', content: fullResponse });
                        }
                        render();
                    },
                    (error) => {
                        if (error) {
                            chatHistory.push({ role: 'assistant', content: error });
                        }
                        render();
                    }
                );
            };

            sendBtn.addEventListener('click', sendMessage);
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
        }
    }

    render();
}

// ── Export for testing ──────────────────────────────────────────────────

export { detectWebGPU, CONFIG };
