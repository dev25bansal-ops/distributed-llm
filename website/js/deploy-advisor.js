/**
 * AI-Powered Deploy Advisor
 *
 * Conversational advisor that guides users through deployment decisions.
 * Uses DistLLM API (dogfooding) with intelligent fallback to rule-based advisor.
 *
 * Features:
 * - Multi-step conversation flow
 * - Hardware assessment
 * - Use case analysis
 * - Budget optimization
 * - Security requirements
 * - Custom deployment plan generation
 *
 * Usage:
 *   <div id="deployAdvisor"></div>
 *   <script type="module">
 *     import { initDeployAdvisor } from './js/deploy-advisor.js';
 *     initDeployAdvisor();
 *   </script>
 */

import { escapeHtml } from './utils.js';

// ── Configuration ──────────────────────────────────────────────────────

const CONFIG = {
    apiEndpoint: null, // Set via data-attribute
    maxMessages: 20,
    thinkingDelay: 1000,
};

// ── Conversation State ─────────────────────────────────────────────────

const state = {
    messages: [],
    currentStep: 0,
    userProfile: {
        hardware: null,
        useCase: null,
        budget: null,
        security: null,
        experience: null,
    },
    isTyping: false,
    plan: null,
};

// ── Conversation Flow ──────────────────────────────────────────────────

const CONVERSATION_FLOW = [
    {
        id: 'welcome',
        message: "👋 Hi! I'm the DistLLM Deploy Advisor. I'll help you create the perfect deployment plan for your needs. Let's start with your hardware setup.",
        question: "What GPUs do you have available?",
        options: [
            { id: 'consumer-nvidia', label: 'Consumer NVIDIA (RTX 30/40 series)', icon: '🎮' },
            { id: 'datacenter-nvidia', label: 'Datacenter NVIDIA (A100/H100)', icon: '🖥️' },
            { id: 'amd', label: 'AMD GPUs (ROCm)', icon: '🔴' },
            { id: 'apple', label: 'Apple Silicon (M1/M2/M3)', icon: '🍎' },
            { id: 'mixed', label: 'Mixed/Multiple types', icon: '🔄' },
            { id: 'unsure', label: "Not sure yet / Need to buy", icon: '❓' },
        ],
        field: 'hardware',
    },
    {
        id: 'gpu-count',
        message: "Great choice! How many GPUs do you have or plan to use?",
        question: "Select your GPU configuration:",
        options: [
            { id: '1', label: '1 GPU (Single machine)', icon: '1️⃣' },
            { id: '2-4', label: '2-4 GPUs (Small cluster)', icon: '2️⃣' },
            { id: '5-8', label: '5-8 GPUs (Medium cluster)', icon: '5️⃣' },
            { id: '8+', label: '8+ GPUs (Large cluster)', icon: '♾️' },
        ],
        field: 'gpuCount',
    },
    {
        id: 'use-case',
        message: "Perfect! Now let's understand your primary use case.",
        question: "What will you primarily use DistLLM for?",
        options: [
            { id: 'chat', label: 'Chat/Conversational AI', icon: '💬' },
            { id: 'code', label: 'Code generation/assistance', icon: '💻' },
            { id: 'rag', label: 'RAG (Retrieval Augmented Generation)', icon: '📚' },
            { id: 'agents', label: 'AI Agents/Automation', icon: '🤖' },
            { id: 'fine-tuning', label: 'Fine-tuning/Training', icon: '🎯' },
            { id: 'production', label: 'Production API serving', icon: '🚀' },
        ],
        field: 'useCase',
    },
    {
        id: 'model-size',
        message: "Based on your use case, let's figure out the right model size.",
        question: "What model size are you considering?",
        options: [
            { id: 'small', label: 'Small (1-7B params)', desc: 'Fast, low VRAM', icon: '⚡' },
            { id: 'medium', label: 'Medium (7-13B params)', desc: 'Balanced', icon: '⚖️' },
            { id: 'large', label: 'Large (30-70B params)', desc: 'High quality', icon: '🧠' },
            { id: 'xlarge', label: 'Extra Large (70B+)', desc: 'Best quality', icon: '🏆' },
            { id: 'unsure', label: "Not sure / Need recommendation", icon: '🤔' },
        ],
        field: 'modelSize',
    },
    {
        id: 'budget',
        message: "Let's talk about budget constraints.",
        question: "What's your monthly budget for inference?",
        options: [
            { id: 'free', label: 'Free (Self-hosted only)', icon: '🆓' },
            { id: 'low', label: 'Under $100/month', icon: '💰' },
            { id: 'medium', label: '$100-500/month', icon: '💰💰' },
            { id: 'high', label: '$500-2000/month', icon: '💰💰💰' },
            { id: 'enterprise', label: '$2000+/month', icon: '🏢' },
        ],
        field: 'budget',
    },
    {
        id: 'security',
        message: "Almost there! Let's discuss security requirements.",
        question: "What level of security do you need?",
        options: [
            { id: 'development', label: 'Development (No auth needed)', icon: '🔓' },
            { id: 'staging', label: 'Staging (Basic auth + TLS)', icon: '🔒' },
            { id: 'production', label: 'Production (Full security suite)', icon: '🛡️' },
            { id: 'compliance', label: 'Compliance (HIPAA/SOC2/GDPR)', icon: '📋' },
        ],
        field: 'security',
    },
    {
        id: 'network',
        message: "One more question about your network setup.",
        question: "How will your cluster be connected?",
        options: [
            { id: 'lan', label: 'Same network (LAN/WiFi)', icon: '🏠' },
            { id: 'wan', label: 'Across internet (WAN)', icon: '🌐' },
            { id: 'multi-cloud', label: 'Multi-cloud deployment', icon: '☁️' },
            { id: 'hybrid', label: 'Hybrid (local + cloud)', icon: '🔄' },
        ],
        field: 'network',
    },
    {
        id: 'experience',
        message: "Last question! What's your experience level with LLM deployment?",
        question: "This helps me tailor the instructions:",
        options: [
            { id: 'beginner', label: 'Beginner (First time)', icon: '🌱' },
            { id: 'intermediate', label: 'Intermediate (Some experience)', icon: '📈' },
            { id: 'advanced', label: 'Advanced (Production experience)', icon: '🎓' },
            { id: 'expert', label: 'Expert (Contributed to ecosystem)', icon: '🏆' },
        ],
        field: 'experience',
    },
];

// ── AI Advisor (with fallback) ─────────────────────────────────────────

async function getAIRecommendation(profile) {
    const endpoint = CONFIG.apiEndpoint;

    if (endpoint) {
        try {
            const prompt = buildPrompt(profile);
            const response = await fetch(`${endpoint}/chat/completions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: 'distributed-llm',
                    messages: [
                        {
                            role: 'system',
                            content: `You are DistLLM Deploy Advisor. Create a deployment plan based on user requirements. 
Be concise, practical, and provide specific commands. Format as markdown.`
                        },
                        { role: 'user', content: prompt }
                    ],
                    max_tokens: 1000,
                    temperature: 0.3,
                }),
            });

            if (response.ok) {
                const data = await response.json();
                return data.choices?.[0]?.message?.content;
            }
        } catch (e) {
            console.warn('[DeployAdvisor] API unavailable, using rule-based:', e.message);
        }
    }

    // Fallback to rule-based recommendation
    return generateRuleBasedPlan(profile);
}

function buildPrompt(profile) {
    return `Create a deployment plan for DistLLM with these requirements:

Hardware: ${profile.hardware || 'Not specified'}
GPU Count: ${profile.gpuCount || 'Not specified'}
Use Case: ${profile.useCase || 'General'}
Model Size: ${profile.modelSize || 'Medium'}
Budget: ${profile.budget || 'Self-hosted'}
Security: ${profile.security || 'Development'}
Network: ${profile.network || 'LAN'}
Experience: ${profile.intermediate || 'Intermediate'}

Provide:
1. Recommended model and quantization
2. Docker compose or deployment commands
3. Configuration tips
4. Estimated performance (tokens/sec)
5. Cost breakdown`;
}

function generateRuleBasedPlan(profile) {
    const recommendations = [];

    // Model recommendation based on use case and hardware
    const modelMap = {
        'chat': { small: 'mistral-7b', medium: 'llama-3.1-8b', large: 'llama-3.1-70b', xlarge: 'llama-3.1-70b' },
        'code': { small: 'codellama-7b', medium: 'codellama-13b', large: 'codellama-34b', xlarge: 'codellama-70b' },
        'rag': { small: 'mistral-7b', medium: 'llama-3.1-8b', large: 'llama-3.1-70b', xlarge: 'qwen-72b' },
        'agents': { small: 'mistral-7b', medium: 'llama-3.1-8b', large: 'mixtral-8x7b', xlarge: 'llama-3.1-70b' },
        'fine-tuning': { small: 'mistral-7b', medium: 'llama-3.1-8b', large: 'llama-3.1-70b', xlarge: 'llama-3.1-70b' },
        'production': { small: 'mistral-7b', medium: 'llama-3.1-8b', large: 'llama-3.1-70b', xlarge: 'qwen-72b' },
    };

    const useCase = profile.useCase || 'chat';
    const modelSize = profile.modelSize || 'medium';
    const recommendedModel = modelMap[useCase]?.[modelSize] || 'llama-3.1-8b';

    // Quantization based on hardware
    const quantMap = {
        'consumer-nvidia': 'int4',
        'datacenter-nvidia': 'fp16',
        'amd': 'int8',
        'apple': 'int4',
        'mixed': 'int8',
        'unsure': 'int4',
    };
    const quantization = quantMap[profile.hardware] || 'int4';

    // Security config
    const securityConfig = {
        'development': '--no-auth',
        'staging': '--auth api-key --tls',
        'production': '--auth mtls --tls --cors --rate-limit',
        'compliance': '--auth mtls --tls --cors --rate-limit --audit-log --encryption',
    };
    const secFlag = securityConfig[profile.security] || '--no-auth';

    // Network config
    const networkConfig = {
        'lan': '',
        'wan': '--wan --transport quic',
        'multi-cloud': '--wan --transport quic --nat-traversal',
        'hybrid': '--wan --transport quic',
    };
    const netFlag = networkConfig[profile.network] || '';

    // Build recommendation
    recommendations.push(`## 📋 Your Deployment Plan

### Recommended Configuration
- **Model**: ${recommendedModel}
- **Quantization**: ${quantization.toUpperCase()}
- **Backend**: vLLM (NVIDIA) or llama.cpp (CPU/AMD/Apple)
- **Security**: ${profile.security || 'Development'}
- **Network**: ${profile.network || 'LAN'}`);

    // Docker Compose
    recommendations.push(`### 🐳 Docker Compose

\`\`\`yaml
version: '3.8'
services:
  coordinator:
    image: ghcr.io/distributed-llm/coordinator:latest
    ports:
      - "8000:8000"
      - "50050:50050"
    command: >
      distllm cluster start
        --model ${recommendedModel}
        --quantization ${quantization}
        ${secFlag}
        ${netFlag}
    volumes:
      - ./models:/models

  worker1:
    image: ghcr.io/distributed-llm/worker:latest
    depends_on: [coordinator]
    command: distllm cluster join --coordinator coordinator
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
\`\`\``);

    // Quick start commands
    recommendations.push(`### 🚀 Quick Start Commands

\`\`\`bash
# Install
pip install distributed-llm[backends]

# Start coordinator
distllm cluster start \\
  --model ${recommendedModel} \\
  --quantization ${quantization} \\
  ${secFlag} ${netFlag}

# Join worker (on another machine)
distllm cluster join --coordinator <coordinator-ip>
\`\`\``);

    // Performance estimate
    const perfEstimates = {
        'consumer-nvidia': { int4: '50-90 tok/s', int8: '35-60 tok/s', fp16: '20-40 tok/s' },
        'datacenter-nvidia': { int4: '100-180 tok/s', int8: '70-120 tok/s', fp16: '40-80 tok/s' },
        'amd': { int4: '30-60 tok/s', int8: '20-40 tok/s', fp16: '10-25 tok/s' },
        'apple': { int4: '20-40 tok/s', int8: '15-30 tok/s', fp16: '8-15 tok/s' },
    };
    const perf = perfEstimates[profile.hardware]?.[quantization] || '30-60 tok/s';

    recommendations.push(`### 📊 Expected Performance
- **Throughput**: ${perf} (single GPU)
- **Latency P50**: 15-30ms
- **VRAM Usage**: ~${modelSize === 'small' ? '4-8' : modelSize === 'medium' ? '8-16' : modelSize === 'large' ? '35-70' : '70-140'}GB
- **Concurrent Requests**: 5-20 (depending on batch size)`);

    // Tips based on experience level
    if (profile.experience === 'beginner') {
        recommendations.push(`### 💡 Tips for Beginners
1. Start with a small model (7B) to learn the basics
2. Use INT4 quantization to minimize VRAM requirements
3. Run everything on one machine first before scaling
4. Check the <a href="/docs.html">documentation</a> for detailed guides`);
    }

    return recommendations.join('\n\n');
}

// ── UI Rendering ───────────────────────────────────────────────────────

function renderAdvisor(container) {
    const { messages, currentStep, isTyping, plan } = state;
    const currentQuestion = CONVERSATION_FLOW[currentStep];

    container.innerHTML = `
        <div class="deploy-advisor">
            <div class="advisor-header">
                <h3>🤖 AI Deploy Advisor</h3>
                <div class="advisor-status">
                    <span class="advisor-step">Step ${currentStep + 1}/${CONVERSATION_FLOW.length}</span>
                    <span class="advisor-progress">${Math.round(((currentStep) / CONVERSATION_FLOW.length) * 100)}%</span>
                </div>
                <div class="advisor-progress-bar">
                    <div class="advisor-progress-fill" style="width: ${(currentStep / CONVERSATION_FLOW.length) * 100}%"></div>
                </div>
            </div>

            <div class="advisor-messages" id="advisorMessages">
                ${messages.map(msg => `
                    <div class="advisor-msg ${msg.role}">
                        ${msg.role === 'assistant' ? '<div class="advisor-avatar">🤖</div>' : ''}
                        <div class="advisor-bubble">
                            ${msg.content}
                        </div>
                        ${msg.role === 'user' ? '<div class="advisor-avatar">👤</div>' : ''}
                    </div>
                `).join('')}

                ${isTyping ? `
                    <div class="advisor-msg assistant">
                        <div class="advisor-avatar">🤖</div>
                        <div class="advisor-bubble typing">
                            <span class="typing-dot"></span>
                            <span class="typing-dot"></span>
                            <span class="typing-dot"></span>
                        </div>
                    </div>
                ` : ''}

                ${!isTyping && currentQuestion && !plan ? `
                    <div class="advisor-question">
                        <p class="advisor-question-text">${currentQuestion.question}</p>
                        <div class="advisor-options">
                            ${currentQuestion.options.map(opt => `
                                <button class="advisor-option" data-field="${currentQuestion.field}" data-value="${opt.id}">
                                    <span class="advisor-option-icon">${opt.icon}</span>
                                    <span class="advisor-option-label">${opt.label}</span>
                                    ${opt.desc ? `<span class="advisor-option-desc">${opt.desc}</span>` : ''}
                                </button>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}

                ${plan ? `
                    <div class="advisor-plan">
                        <div class="advisor-plan-content">${plan}</div>
                        <div class="advisor-actions">
                            <button class="advisor-btn advisor-btn-primary" id="advisorCopy">
                                📋 Copy Plan
                            </button>
                            <button class="advisor-btn advisor-btn-secondary" id="advisorRestart">
                                🔄 Start Over
                            </button>
                        </div>
                    </div>
                ` : ''}
            </div>
        </div>
    `;

    // Scroll to bottom
    const messagesEl = container.querySelector('#advisorMessages');
    if (messagesEl) {
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    // Add event listeners
    setupEventListeners(container);
}

function setupEventListeners(container) {
    // Option buttons
    container.querySelectorAll('.advisor-option').forEach(btn => {
        btn.addEventListener('click', async () => {
            const field = btn.dataset.field;
            const value = btn.dataset.value;

            // Update profile
            state.userProfile[field] = value;

            // Add user message
            const selectedOption = CONVERSATION_FLOW[state.currentStep].options.find(o => o.id === value);
            state.messages.push({
                role: 'user',
                content: `${selectedOption.icon} ${selectedOption.label}`,
            });

            // Move to next step
            state.currentStep++;

            // Check if we have all info
            if (state.currentStep >= CONVERSATION_FLOW.length) {
                // Generate plan
                state.isTyping = true;
                renderAdvisor(container);

                await new Promise(r => setTimeout(r, CONFIG.thinkingDelay));

                const plan = await getAIRecommendation(state.userProfile);
                state.plan = plan;
                state.isTyping = false;

                state.messages.push({
                    role: 'assistant',
                    content: "🎉 Here's your personalized deployment plan!",
                });
            } else {
                // Add next question
                state.isTyping = true;
                renderAdvisor(container);

                await new Promise(r => setTimeout(r, CONFIG.thinkingDelay));

                state.messages.push({
                    role: 'assistant',
                    content: CONVERSATION_FLOW[state.currentStep].message,
                });
                state.isTyping = false;
            }

            renderAdvisor(container);
        });
    });

    // Copy button
    const copyBtn = container.querySelector('#advisorCopy');
    if (copyBtn) {
        copyBtn.addEventListener('click', () => {
            const planContent = container.querySelector('.advisor-plan-content');
            if (planContent) {
                navigator.clipboard.writeText(planContent.textContent).then(() => {
                    copyBtn.textContent = '✅ Copied!';
                    setTimeout(() => { copyBtn.textContent = '📋 Copy Plan'; }, 2000);
                });
            }
        });
    }

    // Restart button
    const restartBtn = container.querySelector('#advisorRestart');
    if (restartBtn) {
        restartBtn.addEventListener('click', () => {
            state.messages = [];
            state.currentStep = 0;
            state.userProfile = {};
            state.plan = null;
            state.isTyping = false;
            initConversation();
            renderAdvisor(container);
        });
    }
}

function initConversation() {
    state.messages = [
        {
            role: 'assistant',
            content: CONVERSATION_FLOW[0].message,
        },
    ];
}

// ── Initialization ─────────────────────────────────────────────────────

export function initDeployAdvisor() {
    const container = document.getElementById('deployAdvisor');
    if (!container) return;

    // Get API endpoint from data attribute
    CONFIG.apiEndpoint = container.dataset.apiEndpoint || null;

    // Initialize conversation
    initConversation();

    // Render
    renderAdvisor(container);
}
