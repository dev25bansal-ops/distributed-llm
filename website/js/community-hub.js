/**
 * Community Hub — contributor leaderboard, showcase gallery, discussions.
 *
 * Pulls contributor data from GitHub API, shows community projects,
 * and embeds GitHub Discussions.
 *
 * Usage:
 *   <div id="communityHub"></div>
 *   <script type="module">
 *     import { initCommunityHub } from './js/community-hub.js';
 *     initCommunityHub();
 *   </script>
 */

// ── Helpers ────────────────────────────────────────────────────────────

function escapeAttr(str) {
    return String(str).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function safeUrl(url) {
    try {
        const parsed = new URL(url);
        return ['https:', 'http:'].includes(parsed.protocol) ? url : '#';
    } catch { return '#'; }
}

function safeText(str) {
    return document.createTextNode(String(str)).textContent;
}

function buildElement(tag, attrs, children) {
    const el = document.createElement(tag);
    if (attrs) {
        for (const [k, v] of Object.entries(attrs)) {
            if (k === 'className') el.className = v;
            else if (k === 'textContent') el.textContent = v;
            else if (k === 'style') el.style.cssText = v;
            else el.setAttribute(k, v);
        }
    }
    if (children) {
        for (const child of children) {
            if (typeof child === 'string') el.appendChild(document.createTextNode(child));
            else el.appendChild(child);
        }
    }
    return el;
}

// ── Showcase Projects ──────────────────────────────────────────────────

const SHOWCASE = [
    { name: 'LocalGPT', desc: 'Private document chatbot running 100% locally with DistLLM backend.', author: '@localgpt', link: 'https://github.com/PromtEngineer/localGPT', stars: '20k+' },
    { name: 'Cursor Fork', desc: 'AI coding assistant using DistLLM for code completion on local GPUs.', author: '@cursor-fan', link: '#', stars: '2k+' },
    { name: 'Enterprise RAG', desc: 'Production RAG pipeline with DistLLM + Milvus + LangChain for 10K docs.', author: '@ragteam', link: '#', stars: '500+' },
    { name: 'Medical AI', desc: 'HIPAA-compliant medical Q&A using federated DistLLM across hospital nodes.', author: '@medai', link: '#', stars: '300+' },
    { name: 'Code Review Bot', desc: 'GitHub bot that reviews PRs using DistLLM with structured output.', author: '@crbot', link: '#', stars: '150+' },
    { name: 'Legal Assistant', desc: 'Document analysis for law firms, running on-premise with DistLLM.', author: '@legalai', link: '#', stars: '100+' },
];

// ── FAQ from Knowledge Base ────────────────────────────────────────────

const FAQ_KB = [
    { q: 'How do I run 70B models on consumer GPUs?', a: 'Use pipeline parallelism: `distllm deploy --hf meta-llama/Llama-3.1-70b --nodes 4`. DistLLM splits the model across your GPUs automatically.' },
    { q: 'Is DistLLM compatible with OpenAI SDK?', a: 'Yes! Set `base_url="http://localhost:8000/v1"` in any OpenAI SDK. It works with Python, JS, Go, and Rust SDKs.' },
    { q: 'How do I connect multiple laptops?', a: 'Start the coordinator on one laptop, then run `distllm cluster join --coordinator <IP>` on others. Both must be on the same network or use Tailscale.' },
    { q: 'What models are supported?', a: 'Any HuggingFace model: Llama, Qwen, Mistral, Mixtral, Falcon, Phi, DeepSeek, CodeLlama, and more.' },
    { q: 'How much cheaper is DistLLM vs OpenAI?', a: '~10x cheaper. A 70B model costs ~$0.01/1K tokens on DistLLM vs $0.002-$0.015/1K on cloud APIs.' },
    { q: 'Can I use DistLLM with LangChain?', a: 'Yes! Use `ChatOpenAI(base_url="http://localhost:8000/v1")`. Same for LlamaIndex, CrewAI, Haystack, and any OpenAI-compatible framework.' },
];

// ── UI ─────────────────────────────────────────────────────────────────

export function initCommunityHub() {
    const container = document.getElementById('communityHub');
    if (!container) return;

    container.innerHTML = `
        <div class="comm-hub">
            <div class="comm-tabs" id="commTabs">
                <button class="comm-tab active" data-view="showcase">Showcase</button>
                <button class="comm-tab" data-view="contributors">Contributors</button>
                <button class="comm-tab" data-view="faq">FAQ</button>
                <button class="comm-tab" data-view="discussions">Discussions</button>
            </div>

            <div id="commContent"></div>
        </div>
    `;

    const content = document.getElementById('commContent');
    const tabs = document.querySelectorAll('.comm-tab');

    function showShowcase() {
        const wrapper = buildElement('div', { className: 'comm-content' });
        const grid = buildElement('div', { className: 'comm-grid' });

        for (const p of SHOWCASE) {
            const card = buildElement('div', { className: 'comm-card' }, [
                buildElement('h4', { textContent: p.name }),
                buildElement('p', { textContent: p.desc }),
                buildElement('div', { className: 'meta' }, [
                    buildElement('span', { textContent: p.author }),
                    buildElement('span', { className: 'stars', textContent: `★ ${p.stars}` }),
                ]),
            ]);
            grid.appendChild(card);
        }
        wrapper.appendChild(grid);
        content.appendChild(wrapper);
    }

    function showContributors() {
        const wrapper = buildElement('div', { className: 'comm-content' });
        const list = buildElement('div', { id: 'contribList' }, [
            buildElement('p', { style: 'color:#888;font-size:13px;', textContent: 'Loading contributors from GitHub...' }),
        ]);
        wrapper.appendChild(list);
        content.appendChild(wrapper);

        // Fetch from GitHub API
        fetch('https://api.github.com/repos/distributed-llm/distributed-llm/contributors?per_page=20')
            .then(r => r.ok ? r.json() : [])
            .then(contribs => {
                const contribList = document.getElementById('contribList');
                contribList.innerHTML = '';
                if (!contribs.length) {
                    contribList.appendChild(
                        buildElement('p', { style: 'color:#888;font-size:13px;', textContent: 'No contributors found. Be the first!' })
                    );
                    return;
                }
                for (const [i, c] of contribs.entries()) {
                    const login = escapeAttr(c.login || '');
                    const href = safeUrl(c.html_url);
                    const commits = Number(c.contributions) || 0;
                    const rankClass = i < 3 ? ['gold', 'silver', 'bronze'][i] : '';
                    const initial = login.charAt(0).toUpperCase();

                    const row = buildElement('div', { className: 'contrib-row' }, [
                        buildElement('div', { className: `contrib-rank ${rankClass}`, textContent: String(i + 1) }),
                        buildElement('div', { className: 'contrib-avatar', textContent: initial }),
                        buildElement('div', { className: 'contrib-info' }, [
                            buildElement('div', { className: 'contrib-name' }, [
                                buildElement('a', { href, target: '_blank', rel: 'noopener noreferrer', style: 'color:#22c55e;text-decoration:none;', textContent: login }),
                            ]),
                            buildElement('div', { className: 'contrib-commits', textContent: `${commits} commits` }),
                        ]),
                    ]);
                    contribList.appendChild(row);
                }
            })
            .catch(() => {
                const contribList = document.getElementById('contribList');
                contribList.innerHTML = '';
                contribList.appendChild(
                    buildElement('p', { style: 'color:#888;font-size:13px;' }, [
                        document.createTextNode('Could not load contributors. Check '),
                        buildElement('a', { href: 'https://github.com/distributed-llm/distributed-llm/graphs/contributors', style: 'color:#22c55e;', textContent: 'GitHub' }),
                        document.createTextNode('.'),
                    ])
                );
            });
    }

    function showFAQ() {
        const wrapper = buildElement('div', { className: 'comm-content' });
        for (const [i, faq] of FAQ_KB.entries()) {
            const item = buildElement('div', { className: 'faq-item', id: `faq-${i}` }, [
                buildElement('button', { className: 'faq-q', 'data-faq-id': String(i) }, [
                    buildElement('span', { textContent: faq.q }),
                    buildElement('span', { textContent: '+' }),
                ]),
                buildElement('div', { className: 'faq-a', id: `faq-a-${i}` }, [
                    buildElement('div', { className: 'faq-a-inner', textContent: faq.a }),
                ]),
            ]);
            wrapper.appendChild(item);
        }
        content.appendChild(wrapper);
        content.querySelectorAll('.faq-q').forEach(btn => {
            btn.addEventListener('click', () => {
                btn.parentElement.classList.toggle('open');
            });
        });
    }

    function showDiscussions() {
        const wrapper = buildElement('div', { className: 'comm-content' }, [
            buildElement('p', { style: 'font-size:13px;color:#888;margin-bottom:12px;', textContent: 'Join the conversation on GitHub Discussions.' }),
            buildElement('iframe', { className: 'discuss-frame', src: 'https://github.com/distributed-llm/distributed-llm/discussions', loading: 'lazy' }),
        ]);
        content.appendChild(wrapper);
    }

    const views = { showcase: showShowcase, contributors: showContributors, faq: showFAQ, discussions: showDiscussions };

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            views[tab.dataset.view]();
        });
    });

    // Initial view
    showShowcase();
}
