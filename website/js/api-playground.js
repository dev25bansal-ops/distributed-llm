/**
 * Interactive API Playground — live API explorer with multi-SDK code generation.
 *
 * Lets users modify parameters and see generated code for:
 * - cURL, Python, JavaScript, Go, Rust
 *
 * Usage:
 *   <div id="apiPlayground"></div>
 *   <script type="module">
 *     import { initApiPlayground } from './js/api-playground.js';
 *     initApiPlayground();
 *   </script>
 */

// ── Code Generators ────────────────────────────────────────────────────

function genCurl(p) {
    const msgs = JSON.stringify(p.messages);
    let cmd = `curl -X POST http://localhost:8000/v1/chat/completions \\\n`;
    cmd += `  -H "Content-Type: application/json" \\\n`;
    cmd += `  -H "Authorization: Bearer $API_KEY" \\\n`;
    cmd += `  -d '${JSON.stringify({ model: p.model, messages: p.messages, max_tokens: p.maxTokens, temperature: p.temperature, top_p: p.topP, stream: p.stream }, null, 2)}'`;
    return cmd;
}

function genPython(p) {
    return `from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key",
)

response = client.chat.completions.create(
    model="${p.model}",
    messages=${JSON.stringify(p.messages, null, 4)},
    max_tokens=${p.maxTokens},
    temperature=${p.temperature},
    top_p=${p.topP},
    stream=${p.stream},
)

print(response.choices[0].message.content)`;
}

function genJavaScript(p) {
    return `import OpenAI from "openai";

const client = new OpenAI({
    baseURL: "http://localhost:8000/v1",
    apiKey: "your-api-key",
});

const response = await client.chat.completions.create({
    model: "${p.model}",
    messages: ${JSON.stringify(p.messages, null, 4)},
    max_tokens: ${p.maxTokens},
    temperature: ${p.temperature},
    top_p: ${p.topP},
    stream: ${p.stream},
});

console.log(response.choices[0].message.content);`;
}

function genGo(p) {
    return `package main

import (
    "context"
    "fmt"
    distllm "github.com/distributed-llm/distributed-llm/sdk/go"
)

func main() {
    client := distllm.NewClient("http://localhost:8000", "your-api-key")
    
    resp, err := client.ChatCompletion(context.Background(), &distllm.ChatRequest{
        Model: "${p.model}",
        Messages: []distllm.Message{
            {Role: "user", Content: ${JSON.stringify(p.messages[0]?.content || "Hello")}},
        },
        MaxTokens: ${p.maxTokens},
        Temperature: ${p.temperature},
    })
    if err != nil {
        panic(err)
    }
    fmt.Println(resp.Choices[0].Message.Content)
}`;
}

function genRust(p) {
    return `use distllm_sdk::{Client, ChatRequest, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new("http://localhost:8000", "your-api-key");
    
    let response = client.chat_completion(&ChatRequest {
        model: "${p.model}".to_string(),
        messages: vec![Message::user(${JSON.stringify(p.messages[0]?.content || "Hello")})],
        max_tokens: Some(${p.maxTokens}),
        temperature: Some(${p.temperature}),
        ..Default::default()
    }).await?;
    
    println!("{}", response.choices[0].message.as_ref().unwrap().content);
    Ok(())
}`;
}

const GENERATORS = {
    curl: { label: 'cURL', fn: genCurl },
    python: { label: 'Python', fn: genPython },
    javascript: { label: 'JavaScript', fn: genJavaScript },
    go: { label: 'Go', fn: genGo },
    rust: { label: 'Rust', fn: genRust },
};

// ── UI ─────────────────────────────────────────────────────────────────

export function initApiPlayground() {
    const container = document.getElementById('apiPlayground');
    if (!container) return;

    container.innerHTML = `
        <div class="api-pg">
            <div class="api-pg-left">
                <h3>API Playground</h3>
                <p class="api-pg-desc">Modify parameters and see generated code for any SDK.</p>
                
                <div class="api-pg-field">
                    <label>Endpoint</label>
                    <select id="pgEndpoint">
                        <option value="chat">POST /v1/chat/completions</option>
                        <option value="completion">POST /v1/completions</option>
                        <option value="embedding">POST /v1/embeddings</option>
                    </select>
                </div>

                <div class="api-pg-field">
                    <label>Model</label>
                    <input type="text" id="pgModel" value="Qwen/Qwen2.5-3B">
                </div>

                <div class="api-pg-field">
                    <label>System Prompt</label>
                    <textarea id="pgSystem" rows="2" placeholder="You are a helpful assistant..."></textarea>
                </div>

                <div class="api-pg-field">
                    <label>User Message</label>
                    <textarea id="pgUser" rows="3">Explain distributed computing in simple terms.</textarea>
                </div>

                <div class="api-pg-row">
                    <div class="api-pg-field">
                        <label>Temperature: <span id="pgTempVal">0.7</span></label>
                        <input type="range" id="pgTemp" min="0" max="2" step="0.1" value="0.7">
                    </div>
                    <div class="api-pg-field">
                        <label>Max Tokens: <span id="pgTokensVal">256</span></label>
                        <input type="range" id="pgTokens" min="1" max="4096" step="1" value="256">
                    </div>
                    <div class="api-pg-field">
                        <label>Top P: <span id="pgTopPVal">0.9</span></label>
                        <input type="range" id="pgTopP" min="0" max="1" step="0.05" value="0.9">
                    </div>
                </div>

                <div class="api-pg-field">
                    <label><input type="checkbox" id="pgStream"> Stream response</label>
                </div>
            </div>

            <div class="api-pg-right">
                <div class="api-pg-tabs" id="pgTabs">
                    <button class="api-pg-tab active" data-sdk="curl">cURL</button>
                    <button class="api-pg-tab" data-sdk="python">Python</button>
                    <button class="api-pg-tab" data-sdk="javascript">JavaScript</button>
                    <button class="api-pg-tab" data-sdk="go">Go</button>
                    <button class="api-pg-tab" data-sdk="rust">Rust</button>
                </div>
                <div class="api-pg-code-wrap">
                    <pre class="api-pg-code" id="pgCode"></pre>
                    <button class="api-pg-copy" id="pgCopy">Copy</button>
                </div>

                <div class="api-pg-response" id="pgResponse">
                    <div class="api-pg-response-header">Response Preview</div>
                    <pre id="pgResponseBody">// Click "Send Request" to see the response</pre>
                </div>
            </div>
        </div>
    `;

    // State
    let activeSdk = 'curl';
    const tabs = document.querySelectorAll('.api-pg-tab');
    const codeEl = document.getElementById('pgCode');
    const copyBtn = document.getElementById('pgCopy');

    function getParams() {
        const messages = [];
        const sys = document.getElementById('pgSystem').value.trim();
        if (sys) messages.push({ role: 'system', content: sys });
        messages.push({ role: 'user', content: document.getElementById('pgUser').value });

        return {
            model: document.getElementById('pgModel').value,
            messages,
            maxTokens: parseInt(document.getElementById('pgTokens').value),
            temperature: parseFloat(document.getElementById('pgTemp').value),
            topP: parseFloat(document.getElementById('pgTopP').value),
            stream: document.getElementById('pgStream').checked,
        };
    }

    function updateCode() {
        const p = getParams();
        const gen = GENERATORS[activeSdk];
        if (gen) codeEl.textContent = gen.fn(p);
    }

    // Tab switching
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            activeSdk = tab.dataset.sdk;
            updateCode();
        });
    });

    // Slider updates
    ['pgTemp', 'pgTokens', 'pgTopP'].forEach(id => {
        const el = document.getElementById(id);
        const valEl = document.getElementById(id + 'Val');
        el.addEventListener('input', () => {
            valEl.textContent = el.value;
            updateCode();
        });
    });

    // Text input updates
    ['pgModel', 'pgSystem', 'pgUser', 'pgStream'].forEach(id => {
        document.getElementById(id).addEventListener('input', updateCode);
    });
    document.getElementById('pgStream').addEventListener('change', updateCode);

    // Copy button
    copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(codeEl.textContent).then(() => {
            copyBtn.textContent = 'Copied!';
            setTimeout(() => { copyBtn.textContent = 'Copy'; }, 2000);
        });
    });

    // Initial render
    updateCode();
}
