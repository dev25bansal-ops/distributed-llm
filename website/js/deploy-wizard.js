/**
 * A8. Deployment Guide Wizard — Generates custom commands based on user inputs.
 */
import { escapeHtml } from './utils.js';

const BACKENDS = [
    { id: 'vllm', name: 'vLLM', desc: 'Best for NVIDIA GPUs with CUDA' },
    { id: 'llamacpp', name: 'llama.cpp', desc: 'CPU, AMD, Apple Silicon' },
    { id: 'exllama', name: 'ExLlamaV2', desc: 'Fast quantized inference' },
];

export function initDeployWizard() {
    const container = document.getElementById('deployWizard');
    if (!container) return;

    container.innerHTML = `
        <div class="wizard-card">
            <div class="wizard-step" id="wizardStep1">
                <h4>How many GPUs?</h4>
                <div class="wizard-options">
                    <button class="wizard-opt" data-val="1">1 GPU</button>
                    <button class="wizard-opt" data-val="2">2 GPUs</button>
                    <button class="wizard-opt active" data-val="3">3-4 GPUs</button>
                    <button class="wizard-opt" data-val="5">5+ GPUs</button>
                </div>
            </div>
            <div class="wizard-step" id="wizardStep2">
                <h4>Network setup?</h4>
                <div class="wizard-options">
                    <button class="wizard-opt active" data-val="lan">LAN / WiFi</button>
                    <button class="wizard-opt" data-val="wan">Internet (WAN)</button>
                </div>
            </div>
            <div class="wizard-step" id="wizardStep3">
                <h4>Backend?</h4>
                <div class="wizard-options">
                    ${BACKENDS.map((b, i) => `<button class="wizard-opt${i === 0 ? ' active' : ''}" data-val="${b.id}">${b.name}<small>${b.desc}</small></button>`).join('')}
                </div>
            </div>
            <div class="wizard-output" id="wizardOutput"></div>
        </div>
    `;

    const steps = container.querySelectorAll('.wizard-step');
        steps.forEach(step => {
            step.querySelectorAll('.wizard-opt').forEach(btn => {
                // Set initial aria-pressed state
                btn.setAttribute('aria-pressed', btn.classList.contains('active'));
                btn.addEventListener('click', () => {
                    step.querySelectorAll('.wizard-opt').forEach(b => {
                        b.classList.remove('active');
                        b.setAttribute('aria-pressed', 'false');
                    });
                    btn.classList.add('active');
                    btn.setAttribute('aria-pressed', 'true');
                    generateOutput();
                });
            });
        });

    generateOutput();
}

function getVal(id) {
    const active = document.querySelector(`#${id} .wizard-opt.active`);
    return active ? active.dataset.val : '';
}

function generateOutput() {
    const gpus = getVal('wizardStep1');
    const network = getVal('wizardStep2');
    const backend = getVal('wizardStep3');

    const pipExtra = ['vllm', 'llamacpp', 'exllama'].includes(backend) ? '[backends]' : '';
    const wanEnv = network === 'wan' ? 'export DISTLLM_WAN_ENABLED=true\nexport DISTLLM_WAN_TRANSPORT=quic\n\n' : '';
    const gpuCount = gpus === '1' ? 1 : gpus === '2' ? 2 : gpus === '3' ? 4 : 8;

    let dockerCompose = `# docker-compose.yml
services:
  coordinator:
    image: ghcr.io/distributed-llm/coordinator:latest
    ports:
      - "8000:8000"
      - "50050:50050"
    command: distllm cluster start --model meta-llama/Llama-3.2-7B --port 50050 --api-port 8000`;

    for (let i = 1; i < Math.min(gpuCount, 4); i++) {
        dockerCompose += `
  worker${i}:
    image: ghcr.io/distributed-llm/worker:latest
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
              # GPU index auto-detected by container runtime
              # Override with: device_ids: ["0"] for specific GPU
    command: distllm cluster join --coordinator coordinator --port 50050 --node-id worker${i}`;
    }

    // Generate K8s manifest
    let k8sManifest = `# kubernetes-deployment.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: distllm-config
  namespace: distllm
data:
  MODEL_NAME: "meta-llama/Llama-3.2-7B"
  LOG_LEVEL: "INFO"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: distllm-coordinator
  namespace: distllm
spec:
  replicas: 1
  selector:
    matchLabels:
      app: distllm-coordinator
  template:
    metadata:
      labels:
        app: distllm-coordinator
    spec:
      containers:
      - name: coordinator
        image: ghcr.io/distributed-llm/coordinator:latest
        ports:
        - containerPort: 8000
        - containerPort: 50050
        envFrom:
        - configMapRef:
            name: distllm-config
---
apiVersion: v1
kind: Service
metadata:
  name: distllm-coordinator
  namespace: distllm
spec:
  selector:
    app: distllm-coordinator
  ports:
  - port: 8000
    targetPort: 8000
    name: api
  - port: 50050
    targetPort: 50050
    name: grpc
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: distllm-worker
  namespace: distllm
spec:
  replicas: ${Math.min(gpuCount, 4)}
  serviceName: distllm-worker
  selector:
    matchLabels:
      app: distllm-worker
  template:
    metadata:
      labels:
        app: distllm-worker
    spec:
      containers:
      - name: worker
        image: ghcr.io/distributed-llm/worker:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        envFrom:
        - configMapRef:
            name: distllm-config
        command: ["distllm", "cluster", "join", "--coordinator", "distllm-coordinator", "--port", "50050"]`;

    const cliCmd = `# Install
pip install "distributed-llm${pipExtra}"

# Start coordinator
${wanEnv}distllm cluster start --model meta-llama/Llama-3.2-7B --qr

# Join from other machines
distllm cluster join --discover`;

    const output = document.getElementById('wizardOutput');
    if (!output) return;

    output.innerHTML = `
        <div class="wizard-code-tabs">
            <button class="wizard-code-tab active" type="button" data-tab="cli">CLI</button>
            <button class="wizard-code-tab" type="button" data-tab="docker">Docker Compose</button>
            <button class="wizard-code-tab" type="button" data-tab="k8s">Kubernetes</button>
        </div>
        <div class="wizard-code-block">
            <div class="wizard-code-actions">
              <button class="code-copy" type="button" data-copy="cli">Copy CLI</button>
              <button class="code-download" type="button" data-download="docker-compose.yml">Download YAML</button>
            </div>
            <div class="wizard-code-content active" data-wtab="cli"><pre><code>${escapeHtml(cliCmd)}</code></pre></div>
            <div class="wizard-code-content" data-wtab="docker"><pre><code>${escapeHtml(dockerCompose)}</code></pre></div>
            <div class="wizard-code-content" data-wtab="k8s"><pre><code>${escapeHtml(k8sManifest)}</code></pre></div>
        </div>
    `;

    output.querySelectorAll('.wizard-code-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            output.querySelectorAll('.wizard-code-tab').forEach(t => t.classList.remove('active'));
            output.querySelectorAll('.wizard-code-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            output.querySelector(`[data-wtab="${tab.dataset.tab}"]`).classList.add('active');
        });
    });

    const copyBtns = output.querySelectorAll('[data-copy]');
    copyBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const activeCode = output.querySelector('.wizard-code-content.active code');
            const text = activeCode?.textContent || '';
            if (!text) return;
            navigator.clipboard.writeText(text).then(() => {
                btn.textContent = 'Copied!';
                setTimeout(() => { btn.textContent = `Copy ${btn.dataset.copy === 'cli' ? 'CLI' : 'Config'}`; }, 2000);
            });
        });
    });

    const dlBtn = output.querySelector('[data-download]');
    if (dlBtn) {
        dlBtn.addEventListener('click', () => {
            const activeCode = output.querySelector('.wizard-code-content.active pre code');
            const text = activeCode?.textContent || '';
            if (!text) return;
            const blob = new Blob([text], { type: 'text/plain' });
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = dlBtn.dataset.download;
            a.click();
            URL.revokeObjectURL(a.href);
        });
    }
}
