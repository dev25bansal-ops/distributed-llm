/**
 * Interactive Quick Start Wizard — platform-aware setup guide with personalized commands.
 *
 * Features:
 * - OS detection + manual override
 * - GPU detection (via GPU checker or manual)
 * - Machine count → cluster config generation
 * - Use case → model recommendation
 * - Personalized command output with copy + download
 *
 * Usage:
 *   <div id="quickstartWizard"></div>
 *   <script type="module">
 *     import { initQuickstartWizard } from './js/quickstart-wizard.js';
 *     initQuickstartWizard();
 *   </script>
 */

import { escapeHtml } from './utils.js';

const STEPS = [
  { title: 'Operating System', key: 'os', description: 'What OS are you using?' },
  { title: 'GPU Availability', key: 'gpu', description: 'Do you have an NVIDIA GPU?' },
  { title: 'Machines', key: 'machines', description: 'How many machines?' },
  { title: 'Use Case', key: 'useCase', description: 'What do you want to run?' },
  { title: 'Your Commands', key: 'done', description: 'Personalized setup guide' },
];

const OS_OPTIONS = [
  { id: 'auto', label: 'Auto-detect', icon: '🖥️' },
  { id: 'windows', label: 'Windows', icon: '🪟' },
  { id: 'macos', label: 'macOS', icon: '🍎' },
  { id: 'linux', label: 'Linux', icon: '🐧' },
];

const GPU_OPTIONS = [
  { id: 'yes', label: 'Yes, NVIDIA GPU', icon: '✅' },
  { id: 'no', label: 'No GPU / Other', icon: '❌' },
  { id: 'unsure', label: 'Not sure', icon: '❓' },
];

const MACHINE_OPTIONS = [
  { id: '1', label: '1 machine', icon: '💻' },
  { id: '2', label: '2 machines', icon: '💻💻' },
  { id: '3', label: '3+ machines', icon: '🏢' },
];

const USECASE_OPTIONS = [
  { id: 'chat', label: 'Chat', icon: '💬', model: 'meta-llama/Llama-3.2-7B' },
  { id: 'code', label: 'Code', icon: '⌨️', model: 'Qwen/Qwen2.5-Coder-7B' },
  { id: 'research', label: 'Research', icon: '🔬', model: 'mistralai/Mistral-7B-Instruct' },
  { id: 'all', label: 'Everything', icon: '🚀', model: 'meta-llama/Llama-3.1-70B' },
];

export function initQuickstartWizard() {
  const container = document.getElementById('quickstartWizard');
  if (!container) return;

  const state = {
    step: 1,
    os: detectOS(),
    gpu: null,
    machines: '1',
    useCase: 'chat',
    model: 'meta-llama/Llama-3.2-7B',
  };

  function detectOS() {
    const p = navigator.platform || '';
    if (p.includes('Win')) return 'windows';
    if (p.includes('Mac')) return 'macos';
    return 'linux';
  }

  function render() {
    container.innerHTML = '';
    const card = document.createElement('div');
    card.className = 'qs-card';

    // Header
    const header = document.createElement('div');
    header.className = 'qs-header';

    // Progress bar
    const progressContainer = document.createElement('div');
    progressContainer.className = 'qs-progress';
    const progressFill = document.createElement('div');
    progressFill.className = 'qs-progress-fill';
    if (state.step <= 4) {
      progressFill.style.width = `${((state.step - 1) / 3) * 100}%`;
    } else {
      progressFill.style.width = '100%';
      progressFill.style.background = 'var(--green)';
    }
    progressContainer.appendChild(progressFill);
    header.appendChild(progressContainer);

    // Step indicator
    const stepLabel = document.createElement('div');
    stepLabel.className = 'qs-step-label';
    if (state.step <= 4) {
      stepLabel.textContent = `Step ${state.step} of 4`;
    } else {
      stepLabel.textContent = 'Ready!';
    }
    header.appendChild(stepLabel);
    card.appendChild(header);
    container.appendChild(card);

    // Content area
    const content = document.createElement('div');
    content.className = 'qs-content';

    if (state.step <= 4) {
      renderStep(content, state);
    } else {
      renderOutput(content, state);
    }

    card.appendChild(content);

    // Navigation
    const nav = document.createElement('div');
    nav.className = 'qs-nav';

    if (state.step > 1) {
      const backBtn = document.createElement('button');
      backBtn.className = 'btn btn-ghost qs-btn';
      backBtn.textContent = '← Back';
      backBtn.addEventListener('click', () => { state.step--; render(); });
      nav.appendChild(backBtn);
    }

    if (state.step < 4) {
      const nextBtn = document.createElement('button');
      nextBtn.className = 'btn btn-primary qs-btn';
      nextBtn.textContent = 'Next →';
      nextBtn.disabled = !canAdvance(state);
      nextBtn.addEventListener('click', () => { if (canAdvance(state)) { state.step++; render(); } });
      nav.appendChild(nextBtn);
    }

    card.appendChild(nav);
  }

  function canAdvance(s) {
    if (s.step === 1) return true; // OS is always detected
    if (s.step === 2) return s.gpu !== null;
    if (s.step === 3) return true;
    return true;
  }

  function renderStep(content, state) {
    const step = STEPS[state.step - 1];
    const desc = document.createElement('p');
    desc.className = 'qs-desc';
    desc.textContent = step.description;
    content.appendChild(desc);

    const options = document.createElement('div');
    options.className = 'qs-options';

    let opts = [];
    if (state.step === 1) opts = OS_OPTIONS;
    else if (state.step === 2) opts = GPU_OPTIONS;
    else if (state.step === 3) opts = MACHINE_OPTIONS;
    else if (state.step === 4) opts = USECASE_OPTIONS;

    for (const opt of opts) {
      const btn = document.createElement('button');
      btn.className = 'qs-option';
      const key = state.step === 1 ? 'os' : state.step === 2 ? 'gpu' : state.step === 3 ? 'machines' : 'useCase';
      if (state[key] === opt.id) btn.classList.add('active');

      btn.innerHTML = `<span class="qs-option-icon">${opt.icon}</span><span class="qs-option-label">${opt.label}</span>`;
      btn.addEventListener('click', () => {
        if (state.step === 1) state.os = opt.id;
        else if (state.step === 2) state.gpu = opt.id;
        else if (state.step === 3) state.machines = opt.id;
        else if (state.step === 4) { state.useCase = opt.id; state.model = opt.model; }
        render();
      });
      options.appendChild(btn);
    }
    content.appendChild(options);

    if (state.step === 1) {
      const detected = document.createElement('p');
      detected.className = 'qs-detected';
      detected.textContent = `Detected: ${state.os.charAt(0).toUpperCase() + state.os.slice(1)}`;
      content.appendChild(detected);
    }
  }

  function renderOutput(content, state) {
    const done = document.createElement('div');
    done.className = 'qs-done';

    const heading = document.createElement('h3');
    heading.textContent = 'Your Personalized Setup';
    done.appendChild(heading);

    const summary = document.createElement('div');
    summary.className = 'qs-summary';
    summary.innerHTML = `
      <span class="qs-summary-tag">${state.os === 'windows' ? '🪟' : state.os === 'macos' ? '🍎' : '🐧'} ${state.os.charAt(0).toUpperCase() + state.os.slice(1)}</span>
      <span class="qs-summary-tag">${state.gpu === 'yes' ? '✅ GPU' : '❌ CPU'}</span>
      <span class="qs-summary-tag">${state.machines === '1' ? '💻 Solo' : state.machines === '2' ? '💻💻 Duo' : '🏢 Cluster'}</span>
      <span class="qs-summary-tag">${USECASE_OPTIONS.find(o => o.id === state.useCase)?.icon} ${state.useCase}</span>
    `;
    done.appendChild(summary);

    const commands = generateCommands(state);
    const codeBlock = document.createElement('div');
    codeBlock.className = 'qs-code';
    codeBlock.innerHTML = `<pre><code>${escapeHtml(commands)}</code></pre>`;
    done.appendChild(codeBlock);

    const actions = document.createElement('div');
    actions.className = 'qs-actions';

    const copyBtn = document.createElement('button');
    copyBtn.className = 'btn btn-primary';
    copyBtn.textContent = '📋 Copy All';
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(commands).then(() => {
        copyBtn.textContent = '✅ Copied!';
        setTimeout(() => { copyBtn.textContent = '📋 Copy All'; }, 2000);
      });
    });
    actions.appendChild(copyBtn);

    const dlBtn = document.createElement('button');
    dlBtn.className = 'btn btn-secondary';
    dlBtn.textContent = '⬇️ Download Script';
    dlBtn.addEventListener('click', () => {
      const ext = state.os === 'windows' ? '.bat' : '.sh';
      const blob = new Blob([state.os === 'windows' ? commands.replace(/\\n/g, '\r\n') : commands], { type: 'text/plain' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `distllm-setup${ext}`;
      a.click();
    });
    actions.appendChild(dlBtn);

    const restartBtn = document.createElement('button');
    restartBtn.className = 'btn btn-ghost';
    restartBtn.textContent = '🔄 Start Over';
    restartBtn.addEventListener('click', () => { state.step = 1; render(); });
    actions.appendChild(restartBtn);

    done.appendChild(actions);
    content.appendChild(done);
  }

  function generateCommands(s) {
    const lines = [];
    const isWin = s.os === 'windows';
    const hasGpu = s.gpu === 'yes';
    const machineCount = parseInt(s.machines, 10) || 1;
    const model = s.model;

    lines.push('# DistLLM Setup');
    lines.push('# Generated for: ' + s.os + ' | GPU: ' + (hasGpu ? 'Yes' : 'No/Other') + ' | Machines: ' + machineCount);
    lines.push('');

    // Install
    if (isWin) {
      lines.push('# Install');
      lines.push('pip install distributed-llm' + (hasGpu ? '[backends]' : ''));
      lines.push('');
      lines.push('# Start coordinator (in one terminal)');
      lines.push('distllm cluster start --model ' + model + ' --qr');
      lines.push('');
      if (machineCount > 1) {
        lines.push('# On worker machines, join the cluster');
        lines.push('distllm cluster join --coordinator <COORDINATOR_IP> --port 50050');
      }
    } else {
      lines.push('# 1. Install');
      lines.push('pip install distributed-llm' + (hasGpu ? '[backends]' : ''));
      lines.push('');
      lines.push('# 2. Start coordinator');
      lines.push('distllm cluster start --model ' + model + ' --qr');
      lines.push('');
      if (machineCount > 1) {
        lines.push('# 3. On each worker machine, join:');
        lines.push('distllm cluster join --coordinator <COORDINATOR_IP> --port 50050');
      }
      lines.push('');
      lines.push('# 4. Test the API');
      lines.push('curl http://localhost:8000/v1/models');
    }

    if (!hasGpu) {
      lines.push('');
      lines.push('# Note: No GPU detected. Using CPU backend (llama.cpp).');
      lines.push('# For better performance, add --quantization bitsandbytes_4bit');
    }

    return lines.join('\n');
  }

  render();
}
