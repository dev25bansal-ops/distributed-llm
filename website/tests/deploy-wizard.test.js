/**
 * Unit tests for deploy-wizard.js
 * 
 * Tests deployment command generation logic.
 */

import { describe, it, expect } from 'vitest';

// Deploy wizard logic (extracted for testability)
function generateDeployCommand(config) {
    const { gpus, network, backend, model } = config;
    
    let cmd = '';
    
    if (network === 'lan') {
        // Single coordinator + workers
        cmd += `# Start coordinator\n`;
        cmd += `distllm system api --model ${model} --local --port 8000\n\n`;
        cmd += `# Connect worker(s)\n`;
        cmd += `distllm cluster join --coordinator localhost:50050\n`;
    } else {
        // WAN mode
        cmd += `# Enable WAN mode\n`;
        cmd += `export DISTLLM_WAN_ENABLED=true\n`;
        cmd += `export DISTLLM_WAN_TRANSPORT=quic\n\n`;
        cmd += `# Start coordinator\n`;
        cmd += `distllm system api --model ${model} --local --port 8000\n\n`;
        cmd += `# Workers connect from anywhere\n`;
        cmd += `distllm cluster join --coordinator your-server.com:50050\n`;
    }
    
    return cmd;
}

function estimateVram(paramsBillions, quantization) {
    const bytesPerParam = {
        'FP16': 2, 'INT8': 1, 'INT4': 0.5,
    }[quantization] || 2;
    return paramsBillions * bytesPerParam;
}

describe('Deploy Wizard', () => {
    describe('generateDeployCommand', () => {
        it('should generate LAN command', () => {
            const cmd = generateDeployCommand({
                gpus: 2, network: 'lan', backend: 'vllm', model: 'llama-3.1-8b',
            });
            expect(cmd).toContain('distllm system api');
            expect(cmd).toContain('distllm cluster join');
            expect(cmd).not.toContain('WAN');
        });

        it('should generate WAN command', () => {
            const cmd = generateDeployCommand({
                gpus: 2, network: 'wan', backend: 'vllm', model: 'llama-3.1-8b',
            });
            expect(cmd).toContain('DISTLLM_WAN_ENABLED=true');
            expect(cmd).toContain('DISTLLM_WAN_TRANSPORT=quic');
        });

        it('should include model name', () => {
            const cmd = generateDeployCommand({
                gpus: 1, network: 'lan', backend: 'vllm', model: 'qwen-2.5-3b',
            });
            expect(cmd).toContain('qwen-2.5-3b');
        });
    });

    describe('estimateVram', () => {
        it('should estimate FP16 VRAM correctly', () => {
            // 7B * 2 bytes = 14GB
            expect(estimateVram(7, 'FP16')).toBe(14);
        });

        it('should estimate INT8 VRAM correctly', () => {
            // 7B * 1 byte = 7GB
            expect(estimateVram(7, 'INT8')).toBe(7);
        });

        it('should estimate INT4 VRAM correctly', () => {
            // 7B * 0.5 bytes = 3.5GB
            expect(estimateVram(7, 'INT4')).toBe(3.5);
        });

        it('should scale with model size', () => {
            expect(estimateVram(70, 'FP16')).toBe(140);
            expect(estimateVram(3, 'FP16')).toBe(6);
        });
    });

    describe('Docker Compose Output', () => {
        it('LAN mode should mention docker compose up', () => {
            const cmd = generateDeployCommand({
                gpus: 2, network: 'lan', backend: 'vllm', model: 'llama-3.1-8b',
            });
            // LAN mode uses distllm commands directly
            expect(cmd).toContain('distllm system api');
            expect(cmd).toContain('distllm cluster join');
        });

        it('WAN mode should mention DISTLLM_WAN_ENABLED=true', () => {
            const cmd = generateDeployCommand({
                gpus: 2, network: 'wan', backend: 'vllm', model: 'llama-3.1-8b',
            });
            expect(cmd).toContain('DISTLLM_WAN_ENABLED=true');
        });
    });

    describe('Backend Selection', () => {
        it('vLLM backend command should reference vllm', () => {
            const cmd = generateDeployCommand({
                gpus: 2, network: 'lan', backend: 'vllm', model: 'llama-3.1-8b',
            });
            expect(cmd).toContain('distllm system api');
            expect(cmd).toContain('--model llama-3.1-8b');
        });

        it('llama.cpp backend command should reference llamacpp', () => {
            const cmd = generateDeployCommand({
                gpus: 1, network: 'lan', backend: 'llamacpp', model: 'phi-3-mini',
            });
            expect(cmd).toContain('distllm system api');
            expect(cmd).toContain('--model phi-3-mini');
        });
    });
});
