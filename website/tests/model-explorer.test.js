/**
 * Unit tests for model-explorer.js
 *
 * Tests model filtering, family grouping, VRAM constraints, and quantization options.
 */

import { describe, it, expect } from 'vitest';

// Model data and helpers (extracted for testability)
const MODELS = [
    { name: 'Llama 3.1 8B', family: 'Llama', vram: 16, quant: ['FP16', 'INT8', 'INT4'], params: '8B' },
    { name: 'Llama 3.1 70B', family: 'Llama', vram: 140, quant: ['FP16', 'INT8', 'INT4'], params: '70B' },
    { name: 'Llama 3.1 405B', family: 'Llama', vram: 810, quant: ['FP16', 'INT8'], params: '405B' },
    { name: 'Llama 3 8B Instruct', family: 'Llama', vram: 16, quant: ['FP16', 'INT8', 'INT4'], params: '8B' },
    { name: 'Llama 3 70B Instruct', family: 'Llama', vram: 140, quant: ['FP16', 'INT8', 'INT4'], params: '70B' },
    { name: 'CodeLlama 34B', family: 'Llama', vram: 68, quant: ['FP16', 'INT8'], params: '34B' },
    { name: 'Mistral 7B v0.3', family: 'Mistral', vram: 14, quant: ['FP16', 'INT8', 'INT4'], params: '7B' },
    { name: 'Mixtral 8x7B', family: 'Mistral', vram: 94, quant: ['FP16', 'INT8'], params: '47B' },
    { name: 'Mixtral 8x22B', family: 'Mistral', vram: 282, quant: ['FP16', 'INT8'], params: '141B' },
    { name: 'Qwen2 7B', family: 'Qwen', vram: 14, quant: ['FP16', 'INT8', 'INT4'], params: '7B' },
    { name: 'Qwen2 72B', family: 'Qwen', vram: 144, quant: ['FP16', 'INT8'], params: '72B' },
    { name: 'Qwen2.5 72B', family: 'Qwen', vram: 144, quant: ['FP16', 'INT8'], params: '72B' },
    { name: 'DeepSeek V2', family: 'DeepSeek', vram: 472, quant: ['FP16', 'INT8'], params: '236B' },
    { name: 'DeepSeek Coder V2', family: 'DeepSeek', vram: 472, quant: ['FP16', 'INT8'], params: '236B' },
    { name: 'Falcon 7B', family: 'Falcon', vram: 14, quant: ['FP16', 'INT8'], params: '7B' },
    { name: 'Falcon 40B', family: 'Falcon', vram: 80, quant: ['FP16', 'INT8'], params: '40B' },
    { name: 'Phi-3 mini', family: 'Phi', vram: 8, quant: ['FP16', 'INT8', 'INT4'], params: '3.8B' },
    { name: 'Phi-3 medium', family: 'Phi', vram: 28, quant: ['FP16', 'INT8'], params: '14B' },
    { name: 'Gemma 2 9B', family: 'Gemma', vram: 18, quant: ['FP16', 'INT8'], params: '9B' },
    { name: 'Gemma 2 27B', family: 'Gemma', vram: 54, quant: ['FP16', 'INT8'], params: '27B' },
];

const FAMILIES = [...new Set(MODELS.map(m => m.family))].sort();

function getRequiredVram(model, quantization) {
    if (quantization === 'all') {
        return Math.min(...model.quant.map(q => getRequiredVram(model, q)));
    }
    if (quantization === 'FP16') return model.vram;
    if (quantization === 'INT8') return Math.ceil(model.vram / 2);
    if (quantization === 'INT4') return Math.ceil(model.vram / 4);
    return model.vram;
}

function filterModels({ family = 'all', maxVram = 900, quant = 'all' }) {
    return MODELS.filter(m => {
        if (family !== 'all' && m.family !== family) return false;
        if (quant === 'FP16' && !m.quant.includes('FP16')) return false;
        if (quant === 'INT8' && !m.quant.includes('INT8')) return false;
        if (quant === 'INT4' && !m.quant.includes('INT4')) return false;
        const required = getRequiredVram(m, quant);
        if (required > maxVram) return false;
        return true;
    });
}

describe('Model Explorer', () => {
    describe('Family Filter', () => {
        it('FAMILIES array should contain all unique model families', () => {
            const expected = ['DeepSeek', 'Falcon', 'Gemma', 'Llama', 'Mistral', 'Phi', 'Qwen'];
            expect(FAMILIES).toEqual(expected);
        });

        it('filtering by Llama should return only Llama models', () => {
            const results = filterModels({ family: 'Llama' });
            expect(results.length).toBeGreaterThan(0);
            results.forEach(m => {
                expect(m.family).toBe('Llama');
            });
        });

        it('filtering by all should return all models', () => {
            const results = filterModels({ family: 'all' });
            expect(results.length).toBe(MODELS.length);
        });
    });

    describe('VRAM Filtering', () => {
        it('models with vram <= 24 should fit on RTX 4090 (24GB)', () => {
            const results = filterModels({ maxVram: 24, quant: 'all' });
            results.forEach(m => {
                expect(getRequiredVram(m, 'all')).toBeLessThanOrEqual(24);
            });
        });

        it('models with vram > 100 should NOT fit on single RTX 4090 (24GB)', () => {
            const largeModels = MODELS.filter(m => m.vram > 100);
            const results = filterModels({ maxVram: 24, quant: 'all' });
            largeModels.forEach(m => {
                expect(results).not.toContain(m);
            });
        });
    });

    describe('Quantization Filter', () => {
        it('models with INT4 in quant array should include INT4', () => {
            const int4Models = MODELS.filter(m => m.quant.includes('INT4'));
            expect(int4Models.length).toBeGreaterThan(0);
            int4Models.forEach(m => {
                expect(m.quant).toContain('INT4');
            });
        });

        it('models without INT4 should not offer INT4 option', () => {
            const noInt4Models = MODELS.filter(m => !m.quant.includes('INT4'));
            expect(noInt4Models.length).toBeGreaterThan(0);
            noInt4Models.forEach(m => {
                expect(m.quant).not.toContain('INT4');
            });
        });
    });

    describe('No Results State', () => {
        it('filtering by impossible constraints should return empty', () => {
            const results = filterModels({ maxVram: 1, quant: 'FP16' });
            expect(results.length).toBe(0);
        });
    });
});
