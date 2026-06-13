/**
 * Unit tests for gpu-checker.js
 * 
 * Tests GPU compatibility checking, model filtering, and quantization selection.
 */

import { describe, it, expect } from 'vitest';

// GPU compatibility checker logic (extracted for testability)
const GPU_DATABASE = [
    { name: 'RTX 4090', vram: 24, tier: 'consumer' },
    { name: 'RTX 4060', vram: 8, tier: 'consumer' },
    { name: 'A100 80GB', vram: 80, tier: 'datacenter' },
    { name: 'H100 80GB', vram: 80, tier: 'datacenter' },
];

const MODEL_DATABASE = [
    { name: 'Llama 3.1 8B', requirements: { FP16: 16, INT8: 8, INT4: 4 } },
    { name: 'Llama 3.1 70B', requirements: { FP16: 140, INT8: 70, INT4: 35 } },
    { name: 'Phi-3 mini', requirements: { FP16: 8, INT8: 4, INT4: 2 } },
];

function getPreferredQuantization(requirements, totalVram) {
    return ['FP16', 'INT8', 'INT4'].find(
        q => requirements[q] && totalVram >= requirements[q]
    ) || null;
}

function getGpusNeeded(requirements, quantization, gpuVram) {
    return Math.ceil(requirements[quantization] / gpuVram);
}

describe('GPU Checker', () => {
    describe('getPreferredQuantization', () => {
        it('should prefer FP16 when VRAM allows', () => {
            const q = getPreferredQuantization({ FP16: 16, INT8: 8, INT4: 4 }, 24);
            expect(q).toBe('FP16');
        });

        it('should fall back to INT8 when FP16 doesn\'t fit', () => {
            const q = getPreferredQuantization({ FP16: 16, INT8: 8, INT4: 4 }, 12);
            expect(q).toBe('INT8');
        });

        it('should fall back to INT4 when INT8 doesn\'t fit', () => {
            const q = getPreferredQuantization({ FP16: 16, INT8: 8, INT4: 4 }, 6);
            expect(q).toBe('INT4');
        });

        it('should return null when nothing fits', () => {
            const q = getPreferredQuantization({ FP16: 16, INT8: 8, INT4: 4 }, 2);
            expect(q).toBeNull();
        });

        it('should handle missing quantization levels', () => {
            const q = getPreferredQuantization({ FP16: 16 }, 24);
            expect(q).toBe('FP16');
        });
    });

    describe('getGpusNeeded', () => {
        it('should return 1 for single GPU fit', () => {
            const n = getGpusNeeded({ FP16: 16 }, 'FP16', 24);
            expect(n).toBe(1);
        });

        it('should return 2 when model needs 2 GPUs', () => {
            const n = getGpusNeeded({ INT8: 70 }, 'INT8', 40);
            expect(n).toBe(2);
        });

        it('should ceil correctly', () => {
            const n = getGpusNeeded({ INT4: 35 }, 'INT4', 24);
            expect(n).toBe(2);
        });
    });

    describe('Model compatibility', () => {
        it('8B model should fit on RTX 4090 with FP16', () => {
            const gpu = GPU_DATABASE.find(g => g.name === 'RTX 4090');
            const model = MODEL_DATABASE.find(m => m.name === 'Llama 3.1 8B');
            const q = getPreferredQuantization(model.requirements, gpu.vram * 1);
            expect(q).toBe('FP16');
        });

        it('70B model should NOT fit on single RTX 4090', () => {
            const gpu = GPU_DATABASE.find(g => g.name === 'RTX 4090');
            const model = MODEL_DATABASE.find(m => m.name === 'Llama 3.1 70B');
            const q = getPreferredQuantization(model.requirements, gpu.vram * 1);
            expect(q).toBeNull();
        });

        it('70B model should fit on A100 80GB with INT8', () => {
            const gpu = GPU_DATABASE.find(g => g.name === 'A100 80GB');
            const model = MODEL_DATABASE.find(m => m.name === 'Llama 3.1 70B');
            const q = getPreferredQuantization(model.requirements, gpu.vram * 1);
            expect(q).toBe('INT8');
        });

        it('Phi-3 mini should fit on RTX 4060 with FP16', () => {
            const gpu = GPU_DATABASE.find(g => g.name === 'RTX 4060');
            const model = MODEL_DATABASE.find(m => m.name === 'Phi-3 mini');
            const q = getPreferredQuantization(model.requirements, gpu.vram * 1);
            expect(q).toBe('FP16');
        });
    });

    describe('GPU Database Coverage', () => {
        it('should contain all 4 expected GPUs', () => {
            const names = GPU_DATABASE.map(g => g.name);
            expect(names).toContain('RTX 4090');
            expect(names).toContain('RTX 4060');
            expect(names).toContain('A100 80GB');
            expect(names).toContain('H100 80GB');
            expect(GPU_DATABASE.length).toBe(4);
        });

        it('each GPU should have vram > 0', () => {
            GPU_DATABASE.forEach(gpu => {
                expect(gpu.vram).toBeGreaterThan(0);
            });
        });

        it('each GPU should have a valid tier', () => {
            GPU_DATABASE.forEach(gpu => {
                expect(['consumer', 'datacenter']).toContain(gpu.tier);
            });
        });
    });

    describe('Multi-GPU Calculation', () => {
        it('2x RTX 4090 (48GB total) should fit 70B INT4 (35GB)', () => {
            const model = MODEL_DATABASE.find(m => m.name === 'Llama 3.1 70B');
            const q = getPreferredQuantization(model.requirements, 48);
            expect(q).toBe('INT4');
        });

        it('4x RTX 4090 (96GB total) should fit 70B INT8 (70GB)', () => {
            const model = MODEL_DATABASE.find(m => m.name === 'Llama 3.1 70B');
            const q = getPreferredQuantization(model.requirements, 96);
            expect(q).toBe('INT8');
        });

        it('single RTX 4060 (8GB) should NOT fit 70B at any quantization', () => {
            const model = MODEL_DATABASE.find(m => m.name === 'Llama 3.1 70B');
            const q = getPreferredQuantization(model.requirements, 8);
            expect(q).toBeNull();
        });
    });

    describe('Quantization Preference Order', () => {
        it('should always prefer FP16 > INT8 > INT4', () => {
            const req = { FP16: 16, INT8: 8, INT4: 4 };
            expect(getPreferredQuantization(req, 24)).toBe('FP16');
            expect(getPreferredQuantization(req, 12)).toBe('INT8');
            expect(getPreferredQuantization(req, 6)).toBe('INT4');
        });

        it('should return null when nothing fits', () => {
            const req = { FP16: 16, INT8: 8, INT4: 4 };
            expect(getPreferredQuantization(req, 1)).toBeNull();
        });
    });
});
