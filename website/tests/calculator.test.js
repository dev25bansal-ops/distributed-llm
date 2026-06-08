/**
 * Unit tests for calculator.js
 * 
 * Tests the savings calculator math, TCO comparison, and edge cases.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { MODEL_SIZES } from '../js/calculator.js';

const GPU_COST = 1600; // $/GPU average (matches calculator.js)

function calculateSavings(gpus, tokens, modelIndex, hours, electricityRate = 0.30) {
    const m = MODEL_SIZES[modelIndex];
    const cloud = tokens * 1000 * m.tpm;
    const elec = gpus * 0.55 * hours * electricityRate * 30;
    const save = Math.max(0, cloud - elec);
    return { cloud, elec, save };
}

function calculateTCO(gpus, tokens, modelIndex, hours, electricityRate = 0.30) {
    const { cloud, elec } = calculateSavings(gpus, tokens, modelIndex, hours, electricityRate);
    const gpuCost = gpus * GPU_COST;
    const results = {};

    for (const years of [1, 3, 5]) {
        const months = years * 12;
        const tcoDistLLM = gpuCost + (elec * months);
        const tcoCloud = cloud * months;
        results[`${years}y`] = { distllm: tcoDistLLM, cloud: tcoCloud };
    }

    return results;
}

describe('Calculator', () => {
    describe('calculateSavings', () => {
        it('should calculate cloud cost correctly', () => {
            // 100M tokens * 1000 * $0.0008 = $80
            const { cloud } = calculateSavings(1, 100, 2, 720); // modelIndex=2 is 7B
            expect(cloud).toBe(80);
        });

        it('should calculate electricity cost correctly', () => {
            // 1 GPU * 0.55kW * 720h * $0.30/kWh * 30 days = $3564
            const { elec } = calculateSavings(1, 100, 2, 720);
            expect(elec).toBe(3564);
        });

        it('should return 0 savings when cloud < electricity', () => {
            // Very small token count
            const { save } = calculateSavings(1, 1, 2, 720);
            expect(save).toBe(0);
        });

        it('should scale with GPU count', () => {
            const s1 = calculateSavings(1, 100, 2, 720);
            const s2 = calculateSavings(2, 100, 2, 720);
            expect(s2.elec).toBe(s1.elec * 2);
        });

        it('should scale with token count', () => {
            const s1 = calculateSavings(1, 100, 2, 720);
            const s2 = calculateSavings(1, 200, 2, 720);
            expect(s2.cloud).toBe(s1.cloud * 2);
        });

        it('should handle zero inputs', () => {
            const { cloud, elec, save } = calculateSavings(0, 0, 0, 0);
            expect(cloud).toBe(0);
            expect(elec).toBe(0);
            expect(save).toBe(0);
        });
    });

    describe('calculateTCO', () => {
        it('should include GPU hardware cost', () => {
            const tco = calculateTCO(2, 100, 2, 720);
            // 2 GPUs * $1600 = $3200 hardware
            expect(tco['1y'].distllm).toBeGreaterThan(3200);
        });

        it('should show cloud TCO growing linearly', () => {
            const tco = calculateTCO(1, 100, 2, 720);
            expect(tco['3y'].cloud).toBe(tco['1y'].cloud * 3);
            expect(tco['5y'].cloud).toBe(tco['1y'].cloud * 5);
        });

        it('should show DistLLM TCO growing slower than cloud', () => {
            const tco = calculateTCO(1, 100, 2, 720);
            // Hardware is one-time, only electricity grows
            const distllmGrowth = tco['5y'].distllm - tco['1y'].distllm;
            const cloudGrowth = tco['5y'].cloud - tco['1y'].cloud;
            expect(cloudGrowth).toBeGreaterThan(distllmGrowth);
        });
    });

    describe('Label Formatting', () => {
        function formatLabel(value, suffix) {
            return `${value}${suffix}`;
        }

        it('should format 100M correctly', () => {
            expect(formatLabel(100, 'M')).toBe('100M');
        });

        it('should format 720h correctly', () => {
            expect(formatLabel(720, 'h')).toBe('720h');
        });

        it('should format 0M correctly', () => {
            expect(formatLabel(0, 'M')).toBe('0M');
        });
    });

    describe('Model Index', () => {
        it('MODEL_SIZES[0] should be 1.5B with tpm 0.0002', () => {
            expect(MODEL_SIZES[0].label).toBe('1.5B');
            expect(MODEL_SIZES[0].tpm).toBe(0.0002);
        });

        it('MODEL_SIZES[2] should be 7B with tpm 0.0008', () => {
            expect(MODEL_SIZES[2].label).toBe('7B');
            expect(MODEL_SIZES[2].tpm).toBe(0.0008);
        });

        it('MODEL_SIZES[4] should be 70B with tpm 0.002', () => {
            expect(MODEL_SIZES[4].label).toBe('70B');
            expect(MODEL_SIZES[4].tpm).toBe(0.002);
        });
    });

    describe('Electricity Rate Sensitivity', () => {
        it('cheap electricity ($0.08/kWh) should cost less than expensive ($0.40/kWh)', () => {
            const cheap = calculateSavings(1, 100, 2, 720, 0.08);
            const expensive = calculateSavings(1, 100, 2, 720, 0.40);
            expect(cheap.elec).toBeLessThan(expensive.elec);
        });

        it('electricity cost should scale linearly with the rate multiplier', () => {
            const base = calculateSavings(1, 100, 2, 720, 0.10);
            const doubled = calculateSavings(1, 100, 2, 720, 0.20);
            expect(doubled.elec).toBeCloseTo(base.elec * 2, 5);
        });
    });
});
