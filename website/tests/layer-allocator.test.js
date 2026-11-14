/**
 * Unit tests for the hero widget's layer allocator (index.html).
 *
 * allocateWeights decides how many model layers each device gets, and its
 * result is what the widget uses to say "fits" or "does not fit". It is the one
 * piece of that widget with arithmetic that can be wrong quietly, and it had no
 * test.
 *
 * The function below is copied verbatim from index.html -- if the two drift,
 * these tests stop pinning the shipped behaviour.
 */

import { describe, it, expect } from 'vitest';

function allocateWeights(totalLayers, weights) {
    var sum = 0, i;
    for (i = 0; i < weights.length; i++) { sum += weights[i]; }
    if (sum <= 0) return weights.map(function () { return 0; });
    var out = [], assigned = 0;
    for (i = 0; i < weights.length; i++) {
        var n = Math.floor(totalLayers * (weights[i] / sum));
        out.push(n); assigned += n;
    }
    var fracs = [];
    for (i = 0; i < weights.length; i++) {
        if (weights[i] > 0) fracs.push({ i: i, f: totalLayers * (weights[i] / sum) - out[i] });
    }
    fracs.sort(function (a, b) { return b.f - a.f; });
    var guard = 0;
    while (assigned < totalLayers && fracs.length && guard < totalLayers * 4) {
        out[fracs[guard % fracs.length].i]++; assigned++; guard++;
    }
    return out;
}

const total = (a) => a.reduce((s, v) => s + v, 0);

describe('allocateWeights', () => {
    it('always distributes exactly totalLayers, never more or fewer', () => {
        // The invariant the widget depends on: every layer is placed once.
        for (const layers of [1, 7, 32, 64, 80, 126]) {
            for (const w of [[1], [1, 1], [1, 1, 1], [3, 1], [10, 1, 1], [1, 1, 1, 1]]) {
                expect(total(allocateWeights(layers, w))).toBe(layers);
            }
        }
    });

    it('gives a bigger device more layers than a smaller one', () => {
        const [small, big] = allocateWeights(64, [1, 3]);
        expect(big).toBeGreaterThan(small);
    });

    it('splits evenly when the weights are equal', () => {
        expect(allocateWeights(60, [1, 1, 1])).toEqual([20, 20, 20]);
        expect(allocateWeights(64, [1, 1])).toEqual([32, 32]);
    });

    it('repairs the rounding remainder instead of dropping it', () => {
        // 10 layers over 3 equal devices floors to 3+3+3=9; the missing layer
        // must still be handed out.
        const out = allocateWeights(10, [1, 1, 1]);
        expect(total(out)).toBe(10);
    });

    it('never assigns to a device with zero free memory', () => {
        // A full device must get 0 layers, and the rest must still cover the
        // whole model.
        const out = allocateWeights(32, [0, 1, 1]);
        expect(out[0]).toBe(0);
        expect(out[1] + out[2]).toBe(32);
    });

    it('returns all zeros when every device has no free memory', () => {
        // sum <= 0 is the "nothing fits" path; it must not divide by zero.
        expect(allocateWeights(32, [0, 0])).toEqual([0, 0]);
    });

    it('handles a single device taking everything', () => {
        expect(allocateWeights(80, [14])).toEqual([80]);
    });

    it('gives fewer layers to a device with less free memory after reserve', () => {
        // Mirrors the widget: free = vram - 0.5 reserve.
        const free = [14, 14, 14, 48].map((v) => Math.max(0, v - 0.5));
        const out = allocateWeights(80, free);
        expect(out[3]).toBeGreaterThan(out[0]);
        expect(total(out)).toBe(80);
    });
});
