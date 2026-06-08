/**
 * Visual regression tests — screenshots at multiple breakpoints.
 *
 * Takes screenshots at 320px, 768px, 1024px, 1440px for both themes.
 *
 * FIRST RUN: These tests require baseline screenshots.
 * Run `npx playwright test --update-snapshots` first to generate baselines.
 * Subsequent runs compare against these baselines.
 */

import { test, expect } from '@playwright/test';

const BREAKPOINTS = [
    { name: 'mobile', width: 320, height: 568 },
    { name: 'tablet', width: 768, height: 1024 },
    { name: 'desktop', width: 1024, height: 768 },
    { name: 'wide', width: 1440, height: 900 },
];

test.describe('Visual Regression', () => {
    for (const bp of BREAKPOINTS) {
        test(`homepage at ${bp.name} (${bp.width}px)`, async ({ page }) => {
            await page.setViewportSize({ width: bp.width, height: bp.height });
            await page.goto('/');
            await page.waitForLoadState('networkidle');
            // Allow for font rendering differences and anti-aliasing
            await expect(page).toHaveScreenshot(`homepage-${bp.name}.png`, {
                maxDiffPixels: 500,
                maxDiffPixelRatio: 0.01,
            });
        });

        test(`docs at ${bp.name} (${bp.width}px)`, async ({ page }) => {
            await page.setViewportSize({ width: bp.width, height: bp.height });
            await page.goto('/docs.html');
            await page.waitForLoadState('networkidle');
            await expect(page).toHaveScreenshot(`docs-${bp.name}.png`, {
                maxDiffPixels: 500,
                maxDiffPixelRatio: 0.01,
            });
        });
    }
});
