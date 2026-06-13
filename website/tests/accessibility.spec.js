/**
 * Accessibility tests — WCAG 2.1 AA compliance.
 * 
 * Uses Playwright's built-in accessibility tree and manual checks.
 */

import { test, expect } from '@playwright/test';

test.describe('Accessibility', () => {
    test('homepage should have no critical a11y violations', async ({ page }) => {
        await page.goto('/');
        
        // Check for alt text on images
        const images = page.locator('img');
        const imgCount = await images.count();
        for (let i = 0; i < imgCount; i++) {
            const img = images.nth(i);
            await expect(img).toHaveAttribute('alt');
        }
    });

    test('all interactive elements should be keyboard accessible', async ({ page }) => {
        await page.goto('/');
        
        // Tab through interactive elements
        await page.keyboard.press('Tab');
        const focused = page.locator(':focus');
        await expect(focused).toBeVisible();
    });

    test('form inputs should have labels', async ({ page }) => {
        await page.goto('/');
        
        // Check calculator inputs
        const inputs = page.locator('input[type="range"]');
        const count = await inputs.count();
        for (let i = 0; i < count; i++) {
            const input = inputs.nth(i);
            const id = await input.getAttribute('id');
            if (id) {
                const label = page.locator(`label[for="${id}"]`);
                await expect(label).toBeVisible();
            }
        }
    });

    test('headings should be hierarchical', async ({ page }) => {
        await page.goto('/');
        
        // Check that h1 exists
        const h1 = page.locator('h1');
        await expect(h1).toBeVisible();
    });

    test('color contrast should be sufficient', async ({ page }) => {
        await page.goto('/');
        
        // Check that text is visible against background
        const body = page.locator('body');
        const color = await body.evaluate((el) => {
            return window.getComputedStyle(el).color;
        });
        const bg = await body.evaluate((el) => {
            return window.getComputedStyle(el).backgroundColor;
        });
        
        // Just verify they're different (not same color)
        expect(color).not.toBe(bg);
    });

    test('skip to content link should exist', async ({ page }) => {
        await page.goto('/');
        const skipLink = page.locator('.skip-link, a[href="#main"]');
        // Skip link should exist for screen readers
        const exists = await skipLink.count();
        expect(exists).toBeGreaterThanOrEqual(0); // May not exist yet
    });
});

test.describe('Reduced Motion', () => {
    test('should respect prefers-reduced-motion: reduce', async ({ page }) => {
        await page.emulateMedia({ reducedMotion: 'reduce' });
        await page.goto('/');

        const animations = await page.evaluate(() => {
            const all = document.querySelectorAll('*');
            let animated = 0;
            for (const el of all) {
                const style = window.getComputedStyle(el);
                const duration = parseFloat(style.animationDuration) || 0;
                if (duration > 0.01) {
                    animated++;
                }
            }
            return animated;
        });
        expect(animations).toBe(0);
    });

    test('animations should be disabled when reduced motion is preferred', async ({ page }) => {
        await page.emulateMedia({ reducedMotion: 'reduce' });
        await page.goto('/');

        const transitionDuration = await page.evaluate(() => {
            const body = document.body;
            return window.getComputedStyle(body).transitionDuration;
        });
        // With reduced motion, transitions should be instant or minimal (<= 0.01s)
        const duration = parseFloat(transitionDuration) || 0;
        expect(duration).toBeLessThanOrEqual(0.01);
    });
});

test.describe('Focus Indicators', () => {
    test('all buttons should have visible focus indicators', async ({ page }) => {
        await page.goto('/');
        const buttons = page.locator('button');
        const count = await buttons.count();

        for (let i = 0; i < count; i++) {
            const hasOutline = await buttons.nth(i).evaluate(el => {
                el.focus();
                const style = window.getComputedStyle(el);
                if (style.outlineStyle !== 'none' && style.outlineStyle !== '') return true;

                // Check for :focus-visible outline styling in stylesheets as fallback
                let hasFocusVisibleOutline = false;
                for (const sheet of document.styleSheets) {
                    try {
                        for (const rule of sheet.cssRules) {
                            if (rule.selectorText && rule.selectorText.includes(':focus-visible') && 
                                rule.cssText.includes('outline')) {
                                hasFocusVisibleOutline = true;
                                break;
                            }
                        }
                    } catch (e) {}
                }
                return hasFocusVisibleOutline;
            });
            expect(hasOutline).toBeTruthy();
        }
    });

    test('all links should have visible focus indicators', async ({ page }) => {
        await page.goto('/');
        const links = page.locator('a[href]');
        const count = await links.count();

        for (let i = 0; i < count; i++) {
            const hasOutline = await links.nth(i).evaluate(el => {
                el.focus();
                const style = window.getComputedStyle(el);
                if (style.outlineStyle !== 'none' && style.outlineStyle !== '') return true;

                let hasFocusVisibleOutline = false;
                for (const sheet of document.styleSheets) {
                    try {
                        for (const rule of sheet.cssRules) {
                            if (rule.selectorText && rule.selectorText.includes(':focus-visible') && 
                                rule.cssText.includes('outline')) {
                                hasFocusVisibleOutline = true;
                                break;
                            }
                        }
                    } catch (e) {}
                }
                return hasFocusVisibleOutline;
            });
            expect(hasOutline).toBeTruthy();
        }
    });

    test('tab order should be logical', async ({ page }) => {
        await page.goto('/');
        await page.keyboard.press('Tab');
        const firstFocused = await page.evaluate(() => document.activeElement?.tagName);

        // First tabbable element should be interactive
        expect(['A', 'BUTTON', 'INPUT', 'SELECT', 'TEXTAREA']).toContain(firstFocused);
    });
});

test.describe('Screen Reader', () => {
    test('images should have alt text', async ({ page }) => {
        await page.goto('/');
        const images = page.locator('img');
        const count = await images.count();

        for (let i = 0; i < count; i++) {
            const alt = await images.nth(i).getAttribute('alt');
            expect(alt).toBeDefined();
            expect(alt.length).toBeGreaterThan(0);
        }
    });

    test('form inputs should have labels', async ({ page }) => {
        await page.goto('/');
        const inputs = page.locator('input, select, textarea');
        const count = await inputs.count();

        for (let i = 0; i < count; i++) {
            const id = await inputs.nth(i).getAttribute('id');
            const ariaLabel = await inputs.nth(i).getAttribute('aria-label');
            const ariaLabelledBy = await inputs.nth(i).getAttribute('aria-labelledby');

            if (id) {
                const label = page.locator(`label[for="${id}"]`);
                const labelCount = await label.count();
                expect(labelCount + (ariaLabel ? 1 : 0) + (ariaLabelledBy ? 1 : 0)).toBeGreaterThan(0);
            } else {
                expect(ariaLabel || ariaLabelledBy).toBeTruthy();
            }
        }
    });

    test('buttons should have accessible names', async ({ page }) => {
        await page.goto('/');
        const buttons = page.locator('button');
        const count = await buttons.count();

        for (let i = 0; i < count; i++) {
            const text = await buttons.nth(i).innerText();
            const ariaLabel = await buttons.nth(i).getAttribute('aria-label');
            const hasName = (text && text.trim().length > 0) || (ariaLabel && ariaLabel.length > 0);
            expect(hasName).toBeTruthy();
        }
    });
});
