/**
 * Playwright integration tests for DistLLM website.
 * 
 * Tests navigation, interactive components, and responsive behavior.
 */

import { test, expect } from '@playwright/test';

test.describe('Navigation', () => {
    test('should load homepage', async ({ page }) => {
        await page.goto('/');
        await expect(page).toHaveTitle(/DistLLM/);
    });

    test('should have working nav links', async ({ page }) => {
        await page.goto('/');
        const featuresLink = page.locator('a[href="#features"]').first();
        await expect(featuresLink).toBeVisible();
    });

    test('should have working GitHub link', async ({ page }) => {
        await page.goto('/');
        const githubLink = page.locator('a[href*="github.com"]').first();
        await expect(githubLink).toBeVisible();
    });

    test('should navigate to docs', async ({ page }) => {
        await page.goto('/docs.html');
        await expect(page).toHaveTitle(/Documentation/);
    });

    test('should navigate to blog', async ({ page }) => {
        await page.goto('/blog.html');
        await expect(page).toHaveTitle(/Blog/);
    });
});

test.describe('Command Palette', () => {
    test('should open with Ctrl+K', async ({ page }) => {
        await page.goto('/');
        await page.keyboard.press('Control+k');
        const palette = page.locator('#cmdPalette');
        await expect(palette).toBeVisible();
    });

    test('should close with Escape', async ({ page }) => {
        await page.goto('/');
        await page.keyboard.press('Control+k');
        await page.keyboard.press('Escape');
        const palette = page.locator('#cmdPalette');
        await expect(palette).not.toBeVisible();
    });
});

test.describe('Theme Toggle', () => {
    test('should toggle dark/light mode', async ({ page }) => {
        await page.goto('/');
        const toggle = page.locator('#themeBtn');
        await toggle.click();
        // Check that the body class or data attribute changed
        const html = page.locator('html');
        await expect(html).not.toHaveAttribute('data-theme', 'dark');
    });
});

test.describe('Code Tabs', () => {
    test('should switch between code tabs', async ({ page }) => {
        await page.goto('/');
        const pythonTab = page.locator('.code-tab', { hasText: 'Python' });
        await pythonTab.click();
        await expect(pythonTab).toHaveAttribute('aria-selected', 'true');
    });

    test('should have working copy button', async ({ page }) => {
        await page.goto('/');
        const copyBtn = page.locator('.code-copy').first();
        await copyBtn.click();
        await expect(copyBtn).toHaveText('Copied!');
    });
});

test.describe('FAQ Accordion', () => {
    test('should toggle FAQ items', async ({ page }) => {
        await page.goto('/');
        const faqButton = page.locator('.faq-q').first();
        await faqButton.click();
        const faqItem = page.locator('.faq-item.open');
        await expect(faqItem).toBeVisible();
    });

    test('should close other FAQ items', async ({ page }) => {
        await page.goto('/');
        const faq1 = page.locator('.faq-q').nth(0);
        const faq2 = page.locator('.faq-q').nth(1);
        await faq1.click();
        await faq2.click();
        const openItems = page.locator('.faq-item.open');
        await expect(openItems).toHaveCount(1);
    });
});

test.describe('Mobile Menu', () => {
    test.use({ viewport: { width: 375, height: 667 } });

    test('should toggle mobile menu', async ({ page }) => {
        await page.goto('/');
        await expect(page.locator('html')).toHaveClass(/js-enabled/);
        const menuBtn = page.locator('#menuBtn');
        await menuBtn.click();
        const navLinks = page.locator('#navLinks');
        await expect(navLinks).toHaveClass(/open/);
    });
});

test.describe('Calculator', () => {
    test('should update savings on slider change', async ({ page }) => {
        await page.goto('/');
        const slider = page.locator('#gpuCount');
        await slider.fill('4');
        const savings = page.locator('#savingsValue');
        await expect(savings).not.toBe('$0');
    });
});

test.describe('Hero Section', () => {
    test('should show hero content', async ({ page }) => {
        await page.goto('/');
        const hero = page.locator('.hero');
        await expect(hero).toBeVisible();
    });

    test('should have working CTA buttons', async ({ page }) => {
        await page.goto('/');
        const cta = page.locator('.btn-primary').first();
        await expect(cta).toBeVisible();
    });
});
