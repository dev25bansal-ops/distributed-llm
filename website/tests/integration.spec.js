/**
 * Playwright integration tests for DistLLM website.
 *
 * Tests navigation, interactive components, and responsive behavior.
 *
 * These tests target the current static homepage (website/index.html). That page
 * was rewritten from the earlier "code tabs / dark-mode toggle / command palette"
 * design into a single hero + calculator + <details> FAQ layout, and the dark-mode
 * toggle was removed outright (commit c1055e1). Tests asserting those retired
 * controls were removed here rather than kept green by resurrecting the feature.
 */

import { test, expect } from '@playwright/test';

test.describe('Navigation', () => {
    test('should load homepage', async ({ page }) => {
        await page.goto('/');
        await expect(page).toHaveTitle(/DistLLM/);
    });

    test('should have working nav links', async ({ page }) => {
        await page.goto('/');
        // At mobile widths the primary nav collapses behind #menuBtn; open it
        // before asserting visibility so the check is viewport-agnostic.
        const menuBtn = page.locator('#menuBtn');
        if (await menuBtn.isVisible()) {
            await menuBtn.click();
        }
        const nav = page.locator('#mainNav');
        await expect(nav.locator('a[href="docs.html"]').first()).toBeVisible();
    });

    test('should have working GitHub link', async ({ page }) => {
        await page.goto('/');
        const menuBtn = page.locator('#menuBtn');
        if (await menuBtn.isVisible()) {
            await menuBtn.click();
        }
        // There are GitHub links in both the header actions and the nav; assert a
        // visible one exists rather than fixing on the first DOM match, whose
        // visibility differs between desktop and the collapsed mobile header.
        const githubLink = page.locator('a[href*="github.com"]:visible').first();
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

    test('should have a skip-to-content link', async ({ page }) => {
        await page.goto('/');
        const skip = page.locator('a.skip-link[href="#main"]');
        await expect(skip).toBeAttached();
        await expect(page.locator('#main')).toBeAttached();
    });
});

test.describe('Code Tabs', () => {
    test('should switch between install tabs', async ({ page }) => {
        await page.goto('/');
        const dockerTab = page.locator('#tab-docker');
        const pipTab = page.locator('#tab-pip');
        await expect(pipTab).toHaveAttribute('aria-selected', 'true');
        await dockerTab.click();
        await expect(dockerTab).toHaveAttribute('aria-selected', 'true');
        await expect(pipTab).toHaveAttribute('aria-selected', 'false');
        await expect(page.locator('#installCmd')).toContainText('docker run');
    });

    test('should have working copy button', async ({ page }) => {
        await page.goto('/');
        const copyBtn = page.locator('#copyBtn');
        await copyBtn.click();
        await expect(copyBtn).toHaveText('Copied');
    });
});

test.describe('FAQ', () => {
    test('FAQ items open on click', async ({ page }) => {
        await page.goto('/');
        // The FAQ is native <details>/<summary>: the first item starts closed and
        // opens when its summary is clicked. Assert the DOM `open` property, not
        // the attribute string — Firefox reflects a valueless boolean attribute
        // such that toHaveAttribute('open', '') is unreliable cross-browser.
        const firstItem = page.locator('#faq details').first();
        expect(await firstItem.evaluate((el) => el.open)).toBe(false);
        // Click the summary, then assert the toggled state through a poll: a
        // single synthesized click can race the native toggle on some engines.
        await firstItem.locator('summary').click();
        await expect.poll(() => firstItem.evaluate((el) => el.open)).toBe(true);
    });

    test('FAQ items are independent', async ({ page }) => {
        await page.goto('/');
        const items = page.locator('#faq details');
        await items.nth(0).locator('summary').click();
        expect(await items.nth(0).evaluate((el) => el.open)).toBe(true);
        // Native <details> do not auto-close each other.
        expect(await items.nth(1).evaluate((el) => el.open)).toBe(false);
    });
});

test.describe('Mobile Menu', () => {
    test.use({ viewport: { width: 375, height: 667 } });

    test('should toggle mobile menu', async ({ page }) => {
        await page.goto('/');
        const menuBtn = page.locator('#menuBtn');
        const navLinks = page.locator('#mainNav');
        await menuBtn.click();
        await expect(navLinks).toHaveClass(/open/);
        await expect(menuBtn).toHaveAttribute('aria-expanded', 'true');
    });
});

test.describe('Calculator', () => {
    test('should update savings on slider change', async ({ page }) => {
        await page.goto('/');
        const cloudBefore = await page.locator('#ccCloud').innerText();
        const slider = page.locator('#ccGpus');
        await slider.fill('8');
        // #ccCloud recomputes on input; a larger GPU count must change it.
        await expect(page.locator('#ccGpusOut')).toHaveText('8');
        await expect(page.locator('#ccCloud')).not.toHaveText(cloudBefore);
        await expect(page.locator('#ccSave')).not.toHaveText('$0');
    });
});

test.describe('Hero Section', () => {
    test('should show hero content', async ({ page }) => {
        await page.goto('/');
        const hero = page.locator('.hero');
        await expect(hero).toBeVisible();
        await expect(page.locator('.hero__heading')).toBeVisible();
    });

    test('should have working CTA buttons', async ({ page }) => {
        await page.goto('/');
        // The hero has a primary "Get Started" CTA and the header actions repeat
        // it; on mobile one of them is collapsed. Assert a visible one.
        const cta = page.locator('a.btn--primary[href="docs.html"]:visible').first();
        await expect(cta).toBeVisible();
        await expect(cta).toHaveAttribute('href', 'docs.html');
    });});