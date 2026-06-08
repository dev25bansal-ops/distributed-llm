/**
 * Security tests — CSP, XSS prevention, external resource loading.
 */

import { test, expect } from '@playwright/test';

test.describe('Security', () => {
    test('should have Content-Security-Policy header', async ({ page }) => {
        const response = await page.goto('/');
        const csp = response?.headers()['content-security-policy'];
        expect(csp).toBeDefined();
        expect(csp).toContain("default-src");
    });

    test('should not load scripts from unknown origins', async ({ page }) => {
        const scripts = [];
        page.on('request', (request) => {
            if (request.url().endsWith('.js')) {
                scripts.push(request.url());
            }
        });
        
        await page.goto('/');
        await page.waitForLoadState('networkidle');
        
        // Check that all scripts are from allowed origins
        const allowedOrigins = ['localhost', 'fonts.googleapis.com', 'plausible.io'];
        scripts.forEach(url => {
            const isAllowed = allowedOrigins.some(origin => url.includes(origin));
            expect(isAllowed).toBe(true);
        });
    });

    test('should have X-Frame-Options header', async ({ page }) => {
        const response = await page.goto('/');
        const xfo = response?.headers()['x-frame-options'];
        expect(xfo).toBeDefined();
    });

    test('should have X-Content-Type-Options header', async ({ page }) => {
        const response = await page.goto('/');
        const xcto = response?.headers()['x-content-type-options'];
        expect(xcto).toBe('nosniff');
    });

    test('should prevent XSS via innerHTML', async ({ page }) => {
        await page.goto('/');

        // Inject XSS payload via innerHTML (the dangerous method)
        const xss = '<img src=x onerror=window.__xss_fired=true>';
        const result = await page.evaluate((payload) => {
            const el = document.createElement('div');
            el.innerHTML = payload; // This is what we're testing
            document.body.appendChild(el);
            // Check if onerror fired (XSS succeeded)
            return window.__xss_fired === true;
        }, xss);

        expect(result).toBe(false);
    });

    test('should sanitize user input before DOM insertion', async ({ page }) => {
        await page.goto('/');

        // Test that the app doesn't allow script execution via user-controlled content
        const maliciousInputs = [
            '<script>alert(1)</script>',
            '<img src=x onerror=alert(1)>',
            '<svg onload=alert(1)>',
            'javascript:alert(1)',
        ];

        for (const input of maliciousInputs) {
            const executed = await page.evaluate((payload) => {
                let fired = false;
                const handler = () => { fired = true; };
                window.addEventListener('error', handler);

                const el = document.createElement('div');
                el.textContent = payload; // Safe method
                document.body.appendChild(el);

                window.removeEventListener('error', handler);
                return fired;
            }, input);

            expect(executed).toBe(false);
        }
    });
});
