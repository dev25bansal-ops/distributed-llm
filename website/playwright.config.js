import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  testMatch: '**/*.spec.js',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : 4,
  reporter: 'html',
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    // clipboard-* permissions are Chromium-only; declaring them globally makes
    // every firefox/webkit test fail at newContext ("Unknown permission").
    // They are granted per-project below instead.
  },
  projects: [
    // All four projects are exercised in CI (ci.yml website-tests installs every
    // browser engine). Locally, run `npx playwright install firefox webkit`
    // first — a cache holding a different Playwright version's binaries makes
    // only chromium launchable, which is why the suite can look chromium-only.
    {
      name: 'chromium',
      use: { browserName: 'chromium', permissions: ['clipboard-read', 'clipboard-write'] },
    },
    {
      name: 'firefox',
      use: { browserName: 'firefox' },
    },
    {
      name: 'mobile-chrome',
      use: {
        browserName: 'chromium',
        ...devices['Pixel 5'],
      },
    },
    {
      name: 'mobile-safari',
      use: {
        browserName: 'webkit',
        ...devices['iPhone 13'],
      },
    },
  ],
  webServer: {
    // The site is static: pages are served as-is, exactly as GitHub Pages
    // serves them from the website/ directory. This used to run `vite dev`,
    // which meant the tests exercised a transformed bundle rather than the
    // files that actually ship.
    command: 'npx serve . -l 3000 --no-clipboard',
    url: 'http://localhost:3000',
    reuseExistingServer: !process.env.CI,
  },
});
