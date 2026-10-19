import { defineConfig } from "@playwright/test";
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { demoFixture } from './tests/fixtures/demo';
const demoPath = path.join(mkdtempSync(path.join(tmpdir(), 'argus-demo-test-')), 'current.json');
writeFileSync(demoPath, JSON.stringify(demoFixture()));
export default defineConfig({
  testDir: "./tests/dashboard",
  fullyParallel: true,
  workers: 8,
  timeout: 30000,
  expect: { timeout: 10000 },
  use: {
    baseURL: "http://localhost:3100",
    viewport: { width: 1440, height: 1000 },
    launchOptions: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE
      ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE }
      : {},
    screenshot: "only-on-failure",
  },
  webServer: {
    command: [
      "pnpm run build",
      "cp -R public .next/playwright/standalone/apps/web/public",
      "cp -R .next/playwright/static .next/playwright/standalone/apps/web/.next/playwright/static",
      "mv .next/playwright/standalone/apps/web/server.js .next/playwright/standalone/apps/web/server.cjs",
      "node .next/playwright/standalone/apps/web/server.cjs",
    ].join(' && '),
    url: "http://localhost:3100",
    reuseExistingServer: false,
    env: { NEXT_DIST_DIR: ".next/playwright", DEMO_BUNDLE_PATH: demoPath, PORT: '3100', HOSTNAME: '127.0.0.1' },
    timeout: 120000,
  },
});
